# Rerank 功能设计提案

## 背景

embedding 服务已稳定运行（`e5-small` 默认模型），用户需要扩展 rerank 功能。
Rerank（重排序）是 RAG 流程的关键环节：对检索到的候选文档按与 query 的相关性重新打分排序，
显著提升最终答案质量。

核心需求：

- 集成到现有 lite-emb 服务（不新建项目）
- **Cohere `/v1/rerank` 格式**（OpenAI 没有 rerank 端点，Cohere 是事实标准）
- 使用 `BAAI/bge-reranker-base`（~1GB cross-encoder，CPU 可运行）
- 轻量回退：e5-small cosine 相似度（0 额外成本）
- 复用现有架构：注册表 → 下载 → 加载 → 路由 → 服务

## 参考来源

- [Cohere Rerank API 文档](https://docs.cohere.com/reference/rerank)
- FlagEmbedding 内置 `FlagReranker` 类（同项目 lab_rag 验证可用）
- 现有 lite-emb 架构：模型注册表、CDN 下载、设备检测、服务编排

## API 设计（Cohere 兼容）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/rerank` | 对文档列表按 query 相关性重排序 |

### 请求格式

```json
{
  "model": "bge-reranker-base",     // 必填，支持别名
  "query": "什么是机器学习？",        // 必填
  "documents": ["文本A", "文本B"],   // 必填，字符串数组
  "top_n": 3,                        // 可选，返回前 N 条
  "return_documents": true           // 可选，是否返回原文
}
```

### 响应格式

```json
{
  "id": "uuid",
  "model": "bge-reranker-base",
  "results": [
    {"index": 0, "relevance_score": 0.98, "document": "文本A"},
    {"index": 1, "relevance_score": 0.42, "document": "文本B"}
  ],
  "meta": {"api_version": "1.0"}
}
```

## 目录结构（新增/修改）

```
lite-emb/
├── lite_emb/
│   ├── config.py                  # [修改] 新增 RERANK_MODEL_NAME
│   ├── models/
│   │   ├── registry.py            # [修改] 注册 bge-reranker-base
│   │   ├── loader.py              # [修改] 新增 RerankerBackend
│   │   └── manager.py             # [修改] Reranker 管理（独立于 Embedding）
│   ├── schemas/
│   │   ├── rerank_request.py      # [新增] Cohere 兼容 RerankRequest
│   │   └── rerank_response.py     # [新增] Cohere 兼容 RerankResponse
│   ├── routers/
│   │   └── rerank.py              # [新增] POST /v1/rerank
│   └── services/
│       └── rerank_service.py      # [新增] Rerank 编排（cosine + cross-encoder）
│
└── tests/
    └── test_rerank_api.py         # [新增]
```

## 模型加载策略（双模式）

```
POST /v1/rerank 到达
  ├─ model == "e5-small" 或其他 Embedding 模型？
  │   └─ cosine 模式：
  │       1. query → embedding_service.embed(query)
  │       2. docs → embedding_service.embed(documents)
  │       3. cosine_similarity → 排序 → top_n
  │       （零额外内存，毫秒级延迟）
  │
  ├─ model == "bge-reranker-base"？
  │   └─ cross-encoder 模式：
  │       1. load FlagReranker (BAAI/bge-reranker-base)
  │       2. compute_score([query, doc] pairs) → scores
  │       3. sort → top_n
  │       （~1GB 内存，每条 ~500ms CPU）
  │
  └─ 其他 cross-encoder 模型 → 自动发现 FlagReranker
```

## 配置管理（新增）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RERANK_MODEL_NAME` | `BAAI/bge-reranker-base` | 默认 Rerank 模型 |
| `RERANK_MAX_BATCH` | `32` | 单次 rerank 最大文档数 |
| `RERANK_DEFAULT_TOP_N` | `10` | 默认返回前 N 条 |

## 实现步骤

### 第1步：Schema 层
- `schemas/rerank_request.py`：Request 模型（Cohere 兼容 + Pydantic 校验）
- `schemas/rerank_response.py`：Response 模型

### 第2步：模型层
- `registry.py`：注册 `bge-reranker-base`，别名映射
- `loader.py`：新增 `RerankerBackend`（FlagReranker + cosine 两种后端）
- `utils/download.py`：添加 reranker 文件列表

### 第3步：业务逻辑层
- `services/rerank_service.py`：cosine 回退 + cross-encoder 主逻辑

### 第4步：API 路由层
- `routers/rerank.py`：POST /v1/rerank

### 第5步：应用入口
- `app.py`：注册路由

### 第6步：测试
- `tests/test_rerank_api.py`：API 集成测试

## 验证方式

```bash
# 1. cosine 模式（秒响应，零额外下载）
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "e5-small",
    "query": "hello world",
    "documents": ["goodbye", "hello everyone", "hi there"]
  }'

# 2. cross-encoder 模式（需先下载 1GB 模型）
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-reranker-base",
    "query": "什么是机器学习",
    "documents": ["深度学习是ML的子集", "今天天气不错", "机器学习是AI分支"]
  }'

# 3. 用 Python Cohere SDK 验证兼容性
import cohere
co = cohere.Client(base_url="http://localhost:8000/v1", api_key="not-needed")
result = co.rerank(model="bge-reranker-base", query="hello", documents=["a", "b"])
```
