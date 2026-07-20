# lite-emb

轻量级 Embedding & Rerank 服务，Embedding 对齐 OpenAI `/v1/embeddings`，Rerank 对齐 Cohere `/v1/rerank`。

> 📄 [English](./README.md)

## 核心特性

| 特性 | 说明 |
|------|------|
| 🔌 **OpenAI Embedding** | `/v1/embeddings` 对齐 OpenAI API，可直接替换 OpenAI SDK，零改动迁移 |
| 🔍 **Cohere Rerank** | `/v1/rerank` 端点，Cohere 兼容。双模式：cosine 轻量排序 / cross-encoder (`bge-reranker-base`) 高精度 |
| 🧠 **双后端引擎** | BGE-M3 走 FlagEmbedding 原生后端（dense 1024 维），通用模型走 SentenceTransformer，自动按模型类型选择最优加载器 |
| 🌍 **通用模型兼容** | 预注册 8+ 个主流模型（Embedding：e5 系列、BGE 系列、MiniLM；Reranker：bge-reranker-base），未知模型自动发现 |
| 📦 **自动下载** | CDN + aria2c 带实时进度条下载，缓存本地，一次下载永不再下 |
| 🔄 **热切换** | 请求中指定不同 `model` 参数即可自动切换，通过 `ALLOW_MODEL_SWITCH` 控制开关（生产环境建议关闭） |
| 🎯 **多设备适配** | 启动时自动检测 CUDA → MPS → CPU，GPU/MPS 自动启用 fp16，CPU 使用 fp32 |
| 💤 **启动预加载** | 启动时自动加载 Embedding + Rerank 两个模型（`PRELOAD_MODEL=true`）。设为 `false` 则首次请求时懒加载 |
| 🧵 **线程安全** | `threading.Lock` 保护模型加载/卸载临界区，PyTorch 推理天然线程安全，支持多请求并发 |
| 📐 **维度截断** | 支持 OpenAI 的 `dimensions` 参数，按需截取前 N 维向量，适配不同下游存储需求 |
| 🐳 **Docker 就绪** | 多阶段构建镜像，内置健康检查，compose 一键编排，模型缓存挂载到本地目录 |
| 📝 **Swagger 文档** | 启动后访问 `/docs` 即获完整交互式 API 文档，所有 Schema 带 description 和 examples |

## 性能对比

在 Apple Silicon (MPS) 上，轻量级模型的吞吐量远超重型模型：

| 模型 | 体积 | 维度 | 内存 | 吞吐量 | 适用场景 |
|------|------|------|------|--------|----------|
| `e5-small` | 120MB | 384 | ~0.1GB | ⚡ 极高 | 开发 / 低资源环境 |
| `all-minilm` | 80MB | 384 | ~0.1GB | ⚡ 极高 | 纯英文场景 |
| `bge-small-zh` | 95MB | 512 | ~0.1GB | ⚡ 高 | 中文场景 |
| `e5-large` | 560MB | 1024 | ~0.5GB | 中等 | 高质量多语言 |
| `bge-large-zh` | 390MB | 1024 | ~0.4GB | 中等 | 高质量中文 |
| `bge-m3` | 2GB+ | 1024 | ~2GB | 🐌 慢 / 易OOM | 需要 dense+sparse+colbert 时 |
| `bge-reranker-base` | 1GB | - | ~1GB | 中等 | Rerank 专用 cross-encoder |

> 💡 **建议**：Embedding 用 `e5-small`，Rerank 用 `bge-reranker-base`。`python preload.py` 一次下载两个。

## Rerank 模式对比

服务支持两种 Rerank 模式 — **cosine**（基于 Embedding，秒回）和 **cross-encoder**（专业 Reranker，高精度）。以下示例展示两者在同一查询上的关键差异。

> 💡 执行 `./compare_rerank.sh` 可在你的机器上重现这些结果。

### 测试1：关键词重叠但语义无关

查询：**"How to lose weight effectively?"**

| # | 文档 | Cosine (e5-small) | Cross-encoder (bge-reranker) |
|---|------|-------------------|------------------------------|
| 1 | Exercise 30 minutes daily and eat less sugar | 0.8789 ✅ | 5.4955 ✅ |
| 2 | My friend tried everything but still failed to lose weight | 0.8429 | **-4.4139** |
| 3 | The price of Bitcoin dropped 5% today | 0.7811 | **-10.1932** |

> 🔍 **Cosine**：三条得分挤在 0.78–0.88 窄区间内 — 无法有效惩罚无关的比特币文档，因为文本间存在表面特征重叠。
> 🎯 **Cross-encoder**：分差达 15.69 — 清晰地区分相关 (+5.50) 和不相关 (-10.19)，Top-1 选择毫无歧义。

### 测试2：中英文混合，语义精确匹配

查询：**"苹果公司最新产品是什么？"**

| # | 文档 | Cosine (e5-small) | Cross-encoder (bge-reranker) |
|---|------|-------------------|------------------------------|
| 1 | Apple announced the new iPhone 16 with AI features | 0.8455 | **-0.7275** ✅ |
| 2 | 苹果是一种营养丰富的水果，含有丰富的维生素C | **0.8720** ❌ | -3.5546 |
| 3 | 今天天气很好适合出去玩 | 0.8202 | -10.1942 |

> 🔍 **Cosine 把水果排在 iPhone 前面！** 水果文档 (0.8720) 得分高于正确答案 (0.8455) — 因为 cosine 将文档视为词袋，"苹果"一词在查询和水果文档中都高频出现，导致误判。
> 🎯 **Cross-encoder** 正确地将 iPhone 发布置顶，理解"苹果公司"指的是企业而非水果。

### 核心结论

| 维度 | Cosine (e5-small) | Cross-encoder (bge-reranker-base) |
|------|-------------------|-----------------------------------|
| 速度 | ⚡ 极快 (~10ms) | 🐢 较慢 (~200ms/对) |
| 内存 | 免费（复用 Embedding 模型） | 额外 ~1GB |
| 分值范围 | [-1, 1]（余弦相似度） | 无界（原始 logit） |
| 区分能力 | 弱 — 得分高度集中 | 强 — 分差显著 |
| 适用场景 | 快速过滤、头部查询 | 精准排序、长尾查询 |
| 歧义处理 | 弱 — "苹果" = 水果 | 强 — "苹果公司" = Apple Inc. |

> 💡 **建议**：用 cosine 对大候选集做快速预筛，再用 cross-encoder 对 Top-N 做精准重排。`compare_rerank.sh` 脚本帮你评估哪种模式更适合你的场景。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 服务信息：当前模型、维度、后端类型、加载耗时 |
| `GET` | `/health` | 健康检查：`{"status": "healthy"}` |
| `GET` | `/v1/models` | 模型列表（OpenAI 兼容格式） |
| `POST` | `/v1/embeddings` | 生成 Embedding（完全兼容 OpenAI API） |
| `POST` | `/v1/rerank` | 按查询相关性重排文档（Cohere 兼容） |
| `GET` | `/docs` | Swagger UI 交互式文档 |
| `GET` | `/redoc` | ReDoc 文档 |
| `GET` | `/openapi.json` | OpenAPI 规范 JSON |

### Embedding 请求示例

```json
{
  "model": "bge-m3",
  "input": "你好，这是一个测试文本",
  "encoding_format": "float",
  "dimensions": null
}
```

### Embedding 响应示例

```json
{
  "object": "list",
  "model": "bge-m3",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0123, -0.0456, 0.0789, ...]
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "total_tokens": 6
  }
}
```

### Rerank 请求与响应

双模式：**cosine**（轻量，复用 Embedding 模型）或 **cross-encoder**（专业 Reranker）。

```json
// 请求: POST /v1/rerank
{
  "model": "bge-reranker-base",
  "query": "什么是机器学习？",
  "documents": [
    "深度学习是ML的子集",
    "今天天气不错",
    "机器学习是AI分支"
  ]
}

// 响应
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "bge-reranker-base",
  "results": [
    {"index": 2, "relevance_score": 0.98, "document": "机器学习是AI分支"},
    {"index": 0, "relevance_score": 0.82, "document": "深度学习是ML的子集"},
    {"index": 1, "relevance_score": 0.12, "document": "今天天气不错"}
  ],
  "meta": {"api_version": "1.0"}
}
```

## 快速开始

> ⚠️ **必须先运行 preload！** 下载 Embedding + Rerank 两个模型（约 1.2GB）。跳过这步首次 API 调用会阻塞等待下载完成。

```bash
# 1. 下载模型（必须先执行！）
cd lite-emb
uv sync
python preload.py
```

```bash
# 2. 启动服务
uv run python main.py
```

可选：通过 `.env` 自定义模型或端口：

```bash
cp .env.example .env
# 编辑 .env 修改 MODEL_NAME、DEV_PORT 等
```

服务默认运行在 `http://localhost:8000`，访问 `/docs` 查看交互式 API 文档。

### 3. 测试 API

```bash
# 健康检查
curl http://localhost:8000/health

# 服务信息（含当前模型、维度等）
curl http://localhost:8000/

# 模型列表
curl http://localhost:8000/v1/models

# 单条文本 Embedding
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "e5-small", "input": "你好，这是一个测试文本"}'

# 批量文本 Embedding
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "e5-small", "input": ["文本一", "文本二", "文本三"]}'

# Rerank（cosine 模式 — 秒回）
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"e5-small","query":"什么是机器学习","documents":["深度学习","天气不错","AI分支"]}'

# Rerank（cross-encoder 模式 — 高精度）
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"bge-reranker-base","query":"什么是机器学习","documents":["深度学习","天气不错","AI分支"]}'
```

### 4. 使用 OpenAI SDK（零改动替换）

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # lite-emb 不需要 API key
)

response = client.embeddings.create(
    model="bge-m3",
    input="你的文本内容"
)

print(response.data[0].embedding)
```

## 支持的模型

### 预注册模型

| 模型别名 | HuggingFace ID | 维度 | 最大长度 | 后端 | 支持语言 |
|----------|---------------|------|----------|------|----------|
| `bge-m3` | `BAAI/bge-m3` | 1024 | 8192 | BGEM3FlagModel | 100+ 语言 |
| `bge-large-zh` | `BAAI/bge-large-zh-v1.5` | 1024 | 512 | SentenceTransformer | 中文 |
| `bge-small-zh` | `BAAI/bge-small-zh-v1.5` | 512 | 512 | SentenceTransformer | 中文 |
| `e5-large` | `intfloat/multilingual-e5-large` | 1024 | 512 | SentenceTransformer | 100+ 语言 |
| `all-minilm` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 | SentenceTransformer | 英文 |
| `all-minilm-l12` | `sentence-transformers/all-MiniLM-L12-v2` | 384 | 256 | SentenceTransformer | 英文 |
| `bge-micro` | `BAAI/bge-micro` | 384 | 512 | SentenceTransformer | 100+ 语言 (~17MB) |

### 自动发现

任何 HuggingFace 上的 sentence-transformers 兼容模型都可以直接使用，无需预注册。系统会自动：

1. 检查是否为已知模型 → 使用注册的后端类型
2. 模型名含 `bge-m3` → 自动走 BGEM3FlagModel 后端
3. 其余模型 → 走 SentenceTransformer 通用后端

```bash
# 直接使用任意 HuggingFace 模型
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "e5-base", "input": "hello"}'
```

## 模型管理

### 预下载

```bash
# 下载默认模型（BAAI/bge-m3）
uv run python preload.py

# 下载指定模型
uv run python preload.py --model sentence-transformers/all-MiniLM-L6-v2

# 离线模式下预下载后启动
HF_HUB_OFFLINE=1 uv run python main.py
```

### 加载策略

```
请求到达 POST /v1/embeddings
  ├─ 模型已加载？ → 直接返回缓存实例
  ├─ 模型不同？
  │   ├─ ALLOW_MODEL_SWITCH=true → 卸载旧模型 → 加载新模型
  │   └─ ALLOW_MODEL_SWITCH=false → 返回 400 错误
  └─ 首次加载？
      ├─ 检查本地缓存是否存在
      ├─ 不存在 → HuggingFace 下载（走 hf-mirror 镜像）
      └─ BGEM3FlagModel / SentenceTransformer 加载到内存
```

## 启动方式详解

| 方式 | 命令 | 适用场景 |
|------|------|----------|
| 开发模式 | `uv run python main.py` | 本地开发，单进程 |
| 开发热重载 | `uv run uvicorn lite_emb.app:app --reload` | 代码修改后自动重启 |
| 生产多 worker | `WORKERS=4 uv run python main.py` | 多核 CPU 并发 |
| 懒加载启动 | `PRELOAD_MODEL=false uv run python main.py` | 首次请求时才加载模型 |
| 离线启动 | `HF_HUB_OFFLINE=1 uv run python main.py` | 无网络环境（需先 preload） |
| 指定模型 | `MODEL_NAME=all-MiniLM-L6-v2 uv run python main.py` | 切换默认模型 |
| Docker | `docker compose up -d` | 容器化部署 |
| Docker (GPU) | 取消 `docker-compose.yml` 中 GPU 注释后 `docker compose up -d` | NVIDIA GPU 加速 |

## Docker 部署

```bash
# 构建并启动
docker compose build --build-arg USE_APT_MIRROR=true --build-arg USE_PYPI_MIRROR=true
docker compose up -d

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

> 🔑 **国内用户**：需在 Docker Desktop → Settings → Resources → Proxies 配置代理，否则拉取基础镜像会超时。代理只影响 docker daemon 拉取 `python:3.11-slim-bookworm`，容器内部的 apt/pip/模型下载仍走国内源直连。

## 配置项

所有配置通过 `.env` 文件或环境变量管理：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `e5-small` | Embedding 模型（别名或 HuggingFace ID） |
| `RERANK_MODEL_NAME` | `bge-reranker-base` | Rerank 模型（别名或 HuggingFace ID） |
| `DEV_HOST` | `0.0.0.0` | 监听地址 |
| `DEV_PORT` | `8000` | 监听端口 |
| `WORKERS` | `1` | uvicorn worker 数（生产建议 CPU 核数） |
| `MAX_BATCH_SIZE` | `32` | 单次编码最大批处理大小 |
| `MAX_SEQUENCE_LENGTH` | `512` | 输入文本最大 token 数 |
| `PRELOAD_MODEL` | `true` | 启动时是否预加载两个模型 |
| `ALLOW_MODEL_SWITCH` | `false` | 是否允许请求中动态切换模型 |
| `EMBEDDING_NORMALIZE` | `true` | 输出向量是否 L2 归一化 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 镜像站 |
| `HF_HUB_ENABLE_HF_TRANSFER` | `true` | 启用 hf-transfer 高速下载 |
| `HF_HUB_OFFLINE` | `0` | 离线模式（1=仅使用本地缓存） |
| `LOG_LEVEL` | `INFO` | 日志级别（DEBUG/INFO/WARNING/ERROR） |
| `LOG_FILE` | `logs/app.log` | 日志文件路径（自动轮转 + 压缩） |

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| Web 框架 | FastAPI | 异步高性能，自动生成 OpenAPI 文档 |
| 服务器 | Uvicorn | ASGI 服务器，支持多 worker |
| 配置管理 | pydantic-settings | 类型安全，.env 加载 |
| 日志 | loguru | 结构化日志，文件轮转压缩 |
| BGE-M3 后端 | FlagEmbedding | 原生支持 dense/sparse/colbert 向量 |
| 通用后端 | sentence-transformers | 兼容所有 HuggingFace 嵌入模型 |
| 模型下载 | huggingface-hub + hf-transfer | 并行高速下载 |
| 深度学习 | PyTorch | CUDA/MPS/CPU 自适应 |
| 依赖管理 | uv | Rust 实现，极速解析安装 |
| 测试 | pytest + httpx | 24 个测试覆盖所有模块 |
| 容器化 | Docker multi-stage | 最小化镜像体积 |

## 许可证

MIT
