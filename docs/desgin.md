# lite-emb 项目实现计划

## 背景

用户需要一个独立的轻量级 embedding 服务，提供 OpenAI 兼容的 API 接口。这是一个全新的项目，位于 `./lite-emb`。

核心需求：

- FastAPI 构建
- OpenAI 兼容的 `/v1/embeddings` 接口
- 默认使用 BGE-M3，兼容 sentence-transformers 等模型
- 自动从 HuggingFace 下载模型
- 不计成本，做到生产级别

## 参考来源

现有项目 `lab_rag` 提供了以下可复用的模式：

- `FlagEmbedding.BGEM3FlagModel` 加载 BGE-M3
- `huggingface_hub.snapshot_download` 下载模型
- `pydantic-settings.BaseSettings` 管理配置
- 设备检测逻辑：CUDA > MPS > CPU，fp16 仅在 GPU/MPS
- 单例模式管理模型实例
- `loguru` + `InterceptHandler` 日志模式

## 目录结构

```
./lite-emb/
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── main.py                       # uvicorn 入口
├── preload.py                    # 模型预下载脚本
├── Dockerfile
├── docker-compose.yml
│
├── lite_emb/
│   ├── __init__.py
│   ├── app.py                    # FastAPI 应用工厂 + 生命周期
│   ├── config.py                 # pydantic-settings 配置
│   ├── logger.py                 # loguru 日志初始化
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py           # 模型注册表
│   │   ├── loader.py             # 模型加载器（BGEM3 + SentenceTransformer）
│   │   └── manager.py            # 模型生命周期管理（懒加载、缓存、热切换）
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── request.py            # OpenAI 兼容 EmbeddingRequest
│   │   └── response.py           # OpenAI 兼容 EmbeddingResponse, UsageInfo
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── embeddings.py         # POST /v1/embeddings
│   │   ├── models.py             # GET /v1/models
│   │   └── health.py             # GET /health, GET /
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   └── embedding_service.py  # embedding 推理编排
│   │
│   └── utils/
│       ├── __init__.py
│       ├── device.py             # 设备检测
│       └── download.py           # HuggingFace 下载封装
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_embeddings_api.py
    ├── test_model_manager.py
    └── test_device.py
```

## 架构设计要点

### API 路由（OpenAI 兼容）

| 方法 | 路径             | 说明                                  |
| ---- | ---------------- | ------------------------------------- |
| POST | `/v1/embeddings` | 生成 embeddings，完全兼容 OpenAI 格式 |
| GET  | `/v1/models`     | 列出可用模型                          |
| GET  | `/health`        | 健康检查                              |
| GET  | `/`              | 服务信息                              |

### 模型加载策略

- **BGEM3FlagModel 后端**：BGE-M3 专用，提供 1024 维 dense 向量
- **SentenceTransformer 后端**：通用回退，支持任意 HuggingFace 兼容模型
- **自动发现**：未知模型名自动尝试 SentenceTransformer → BGEM3FlagModel
- **懒加载**：首次请求时加载模型，避免启动延迟
- **线程安全**：`threading.Lock` 保护模型切换，推理天然线程安全

### 配置管理

环境变量通过 `.env` 文件配置（pydantic-settings）：

- `MODEL_NAME`：模型 ID（默认 BAAI/bge-m3）
- `MAX_BATCH_SIZE`：批处理大小（默认 32）
- `HF_ENDPOINT`：HuggingFace 镜像地址（默认 hf-mirror.com）
- `PRELOAD_MODEL`：是否启动时预加载（默认 false）
- `ALLOW_MODEL_SWITCH`：是否允许请求中切换模型（默认 false）

### 自动下载机制

三层下载：

1. `preload.py` 脚本：构建时预先下载
2. 运行时懒下载：首次使用自动 snapshot_download
3. 离线模式：`HF_HUB_OFFLINE=1` 时仅使用本地缓存

## 实现步骤

### 第1步：项目初始化

- 创建目录结构
- 编写 `pyproject.toml`（uv 项目，包含所有依赖）
- 创建 `.python-version`、`.gitignore`、`.env.example`

### 第2步：配置与工具模块

- `config.py`：pydantic-settings 配置类
- `logger.py`：loguru 日志初始化
- `utils/device.py`：设备检测（CUDA > MPS > CPU）
- `utils/download.py`：HuggingFace 下载封装

### 第3步：模型层

- `models/registry.py`：模型注册表 + ModelInfo
- `models/loader.py`：BGEM3FlagModel 和 SentenceTransformer 后端
- `models/manager.py`：模型生命周期管理

### 第4步：Schema 层

- `schemas/request.py`：OpenAI 兼容请求模型
- `schemas/response.py`：OpenAI 兼容响应模型

### 第5步：业务逻辑层

- `services/embedding_service.py`：embedding 推理编排

### 第6步：API 路由层

- `routers/embeddings.py`：POST /v1/embeddings
- `routers/models.py`：GET /v1/models
- `routers/health.py`：健康检查

### 第7步：应用入口

- `app.py`：FastAPI 应用工厂
- `main.py`：uvicorn 启动入口
- `preload.py`：模型预下载脚本

### 第8步：Docker 支持

- `Dockerfile`
- `docker-compose.yml`

### 第9步：测试

- API 集成测试
- 模型管理器单元测试
- 设备检测单元测试

## 验证方式

1. 启动服务：`cd ./lite-emb && uv run python main.py`
2. 测试健康检查：`curl http://localhost:8000/health`
3. 测试 embedding API：`curl -X POST http://localhost:8000/v1/embeddings -H "Content-Type: application/json" -d '{"model":"bge-m3","input":"Hello world"}'`
4. 测试模型列表：`curl http://localhost:8000/v1/models`
5. 使用 OpenAI Python SDK 验证兼容性：`openai.OpenAI(base_url="http://localhost:8000/v1").embeddings.create(model="bge-m3", input="test")`
6. 运行测试：`uv run pytest tests/`
