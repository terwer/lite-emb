# lite-emb

轻量级 OpenAI 兼容的 Embedding 服务，基于 BGE-M3 和 sentence-transformers。

> 📄 [English](./README.md)

## 核心特性

| 特性 | 说明 |
|------|------|
| 🔌 **OpenAI 兼容** | `/v1/embeddings` 请求/响应格式完全对齐 OpenAI API，可直接替换 OpenAI SDK，零改动迁移 |
| 🧠 **双后端引擎** | BGE-M3 走 FlagEmbedding 原生后端（dense 1024 维），通用模型走 SentenceTransformer，自动按模型类型选择最优加载器 |
| 🌍 **通用模型兼容** | 预注册 5 个主流模型（BGE-M3 / BGE-large-zh / BGE-small-zh / all-MiniLM / multilingual-e5），未知模型自动走 SentenceTransformer 发现 |
| 📦 **自动下载** | 首次请求时自动从 HuggingFace 下载模型到本地缓存，默认走 hf-mirror.com 国内镜像，支持 hf-transfer 高速并行传输 |
| 🔄 **热切换** | 请求中指定不同 `model` 参数即可自动切换，通过 `ALLOW_MODEL_SWITCH` 控制开关（生产环境建议关闭） |
| 🎯 **多设备适配** | 启动时自动检测 CUDA → MPS → CPU，GPU/MPS 自动启用 fp16，CPU 使用 fp32 |
| 💤 **懒加载** | 默认不在启动时加载模型，首次 API 请求触发加载。可通过 `PRELOAD_MODEL=true` 改为启动预加载 |
| 🧵 **线程安全** | `threading.Lock` 保护模型加载/卸载临界区，PyTorch 推理天然线程安全，支持多请求并发 |
| 📐 **维度截断** | 支持 OpenAI 的 `dimensions` 参数，按需截取前 N 维向量，适配不同下游存储需求 |
| 🐳 **Docker 就绪** | 多阶段构建镜像，内置健康检查，compose 一键编排，HuggingFace 缓存持久化 volume |
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

> 💡 **建议**：默认使用 `e5-small`，384 维覆盖 100+ 语言，资源占用极小。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 服务信息：当前模型、维度、后端类型、加载耗时 |
| `GET` | `/health` | 健康检查：`{"status": "healthy"}` |
| `GET` | `/v1/models` | 模型列表（OpenAI 兼容格式） |
| `POST` | `/v1/embeddings` | 生成 Embedding（完全兼容 OpenAI API） |
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

## 快速开始

### 1. 安装依赖

```bash
cd /Volumes/workspace/myproject/lite-emb
uv sync
```

### 2. 配置（可选）

```bash
cp .env.example .env
# 编辑 .env 修改默认模型、端口等
```

### 3. 启动服务

```bash
uv run python main.py
```

服务默认运行在 `http://localhost:8000`，访问 `/docs` 查看交互式 API 文档。

### 4. 测试 API

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
  -d '{"model": "bge-m3", "input": "你好，这是一个测试文本"}'

# 批量文本 Embedding
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": ["文本一", "文本二", "文本三"]}'
```

### 5. 使用 OpenAI SDK（零改动替换）

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
| 预加载启动 | `PRELOAD_MODEL=true uv run python main.py` | 避免首次请求冷启动延迟 |
| 离线启动 | `HF_HUB_OFFLINE=1 uv run python main.py` | 无网络环境（需先 preload） |
| 指定模型 | `MODEL_NAME=all-MiniLM-L6-v2 uv run python main.py` | 切换默认模型 |
| Docker | `docker compose up -d` | 容器化部署 |
| Docker (GPU) | 取消 `docker-compose.yml` 中 GPU 注释后 `docker compose up -d` | NVIDIA GPU 加速 |

## Docker 部署

```bash
# 构建镜像（不预下载模型）
docker build -t lite-emb .

# 构建镜像并预下载 BGE-M3
docker build --build-arg PRELOAD_MODEL=true -t lite-emb .

# 使用 compose 启动
docker compose up -d

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

## 配置项

所有配置通过 `.env` 文件或环境变量管理：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | `BAAI/bge-m3` | 默认模型 HuggingFace ID |
| `DEV_HOST` | `0.0.0.0` | 监听地址 |
| `DEV_PORT` | `8000` | 监听端口 |
| `WORKERS` | `1` | uvicorn worker 数（生产建议 CPU 核数） |
| `MAX_BATCH_SIZE` | `32` | 单次编码最大批处理大小 |
| `MAX_SEQUENCE_LENGTH` | `8192` | 输入文本最大 token 数 |
| `PRELOAD_MODEL` | `false` | 启动时是否预加载模型 |
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
