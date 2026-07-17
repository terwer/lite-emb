# lite-emb

A lightweight, OpenAI-compatible embedding service powered by BGE-M3 and sentence-transformers.

> 📄 [中文](./README_zh_CN.md)

## Features

| Feature | Description |
|---------|-------------|
| 🔌 **OpenAI Compatible** | Drop-in replacement for OpenAI's `/v1/embeddings` — same request/response format, works with the official Python SDK with zero code changes |
| 🧠 **Dual Backend** | BGE-M3 via FlagEmbedding (dense 1024-dim), general models via SentenceTransformer — auto-selects the optimal loader per model type |
| 🌍 **Broad Model Support** | 5 pre-registered models (BGE-M3 / BGE-large-zh / BGE-small-zh / all-MiniLM / multilingual-e5), plus auto-discovery for any HuggingFace sentence-transformers model |
| 📦 **Auto Download** | Downloads models from HuggingFace on first request, cached locally. Uses hf-mirror.com by default for faster downloads in China, with hf-transfer for parallel transfer |
| 🔄 **Hot Swap** | Switch models on the fly by passing a different `model` parameter. Controlled via `ALLOW_MODEL_SWITCH` (recommended off in production) |
| 🎯 **Device Auto-detect** | CUDA → MPS → CPU, with fp16 on GPU/MPS and fp32 on CPU |
| 💤 **Lazy Loading** | Models load on first API request by default. Set `PRELOAD_MODEL=true` for eager loading at startup |
| 🧵 **Thread Safe** | `threading.Lock` guards model load/unload critical sections. PyTorch inference is natively thread-safe for concurrent requests |
| 📐 **Dimension Truncation** | Supports OpenAI's `dimensions` parameter — truncate vectors to the first N dimensions for downstream storage needs |
| 🐳 **Docker Ready** | Multi-stage Docker build, built-in health check, compose one-liner, persistent HuggingFace cache volume |
| 📝 **Swagger Docs** | Visit `/docs` after startup for full interactive API docs — all schemas include descriptions and examples |

## Performance Comparison

On Apple Silicon (MPS), lightweight models deliver dramatically better throughput than heavy ones:

| Model | Size | Dims | Memory | Throughput | Suitable For |
|-------|------|------|--------|------------|--------------|
| `e5-small` | 120MB | 384 | ~0.1GB | ⚡ Very High | 开发 / 低资源环境 |
| `all-minilm` | 80MB | 384 | ~0.1GB | ⚡ Very High | 英文场景 |
| `bge-small-zh` | 95MB | 512 | ~0.1GB | ⚡ High | 中文场景 |
| `e5-large` | 560MB | 1024 | ~0.5GB | Medium | 高质量多语言 |
| `bge-large-zh` | 390MB | 1024 | ~0.4GB | Medium | 高质量中文 |
| `bge-m3` | 2GB+ | 1024 | ~2GB | 🐌 Slow / OOM | 需要 dense+sparse+colbert 时 |
| `bge-reranker-base` | 1GB | - | ~1GB | Medium | Reranker 专用 cross-encoder |

> 💡 **Recommendation**: Start with `e5-small` (default) for embeddings, `bge-reranker-base` for reranking. Both pre-loaded by `python preload.py`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info: current model, dimension, backend type, load time |
| `GET` | `/health` | Health check: `{"status": "healthy"}` |
| `GET` | `/v1/models` | List models (OpenAI-compatible format) |
| `POST` | `/v1/embeddings` | Generate embeddings (fully OpenAI-compatible) |
| `POST` | `/v1/rerank` | Rerank documents by query relevance (Cohere-compatible) |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc |
| `GET` | `/openapi.json` | OpenAPI spec |

### Embedding Request

```json
{
  "model": "bge-m3",
  "input": "The quick brown fox jumps over the lazy dog",
  "encoding_format": "float",
  "dimensions": null
}
```

### Embedding Response

```json
{
  "object": "list",
  "model": "bge-m3",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0123, -0.0456, 0.0789, "..."]
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

### Rerank Request & Response

Dual-mode: **cosine** (lightweight, uses embedding model) or **cross-encoder** (professional reranker).

```json
// Request: POST /v1/rerank
{
  "model": "bge-reranker-base",
  "query": "What is machine learning?",
  "documents": [
    "Deep learning is a subset of ML",
    "The weather is nice today",
    "Machine learning is a branch of AI"
  ]
}

// Response
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "bge-reranker-base",
  "results": [
    {"index": 2, "relevance_score": 0.98, "document": "Machine learning is a branch of AI"},
    {"index": 0, "relevance_score": 0.82, "document": "Deep learning is a subset of ML"},
    {"index": 1, "relevance_score": 0.12, "document": "The weather is nice today"}
  ],
  "meta": {"api_version": "1.0"}
}
```

## Quick Start

> ⚠️ **Run preload first!** Downloads both embedding and reranker models (~1.2GB total). Skip this and the first API call will block until the download completes.

```bash
# 1. Download models (required — do this before starting!)
cd lite-emb
uv sync
python preload.py
```

```bash
# 2. Start the server
uv run python main.py
```

Optional: customize models or port in `.env`:

```bash
cp .env.example .env
# Edit .env to change MODEL_NAME, DEV_PORT, etc.
```

The service runs at `http://localhost:8000`. Visit `/docs` for interactive API docs.

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Service info (current model, dimension, etc.)
curl http://localhost:8000/

# Model list
curl http://localhost:8000/v1/models

# Single text embedding
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": "Hello, world!"}'

# Batch embedding
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": ["text one", "text two", "text three"]}'

# Rerank (cosine mode — instant)
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"e5-small","query":"What is ML","documents":["Deep learning","Nice weather","AI branch"]}'

# Rerank (cross-encoder mode — higher accuracy)
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"bge-reranker-base","query":"What is ML","documents":["Deep learning","Nice weather","AI branch"]}'
```

### 4. Use with OpenAI SDK (Drop-in Replacement)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # lite-emb requires no API key
)

response = client.embeddings.create(
    model="bge-m3",
    input="Your text here"
)

print(response.data[0].embedding)
```

## Supported Models

### Pre-registered Models

| Alias | HuggingFace ID | Dims | Max Length | Backend | Languages |
|-------|---------------|------|------------|---------|-----------|
| `bge-m3` | `BAAI/bge-m3` | 1024 | 8192 | BGEM3FlagModel | 100+ |
| `bge-large-zh` | `BAAI/bge-large-zh-v1.5` | 1024 | 512 | SentenceTransformer | Chinese |
| `bge-small-zh` | `BAAI/bge-small-zh-v1.5` | 512 | 512 | SentenceTransformer | Chinese |
| `e5-large` | `intfloat/multilingual-e5-large` | 1024 | 512 | SentenceTransformer | 100+ |
| `all-minilm` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 | SentenceTransformer | English |
| `all-minilm-l12` | `sentence-transformers/all-MiniLM-L12-v2` | 384 | 256 | SentenceTransformer | English |
| `bge-micro` | `BAAI/bge-micro` | 384 | 512 | SentenceTransformer | 100+ (~17MB) |
| `e5-large` | `intfloat/multilingual-e5-large` | 1024 | 512 | SentenceTransformer | 100+ |

### Auto Discovery

Any HuggingFace sentence-transformers compatible model works out of the box — no pre-registration needed. The system will:

1. Check if it is a known model → use the registered backend
2. Model name contains `bge-m3` → use BGEM3FlagModel backend
3. Everything else → use SentenceTransformer backend

```bash
# Use any HuggingFace model directly
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "e5-base", "input": "hello"}'
```

## Model Management

### Pre-download

```bash
# Download the default model (BAAI/bge-m3)
uv run python preload.py

# Download a specific model
uv run python preload.py --model sentence-transformers/all-MiniLM-L6-v2

# Start in offline mode after pre-downloading
HF_HUB_OFFLINE=1 uv run python main.py
```

### Loading Strategy

```
Request arrives at POST /v1/embeddings
  ├─ Model already loaded? → Return cached instance
  ├─ Different model requested?
  │   ├─ ALLOW_MODEL_SWITCH=true → Unload old model → Load new model
  │   └─ ALLOW_MODEL_SWITCH=false → Return 400 error
  └─ First load?
      ├─ Check local cache
      ├─ Not cached → Download from HuggingFace (via hf-mirror)
      └─ Load via BGEM3FlagModel / SentenceTransformer into memory
```

## Startup Modes

| Mode | Command | Use Case |
|------|---------|----------|
| Development | `uv run python main.py` | Local dev, single process |
| Hot Reload | `uv run uvicorn lite_emb.app:app --reload` | Auto-reload on code changes |
| Multi-worker | `WORKERS=4 uv run python main.py` | Multi-core concurrency |
| Eager Loading | `PRELOAD_MODEL=true uv run python main.py` | Avoid cold-start latency |
| Offline | `HF_HUB_OFFLINE=1 uv run python main.py` | No network (preload first) |
| Custom Model | `MODEL_NAME=all-MiniLM-L6-v2 uv run python main.py` | Switch default model |
| Docker | `docker compose up -d` | Containerized deployment |
| Docker (GPU) | Uncomment GPU section in `docker-compose.yml`, then `docker compose up -d` | NVIDIA GPU acceleration |

## Docker Deployment

```bash
# Build image (without pre-downloading the model)
docker build -t lite-emb .

# Build image with BGE-M3 pre-downloaded
docker build --build-arg PRELOAD_MODEL=true -t lite-emb .

# Start with compose
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

## Configuration

All settings are managed via `.env` file or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `BAAI/bge-m3` | Default model HuggingFace ID |
| `DEV_HOST` | `0.0.0.0` | Listen address |
| `DEV_PORT` | `8000` | Listen port |
| `WORKERS` | `1` | Uvicorn workers (set to CPU count in production) |
| `MAX_BATCH_SIZE` | `32` | Max batch size per encoding call |
| `MAX_SEQUENCE_LENGTH` | `8192` | Max input tokens |
| `PRELOAD_MODEL` | `false` | Preload model at startup |
| `ALLOW_MODEL_SWITCH` | `false` | Allow dynamic model switching per request |
| `EMBEDDING_NORMALIZE` | `true` | L2-normalize output vectors |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace mirror endpoint |
| `HF_HUB_ENABLE_HF_TRANSFER` | `true` | Enable hf-transfer for faster downloads |
| `HF_HUB_OFFLINE` | `0` | Offline mode (1 = local cache only) |
| `LOG_LEVEL` | `INFO` | Log level (DEBUG/INFO/WARNING/ERROR) |
| `LOG_FILE` | `logs/app.log` | Log file path (auto rotation + compression) |

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Web Framework | FastAPI | Async, auto-generated OpenAPI docs |
| Server | Uvicorn | ASGI server, multi-worker support |
| Configuration | pydantic-settings | Type-safe, .env loading |
| Logging | loguru | Structured logging, file rotation & compression |
| BGE-M3 Backend | FlagEmbedding | Native dense/sparse/colbert vector support |
| General Backend | sentence-transformers | Compatible with all HuggingFace embedding models |
| Model Download | huggingface-hub + hf-transfer | Parallel high-speed downloads |
| Deep Learning | PyTorch | CUDA/MPS/CPU adaptive |
| Package Manager | uv | Rust-based, fast resolution & installation |
| Testing | pytest + httpx | 24 tests covering all modules |
| Containerization | Docker multi-stage | Minimized image size |

## License

MIT