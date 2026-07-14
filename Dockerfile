# lite-emb Docker 镜像
# 基于 Python 3.11 slim 镜像，使用 uv 管理依赖

FROM python:3.11-slim-bookworm AS builder

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN pip install uv --no-cache-dir

WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存层
COPY pyproject.toml .python-version ./

# 安装 Python 依赖
RUN uv pip install --system -r pyproject.toml

# ============================================================
# 生产阶段
# ============================================================
FROM python:3.11-slim-bookworm AS production

# 安装运行时系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从 builder 复制已安装的 Python 包
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制应用代码
COPY lite_emb/ ./lite_emb/
COPY main.py preload.py ./

# 预下载模型（可选，在构建时通过 --build-arg 控制）
ARG PRELOAD_MODEL=false
ENV PRELOAD_MODEL=${PRELOAD_MODEL}
RUN if [ "$PRELOAD_MODEL" = "true" ]; then \
    python preload.py; \
    else \
    echo "跳过模型预下载（设置 --build-arg PRELOAD_MODEL=true 来启用）"; \
    fi

# 创建日志目录
RUN mkdir -p logs

# 创建非 root 用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动服务
CMD ["python", "main.py"]
