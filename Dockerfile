# lite-emb Docker 镜像
# 基于 Python 3.11 slim 镜像，使用 uv 管理依赖
#
# 国内用户构建时可通过 --build-arg 启用镜像加速：
#   docker build --build-arg USE_APT_MIRROR=true --build-arg USE_PYPI_MIRROR=true -t lite-emb .
#   docker compose build --build-arg USE_APT_MIRROR=true --build-arg USE_PYPI_MIRROR=true

FROM python:3.11-slim-bookworm AS builder

# 清空代理（只让 daemon 拉基础镜像时走代理，容器内直连）
ENV HTTP_PROXY="" HTTPS_PROXY="" http_proxy="" https_proxy=""

# -- 国内镜像加速（默认关闭，国内用户通过 --build-arg 启用） --
ARG USE_APT_MIRROR=false
ARG APT_MIRROR=mirrors.aliyun.com
ARG USE_PYPI_MIRROR=false
ARG PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple

# 安装系统依赖
RUN if [ "$USE_APT_MIRROR" = "true" ]; then \
    sed -i "s|http://deb.debian.org|http://${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; \
    fi \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN if [ "$USE_PYPI_MIRROR" = "true" ]; then \
    pip install uv --no-cache-dir -i ${PYPI_MIRROR}; \
    else \
    pip install uv --no-cache-dir; \
    fi

WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存层
COPY pyproject.toml .python-version ./

# 安装 Python 依赖
RUN if [ "$USE_PYPI_MIRROR" = "true" ]; then \
    uv pip install --system --index-url ${PYPI_MIRROR} -r pyproject.toml; \
    else \
    uv pip install --system -r pyproject.toml; \
    fi

# ============================================================
# 生产阶段
# ============================================================
FROM python:3.11-slim-bookworm AS production

# 清空代理
ENV HTTP_PROXY="" HTTPS_PROXY="" http_proxy="" https_proxy=""

# -- 国内镜像加速 --
ARG USE_APT_MIRROR=false
ARG APT_MIRROR=mirrors.aliyun.com

# 安装运行时系统依赖
RUN if [ "$USE_APT_MIRROR" = "true" ]; then \
    sed -i "s|http://deb.debian.org|http://${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; \
    fi \
    && apt-get update && apt-get install -y --no-install-recommends \
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

# 创建非 root 用户，预先准备缓存目录权限
RUN useradd -m -u 1000 appuser \
    && mkdir -p /home/appuser/.cache/huggingface /home/appuser/.cache/lite-emb \
    && chown -R appuser:appuser /app /home/appuser/.cache
USER appuser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动服务
CMD ["python", "main.py"]
