"""
FastAPI 应用工厂。

负责：
- 创建和配置 FastAPI 应用实例
- 注册路由和中间件
- 管理应用生命周期（启动/关闭）
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from lite_emb.config import settings
from lite_emb.logger import setup_logging
from lite_emb.models.manager import model_manager
from lite_emb.routers import embeddings, health, models, rerank
from lite_emb.services.rerank_service import rerank_service
from lite_emb.utils.device import get_device_and_precision
from lite_emb.utils.preload import preload_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理。

    启动时初始化日志和模型（如果需要预加载），
    关闭时卸载模型释放资源。
    """
    # === 启动阶段 ===
    setup_logging()

    device, use_fp16 = get_device_and_precision()
    logger.info("=" * 60)
    logger.info("lite-emb 启动中...")
    logger.info("版本: 0.1.0")
    logger.info("Embedding 模型: {}", settings.MODEL_NAME)
    logger.info("Rerank 模型: {}", settings.RERANK_MODEL_NAME)
    logger.info("计算设备: {} (fp16: {})", device, use_fp16)
    logger.info("预加载: {}", settings.PRELOAD_MODEL)
    logger.info("允许模型切换: {}", settings.ALLOW_MODEL_SWITCH)
    logger.info("监听地址: {}:{}", settings.DEV_HOST, settings.DEV_PORT)
    logger.info("=" * 60)

    if settings.PRELOAD_MODEL:
        try:
            preload_models()  # 下载 embedding + reranker 到本地
            model_manager.preload(settings.MODEL_NAME)  # 加载 embedding 到内存
            rerank_service.preload(settings.RERANK_MODEL_NAME)  # 加载 reranker 到内存
        except Exception as e:
            logger.error("预加载失败: {}", str(e))
            logger.warning("服务将继续启动，模型将在首次请求时加载")
    else:
        logger.info("模型将在首次请求时懒加载")

    yield  # 应用运行期间

    # === 关闭阶段 ===
    logger.info("lite-emb 正在关闭...")
    # 模型会在 ModelManager 析构时自动清理
    logger.info("lite-emb 已关闭")


def create_app() -> FastAPI:
    """
    创建并配置 FastAPI 应用实例。

    Returns:
        FastAPI: 配置完成的应用实例
    """
    app = FastAPI(
        title="lite-emb",
        description="Lightweight embedding & rerank service — OpenAI /v1/embeddings + Cohere /v1/rerank",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS 中间件（默认允许所有来源，适合嵌入服务场景）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(embeddings.router)
    app.include_router(rerank.router)

    # 全局异常处理器
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """处理值错误（400）。"""
        logger.warning("ValueError: {}", str(exc))
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(exc), "type": "invalid_request_error"}},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """处理未预期的异常（500）。"""
        logger.exception("未处理的异常")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"内部服务器错误: {str(exc)}",
                    "type": "internal_error",
                }
            },
        )

    return app


# 创建应用实例（供 uvicorn 导入）
app = create_app()
