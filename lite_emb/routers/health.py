"""
健康检查 API 路由。

提供 GET /health 和 GET / 端点，用于服务存活性探测。
"""

from fastapi import APIRouter

from lite_emb.models.manager import model_manager

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="健康检查",
    description="返回服务健康状态。",
)
async def health_check() -> dict:
    """
    健康检查端点。

    返回服务是否正在运行。可以扩展为检查模型加载状态等。
    """
    return {"status": "healthy"}


@router.get(
    "/",
    summary="服务信息",
    description="返回服务基本信息，包括当前模型和维度。",
)
async def root() -> dict:
    """
    根路径端点。

    返回服务的基本信息，包括当前加载的模型名称和向量维度。
    """
    model_info = model_manager.get_model_info()
    return {
        "message": "lite-emb embedding service is running",
        "version": "0.1.0",
        "model": model_info["display_name"],
        "model_id": model_info["model_id"],
        "dimension": model_info["dimension"],
        "backend": model_info["backend"],
        "device": model_info["device"],
        "loaded": model_info["loaded"],
        "load_time_seconds": model_info["load_time_seconds"],
    }
