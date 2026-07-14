"""
模型列表 API 路由。

提供 OpenAI 兼容的 GET /v1/models 端点。
"""

from fastapi import APIRouter

from lite_emb.models.registry import ModelRegistry
from lite_emb.schemas.response import ModelListResponse, ModelObject

router = APIRouter(prefix="/v1", tags=["models"])


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="列出可用模型",
    description="返回所有已注册的嵌入模型列表（OpenAI 兼容格式）。",
)
async def list_models() -> ModelListResponse:
    """
    获取可用的嵌入模型列表。

    返回所有预注册模型的信息，包括模型 ID、类型和所属方。
    """
    models_data = ModelRegistry.list_models()
    model_objects = [ModelObject(**m) for m in models_data]
    return ModelListResponse(data=model_objects)
