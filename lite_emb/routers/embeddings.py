"""
Embedding API 路由。

提供 OpenAI 兼容的 POST /v1/embeddings 端点。
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from lite_emb.schemas.request import EmbeddingRequest
from lite_emb.schemas.response import EmbeddingResponse
from lite_emb.services.embedding_service import embedding_service

router = APIRouter(prefix="/v1", tags=["embeddings"])


@router.post(
    "/embeddings",
    response_model=EmbeddingResponse,
    summary="创建嵌入向量",
    description="OpenAI 兼容的 embedding 端点。将输入文本编码为向量表示。",
)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    生成文本的嵌入向量。

    - **model**: 模型名称（默认 bge-m3）
    - **input**: 要嵌入的文本（字符串或字符串数组）
    - **encoding_format**: 返回格式（仅支持 float）
    - **dimensions**: 可选的输出维度截断
    - **user**: 可选用户标识符
    """
    try:
        return embedding_service.embed(request)
    except ValueError as e:
        logger.warning("请求验证失败: {}", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error("模型加载失败: {}", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Embedding 处理异常")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")
