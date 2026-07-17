"""Rerank API 路由 — Cohere /v1/rerank 兼容。"""  # noqa: D205

from fastapi import APIRouter, HTTPException

from lite_emb.schemas.rerank_request import RerankRequest
from lite_emb.schemas.rerank_response import RerankResponse
from lite_emb.services.embedding_service import embedding_service
from lite_emb.services.rerank_service import rerank_service

router = APIRouter()


@router.post(
    "/v1/rerank",
    response_model=RerankResponse,
    summary="Rerank documents by relevance to a query",
    description="""
Cohere-compatible rerank endpoint. Supports two modes:

- **Cross-encoder** (`bge-reranker-base`): Professional reranker model for highest accuracy.
- **Cosine** (`e5-small`, etc.): Uses the embedding model for lightweight cosine similarity ranking.

Request format matches [Cohere Rerank API](https://docs.cohere.com/reference/rerank).
""",
)
def rerank(request: RerankRequest) -> RerankResponse:
    """POST /v1/rerank — 文档重排序。"""
    try:
        return rerank_service.rerank(request, embedding_service=embedding_service)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {e}") from e
