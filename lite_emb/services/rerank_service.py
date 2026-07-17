"""
Rerank 推理编排服务。

支持双模式：
- cosine 模式：用 Embedding 模型做余弦相似度排序（轻量、零额外成本）
- cross-encoder 模式：用专业 Reranker 模型（高精度）
"""

import uuid
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from lite_emb.config import settings
from lite_emb.models.loader import RerankerBackend, create_backend
from lite_emb.models.registry import BackendType, ModelRegistry
from lite_emb.schemas.rerank_request import RerankRequest
from lite_emb.schemas.rerank_response import (
    RerankMeta,
    RerankResponse,
    RerankResult,
)
from lite_emb.utils.device import get_device_and_precision
from lite_emb.utils.download import ensure_model_downloaded

if TYPE_CHECKING:
    from lite_emb.services.embedding_service import EmbeddingService


def _is_reranker_model(model_id: str) -> bool:
    """判断是否为专业 Reranker 模型。"""
    backend_type = ModelRegistry.get_backend_type(model_id)
    return backend_type == BackendType.RERANKER


class RerankService:
    """Rerank 推理编排服务。"""

    def __init__(self) -> None:
        self._reranker: RerankerBackend | None = None
        self._reranker_model_id: str | None = None

    def rerank(
        self,
        request: RerankRequest,
        embedding_service: "EmbeddingService | None" = None,
    ) -> RerankResponse:
        """
        执行 Rerank。

        Args:
            request: Rerank 请求
            embedding_service: Embedding 服务（cosine 模式需要）

        Returns:
            RerankResponse: Cohere 兼容响应
        """
        model_id = ModelRegistry.resolve_model_id(request.model)

        if _is_reranker_model(model_id):
            return self._cross_encoder_rerank(request, model_id)
        else:
            return self._cosine_rerank(request, model_id, embedding_service)

    def _cosine_rerank(
        self,
        request: RerankRequest,
        model_id: str,
        embedding_service: "EmbeddingService | None",
    ) -> RerankResponse:
        """使用 Embedding 模型做 cosine 相似度排序。"""
        if embedding_service is None:
            raise RuntimeError("cosine rerank 需要 embedding_service")

        from lite_emb.schemas.request import EmbeddingRequest

        logger.info(
            "cosine rerank: model={}, docs={}",
            model_id,
            len(request.documents),
        )

        # 1. Embed query
        q_req = EmbeddingRequest(model=request.model, input=request.query)
        q_resp = embedding_service.embed(q_req)
        query_vec = np.array(q_resp.data[0].embedding)

        # 2. Embed documents
        d_req = EmbeddingRequest(model=request.model, input=request.documents)
        d_resp = embedding_service.embed(d_req)
        doc_vecs = np.array([d.embedding for d in d_resp.data])

        # 3. Cosine similarity
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
        scores = np.dot(doc_norms, query_norm)

        # 4. Sort & top_n
        return self._build_response(
            request=request,
            model_id=model_id,
            scores=scores.tolist(),
        )

    def _cross_encoder_rerank(
        self,
        request: RerankRequest,
        model_id: str,
    ) -> RerankResponse:
        """使用 Cross-Encoder Reranker。"""
        logger.info(
            "cross-encoder rerank: model={}, docs={}",
            model_id,
            len(request.documents),
        )

        # 1. Load model if needed
        if self._reranker_model_id != model_id or self._reranker is None:
            self._load_reranker(model_id)

        # 2. Build (query, doc) pairs
        pairs = [[request.query, doc] for doc in request.documents]

        # 3. Compute scores
        scores = self._reranker.compute_score(pairs)  # type: ignore[union-attr]

        # 4. Sort & top_n
        return self._build_response(
            request=request,
            model_id=model_id,
            scores=scores,
        )

    def _load_reranker(self, model_id: str) -> None:
        """加载 Reranker 模型。"""
        logger.info("开始加载 Reranker: {}", model_id)

        device, use_fp16 = get_device_and_precision()
        is_offline = settings.HF_HUB_OFFLINE == 1

        local_path = ensure_model_downloaded(
            model_id=model_id,
            local_files_only=is_offline,
        )

        backend = create_backend(BackendType.RERANKER)
        backend.load(model_path=local_path, device=device, use_fp16=use_fp16)

        self._reranker = backend  # type: ignore[assignment]
        self._reranker_model_id = model_id

        logger.info("Reranker 加载完成: {}", model_id)

    def _build_response(
        self,
        request: RerankRequest,
        model_id: str,
        scores: list[float],
    ) -> RerankResponse:
        """按分数排序，截取 top_n，构建响应。"""
        # 按分数降序排列
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        top_n = request.top_n
        if top_n is not None:
            indexed = indexed[:top_n]

        results = []
        for idx, score in indexed:
            doc = request.documents[idx] if request.return_documents else None
            results.append(
                RerankResult(
                    index=idx,
                    relevance_score=float(score),
                    document=doc,
                )
            )

        return RerankResponse(
            id=str(uuid.uuid4()),
            model=model_id,
            results=results,
            meta=RerankMeta(),
        )


# 全局单例
rerank_service = RerankService()
