"""
Embedding 推理编排服务。

负责：
- 输入标准化（统一转为文本列表）
- token 数量估算
- 向量维度截断（如果请求了 dimensions 参数）
- 响应格式化
"""

import numpy as np
from loguru import logger

from lite_emb.config import settings
from lite_emb.models.manager import model_manager
from lite_emb.schemas.request import EmbeddingRequest
from lite_emb.schemas.response import (
    EmbeddingObject,
    EmbeddingResponse,
    UsageInfo,
)

# 中英文混合的 token 估算系数
# 粗略估计：中文约 1.5 字符/token，英文约 4 字符/token
TOKEN_ESTIMATE_RATIO = 2.5


def _estimate_tokens(texts: list[str]) -> int:
    """
    估算文本的 token 数量。

    这是一个简化的估算，用于 usage 统计
    （因为并非所有模型都暴露 tokenizer）。

    Args:
        texts: 文本列表

    Returns:
        int: 估算的 token 总数
    """
    total_chars = sum(len(t) for t in texts)
    return max(1, int(total_chars / TOKEN_ESTIMATE_RATIO))


def _normalize_input(request: EmbeddingRequest) -> list[str]:
    """
    将输入标准化为字符串列表。

    Args:
        request: Embedding 请求

    Returns:
        list[str]: 标准化的文本列表
    """
    raw_input = request.input

    if isinstance(raw_input, str):
        return [raw_input]

    if isinstance(raw_input, list):
        if len(raw_input) == 0:
            raise ValueError("输入不能为空列表")
        return [str(item) for item in raw_input]

    raise ValueError(f"不支持的输入类型: {type(raw_input)}")


def _truncate_dimensions(
    embeddings: np.ndarray, target_dim: int | None
) -> np.ndarray:
    """
    截断向量到指定维度（取前 N 维）。

    Args:
        embeddings: shape (N, D) 的向量数组
        target_dim: 目标维度，None 表示不截断

    Returns:
        np.ndarray: 截断后的向量数组
    """
    if target_dim is None:
        return embeddings

    current_dim = embeddings.shape[1]
    if target_dim > current_dim:
        logger.warning(
            "请求维度 {} 超过模型维度 {}，将使用模型维度",
            target_dim,
            current_dim,
        )
        return embeddings

    if target_dim < current_dim:
        logger.info("截断向量从 {} 维到 {} 维", current_dim, target_dim)
        return embeddings[:, :target_dim]

    return embeddings


class EmbeddingService:
    """
    Embedding 推理编排服务。

    将模型管理、编码和响应格式化组合在一起。
    """

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        执行 embedding 推理并返回 OpenAI 兼容响应。

        Args:
            request: Embedding 请求

        Returns:
            EmbeddingResponse: OpenAI 兼容的响应
        """
        # 1. 标准化输入
        texts = _normalize_input(request)
        logger.info("Embedding 请求: model={}, 文本数={}", request.model, len(texts))

        # 2. 获取模型后端
        backend = model_manager.get_model(request.model)
        model_id = model_manager.current_model_id
        display_name = request.model

        # 3. 编码
        embeddings_array = backend.encode(
            texts=texts,
            batch_size=min(settings.MAX_BATCH_SIZE, len(texts)),
            max_length=settings.MAX_SEQUENCE_LENGTH,
            normalize=settings.EMBEDDING_NORMALIZE,
        )

        # 4. 维度截断（如果需要）
        embeddings_array = _truncate_dimensions(embeddings_array, request.dimensions)

        # 5. 构建响应
        data = []
        for i, emb in enumerate(embeddings_array):
            data.append(
                EmbeddingObject(
                    index=i,
                    embedding=emb.tolist(),
                )
            )

        token_count = _estimate_tokens(texts)
        usage = UsageInfo(
            prompt_tokens=token_count,
            total_tokens=token_count,
        )

        logger.info(
            "Embedding 完成: model={}, 文本数={}, 维度={}",
            display_name,
            len(texts),
            embeddings_array.shape[1],
        )

        return EmbeddingResponse(
            model=display_name,
            data=data,
            usage=usage,
        )


# 全局单例服务
embedding_service = EmbeddingService()
