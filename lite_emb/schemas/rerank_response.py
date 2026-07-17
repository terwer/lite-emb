"""Cohere 兼容的 Rerank 响应模型。"""  # noqa: D205

from pydantic import BaseModel, Field


class RerankResult(BaseModel):
    """单条 Rerank 结果。"""

    index: int = Field(..., description="文档在原始列表中的索引")
    relevance_score: float = Field(..., description="相关性分数，越高越相关")
    document: str | None = Field(
        default=None,
        description="原文档文本（return_documents=true 时返回）",
    )


class RerankMeta(BaseModel):
    """Rerank 元信息。"""

    api_version: str = Field(default="1.0")


class RerankResponse(BaseModel):
    """Cohere 兼容的 Rerank 响应。"""

    id: str = Field(..., description="请求 ID (UUID)")
    model: str = Field(..., description="使用的模型 ID")
    results: list[RerankResult] = Field(default_factory=list)
    meta: RerankMeta = Field(default_factory=RerankMeta)
