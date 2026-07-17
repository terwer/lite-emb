"""Cohere 兼容的 Rerank 请求模型。"""  # noqa: D205

from pydantic import BaseModel, Field, field_validator


class RerankRequest(BaseModel):
    """Cohere 兼容的 Rerank 请求。"""

    model: str = Field(
        default="e5-small",
        description="Rerank 模型 ID。Default 使用 Embedding 模型做 cosine 排序",
        examples=["e5-small", "bge-reranker-base"],
    )
    query: str = Field(
        ...,
        min_length=1,
        description="查询文本",
        examples=["什么是机器学习？"],
    )
    documents: list[str] = Field(
        ...,
        min_length=1,
        description="待排序文档列表",
        examples=[["文本A", "文本B", "文本C"]],
    )
    top_n: int | None = Field(
        default=None,
        ge=1,
        description="返回前 N 条结果。None 表示返回全部",
        examples=[3],
    )
    return_documents: bool = Field(
        default=True,
        description="是否在结果中返回原文档内容",
    )

    @field_validator("documents")
    @classmethod
    def validate_documents(cls, v: list[str]) -> list[str]:
        """校样文档列表不为空且每个文档非空。"""
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("documents 不能为空")
        for i, doc in enumerate(v):
            if not isinstance(doc, str) or not doc.strip():
                raise ValueError(f"documents[{i}] 不能为空字符串")
        return v
