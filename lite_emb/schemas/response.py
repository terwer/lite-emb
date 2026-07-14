"""
OpenAI 兼容的响应模型。

完全对齐 OpenAI Embeddings API 的响应格式：
https://platform.openai.com/docs/api-reference/embeddings/object
"""

from pydantic import BaseModel, Field


class UsageInfo(BaseModel):
    """Token 使用量信息。"""

    prompt_tokens: int = Field(
        ...,
        description="输入文本的 token 数量",
        examples=[10],
    )
    total_tokens: int = Field(
        ...,
        description="请求的总 token 数量",
        examples=[10],
    )


class EmbeddingObject(BaseModel):
    """单个嵌入向量对象。"""

    object: str = Field(
        default="embedding",
        description="对象类型，固定为 'embedding'",
    )
    index: int = Field(
        ...,
        description="嵌入向量在输入列表中的索引位置",
        examples=[0],
    )
    embedding: list[float] = Field(
        ...,
        description="嵌入向量（浮点数列表）",
        examples=[[0.01, 0.02, -0.03, 0.04]],
    )


class EmbeddingResponse(BaseModel):
    """OpenAI 兼容的 Embedding 响应。"""

    object: str = Field(
        default="list",
        description="对象类型，固定为 'list'",
    )
    model: str = Field(
        ...,
        description="使用的模型名称",
        examples=["bge-m3"],
    )
    data: list[EmbeddingObject] = Field(
        ...,
        description="嵌入向量列表",
    )
    usage: UsageInfo = Field(
        ...,
        description="Token 使用量统计",
    )


class ModelObject(BaseModel):
    """OpenAI 兼容的模型信息对象。"""

    id: str = Field(
        ...,
        description="模型标识符",
    )
    object: str = Field(
        default="model",
        description="对象类型，固定为 'model'",
    )
    created: int = Field(
        ...,
        description="模型注册时间（Unix 时间戳）",
    )
    owned_by: str = Field(
        default="lite-emb",
        description="模型所有者",
    )


class ModelListResponse(BaseModel):
    """OpenAI 兼容的模型列表响应。"""

    object: str = Field(
        default="list",
        description="对象类型，固定为 'list'",
    )
    data: list[ModelObject] = Field(
        ...,
        description="可用模型列表",
    )
