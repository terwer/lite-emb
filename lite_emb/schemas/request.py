"""
OpenAI 兼容的请求模型。

完全对齐 OpenAI Embeddings API 的请求格式：
https://platform.openai.com/docs/api-reference/embeddings/create
"""

from typing import Literal, Union

from pydantic import BaseModel, Field, field_validator


class EmbeddingRequest(BaseModel):
    """
    OpenAI 兼容的 Embedding 请求。

    对应的 API: POST /v1/embeddings
    """

    model: str = Field(
        default="bge-m3",
        description="要使用的嵌入模型 ID。支持别名（如 bge-m3）和完整 HuggingFace ID",
        examples=["bge-m3", "BAAI/bge-m3", "all-MiniLM-L6-v2"],
    )
    input: Union[str, list[str], list[int], list[list[int]]] = Field(
        ...,
        description=(
            "要嵌入的输入文本。可以是字符串、字符串数组、token 数组或嵌套 token 数组。"
            "注意：当前仅支持字符串和字符串数组输入。"
        ),
        examples=["The quick brown fox", ["hello", "world"]],
    )
    encoding_format: Literal["float", "base64"] = Field(
        default="float",
        description="返回嵌入向量的格式。base64 格式当前不支持。",
    )
    dimensions: int | None = Field(
        default=None,
        description=(
            "期望的输出向量维度。如果模型支持，将截断到指定维度。"
            "设置为 None 表示使用模型默认维度。"
        ),
        ge=1,
    )
    @field_validator("input")
    @classmethod
    def validate_input_type(cls, v: Union[str, list]) -> Union[str, list]:
        """验证输入类型，仅支持文本输入。"""
        if isinstance(v, list) and len(v) > 0:
            if isinstance(v[0], int) or (
                isinstance(v[0], list) and len(v[0]) > 0 and isinstance(v[0][0], int)
            ):
                raise ValueError(
                    "当前不支持 token 数组输入，请使用文本字符串或字符串数组。"
                )
        return v

    @field_validator("encoding_format")
    @classmethod
    def validate_encoding_format(cls, v: str) -> str:
        """验证编码格式，base64 暂不支持。"""
        if v == "base64":
            raise ValueError("base64 编码格式当前不支持，请使用 'float'。")
        return v
