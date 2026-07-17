"""
模型注册表模块。

定义所有已知的 embedding 模型元数据，支持自动发现未知模型。
"""

from dataclasses import dataclass, field
from enum import Enum


class BackendType(str, Enum):
    """模型后端类型枚举。"""

    BGE_M3 = "bge_m3"
    """BGE-M3 专用后端，使用 FlagEmbedding.BGEM3FlagModel"""

    SENTENCE_TRANSFORMER = "sentence_transformer"
    """通用 sentence-transformers 后端"""

    RERANKER = "reranker"
    """Cross-encoder reranker 后端，使用 FlagEmbedding.FlagReranker"""


@dataclass
class ModelInfo:
    """模型元数据信息。"""

    model_id: str
    """HuggingFace 模型 ID"""

    display_name: str
    """显示名称"""

    dimension: int
    """向量维度"""

    max_seq_length: int
    """最大序列长度"""

    backend_type: BackendType
    """推荐的后端类型"""

    description: str = ""
    """模型描述"""

    tags: list[str] = field(default_factory=list)
    """模型标签（如 multilingual, chinese, english 等）"""


class ModelRegistry:
    """
    模型注册表。

    维护已知模型的元数据清单，支持按模型 ID 查询。
    未知模型会触发自动发现机制。
    """

    # 预注册的已知模型
    KNOWN_MODELS: dict[str, ModelInfo] = {
        "BAAI/bge-m3": ModelInfo(
            model_id="BAAI/bge-m3",
            display_name="bge-m3",
            dimension=1024,
            max_seq_length=8192,
            backend_type=BackendType.BGE_M3,
            description="BGE-M3 多语言嵌入模型，支持中英文及100+语言",
            tags=["multilingual", "chinese", "english", "dense", "sparse", "colbert"],
        ),
        "BAAI/bge-large-zh-v1.5": ModelInfo(
            model_id="BAAI/bge-large-zh-v1.5",
            display_name="bge-large-zh-v1.5",
            dimension=1024,
            max_seq_length=512,
            backend_type=BackendType.SENTENCE_TRANSFORMER,
            description="BGE 中文大模型 v1.5",
            tags=["chinese", "dense"],
        ),
        "BAAI/bge-small-zh-v1.5": ModelInfo(
            model_id="BAAI/bge-small-zh-v1.5",
            display_name="bge-small-zh-v1.5",
            dimension=512,
            max_seq_length=512,
            backend_type=BackendType.SENTENCE_TRANSFORMER,
            description="BGE 中文小模型 v1.5",
            tags=["chinese", "dense", "lightweight"],
        ),
        "sentence-transformers/all-MiniLM-L6-v2": ModelInfo(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            display_name="all-MiniLM-L6-v2",
            dimension=384,
            max_seq_length=256,
            backend_type=BackendType.SENTENCE_TRANSFORMER,
            description="轻量级英文句子嵌入模型",
            tags=["english", "dense", "lightweight"],
        ),
        "intfloat/multilingual-e5-large": ModelInfo(
            model_id="intfloat/multilingual-e5-large",
            display_name="multilingual-e5-large",
            dimension=1024,
            max_seq_length=512,
            backend_type=BackendType.SENTENCE_TRANSFORMER,
            description="多语言 E5 大模型，支持 100+ 语言",
            tags=["multilingual", "dense", "e5"],
        ),
        "intfloat/multilingual-e5-base": ModelInfo(
            model_id="intfloat/multilingual-e5-base",
            display_name="multilingual-e5-base",
            dimension=768,
            max_seq_length=512,
            backend_type=BackendType.SENTENCE_TRANSFORMER,
            description="多语言 E5 中等模型，支持 100+ 语言",
            tags=["multilingual", "dense", "e5"],
        ),
        "intfloat/multilingual-e5-small": ModelInfo(
            model_id="intfloat/multilingual-e5-small",
            display_name="multilingual-e5-small",
            dimension=384,
            max_seq_length=512,
            backend_type=BackendType.SENTENCE_TRANSFORMER,
            description="多语言 E5 小模型，支持 100+ 语言，仅 120MB",
            tags=["multilingual", "dense", "e5", "lightweight"],
        ),
        "BAAI/bge-reranker-base": ModelInfo(
            model_id="BAAI/bge-reranker-base",
            display_name="bge-reranker-base",
            dimension=0,  # cross-encoder 无固定维度
            max_seq_length=512,
            backend_type=BackendType.RERANKER,
            description="BGE Cross-Encoder Reranker，中英文重排序",
            tags=["multilingual", "chinese", "english", "reranker", "cross-encoder"],
        ),
        "BAAI/bge-micro": ModelInfo(
            model_id="BAAI/bge-micro",
            display_name="bge-micro",
            dimension=384,
            max_seq_length=512,
            backend_type=BackendType.SENTENCE_TRANSFORMER,
            description="BGE 超轻量多语言模型，仅 17MB",
            tags=["multilingual", "dense", "ultralight"],
        ),
        "sentence-transformers/all-MiniLM-L12-v2": ModelInfo(
            model_id="sentence-transformers/all-MiniLM-L12-v2",
            display_name="all-MiniLM-L12-v2",
            dimension=384,
            max_seq_length=256,
            backend_type=BackendType.SENTENCE_TRANSFORMER,
            description="轻量英文嵌入模型 L12 版本",
            tags=["english", "dense", "lightweight"],
        ),
    }

    # 模型 ID 到别名的映射（用于请求中的 model 参数）
    ALIAS_MAP: dict[str, str] = {
        "bge-m3": "BAAI/bge-m3",
        "bge-m3-zh": "BAAI/bge-m3",
        "bge-large-zh": "BAAI/bge-large-zh-v1.5",
        "bge-small-zh": "BAAI/bge-small-zh-v1.5",
        "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "e5-large": "intfloat/multilingual-e5-large",
        "e5-base": "intfloat/multilingual-e5-base",
        "e5-small": "intfloat/multilingual-e5-small",
        "bge-micro": "BAAI/bge-micro",
        "all-minilm-l12": "sentence-transformers/all-MiniLM-L12-v2",
        "bge-reranker-base": "BAAI/bge-reranker-base",
    }

    @classmethod
    def resolve_model_id(cls, model_name: str) -> str:
        """
        解析模型名称。

        先查别名映射，再查已知模型，最后返回原始名称（触发自动发现）。

        Args:
            model_name: 用户提供的模型名称（可能是别名或完整 ID）

        Returns:
            str: 解析后的 HuggingFace 模型 ID
        """
        # 1. 检查别名
        if model_name.lower() in cls.ALIAS_MAP:
            return cls.ALIAS_MAP[model_name.lower()]

        # 2. 检查已知模型
        if model_name in cls.KNOWN_MODELS:
            return model_name

        # 3. 直接作为 HuggingFace ID 使用（自动发现）
        return model_name

    @classmethod
    def get_model_info(cls, model_id: str) -> ModelInfo | None:
        """
        获取模型元数据。

        Args:
            model_id: HuggingFace 模型 ID

        Returns:
            ModelInfo | None: 模型信息，未知模型返回 None
        """
        return cls.KNOWN_MODELS.get(model_id)

    @classmethod
    def get_backend_type(cls, model_id: str) -> BackendType:
        """
        确定应该使用哪个后端加载模型。

        已知模型使用注册的后端类型，未知模型默认使用 SentenceTransformer。

        Args:
            model_id: HuggingFace 模型 ID

        Returns:
            BackendType: 后端类型
        """
        info = cls.get_model_info(model_id)
        if info is not None:
            return info.backend_type

        # 自动发现：BGE-M3 系列使用 BGE_M3 后端
        if "bge-m3" in model_id.lower():
            return BackendType.BGE_M3

        # 自动发现：Reranker 模型
        if "reranker" in model_id.lower():
            return BackendType.RERANKER

        # 默认使用 SentenceTransformer
        return BackendType.SENTENCE_TRANSFORMER

    @classmethod
    def list_models(cls) -> list[dict]:
        """
        以 OpenAI 兼容格式返回所有已知模型列表。

        Returns:
            list[dict]: 模型列表
        """
        import time

        models = []
        for model_id, info in cls.KNOWN_MODELS.items():
            models.append(
                {
                    "id": info.display_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "lite-emb",
                }
            )
        return models
