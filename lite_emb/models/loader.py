"""
模型加载器模块。

提供多种模型后端的统一抽象接口：
- BGEM3Loader: 基于 FlagEmbedding 的 BGE-M3 专用后端
- SentenceTransformerLoader: 基于 sentence-transformers 的通用后端
"""

from abc import ABC, abstractmethod

import numpy as np
from loguru import logger

from lite_emb.models.registry import BackendType


class EmbeddingBackend(ABC):
    """Embedding 后端抽象基类。"""

    @abstractmethod
    def load(self, model_path: str, device: str, use_fp16: bool) -> None:
        """
        加载模型到内存。

        Args:
            model_path: 模型本地路径
            device: 计算设备（cpu/cuda/mps）
            use_fp16: 是否使用 fp16 精度
        """

    @abstractmethod
    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        max_length: int = 8192,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        将文本列表编码为向量。

        Args:
            texts: 输入文本列表
            batch_size: 批处理大小
            max_length: 最大序列长度
            normalize: 是否对输出向量进行 L2 归一化

        Returns:
            np.ndarray: shape 为 (len(texts), dimension) 的向量数组
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """模型输出向量维度。"""


class BGEM3Loader(EmbeddingBackend):
    """
    BGE-M3 专用后端。

    使用 FlagEmbedding.BGEM3FlagModel 加载 BGE-M3，
    支持 dense、sparse 和 ColBERT 向量，API 暴露 dense 向量。
    """

    def __init__(self) -> None:
        self._model = None
        self._dimension = 1024

    @property
    def dimension(self) -> int:
        return self._dimension

    def load(self, model_path: str, device: str, use_fp16: bool) -> None:
        """加载 BGE-M3 模型。"""
        from FlagEmbedding import BGEM3FlagModel

        logger.info(
            "正在加载 BGE-M3 模型: {} (设备: {}, fp16: {})",
            model_path,
            device,
            use_fp16,
        )

        self._model = BGEM3FlagModel(
            model_path,
            use_fp16=use_fp16,
            device=device,
        )
        self._dimension = 1024
        logger.info("BGE-M3 模型加载完成，向量维度: {}", self._dimension)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        max_length: int = 8192,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        BGE-M3 编码文本。

        BGE-M3 返回包含 dense_vecs、sparse_vecs 和 colbert_vecs 的字典，
        此处提取 dense_vecs。
        """
        if self._model is None:
            raise RuntimeError("模型尚未加载，请先调用 load()")

        result = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        # BGE-M3 在 return_dense=True 时返回 dict
        if isinstance(result, dict):
            embeddings = result["dense_vecs"]
        else:
            embeddings = result

        embeddings = np.array(embeddings)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)  # 避免除零
            embeddings = embeddings / norms

        return embeddings


class SentenceTransformerLoader(EmbeddingBackend):
    """
    通用 SentenceTransformer 后端。

    兼容所有 HuggingFace sentence-transformers 模型。
    """

    def __init__(self) -> None:
        self._model = None
        self._dimension = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def load(self, model_path: str, device: str, use_fp16: bool) -> None:
        """加载 sentence-transformers 模型。"""
        from sentence_transformers import SentenceTransformer

        logger.info(
            "正在加载 SentenceTransformer 模型: {} (设备: {}, fp16: {})",
            model_path,
            device,
            use_fp16,
        )

        model_kwargs = {}
        if device == "mps":
            model_kwargs["device"] = "mps"
            # MPS 上推荐使用 float32 (sentence-transformers 对 MPS 的 auto 支持不稳定)
            self._model = SentenceTransformer(
                model_path, device="mps", model_kwargs=model_kwargs
            )
        else:
            self._model = SentenceTransformer(model_path, device=device)

        # 获取向量维度
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info("模型加载完成，向量维度: {}", self._dimension)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        max_length: int = 8192,
        normalize: bool = True,
    ) -> np.ndarray:
        """SentenceTransformer 编码文本。"""
        if self._model is None:
            raise RuntimeError("模型尚未加载，请先调用 load()")

        # 设置最大序列长度
        if hasattr(self._model, "max_seq_length"):
            self._model.max_seq_length = max_length

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return np.array(embeddings)


# 后端工厂函数
BACKEND_CLASS_MAP = {
    BackendType.BGE_M3: BGEM3Loader,
    BackendType.SENTENCE_TRANSFORMER: SentenceTransformerLoader,
}


def create_backend(backend_type: BackendType) -> EmbeddingBackend:
    """
    根据后端类型创建加载器实例。

    Args:
        backend_type: 后端类型枚举值

    Returns:
        EmbeddingBackend: 后端实例

    Raises:
        ValueError: 不支持的后端类型
    """
    backend_cls = BACKEND_CLASS_MAP.get(backend_type)
    if backend_cls is None:
        raise ValueError(f"不支持的后端类型: {backend_type}")
    return backend_cls()
