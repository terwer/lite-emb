"""
模型层模块。

提供模型注册、加载和生命周期管理。
"""

from lite_emb.models.registry import ModelInfo, ModelRegistry
from lite_emb.models.loader import EmbeddingBackend, create_backend
from lite_emb.models.manager import ModelManager, model_manager

__all__ = [
    "ModelInfo",
    "ModelRegistry",
    "EmbeddingBackend",
    "create_backend",
    "ModelManager",
    "model_manager",
]
