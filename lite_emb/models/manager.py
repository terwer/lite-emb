"""
模型生命周期管理器。

负责：
- 模型的懒加载与缓存
- 线程安全的模型切换
- 模型状态查询
"""

import threading
import time

from loguru import logger

from lite_emb.config import settings
from lite_emb.models.loader import EmbeddingBackend, create_backend
from lite_emb.models.registry import BackendType, ModelInfo, ModelRegistry
from lite_emb.utils.device import get_device_and_precision
from lite_emb.utils.download import ensure_model_downloaded


class ModelManager:
    """
    模型生命周期管理器。

    维护已加载模型实例的缓存，支持懒加载、热切换和状态查询。
    所有状态变更操作都是线程安全的。

    使用方式:
        manager = ModelManager()
        backend = manager.get_model("BAAI/bge-m3")
        embeddings = backend.encode(["hello world"])
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_model_id: str | None = None
        self._current_backend: EmbeddingBackend | None = None
        self._load_time: float = 0.0

    @property
    def current_model_id(self) -> str | None:
        """当前已加载的模型 ID。"""
        return self._current_model_id

    @property
    def current_backend(self) -> EmbeddingBackend | None:
        """当前已加载的后端实例。"""
        return self._current_backend

    @property
    def current_dimension(self) -> int:
        """当前模型的向量维度。"""
        if self._current_backend is None:
            return 0
        return self._current_backend.dimension

    def get_model(self, model_name: str) -> EmbeddingBackend:
        """
        获取模型实例（懒加载 + 缓存）。

        如果请求的模型与当前加载的不同：
        - 如果 ALLOW_MODEL_SWITCH=true，自动切换
        - 如果 ALLOW_MODEL_SWITCH=false，抛出错误

        Args:
            model_name: 模型名称或 HuggingFace ID

        Returns:
            EmbeddingBackend: 已加载的后端实例

        Raises:
            ValueError: 模型切换被禁用时请求不同模型
            RuntimeError: 模型加载失败
        """
        model_id = ModelRegistry.resolve_model_id(model_name)

        # 如果已经是当前模型，直接返回
        if self._current_model_id == model_id and self._current_backend is not None:
            return self._current_backend

        # 需要切换模型
        with self._lock:
            # 双重检查锁定
            if self._current_model_id == model_id and self._current_backend is not None:
                return self._current_backend

            if self._current_backend is not None and not settings.ALLOW_MODEL_SWITCH:
                raise ValueError(
                    f"模型切换被禁用（ALLOW_MODEL_SWITCH=false）。"
                    f"当前模型: {self._current_model_id}，"
                    f"请求模型: {model_id}。"
                    f"请使用相同模型或启用 ALLOW_MODEL_SWITCH。"
                )

            return self._load_model(model_id)

    def switch_model(self, model_name: str) -> EmbeddingBackend:
        """
        显式切换到指定模型。

        Args:
            model_name: 目标模型名称或 HuggingFace ID

        Returns:
            EmbeddingBackend: 新加载的后端实例
        """
        model_id = ModelRegistry.resolve_model_id(model_name)

        with self._lock:
            if self._current_model_id == model_id and self._current_backend is not None:
                logger.info("模型已加载，无需切换: {}", model_id)
                return self._current_backend

            # 卸载旧模型
            if self._current_backend is not None:
                logger.info("正在卸载旧模型: {}", self._current_model_id)
                self._unload_current()

            return self._load_model(model_id)

    def get_model_info(self) -> dict:
        """
        获取当前模型的摘要信息。

        Returns:
            dict: 包含 model_id、dimension、load_time 等信息
        """
        backend = self._current_backend
        model_id = self._current_model_id or "N/A"

        info = ModelRegistry.get_model_info(model_id) if model_id != "N/A" else None

        return {
            "model_id": model_id,
            "display_name": info.display_name if info else model_id,
            "dimension": backend.dimension if backend else 0,
            "backend": type(backend).__name__ if backend else "N/A",
            "loaded": backend is not None,
            "load_time_seconds": round(self._load_time, 2) if self._load_time else 0,
            "device": get_device_and_precision()[0],
        }

    def preload(self, model_name: str | None = None) -> EmbeddingBackend:
        """
        启动时预加载模型。

        Args:
            model_name: 模型名称，默认使用配置中的 MODEL_NAME

        Returns:
            EmbeddingBackend: 已加载的后端实例
        """
        target = model_name or settings.MODEL_NAME
        logger.info("启动预加载模型: {}", target)
        return self.get_model(target)

    def _load_model(self, model_id: str) -> EmbeddingBackend:
        """
        内部方法：加载指定模型。

        调用此方法前必须持有 self._lock。
        """
        logger.info("=" * 60)
        logger.info("开始加载模型: {}", model_id)
        start_time = time.time()

        # 1. 确定后端类型
        backend_type = ModelRegistry.get_backend_type(model_id)
        logger.info("后端类型: {}", backend_type.value)

        # 2. 创建后端实例
        backend = create_backend(backend_type)

        # 3. 检测设备
        device, use_fp16 = get_device_and_precision()
        logger.info("计算设备: {} (fp16: {})", device, use_fp16)

        # 4. 下载/检查模型文件
        is_offline = settings.HF_HUB_OFFLINE == 1
        local_path = ensure_model_downloaded(
            model_id=model_id,
            local_files_only=is_offline,
        )

        # 5. 加载模型
        backend.load(model_path=local_path, device=device, use_fp16=use_fp16)

        # 6. 更新状态
        self._current_model_id = model_id
        self._current_backend = backend
        self._load_time = time.time() - start_time

        logger.info(
            "模型加载完成: {} (耗时: {:.2f}s, 维度: {})",
            model_id,
            self._load_time,
            backend.dimension,
        )
        logger.info("=" * 60)

        return backend

    def _unload_current(self) -> None:
        """
        内部方法：卸载当前模型，释放 GPU 内存。

        调用此方法前必须持有 self._lock。
        """
        import gc

        import torch

        if self._current_backend is not None:
            del self._current_backend
            self._current_backend = None
            self._current_model_id = None
            self._load_time = 0.0

        # 清理 GPU 缓存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        logger.info("旧模型已卸载，GPU 缓存已清理")


# 全局单例模型管理器
model_manager = ModelManager()
