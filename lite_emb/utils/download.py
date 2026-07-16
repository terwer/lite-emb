"""
HuggingFace 模型下载封装。

提供自动下载和本地缓存检测功能，支持：
- 国内镜像加速
- 离线模式
- 选择性下载文件类型
- 实时进度条（禁用 hf-transfer 时）
"""

import os

from huggingface_hub import snapshot_download
from huggingface_hub.utils import enable_progress_bars
from loguru import logger

from lite_emb.config import settings

# 默认只下载模型权重和配置文件
DEFAULT_ALLOW_PATTERNS = [
    "*.bin",
    "*.safetensors",
    "*.json",
    "*.txt",
    "*.model",
    "tokenizer*",
    "vocab*",
    "*.md",
]


def _setup_download_env() -> None:
    """配置 HuggingFace 下载环境变量。"""
    if settings.HF_ENDPOINT:
        os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT


def ensure_model_downloaded(
    model_id: str,
    local_files_only: bool = False,
    allow_patterns: list[str] | None = None,
) -> str:
    """
    确保模型文件已下载到本地缓存。

    优先使用本地缓存，无缓存时自动下载并显示 tqdm 进度条。

    Args:
        model_id: HuggingFace 模型 ID（如 "BAAI/bge-m3"）
        local_files_only: 是否仅使用本地文件（离线模式）
        allow_patterns: 允许下载的文件匹配模式

    Returns:
        str: 本地缓存路径

    Raises:
        RuntimeError: 离线模式下模型文件缺失时抛出或下载失败
    """
    if allow_patterns is None:
        allow_patterns = DEFAULT_ALLOW_PATTERNS

    _setup_download_env()

    # 1. 优先本地缓存
    try:
        local_path = snapshot_download(
            repo_id=model_id,
            local_files_only=True,
            allow_patterns=allow_patterns,
        )
        logger.info("模型就绪（本地缓存），路径: {}", local_path)
        return local_path
    except Exception:
        pass

    if local_files_only:
        raise RuntimeError(
            f"离线模式下未找到模型 {model_id} 的本地缓存文件。"
            f"请先运行 'python preload.py' 下载模型，或设置 HF_HUB_OFFLINE=0 启用在线下载。"
        )

    # 2. 下载模式：禁用 hf-transfer 以显示 tqdm 进度条
    logger.info("正在下载模型: {} ...", model_id)
    enable_progress_bars()

    hf_transfer_original = os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    try:
        local_path = snapshot_download(
            repo_id=model_id,
            local_files_only=False,
            allow_patterns=allow_patterns,
            resume_download=True,
        )
        logger.info("模型下载完成，路径: {}", local_path)
        return local_path
    except Exception as e:
        raise RuntimeError(f"下载模型 {model_id} 失败: {e}") from e
    finally:
        if hf_transfer_original is not None:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = hf_transfer_original
