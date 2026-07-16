"""
HuggingFace 模型下载封装。

提供自动下载和本地缓存检测功能，支持：
- 国内镜像加速
- hf-transfer 高速下载
- 离线模式
- 选择性下载文件类型
- tqdm 实时进度显示
"""

import os
import sys

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

    if settings.HF_HUB_ENABLE_HF_TRANSFER:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def ensure_model_downloaded(
    model_id: str,
    local_files_only: bool = False,
    allow_patterns: list[str] | None = None,
) -> str:
    """
    确保模型文件已下载到本地缓存。

    如果模型未缓存且不处于离线模式，自动从 HuggingFace 下载，
    并显示 tqdm 实时进度条。

    Args:
        model_id: HuggingFace 模型 ID（如 "BAAI/bge-m3"）
        local_files_only: 是否仅使用本地文件（离线模式）
        allow_patterns: 允许下载的文件匹配模式

    Returns:
        str: 本地缓存路径

    Raises:
        RuntimeError: 离线模式下模型文件缺失时抛出
    """
    if allow_patterns is None:
        allow_patterns = DEFAULT_ALLOW_PATTERNS

    _setup_download_env()

    # 优先尝试本地缓存，避免不必要的网络请求
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

    # 本地无缓存，检查是否强制离线
    if local_files_only:
        raise RuntimeError(
            f"离线模式下未找到模型 {model_id} 的本地缓存文件。"
            f"请先运行 'python preload.py' 下载模型，或设置 HF_HUB_OFFLINE=0 启用在线下载。"
        )

    logger.info(
        "本地无缓存，开始下载模型: {} (镜像: {})",
        model_id,
        settings.HF_ENDPOINT,
    )

    # 临时禁用 hf-transfer 以启用 tqdm 进度条
    # hf-transfer (Rust) 不输出进度，下载大文件时用户无法判断状态
    hf_transfer_original = os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

    try:
        # 强制启用 huggingface_hub 的 tqdm 进度条
        enable_progress_bars()

        local_path = snapshot_download(
            repo_id=model_id,
            local_files_only=False,
            allow_patterns=allow_patterns,
            resume_download=True,
            # tqdm 输出到 stderr，确保在终端可见
            tqdm_class=None,  # 使用默认 tqdm
        )
        logger.info("模型下载完成，路径: {}", local_path)
        return local_path
    except Exception as e:
        raise RuntimeError(f"下载模型 {model_id} 失败: {e}") from e
    finally:
        # 恢复 hf-transfer 设置
        if hf_transfer_original is not None:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = hf_transfer_original