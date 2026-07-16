"""
HuggingFace 模型下载封装。

提供自动下载和本地缓存检测功能，支持：
- 国内镜像加速
- hf-transfer 高速下载（可选关闭以显示进度）
- 离线模式
- 选择性下载文件类型
- 逐个文件进度日志 + tqdm 进度条
"""

import os

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
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


def _matches_patterns(filename: str, patterns: list[str]) -> bool:
    """检查文件名是否匹配任一 glob 模式。"""
    import fnmatch

    return any(fnmatch.fnmatch(filename, p) for p in patterns)


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

    如果模型未缓存且不处于离线模式，自动从 HuggingFace 下载。
    逐个文件下载，每个文件有独立日志和 tqdm 进度条。

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

    logger.info("正在获取文件列表: {} ...", model_id)

    try:
        files = list_repo_files(model_id)
    except Exception as e:
        raise RuntimeError(f"获取模型文件列表失败: {model_id}，错误: {e}") from e

    target_files = [f for f in files if _matches_patterns(f, allow_patterns)]
    logger.info("共 {} 个文件需下载，总 {} 个文件", len(target_files), len(files))

    # 禁用 hf-transfer 以确保 tqdm 进度条可见
    hf_transfer_original = os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    enable_progress_bars()

    try:
        for i, filename in enumerate(target_files, 1):
            logger.info("[{}/{}] 下载: {}", i, len(target_files), filename)
            try:
                hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    resume_download=True,
                )
            except Exception as e:
                logger.error("下载失败: {} - {}", filename, e)
                raise

        logger.info("所有文件下载完成 ({} 个)", len(target_files))
    finally:
        if hf_transfer_original is not None:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = hf_transfer_original

    # 再次调用 snapshot_download 以获取统一路径（此时全部命中缓存）
    local_path = snapshot_download(
        repo_id=model_id,
        local_files_only=True,
        allow_patterns=allow_patterns,
    )
    logger.info("模型就绪，路径: {}", local_path)
    return local_path
