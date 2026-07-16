"""
HuggingFace 模型下载封装。

提供自动下载和本地缓存检测功能，支持：
- 直连 CDN 下载，绕过 HF API（国内网络友好）
- 逐个文件实时 tqdm 进度条
- HF 标准缓存回退
"""

import os
from pathlib import Path

import requests
from huggingface_hub import snapshot_download
from huggingface_hub.utils import enable_progress_bars
from loguru import logger
from tqdm import tqdm

from lite_emb.config import settings

# 已知模型的文件列表（用于直连 CDN 下载，无需 HF API）
_MODEL_FILES: dict[str, list[str]] = {
    "BAAI/bge-micro": [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentence_bert_config.json",
        "modules.json",
        "1_Pooling/config.json",
    ],
    "sentence-transformers/all-MiniLM-L6-v2": [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentence_bert_config.json",
        "modules.json",
        "1_Pooling/config.json",
    ],
}

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

# 独立缓存目录（直连下载时的存储位置）
_CACHE_ROOT = Path.home() / ".cache" / "lite-emb" / "models"


def _get_endpoint() -> str:
    ep = settings.HF_ENDPOINT or "https://huggingface.co"
    return ep.rstrip("/")


def _download_with_progress(url: str, dest: Path, desc: str) -> None:
    """带 tqdm 进度条下载文件。"""
    resp = requests.get(url, stream=True, timeout=(10, 60))
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=desc,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))


def ensure_model_downloaded(
    model_id: str,
    local_files_only: bool = False,
    allow_patterns: list[str] | None = None,
) -> str:
    """
    确保模型文件就绪，返回本地路径。

    策略：
    1. 有 HF 缓存 → 直接返回
    2. 已知文件列表 → 直连 CDN 下载，存入独立缓存
    3. 未知模型 → 回退到 snapshot_download

    Args:
        model_id: HuggingFace 模型 ID
        local_files_only: 是否仅使用本地文件
        allow_patterns: 文件匹配模式

    Returns:
        str: 模型本地路径
    """
    if allow_patterns is None:
        allow_patterns = DEFAULT_ALLOW_PATTERNS

    if settings.HF_ENDPOINT:
        os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT

    # 1. HF 标准缓存
    try:
        local_path = snapshot_download(
            repo_id=model_id,
            local_files_only=True,
            allow_patterns=allow_patterns,
        )
        logger.info("模型就绪（HF 缓存），路径: {}", local_path)
        return local_path
    except Exception:
        pass

    # 2. 独立 lite-emb 缓存
    cache_dir = _CACHE_ROOT / model_id
    key_file = cache_dir / "config.json"
    if key_file.exists():
        logger.info("模型就绪（lite-emb 缓存），路径: {}", cache_dir)
        return str(cache_dir)

    if local_files_only:
        raise RuntimeError(
            f"离线模式下未找到模型 {model_id}。"
            f"请先运行 'python preload.py' 下载模型，或设置 HF_HUB_OFFLINE=0。"
        )

    # 3. 已知模型：直连 CDN 下载
    files = _MODEL_FILES.get(model_id)
    if files is not None:
        return _download_from_cdn(model_id, files, cache_dir)

    # 4. 未知模型：回退 snapshot_download
    return _download_via_snapshot(model_id, allow_patterns)


def _download_from_cdn(model_id: str, files: list[str], dest: Path) -> str:
    """直连 CDN 下载文件到独立缓存目录。"""
    endpoint = _get_endpoint()
    logger.info("直连 CDN 下载: {} ({} 个文件)", model_id, len(files))

    for i, filename in enumerate(files, 1):
        url = f"{endpoint}/{model_id}/resolve/main/{filename}"
        dest_file = dest / filename

        if dest_file.exists():
            logger.info("[{}/{}] {} ✓", i, len(files), filename)
            continue

        logger.info("[{}/{}] {} ...", i, len(files), filename)
        try:
            _download_with_progress(url, dest_file, f"[{i}/{len(files)}] {filename}")
        except Exception as e:
            logger.error("下载失败: {} - {}", filename, e)
            raise RuntimeError(f"下载文件失败 {filename}: {e}") from e

    logger.info("模型下载完成 ({} 个文件), 路径: {}", len(files), dest)
    return str(dest)


def _download_via_snapshot(model_id: str, allow_patterns: list[str]) -> str:
    """回退方案：使用 snapshot_download（需要 HF API 可用）。"""
    logger.info("使用 snapshot_download: {} ...", model_id)
    enable_progress_bars()

    hf_transfer_original = os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    try:
        local_path = snapshot_download(
            repo_id=model_id,
            local_files_only=False,
            allow_patterns=allow_patterns,
            resume_download=True,
        )
        logger.info("下载完成，路径: {}", local_path)
        return local_path
    except Exception as e:
        raise RuntimeError(f"下载模型 {model_id} 失败: {e}") from e
    finally:
        if hf_transfer_original is not None:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = hf_transfer_original
