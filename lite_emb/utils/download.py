"""
HuggingFace 模型下载封装。

优先 aria2c（多线程 + 续传 + 进度条），回退 tqdm。
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import snapshot_download
from loguru import logger
from tqdm import tqdm

from lite_emb.config import settings

_MODEL_FILES: dict[str, list[str]] = {
    "intfloat/multilingual-e5-base": [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentence_bert_config.json",
        "modules.json",
        "1_Pooling/config.json",
    ],
    "intfloat/multilingual-e5-small": [
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
    "sentence-transformers/all-MiniLM-L12-v2": [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentence_bert_config.json",
        "modules.json",
        "1_Pooling/config.json",
    ],
    "BAAI/bge-reranker-base": [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ],
    "BAAI/bge-small-zh-v1.5": [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentence_bert_config.json",
        "modules.json",
        "1_Pooling/config.json",
    ],
    "intfloat/multilingual-e5-small": [
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
    "*.bin", "*.safetensors", "*.json", "*.txt",
    "*.model", "tokenizer*", "vocab*", "*.md",
]

_CACHE_ROOT = Path.home() / ".cache" / "lite-emb" / "models"


def _get_endpoint() -> str:
    ep = settings.HF_ENDPOINT or "https://huggingface.co"
    return ep.rstrip("/")


def _has_aria2c() -> bool:
    return shutil.which("aria2c") is not None


def _download_file(url: str, dest: Path, desc: str) -> None:
    """下载单个文件：aria2c 优先，回退 tqdm。"""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if _has_aria2c():
        _download_aria2(url, dest)
    else:
        _download_tqdm(url, dest, desc)


def _download_aria2(url: str, dest: Path) -> None:
    """aria2c 下载 — 自带进度条、速度、ETA。"""
    cmd = [
        "aria2c",
        "--console-log-level=error",
        "--summary-interval=0",
        "-x", "4",           # 4 连接
        "-s", "4",           # 4 线程
        "-k", "1M",          # 分片 1M
        "--continue=true",
        "--dir", str(dest.parent),
        "--out", dest.name,
        url,
    ]
    subprocess.run(cmd, check=True)


def _download_tqdm(url: str, dest: Path, desc: str) -> None:
    """tqdm 回退方案。"""
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, stream=True, timeout=(15, 120))
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))

            with open(dest, "wb") as f:
                with tqdm(
                    total=total, unit="B", unit_scale=True, unit_divisor=1024,
                    desc=desc, file=sys.stderr, miniters=1,
                ) as bar:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                        bar.update(len(chunk))
            return
        except Exception as e:
            if attempt == 3:
                raise RuntimeError(f"下载失败: {e}") from e
            logger.warning("重试 {}/3: {}", attempt + 1, e)
            time.sleep(2)


def ensure_model_downloaded(
    model_id: str,
    local_files_only: bool = False,
    allow_patterns: list[str] | None = None,
) -> str:
    """确保模型文件就绪，返回本地路径。"""
    # 解析别名（懒加载避免循环导入）
    from lite_emb.models.registry import ModelRegistry
    model_id = ModelRegistry.resolve_model_id(model_id)

    if allow_patterns is None:
        allow_patterns = DEFAULT_ALLOW_PATTERNS

    if settings.HF_ENDPOINT:
        os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT

    # 1. HF 缓存
    try:
        local_path = snapshot_download(
            repo_id=model_id,
            local_files_only=True,
            allow_patterns=allow_patterns,
        )
        logger.info("模型就绪（HF 缓存）: {}", local_path)
        return local_path
    except Exception:
        pass

    # 2. lite-emb 缓存
    cache_dir = _CACHE_ROOT / model_id
    if (cache_dir / "config.json").exists():
        logger.info("模型就绪（lite-emb 缓存）: {}", cache_dir)
        return str(cache_dir)

    if local_files_only:
        raise RuntimeError(
            f"离线模式下未找到模型 {model_id}，请先运行 python preload.py"
        )

    # 3. 下载
    files = _MODEL_FILES.get(model_id)
    if files is None:
        logger.info("未知模型，使用 snapshot_download ...")
        try:
            return snapshot_download(
                repo_id=model_id, local_files_only=False,
                allow_patterns=allow_patterns, resume_download=True,
            )
        except Exception as e:
            raise RuntimeError(f"下载失败: {e}") from e

    endpoint = _get_endpoint()
    method = "aria2c" if _has_aria2c() else "tqdm"
    logger.info("下载模型: {} ({} 个文件, {})", model_id, len(files), method)

    for i, fname in enumerate(files, 1):
        dest_file = cache_dir / fname
        if dest_file.exists():
            logger.info("[{}/{}] {} ✓", i, len(files), fname)
            continue

        url = f"{endpoint}/{model_id}/resolve/main/{fname}"
        logger.info("[{}/{}] {}", i, len(files), fname)
        _download_file(url, dest_file, f"[{i}/{len(files)}] {fname}")

    logger.info("下载完成: {}", cache_dir)
    return str(cache_dir)
