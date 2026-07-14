"""
模型预下载脚本。

在构建 Docker 镜像或离线部署前运行此脚本，
将配置文件中的模型预先下载到本地缓存。

用法:
    python preload.py
    python preload.py --model BAAI/bge-m3
    python preload.py --model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import os
import sys

# 将项目根目录加入 sys.path，确保可以导入 lite_emb 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lite_emb.config import settings
from lite_emb.logger import setup_logging
from lite_emb.utils.download import ensure_model_downloaded

setup_logging()


def main() -> None:
    """预下载模型的主函数。"""
    parser = argparse.ArgumentParser(
        description="预下载 HuggingFace 嵌入模型到本地缓存"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"要下载的模型 ID（默认使用配置中的 MODEL_NAME: {settings.MODEL_NAME}）",
    )
    args = parser.parse_args()

    model_id = args.model or settings.MODEL_NAME
    is_offline = settings.HF_HUB_OFFLINE == 1

    print(f"\n{'=' * 60}")
    print(f"lite-emb 模型预下载工具")
    print(f"{'=' * 60}")
    print(f"模型 ID:    {model_id}")
    print(f"HuggingFace 镜像: {settings.HF_ENDPOINT}")
    print(f"高速传输:  {settings.HF_HUB_ENABLE_HF_TRANSFER}")
    print(f"离线模式:  {bool(is_offline)}")
    print(f"{'=' * 60}\n")

    try:
        local_path = ensure_model_downloaded(
            model_id=model_id,
            local_files_only=is_offline,
        )
        print(f"\n✅ 模型下载完成！")
        print(f"   本地路径: {local_path}")
    except Exception as e:
        print(f"\n❌ 模型下载失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
