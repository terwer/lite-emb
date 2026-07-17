"""
模型预下载脚本。

默认下载 Embedding + Rerank 两个模型，也支持指定。
用法:
    python preload.py                              # 默认两个
    python preload.py --model bge-reranker-base    # 单独
    python preload.py --model e5-small,bge-m3      # 逗号分隔
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lite_emb.config import settings
from lite_emb.logger import setup_logging
from lite_emb.utils.preload import DEFAULT_MODELS, preload_models

setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="预下载 HuggingFace 模型到本地缓存"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"逗号分隔，默认: {','.join(DEFAULT_MODELS)}",
    )
    args = parser.parse_args()

    models = args.model.split(",") if args.model else DEFAULT_MODELS
    seen: set[str] = set()
    models = [m.strip() for m in models if not (m.strip() in seen or seen.add(m.strip()))]  # type: ignore[func-returns-value]

    print(f"\n{'=' * 60}")
    print("lite-emb 模型预下载工具")
    print(f"{'=' * 60}")
    print(f"模型:  {', '.join(models)}")
    print(f"镜像:  {settings.HF_ENDPOINT}")
    print(f"{'=' * 60}\n")

    try:
        preload_models(models)
        print(f"\n 全部 {len(models)} 个模型下载完成")
    except Exception as e:
        print(f"\n 下载失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
