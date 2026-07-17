"""
模型预下载脚本。

默认下载 Embedding + Rerank 两个模型，也支持指定。
用法:
    python preload.py                        # 下载默认两个
    python preload.py --model bge-reranker-base   # 单独下载
    python preload.py --model e5-small,bge-m3  # 逗号分隔多个
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lite_emb.config import settings
from lite_emb.logger import setup_logging
from lite_emb.utils.download import ensure_model_downloaded

setup_logging()

# 默认预下载的模型列表
DEFAULT_MODELS = [settings.MODEL_NAME, settings.RERANK_MODEL_NAME]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="预下载 HuggingFace 模型到本地缓存"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "模型 ID，逗号分隔多个。"
            f"默认: {','.join(DEFAULT_MODELS)}"
        ),
    )
    args = parser.parse_args()

    models = args.model.split(",") if args.model else DEFAULT_MODELS
    # 去重留序
    seen = set()
    models = [m.strip() for m in models if not (m.strip() in seen or seen.add(m.strip()))]  # type: ignore[func-returns-value]

    print(f"\n{'=' * 60}")
    print("lite-emb 模型预下载工具")
    print(f"{'=' * 60}")
    print(f"模型列表:    {', '.join(models)}")
    print(f"镜像:        {settings.HF_ENDPOINT}")
    print(f"离线模式:    {bool(settings.HF_HUB_OFFLINE == 1)}")
    print(f"{'=' * 60}\n")

    failed = []
    for model in models:
        try:
            local_path = ensure_model_downloaded(model_id=model)
            print(f"  {model} -> {local_path}")
        except Exception as e:
            print(f"  {model} -> 失败: {e}")
            failed.append(model)

    if failed:
        print(f"\n{len(failed)} 个模型下载失败: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\n全部 {len(models)} 个模型下载完成")


if __name__ == "__main__":
    main()
