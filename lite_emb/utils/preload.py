"""共享预下载逻辑，preload.py 和 app.py 共用。"""  # noqa: D205

from loguru import logger

from lite_emb.config import settings
from lite_emb.utils.download import ensure_model_downloaded

# 默认预下载模型列表（与 preload.py 脚本保持一致）
DEFAULT_MODELS = [settings.MODEL_NAME, settings.RERANK_MODEL_NAME]


def preload_models(models: list[str] | None = None) -> None:
    """
    下载模型到本地缓存。

    下载后不加载到内存（加载由 ModelManager/RerankService 各自处理）。

    Args:
        models: 模型 ID 列表，默认 [embedding, reranker]
    """
    targets = models or DEFAULT_MODELS
    logger.info("预下载模型: {}", ", ".join(targets))

    for model in targets:
        logger.info("下载: {}", model)
        ensure_model_downloaded(model_id=model)

    logger.info("预下载完成 ({} 个模型)", len(targets))
