"""
日志初始化模块。

使用 loguru 提供结构化日志，包含：
- 控制台彩色输出
- 文件日志轮转
- 标准库 logging 桥接
"""

import logging
import sys

from loguru import logger

from lite_emb.config import settings


class InterceptHandler(logging.Handler):
    """将标准库 logging 日志桥接到 loguru。"""

    def emit(self, record: logging.LogRecord) -> None:
        """将 logging.LogRecord 转发到 loguru。"""
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging() -> None:
    """初始化 loguru 日志系统。"""
    # 移除默认 handler
    logger.remove()

    # 控制台输出（彩色）
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=settings.LOG_LEVEL,
        colorize=True,
    )

    # 文件输出（带轮转）
    logger.add(
        settings.LOG_FILE,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.LOG_LEVEL,
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        enqueue=True,
    )

    # 桥接标准库 logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # 静默一些过于冗长的第三方日志
    for lib in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(lib).handlers = [InterceptHandler()]

    # 保留 huggingface_hub 的原生日志输出（含下载进度）
    for lib in ("huggingface_hub", "huggingface_hub.file_download"):
        logging.getLogger(lib).handlers = []

    logger.info("日志系统初始化完成，级别: {}", settings.LOG_LEVEL)
