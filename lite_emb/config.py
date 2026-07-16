"""
应用配置管理模块。

使用 pydantic-settings 从 .env 文件和环境变量加载配置，
所有配置项都有合理的默认值。
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用全局配置，从 .env 文件加载。"""

    model_config = {"env_file": ".env", "extra": "forbid"}

    # -- 服务器配置 --
    DEV_HOST: str = "0.0.0.0"
    DEV_PORT: int = 8000
    WORKERS: int = 1

    # -- 模型配置 --
    MODEL_NAME: str = "e5-small"
    MAX_BATCH_SIZE: int = 32
    MAX_SEQUENCE_LENGTH: int = 8192

    # -- 预加载配置 --
    PRELOAD_MODEL: bool = False
    ALLOW_MODEL_SWITCH: bool = False

    # -- Embedding 配置 --
    EMBEDDING_NORMALIZE: bool = True

    # -- HuggingFace 配置 --
    HF_ENDPOINT: str = "https://hf-mirror.com"
    HF_HUB_ENABLE_HF_TRANSFER: bool = True
    HF_HUB_OFFLINE: int = 0

    # -- 日志配置 --
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"


# 全局单例配置实例
settings = Settings()
