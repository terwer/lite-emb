"""
lite-emb 服务入口。

使用 uvicorn 启动 FastAPI 应用。
"""

import uvicorn

from lite_emb.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "lite_emb.app:app",
        host=settings.DEV_HOST,
        port=settings.DEV_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        workers=settings.WORKERS if settings.WORKERS > 1 else None,
        reload=False,
    )
