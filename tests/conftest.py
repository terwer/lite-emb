"""
pytest 配置和共享 fixtures。

提供 FastAPI TestClient，用于集成测试。
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 设置测试环境变量（在导入应用前）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("LOG_FILE", "/dev/null")


@pytest.fixture
def client() -> TestClient:
    """创建 FastAPI 测试客户端。"""
    from lite_emb.app import create_app

    app = create_app()
    return TestClient(app)
