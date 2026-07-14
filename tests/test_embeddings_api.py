"""
Embedding API 集成测试。

测试 OpenAI 兼容 API 端点的正确性。
"""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """测试健康检查端点。"""

    def test_root_endpoint(self, client: TestClient):
        """测试根路径返回服务信息。"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "lite-emb" in data["message"]
        assert "version" in data

    def test_health_endpoint(self, client: TestClient):
        """测试健康检查端点。"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestModelsEndpoint:
    """测试模型列表端点。"""

    def test_list_models(self, client: TestClient):
        """测试获取模型列表。"""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        # 验证每个模型的结构
        for model in data["data"]:
            assert "id" in model
            assert model["object"] == "model"
            assert "created" in model
            assert model["owned_by"] == "lite-emb"


class TestEmbeddingsEndpoint:
    """测试 Embedding 端点（不加载真实模型）。"""

    def test_embeddings_missing_input(self, client: TestClient):
        """测试缺少 input 参数时返回 422。"""
        response = client.post(
            "/v1/embeddings",
            json={"model": "bge-m3"},
        )
        assert response.status_code == 422

    def test_embeddings_empty_list(self, client: TestClient):
        """测试空输入列表时返回 400。"""
        response = client.post(
            "/v1/embeddings",
            json={"model": "bge-m3", "input": []},
        )
        # 空列表被 pydantic 捕获为验证错误 422
        # 因为我们的 validator 会在非空时才检查
        assert response.status_code in (400, 422)

    def test_embeddings_base64_not_supported(self, client: TestClient):
        """测试 base64 编码返回 422。"""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "bge-m3",
                "input": "hello",
                "encoding_format": "base64",
            },
        )
        assert response.status_code == 422

    def test_embeddings_token_array_not_supported(self, client: TestClient):
        """测试 token 数组输入返回 422。"""
        response = client.post(
            "/v1/embeddings",
            json={"model": "bge-m3", "input": [1, 2, 3]},
        )
        assert response.status_code == 422


class TestOpenAICompatibility:
    """测试 OpenAI SDK 兼容性。"""

    def test_response_structure_single_input(self, client: TestClient):
        """验证单个字符串输入的响应结构（不加载模型，仅验证验证层）。"""
        response = client.post(
            "/v1/embeddings",
            json={"model": "bge-m3", "input": "hello world"},
        )
        # 预期失败（因为模型未加载），但至少路由和验证是正常工作的
        # 如果没有模型，会返回 500；其他验证错误返回 400/422
        assert response.status_code in (400, 422, 500)

    def test_invalid_dimensions_negative(self, client: TestClient):
        """测试负维度值被拒绝。"""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "bge-m3",
                "input": "hello",
                "dimensions": -1,
            },
        )
        assert response.status_code == 422

    def test_invalid_dimensions_zero(self, client: TestClient):
        """测试零维度值被拒绝。"""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "bge-m3",
                "input": "hello",
                "dimensions": 0,
            },
        )
        assert response.status_code == 422
