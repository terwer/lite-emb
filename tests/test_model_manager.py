"""
模型管理器单元测试。
"""

import pytest

from lite_emb.config import settings


class TestModelManager:
    """测试模型管理器功能。"""

    def test_singleton_exists(self):
        """验证全局单例管理器存在。"""
        from lite_emb.models.manager import model_manager

        assert model_manager is not None

    def test_get_model_info_unloaded(self):
        """验证未加载模型时的状态信息。"""
        from lite_emb.models.manager import model_manager

        info = model_manager.get_model_info()

        assert info["loaded"] is False
        assert info["dimension"] == 0
        assert info["model_id"] == "N/A"


class TestModelRegistry:
    """测试模型注册表功能。"""

    def test_resolve_alias_bge_m3(self):
        """验证别名解析：bge-m3。"""
        from lite_emb.models.registry import ModelRegistry

        resolved = ModelRegistry.resolve_model_id("bge-m3")
        assert resolved == "BAAI/bge-m3"

    def test_resolve_alias_case_insensitive(self):
        """验证别名解析的大小写不敏感性。"""
        from lite_emb.models.registry import ModelRegistry

        resolved = ModelRegistry.resolve_model_id("BGE-M3")
        assert resolved == "BAAI/bge-m3"

    def test_resolve_full_id(self):
        """验证完整模型 ID 直接通过。"""
        from lite_emb.models.registry import ModelRegistry

        resolved = ModelRegistry.resolve_model_id("BAAI/bge-m3")
        assert resolved == "BAAI/bge-m3"

    def test_resolve_unknown_model_passthrough(self):
        """验证未知模型名称原样返回（触发自动发现）。"""
        from lite_emb.models.registry import ModelRegistry

        resolved = ModelRegistry.resolve_model_id("some/unknown-model")
        assert resolved == "some/unknown-model"

    def test_get_model_info_known(self):
        """验证已知模型返回正确信息。"""
        from lite_emb.models.registry import ModelRegistry

        info = ModelRegistry.get_model_info("BAAI/bge-m3")
        assert info is not None
        assert info.dimension == 1024
        assert info.max_seq_length == 8192

    def test_get_model_info_unknown(self):
        """验证未知模型返回 None。"""
        from lite_emb.models.registry import ModelRegistry

        info = ModelRegistry.get_model_info("unknown/model")
        assert info is None

    def test_list_models_not_empty(self):
        """验证模型列表不为空。"""
        from lite_emb.models.registry import ModelRegistry

        models = ModelRegistry.list_models()
        assert len(models) > 0
        assert all("id" in m for m in models)
        assert all("object" in m for m in models)

    def test_bge_m3_backend_type(self):
        """验证 BGE-M3 使用正确的后端类型。"""
        from lite_emb.models.registry import BackendType, ModelRegistry

        backend = ModelRegistry.get_backend_type("BAAI/bge-m3")
        assert backend == BackendType.BGE_M3

    def test_sentence_transformer_backend_type(self):
        """验证 sentence-transformers 模型使用正确后端。"""
        from lite_emb.models.registry import BackendType, ModelRegistry

        backend = ModelRegistry.get_backend_type(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        assert backend == BackendType.SENTENCE_TRANSFORMER
