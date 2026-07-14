"""
设备检测单元测试。
"""

import pytest


class TestDeviceDetection:
    """测试设备检测功能。"""

    def test_get_device_returns_tuple(self):
        """验证返回值是 (str, bool) 元组。"""
        from lite_emb.utils.device import get_device_and_precision

        device, use_fp16 = get_device_and_precision()

        assert isinstance(device, str)
        assert isinstance(use_fp16, bool)

    def test_device_is_valid(self):
        """验证设备名是有效的设备类型。"""
        from lite_emb.utils.device import get_device_and_precision

        device, _ = get_device_and_precision()

        assert device in ("cuda", "mps", "cpu")

    def test_fp16_only_on_gpu(self):
        """验证 fp16 仅在 GPU/MPS 上启用。"""
        from lite_emb.utils.device import get_device_and_precision

        device, use_fp16 = get_device_and_precision()

        if device == "cpu":
            assert use_fp16 is False
        else:
            assert use_fp16 is True
