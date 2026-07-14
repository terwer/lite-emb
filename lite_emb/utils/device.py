"""
设备检测工具。

自动检测可用的计算设备并返回最佳精度策略：
- CUDA (NVIDIA GPU) -> 使用 fp16
- MPS (Apple Silicon) -> 使用 fp16
- CPU -> 使用 fp32
"""

import torch


def get_device_and_precision() -> tuple[str, bool]:
    """
    检测最佳可用设备并返回设备名称与是否启用 fp16。

    Returns:
        tuple[str, bool]: (设备名称, 是否使用 fp16)
    """
    if torch.cuda.is_available():
        return "cuda", True
    if torch.backends.mps.is_available():
        return "mps", True
    return "cpu", False
