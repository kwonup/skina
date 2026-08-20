"""학습과 추론에서 공통으로 사용할 PyTorch device를 선택한다."""

import torch


def get_device() -> torch.device:
    """CUDA, Apple MPS, CPU 순서로 사용할 장치를 선택한다."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
