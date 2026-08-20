"""팀원별 모델 구현을 이름 하나로 생성할 수 있게 연결한다."""

import torch.nn as nn

from src.models.cnn import CustomCNN, create_cnn
from src.models.efficientnet_b0 import create_efficientnet_b0
from src.models.mobilenet_v3 import create_mobilenet_v3
from src.models.resnet18 import create_resnet18


MODEL_NAMES = ("cnn", "resnet18", "efficientnet_b0", "mobilenet_v3")


def create_model(
    model_name: str, num_classes: int, pretrained: bool = True
) -> nn.Module:
    """이름에 맞는 팀원별 모델 생성 함수를 호출한다."""
    model_name = model_name.lower()
    creators = {
        "cnn": create_cnn,
        "resnet18": create_resnet18,
        "efficientnet_b0": create_efficientnet_b0,
        "mobilenet_v3": create_mobilenet_v3,
    }
    if model_name not in creators:
        supported = ", ".join(MODEL_NAMES)
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}. 사용 가능: {supported}")
    return creators[model_name](num_classes=num_classes, pretrained=pretrained)


__all__ = ["CustomCNN", "MODEL_NAMES", "create_model"]
