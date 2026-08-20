"""팀원 C가 담당하는 ImageNet pretrained EfficientNet-B0 모델."""

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def create_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0의 마지막 classifier를 클래스 수에 맞게 교체한다."""
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model
