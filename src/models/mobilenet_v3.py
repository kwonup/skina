"""팀원 D가 담당하는 ImageNet pretrained MobileNetV3-Large 모델."""

import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def create_mobilenet_v3(
    num_classes: int = 10, pretrained: bool = True
) -> nn.Module:
    """MobileNetV3-Large의 마지막 classifier를 클래스 수에 맞게 교체한다."""
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model
