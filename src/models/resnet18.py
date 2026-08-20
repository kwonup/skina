"""팀원 B가 담당하는 ImageNet pretrained ResNet18 모델."""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def create_resnet18(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """ResNet18의 마지막 FC를 피부종양 클래스 수에 맞게 교체한다."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
