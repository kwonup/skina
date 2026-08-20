"""팀원 A가 담당하는 Custom CNN 모델."""

import torch.nn as nn


class CustomCNN(nn.Module):
    """AdaptiveAvgPool을 사용해 FC 입력 크기 계산을 단순화한 기본 CNN."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, images):
        features = self.features(images)
        pooled = self.pool(features)
        return self.classifier(pooled)


def create_cnn(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Custom CNN을 만든다. pretrained 인자는 공통 interface를 위해 받기만 한다."""
    del pretrained
    return CustomCNN(num_classes=num_classes)
