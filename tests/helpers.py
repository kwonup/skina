"""테스트용 작은 checkpoint와 설정을 만든다."""

from pathlib import Path

import torch

from service.api.settings import Settings
from src.classes import CLASS_NAMES
from src.models import create_model


def create_test_checkpoint(path: Path, image_size: int = 32) -> Path:
    model = create_model("cnn", len(CLASS_NAMES), pretrained=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "cnn",
            "class_names": list(CLASS_NAMES),
            "image_size": image_size,
        },
        path,
    )
    return path


def create_test_settings(checkpoint_path: Path) -> Settings:
    return Settings(
        model_path=checkpoint_path,
        allowed_origins=("http://localhost:3000",),
        max_upload_size_bytes=1024 * 1024,
        enable_gradcam=False,
    )
