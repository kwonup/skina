"""CLI와 웹 서비스가 공유하는 모델 로딩 및 이미지 추론 로직."""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image

from src.classes import validate_class_names
from src.data.dataset import create_eval_transform
from src.device import get_device
from src.models import MODEL_NAMES, create_model


@dataclass(frozen=True)
class Prediction:
    """Top-K의 단일 예측 항목."""

    rank: int
    class_name: str
    probability: float


@dataclass(frozen=True)
class PredictionOutput:
    """한 이미지의 Top-K 결과와 순수 모델 실행 시간."""

    predictions: tuple[Prediction, ...]
    inference_time_ms: float


@dataclass
class InferenceModel:
    """서버 시작 시 한 번 로드해 요청마다 재사용하는 추론 모델."""

    model: nn.Module
    model_name: str
    class_names: tuple[str, ...]
    image_size: int
    device: torch.device
    checkpoint_path: Path
    lock: Lock

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """검증/테스트와 동일한 transform으로 PIL 이미지를 tensor로 바꾼다."""
        rgb_image = image.convert("RGB")
        return create_eval_transform(self.image_size)(rgb_image).unsqueeze(0).to(
            self.device
        )

    def predict(self, image: Image.Image, top_k: int = 3) -> PredictionOutput:
        """로드된 모델로 이미지 한 장의 Top-K를 계산한다."""
        if not 1 <= top_k <= len(self.class_names):
            raise ValueError(
                f"top_k는 1~{len(self.class_names)} 범위여야 합니다: {top_k}"
            )

        image_tensor = self.preprocess(image)
        with self.lock, torch.inference_mode():
            _synchronize(self.device)
            started_at = perf_counter()
            probabilities = torch.softmax(self.model(image_tensor), dim=1)
            _synchronize(self.device)
            inference_time_ms = (perf_counter() - started_at) * 1000
            top_probabilities, top_indices = torch.topk(
                probabilities, k=top_k, dim=1
            )

        predictions = tuple(
            Prediction(
                rank=rank,
                class_name=self.class_names[index.item()],
                probability=probability.item(),
            )
            for rank, (probability, index) in enumerate(
                zip(top_probabilities[0].cpu(), top_indices[0].cpu()), start=1
            )
        )
        return PredictionOutput(
            predictions=predictions,
            inference_time_ms=inference_time_ms,
        )


def _synchronize(device: torch.device) -> None:
    """비동기 CUDA 연산을 latency 측정 전에 동기화한다."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _last_linear_output_size(model: nn.Module) -> int:
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    if not linear_layers:
        raise ValueError("모델에서 classifier Linear layer를 찾지 못했습니다.")
    return linear_layers[-1].out_features


def load_inference_model(
    checkpoint_path: Path,
    *,
    expected_model_name: Optional[str] = None,
    expected_image_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> InferenceModel:
    """checkpoint를 검증하고 서비스에서 재사용할 모델을 한 번 로드한다."""
    resolved_path = Path(checkpoint_path).resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"checkpoint를 찾을 수 없습니다: {resolved_path}")

    selected_device = device or get_device()
    checkpoint = torch.load(
        resolved_path,
        map_location=selected_device,
        weights_only=True,
    )
    required_keys = {"model_state_dict", "model_name", "class_names", "image_size"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(
            "checkpoint에 필수 항목이 없습니다: " + ", ".join(sorted(missing_keys))
        )

    model_name = checkpoint["model_name"]
    if model_name not in MODEL_NAMES:
        raise ValueError(f"지원하지 않는 checkpoint 모델입니다: {model_name}")
    if expected_model_name is not None and model_name != expected_model_name:
        raise ValueError(
            f"요청 모델은 {expected_model_name}이지만 checkpoint 모델은 "
            f"{model_name}입니다."
        )

    image_size = checkpoint["image_size"]
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError(f"checkpoint image_size가 올바르지 않습니다: {image_size}")
    if expected_image_size is not None and image_size != expected_image_size:
        raise ValueError(
            "checkpoint 학습 image_size와 설정의 image_size가 다릅니다: "
            f"{image_size} != {expected_image_size}"
        )

    class_names = validate_class_names(
        checkpoint["class_names"], source="checkpoint"
    )
    model = create_model(model_name, len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if _last_linear_output_size(model) != len(class_names):
        raise ValueError("모델 classifier 출력과 checkpoint 클래스 수가 다릅니다.")

    model = model.to(selected_device)
    model.eval()
    return InferenceModel(
        model=model,
        model_name=model_name,
        class_names=class_names,
        image_size=image_size,
        device=selected_device,
        checkpoint_path=resolved_path,
        lock=Lock(),
    )
