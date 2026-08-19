"""JSON 실험 설정을 읽고 기본값과 프로젝트 기준 경로를 관리한다."""

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAMES = {"cnn", "resnet18", "efficientnet_b0", "mobilenet_v3"}

DEFAULT_CONFIG = {
    "model": "resnet18",
    "data": {
        "image_size": 224,
        "batch_size": 32,
        "num_workers": 0,
        "seed": 42,
    },
    "training": {
        "epochs": 15,
        "learning_rate": 0.0001,
        "optimizer": "Adam",
        "weight_decay": 0.0,
        "pretrained": True,
    },
    "wandb": {
        "enabled": True,
        "entity": "sesac08",
        "project": "skina",
        "run_name": "resnet18_baseline",
    },
    "inference": {
        "top_k": 3,
        "checkpoint": "outputs/models/resnet18_best.pth",
    },
}


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """중첩 딕셔너리에서 사용자가 적은 값만 기본값 위에 덮어쓴다."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def resolve_project_path(path: Optional[str]) -> Optional[Path]:
    """상대 경로는 skina 프로젝트 루트를 기준으로 절대 경로로 바꾼다."""
    if path is None:
        return None
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def get_checkpoint_path(
    config: Dict[str, Any], model_name: Optional[str] = None
) -> Path:
    """config 모델은 설정 경로를, CLI로 바꾼 모델은 모델별 기본 경로를 쓴다."""
    selected_model = model_name or config["model"]
    if selected_model == config["model"]:
        return resolve_project_path(config["inference"]["checkpoint"])
    return PROJECT_ROOT / "outputs" / "models" / f"{selected_model}_best.pth"


def validate_config(config: Dict[str, Any]) -> None:
    """실행 전에 자주 발생하는 config 오타와 잘못된 범위를 확인한다."""
    if config["model"] not in MODEL_NAMES:
        raise ValueError(
            f"지원하지 않는 model입니다: {config['model']}. "
            f"사용 가능: {', '.join(sorted(MODEL_NAMES))}"
        )

    positive_values = {
        "data.image_size": config["data"]["image_size"],
        "data.batch_size": config["data"]["batch_size"],
        "training.epochs": config["training"]["epochs"],
        "training.learning_rate": config["training"]["learning_rate"],
        "inference.top_k": config["inference"]["top_k"],
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name}은 0보다 커야 합니다: {value}")

    if config["data"]["num_workers"] < 0:
        raise ValueError("data.num_workers는 0 이상이어야 합니다.")
    if config["training"]["weight_decay"] < 0:
        raise ValueError("training.weight_decay는 0 이상이어야 합니다.")
    if config["training"]["optimizer"].lower() not in {"adam", "adamw"}:
        raise ValueError("training.optimizer는 Adam 또는 AdamW를 사용하세요.")


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """JSON 파일을 읽어 누락된 값은 DEFAULT_CONFIG로 채운 뒤 검증한다."""
    user_config: Dict[str, Any] = {}
    if config_path is not None:
        path = resolve_project_path(str(config_path))
        if not path.is_file():
            raise FileNotFoundError(f"config 파일을 찾을 수 없습니다: {path}")
        with path.open("r", encoding="utf-8") as file:
            user_config = json.load(file)

    config = _merge_dict(DEFAULT_CONFIG, user_config)
    model_name = config["model"]
    if "run_name" not in user_config.get("wandb", {}):
        config["wandb"]["run_name"] = f"{model_name}_baseline"
    if "checkpoint" not in user_config.get("inference", {}):
        config["inference"]["checkpoint"] = (
            f"outputs/models/{model_name}_best.pth"
        )
    validate_config(config)
    return config
