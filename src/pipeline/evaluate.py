"""Best checkpoint를 최종 test set에서 평가하고 지표와 confusion matrix를 저장한다."""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from src.config import get_checkpoint_path, load_config
from src.classes import validate_class_names
from src.data.dataset import create_dataloaders
from src.models import MODEL_NAMES, create_model
from src.pipeline.train import get_device


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Best 모델을 test set에서 최종 평가합니다.")
    parser.add_argument("--config", type=Path, help="사용할 JSON config 파일")
    parser.add_argument("--model", choices=MODEL_NAMES, help="config의 model을 덮어씁니다.")
    parser.add_argument("--batch-size", type=int, help="config의 batch_size를 덮어씁니다.")
    parser.add_argument(
        "--checkpoint", type=Path, help="생략하면 outputs/models/{model}_best.pth"
    )
    return parser.parse_args()


def run_evaluation(
    model_name: str,
    batch_size: int = 32,
    checkpoint_path: Optional[Path] = None,
    image_size: int = 224,
    num_workers: int = 0,
    seed: int = 42,
) -> dict:
    """지정 모델의 best checkpoint를 test set에서 평가하고 결과를 반환한다."""
    device = get_device()
    print("Device:", device)

    checkpoint_path = checkpoint_path or (
        PROJECT_ROOT / "outputs" / "models" / f"{model_name}_best.pth"
    )
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint를 찾을 수 없습니다: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_model = checkpoint.get("model_name", model_name)
    if checkpoint_model != model_name:
        raise ValueError(
            f"요청 모델은 {model_name}이지만 checkpoint 모델은 {checkpoint_model}입니다."
        )
    if checkpoint.get("image_size", image_size) != image_size:
        raise ValueError(
            "checkpoint 학습 image_size와 config의 image_size가 다릅니다: "
            f"{checkpoint['image_size']} != {image_size}"
        )

    _, _, test_loader, dataset_class_names = create_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        seed=seed,
    )
    class_names = list(
        validate_class_names(checkpoint["class_names"], source="checkpoint")
    )
    if list(dataset_class_names) != list(class_names):
        raise ValueError("checkpoint와 test dataset의 클래스 순서가 다릅니다.")

    model = create_model(model_name, len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            predictions = model(images.to(device)).argmax(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )
    metrics = {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "validation_accuracy": checkpoint["val_accuracy"],
        "validation_f1_macro": checkpoint["val_f1_macro"],
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_f1_macro": f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "test_precision_macro": precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "test_recall_macro": recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "classification_report": report,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{model_name}_metrics.json"
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    figure, axis = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
        ax=axis, cmap="Blues", xticks_rotation=90, colorbar=False
    )
    figure.tight_layout()
    plot_path = PLOTS_DIR / f"confusion_matrix_{model_name}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    print("\n=== Test Metrics ===")
    print(f"Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Macro F1: {metrics['test_f1_macro']:.4f}")
    print(f"Macro Precision: {metrics['test_precision_macro']:.4f}")
    print(f"Macro Recall: {metrics['test_recall_macro']:.4f}\n")
    print(report_text)
    print("Metrics saved:", result_path)
    print("Confusion matrix saved:", plot_path)
    return metrics


def run_evaluation_from_config(
    config_path: Path,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
) -> dict:
    """JSON config의 모델·데이터·checkpoint 설정으로 최종 평가한다."""
    config = load_config(config_path)
    data_config = config["data"]
    selected_model = model_name or config["model"]
    selected_checkpoint = checkpoint_path or get_checkpoint_path(
        config, selected_model
    )
    return run_evaluation(
        model_name=selected_model,
        batch_size=(
            batch_size if batch_size is not None else data_config["batch_size"]
        ),
        checkpoint_path=selected_checkpoint,
        image_size=data_config["image_size"],
        num_workers=data_config["num_workers"],
        seed=data_config["seed"],
    )


def main() -> None:
    args = parse_args()
    if args.config is None and args.model is None:
        raise SystemExit("--config 또는 --model 중 하나는 반드시 지정해야 합니다.")

    if args.config is not None:
        run_evaluation_from_config(
            config_path=args.config,
            model_name=args.model,
            batch_size=args.batch_size,
            checkpoint_path=args.checkpoint,
        )
    else:
        run_evaluation(
            model_name=args.model,
            batch_size=args.batch_size if args.batch_size is not None else 32,
            checkpoint_path=args.checkpoint,
        )


if __name__ == "__main__":
    main()
