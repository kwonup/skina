"""두 checkpoint를 동일한 외부 이미지셋에서 비교한다.

외부 데이터는 클래스별 하위 폴더와 루트의 ``<class>_sample`` 파일을 모두
지원한다. 서로 클래스 수가 다른 checkpoint도 class index가 아니라 class name을
기준으로 비교하므로, 한 모델에만 있는 출력 클래스 역시 정상적으로 오답 처리된다.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 서버/CI 및 Tcl/Tk가 없는 환경에서도 PNG를 저장할 수 있게 한다.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.classes import CLASS_NAMES
from src.data.dataset import create_eval_transform
from src.device import get_device
from src.models import create_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_SPELLING_FIXES = {
    "hermangioma": "hemangioma",
    "seaborrheic": "seborrheic",
}


@dataclass(frozen=True)
class ExternalSample:
    path: Path
    label: str


class ExternalDataset(Dataset):
    def __init__(self, samples: list[ExternalSample], image_size: int) -> None:
        self.samples = samples
        self.transform = create_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, sample.label, str(sample.path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="이전/최근 모델의 외부셋 성능 비교")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--recent", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "external_comparison",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def _normalized_text(path: Path, data_dir: Path) -> str:
    relative = str(path.relative_to(data_dir)).lower()
    for wrong, correct in LABEL_SPELLING_FIXES.items():
        relative = relative.replace(wrong, correct)
    return relative


def discover_samples(data_dir: Path) -> list[ExternalSample]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"외부 데이터 폴더를 찾을 수 없습니다: {data_dir}")

    samples: list[ExternalSample] = []
    unknown: list[Path] = []
    for path in sorted(data_dir.rglob("*"), key=lambda value: str(value).lower()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        normalized = _normalized_text(path, data_dir)
        matches = [name for name in CLASS_NAMES if name in normalized]
        if len(matches) != 1:
            unknown.append(path)
            continue
        samples.append(ExternalSample(path=path.resolve(), label=matches[0]))

    if unknown:
        rendered = "\n".join(f"- {path}" for path in unknown)
        raise ValueError(f"정답 클래스를 하나로 판별하지 못한 이미지가 있습니다:\n{rendered}")
    if not samples:
        raise ValueError(f"평가할 이미지가 없습니다: {data_dir}")

    # 추론을 시작하기 전에 손상 이미지를 찾아 전체 평가의 완전성을 보장한다.
    invalid: list[str] = []
    for sample in samples:
        try:
            with Image.open(sample.path) as image:
                image.verify()
        except Exception as error:  # Pillow가 포맷별 예외를 다양하게 발생시킨다.
            invalid.append(f"{sample.path}: {error}")
    if invalid:
        raise ValueError("열 수 없는 이미지가 있습니다:\n" + "\n".join(invalid))
    return samples


def load_checkpoint_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint를 찾을 수 없습니다: {resolved}")
    checkpoint = torch.load(resolved, map_location=device, weights_only=True)
    required = {"model_state_dict", "model_name", "class_names", "image_size"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"{resolved.name}에 필수 항목이 없습니다: {sorted(missing)}")

    class_names = list(checkpoint["class_names"])
    if not class_names or len(class_names) != len(set(class_names)):
        raise ValueError(f"{resolved.name}의 클래스 목록이 올바르지 않습니다.")
    model = create_model(checkpoint["model_name"], len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    metadata = {
        "path": str(resolved),
        "model_name": checkpoint["model_name"],
        "class_names": class_names,
        "image_size": int(checkpoint["image_size"]),
        "epoch": checkpoint.get("epoch"),
        "validation_accuracy": checkpoint.get("val_accuracy"),
        "validation_f1_macro": checkpoint.get("val_f1_macro"),
    }
    return model, metadata


def predict(
    model: torch.nn.Module,
    class_names: list[str],
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> tuple[list[str], list[str], list[float]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    with torch.inference_mode():
        for images, labels, _ in tqdm(loader, desc=description):
            probabilities = torch.softmax(model(images.to(device)), dim=1)
            confidence, indices = probabilities.max(dim=1)
            y_true.extend(labels)
            y_pred.extend(class_names[index] for index in indices.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
    return y_true, y_pred, confidences


def calculate_metrics(
    y_true: list[str], y_pred: list[str], target_classes: list[str]
) -> dict:
    recalls = recall_score(
        y_true, y_pred, labels=target_classes, average=None, zero_division=0
    )
    support = Counter(y_true)
    extra_predictions = Counter(pred for pred in y_pred if pred not in target_classes)
    correct_count = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=target_classes, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(
                y_true, y_pred, labels=target_classes, average="macro", zero_division=0
            )
        ),
        "correct_count": correct_count,
        "total_count": len(y_true),
        "class_recall": {
            name: {"recall": float(value), "support": support[name]}
            for name, value in zip(target_classes, recalls)
        },
        "predictions_outside_target_classes": dict(sorted(extra_predictions.items())),
    }


def save_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    target_classes: list[str],
    prediction_classes: list[str],
    title: str,
    output_path: Path,
) -> None:
    row_index = {name: index for index, name in enumerate(target_classes)}
    column_index = {name: index for index, name in enumerate(prediction_classes)}
    matrix = np.zeros((len(target_classes), len(prediction_classes)), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[row_index[actual], column_index[predicted]] += 1

    width = max(12, len(prediction_classes) * 0.8)
    height = max(9, len(target_classes) * 0.7)
    figure, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.03, pad=0.03)
    axis.set(
        title=title,
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(prediction_classes)),
        yticks=np.arange(len(target_classes)),
        xticklabels=prediction_classes,
        yticklabels=target_classes,
    )
    axis.tick_params(axis="x", rotation=60)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if value:
                axis.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=8,
                )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_improved_examples(
    rows: list[dict], output_path: Path, maximum: int = 8
) -> int:
    improved = [row for row in rows if row["previous_correct"] == "False" and row["recent_correct"] == "True"]
    selected = improved[:maximum]
    if not selected:
        return 0

    columns = min(4, len(selected))
    rows_count = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(rows_count, columns, figsize=(4 * columns, 4 * rows_count))
    axes_array = np.atleast_1d(axes).reshape(-1)
    for axis, row in zip(axes_array, selected):
        with Image.open(row["path"]) as image:
            display = ImageOps.exif_transpose(image).convert("RGB")
            axis.imshow(display)
        axis.set_title(
            f"True: {row['true_label']}\nBefore: {row['previous_prediction']} -> New: {row['recent_prediction']}",
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes_array[len(selected):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return len(selected)


def save_performance_comparison(result: dict, output_path: Path) -> None:
    """핵심 지표와 클래스별 Recall을 한 장의 비교 그래프로 저장한다."""
    previous = result["models"]["previous"]["metrics"]
    recent = result["models"]["recent"]["metrics"]
    target_classes = result["dataset"]["target_classes"]
    colors = {"previous": "#94a3b8", "recent": "#2563eb"}

    figure = plt.figure(figsize=(14, 12))
    grid = figure.add_gridspec(2, 1, height_ratios=(1, 2.4), hspace=0.38)

    metric_axis = figure.add_subplot(grid[0])
    metric_names = ["Accuracy", "Macro F1", "Macro Recall"]
    previous_values = [
        previous["accuracy"], previous["macro_f1"], previous["macro_recall"]
    ]
    recent_values = [recent["accuracy"], recent["macro_f1"], recent["macro_recall"]]
    x_positions = np.arange(len(metric_names))
    width = 0.34
    old_bars = metric_axis.bar(
        x_positions - width / 2,
        previous_values,
        width,
        label="Previous",
        color=colors["previous"],
    )
    new_bars = metric_axis.bar(
        x_positions + width / 2,
        recent_values,
        width,
        label="Recent",
        color=colors["recent"],
    )
    metric_axis.bar_label(old_bars, fmt="%.3f", padding=3)
    metric_axis.bar_label(new_bars, fmt="%.3f", padding=3)
    metric_axis.set(
        title="External performance: previous vs recent",
        ylabel="Score (0-1)",
        xticks=x_positions,
        xticklabels=metric_names,
        ylim=(0, 1.08),
    )
    metric_axis.grid(axis="y", alpha=0.25)
    metric_axis.legend(frameon=False, loc="upper left")

    recall_axis = figure.add_subplot(grid[1])
    y_positions = np.arange(len(target_classes))
    previous_recalls = [
        previous["class_recall"][name]["recall"] for name in target_classes
    ]
    recent_recalls = [recent["class_recall"][name]["recall"] for name in target_classes]
    supports = [previous["class_recall"][name]["support"] for name in target_classes]
    display_names = [
        f"{name}  (n={support})" for name, support in zip(target_classes, supports)
    ]
    recall_axis.barh(
        y_positions + width / 2,
        previous_recalls,
        width,
        label="Previous",
        color=colors["previous"],
    )
    recall_axis.barh(
        y_positions - width / 2,
        recent_recalls,
        width,
        label="Recent",
        color=colors["recent"],
    )
    for index, (old_value, new_value) in enumerate(zip(previous_recalls, recent_recalls)):
        recall_axis.text(
            old_value + 0.012,
            index + width / 2,
            f"{old_value:.2f}",
            va="center",
            fontsize=8,
        )
        recall_axis.text(
            new_value + 0.012,
            index - width / 2,
            f"{new_value:.2f}",
            va="center",
            fontsize=8,
        )
    recall_axis.set(
        title="Class recall comparison",
        xlabel="Recall (0-1)",
        yticks=y_positions,
        yticklabels=display_names,
        xlim=(0, 1.12),
    )
    recall_axis.invert_yaxis()
    recall_axis.grid(axis="x", alpha=0.25)
    recall_axis.legend(frameon=False, loc="lower right")

    figure.subplots_adjust(left=0.27, right=0.97, top=0.95, bottom=0.07)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def save_markdown_report(result: dict, output_path: Path) -> None:
    previous = result["models"]["previous"]["metrics"]
    recent = result["models"]["recent"]["metrics"]
    delta = result["improvement"]
    lines = [
        "# 외부 데이터 모델 성능 비교",
        "",
        f"- 평가 이미지: {result['dataset']['total_images']}장",
        f"- 전처리: Resize({result['dataset']['image_size']}×{result['dataset']['image_size']}) + ImageNet Normalize",
        "- 평가지표의 Macro F1/Recall은 외부셋 정답 10개 클래스 기준",
        "",
        "| Model | External Accuracy | Macro F1 | Macro Recall | 정답 수 |",
        "|---|---:|---:|---:|---:|",
        f"| 이전 모델 | {format_percent(previous['accuracy'])} | {previous['macro_f1']:.4f} | {previous['macro_recall']:.4f} | {previous['correct_count']}/{previous['total_count']} |",
        f"| 최근 모델 | {format_percent(recent['accuracy'])} | {recent['macro_f1']:.4f} | {recent['macro_recall']:.4f} | {recent['correct_count']}/{recent['total_count']} |",
        f"| 개선폭 | {delta['accuracy_percentage_points']:+.2f}%p | {delta['macro_f1']:+.4f} | {delta['macro_recall']:+.4f} | {delta['correct_count']:+d}장 |",
        "",
        "## 클래스별 Recall",
        "",
        "| Class | Support | 이전 | 최근 | 변화 |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_name in result["dataset"]["target_classes"]:
        old_row = previous["class_recall"][class_name]
        new_row = recent["class_recall"][class_name]
        lines.append(
            f"| {class_name} | {old_row['support']} | {old_row['recall']:.4f} | "
            f"{new_row['recall']:.4f} | {new_row['recall'] - old_row['recall']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## 정오답 전환",
            "",
            f"- 이전 오답 → 최근 정답: {result['transitions']['previous_wrong_to_recent_correct']}장",
            f"- 이전 정답 → 최근 오답: {result['transitions']['previous_correct_to_recent_wrong']}장",
            f"- 두 모델 모두 정답: {result['transitions']['both_correct']}장",
            f"- 두 모델 모두 오답: {result['transitions']['both_wrong']}장",
            "",
            "이전 모델에만 존재하는 추가 출력 클래스로 예측한 경우도 오답으로 포함했습니다.",
            "",
            "## 해석 시 주의사항",
            "",
            f"- 이전 checkpoint는 {len(result['models']['previous']['checkpoint']['class_names'])}개, "
            f"최근 checkpoint는 {len(result['models']['recent']['checkpoint']['class_names'])}개 클래스 모델입니다. "
            "따라서 이 결과는 외부 10클래스 과제에서의 실제 Top-1 성능 비교이며, "
            "클래스 구성 외의 학습 변경 효과만 분리한 통제 실험은 아닙니다.",
            "- actinic_keratosis, basal_cell_carcinoma, dermatofibroma는 각 1장, "
            "hemangioma는 3장뿐이므로 해당 클래스 Recall과 Macro 지표의 표본 불확실성이 큽니다.",
            "",
            "## 그래프",
            "",
            "- `performance_comparison.png`: 핵심 지표와 클래스별 Recall 비교",
            "- `confusion_matrix_previous.png`, `confusion_matrix_recent.png`: 모델별 혼동행렬",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print("Device:", device)

    previous_model, previous_meta = load_checkpoint_model(args.previous, device)
    recent_model, recent_meta = load_checkpoint_model(args.recent, device)
    if previous_meta["image_size"] != recent_meta["image_size"]:
        raise ValueError("두 checkpoint의 image_size가 달라 동일 전처리 비교가 불가능합니다.")

    samples = discover_samples(args.data_dir.resolve())
    target_classes = list(CLASS_NAMES)
    missing_by_model = {
        "previous": sorted(set(target_classes) - set(previous_meta["class_names"])),
        "recent": sorted(set(target_classes) - set(recent_meta["class_names"])),
    }
    if any(missing_by_model.values()):
        raise ValueError(f"외부 정답 클래스가 checkpoint에 없습니다: {missing_by_model}")

    loader = DataLoader(
        ExternalDataset(samples, previous_meta["image_size"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    y_true, previous_pred, previous_conf = predict(
        previous_model, previous_meta["class_names"], loader, device, "Previous"
    )
    y_true_recent, recent_pred, recent_conf = predict(
        recent_model, recent_meta["class_names"], loader, device, "Recent"
    )
    if y_true != y_true_recent:
        raise RuntimeError("두 추론의 이미지/정답 순서가 달라졌습니다.")

    rows: list[dict] = []
    for sample, actual, old_pred, new_pred, old_conf, new_conf in zip(
        samples, y_true, previous_pred, recent_pred, previous_conf, recent_conf
    ):
        rows.append(
            {
                "path": str(sample.path),
                "true_label": actual,
                "previous_prediction": old_pred,
                "previous_confidence": f"{old_conf:.8f}",
                "previous_correct": str(old_pred == actual),
                "recent_prediction": new_pred,
                "recent_confidence": f"{new_conf:.8f}",
                "recent_correct": str(new_pred == actual),
            }
        )
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    previous_metrics = calculate_metrics(y_true, previous_pred, target_classes)
    recent_metrics = calculate_metrics(y_true, recent_pred, target_classes)
    old_correct = [actual == predicted for actual, predicted in zip(y_true, previous_pred)]
    new_correct = [actual == predicted for actual, predicted in zip(y_true, recent_pred)]
    transitions = {
        "previous_wrong_to_recent_correct": sum(not old and new for old, new in zip(old_correct, new_correct)),
        "previous_correct_to_recent_wrong": sum(old and not new for old, new in zip(old_correct, new_correct)),
        "both_correct": sum(old and new for old, new in zip(old_correct, new_correct)),
        "both_wrong": sum(not old and not new for old, new in zip(old_correct, new_correct)),
    }
    result = {
        "dataset": {
            "path": str(args.data_dir.resolve()),
            "total_images": len(samples),
            "class_distribution": dict(sorted(Counter(y_true).items())),
            "target_classes": target_classes,
            "image_size": previous_meta["image_size"],
        },
        "models": {
            "previous": {"checkpoint": previous_meta, "metrics": previous_metrics},
            "recent": {"checkpoint": recent_meta, "metrics": recent_metrics},
        },
        "improvement": {
            "accuracy_percentage_points": (recent_metrics["accuracy"] - previous_metrics["accuracy"]) * 100,
            "macro_f1": recent_metrics["macro_f1"] - previous_metrics["macro_f1"],
            "macro_recall": recent_metrics["macro_recall"] - previous_metrics["macro_recall"],
            "correct_count": recent_metrics["correct_count"] - previous_metrics["correct_count"],
        },
        "transitions": transitions,
    }
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    union_classes = list(dict.fromkeys(previous_meta["class_names"] + recent_meta["class_names"]))
    save_confusion_matrix(
        y_true, previous_pred, target_classes, union_classes,
        "Previous model - external data", output_dir / "confusion_matrix_previous.png",
    )
    save_confusion_matrix(
        y_true, recent_pred, target_classes, union_classes,
        "Recent model - external data", output_dir / "confusion_matrix_recent.png",
    )
    save_performance_comparison(result, output_dir / "performance_comparison.png")
    example_count = save_improved_examples(rows, output_dir / "improved_examples.png")
    result["representative_improved_examples"] = example_count
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_markdown_report(result, output_dir / "comparison_report.md")

    print("\n=== External comparison ===")
    print(f"Images: {len(samples)}")
    print(f"Previous: accuracy={previous_metrics['accuracy']:.4f}, macro_f1={previous_metrics['macro_f1']:.4f}, correct={previous_metrics['correct_count']}")
    print(f"Recent:   accuracy={recent_metrics['accuracy']:.4f}, macro_f1={recent_metrics['macro_f1']:.4f}, correct={recent_metrics['correct_count']}")
    print(
        "Improvement: "
        f"accuracy={result['improvement']['accuracy_percentage_points']:+.2f}%p, "
        f"macro_f1={result['improvement']['macro_f1']:+.4f}, "
        f"correct={result['improvement']['correct_count']:+d}"
    )
    print(f"Previous wrong -> recent correct: {transitions['previous_wrong_to_recent_correct']}")
    print("\n=== Class recall (previous -> recent) ===")
    for class_name in target_classes:
        old_recall = previous_metrics["class_recall"][class_name]["recall"]
        new_recall = recent_metrics["class_recall"][class_name]["recall"]
        support = previous_metrics["class_recall"][class_name]["support"]
        print(
            f"{class_name:27s} n={support:3d}  "
            f"{old_recall:.4f} -> {new_recall:.4f}  "
            f"({new_recall - old_recall:+.4f})"
        )
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()
