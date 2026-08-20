"""선택한 모델을 공통 조건으로 학습하고 Validation Macro F1 최고 모델을 저장한다."""

import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.config import get_checkpoint_path, load_config
from src.data.dataset import IMAGE_SIZE, SEED, create_dataloaders
from src.device import get_device
from src.models import MODEL_NAMES, create_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"


def set_seed(seed: int = SEED) -> None:
    """Python, NumPy, PyTorch의 난수 seed를 한 번에 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """한 epoch 동안 forward, backward, optimizer update를 수행한다."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    """검증 loss, accuracy와 전체 예측을 모아 Macro F1을 계산한다."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (predictions == labels).sum().item()
            total += batch_size
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    val_f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return running_loss / total, correct / total, val_f1_macro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="skina 공통 baseline 모델을 학습합니다.")
    parser.add_argument("--config", type=Path, help="사용할 JSON config 파일")
    parser.add_argument("--model", choices=MODEL_NAMES, help="config의 model을 덮어씁니다.")
    parser.add_argument("--epochs", type=int, help="config의 epochs를 덮어씁니다.")
    parser.add_argument("--batch-size", type=int, help="config의 batch_size를 덮어씁니다.")
    parser.add_argument("--lr", type=float, help="config의 learning_rate를 덮어씁니다.")
    parser.add_argument(
        "--no-wandb", action="store_true", help="W&B 기록 없이 로컬에서 학습합니다."
    )
    return parser.parse_args()


def create_optimizer(model, name: str, learning_rate: float, weight_decay: float):
    """config에서 지원하는 Adam 또는 AdamW optimizer를 만든다."""
    if name.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    if name.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    raise ValueError(f"지원하지 않는 optimizer입니다: {name}")


def run_training(
    model_name: str,
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    use_wandb: bool = True,
    image_size: int = IMAGE_SIZE,
    num_workers: int = 0,
    seed: int = SEED,
    optimizer_name: str = "Adam",
    weight_decay: float = 0.0,
    pretrained: bool = True,
    wandb_project: str = "skina",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
) -> Path:
    """공통 조건으로 지정 모델을 학습하고 best checkpoint 경로를 반환한다."""
    set_seed(seed)
    device = get_device()
    print("Device:", device)

    train_loader, val_loader, _, class_names = create_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        seed=seed,
    )
    print(f"Classes ({len(class_names)}):", class_names)

    model = create_model(
        model_name, num_classes=len(class_names), pretrained=pretrained
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(
        model, optimizer_name, learning_rate, weight_decay
    )

    run = None
    if use_wandb:
        import wandb

        run = wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=wandb_run_name or f"{model_name}_baseline",
            config={
                "model": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "optimizer": optimizer_name,
                "weight_decay": weight_decay,
                "image_size": image_size,
                "seed": seed,
                "pretrained": pretrained,
            },
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_path or MODEL_DIR / f"{model_name}_best.pth"
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0

    try:
        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_accuracy, val_f1_macro = validate(
                model, val_loader, criterion, device
            )
            learning_rate = optimizer.param_groups[0]["lr"]

            print(f"\nEpoch {epoch}/{epochs}\n")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_accuracy:.4f}\n")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            print(f"Val Macro F1: {val_f1_macro:.4f}")

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_f1_macro": val_f1_macro,
                "learning_rate": learning_rate,
            }
            if run is not None:
                run.log(metrics)

            if val_f1_macro > best_val_f1:
                best_val_f1 = val_f1_macro
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_name": model_name,
                        "class_names": class_names,
                        "epoch": epoch,
                        "val_accuracy": val_accuracy,
                        "val_f1_macro": val_f1_macro,
                        "image_size": image_size,
                        "seed": seed,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "optimizer": optimizer_name,
                    },
                    checkpoint_path,
                )
                print(f"Best model saved: {checkpoint_path}")
    finally:
        if run is not None:
            run.finish()

    print(f"\nBest Validation Macro F1: {best_val_f1:.4f}")
    return checkpoint_path


def run_training_from_config(
    config_path: Path,
    model_name: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    disable_wandb: bool = False,
) -> Path:
    """JSON config를 읽고 선택적인 CLI 값만 덮어써 학습을 실행한다."""
    config = load_config(config_path)
    data_config = config["data"]
    training_config = config["training"]
    wandb_config = config["wandb"]
    selected_model = model_name or config["model"]
    selected_run_name = (
        wandb_config["run_name"]
        if selected_model == config["model"]
        else f"{selected_model}_baseline"
    )

    return run_training(
        model_name=selected_model,
        epochs=epochs if epochs is not None else training_config["epochs"],
        batch_size=(
            batch_size if batch_size is not None else data_config["batch_size"]
        ),
        learning_rate=(
            learning_rate
            if learning_rate is not None
            else training_config["learning_rate"]
        ),
        use_wandb=wandb_config["enabled"] and not disable_wandb,
        image_size=data_config["image_size"],
        num_workers=data_config["num_workers"],
        seed=data_config["seed"],
        optimizer_name=training_config["optimizer"],
        weight_decay=training_config["weight_decay"],
        pretrained=training_config["pretrained"],
        wandb_project=wandb_config["project"],
        wandb_entity=wandb_config["entity"],
        wandb_run_name=selected_run_name,
        checkpoint_path=get_checkpoint_path(config, selected_model),
    )


def main() -> None:
    args = parse_args()
    if args.config is None and args.model is None:
        raise SystemExit("--config 또는 --model 중 하나는 반드시 지정해야 합니다.")

    if args.config is not None:
        run_training_from_config(
            config_path=args.config,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            disable_wandb=args.no_wandb,
        )
    else:
        run_training(
            model_name=args.model,
            epochs=args.epochs if args.epochs is not None else 15,
            batch_size=args.batch_size if args.batch_size is not None else 32,
            learning_rate=args.lr if args.lr is not None else 1e-4,
            use_wandb=not args.no_wandb,
        )


if __name__ == "__main__":
    main()
