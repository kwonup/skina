"""공통 이미지 transform과 train/val/test DataLoader를 만든다."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.classes import validate_class_names


SEED = 42
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def create_train_transform(image_size: int = IMAGE_SIZE):
    """config의 image_size를 반영한 학습용 랜덤 증강을 만든다."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def create_eval_transform(image_size: int = IMAGE_SIZE):
    """config의 image_size를 반영한 validation/test/추론 transform을 만든다."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# 별도 설정을 전달하지 않는 코드와 notebook을 위한 baseline transform이다.
train_transform = create_train_transform()
eval_transform = create_eval_transform()


def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = IMAGE_SIZE,
    seed: int = SEED,
):
    """세 DataLoader와 ImageFolder가 정한 알파벳순 클래스명을 반환한다."""
    current_train_transform = create_train_transform(image_size)
    current_eval_transform = create_eval_transform(image_size)
    train_dataset = datasets.ImageFolder(
        PROCESSED_DIR / "train", transform=current_train_transform
    )
    val_dataset = datasets.ImageFolder(
        PROCESSED_DIR / "val", transform=current_eval_transform
    )
    test_dataset = datasets.ImageFolder(
        PROCESSED_DIR / "test", transform=current_eval_transform
    )

    if not (
        train_dataset.class_to_idx
        == val_dataset.class_to_idx
        == test_dataset.class_to_idx
    ):
        raise ValueError("train/val/test의 클래스 폴더 구성이 서로 다릅니다.")

    class_names = validate_class_names(
        train_dataset.classes, source="processed dataset"
    )

    # generator를 고정하면 train shuffle 순서도 같은 환경에서 재현할 수 있다.
    generator = torch.Generator().manual_seed(seed)
    common_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **common_options
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_options)

    return train_loader, val_loader, test_loader, list(class_names)


if __name__ == "__main__":
    loaders = create_dataloaders()
    images, labels = next(iter(loaders[0]))
    print("Classes:", loaders[3])
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
