"""공통 이미지 transform과 train/validation/test DataLoader를 만든다."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


SEED = 42
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLASS_NAMES_PATH = PROCESSED_DIR / "class_names.json"


class OrderedImageFolder(datasets.ImageFolder):
    """ImageFolder variant that preserves the explicit project class order."""

    def __init__(self, root: Path, class_names: list[str], **kwargs):
        self.ordered_class_names = class_names
        super().__init__(root, **kwargs)

    def find_classes(self, directory: str):
        """Use class_names.json instead of ImageFolder's alphabetic ordering."""
        directory_path = Path(directory)
        found = {path.name for path in directory_path.iterdir() if path.is_dir()}
        expected = set(self.ordered_class_names)
        if found != expected:
            raise ValueError(
                f"{directory_path} 클래스 폴더 불일치: "
                f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
            )
        return self.ordered_class_names, {
            name: index for index, name in enumerate(self.ordered_class_names)
        }


def load_class_names() -> list[str]:
    """Load and minimally validate the dataset's canonical class order."""
    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            f"클래스 순서 파일이 없습니다: {CLASS_NAMES_PATH}. "
            "먼저 scripts/prepare_dataset.py를 실행하세요."
        )
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        class_names = json.load(file)
    if not isinstance(class_names, list) or len(class_names) != 10:
        raise ValueError("class_names.json은 중복 없는 10개 클래스 배열이어야 합니다.")
    if len(set(class_names)) != len(class_names) or not all(
        isinstance(name, str) and name for name in class_names
    ):
        raise ValueError("class_names.json에 중복 또는 잘못된 클래스명이 있습니다.")
    return class_names


def create_train_transform(image_size: int = IMAGE_SIZE):
    """병변 전체를 유지하면서 위치·크기·조명 변화를 적용한다."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=15,
                        translate=(0.05, 0.05),
                        scale=(0.90, 1.05),
                        fill=(124, 116, 104),
                    )
                ],
                p=0.5,
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.15,
                        contrast=0.15,
                        saturation=0.10,
                        hue=0.02,
                    )
                ],
                p=0.5,
            ),
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
    class_names = load_class_names()
    train_dataset = OrderedImageFolder(
        PROCESSED_DIR / "train",
        class_names,
        transform=current_train_transform,
    )
    val_dataset = OrderedImageFolder(
        PROCESSED_DIR / "validation",
        class_names,
        transform=current_eval_transform,
    )
    test_dataset = OrderedImageFolder(
        PROCESSED_DIR / "test",
        class_names,
        transform=current_eval_transform,
    )

    if not (
        train_dataset.class_to_idx
        == val_dataset.class_to_idx
        == test_dataset.class_to_idx
    ):
        raise ValueError("train/validation/test의 클래스 폴더 구성이 서로 다릅니다.")

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

    return train_loader, val_loader, test_loader, train_dataset.classes


if __name__ == "__main__":
    loaders = create_dataloaders()
    images, labels = next(iter(loaders[0]))
    print("Classes:", loaders[3])
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
