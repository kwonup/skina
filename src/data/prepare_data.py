"""AI-Hub 원본 이미지를 공통 train/val/test 데이터로 준비한다.

raw/train은 모두 train으로 복사하고, raw/validation은 클래스마다 seed 42로
50장씩 val과 test로 나눈다. 기존 processed 데이터는 --overwrite 없이는
수정하지 않아 실수로 서로 다른 split이 섞이는 것을 막는다.
"""

import argparse
import random
import shutil
from pathlib import Path


SEED = 42
EXPECTED_TRAIN_PER_CLASS = 800
EXPECTED_VALIDATION_PER_CLASS = 100
VAL_PER_CLASS = 50
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CLASS_NAMES = [
    "actinic_keratosis",
    "basal_cell_carcinoma",
    "bowen_disease",
    "dermatofibroma",
    "epidermal_cyst",
    "hemangioma",
    "lentigo",
    "malignant_melanoma",
    "melanocytic_nevus",
    "milia",
    "pyogenic_granuloma",
    "sebaceous_hyperplasia",
    "seborrheic_keratosis",
    "squamous_cell_carcinoma",
    "wart",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "train"
RAW_VALIDATION_DIR = PROJECT_ROOT / "data" / "raw" / "validation"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BUILD_DIR = PROJECT_ROOT / "data" / ".processed_build"


def list_images(directory: Path) -> list[Path]:
    """폴더 바로 아래의 지원 이미지 파일을 이름순으로 반환한다."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_raw_data() -> dict[str, tuple[list[Path], list[Path]]]:
    """복사를 시작하기 전에 클래스와 이미지 수가 예상과 같은지 확인한다."""
    if not RAW_TRAIN_DIR.is_dir() or not RAW_VALIDATION_DIR.is_dir():
        raise FileNotFoundError(
            "원본 데이터 폴더가 없습니다. data/raw/train과 "
            "data/raw/validation 아래에 클래스 폴더를 배치하세요."
        )

    train_classes = sorted(path.name for path in RAW_TRAIN_DIR.iterdir() if path.is_dir())
    validation_classes = sorted(
        path.name for path in RAW_VALIDATION_DIR.iterdir() if path.is_dir()
    )
    if train_classes != CLASS_NAMES or validation_classes != CLASS_NAMES:
        raise ValueError(
            "원본 데이터의 클래스 폴더가 프로젝트의 15개 클래스와 일치하지 않습니다.\n"
            f"기대 클래스: {CLASS_NAMES}\n"
            f"train 클래스: {train_classes}\n"
            f"validation 클래스: {validation_classes}"
        )

    files_by_class = {}
    for class_name in CLASS_NAMES:
        train_files = list_images(RAW_TRAIN_DIR / class_name)
        validation_files = list_images(RAW_VALIDATION_DIR / class_name)
        if len(train_files) != EXPECTED_TRAIN_PER_CLASS:
            raise ValueError(
                f"{class_name}: train 이미지가 {EXPECTED_TRAIN_PER_CLASS}장이 아니라 "
                f"{len(train_files)}장입니다."
            )
        if len(validation_files) != EXPECTED_VALIDATION_PER_CLASS:
            raise ValueError(
                f"{class_name}: validation 이미지가 "
                f"{EXPECTED_VALIDATION_PER_CLASS}장이 아니라 "
                f"{len(validation_files)}장입니다."
            )
        files_by_class[class_name] = (train_files, validation_files)

    return files_by_class


def has_contents(directory: Path) -> bool:
    """폴더에 실제 파일이 있는지 확인한다(빈 기본 split 폴더는 허용)."""
    return directory.exists() and any(path.is_file() for path in directory.rglob("*"))


def prepare_data(overwrite: bool = False) -> None:
    """검증된 원본을 임시 폴더에 만든 후 processed 폴더로 교체한다."""
    files_by_class = validate_raw_data()

    if has_contents(PROCESSED_DIR) and not overwrite:
        raise FileExistsError(
            f"{PROCESSED_DIR}에 기존 데이터가 있습니다. 동일 split을 유지하려면 "
            "그대로 사용하고, seed 42로 다시 만들려면 --overwrite를 사용하세요."
        )

    # 중간 실패가 기존 processed 데이터를 오염시키지 않도록 별도 폴더에 먼저 만든다.
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    try:
        for split_name in ("train", "val", "test"):
            (BUILD_DIR / split_name).mkdir(parents=True, exist_ok=True)

        rng = random.Random(SEED)
        for class_name in CLASS_NAMES:
            train_files, validation_files = files_by_class[class_name]
            shuffled_validation = validation_files.copy()
            rng.shuffle(shuffled_validation)
            val_files = shuffled_validation[:VAL_PER_CLASS]
            test_files = shuffled_validation[VAL_PER_CLASS:]

            split_files = {
                "train": train_files,
                "val": val_files,
                "test": test_files,
            }
            for split_name, image_files in split_files.items():
                destination = BUILD_DIR / split_name / class_name
                destination.mkdir(parents=True, exist_ok=True)
                for image_path in image_files:
                    shutil.copy2(image_path, destination / image_path.name)

            print(
                f"{class_name}: train={len(train_files)}, "
                f"val={len(val_files)}, test={len(test_files)}"
            )

        if PROCESSED_DIR.exists():
            shutil.rmtree(PROCESSED_DIR)
        BUILD_DIR.rename(PROCESSED_DIR)
    except Exception:
        if BUILD_DIR.exists():
            shutil.rmtree(BUILD_DIR)
        raise

    print("\n데이터 준비 완료: train=12000, val=750, test=750 (seed=42)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Hub 원본을 공통 train/val/test split으로 준비합니다."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 data/processed를 검증된 새 split으로 교체합니다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_data(overwrite=args.overwrite)
