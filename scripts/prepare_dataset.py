"""Build the final 10-class Synthetic + Real image dataset.

The source datasets are treated as immutable.  Files are copied into a temporary
build directory, validated, and only then swapped into ``data/processed``.
Run with ``--overwrite`` when an existing processed dataset should be replaced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageOps


SEED = 42
SYNTHETIC_TRAIN_PER_CLASS = 500
REAL_TRAIN_PER_CLASS = 500
VALIDATION_PER_CLASS = 100
TEST_PER_CLASS = 100
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CLASS_NAMES = [
    "actinic_keratosis",
    "basal_cell_carcinoma",
    "dermatofibroma",
    "hemangioma",
    "lentigo",
    "malignant_melanoma",
    "melanocytic_nevus",
    "squamous_cell_carcinoma",
    "seborrheic_keratosis",
    "wart",
]

REAL_FOLDER_NAMES = {
    "actinic_keratosis": "AK(광선각화증)",
    "basal_cell_carcinoma": "BCC(기저세포암)",
    "dermatofibroma": "Dermatofibroma(피부섬유종)",
    "hemangioma": "Hemangioma(혈관종)",
    "lentigo": "Lentigo(흑자)",
    "malignant_melanoma": "Melanoma(흑색종)",
    "melanocytic_nevus": "Nevus(모반)",
    "squamous_cell_carcinoma": "SCC(편평세포암)",
    "seborrheic_keratosis": "Seborrheic keratosis(지루각화증)",
    "wart": "Wart(사마귀)",
}

EXPECTED_REAL_COUNTS = {
    "actinic_keratosis": 588,
    "basal_cell_carcinoma": 972,
    "dermatofibroma": 1123,
    "hemangioma": 2442,
    "lentigo": 1075,
    "malignant_melanoma": 540,
    "melanocytic_nevus": 2438,
    "squamous_cell_carcinoma": 1109,
    "seborrheic_keratosis": 1282,
    "wart": 2689,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNTHETIC_ROOT = PROJECT_ROOT / "data" / "raw" / "train"
DEFAULT_REAL_ROOT = Path(
    r"C:\Users\user\Downloads\cropped_images\cropped_images\10개클래스_합산"
)
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MANIFEST_FIELDS = [
    "original_path",
    "original_filename",
    "new_path",
    "new_filename",
    "source",
    "class",
    "split",
    "is_augmented",
    "augmentation_parent",
]

SUMMARY_FIELDS = [
    "class",
    "synthetic_train",
    "real_train_original",
    "real_train_augmented",
    "validation",
    "test",
    "train_total",
    "predict_pool_real",
    "predict_pool_synthetic",
    "official_split_hash_overlaps",
    "predict_pool_hash_overlaps",
    "data_leakage_check",
    "seed",
]


def list_images(directory: Path) -> list[Path]:
    """Return supported image files directly below a directory in stable order."""
    if not directory.is_dir():
        raise FileNotFoundError(f"이미지 폴더를 찾을 수 없습니다: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_images(
    synthetic_root: Path, real_root: Path
) -> dict[str, dict[str, list[Path]]]:
    """Discover and validate the ten configured Synthetic and Real classes."""
    collected: dict[str, dict[str, list[Path]]] = {}
    for class_name in CLASS_NAMES:
        synthetic = list_images(synthetic_root / class_name)
        real = list_images(real_root / REAL_FOLDER_NAMES[class_name])
        if len(synthetic) != 800:
            raise ValueError(
                f"{class_name}: Synthetic train이 800장이 아니라 {len(synthetic)}장입니다."
            )
        expected_real = EXPECTED_REAL_COUNTS[class_name]
        if len(real) != expected_real:
            raise ValueError(
                f"{class_name}: Real 원본이 {expected_real}장이 아니라 {len(real)}장입니다."
            )
        collected[class_name] = {"synthetic": synthetic, "real": real}
    return collected


def sample_synthetic_images(
    images: list[Path], class_index: int
) -> tuple[list[Path], list[Path]]:
    """Select 500 Synthetic train images and return the 300-image remainder."""
    shuffled = images.copy()
    random.Random(SEED + class_index).shuffle(shuffled)
    return (
        shuffled[:SYNTHETIC_TRAIN_PER_CLASS],
        shuffled[SYNTHETIC_TRAIN_PER_CLASS:],
    )


def split_real_images(
    images: list[Path], class_index: int
) -> dict[str, list[Path]]:
    """Split Real originals before augmentation using the fixed random seed."""
    shuffled = images.copy()
    random.Random(SEED + 1000 + class_index).shuffle(shuffled)
    test = shuffled[:TEST_PER_CLASS]
    validation = shuffled[
        TEST_PER_CLASS : TEST_PER_CLASS + VALIDATION_PER_CLASS
    ]
    candidates = shuffled[TEST_PER_CLASS + VALIDATION_PER_CLASS :]
    train = candidates[:REAL_TRAIN_PER_CLASS]
    predict_pool = candidates[REAL_TRAIN_PER_CLASS:]
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "predict_pool": predict_pool,
    }


def logical_path(path: Path) -> str:
    """Represent project files relative to the repository when possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def make_manifest_row(
    *,
    source_path: Path,
    logical_destination: Path,
    source: str,
    class_name: str,
    split: str,
    content_hash: str,
    is_augmented: bool = False,
    augmentation_parent: str = "",
) -> dict[str, Any]:
    """Create a manifest row plus its internal validation hash."""
    return {
        "original_path": str(source_path.resolve()),
        "original_filename": source_path.name,
        "new_path": logical_path(logical_destination),
        "new_filename": logical_destination.name,
        "source": source,
        "class": class_name,
        "split": split,
        "is_augmented": str(is_augmented).lower(),
        "augmentation_parent": augmentation_parent,
        "_sha256": content_hash,
    }


def copy_and_rename(
    source_path: Path,
    build_destination: Path,
    logical_destination: Path,
    *,
    source: str,
    class_name: str,
    split: str,
    hash_cache: dict[Path, str],
) -> dict[str, Any]:
    """Copy one immutable source image and return its traceability record."""
    build_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, build_destination)
    content_hash = hash_cache.setdefault(source_path, sha256_file(source_path))
    return make_manifest_row(
        source_path=source_path,
        logical_destination=logical_destination,
        source=source,
        class_name=class_name,
        split=split,
        content_hash=content_hash,
    )


def save_augmented_image(source_path: Path, destination: Path, rng: random.Random) -> None:
    """Create one conservative, deterministic augmentation of a Real train image."""
    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    if rng.random() < 0.5:
        image = ImageOps.mirror(image)

    width, height = image.size
    crop_scale = rng.uniform(0.96, 0.995)
    crop_width = max(1, round(width * crop_scale))
    crop_height = max(1, round(height * crop_scale))
    max_left = width - crop_width
    max_top = height - crop_height
    left = rng.randint(0, max_left) if max_left else 0
    top = rng.randint(0, max_top) if max_top else 0
    image = image.crop((left, top, left + crop_width, top + crop_height))
    image = image.resize((width, height), Image.Resampling.BICUBIC)

    angle = rng.uniform(3.0, 10.0) * (-1 if rng.random() < 0.5 else 1)
    edge_color = image.resize((1, 1), Image.Resampling.BILINEAR).getpixel((0, 0))
    image = image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=edge_color,
    )
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.96, 1.04))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.96, 1.04))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.97, 1.03))

    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(destination, quality=95, subsampling=0)
    elif suffix == ".png":
        image.save(destination, compress_level=6)
    else:
        image.save(destination)


def augment_training_images(
    *,
    class_name: str,
    class_index: int,
    train_originals: list[Path],
    source_indices: dict[Path, int],
    parent_filenames: dict[Path, str],
    needed: int,
    build_class_dir: Path,
    logical_class_dir: Path,
) -> list[dict[str, Any]]:
    """Generate only the number of Real train files needed to reach 500."""
    if needed <= 0:
        return []
    if not train_originals:
        raise ValueError(f"{class_name}: 증강할 Real train 원본이 없습니다.")

    parents = train_originals.copy()
    rng = random.Random(SEED + 2000 + class_index)
    rng.shuffle(parents)
    per_parent_counter: Counter[Path] = Counter()
    rows: list[dict[str, Any]] = []
    for position in range(needed):
        parent = parents[position % len(parents)]
        per_parent_counter[parent] += 1
        parent_index = source_indices[parent]
        new_name = (
            f"real_{class_name}_{parent_index:04d}_"
            f"aug{per_parent_counter[parent]:02d}{parent.suffix.lower()}"
        )
        build_destination = build_class_dir / new_name
        logical_destination = logical_class_dir / new_name
        save_augmented_image(parent, build_destination, rng)
        rows.append(
            make_manifest_row(
                source_path=parent,
                logical_destination=logical_destination,
                source="real",
                class_name=class_name,
                split="train",
                content_hash=sha256_file(build_destination),
                is_augmented=True,
                augmentation_parent=parent_filenames[parent],
            )
        )
    return rows


def build_manifest(rows: list[dict[str, Any]], destination: Path) -> None:
    """Save traceability records for official splits and both predict pools."""
    with destination.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in MANIFEST_FIELDS})


def validate_counts(rows: list[dict[str, Any]]) -> None:
    """Validate per-class source/split counts and the 12,000 official total."""
    errors: list[str] = []
    for class_name in CLASS_NAMES:
        class_rows = [row for row in rows if row["class"] == class_name]
        expected = {
            "synthetic train": SYNTHETIC_TRAIN_PER_CLASS,
            "real train": REAL_TRAIN_PER_CLASS,
            "validation": VALIDATION_PER_CLASS,
            "test": TEST_PER_CLASS,
        }
        actual = {
            "synthetic train": sum(
                row["source"] == "synthetic" and row["split"] == "train"
                for row in class_rows
            ),
            "real train": sum(
                row["source"] == "real" and row["split"] == "train"
                for row in class_rows
            ),
            "validation": sum(row["split"] == "validation" for row in class_rows),
            "test": sum(row["split"] == "test" for row in class_rows),
        }
        for label, expected_count in expected.items():
            if actual[label] != expected_count:
                errors.append(
                    f"{class_name}: {label}={actual[label]} (기대 {expected_count})"
                )

        invalid_eval = [
            row
            for row in class_rows
            if row["split"] in {"validation", "test"}
            and (row["source"] != "real" or row["is_augmented"] != "false")
        ]
        if invalid_eval:
            errors.append(f"{class_name}: Validation/Test에 Real 원본이 아닌 파일이 있습니다.")

    official_total = sum(
        row["split"] in {"train", "validation", "test"} for row in rows
    )
    if official_total != 12000:
        errors.append(f"공식 split 총합={official_total} (기대 12000)")
    if errors:
        raise ValueError("이미지 수 검증 실패:\n- " + "\n- ".join(errors))


def hash_sets_by_split(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Return content hashes grouped by manifest split."""
    result: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        result[row["split"]].add(row["_sha256"])
    return result


def validate_duplicates(rows: list[dict[str, Any]]) -> int:
    """Reject identical image bytes occurring in different official splits."""
    hashes = hash_sets_by_split(rows)
    overlaps = {
        "train/validation": hashes["train"] & hashes["validation"],
        "train/test": hashes["train"] & hashes["test"],
        "validation/test": hashes["validation"] & hashes["test"],
    }
    count = sum(len(values) for values in overlaps.values())
    if count:
        details = ", ".join(f"{name}={len(values)}" for name, values in overlaps.items())
        raise ValueError(f"공식 split SHA-256 중복 검출: {details}")
    return count


def validate_data_leakage(rows: list[dict[str, Any]]) -> int:
    """Validate augmentation parents and isolation of the predict pool."""
    originals_by_name = {
        row["new_filename"]: row
        for row in rows
        if row["source"] == "real" and row["is_augmented"] == "false"
    }
    for row in rows:
        if row["is_augmented"] != "true":
            continue
        parent = originals_by_name.get(row["augmentation_parent"])
        if parent is None:
            raise ValueError(
                f"증강 parent를 manifest에서 찾을 수 없습니다: {row['augmentation_parent']}"
            )
        if parent["split"] != "train" or parent["class"] != row["class"]:
            raise ValueError(
                f"증강 parent 누수: {row['new_filename']} -> "
                f"{parent['split']}/{parent['new_filename']}"
            )

    hashes = hash_sets_by_split(rows)
    official_hashes = hashes["train"] | hashes["validation"] | hashes["test"]
    pool_overlap = hashes["predict_pool"] & official_hashes
    if pool_overlap:
        raise ValueError(
            f"Predict Pool과 공식 split 사이 SHA-256 중복 {len(pool_overlap)}건을 검출했습니다."
        )
    return len(pool_overlap)


def save_summary(
    rows: list[dict[str, Any]],
    destination: Path,
    official_overlap_count: int,
    pool_overlap_count: int,
) -> None:
    """Save the requested class-level count and validation report."""
    summary_rows: list[dict[str, Any]] = []
    for class_name in CLASS_NAMES:
        class_rows = [row for row in rows if row["class"] == class_name]
        row = {
            "class": class_name,
            "synthetic_train": sum(
                item["source"] == "synthetic" and item["split"] == "train"
                for item in class_rows
            ),
            "real_train_original": sum(
                item["source"] == "real"
                and item["split"] == "train"
                and item["is_augmented"] == "false"
                for item in class_rows
            ),
            "real_train_augmented": sum(
                item["source"] == "real"
                and item["split"] == "train"
                and item["is_augmented"] == "true"
                for item in class_rows
            ),
            "validation": sum(item["split"] == "validation" for item in class_rows),
            "test": sum(item["split"] == "test" for item in class_rows),
            "train_total": sum(item["split"] == "train" for item in class_rows),
            "predict_pool_real": sum(
                item["source"] == "real" and item["split"] == "predict_pool"
                for item in class_rows
            ),
            "predict_pool_synthetic": sum(
                item["source"] == "synthetic" and item["split"] == "predict_pool"
                for item in class_rows
            ),
            "official_split_hash_overlaps": official_overlap_count,
            "predict_pool_hash_overlaps": pool_overlap_count,
            "data_leakage_check": "passed",
            "seed": SEED,
        }
        summary_rows.append(row)

    total: dict[str, Any] = {"class": "TOTAL"}
    numeric_fields = [
        "synthetic_train",
        "real_train_original",
        "real_train_augmented",
        "validation",
        "test",
        "train_total",
        "predict_pool_real",
        "predict_pool_synthetic",
    ]
    for field in numeric_fields:
        total[field] = sum(int(row[field]) for row in summary_rows)
    total.update(
        {
            "official_split_hash_overlaps": official_overlap_count,
            "predict_pool_hash_overlaps": pool_overlap_count,
            "data_leakage_check": "passed",
            "seed": SEED,
        }
    )
    summary_rows.append(total)

    with destination.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)


def prepare_dataset(
    *,
    synthetic_root: Path,
    real_root: Path,
    processed_dir: Path,
    overwrite: bool,
) -> None:
    """Build, validate, and atomically install the final processed dataset."""
    synthetic_root = synthetic_root.resolve()
    real_root = real_root.resolve()
    processed_dir = processed_dir.resolve()
    build_dir = processed_dir.parent / ".processed_build"
    backup_dir = processed_dir.parent / ".processed_backup"

    if processed_dir.exists() and any(processed_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{processed_dir}에 기존 데이터가 있습니다. 재생성하려면 --overwrite를 전달하세요."
        )
    if backup_dir.exists():
        raise FileExistsError(
            f"이전 교체 작업의 백업이 남아 있습니다: {backup_dir}. 내용을 확인해 주세요."
        )
    if build_dir.exists():
        shutil.rmtree(build_dir)

    collected = collect_images(synthetic_root, real_root)
    rows: list[dict[str, Any]] = []
    hash_cache: dict[Path, str] = {}

    try:
        for split in ("train", "validation", "test"):
            for class_name in CLASS_NAMES:
                (build_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        for source in ("real", "synthetic"):
            for class_name in CLASS_NAMES:
                (build_dir / "predict_pool" / source / class_name).mkdir(
                    parents=True, exist_ok=True
                )

        for class_index, class_name in enumerate(CLASS_NAMES):
            synthetic_images = collected[class_name]["synthetic"]
            real_images = collected[class_name]["real"]
            synthetic_indices = {
                path: index for index, path in enumerate(synthetic_images, start=1)
            }
            real_indices = {path: index for index, path in enumerate(real_images, start=1)}

            synthetic_train, synthetic_pool = sample_synthetic_images(
                synthetic_images, class_index
            )
            for source_path in sorted(synthetic_train, key=synthetic_indices.get):
                new_name = (
                    f"syn_{class_name}_{synthetic_indices[source_path]:04d}"
                    f"{source_path.suffix.lower()}"
                )
                rows.append(
                    copy_and_rename(
                        source_path,
                        build_dir / "train" / class_name / new_name,
                        processed_dir / "train" / class_name / new_name,
                        source="synthetic",
                        class_name=class_name,
                        split="train",
                        hash_cache=hash_cache,
                    )
                )
            for pool_index, source_path in enumerate(synthetic_pool, start=1):
                new_name = (
                    f"syn_{class_name}_predict_{pool_index:04d}"
                    f"{source_path.suffix.lower()}"
                )
                rows.append(
                    copy_and_rename(
                        source_path,
                        build_dir / "predict_pool" / "synthetic" / class_name / new_name,
                        processed_dir
                        / "predict_pool"
                        / "synthetic"
                        / class_name
                        / new_name,
                        source="synthetic",
                        class_name=class_name,
                        split="predict_pool",
                        hash_cache=hash_cache,
                    )
                )

            real_split = split_real_images(real_images, class_index)
            parent_filenames: dict[Path, str] = {}
            for split in ("train", "validation", "test"):
                for source_path in sorted(real_split[split], key=real_indices.get):
                    new_name = (
                        f"real_{class_name}_{real_indices[source_path]:04d}"
                        f"{source_path.suffix.lower()}"
                    )
                    if split == "train":
                        parent_filenames[source_path] = new_name
                    rows.append(
                        copy_and_rename(
                            source_path,
                            build_dir / split / class_name / new_name,
                            processed_dir / split / class_name / new_name,
                            source="real",
                            class_name=class_name,
                            split=split,
                            hash_cache=hash_cache,
                        )
                    )

            for pool_index, source_path in enumerate(real_split["predict_pool"], start=1):
                new_name = (
                    f"real_{class_name}_predict_{pool_index:04d}"
                    f"{source_path.suffix.lower()}"
                )
                rows.append(
                    copy_and_rename(
                        source_path,
                        build_dir / "predict_pool" / "real" / class_name / new_name,
                        processed_dir / "predict_pool" / "real" / class_name / new_name,
                        source="real",
                        class_name=class_name,
                        split="predict_pool",
                        hash_cache=hash_cache,
                    )
                )

            augmentation_needed = REAL_TRAIN_PER_CLASS - len(real_split["train"])
            rows.extend(
                augment_training_images(
                    class_name=class_name,
                    class_index=class_index,
                    train_originals=real_split["train"],
                    source_indices=real_indices,
                    parent_filenames=parent_filenames,
                    needed=augmentation_needed,
                    build_class_dir=build_dir / "train" / class_name,
                    logical_class_dir=processed_dir / "train" / class_name,
                )
            )

            print(
                f"{class_name}: synthetic train=500, real train original="
                f"{len(real_split['train'])}, augmented={augmentation_needed}, "
                f"validation=100, test=100, real pool="
                f"{len(real_split['predict_pool'])}, synthetic pool=300"
            )

        validate_counts(rows)
        official_overlap_count = validate_duplicates(rows)
        pool_overlap_count = validate_data_leakage(rows)

        with (build_dir / "class_names.json").open("w", encoding="utf-8") as file:
            json.dump(CLASS_NAMES, file, ensure_ascii=False, indent=2)
            file.write("\n")
        build_manifest(rows, build_dir / "dataset_manifest.csv")
        save_summary(
            rows,
            build_dir / "split_summary.csv",
            official_overlap_count,
            pool_overlap_count,
        )

        physical_images = sum(
            1
            for path in build_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if physical_images != len(rows):
            raise ValueError(
                f"Manifest/파일 수 불일치: manifest={len(rows)}, files={physical_images}"
            )

        if processed_dir.exists():
            processed_dir.rename(backup_dir)
        try:
            build_dir.rename(processed_dir)
        except Exception:
            if backup_dir.exists() and not processed_dir.exists():
                backup_dir.rename(processed_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    except Exception:
        if build_dir.exists():
            shutil.rmtree(build_dir)
        raise

    official_total = sum(
        row["split"] in {"train", "validation", "test"} for row in rows
    )
    real_pool_total = sum(
        row["split"] == "predict_pool" and row["source"] == "real" for row in rows
    )
    synthetic_pool_total = sum(
        row["split"] == "predict_pool" and row["source"] == "synthetic"
        for row in rows
    )
    print("\n데이터셋 준비 및 검증 완료")
    print("Train=10000, Validation=1000, Test=1000, Total=12000")
    print(
        f"Predict Pool: real={real_pool_total}, synthetic={synthetic_pool_total}"
    )
    print(
        f"SHA-256 overlaps: official={official_overlap_count}, "
        f"predict_pool={pool_overlap_count}; seed={SEED}; manifest rows={len(rows)}"
    )
    if official_total != 12000:
        raise AssertionError("검증 후 공식 split 총합이 변경되었습니다.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="10클래스 Synthetic + Real 최종 데이터셋을 준비합니다."
    )
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=DEFAULT_SYNTHETIC_ROOT,
        help="클래스당 800장의 기존 Synthetic train 루트",
    )
    parser.add_argument(
        "--real-root",
        type=Path,
        default=DEFAULT_REAL_ROOT,
        help="10개 한글 클래스 폴더가 있는 Real 원본 루트",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="최종 processed 데이터셋 경로",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="검증 완료 후 기존 processed 데이터셋을 교체합니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(
        synthetic_root=args.synthetic_root,
        real_root=args.real_root,
        processed_dir=args.processed_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
