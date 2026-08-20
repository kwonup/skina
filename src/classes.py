"""프로젝트 전체에서 사용하는 클래스 순서와 검증 함수를 제공한다."""

import json
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES_PATH = PROJECT_ROOT / "configs" / "class_names.json"


def load_class_names(path: Path = CLASS_NAMES_PATH) -> tuple[str, ...]:
    """JSON에 저장된 클래스 이름을 순서를 보존해 읽는다."""
    with path.open("r", encoding="utf-8") as file:
        values = json.load(file)

    if not isinstance(values, list) or not values:
        raise ValueError(f"클래스 목록은 비어 있지 않은 JSON 배열이어야 합니다: {path}")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"모든 클래스 이름은 비어 있지 않은 문자열이어야 합니다: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"중복된 클래스 이름이 있습니다: {path}")
    return tuple(values)


CLASS_NAMES = load_class_names()
NUM_CLASSES = len(CLASS_NAMES)


def validate_class_names(
    actual: Iterable[str],
    *,
    source: str,
    expected: Sequence[str] = CLASS_NAMES,
) -> tuple[str, ...]:
    """클래스 개수와 index 순서가 중앙 목록과 같은지 확인한다."""
    actual_names = tuple(actual)
    expected_names = tuple(expected)
    if actual_names != expected_names:
        raise ValueError(
            f"{source}의 클래스 순서가 configs/class_names.json과 다릅니다.\n"
            f"기대 클래스: {list(expected_names)}\n"
            f"실제 클래스: {list(actual_names)}"
        )
    return actual_names
