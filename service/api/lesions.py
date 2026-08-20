"""검수된 정적 병변 정보를 읽고 class mapping과 일치하는지 검증한다."""

import json
from pathlib import Path
from typing import Mapping, Sequence

from service.api.schemas import LesionInformation


LESION_INFO_PATH = Path(__file__).with_name("lesion_info.json")
MEDICAL_DISCLAIMER = (
    "본 결과는 학습 프로젝트의 이미지 분류 모델이 제공하는 참고용 예측 "
    "결과이며, 의료적 진단을 대체하지 않습니다."
)


def load_lesion_catalog(
    class_names: Sequence[str], path: Path = LESION_INFO_PATH
) -> dict[str, LesionInformation]:
    """클래스 순서대로 병변 정보를 반환하고 누락/추가 key를 거부한다."""
    with path.open("r", encoding="utf-8") as file:
        raw_catalog = json.load(file)
    if not isinstance(raw_catalog, Mapping):
        raise ValueError("lesion_info.json의 최상위 값은 객체여야 합니다.")

    expected_keys = set(class_names)
    actual_keys = set(raw_catalog)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            "병변 정보와 checkpoint 클래스가 일치하지 않습니다. "
            f"누락={missing}, 추가={extra}"
        )

    return {
        class_name: LesionInformation(
            class_name=class_name,
            **raw_catalog[class_name],
        )
        for class_name in class_names
    }
