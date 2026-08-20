"""환경변수 기반의 단순한 서비스 설정."""

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_path: Path
    allowed_origins: tuple[str, ...]
    max_upload_size_bytes: int
    enable_gradcam: bool

    @classmethod
    def from_env(cls) -> "Settings":
        origins = tuple(
            origin.strip()
            for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
            if origin.strip()
        )
        max_upload_size_mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
        if max_upload_size_mb <= 0:
            raise ValueError("MAX_UPLOAD_SIZE_MB는 0보다 커야 합니다.")
        return cls(
            model_path=_resolve_project_path(
                os.getenv(
                    "MODEL_PATH", "outputs/models/efficientnet_b0_best.pth"
                )
            ),
            allowed_origins=origins,
            max_upload_size_bytes=max_upload_size_mb * 1024 * 1024,
            enable_gradcam=_parse_bool(os.getenv("ENABLE_GRADCAM", "false")),
        )
