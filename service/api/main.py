"""skina FastAPI application."""

import logging
from io import BytesIO
from pathlib import Path
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from service.api.schemas import (
    HealthResponse,
    PredictResponse,
    PredictionSummary,
    TopPrediction,
)
from service.api.settings import Settings
from src.pipeline.inference import InferenceModel, load_inference_model


ModelLoader = Callable[..., InferenceModel]
LOGGER = logging.getLogger(__name__)
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


async def _read_validated_image(
    upload: UploadFile, max_size_bytes: int
) -> Image.Image:
    """업로드 크기·형식·실제 decode 가능 여부를 검사한다."""
    content_type = (upload.content_type or "").lower()
    extension = Path(upload.filename or "").suffix.lower()
    if content_type not in ALLOWED_IMAGE_TYPES or extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="JPG, JPEG, PNG 이미지 파일만 업로드할 수 있습니다.",
        )

    contents = await upload.read(max_size_bytes + 1)
    await upload.close()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="업로드한 이미지 파일이 비어 있습니다.",
        )
    if len(contents) > max_size_bytes:
        max_size_mb = max_size_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"이미지는 최대 {max_size_mb}MB까지 업로드할 수 있습니다.",
        )

    try:
        with Image.open(BytesIO(contents)) as opened_image:
            opened_image.load()
            image = opened_image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미지를 읽을 수 없습니다. 파일이 손상되지 않았는지 확인해 주세요.",
        ) from None
    return image


def create_app(
    settings: Optional[Settings] = None,
    model_loader: ModelLoader = load_inference_model,
) -> FastAPI:
    service_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.settings = service_settings
        application.state.inference_model = model_loader(
            service_settings.model_path
        )
        yield

    application = FastAPI(
        title="skina API",
        description="피부 이미지 분류 모델의 참고용 예측 API",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(service_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @application.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        inference_model: InferenceModel = request.app.state.inference_model
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=inference_model.model_name,
            num_classes=len(inference_model.class_names),
            device=str(inference_model.device),
        )

    @application.post("/predict", response_model=PredictResponse)
    async def predict(
        request: Request,
        image: UploadFile = File(..., description="JPG, JPEG 또는 PNG 피부 이미지"),
    ) -> PredictResponse:
        settings: Settings = request.app.state.settings
        inference_model: InferenceModel = request.app.state.inference_model
        decoded_image = await _read_validated_image(
            image, settings.max_upload_size_bytes
        )
        try:
            output = await run_in_threadpool(
                inference_model.predict, decoded_image, 3
            )
        except Exception as error:
            LOGGER.exception("이미지 추론 중 오류가 발생했습니다.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="이미지를 분석하지 못했습니다. 잠시 후 다시 시도해 주세요.",
            ) from error

        top3 = [
            TopPrediction(
                rank=item.rank,
                class_name=item.class_name,
                probability=item.probability,
            )
            for item in output.predictions
        ]
        return PredictResponse(
            prediction=PredictionSummary(
                class_name=top3[0].class_name,
                confidence=top3[0].probability,
            ),
            top3=top3,
            inference_time_ms=output.inference_time_ms,
        )

    return application


app = create_app()
