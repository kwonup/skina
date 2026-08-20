"""skina FastAPI application."""

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from service.api.schemas import HealthResponse
from service.api.settings import Settings
from src.pipeline.inference import InferenceModel, load_inference_model


ModelLoader = Callable[..., InferenceModel]


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

    return application


app = create_app()
