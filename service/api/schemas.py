"""FastAPI 응답 계약."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    num_classes: int
    device: str


class PredictionSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(alias="class")
    confidence: float


class TopPrediction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    rank: int
    class_name: str = Field(alias="class")
    probability: float


class PredictResponse(BaseModel):
    prediction: PredictionSummary
    top3: list[TopPrediction]
    inference_time_ms: float
    information: Optional[dict] = None
    gradcam: Optional[dict] = None
