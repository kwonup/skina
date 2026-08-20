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
    name_ko: str = ""
    name_en: str = ""
    confidence: float


class TopPrediction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    rank: int
    class_name: str = Field(alias="class")
    name_ko: str = ""
    probability: float


class LesionInformation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(alias="class")
    name_ko: str = ""
    name_en: str = ""
    category: str = ""
    description: str = ""
    features: list[str] = Field(default_factory=list)
    precautions: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    prediction: PredictionSummary
    top3: list[TopPrediction]
    inference_time_ms: float
    information: Optional[LesionInformation] = None
    gradcam: Optional[dict] = None
    disclaimer: str


class LesionsResponse(BaseModel):
    lesions: list[LesionInformation]
    disclaimer: str
