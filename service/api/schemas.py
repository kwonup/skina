"""FastAPI 응답 계약."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    num_classes: int
    device: str
