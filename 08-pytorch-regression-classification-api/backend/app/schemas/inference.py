from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float]


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[dict[str, float]] = Field(min_length=1)
