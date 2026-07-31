from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ReconstructRequest(BaseModel):
    sample_id: str = Field(min_length=1, max_length=64)
    model_id: str = Field(min_length=1, max_length=32)


class DenoiseRequest(BaseModel):
    sample_id: str = Field(min_length=1, max_length=64)
    corruption_type: Literal["gaussian", "masking"]
    corruption_level: float
    seed: Literal[7, 21, 42] = 42
    model_ids: list[str] = Field(default_factory=lambda: ["conv-ae", "denoising-ae"])

    @field_validator("corruption_level")
    @classmethod
    def validate_corruption_level(cls, value: float) -> float:
        if value not in (0.1, 0.2, 0.3):
            raise ValueError("corruption_level must be 0.1, 0.2 or 0.3")
        return value

    @field_validator("model_ids")
    @classmethod
    def validate_model_ids(cls, value: list[str]) -> list[str]:
        allowed = {"conv-ae", "denoising-ae"}
        if not 1 <= len(value) <= 2 or not set(value) <= allowed:
            raise ValueError("model_ids must contain one or both approved denoising models")
        return list(dict.fromkeys(value))


class LatentDecodeRequest(BaseModel):
    model_id: Literal["latent-2d"] = "latent-2d"
    x: float
    y: float


class InterpolateRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=32)
    sample_id_a: str = Field(min_length=1, max_length=64)
    sample_id_b: str = Field(min_length=1, max_length=64)
    steps: int = Field(default=7, ge=3, le=12)
