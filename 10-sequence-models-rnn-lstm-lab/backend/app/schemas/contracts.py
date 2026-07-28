from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictionRequest(StrictModel):
    sample_id: str = Field(min_length=5, max_length=64, pattern=r"^[a-z0-9-]+$")
    model_id: Literal["active", "rnn", "lstm", "gru"] = "active"


def _default_model_ids() -> list[Literal["rnn", "lstm", "gru"]]:
    return ["rnn", "lstm", "gru"]


class CompareRequest(StrictModel):
    sample_id: str = Field(min_length=5, max_length=64, pattern=r"^[a-z0-9-]+$")
    model_ids: list[Literal["rnn", "lstm", "gru"]] = Field(
        default_factory=_default_model_ids, min_length=2, max_length=3
    )

    @field_validator("model_ids")
    @classmethod
    def unique_models(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("model_ids must be unique")
        return values


class SampleTraceRequest(StrictModel):
    sample_id: str = Field(min_length=5, max_length=64, pattern=r"^[a-z0-9-]+$")
    model_id: Literal["rnn", "lstm", "gru"]
    selected_units: list[int] = Field(default_factory=lambda: [0, 1, 2], max_length=6)
    start_timestep: int = Field(default=0, ge=0, le=127)
    end_timestep: int = Field(default=31, ge=0, le=127)

    @field_validator("selected_units")
    @classmethod
    def valid_units(cls, values: list[int]) -> list[int]:
        if len(values) != len(set(values)) or any(value < 0 or value >= 24 for value in values):
            raise ValueError("selected units must be unique values in 0..23")
        return values


class CellTraceRequest(StrictModel):
    cell_type: Literal["rnn", "lstm", "gru"]
    example_id: Literal["balanced-memory"] = "balanced-memory"
