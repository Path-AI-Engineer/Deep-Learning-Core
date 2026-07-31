from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator

TaskId = Literal["copy", "reverse", "recall"]
TraceType = Literal["encoder_self", "decoder_self", "cross"]


class PredictRequest(BaseModel):
    task: TaskId
    source_symbols: list[str] | None = Field(default=None, max_length=22)
    sample_id: str | None = Field(default=None, max_length=120)
    model_id: Literal["transformer-v1.0.0"] = "transformer-v1.0.0"
    max_new_tokens: int = Field(default=24, ge=1, le=28)

    @model_validator(mode="after")
    def exactly_one_source(self) -> PredictRequest:
        if (self.source_symbols is None) == (self.sample_id is None):
            raise ValueError("Provide exactly one of source_symbols or sample_id.")
        return self


class TraceRequest(PredictRequest):
    trace_type: TraceType = "cross"
    layer: int = Field(default=0, ge=0, le=1)
    head: int = Field(default=0, ge=0, le=3)


class AttentionComputeRequest(BaseModel):
    query: list[list[float]] = Field(min_length=1, max_length=8)
    key: list[list[float]] = Field(min_length=1, max_length=8)
    value: list[list[float]] = Field(min_length=1, max_length=8)
    mask: list[list[bool]] | None = None

    @model_validator(mode="after")
    def validate_matrices(self) -> AttentionComputeRequest:
        widths = {
            "query": {len(row) for row in self.query},
            "key": {len(row) for row in self.key},
            "value": {len(row) for row in self.value},
        }
        if any(len(values) != 1 or not next(iter(values), 0) for values in widths.values()):
            raise ValueError("Each matrix must be rectangular and non-empty.")
        if next(iter(widths["query"])) != next(iter(widths["key"])):
            raise ValueError("Query and key feature dimensions must match.")
        if len(self.key) != len(self.value):
            raise ValueError("Key and value sequence lengths must match.")
        if self.mask is not None:
            if len(self.mask) != len(self.query) or any(
                len(row) != len(self.key) for row in self.mask
            ):
                raise ValueError("Mask must have shape [T_query, T_key].")
            if any(all(row) for row in self.mask):
                raise ValueError("Mask cannot block every key in a query row.")
        values = [*self.query, *self.key, *self.value]
        if not all(math.isfinite(value) for row in values for value in row):
            raise ValueError("Matrices must contain finite values.")
        return self

