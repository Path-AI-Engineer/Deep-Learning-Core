from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SamplePredictionRequest(Contract):
    sample_id: str = Field(pattern=r"^test-\d{5}$")


class ActivationRequest(Contract):
    sample_id: str = Field(pattern=r"^test-\d{5}$")
    layer_id: str
    limit: int = Field(default=8, ge=1, le=12)


MatrixValue = Annotated[float, Field(ge=-100.0, le=100.0)]


class ConvolutionRequest(Contract):
    matrix: list[list[MatrixValue]]
    kernel: list[list[MatrixValue]]
    stride: int = Field(default=1, ge=1, le=2)
    padding: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_shapes(self) -> ConvolutionRequest:
        if not self.matrix or len(self.matrix) > 12:
            raise ValueError("matrix height must be between 1 and 12.")
        width = len(self.matrix[0])
        if width < 1 or width > 12 or any(len(row) != width for row in self.matrix):
            raise ValueError("matrix rows must share a width between 1 and 12.")
        if len(self.kernel) != 3 or any(len(row) != 3 for row in self.kernel):
            raise ValueError("kernel must have shape [3, 3].")
        return self

