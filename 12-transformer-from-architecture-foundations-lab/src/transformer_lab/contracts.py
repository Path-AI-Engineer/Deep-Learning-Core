from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

TaskId = Literal["copy", "reverse", "recall"]
PositionKind = Literal["sinusoidal", "learned", "none"]
NormalizationKind = Literal["post", "pre"]


@dataclass(frozen=True)
class ModelConfig:
    vocabulary_size: int = 39
    d_model: int = 48
    num_heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.0
    positional_encoding: PositionKind = "sinusoidal"
    normalization: NormalizationKind = "post"
    activation: Literal["relu", "gelu"] = "gelu"
    max_length: int = 32

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.vocabulary_size < 8:
            raise ValueError("vocabulary_size is too small for the task contract.")
        if self.max_length < 4:
            raise ValueError("max_length must support BOS, task and EOS tokens.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SequenceExample:
    example_id: str
    task: TaskId
    split: str
    seed: int
    source_tokens: tuple[str, ...]
    target_tokens: tuple[str, ...]
    content_length: int
    canonical_hash: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

