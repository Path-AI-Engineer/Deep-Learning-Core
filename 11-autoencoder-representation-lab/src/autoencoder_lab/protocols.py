from __future__ import annotations

from typing import Protocol

from torch import Tensor


class EncoderDecoder(Protocol):
    def encode(self, inputs: Tensor) -> Tensor: ...

    def decode(self, latent: Tensor) -> Tensor: ...
