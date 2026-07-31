from __future__ import annotations

import torch
from torch import Tensor

from autoencoder_lab.contracts import CorruptionType


def corrupt(
    images: Tensor,
    corruption_type: CorruptionType,
    level: float,
    seed: int = 42,
) -> Tensor:
    if level not in (0.1, 0.2, 0.3):
        raise ValueError("corruption level must be 0.1, 0.2 or 0.3")
    generator = torch.Generator(device=images.device).manual_seed(seed)
    if corruption_type == "gaussian":
        noise = torch.randn(images.shape, generator=generator, device=images.device)
        return (images + noise * level).clamp(0, 1)
    if corruption_type == "masking":
        mask = torch.rand(images.shape, generator=generator, device=images.device) >= level
        return images * mask
    raise ValueError("corruption type must be gaussian or masking")
