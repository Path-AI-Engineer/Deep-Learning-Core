from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ReconstructionMetrics:
    mse: float
    mae: float
    psnr: float | None
    ssim: float

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def reconstruction_metrics(target: Tensor, prediction: Tensor) -> ReconstructionMetrics:
    if target.shape != prediction.shape:
        raise ValueError("target and prediction shapes must match")
    if target.ndim != 4 or target.shape[1:] != (1, 28, 28):
        raise ValueError("images must have shape [N, 1, 28, 28]")
    if not torch.isfinite(target).all() or not torch.isfinite(prediction).all():
        raise ValueError("images must be finite")
    mse = float(torch.mean((prediction - target) ** 2).item())
    mae = float(torch.mean(torch.abs(prediction - target)).item())
    psnr = None if mse == 0 else 10.0 * math.log10(1.0 / mse)
    target_np = target.detach().cpu().numpy()
    prediction_np = prediction.detach().cpu().numpy()
    similarity = cast(Callable[..., float], structural_similarity)
    scores = [
        similarity(
            target_np[index, 0],
            prediction_np[index, 0],
            data_range=1.0,
        )
        for index in range(len(target_np))
    ]
    return ReconstructionMetrics(
        mse=round(mse, 7),
        mae=round(mae, 7),
        psnr=None if psnr is None else round(psnr, 5),
        ssim=round(float(np.mean(scores)), 6),
    )


def per_sample_mse(target: Tensor, prediction: Tensor) -> Tensor:
    return torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
