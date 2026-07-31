from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class TransformerSchedule(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        d_model: int,
        warmup_steps: int,
        factor: float = 1.0,
    ) -> None:
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be positive.")
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        step = max(self.last_epoch + 1, 1)
        scale = (
            self.factor
            * self.d_model ** -0.5
            * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        )
        return [scale for _ in self.optimizer.param_groups]


def schedule_value(
    step: int,
    *,
    d_model: int,
    warmup_steps: int,
    factor: float = 1.0,
) -> float:
    if step < 1:
        raise ValueError("Schedule steps start at one.")
    return (
        factor
        * math.pow(d_model, -0.5)
        * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))
    )

