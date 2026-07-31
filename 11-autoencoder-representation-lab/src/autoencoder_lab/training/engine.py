from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from autoencoder_lab.contracts import TrainingConfig
from autoencoder_lab.corruption import corrupt


@dataclass(frozen=True, slots=True)
class EpochResult:
    epoch: int
    train_mse: float
    validation_mse: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    best_epoch: int
    best_validation_mse: float
    stopped_early: bool
    elapsed_seconds: float
    history: list[EpochResult]
    state_dict: dict[str, Tensor]

    def history_dicts(self) -> list[dict[str, int | float]]:
        return [asdict(item) for item in self.history]


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _images(batch: Tensor | tuple[Tensor, ...]) -> Tensor:
    values = batch[0] if isinstance(batch, tuple | list) else batch
    return values.to(dtype=torch.float32)


def evaluate(model: nn.Module, loader: DataLoader[Tensor]) -> float:
    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = _images(batch)
            losses.append(float(nn.functional.mse_loss(model(images), images).item()))
    if not losses:
        raise ValueError("evaluation loader is empty")
    return float(np.mean(losses))


def train(
    model: nn.Module,
    training_loader: DataLoader[Tensor],
    validation_loader: DataLoader[Tensor],
    config: TrainingConfig,
    denoising: bool = False,
    corruption_type: Literal["gaussian", "masking"] = "gaussian",
    corruption_level: float = 0.2,
) -> TrainingResult:
    config.validate()
    set_reproducible(config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] = {}
    stale = 0
    history: list[EpochResult] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_index, batch in enumerate(training_loader):
            clean = _images(batch)
            inputs = (
                corrupt(
                    clean,
                    corruption_type,
                    corruption_level,
                    seed=config.seed + epoch * 1000 + batch_index,
                )
                if denoising
                else clean
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = nn.functional.mse_loss(prediction, clean)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            losses.append(float(loss.item()))
        validation_mse = evaluate(model, validation_loader)
        history.append(
            EpochResult(
                epoch=epoch,
                train_mse=float(np.mean(losses)),
                validation_mse=validation_mse,
            )
        )
        if validation_mse < best_loss - 1e-8:
            best_loss = validation_mse
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                break
    model.load_state_dict(best_state)
    return TrainingResult(
        best_epoch=best_epoch,
        best_validation_mse=best_loss,
        stopped_early=len(history) < config.epochs,
        elapsed_seconds=time.perf_counter() - started,
        history=history,
        state_dict=copy.deepcopy(best_state),
    )
