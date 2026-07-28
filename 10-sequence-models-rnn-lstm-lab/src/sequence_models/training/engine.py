from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader

from sequence_models.contracts import TrainingConfig
from sequence_models.evaluation import classification_metrics


@dataclass(frozen=True, slots=True)
class EpochResult:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_accuracy: float
    validation_macro_f1: float
    gradient_norm_before_clip: float
    gradient_norm_after_clip: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    best_epoch: int
    best_validation_macro_f1: float
    stopped_early: bool
    elapsed_seconds: float
    history: list[EpochResult]
    state_dict: dict[str, Tensor]

    def history_dicts(self) -> list[dict[str, float | int]]:
        return [asdict(epoch) for epoch in self.history]


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _gradient_norm(parameters: list[nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().norm(2).item()) ** 2
    return float(squared**0.5)


def _move_batch(batch: tuple[Tensor, Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    inputs, labels = batch
    return inputs.to(device=device, dtype=torch.float32), labels.to(device=device, dtype=torch.long)


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device | None = None,
) -> tuple[float, NDArray[np.int64], NDArray[np.int64]]:
    target_device = device or torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    truth: list[int] = []
    predictions: list[int] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs, labels = _move_batch(batch, target_device)
            logits = model(inputs)
            losses.append(float(criterion(logits, labels).item()))
            truth.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    if not losses:
        raise ValueError("evaluation loader must contain at least one batch")
    return (
        float(np.mean(losses)),
        np.asarray(truth, dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
    )


def train(
    model: nn.Module,
    training_loader: DataLoader[tuple[Tensor, Tensor]],
    validation_loader: DataLoader[tuple[Tensor, Tensor]],
    config: TrainingConfig,
    device: torch.device | None = None,
) -> TrainingResult:
    config.validate()
    set_reproducible(config.seed)
    target_device = device or torch.device("cpu")
    model.to(target_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, Tensor] = {}
    stale_epochs = 0
    history: list[EpochResult] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses: list[float] = []
        before_norms: list[float] = []
        after_norms: list[float] = []
        for batch in training_loader:
            inputs, labels = _move_batch(batch, target_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss detected")
            loss.backward()
            before_norms.append(_gradient_norm(parameters))
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
            after_norms.append(_gradient_norm(parameters))
            optimizer.step()
            train_losses.append(float(loss.item()))
        validation_loss, truth, prediction = evaluate(model, validation_loader, target_device)
        metrics = classification_metrics(truth, prediction)
        result = EpochResult(
            epoch=epoch,
            train_loss=float(np.mean(train_losses)),
            validation_loss=validation_loss,
            validation_accuracy=metrics.accuracy,
            validation_macro_f1=metrics.macro_f1,
            gradient_norm_before_clip=float(np.mean(before_norms)),
            gradient_norm_after_clip=float(np.mean(after_norms)),
        )
        history.append(result)
        if metrics.macro_f1 > best_score + 1e-8:
            best_score = metrics.macro_f1
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break
    model.load_state_dict(best_state)
    return TrainingResult(
        best_epoch=best_epoch,
        best_validation_macro_f1=best_score,
        stopped_early=len(history) < config.epochs,
        elapsed_seconds=time.perf_counter() - started,
        history=history,
        state_dict=copy.deepcopy(best_state),
    )
