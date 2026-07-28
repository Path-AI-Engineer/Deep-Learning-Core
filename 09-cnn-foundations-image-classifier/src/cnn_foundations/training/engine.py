from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    examples: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    history: tuple[dict[str, Any], ...]
    best_state: dict[str, torch.Tensor]


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        if patience < 1:
            raise ValueError("patience must be positive.")
        self.patience = patience
        self.best_loss = float("inf")
        self.wait = 0

    def update(self, validation_loss: float) -> bool:
        if validation_loss < self.best_loss - 1e-8:
            self.best_loss = validation_loss
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    context = torch.enable_grad() if training else torch.inference_mode()  # type: ignore[no-untyped-call]
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("training produced a non-finite loss.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            examples = targets.shape[0]
            total_loss += float(loss.detach()) * examples
            total_correct += int((logits.argmax(dim=1) == targets).sum())
            total_examples += examples
    if not total_examples:
        raise ValueError("data loader produced no examples.")
    return EpochMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def fit(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    validation_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
) -> TrainingResult:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    stopper = EarlyStopping(patience)
    history: list[dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device=device, optimizer=optimizer
        )
        validation_metrics = run_epoch(
            model, validation_loader, criterion, device=device
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics.as_dict(),
                "validation": validation_metrics.as_dict(),
            }
        )
        if validation_metrics.loss < best_loss:
            best_loss = validation_metrics.loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if stopper.update(validation_metrics.loss):
            break
    model.load_state_dict(best_state)
    return TrainingResult(best_epoch, tuple(history), best_state)
