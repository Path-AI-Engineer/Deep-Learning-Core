from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_tabular.contracts import ExperimentConfig


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[float]]:
        return asdict(self)


@dataclass
class FitResult:
    history: History
    best_epoch: int
    best_validation_loss: float
    state_dict: dict[str, torch.Tensor]
    stopped_early: bool


def _criterion(task: str) -> nn.Module:
    return nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    observations = 0
    for features, target in loader:
        features, target = features.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features), target)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(features)
        observations += len(features)
    return total_loss / max(observations, 1)


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    observations = 0
    with torch.inference_mode():
        for features, target in loader:
            features, target = features.to(device), target.to(device)
            loss = criterion(model(features), target)
            total_loss += float(loss) * len(features)
            observations += len(features)
    return total_loss / max(observations, 1)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device | None = None,
) -> FitResult:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = _criterion(config.task)
    optimizer_type = torch.optim.Adam if config.optimizer == "adam" else torch.optim.SGD
    optimizer = optimizer_type(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history = History()
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    for epoch in range(config.epochs):
        history.train_loss.append(
            train_epoch(model, train_loader, optimizer, criterion, device)
        )
        validation_loss = validate_epoch(model, validation_loader, criterion, device)
        history.validation_loss.append(validation_loss)
        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    model.to("cpu")
    return FitResult(
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        state_dict=best_state,
        stopped_early=len(history.train_loss) < config.epochs,
    )
