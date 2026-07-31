from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.nn import functional

from transformer_lab.contracts import SequenceExample
from transformer_lab.data import collate_examples
from transformer_lab.models import SequenceTransformer
from transformer_lab.tokenization import PAD
from transformer_lab.training.schedule import TransformerSchedule


@dataclass(frozen=True)
class TrainingResult:
    best_state: dict[str, torch.Tensor]
    best_epoch: int
    history: tuple[dict[str, Any], ...]
    training_seconds: float
    optimizer_steps: int
    stopped_by: str


def sequence_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape[:2] != target.shape:
        raise ValueError("Logits and shifted targets must share batch and time shapes.")
    return functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        ignore_index=PAD,
    )


def _batches(
    examples: list[SequenceExample],
    *,
    batch_size: int,
    seed: int,
) -> list[list[SequenceExample]]:
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    return [
        [examples[index] for index in indices[offset : offset + batch_size]]
        for offset in range(0, len(indices), batch_size)
    ]


def evaluate_loss(
    model: SequenceTransformer,
    examples: list[SequenceExample],
    *,
    batch_size: int,
) -> float:
    model.eval()
    weighted_loss = 0.0
    tokens = 0
    with torch.inference_mode():
        for batch_examples in _batches(examples, batch_size=batch_size, seed=0):
            batch = collate_examples(batch_examples)
            logits, _ = model(
                batch["source_ids"],
                batch["target_input"],
                source_padding_mask=batch["source_padding_mask"],
                target_padding_mask=batch["target_padding_mask"],
            )
            loss = sequence_loss(logits, batch["target_output"])
            valid_tokens = int(batch["target_output"].ne(PAD).sum())
            weighted_loss += float(loss) * valid_tokens
            tokens += valid_tokens
    return weighted_loss / max(tokens, 1)


def train(
    model: SequenceTransformer,
    training_examples: list[SequenceExample],
    validation_examples: list[SequenceExample],
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    patience: int = 6,
    warmup_steps: int = 40,
    gradient_clip: float = 1.0,
) -> TrainingResult:
    torch.manual_seed(seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98))
    scheduler = TransformerSchedule(
        optimizer,
        d_model=model.config.d_model,
        warmup_steps=warmup_steps,
        factor=0.7,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    optimizer_steps = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    stopped_by = "epoch_budget"
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        gradient_norms: list[float] = []
        for batch_examples in _batches(
            training_examples,
            batch_size=batch_size,
            seed=seed + epoch,
        ):
            batch = collate_examples(batch_examples)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(
                batch["source_ids"],
                batch["target_input"],
                source_padding_mask=batch["source_padding_mask"],
                target_padding_mask=batch["target_padding_mask"],
            )
            loss = sequence_loss(logits, batch["target_output"])
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss.")
            loss.backward()  # type: ignore[no-untyped-call]
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Training produced a non-finite gradient norm.")
            optimizer.step()
            scheduler.step()
            optimizer_steps += 1
            losses.append(float(loss.detach()))
            gradient_norms.append(float(gradient_norm))
        validation_loss = evaluate_loss(
            model, validation_examples, batch_size=batch_size
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": sum(losses) / len(losses),
                "validation_loss": validation_loss,
                "gradient_norm": sum(gradient_norms) / len(gradient_norms),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                stopped_by = "early_stopping"
                break
    model.load_state_dict(best_state)
    return TrainingResult(
        best_state=best_state,
        best_epoch=best_epoch,
        history=tuple(history),
        training_seconds=time.perf_counter() - started,
        optimizer_steps=optimizer_steps,
        stopped_by=stopped_by,
    )


def result_metadata(result: TrainingResult) -> dict[str, Any]:
    payload = asdict(result)
    payload.pop("best_state")
    payload["history"] = list(result.history)
    return payload
