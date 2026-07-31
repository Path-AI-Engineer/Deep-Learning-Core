from __future__ import annotations

import math
from typing import Any, cast

import torch
from fastapi import APIRouter, HTTPException, Query, Request

from backend.app.schemas import (
    AttentionComputeRequest,
    PredictRequest,
    TraceRequest,
)
from transformer_lab.attention import scaled_dot_product_attention
from transformer_lab.contracts import SequenceExample, TaskId
from transformer_lab.inference import ModelRegistry
from transformer_lab.tokenization import Vocabulary

router = APIRouter(prefix="/api/v1")


def registry(request: Request) -> ModelRegistry:
    return cast(ModelRegistry, request.app.state.registry)


def _bundle_files(request: Request) -> dict[str, Any]:
    return registry(request).require_bundle().files


def _example(current: ModelRegistry, payload: PredictRequest) -> SequenceExample:
    try:
        if payload.sample_id is not None:
            sample = current.sample(payload.sample_id)
            if sample.task != payload.task:
                raise ValueError("Selected sample does not belong to the requested task.")
            return sample
        return current.custom_example(payload.task, payload.source_symbols or [])
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    current = registry(request)
    bundle = current.bundle
    return {
        "status": "ready" if current.ready else "degraded",
        "api_version": "1.0.0",
        "model_registry_status": "ready" if bundle else "unavailable",
        "active_model": None if bundle is None else bundle.manifest["model_version"],
        "experiment_registry_status": (
            "reference_validation" if bundle else "unavailable"
        ),
        "detail": current.error,
    }


@router.get("/model-card")
def model_card(request: Request) -> dict[str, Any]:
    bundle = registry(request).require_bundle()
    return {
        "purpose": "Inspect a small encoder-decoder Transformer built from primitives.",
        "architecture": bundle.files["model_config.json"],
        "tasks": ["copy", "reverse", "recall"],
        "data": bundle.files["data_manifest.json"],
        "metrics": bundle.files["metrics.json"],
        "limitations": bundle.manifest["limitations"],
        "version": bundle.manifest["model_version"],
    }


@router.get("/architecture")
def architecture(request: Request) -> dict[str, Any]:
    config = _bundle_files(request)["model_config.json"]
    return {
        "components": [
            {"id": "source_embedding", "responsibility": "Map source token IDs to vectors."},
            {"id": "position", "responsibility": "Inject sequence order."},
            {"id": "encoder_self_attention", "responsibility": "Mix source context."},
            {"id": "decoder_masked_attention", "responsibility": "Block future targets."},
            {"id": "cross_attention", "responsibility": "Query encoder memory."},
            {"id": "feed_forward", "responsibility": "Transform each position independently."},
            {"id": "projection", "responsibility": "Produce vocabulary logits."},
        ],
        "shapes": {
            "sequence": "[B, T, D]",
            "heads": "[B, H, T, D_k]",
            "scores": "[B, H, T_query, T_key]",
            "logits": "[B, T_target, vocabulary_size]",
        },
        "active_config": config,
        "normalization": config["normalization"],
        "positional_encoding": config["positional_encoding"],
        "trace_limits": {"layers": 2, "heads": 4, "batch": 1, "top_k": 5},
    }


@router.get("/tasks")
def tasks() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "copy",
            "name": "Copy",
            "description": "Reproduce the input symbol sequence.",
            "rules": "COPY + symbols + EOS",
            "trained_lengths": [5, 16],
            "evaluated_ood_lengths": [17, 20],
        },
        {
            "task_id": "reverse",
            "name": "Reverse",
            "description": "Generate symbols in reverse order.",
            "rules": "REVERSE + symbols + EOS",
            "trained_lengths": [5, 16],
            "evaluated_ood_lengths": [17, 20],
        },
        {
            "task_id": "recall",
            "name": "Associative recall",
            "description": "Retrieve a value for a present key.",
            "rules": "RECALL + key/value pairs + SEP + query + EOS",
            "trained_lengths": [2, 6],
            "evaluated_ood_lengths": [7, 8],
        },
    ]


@router.get("/tokens")
def tokens() -> dict[str, Any]:
    return {
        **Vocabulary().as_dict(),
        "input_rules": {
            "content": "SYMBOL_00 through SYMBOL_31",
            "recall_separator": "SEP",
            "unknown_ids": "rejected",
        },
    }


@router.get("/models")
def models(request: Request) -> list[dict[str, Any]]:
    bundle = registry(request).require_bundle()
    return [
        {
            "model_id": "transformer-v1.0.0",
            "family": "encoder-decoder-transformer",
            "version": bundle.manifest["model_version"],
            "config": bundle.files["model_config.json"],
            "capabilities": ["predict", "trace", "attention"],
            "metrics": bundle.files["metrics.json"],
            "status": bundle.manifest["evidence_status"],
        }
    ]


@router.get("/samples")
def samples(
    request: Request,
    task: TaskId | None = None,
    split: str | None = None,
    limit: int = Query(default=12, ge=1, le=24),
) -> dict[str, Any]:
    rows = registry(request).samples
    if task is not None:
        rows = [row for row in rows if row.task == task]
    if split is not None:
        rows = [row for row in rows if row.split == split]
    return {
        "items": [row.as_dict() for row in rows[:limit]],
        "count": min(len(rows), limit),
        "total": len(rows),
    }


@router.get("/samples/{sample_id}")
def sample_detail(request: Request, sample_id: str) -> dict[str, Any]:
    try:
        return registry(request).sample(sample_id).as_dict()
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/predict")
def predict(request: Request, payload: PredictRequest) -> dict[str, Any]:
    current = registry(request)
    example = _example(current, payload)
    try:
        result, metrics, latency = current.predict(
            example,
            max_new_tokens=payload.max_new_tokens,
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    ood_limit = 6 if example.task == "recall" else 16
    is_ood = example.content_length > ood_limit
    return {
        "request_id": f"req-{example.canonical_hash[:12]}",
        "task": example.task,
        "normalized_source": list(example.source_tokens),
        "prediction": current.prediction_tokens(result),
        "target": list(example.target_tokens[1:]),
        **metrics,
        "eos_status": result.stopped_by,
        "decoding_steps": list(result.steps),
        "latency_ms": latency,
        "model_version": current.require_bundle().manifest["model_version"],
        "length_regime": "ood_length" if is_ood else "in_distribution",
        "warning": (
            "Length exceeds the training range; this is controlled OOD evaluation."
            if is_ood
            else None
        ),
    }


@router.post("/trace")
def trace(request: Request, payload: TraceRequest) -> dict[str, Any]:
    current = registry(request)
    example = _example(current, payload)
    try:
        result, _, _ = current.predict(
            example,
            max_new_tokens=payload.max_new_tokens,
            trace=True,
        )
        matrix = current.trace_matrix(
            result,
            trace_type=payload.trace_type,
            layer=payload.layer,
            head=payload.head,
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    prediction = current.prediction_tokens(result)
    source_axis = list(example.source_tokens)
    target_axis = ["BOS", *prediction[:-1]]
    if payload.trace_type == "encoder_self":
        query_axis, key_axis = source_axis, source_axis
    elif payload.trace_type == "decoder_self":
        query_axis, key_axis = target_axis, target_axis
    else:
        query_axis, key_axis = target_axis, source_axis
    return {
        "schema_version": "1.0.0",
        "task": example.task,
        "trace_type": payload.trace_type,
        "layer": payload.layer,
        "head": payload.head,
        "query_tokens": query_axis[-len(matrix) :],
        "key_tokens": key_axis[-len(matrix[0]) :] if matrix else [],
        "weights": matrix,
        "shape": [len(matrix), len(matrix[0]) if matrix else 0],
        "entropy": [
            -sum(value * math.log(value + 1e-12) for value in row)
            for row in matrix
        ],
        "warning": "Attention weights are descriptive, not causal explanations.",
    }


@router.post("/attention/compute")
def attention_compute(payload: AttentionComputeRequest) -> dict[str, Any]:
    query = torch.tensor(payload.query, dtype=torch.float64)[None, None, :, :]
    key = torch.tensor(payload.key, dtype=torch.float64)[None, None, :, :]
    value = torch.tensor(payload.value, dtype=torch.float64)[None, None, :, :]
    mask = (
        None
        if payload.mask is None
        else torch.tensor(payload.mask, dtype=torch.bool)[None, None, :, :]
    )
    output, weights, scores = scaled_dot_product_attention(
        query,
        key,
        value,
        mask=mask,
    )
    raw_scores = query @ key.transpose(-2, -1)
    reference = torch.softmax(scores, dim=-1) @ value
    return {
        "raw_scores": raw_scores[0, 0].tolist(),
        "scale": math.sqrt(query.shape[-1]),
        "scaled_scores": (raw_scores / math.sqrt(query.shape[-1]))[0, 0].tolist(),
        "masked_scores": scores[0, 0].tolist(),
        "weights": weights[0, 0].tolist(),
        "output": output[0, 0].tolist(),
        "reference_difference": float((output - reference).abs().max()),
        "shapes": {
            "query": list(query.shape),
            "key": list(key.shape),
            "value": list(value.shape),
            "weights": list(weights.shape),
        },
    }


@router.get("/experiments")
def experiments(request: Request) -> list[dict[str, Any]]:
    bundle = registry(request).require_bundle()
    return [
        {
            "experiment_id": "reference-validation",
            "family": "transformer-reference",
            "status": bundle.manifest["evidence_status"],
            "runs": 1,
            "seeds": [bundle.manifest["seed"]],
            "frozen": False,
            "summary": bundle.files["metrics.json"],
        }
    ]


@router.get("/experiments/{experiment_id}")
def experiment_detail(request: Request, experiment_id: str) -> dict[str, Any]:
    if experiment_id != "reference-validation":
        raise HTTPException(status_code=404, detail="Unknown experiment ID.")
    bundle = registry(request).require_bundle()
    return {
        "experiment_id": experiment_id,
        "config": bundle.files["model_config.json"],
        "included_runs": [{"seed": bundle.manifest["seed"], "status": "completed"}],
        "excluded_runs": [],
        "aggregate_metrics": bundle.files["metrics.json"],
        "figures": ["reference-validation"],
        "limitations": bundle.manifest["limitations"],
    }


@router.get("/evaluation/summary")
def evaluation_summary(request: Request) -> dict[str, Any]:
    files = _bundle_files(request)
    return {
        "models": ["transformer-v1.0.0"],
        "id": files["metrics.json"]["validation_id"],
        "ood": files["metrics.json"]["validation_ood"],
        "generalization_gap": files["metrics.json"]["generalization_gap_exact_match"],
        "per_task": files["per_task_metrics.json"],
        "cost": {
            "latency": files["latency.json"],
            "training": files["training_history.json"],
        },
        "selected_bundle": "transformer-v1.0.0",
        "decision": "Reference validation bundle; frozen test remains unopened.",
    }


@router.get("/evaluation/by-length")
def evaluation_by_length(request: Request) -> dict[str, Any]:
    return cast(dict[str, Any], _bundle_files(request)["per_length_metrics.json"])


@router.get("/evaluation/errors")
def evaluation_errors(
    request: Request,
    task: TaskId | None = None,
    limit: int = Query(default=12, ge=1, le=50),
) -> dict[str, Any]:
    path = (
        registry(request).bundle_path.parents[3]
        / "reports/predictions/reference-validation.json"
    )
    if not path.is_file():
        return {"items": [], "count": 0}
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    rows = [
        row
        for split_rows in payload.values()
        for row in split_rows
        if not row["exact_match"] and (task is None or row["task"] == task)
    ][:limit]
    return {"items": rows, "count": len(rows)}


@router.get("/research")
def research(request: Request) -> dict[str, Any]:
    bundle = registry(request).require_bundle()
    return {
        "research_questions": [
            "Can a manual Transformer learn copy, reverse and associative recall?",
            "How does validation performance change at unseen lengths?",
            "What changes with positional signal, head count and LayerNorm placement?",
        ],
        "hypotheses_status": "preregistered_in_protocol",
        "protocol": (
            "Validation-only reference run; final multi-seed matrix and test are frozen."
        ),
        "results_status": bundle.manifest["evidence_status"],
        "threats": [
            "Synthetic tasks do not represent language.",
            "One reference seed cannot establish training stability.",
            "Attention weights are not causal explanations.",
        ],
        "paper_assets": [
            "paper/data/reference-validation.json",
            "paper/tables/reference-validation.md",
        ],
    }
