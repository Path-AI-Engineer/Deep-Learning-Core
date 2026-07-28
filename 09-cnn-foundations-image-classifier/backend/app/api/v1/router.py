from __future__ import annotations

import uuid
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as functional
from fastapi import APIRouter, HTTPException, Query, Request

from app.core.registry import ModelRegistry
from app.schemas.contracts import (
    ActivationRequest,
    ConvolutionRequest,
    SamplePredictionRequest,
)
from app.services.image_validation import extract_image
from cnn_foundations.contracts.config import CLASS_NAMES
from cnn_foundations.operations.manual_convolution import cross_correlate_2d

router = APIRouter(prefix="/api/v1")


def registry(request: Request) -> ModelRegistry:
    return cast(ModelRegistry, request.app.state.registry)


def request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", uuid.uuid4()))


def unavailable(error: RuntimeError) -> HTTPException:
    return HTTPException(status_code=503, detail=str(error))


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    current = registry(request)
    version = (
        str(current.bundle.metadata.get("model_version"))
        if current.bundle is not None
        else None
    )
    return {
        "status": "ready" if current.ready else "degraded",
        "api_version": "1.0.0",
        "model_available": current.predictor is not None,
        "gallery_available": current.gallery is not None,
        "model_version": version,
        "training_mode": False,
    }


@router.get("/classes")
def classes() -> dict[str, Any]:
    descriptions = (
        "Casual upper-body garment.",
        "Full-length leg garment.",
        "Knitted upper-body garment.",
        "One-piece garment.",
        "Outerwear garment.",
        "Open warm-weather footwear.",
        "Collared or casual shirt.",
        "Athletic closed footwear.",
        "Hand-carried accessory.",
        "Ankle-height boot.",
    )
    return {
        "classes": [
            {"index": index, "name": name, "description": descriptions[index]}
            for index, name in enumerate(CLASS_NAMES)
        ]
    }


@router.get("/model-card")
def model_card(request: Request) -> dict[str, Any]:
    try:
        bundle = registry(request).require_bundle()
    except RuntimeError as error:
        raise unavailable(error) from error
    return {
        "dataset": "FashionMNIST",
        "classes": list(CLASS_NAMES),
        "architecture": bundle.files["model_config.json"],
        "input": {"shape": [1, 1, 28, 28], "dtype": "float32"},
        "metrics": bundle.files["metrics.json"],
        "baseline": bundle.files["comparison_with_mlp.json"],
        "limitations": bundle.metadata.get("limitations", []),
        "domain": (
            "FashionMNIST grayscale catalog images. It is not a general "
            "photograph or garment-quality classifier."
        ),
    }


@router.get("/samples")
def samples(
    request: Request,
    class_index: int | None = Query(default=None, ge=0, le=9),
    limit: int = Query(default=20, ge=1, le=40),
    offset: int = Query(default=0, ge=0, le=9999),
) -> dict[str, Any]:
    try:
        rows = registry(request).require_gallery().list(
            class_index=class_index,
            limit=limit,
            offset=offset,
        )
    except RuntimeError as error:
        raise unavailable(error) from error
    return {"items": [row.as_dict() for row in rows], "limit": limit, "offset": offset}


@router.get("/samples/{sample_id}")
def sample_detail(sample_id: str, request: Request) -> dict[str, Any]:
    try:
        sample, _ = registry(request).require_gallery().get(sample_id)
    except RuntimeError as error:
        raise unavailable(error) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail="sample not found.") from error
    return sample.as_dict()


def _prediction_payload(
    prediction: Any,
    *,
    request: Request,
    true_class: dict[str, Any] | None,
    preprocessing: Any,
    warning: str | None,
) -> dict[str, Any]:
    return {
        "predicted_index": prediction.predicted_index,
        "predicted_class": prediction.predicted_class,
        "true_class": true_class,
        "probabilities": list(prediction.probabilities),
        "top_k": list(prediction.top_k),
        "model_version": prediction.model_version,
        "inference_time_ms": round(prediction.inference_time_ms, 3),
        "preprocessed_preview": preprocessing.preview_data_url,
        "preprocessing_summary": list(preprocessing.preprocessing_summary),
        "warnings": [warning] if warning else [],
        "request_id": request_id(request),
    }


@router.post("/predictions/sample")
def predict_sample(
    payload: SamplePredictionRequest,
    request: Request,
) -> dict[str, Any]:
    try:
        sample, processed = registry(request).process_sample(payload.sample_id)
        prediction = registry(request).require_predictor().predict(processed.tensor)
    except RuntimeError as error:
        raise unavailable(error) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail="sample not found.") from error
    return _prediction_payload(
        prediction,
        request=request,
        true_class={
            "index": sample["label_index"],
            "name": sample["class_name"],
        },
        preprocessing=processed,
        warning=None,
    )


@router.post("/predictions/upload")
async def predict_upload(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    body = await request.body()
    try:
        content, mime_type = extract_image(content_type, body)
        processed = registry(request).preprocessing(content, mime_type)
        prediction = registry(request).require_predictor().predict(processed.tensor)
    except OverflowError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except TypeError as error:
        raise HTTPException(status_code=415, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise unavailable(error) from error
    return _prediction_payload(
        prediction,
        request=request,
        true_class=None,
        preprocessing=processed,
        warning=(
            "Uploaded photographs are outside the FashionMNIST training domain. "
            "Treat this output as an educational model response, not certainty."
        ),
    )


@router.post("/labs/convolution")
def convolution_lab(payload: ConvolutionRequest) -> dict[str, Any]:
    matrix = np.asarray(payload.matrix, dtype=np.float64)
    kernel = np.asarray(payload.kernel, dtype=np.float64)
    try:
        output, trace = cross_correlate_2d(
            matrix,
            kernel,
            stride=payload.stride,
            padding=payload.padding,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    source = torch.tensor(matrix, dtype=torch.float64)[None, None]
    weights = torch.tensor(kernel, dtype=torch.float64)[None, None]
    torch_output = functional.conv2d(
        source,
        weights,
        stride=payload.stride,
        padding=payload.padding,
    )[0, 0].numpy()
    difference = float(np.max(np.abs(output - torch_output)))
    return {
        "input": matrix.tolist(),
        "kernel": kernel.tolist(),
        "output": output.tolist(),
        "output_shape": list(output.shape),
        "operation_trace": trace,
        "parity_result": {
            "passed": difference <= 1e-10,
            "max_absolute_error": difference,
            "operation": (
                "PyTorch Conv2d performs cross-correlation; the kernel is not flipped."
            ),
        },
    }


@router.get("/explanations/filters")
def filters(request: Request, limit: int = Query(default=8, ge=1, le=16)) -> dict[str, Any]:
    current = registry(request)
    if current.inspector is None:
        raise unavailable(RuntimeError("The approved CNN bundle is not available."))
    return current.inspector.filters(limit)


@router.post("/explanations/activations")
def activations(payload: ActivationRequest, request: Request) -> dict[str, Any]:
    current = registry(request)
    if current.inspector is None:
        raise unavailable(RuntimeError("The approved CNN bundle is not available."))
    try:
        sample, processed = current.process_sample(payload.sample_id)
        result = current.inspector.capture(
            processed.tensor,
            layer_id=payload.layer_id,
            limit=payload.limit,
        )
    except RuntimeError as error:
        raise unavailable(error) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {
        **result,
        "sample": sample,
        "request_id": request_id(request),
    }


@router.get("/evaluation/summary")
def evaluation_summary(request: Request) -> dict[str, Any]:
    try:
        bundle = registry(request).require_bundle()
    except RuntimeError as error:
        raise unavailable(error) from error
    return {
        "metrics": bundle.files["metrics.json"],
        "per_class_metrics": bundle.files["per_class_metrics.json"],
        "confusion_matrix": bundle.files["confusion_matrix.json"],
        "training_history": bundle.files["training_history.json"],
        "comparison": bundle.files["comparison_with_mlp.json"],
        "limitations": bundle.metadata.get("limitations", []),
    }


@router.get("/evaluation/errors")
def evaluation_errors(
    request: Request,
    true_class: int | None = Query(default=None, ge=0, le=9),
    predicted_class: int | None = Query(default=None, ge=0, le=9),
    limit: int = Query(default=20, ge=1, le=40),
) -> dict[str, Any]:
    try:
        bundle = registry(request).require_bundle()
    except RuntimeError as error:
        raise unavailable(error) from error
    records = list(bundle.files["error_analysis.json"].get("errors", []))
    if true_class is not None:
        records = [row for row in records if row["true_index"] == true_class]
    if predicted_class is not None:
        records = [
            row for row in records if row["predicted_index"] == predicted_class
        ]
    return {"items": records[:limit], "limit": limit}
