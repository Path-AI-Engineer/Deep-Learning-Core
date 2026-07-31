from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from app.core import LabRegistry, get_registry
from app.schemas import DenoiseRequest, ReconstructRequest
from autoencoder_lab.inference import decode_upload, reconstruct

router = APIRouter(tags=["reconstruction"])


@router.post("/reconstruct/sample")
def reconstruct_sample(
    request: ReconstructRequest,
    registry: LabRegistry = Depends(get_registry),
) -> dict[str, object]:
    return registry.reconstruct(request.sample_id, request.model_id)


@router.post("/reconstruct/upload")
async def reconstruct_upload(
    request: Request,
    model_id: str = Query(min_length=1, max_length=32),
    registry: LabRegistry = Depends(get_registry),
) -> dict[str, object]:
    if request.headers.get("content-type") not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError("upload must be PNG, JPEG or WebP")
    if model_id not in registry.bundles:
        raise ValueError("upload reconstruction requires an autoencoder model")
    content = await request.body()
    image = decode_upload(content)
    result = reconstruct(registry.bundles[model_id].model, image)
    return {
        **result,
        "model_id": model_id,
        "model_version": registry.version,
        "persisted": False,
        "domain_warning": "Uploaded images may differ from FashionMNIST-like inputs.",
    }


@router.post("/denoise")
@router.post("/denoise/sample", include_in_schema=False)
def denoise_sample(
    request: DenoiseRequest,
    registry: LabRegistry = Depends(get_registry),
) -> dict[str, object]:
    return registry.denoise(
        request.sample_id,
        request.corruption_type,
        request.corruption_level,
        request.seed,
        request.model_ids,
    )
