from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core import LabRegistry, get_registry

router = APIRouter(tags=["metadata"])


@router.get("/health")
def health(registry: LabRegistry = Depends(get_registry)) -> dict[str, object]:
    return {
        "status": "ready",
        "api_version": "v1",
        "data_mode": registry.data_mode,
        "active_model": registry.active_model,
        "bundles_available": sorted(registry.bundles),
    }


@router.get("/model-card")
def model_card(registry: LabRegistry = Depends(get_registry)) -> dict[str, object]:
    return {
        "title": "Latent Representation Lab",
        "task": "Deterministic reconstruction and representation analysis.",
        "official_dataset_target": "FashionMNIST",
        "runtime_data_mode": registry.data_mode,
        "labels_used_for_autoencoder_training": False,
        "active_model": registry.active_model,
        "metrics": registry.comparison,
        "limitations": [
            "Fixture metrics validate software and are not FashionMNIST benchmarks.",
            "Reconstruction quality does not prove representation usefulness.",
            "Linear probe performance is evidence of linear accessibility, not causality.",
            "The deterministic latent space is not a probability distribution.",
            "Upload inference is ephemeral and may be out of domain.",
        ],
    }


@router.get("/models")
def models(registry: LabRegistry = Depends(get_registry)) -> dict[str, object]:
    return {"items": registry.model_rows(), "active_model": registry.active_model}


@router.get("/classes")
def classes(registry: LabRegistry = Depends(get_registry)) -> dict[str, object]:
    return {"items": registry.classes()}
