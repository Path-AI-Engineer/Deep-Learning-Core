from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core import LabRegistry, get_registry
from sequence_models.contracts import CHANNELS

router = APIRouter(tags=["metadata"])


@router.get("/health")
def health(registry: LabRegistry = Depends(get_registry)) -> dict[str, object]:
    return {
        "status": "ready" if registry.bundles else "degraded",
        "api_version": "v1",
        "data_mode": registry.data_mode,
        "bundles_available": sorted(registry.bundles),
        "active_model": registry.active_model,
        "model_versions": {
            model_id: bundle.version for model_id, bundle in registry.bundles.items()
        },
    }


@router.get("/model-card")
def model_card(registry: LabRegistry = Depends(get_registry)) -> dict[str, object]:
    return {
        "title": "Sequence Memory Lab model card",
        "task": "Many-to-one classification of 128-step multivariate inertial sequences.",
        "official_dataset": {
            "name": "Human Activity Recognition Using Smartphones",
            "source": "UCI Machine Learning Repository",
            "doi": "10.24432/C54S4K",
            "license": "CC BY 4.0",
        },
        "runtime_data_mode": registry.data_mode,
        "signals": list(CHANNELS),
        "classes": registry.classes(),
        "approved_model": registry.active_model,
        "comparison": registry.comparison,
        "domain": "Controlled educational analysis of smartphone inertial windows.",
        "limitations": [
            "Fixture metrics validate software behavior and are not UCI benchmark claims.",
            "Softmax scores are confidence estimates, not certainty.",
            "Hidden states and gates do not establish causal explanations.",
            "The API accepts controlled sample IDs, never arbitrary sensor files or checkpoints.",
        ],
    }


@router.get("/models")
def models(registry: LabRegistry = Depends(get_registry)) -> dict[str, object]:
    return {"models": registry.model_rows(), "active_model": registry.active_model}


@router.get("/classes")
def classes(registry: LabRegistry = Depends(get_registry)) -> dict[str, object]:
    return {"classes": registry.classes()}
