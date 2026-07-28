from __future__ import annotations

from fastapi import APIRouter

from .routes import evaluation, labs, metadata, predictions, samples, traces

router = APIRouter(prefix="/api/v1")
router.include_router(metadata.router)
router.include_router(samples.router)
router.include_router(predictions.router)
router.include_router(evaluation.router)
router.include_router(traces.router)
router.include_router(labs.router)
