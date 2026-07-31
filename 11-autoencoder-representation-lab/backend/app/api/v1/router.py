from __future__ import annotations

from fastapi import APIRouter

from .routes import evaluation, latent, metadata, reconstruction, samples

router = APIRouter(prefix="/api/v1")
router.include_router(metadata.router)
router.include_router(samples.router)
router.include_router(reconstruction.router)
router.include_router(latent.router)
router.include_router(evaluation.router)
