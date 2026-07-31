from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.v1 import router
from transformer_lab.inference import ModelRegistry

ROOT = Path(__file__).parents[2]
STATIC_ROOT = ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    current = ModelRegistry()
    current.load()
    application.state.registry = current
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        title="Transformer Architecture Lab API",
        version="1.0.0",
        description=(
            "Manual encoder-decoder Transformer, controlled sequence transduction, "
            "bounded traces and reproducible validation evidence."
        ),
        lifespan=lifespan,
    )
    application.include_router(router)
    assets = STATIC_ROOT / "assets"
    if assets.is_dir():
        application.mount("/assets", StaticFiles(directory=assets), name="assets")

    @application.get("/{path:path}", include_in_schema=False, response_model=None)
    def spa(path: str) -> FileResponse | dict[str, object]:
        index = STATIC_ROOT / "index.html"
        if index.is_file():
            return FileResponse(index)
        return {
            "name": "Transformer Architecture Lab",
            "status": "frontend_not_built",
            "api": "/api/v1/health",
            "docs": "/docs",
        }

    return application


app = create_app()
