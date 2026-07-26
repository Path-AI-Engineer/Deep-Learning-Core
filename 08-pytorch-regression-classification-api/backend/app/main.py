from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1 import router
from app.core import get_settings
from app.services import ModelRegistry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    model_registry = ModelRegistry(settings.resolved_bundle_root())
    model_registry.load()
    app.state.registry = model_registry
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Versioned inference API for two approved PyTorch tabular models.",
        lifespan=lifespan,
    )
    app.include_router(router)
    frontend = settings.project_root / "frontend" / "dist"
    assets = frontend / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str, request: Request):
        if path.startswith("api/"):
            return JSONResponse(
                status_code=404,
                content={"detail": "API route not found."},
            )
        index = frontend / "index.html"
        if index.is_file():
            return FileResponse(index)
        return {
            "message": "PyTorch Tabular Studio API",
            "docs": str(request.base_url).rstrip("/") + "/docs",
        }

    return app


app = create_app()
