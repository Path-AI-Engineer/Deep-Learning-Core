from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from app.api.v1.router import router
from app.core.registry import ModelRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


def create_app(
    *,
    bundle_path: Path | None = None,
    data_root: Path | None = None,
) -> FastAPI:
    registry = ModelRegistry(bundle_path=bundle_path, data_root=data_root)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        registry.load()
        app.state.registry = registry
        yield

    application = FastAPI(
        title="CNN Vision Lab API",
        version="1.0.0",
        description=(
            "FashionMNIST CNN classification, controlled activations, "
            "evaluation evidence and an educational cross-correlation lab."
        ),
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    application.include_router(router)

    @application.exception_handler(Exception)
    async def unhandled(_request: Request, _error: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected internal error occurred."},
        )

    if FRONTEND_DIST.exists():
        assets = FRONTEND_DIST / "assets"
        if assets.exists():
            application.mount("/assets", StaticFiles(directory=assets), name="assets")

        @application.get(
            "/{full_path:path}",
            include_in_schema=False,
            response_model=None,
        )
        async def spa(full_path: str) -> FileResponse | JSONResponse:
            if full_path.startswith("api/"):
                return JSONResponse(status_code=404, content={"detail": "not found"})
            requested = FRONTEND_DIST / full_path
            if requested.is_file() and FRONTEND_DIST in requested.parents:
                return FileResponse(requested)
            return FileResponse(FRONTEND_DIST / "index.html")
    else:

        @application.get("/", include_in_schema=False)
        async def frontend_not_built() -> JSONResponse:
            return JSONResponse(
                status_code=200,
                content={
                    "product": "CNN Vision Lab",
                    "frontend": "not built",
                    "docs": "/docs",
                    "health": "/api/v1/health",
                },
            )
    return application


app = create_app()
