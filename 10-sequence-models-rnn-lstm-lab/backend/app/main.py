from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.v1 import router

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = Path(os.getenv("SEQUENCE_FRONTEND_DIST", PROJECT_ROOT / "frontend/dist"))
if not FRONTEND_DIST.is_absolute():
    FRONTEND_DIST = PROJECT_ROOT / FRONTEND_DIST

app = FastAPI(
    title="Sequence Memory Lab API",
    version="1.0.0",
    description="Bounded inference and educational traces for RNN, LSTM and GRU.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(router)


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, error: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(error), "code": "validation_error"})


if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/{full_path:path}", include_in_schema=False, response_model=None)
def spa_fallback(full_path: str) -> Response:
    if full_path.startswith("api/"):
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    requested = FRONTEND_DIST / full_path
    if full_path and requested.is_file() and FRONTEND_DIST in requested.resolve().parents:
        return FileResponse(requested)
    index = FRONTEND_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Frontend build is not available. Run npm install and npm run build.",
            "api_docs": "/docs",
        },
    )
