from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "PyTorch Tabular Studio API"
    environment: str = os.getenv("APP_ENV", "development")
    project_root: Path = Path(__file__).resolve().parents[3]
    bundle_root: Path | None = None

    def resolved_bundle_root(self) -> Path:
        return self.bundle_root or self.project_root / "artifacts" / "models"


@lru_cache
def get_settings() -> Settings:
    configured = os.getenv("MODEL_BUNDLE_ROOT")
    return Settings(bundle_root=Path(configured) if configured else None)
