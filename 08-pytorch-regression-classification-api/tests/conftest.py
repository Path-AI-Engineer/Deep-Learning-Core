from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def runtime_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / ".runtime" / "tests"
    directory = root / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    yield directory
    shutil.rmtree(directory, ignore_errors=True)
