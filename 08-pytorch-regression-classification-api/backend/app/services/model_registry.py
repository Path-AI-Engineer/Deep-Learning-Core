from __future__ import annotations

from pathlib import Path

from pytorch_tabular.inference import ModelPredictor


class ModelRegistry:
    def __init__(self, bundle_root: Path) -> None:
        self.bundle_root = bundle_root
        self._models: dict[str, ModelPredictor] = {}

    def load(self) -> None:
        loaded: dict[str, ModelPredictor] = {}
        for task in ("regression", "classification"):
            bundle = self.bundle_root / task / "v1.0.0"
            if bundle.is_dir():
                loaded[task] = ModelPredictor(bundle)
        self._models = loaded

    def get(self, task: str) -> ModelPredictor:
        try:
            return self._models[task]
        except KeyError as error:
            raise LookupError(f"No approved {task} model is available.") from error

    def readiness(self) -> dict[str, bool]:
        return {task: task in self._models for task in ("regression", "classification")}

    def tasks(self) -> list[dict[str, object]]:
        return [
            {
                "task": task,
                "available": task in self._models,
                "model_version": (
                    self._models[task].metadata["model_version"]
                    if task in self._models
                    else None
                ),
                "dataset": (
                    self._models[task].metadata["dataset"] if task in self._models else None
                ),
            }
            for task in ("regression", "classification")
        ]
