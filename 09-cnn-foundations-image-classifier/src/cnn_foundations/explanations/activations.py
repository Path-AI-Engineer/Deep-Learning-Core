from __future__ import annotations

from typing import Any

import numpy as np
import torch

from cnn_foundations.models.cnn import FashionCNN

ALLOWED_LAYERS = ("conv1", "pool1", "conv2", "pool2")


def _serialize_maps(tensor: torch.Tensor, limit: int) -> list[list[list[float]]]:
    maps = tensor.detach().cpu()[0, :limit]
    output: list[list[list[float]]] = []
    for feature_map in maps:
        minimum = float(feature_map.min())
        maximum = float(feature_map.max())
        normalized = (feature_map - minimum) / (maximum - minimum + 1e-8)
        output.append(np.round(normalized.numpy(), 5).tolist())
    return output


class ActivationInspector:
    def __init__(self, model: FashionCNN) -> None:
        self.model = model

    def filters(self, limit: int = 8) -> dict[str, Any]:
        if limit < 1 or limit > 16:
            raise ValueError("filter limit must be between 1 and 16.")
        weights = self.model.conv1.weight.detach().cpu()[:limit, 0]
        return {
            "layer_id": "conv1",
            "weight_shape": list(self.model.conv1.weight.shape),
            "filters": np.round(weights.numpy(), 6).tolist(),
        }

    def capture(
        self,
        tensor: torch.Tensor,
        *,
        layer_id: str,
        limit: int = 8,
    ) -> dict[str, Any]:
        if layer_id not in ALLOWED_LAYERS:
            raise KeyError(f"layer is not explainable: {layer_id}")
        if limit < 1 or limit > 12:
            raise ValueError("feature-map limit must be between 1 and 12.")
        captured: dict[str, torch.Tensor] = {}

        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured["value"] = output.detach()

        module = getattr(self.model, layer_id)
        handle = module.register_forward_hook(hook)
        try:
            self.model.eval()
            with torch.inference_mode():
                logits = self.model(tensor.cpu())
        finally:
            handle.remove()
        value = captured["value"]
        return {
            "layer_id": layer_id,
            "tensor_shape": list(value.shape),
            "feature_maps": _serialize_maps(value, limit),
            "predicted_index": int(logits.argmax(dim=1)[0]),
            "interpretation_warning": (
                "Feature maps show recorded activations. They do not establish "
                "causality or reveal human-like reasoning."
            ),
        }
