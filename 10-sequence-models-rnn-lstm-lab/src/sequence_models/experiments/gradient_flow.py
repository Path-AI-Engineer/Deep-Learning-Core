from __future__ import annotations

import math

import torch
from torch import Tensor


def permute_timesteps(inputs: Tensor, seed: int = 42) -> Tensor:
    if inputs.ndim != 3:
        raise ValueError("inputs must have shape [batch, time, features]")
    generator = torch.Generator(device=inputs.device).manual_seed(seed)
    order = torch.randperm(inputs.shape[1], generator=generator, device=inputs.device)
    return inputs[:, order, :]


def _single_gradient(length: int, recurrent_scale: float, clipping: bool) -> dict[str, float]:
    hidden = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
    initial = hidden
    for timestep in range(length):
        value = 0.03 if timestep == 0 else 0.0
        hidden = torch.tanh(recurrent_scale * hidden + value)
    loss = hidden.square().sum()
    loss.backward()  # type: ignore[no-untyped-call]
    before = abs(float(initial.grad.item())) if initial.grad is not None else 0.0
    after = min(before, 1.0) if clipping else before
    return {
        "length": float(length),
        "gradient_norm_before": before,
        "gradient_norm_after": after,
    }


def gradient_flow_experiment() -> dict[str, object]:
    lengths = (4, 8, 16, 32, 64, 96)
    scenarios = {
        "vanishing": 0.55,
        "stable": 1.0,
        "growing": 1.45,
    }
    results: list[dict[str, object]] = []
    all_finite = True
    for name, scale in scenarios.items():
        points = [_single_gradient(length, scale, clipping=name == "growing") for length in lengths]
        all_finite = all_finite and all(
            math.isfinite(point["gradient_norm_before"]) for point in points
        )
        results.append(
            {
                "scenario": name,
                "recurrent_scale": scale,
                "clipping_threshold": 1.0 if name == "growing" else None,
                "points": points,
            }
        )
    return {
        "experiment": "delayed-dependency-gradient-flow",
        "reproducible": True,
        "scenarios": results,
        "interpretation": (
            "Repeated Jacobian products can shrink or amplify gradients. Clipping caps large "
            "norms after backward, but it does not recover vanished information."
        ),
        "finite": all_finite,
    }
