from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ReceptiveFieldStep:
    layer: str
    receptive_field: int
    jump: int


def receptive_field_trace(
    layers: tuple[tuple[str, int, int], ...] = (
        ("conv1", 3, 1),
        ("pool1", 2, 2),
        ("conv2", 3, 1),
        ("pool2", 2, 2),
    ),
) -> list[dict[str, int | str]]:
    receptive_field = 1
    jump = 1
    steps: list[dict[str, int | str]] = []
    for name, kernel, stride in layers:
        if kernel <= 0 or stride <= 0:
            raise ValueError("kernel and stride must be positive.")
        receptive_field += (kernel - 1) * jump
        jump *= stride
        steps.append(asdict(ReceptiveFieldStep(name, receptive_field, jump)))
    return steps

