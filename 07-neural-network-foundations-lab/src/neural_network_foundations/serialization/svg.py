"""Dependency-free SVG evidence for two-dimensional decision boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _color(probability: float) -> str:
    red = round(93 + (1.0 - probability) * 66)
    green = round(70 + probability * 35)
    blue = round(116 + probability * 94)
    return f"rgb({red},{green},{blue})"


def render_boundary_svg(
    path: str | Path,
    *,
    title: str,
    boundary: dict[str, Any],
    features: FloatArray,
    targets: FloatArray,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plot_size = 420
    margin = 54
    width = plot_size + 2 * margin
    height = plot_size + 100
    resolution = int(boundary["resolution"])
    cell = plot_size / resolution
    x_axis = np.asarray(boundary["x"], dtype=np.float64)
    y_axis = np.asarray(boundary["y"], dtype=np.float64)
    probabilities = np.asarray(boundary["probabilities"], dtype=np.float64)
    points = np.asarray(features, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.float64).reshape(-1)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{title}">',
        '<rect width="100%" height="100%" rx="18" fill="#0b0f18"/>',
        f'<text x="{margin}" y="34" fill="#f4f5fb" font-family="Segoe UI" '
        f'font-size="18" font-weight="600">{title}</text>',
        f'<g transform="translate({margin} 55)">',
    ]
    for row in range(resolution):
        for column in range(resolution):
            x = column * cell
            y = (resolution - row - 1) * cell
            elements.append(
                f'<rect x="{x:.3f}" y="{y:.3f}" width="{cell + 0.2:.3f}" '
                f'height="{cell + 0.2:.3f}" fill="{_color(float(probabilities[row, column]))}"/>'
            )
    for point, label in zip(points, labels, strict=True):
        x = (point[0] - x_axis.min()) / (x_axis.max() - x_axis.min()) * plot_size
        y = plot_size - (point[1] - y_axis.min()) / (y_axis.max() - y_axis.min()) * plot_size
        fill = "#54d5ff" if label == 1.0 else "#ff9f66"
        elements.append(
            f'<circle cx="{x:.3f}" cy="{y:.3f}" r="7" fill="{fill}" '
            'stroke="#ffffff" stroke-width="2"/>'
        )
    elements.extend(
        [
            f'<rect x="0" y="0" width="{plot_size}" height="{plot_size}" '
            'fill="none" stroke="#343d52" stroke-width="1"/>',
            "</g>",
            '<text x="54" y="500" fill="#8e96a9" font-family="Segoe UI" font-size="12">'
            "Warm = class 0 · Blue-violet = class 1</text>",
            "</svg>",
        ]
    )
    destination.write_text("\n".join(elements), encoding="utf-8")
    return destination
