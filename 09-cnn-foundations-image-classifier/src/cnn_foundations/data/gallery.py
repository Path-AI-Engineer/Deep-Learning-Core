from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torchvision.datasets import FashionMNIST

from cnn_foundations.contracts.config import CLASS_NAMES


@dataclass(frozen=True)
class GallerySample:
    sample_id: str
    label_index: int
    class_name: str
    image_data_url: str
    split: str = "test"

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "label_index": self.label_index,
            "class_name": self.class_name,
            "image_data_url": self.image_data_url,
            "split": self.split,
        }


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class SampleGallery:
    def __init__(self, root: Path) -> None:
        try:
            self.dataset = FashionMNIST(root=root, train=False, download=False)
        except RuntimeError as error:
            raise RuntimeError(
                "FashionMNIST test samples are unavailable; run prepare_data.py."
            ) from error

    def get(self, sample_id: str) -> tuple[GallerySample, Image.Image]:
        if not sample_id.startswith("test-"):
            raise KeyError(sample_id)
        try:
            index = int(sample_id.removeprefix("test-"))
        except ValueError as error:
            raise KeyError(sample_id) from error
        if index < 0 or index >= len(self.dataset):
            raise KeyError(sample_id)
        image, label = self.dataset[index]
        sample = GallerySample(
            sample_id=f"test-{index:05d}",
            label_index=int(label),
            class_name=CLASS_NAMES[int(label)],
            image_data_url=_data_url(image),
        )
        return sample, image

    def list(
        self,
        *,
        class_index: int | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[GallerySample]:
        if limit < 1 or limit > 40:
            raise ValueError("sample limit must be between 1 and 40.")
        if class_index is not None and class_index not in range(10):
            raise ValueError("class_index must be between 0 and 9.")
        results: list[GallerySample] = []
        for index in range(offset, len(self.dataset)):
            image, label = self.dataset[index]
            if class_index is not None and int(label) != class_index:
                continue
            results.append(
                GallerySample(
                    sample_id=f"test-{index:05d}",
                    label_index=int(label),
                    class_name=CLASS_NAMES[int(label)],
                    image_data_url=_data_url(image),
                )
            )
            if len(results) >= limit:
                break
        return results

