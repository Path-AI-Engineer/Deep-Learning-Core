from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from autoencoder_lab.contracts import CLASS_MAPPING


@dataclass(frozen=True, slots=True)
class ImageRecord:
    sample_id: str
    image: NDArray[np.float32]
    label: int
    split: str

    @property
    def class_name(self) -> str:
        return CLASS_MAPPING[self.label]


def _template(label: int) -> NDArray[np.float32]:
    image = np.zeros((28, 28), dtype=np.float32)
    yy, xx = np.mgrid[:28, :28]
    if label in (0, 2, 4, 6):
        width = {0: 8, 2: 9, 4: 10, 6: 7}[label]
        body = (yy >= 8) & (yy <= 23) & (np.abs(xx - 14) <= width)
        sleeves = (yy >= 8) & (yy <= 14) & (np.abs(xx - 14) <= width + 4)
        neck = (yy <= 10) & (((xx - 14) ** 2 + (yy - 8) ** 2) < 9)
        image[body | sleeves] = 0.72 + label * 0.02
        image[neck] = 0
    elif label == 1:
        image[(yy >= 5) & (yy <= 24) & (np.abs(xx - 14) <= 7)] = 0.84
        image[(yy >= 14) & (xx >= 13) & (xx <= 14)] = 0
    elif label == 3:
        top = (yy >= 6) & (yy < 15) & (np.abs(xx - 14) <= 5)
        skirt = (yy >= 14) & (yy <= 24) & (np.abs(xx - 14) <= (yy - 10) * 0.65)
        image[top | skirt] = 0.82
    elif label == 8:
        bag = (yy >= 11) & (yy <= 23) & (np.abs(xx - 14) <= 8)
        handle = ((xx - 14) ** 2 + (yy - 12) ** 2 <= 30) & (yy <= 13)
        inner = ((xx - 14) ** 2 + (yy - 12) ** 2 <= 16) & (yy <= 13)
        image[bag | handle] = 0.9
        image[inner] = 0
    else:
        sole = (yy >= 20) & (yy <= 23) & (xx >= 5) & (xx <= 23)
        upper = ((xx - 14) / 9) ** 2 + ((yy - 17) / 5) ** 2 <= 1
        if label == 9:
            upper |= (yy >= 8) & (yy <= 19) & (xx >= 7) & (xx <= 13)
        image[sole | upper] = 0.78 + (label - 5) * 0.03
    return image


def build_fixture_records(samples_per_class: int = 15) -> list[ImageRecord]:
    if not 6 <= samples_per_class <= 30:
        raise ValueError("samples_per_class must be between 6 and 30")
    records: list[ImageRecord] = []
    for label in CLASS_MAPPING:
        base = _template(label)
        for index in range(samples_per_class):
            rng = np.random.default_rng(11_000 + label * 101 + index)
            shifted = np.roll(base, shift=(index % 3 - 1, (index // 3) % 3 - 1), axis=(0, 1))
            image = np.clip(shifted + rng.normal(0, 0.035, shifted.shape), 0, 1)
            split = "train" if index < samples_per_class - 5 else (
                "validation" if index < samples_per_class - 2 else "test"
            )
            records.append(
                ImageRecord(
                    sample_id=f"fixture-{label}-{index:02d}",
                    image=image[None, ...].astype(np.float32),
                    label=label,
                    split=split,
                )
            )
    return records


def stack_records(
    records: list[ImageRecord],
    split: str,
) -> tuple[NDArray[np.float32], NDArray[np.int64], list[str]]:
    selected = [record for record in records if record.split == split]
    return (
        np.stack([record.image for record in selected]).astype(np.float32),
        np.asarray([record.label for record in selected], dtype=np.int64),
        [record.sample_id for record in selected],
    )
