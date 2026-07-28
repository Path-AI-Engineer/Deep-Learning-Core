from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sequence_models.contracts import CHANNELS, CLASS_MAPPING


@dataclass(frozen=True, slots=True)
class SequenceRecord:
    sample_id: str
    values: NDArray[np.float32]
    label: int
    subject_id: int
    split: str = "fixture"

    @property
    def activity(self) -> str:
        return CLASS_MAPPING[self.label]


def _activity_signal(label: int, sample_index: int) -> NDArray[np.float32]:
    rng = np.random.default_rng(10_000 + label * 101 + sample_index)
    t = np.linspace(0, 2.56, 128, endpoint=False, dtype=np.float32)
    dynamic = label < 3
    base_frequency = (1.65, 2.05, 2.45, 0.12, 0.08, 0.04)[label]
    amplitude = (0.95, 1.15, 1.28, 0.08, 0.055, 0.035)[label]
    phase = np.linspace(0, np.pi, len(CHANNELS), dtype=np.float32)
    channels: list[NDArray[np.float32]] = []
    for channel_index in range(len(CHANNELS)):
        harmonic = np.sin(2 * np.pi * base_frequency * t + phase[channel_index])
        secondary = np.cos(
            2 * np.pi * base_frequency * 0.5 * t + phase[channel_index] * 0.35
        )
        orientation = (label - 3) * 0.13 if not dynamic and channel_index >= 6 else 0.0
        noise = rng.normal(0, 0.035 if dynamic else 0.012, size=t.shape)
        channel = (
            amplitude * (1 - channel_index * 0.045) * harmonic
            + amplitude * 0.18 * secondary
            + orientation
            + noise
        )
        channels.append(channel.astype(np.float32))
    return np.stack(channels, axis=1).astype(np.float32)


def build_demo_records(samples_per_class: int = 4) -> list[SequenceRecord]:
    if not 1 <= samples_per_class <= 20:
        raise ValueError("samples_per_class must be between 1 and 20")
    records: list[SequenceRecord] = []
    for label in CLASS_MAPPING:
        for sample_index in range(samples_per_class):
            records.append(
                SequenceRecord(
                    sample_id=f"demo-{label}-{sample_index:02d}",
                    values=_activity_signal(label, sample_index),
                    label=label,
                    subject_id=90 + sample_index,
                )
            )
    return records
