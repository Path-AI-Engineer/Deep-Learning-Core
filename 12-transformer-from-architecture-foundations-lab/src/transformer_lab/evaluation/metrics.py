from __future__ import annotations

from dataclasses import asdict, dataclass

from transformer_lab.tokenization import EOS, PAD


def normalize_sequence(values: list[int] | tuple[int, ...]) -> list[int]:
    normalized: list[int] = []
    for value in values:
        if value == PAD:
            continue
        normalized.append(int(value))
        if value == EOS:
            break
    return normalized


@dataclass(frozen=True)
class SequenceMetrics:
    exact_match: float
    token_accuracy: float
    eos_correct: float
    evaluated_tokens: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_sequence(
    prediction: list[int] | tuple[int, ...],
    target: list[int] | tuple[int, ...],
) -> SequenceMetrics:
    predicted = normalize_sequence(prediction)
    expected = normalize_sequence(target)
    width = max(len(predicted), len(expected), 1)
    correct = sum(
        int(
            index < len(predicted)
            and index < len(expected)
            and predicted[index] == expected[index]
        )
        for index in range(width)
    )
    return SequenceMetrics(
        exact_match=float(predicted == expected),
        token_accuracy=correct / width,
        eos_correct=float(
            bool(predicted)
            and bool(expected)
            and predicted[-1] == expected[-1] == EOS
        ),
        evaluated_tokens=width,
    )
