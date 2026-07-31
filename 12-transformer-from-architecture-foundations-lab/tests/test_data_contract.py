from __future__ import annotations

import pytest

from transformer_lab.data import (
    assert_disjoint,
    collate_examples,
    generate_balanced_suite,
    generate_example,
    oracle,
)
from transformer_lab.tokenization import BOS, EOS, PAD, Vocabulary


def test_fixed_vocabulary_contract_round_trips() -> None:
    vocabulary = Vocabulary()
    tokens = ["BOS", "COPY", "SYMBOL_00", "SYMBOL_31", "EOS"]
    assert vocabulary.decode(vocabulary.encode(tokens)) == tokens
    assert (PAD, BOS, EOS) == (0, 1, 2)
    with pytest.raises(ValueError, match="Unknown token"):
        vocabulary.encode(["NOT_A_TOKEN"])


@pytest.mark.parametrize(
    ("task", "content", "expected"),
    [
        ("copy", ["SYMBOL_01", "SYMBOL_02"], ["SYMBOL_01", "SYMBOL_02"]),
        ("reverse", ["SYMBOL_01", "SYMBOL_02"], ["SYMBOL_02", "SYMBOL_01"]),
        (
            "recall",
            ["SYMBOL_01", "SYMBOL_17", "SYMBOL_02", "SYMBOL_18", "SEP", "SYMBOL_02"],
            ["SYMBOL_18"],
        ),
    ],
)
def test_task_oracles(task: str, content: list[str], expected: list[str]) -> None:
    assert oracle(task, content) == expected  # type: ignore[arg-type]


def test_generation_is_deterministic_and_splits_are_disjoint() -> None:
    first = generate_example("reverse", seed=48, length=8)
    second = generate_example("reverse", seed=48, length=8)
    assert first == second
    training = generate_balanced_suite(
        seed=100,
        count_per_task=4,
        split="train",
        copy_reverse_range=(5, 7),
        recall_range=(2, 3),
    )
    validation = generate_balanced_suite(
        seed=101,
        count_per_task=4,
        split="validation",
        copy_reverse_range=(5, 7),
        recall_range=(2, 3),
    )
    assert_disjoint(training, validation)


def test_collation_shifts_target_and_marks_padding() -> None:
    examples = [
        generate_example("copy", seed=1, length=3),
        generate_example("reverse", seed=2, length=5),
    ]
    batch = collate_examples(examples)
    assert batch["source_ids"].shape[0] == 2
    assert batch["target_input"][0, 0].item() == BOS
    assert batch["target_output"][0, 3].item() == EOS
    assert str(batch["target_padding_mask"].dtype) == "torch.bool"
