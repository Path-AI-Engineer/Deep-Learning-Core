from __future__ import annotations

from collections.abc import Sequence

import torch

from transformer_lab.contracts import SequenceExample
from transformer_lab.tokenization import PAD, Vocabulary


def pad_sequences(rows: Sequence[Sequence[int]], pad_id: int = PAD) -> torch.Tensor:
    if not rows:
        raise ValueError("Cannot pad an empty batch.")
    width = max(len(row) for row in rows)
    output = torch.full((len(rows), width), pad_id, dtype=torch.long)
    for index, row in enumerate(rows):
        output[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return output


def collate_examples(
    examples: Sequence[SequenceExample],
    vocabulary: Vocabulary | None = None,
) -> dict[str, torch.Tensor]:
    vocabulary = vocabulary or Vocabulary()
    source = pad_sequences([vocabulary.encode(example.source_tokens) for example in examples])
    target = pad_sequences([vocabulary.encode(example.target_tokens) for example in examples])
    return {
        "source_ids": source,
        "source_padding_mask": source.eq(PAD),
        "target_input": target[:, :-1],
        "target_output": target[:, 1:],
        "target_padding_mask": target[:, :-1].eq(PAD),
    }

