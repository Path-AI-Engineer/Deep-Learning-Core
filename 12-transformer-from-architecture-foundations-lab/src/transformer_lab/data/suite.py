from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import replace

from transformer_lab.contracts import SequenceExample, TaskId
from transformer_lab.tokenization import SYMBOLS

TASK_TOKEN: dict[TaskId, str] = {
    "copy": "COPY",
    "reverse": "REVERSE",
    "recall": "RECALL",
}


def canonical_hash(task: TaskId, source: Iterable[str], target: Iterable[str]) -> str:
    payload = {
        "task": task,
        "source": list(source),
        "target": list(target),
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def oracle(task: TaskId, source_content: list[str]) -> list[str]:
    if task == "copy":
        return source_content.copy()
    if task == "reverse":
        return list(reversed(source_content))
    if "SEP" not in source_content:
        raise ValueError("Associative recall requires SEP between memory and query.")
    separator = source_content.index("SEP")
    memory, query = source_content[:separator], source_content[separator + 1 :]
    if len(memory) < 4 or len(memory) % 2 or len(query) != 1:
        raise ValueError("Recall requires key/value pairs, SEP and exactly one query.")
    pairs = dict(zip(memory[::2], memory[1::2], strict=True))
    if len(pairs) != len(memory) // 2:
        raise ValueError("Recall keys must be unique.")
    if query[0] not in pairs:
        raise ValueError("Recall query must reference a present key.")
    return [pairs[query[0]]]


def generate_example(
    task: TaskId,
    *,
    seed: int,
    split: str = "demo",
    length: int = 6,
) -> SequenceExample:
    rng = random.Random(seed)
    if task in {"copy", "reverse"}:
        content = [rng.choice(SYMBOLS) for _ in range(length)]
        target_content = oracle(task, content)
    else:
        if not 2 <= length <= 10:
            raise ValueError("Recall pair count must be between 2 and 10.")
        keys = rng.sample(SYMBOLS[:16], length)
        values = [rng.choice(SYMBOLS[16:]) for _ in range(length)]
        query_index = rng.randrange(length)
        content = [item for pair in zip(keys, values, strict=True) for item in pair]
        content += ["SEP", keys[query_index]]
        target_content = oracle(task, content)
    source = (TASK_TOKEN[task], *content, "EOS")
    target = ("BOS", *target_content, "EOS")
    digest = canonical_hash(task, source, target)
    return SequenceExample(
        example_id=f"{task}-{split}-{seed}-{digest[:8]}",
        task=task,
        split=split,
        seed=seed,
        source_tokens=source,
        target_tokens=target,
        content_length=length,
        canonical_hash=digest,
    )


def generate_balanced_suite(
    *,
    seed: int,
    count_per_task: int,
    split: str,
    copy_reverse_range: tuple[int, int],
    recall_range: tuple[int, int],
) -> list[SequenceExample]:
    examples: list[SequenceExample] = []
    seen: set[str] = set()
    for task_offset, task in enumerate(("copy", "reverse", "recall")):
        index = 0
        while index < count_per_task:
            example_seed = seed + task_offset * 100_000 + index * 997
            rng = random.Random(example_seed)
            limits = recall_range if task == "recall" else copy_reverse_range
            length = rng.randint(*limits)
            example = generate_example(task, seed=example_seed, split=split, length=length)  # type: ignore[arg-type]
            if example.canonical_hash in seen:
                continue
            seen.add(example.canonical_hash)
            examples.append(replace(example, split=split))
            index += 1
    return examples


def assert_disjoint(*splits: list[SequenceExample]) -> None:
    sets = [{example.canonical_hash for example in split} for split in splits]
    for left_index, left in enumerate(sets):
        for right in sets[left_index + 1 :]:
            if left & right:
                raise ValueError("Dataset splits contain duplicate canonical examples.")

