from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import cast

import torch

from transformer_lab.artifacts import Bundle, load_bundle
from transformer_lab.contracts import SequenceExample, TaskId
from transformer_lab.data import canonical_hash, oracle
from transformer_lab.decoding import DecodingResult, greedy_decode
from transformer_lab.evaluation import evaluate_sequence
from transformer_lab.tokenization import EOS, Vocabulary


class ModelRegistry:
    def __init__(
        self,
        *,
        bundle_path: Path | None = None,
        sample_path: Path | None = None,
    ) -> None:
        self.bundle_path = bundle_path or Path(
            os.getenv(
                "TRANSFORMER_BUNDLE_PATH",
                "artifacts/models/transformer/v1.0.0-reference",
            )
        )
        self.sample_path = sample_path or Path(
            os.getenv(
                "TRANSFORMER_SAMPLE_PATH",
                "data/samples/demo_catalog.json",
            )
        )
        self.bundle: Bundle | None = None
        self.samples: list[SequenceExample] = []
        self.error: str | None = None

    def load(self) -> None:
        try:
            self.bundle = load_bundle(self.bundle_path)
            payload = json.loads(self.sample_path.read_text(encoding="utf-8"))
            self.samples = [
                SequenceExample(
                    example_id=row["example_id"],
                    task=cast(TaskId, row["task"]),
                    split=row["split"],
                    seed=int(row["seed"]),
                    source_tokens=tuple(row["source_tokens"]),
                    target_tokens=tuple(row["target_tokens"]),
                    content_length=int(row["content_length"]),
                    canonical_hash=row["canonical_hash"],
                )
                for row in payload["samples"]
            ]
            self.error = None
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            self.bundle = None
            self.error = str(error)

    @property
    def ready(self) -> bool:
        return self.bundle is not None and bool(self.samples)

    def require_bundle(self) -> Bundle:
        if self.bundle is None:
            raise RuntimeError("The approved Transformer bundle is unavailable.")
        return self.bundle

    def sample(self, sample_id: str) -> SequenceExample:
        for sample in self.samples:
            if sample.example_id == sample_id:
                return sample
        raise ValueError(f"Unknown sample ID: {sample_id}.")

    def custom_example(self, task: TaskId, symbols: list[str]) -> SequenceExample:
        if task == "recall":
            target_content = oracle(task, symbols)
            source = ("RECALL", *symbols, "EOS")
            target = ("BOS", *target_content, "EOS")
            return SequenceExample(
                example_id="custom",
                task=task,
                split="custom",
                seed=0,
                source_tokens=source,
                target_tokens=target,
                content_length=(len(symbols) - 2) // 2,
                canonical_hash=canonical_hash(task, source, target),
            )
        target_content = oracle(task, symbols)
        task_token = "COPY" if task == "copy" else "REVERSE"
        source = (task_token, *symbols, "EOS")
        target = ("BOS", *target_content, "EOS")
        return SequenceExample(
            example_id="custom",
            task=task,
            split="custom",
            seed=0,
            source_tokens=source,
            target_tokens=target,
            content_length=len(symbols),
            canonical_hash=canonical_hash(task, source, target),
        )

    def predict(
        self,
        example: SequenceExample,
        *,
        max_new_tokens: int,
        trace: bool = False,
    ) -> tuple[DecodingResult, dict[str, float | int], float]:
        bundle = self.require_bundle()
        vocabulary = Vocabulary()
        source_ids = torch.tensor(
            [vocabulary.encode(example.source_tokens)],
            dtype=torch.long,
        )
        started = time.perf_counter()
        result = greedy_decode(
            bundle.model,
            source_ids,
            max_new_tokens=max_new_tokens,
            trace=trace,
        )
        elapsed = (time.perf_counter() - started) * 1000
        target_ids = vocabulary.encode(example.target_tokens[1:])
        metrics = evaluate_sequence(result.token_ids, target_ids).as_dict()
        return result, metrics, elapsed

    @staticmethod
    def trace_matrix(
        result: DecodingResult,
        *,
        trace_type: str,
        layer: int,
        head: int,
    ) -> list[list[float]]:
        if result.trace is None:
            raise RuntimeError("Trace data was not requested.")
        key = {
            "encoder_self": "encoder_self",
            "decoder_self": "decoder_self",
            "cross": "cross",
        }.get(trace_type)
        if key is None:
            raise ValueError("Unknown trace type.")
        layers = result.trace[key]
        if not isinstance(layers, list) or not 0 <= layer < len(layers):
            raise ValueError("Layer is outside the trace whitelist.")
        tensor = layers[layer]
        if not isinstance(tensor, torch.Tensor) or not 0 <= head < tensor.shape[1]:
            raise ValueError("Head is outside the trace whitelist.")
        return [
            [round(float(value), 8) for value in row]
            for row in tensor[0, head].cpu()
        ]

    def prediction_tokens(self, result: DecodingResult) -> list[str]:
        values = list(result.token_ids)
        if EOS in values:
            values = values[: values.index(EOS) + 1]
        return Vocabulary().decode(values)
