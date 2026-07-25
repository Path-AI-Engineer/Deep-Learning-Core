"""Contract tests for the cross-repository execution trace."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from neural_network_foundations.contracts import ExperimentConfig
from neural_network_foundations.datasets import get_dataset
from neural_network_foundations.models import MLP


def test_trace_schema_is_valid_and_accepts_a_real_engine_trace() -> None:
    schema = json.loads(Path("contracts/trace-v1.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    config = ExperimentConfig()
    dataset = get_dataset(config.dataset)
    model = MLP(config.network, seed=config.seed)
    model.forward(dataset.features)
    model.backward(dataset.targets, loss_name=config.loss)
    trace = model.trace_sample(
        dataset=config.dataset,
        features=dataset.features,
        targets=dataset.targets,
        sample_index=1,
        loss_name=config.loss,
        configuration=config.to_dict(),
    ).to_dict()

    Draft202012Validator(schema).validate(trace)
