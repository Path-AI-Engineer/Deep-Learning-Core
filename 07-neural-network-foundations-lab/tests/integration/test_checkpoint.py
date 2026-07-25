from pathlib import Path
from uuid import uuid4

import numpy as np

from neural_network_foundations.contracts import ExperimentConfig
from neural_network_foundations.datasets import get_dataset
from neural_network_foundations.models import MLP
from neural_network_foundations.serialization import load_checkpoint, save_checkpoint


def test_checkpoint_restores_identical_predictions() -> None:
    output = Path(".runtime/tests") / f"checkpoint-{uuid4().hex}"
    config = ExperimentConfig()
    dataset = get_dataset("xor")
    source = MLP(config.network, seed=config.seed)
    expected = source.forward(dataset.features)
    checkpoint, _ = save_checkpoint(
        source,
        output / "model.npz",
        metadata={"seed": config.seed},
    )
    parameters, metadata = load_checkpoint(checkpoint)
    restored = MLP(config.network, seed=99)
    restored.load_parameters(parameters)
    np.testing.assert_allclose(restored.forward(dataset.features), expected)
    assert metadata["metadata"]["seed"] == config.seed
