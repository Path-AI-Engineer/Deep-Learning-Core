import json

import numpy as np

from neural_network_foundations.contracts import ExperimentConfig
from neural_network_foundations.datasets import get_dataset
from neural_network_foundations.models import MLP


def test_mlp_forward_is_deterministic_and_has_expected_shape() -> None:
    config = ExperimentConfig()
    dataset = get_dataset("xor")
    first = MLP(config.network, seed=config.seed)
    second = MLP(config.network, seed=config.seed)
    first_predictions = first.forward(dataset.features)
    second_predictions = second.forward(dataset.features)
    assert first_predictions.shape == (4, 1)
    np.testing.assert_allclose(first_predictions, second_predictions)


def test_trace_contains_reconstructable_neuron_calculation() -> None:
    config = ExperimentConfig()
    dataset = get_dataset("xor")
    model = MLP(config.network, seed=config.seed)
    model.forward(dataset.features)
    trace = model.trace_sample(
        dataset="xor",
        features=dataset.features,
        targets=dataset.targets,
        sample_index=1,
        loss_name=config.loss,
        configuration=config.to_dict(),
    ).to_dict()
    assert trace["schema_version"] == "1.0"
    assert len(trace["nodes"]) == config.network.hidden_units + 1
    first_node = trace["nodes"][0]
    reconstructed_z = np.dot(first_node["inputs"], first_node["weights"]) + first_node["bias"]
    assert reconstructed_z == first_node["z"]
    json.dumps(trace, allow_nan=False)


def test_parameter_loading_reproduces_predictions() -> None:
    config = ExperimentConfig()
    dataset = get_dataset("xor")
    source = MLP(config.network, seed=1)
    expected = source.forward(dataset.features)
    restored = MLP(config.network, seed=99)
    restored.load_parameters({name: values.copy() for name, values in source.parameters().items()})
    np.testing.assert_allclose(restored.forward(dataset.features), expected)
