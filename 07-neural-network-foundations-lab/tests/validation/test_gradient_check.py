import pytest

from neural_network_foundations.contracts import ExperimentConfig, NetworkConfig
from neural_network_foundations.datasets import get_dataset
from neural_network_foundations.evaluation import check_model_gradients
from neural_network_foundations.models import MLP


@pytest.mark.parametrize("hidden_activation", ["sigmoid", "tanh", "relu"])
def test_all_parameter_gradients_match_finite_differences(hidden_activation: str) -> None:
    config = ExperimentConfig(
        network=NetworkConfig(hidden_activation=hidden_activation),
        seed=11,
    )
    dataset = get_dataset("xor")
    model = MLP(config.network, seed=config.seed)
    if hidden_activation == "relu":
        # ReLU is not differentiable at z=0. The XOR sample [0, 0] with a
        # zero bias lands exactly on that kink, so this parity check uses a
        # small non-zero bias while the separate activation test documents
        # the chosen derivative at zero.
        model.hidden.bias[...] = 0.123
    report = check_model_gradients(
        model,
        dataset.features,
        dataset.targets,
        loss_name=config.loss,
    )
    assert report.passed, report.to_dict()
    assert sum(item.checked_values for item in report.parameters) == 17
