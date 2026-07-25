import pytest

torch = pytest.importorskip("torch")

from neural_network_foundations.contracts import ExperimentConfig  # noqa: E402
from neural_network_foundations.datasets import get_dataset  # noqa: E402
from neural_network_foundations.models import MLP  # noqa: E402
from neural_network_foundations.models.pytorch_reference import (  # noqa: E402
    compare_with_pytorch,
)


def test_numpy_matches_pytorch_forward_loss_gradients_and_update() -> None:
    config = ExperimentConfig()
    dataset = get_dataset("xor")
    report = compare_with_pytorch(
        MLP(config.network, seed=config.seed),
        dataset.features,
        dataset.targets,
        loss_name=config.loss,
        learning_rate=config.learning_rate,
    )
    assert report.passed, report.to_dict()
