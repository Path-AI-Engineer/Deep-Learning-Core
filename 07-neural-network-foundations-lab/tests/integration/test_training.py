from neural_network_foundations.contracts import ExperimentConfig
from neural_network_foundations.datasets import get_dataset
from neural_network_foundations.evaluation import binary_accuracy, decision_boundary
from neural_network_foundations.models import MLP
from neural_network_foundations.training import train


def test_training_reduces_loss_on_xor() -> None:
    config = ExperimentConfig(epochs=500)
    dataset = get_dataset("xor")
    model = MLP(config.network, seed=config.seed)
    history = train(
        model,
        dataset.features,
        dataset.targets,
        loss_name=config.loss,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
    )
    assert history.status == "completed"
    assert history.loss[-1] < history.loss[0]


def test_approved_configuration_solves_xor() -> None:
    config = ExperimentConfig()
    dataset = get_dataset("xor")
    model = MLP(config.network, seed=config.seed)
    history = train(
        model,
        dataset.features,
        dataset.targets,
        loss_name=config.loss,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
    )
    assert history.loss[-1] < 0.02
    assert binary_accuracy(model.forward(dataset.features), dataset.targets) == 1.0


def test_training_is_reproducible_and_boundary_is_bounded() -> None:
    config = ExperimentConfig(epochs=50, grid_resolution=20)
    dataset = get_dataset("xor")
    first = MLP(config.network, seed=config.seed)
    second = MLP(config.network, seed=config.seed)
    first_history = train(
        first,
        dataset.features,
        dataset.targets,
        loss_name=config.loss,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
    )
    second_history = train(
        second,
        dataset.features,
        dataset.targets,
        loss_name=config.loss,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
    )
    assert first_history.to_dict() == second_history.to_dict()
    boundary = decision_boundary(first, dataset.features, resolution=config.grid_resolution)
    assert len(boundary["probabilities"]) == 20
    assert len(boundary["probabilities"][0]) == 20
