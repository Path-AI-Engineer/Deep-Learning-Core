import pytest

from neural_network_foundations.contracts import ExperimentConfig, NetworkConfig, ValidationError


def test_default_experiment_is_serializable() -> None:
    config = ExperimentConfig()
    payload = config.to_dict()
    assert payload["dataset"] == "xor"
    assert payload["network"]["hidden_units"] == 4
    assert payload["schema_version"] == "1.0"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hidden_units", 1),
        ("hidden_units", 9),
        ("hidden_activation", "softmax"),
        ("initialization", "mystery"),
    ],
)
def test_invalid_network_configuration_is_rejected(field: str, value: object) -> None:
    data = NetworkConfig().to_dict()
    data[field] = value
    with pytest.raises(ValidationError):
        NetworkConfig.from_dict(data)


def test_experiment_limits_are_enforced() -> None:
    with pytest.raises(ValidationError, match="epochs"):
        ExperimentConfig(epochs=5001)
    with pytest.raises(ValidationError, match="learning_rate"):
        ExperimentConfig(learning_rate=2.0)
