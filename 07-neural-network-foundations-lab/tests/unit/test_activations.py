import numpy as np
import pytest

from neural_network_foundations.activations import activation, derivative


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "relu"])
def test_activation_derivative_matches_finite_difference(name: str) -> None:
    values = np.array([[-2.0, -0.5, 0.5, 2.0]])
    epsilon = 1e-6
    numerical = (activation(name, values + epsilon) - activation(name, values - epsilon)) / (
        2.0 * epsilon
    )
    np.testing.assert_allclose(derivative(name, values), numerical, rtol=1e-5, atol=1e-6)


def test_sigmoid_is_stable_for_extreme_values() -> None:
    result = activation("sigmoid", np.array([[-1000.0, 1000.0]]))
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, np.array([[0.0, 1.0]]), atol=1e-12)
