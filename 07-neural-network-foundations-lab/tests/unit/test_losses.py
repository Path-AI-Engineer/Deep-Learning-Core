import numpy as np

from neural_network_foundations.losses import loss, loss_derivative


def test_mean_squared_error_known_value() -> None:
    predictions = np.array([[0.0], [0.5], [1.0]])
    targets = np.array([[0.0], [1.0], [1.0]])
    assert loss("mean_squared_error", predictions, targets) == 1.0 / 12.0


def test_binary_cross_entropy_is_stable_at_boundaries() -> None:
    predictions = np.array([[0.0], [1.0]])
    targets = np.array([[0.0], [1.0]])
    value = loss("binary_cross_entropy", predictions, targets)
    assert np.isfinite(value)
    assert value < 1e-9


def test_loss_derivatives_match_finite_difference() -> None:
    predictions = np.array([[0.2], [0.7]])
    targets = np.array([[0.0], [1.0]])
    epsilon = 1e-6
    for name in ("mean_squared_error", "binary_cross_entropy"):
        numerical = np.zeros_like(predictions)
        for row in range(predictions.shape[0]):
            plus = predictions.copy()
            minus = predictions.copy()
            plus[row, 0] += epsilon
            minus[row, 0] -= epsilon
            numerical[row, 0] = (loss(name, plus, targets) - loss(name, minus, targets)) / (
                2.0 * epsilon
            )
        np.testing.assert_allclose(
            loss_derivative(name, predictions, targets), numerical, rtol=1e-5
        )
