import numpy as np

from neural_network_foundations.optimizers import SGD


def test_sgd_applies_expected_update() -> None:
    parameters = {"weight": np.array([[2.0, -1.0]])}
    gradients = {"weight": np.array([[0.5, -0.25]])}
    SGD(learning_rate=0.1).step(parameters, gradients)
    np.testing.assert_allclose(parameters["weight"], np.array([[1.95, -0.975]]))
