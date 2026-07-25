import numpy as np
import pytest

from neural_network_foundations.layers import Dense


def test_dense_forward_matches_manual_calculation() -> None:
    layer = Dense(
        input_units=2,
        output_units=2,
        weights=np.array([[0.5, -1.0], [2.0, 0.25]]),
        bias=np.array([[0.1, -0.2]]),
        layer_id="hidden",
    )
    result = layer.forward(np.array([[1.0, 3.0]]))
    np.testing.assert_allclose(result, np.array([[6.6, -0.45]]))


def test_dense_rejects_wrong_batch_shape() -> None:
    layer = Dense.small_normal(2, 2, rng=np.random.default_rng(1), layer_id="hidden")
    with pytest.raises(ValueError, match="shape"):
        layer.forward(np.array([[1.0, 2.0, 3.0]]))


def test_dense_backward_shapes_and_values() -> None:
    layer = Dense(
        2,
        1,
        np.array([[2.0], [-1.0]]),
        np.array([[0.5]]),
        "output",
    )
    layer.forward(np.array([[1.0, 3.0], [2.0, 4.0]]))
    input_gradient = layer.backward(np.array([[0.25], [-0.5]]))
    assert layer.grad_weights is not None
    assert layer.grad_bias is not None
    np.testing.assert_allclose(layer.grad_weights, np.array([[-0.75], [-1.25]]))
    np.testing.assert_allclose(layer.grad_bias, np.array([[-0.25]]))
    np.testing.assert_allclose(input_gradient, np.array([[0.5, -0.25], [-1.0, 0.5]]))
