import numpy as np
import pytest

from neural_network_foundations.layers import initialize_parameters


@pytest.mark.parametrize("strategy", ["zeros", "small_normal", "xavier", "he"])
def test_initialization_is_reproducible_and_has_expected_shapes(strategy: str) -> None:
    first = initialize_parameters(3, 5, strategy=strategy, rng=np.random.default_rng(42))
    second = initialize_parameters(3, 5, strategy=strategy, rng=np.random.default_rng(42))
    assert first[0].shape == (3, 5)
    assert first[1].shape == (1, 5)
    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])


def test_zero_initialization_preserves_hidden_symmetry() -> None:
    weights, _ = initialize_parameters(2, 4, strategy="zeros", rng=np.random.default_rng(1))
    assert np.unique(weights, axis=1).shape[1] == 1
