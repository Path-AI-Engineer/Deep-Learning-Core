import numpy as np
import pytest

from neural_network_foundations.datasets import DatasetBundle, get_dataset, list_datasets


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("and", [0, 0, 0, 1]),
        ("or", [0, 1, 1, 1]),
        ("xor", [0, 1, 1, 0]),
    ],
)
def test_truth_tables_have_known_labels(name: str, expected: list[int]) -> None:
    dataset = get_dataset(name)
    assert dataset.features.shape == (4, 2)
    assert dataset.targets.reshape(-1).tolist() == expected


def test_circles_are_deterministic() -> None:
    first = get_dataset("circles", seed=23, samples=80)
    second = get_dataset("circles", seed=23, samples=80)
    np.testing.assert_allclose(first.features, second.features)
    np.testing.assert_array_equal(first.targets, second.targets)


def test_invalid_dataset_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="features"):
        DatasetBundle("bad", "", "", np.array([1.0, 2.0]), np.array([[1.0]]), 0)


def test_catalog_is_serializable() -> None:
    catalog = list_datasets()
    assert [item["name"] for item in catalog] == ["and", "or", "xor", "circles"]
    assert all(item["problem_type"] == "binary_classification" for item in catalog)
