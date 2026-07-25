"""Inspectable neural-network mechanics implemented with NumPy."""

from neural_network_foundations.contracts.config import ExperimentConfig, NetworkConfig
from neural_network_foundations.datasets.catalog import DatasetBundle, get_dataset

__all__ = ["DatasetBundle", "ExperimentConfig", "NetworkConfig", "get_dataset"]
__version__ = "0.1.0"
