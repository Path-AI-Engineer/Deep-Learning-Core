"""CNN foundations package."""

from cnn_foundations.contracts.config import CLASS_NAMES, DatasetConfig, ExperimentConfig
from cnn_foundations.models.cnn import FashionCNN
from cnn_foundations.models.mlp import FashionMLP

__all__ = [
    "CLASS_NAMES",
    "DatasetConfig",
    "ExperimentConfig",
    "FashionCNN",
    "FashionMLP",
]

