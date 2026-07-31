from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from _common import PROJECT_ROOT, configure_imports

configure_imports()

from autoencoder_lab.artifacts import save_bundle  # noqa: E402
from autoencoder_lab.contracts import ModelConfig, ModelType, TrainingConfig  # noqa: E402
from autoencoder_lab.evaluation import reconstruction_metrics  # noqa: E402
from autoencoder_lab.models import build_model, count_parameters  # noqa: E402
from autoencoder_lab.representations import extract_embeddings, train_linear_probe  # noqa: E402
from autoencoder_lab.training import train  # noqa: E402


def _dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError("Run scripts/prepare_data.py before training.")
    with np.load(path) as values:
        return {key: values[key] for key in values.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["dense-ae", "conv-ae", "denoising-ae", "latent-2d"],
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    model_type = cast(ModelType, args.model)
    latent_dim = 2 if model_type == "latent-2d" else 16
    model_config = ModelConfig(model_type=model_type, latent_dim=latent_dim)
    training_config = TrainingConfig(epochs=args.epochs)
    values = _dataset(PROJECT_ROOT / "data/processed/fashion-mnist/dataset.npz")
    train_images = torch.from_numpy(values["train_images"])
    validation_images = torch.from_numpy(values["validation_images"])
    test_images = torch.from_numpy(values["test_images"])
    training_loader = DataLoader(
        TensorDataset(train_images),
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(training_config.seed),
    )
    validation_loader = DataLoader(
        TensorDataset(validation_images),
        batch_size=training_config.batch_size,
    )
    model = build_model(model_config)
    parameters = count_parameters(model)
    result = train(
        model,
        training_loader,
        validation_loader,
        training_config,
        denoising=model_type == "denoising-ae",
    )
    model.eval()
    with torch.inference_mode():
        reconstruction = model(test_images)
    reconstruction = reconstruction_metrics(test_images, reconstruction)
    train_embeddings = extract_embeddings(model, train_images)
    validation_embeddings = extract_embeddings(model, validation_images)
    probe = train_linear_probe(
        train_embeddings,
        values["train_labels"],
        validation_embeddings,
        values["validation_labels"],
    )
    metrics = {
        "reconstruction": reconstruction,
        "representation": {
            "linear_probe": probe,
            "labels_used_for_autoencoder_training": False,
        },
        "robustness": {
            "trained_with_corruption": model_type == "denoising-ae",
            "target": "clean",
        },
    }
    version = "v1.0.0"
    output = PROJECT_ROOT / "artifacts" / "official" / model_type / version
    save_bundle(
        output,
        model,
        model_config,
        metrics,
        result.history_dicts(),
        {
            "model_id": model_type,
            "model_version": version,
            "data_mode": "official_fashion_mnist",
            "parameters": parameters,
            "training_seconds": result.elapsed_seconds,
            "best_epoch": result.best_epoch,
            "best_validation_mse": result.best_validation_mse,
        },
    )
    print(json.dumps({"bundle": str(output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
