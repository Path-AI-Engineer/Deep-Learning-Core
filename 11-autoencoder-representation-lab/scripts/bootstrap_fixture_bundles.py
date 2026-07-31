from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from _common import PROJECT_ROOT, SRC_ROOT, loader, reset, write_json

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from autoencoder_lab.artifacts import load_bundle, save_bundle  # noqa: E402
from autoencoder_lab.baselines import MeanImageBaseline, PCABaseline  # noqa: E402
from autoencoder_lab.contracts import CLASS_MAPPING, ModelConfig, TrainingConfig  # noqa: E402
from autoencoder_lab.corruption import corrupt  # noqa: E402
from autoencoder_lab.data import build_fixture_records, stack_records  # noqa: E402
from autoencoder_lab.evaluation import reconstruction_metrics  # noqa: E402
from autoencoder_lab.inference import image_data_url  # noqa: E402
from autoencoder_lab.models import build_model, count_parameters  # noqa: E402
from autoencoder_lab.representations import (  # noqa: E402
    extract_embeddings,
    nearest_neighbors,
    train_linear_probe,
)
from autoencoder_lab.training import train  # noqa: E402


def _metric_payload(
    model: torch.nn.Module,
    validation: torch.Tensor,
    train_images: torch.Tensor,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    denoising: bool,
) -> dict[str, object]:
    model.eval()
    with torch.inference_mode():
        reconstructed = model(validation)
        noisy = corrupt(validation, "gaussian", 0.2, 42)
        denoised = model(noisy)
    train_embeddings = extract_embeddings(model, train_images)
    validation_embeddings = extract_embeddings(model, validation)
    probe = train_linear_probe(
        train_embeddings,
        train_labels,
        validation_embeddings,
        validation_labels,
    )
    return {
        "reconstruction": reconstruction_metrics(validation, reconstructed).to_dict(),
        "representation": {
            "linear_probe": probe,
            "labels_used_for_autoencoder_training": False,
        },
        "robustness": {
            "gaussian_0.2": reconstruction_metrics(validation, denoised).to_dict(),
            "trained_with_corruption": denoising,
            "target": "clean",
        },
    }


def build(force: bool = False) -> dict[str, object]:
    records = build_fixture_records(15)
    train_images, train_labels, train_ids = stack_records(records, "train")
    validation_images, validation_labels, _ = stack_records(records, "validation")
    test_images, test_labels, test_ids = stack_records(records, "test")
    train_tensor = torch.from_numpy(train_images)
    validation_tensor = torch.from_numpy(validation_images)
    test_tensor = torch.from_numpy(test_images)
    training_loader = loader(train_images, 32, True)
    validation_loader = loader(validation_images, 32, False)
    comparison: list[dict[str, object]] = []
    models: dict[str, torch.nn.Module] = {}
    mean_baseline = MeanImageBaseline().fit(train_images)
    mean_prediction = torch.from_numpy(mean_baseline.reconstruct(validation_images))
    mean_metrics = {
        "model_id": "mean-image",
        "reconstruction": reconstruction_metrics(
            validation_tensor, mean_prediction
        ).to_dict(),
        "representation": {
            "linear_probe": None,
            "labels_used_for_autoencoder_training": False,
        },
        "robustness": {"not_applicable": True},
        "validation_mse": float(
            torch.mean((mean_prediction - validation_tensor) ** 2).item()
        ),
        "parameters": 0,
        "training_seconds": 0.0,
    }
    mean_directory = PROJECT_ROOT / "artifacts" / "models" / "mean-image" / "v1.0.0"
    reset(mean_directory, force)
    mean_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mean_directory / "baseline.npz", mean_image=mean_baseline.mean_image)
    write_json(mean_directory / "metrics.json", mean_metrics)
    write_json(
        mean_directory / "manifest.json",
        {
            "model_id": "mean-image",
            "model_version": "v1.0.0",
            "dataset_status": "fixture-not-fashionmnist",
            "latent_dim": None,
            "warning": "Fixture evidence only.",
        },
    )
    comparison.append(mean_metrics)
    pca = PCABaseline(16).fit(train_images)
    pca_prediction_np = pca.reconstruct(validation_images)
    pca_prediction = torch.from_numpy(pca_prediction_np)
    train_pca = pca.encode(train_images)
    validation_pca = pca.encode(validation_images)
    pca_probe = train_linear_probe(
        train_pca,
        train_labels,
        validation_pca,
        validation_labels,
    )
    pca_metrics = {
        "model_id": "pca",
        "reconstruction": reconstruction_metrics(
            validation_tensor, pca_prediction
        ).to_dict(),
        "representation": {
            "linear_probe": pca_probe,
            "labels_used_for_autoencoder_training": False,
        },
        "robustness": {"not_applicable": True},
        "validation_mse": float(
            torch.mean((pca_prediction - validation_tensor) ** 2).item()
        ),
        "parameters": int(16 * 784),
        "training_seconds": 0.0,
    }
    pca_directory = PROJECT_ROOT / "artifacts" / "models" / "pca" / "v1.0.0"
    reset(pca_directory, force)
    pca_directory.mkdir(parents=True, exist_ok=True)
    assert pca.pca is not None
    np.savez_compressed(
        pca_directory / "pca.npz",
        mean=pca.pca.mean_.astype(np.float32),
        components=pca.pca.components_.astype(np.float32),
        explained_variance_ratio=pca.pca.explained_variance_ratio_.astype(np.float32),
    )
    write_json(pca_directory / "metrics.json", pca_metrics)
    write_json(
        pca_directory / "manifest.json",
        {
            "model_id": "pca",
            "model_version": "v1.0.0",
            "dataset_status": "fixture-not-fashionmnist",
            "latent_dim": 16,
            "fit_split": "fixture training only",
            "warning": "Fixture evidence only.",
        },
    )
    comparison.append(pca_metrics)
    for model_id, latent_dim in (
        ("dense-ae", 16),
        ("conv-ae", 16),
        ("denoising-ae", 16),
        ("latent-2d", 2),
    ):
        torch.manual_seed(42)
        config = ModelConfig(model_type=model_id, latent_dim=latent_dim)  # type: ignore[arg-type]
        model = build_model(config)
        result = train(
            model,
            training_loader,
            validation_loader,
            TrainingConfig(epochs=10, batch_size=32),
            denoising=model_id == "denoising-ae",
        )
        parameters = count_parameters(model)
        metrics = _metric_payload(
            model,
            validation_tensor,
            train_tensor,
            train_labels,
            validation_labels,
            model_id == "denoising-ae",
        )
        metrics["validation_mse"] = result.best_validation_mse
        metrics["parameters"] = parameters
        metrics["training_seconds"] = round(result.elapsed_seconds, 4)
        directory = PROJECT_ROOT / "artifacts" / "models" / model_id / "v1.0.0"
        reset(directory, force)
        save_bundle(
            directory,
            model,
            config,
            metrics,
            result.history_dicts(),
            {
                "task": "deterministic image reconstruction",
                "model_id": model_id,
                "model_type": model_id,
                "model_version": "v1.0.0",
                "dataset": "FashionMNIST-shaped educational fixture",
                "dataset_status": "fixture-not-fashionmnist",
                "official_dataset_target": "FashionMNIST",
                "source": "local deterministic generator",
                "license": "project fixture",
                "seed": 42,
                "input_shape": [None, 1, 28, 28],
                "dtype": "float32",
                "pixel_range": [0.0, 1.0],
                "latent_dim": latent_dim,
                "labels_used_for_autoencoder_training": False,
                "device": "cpu",
                "warning": "Fixture metrics validate software, not FashionMNIST performance.",
            },
        )
        models[model_id] = model
        comparison.append({"model_id": model_id, **metrics})
    active_model = "conv-ae"
    comparison_payload = {
        "version": "v1.0.0",
        "data_mode": "educational_fixture",
        "active_model": active_model,
        "selection": (
            "Conv AE is the active neural bundle for reconstruction, encoding and matched "
            "robustness workflows. PCA remains the strongest fixture reconstruction baseline; "
            "the conflict is retained rather than hidden."
        ),
        "models": comparison,
        "warning": "No value in this comparison is a FashionMNIST benchmark.",
    }
    write_json(
        PROJECT_ROOT / "artifacts" / "comparisons" / "v1.0.0" / "model_comparison.json",
        comparison_payload,
    )
    latent_model = models["latent-2d"]
    embeddings = extract_embeddings(latent_model, test_tensor)
    bounds = {
        "x": [float(embeddings[:, 0].min()), float(embeddings[:, 0].max())],
        "y": [float(embeddings[:, 1].min()), float(embeddings[:, 1].max())],
    }
    sample_index = []
    with torch.inference_mode():
        reconstructions = latent_model(test_tensor)
    for index, sample_id in enumerate(test_ids):
        neighbors = nearest_neighbors(
            embeddings[index],
            embeddings,
            test_ids,
            limit=5,
            exclude_id=sample_id,
        )
        sample_index.append(
            {
                "sample_id": sample_id,
                "label": int(test_labels[index]),
                "class_name": CLASS_MAPPING[int(test_labels[index])],
                "x": round(float(embeddings[index, 0]), 6),
                "y": round(float(embeddings[index, 1]), 6),
                "image": image_data_url(test_tensor[index]),
                "reconstruction": image_data_url(reconstructions[index]),
                "neighbors": neighbors,
            }
        )
    latent_directory = PROJECT_ROOT / "artifacts" / "models" / "latent-2d" / "v1.0.0"
    write_json(latent_directory / "latent_bounds.json", bounds)
    write_json(latent_directory / "sample_index.json", {"items": sample_index})
    gallery = []
    for record in records:
        if record.split == "test":
            gallery.append(
                {
                    "sample_id": record.sample_id,
                    "label": record.label,
                    "class_name": record.class_name,
                    "split": record.split,
                    "image": image_data_url(record.image),
                }
            )
    write_json(
        PROJECT_ROOT / "data" / "samples" / "fixture_gallery.json",
        {"items": gallery, "data_mode": "educational_fixture"},
    )
    write_json(
        PROJECT_ROOT / "data" / "samples" / "fixture_tensors.json",
        {
            "sample_ids": test_ids,
            "labels": test_labels.tolist(),
            "images": test_images.tolist(),
        },
    )
    return comparison_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    comparison_path = (
        PROJECT_ROOT / "artifacts/comparisons/v1.0.0/model_comparison.json"
    )
    if comparison_path.is_file() and not args.force:
        for model_id in ("dense-ae", "conv-ae", "denoising-ae", "latent-2d"):
            load_bundle(PROJECT_ROOT / "artifacts/models" / model_id / "v1.0.0")
        print(
            json.dumps(
                {
                    "status": "existing fixture bundles are valid",
                    "comparison": str(comparison_path),
                    "hint": "Pass --force only to regenerate deterministic artifacts.",
                },
                indent=2,
            )
        )
        return
    print(json.dumps(build(args.force), indent=2))


if __name__ == "__main__":
    main()
