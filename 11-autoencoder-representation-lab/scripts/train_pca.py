from __future__ import annotations

import json

import joblib
import numpy as np
import torch

from _common import PROJECT_ROOT, configure_imports

configure_imports()

from autoencoder_lab.baselines import PCABaseline  # noqa: E402
from autoencoder_lab.evaluation import reconstruction_metrics  # noqa: E402


def main() -> None:
    dataset = PROJECT_ROOT / "data/processed/fashion-mnist/dataset.npz"
    if not dataset.is_file():
        raise FileNotFoundError("Run scripts/prepare_data.py before training PCA.")
    with np.load(dataset) as values:
        train_images = values["train_images"]
        validation_images = values["validation_images"]
    model = PCABaseline(latent_dim=16)
    model.fit(train_images)
    output = PROJECT_ROOT / "artifacts/official/pca/v1.0.0"
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "pca.joblib")
    reconstruction = model.reconstruct(validation_images)
    metrics = reconstruction_metrics(
        torch.from_numpy(validation_images),
        torch.from_numpy(reconstruction),
    ).to_dict()
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"bundle": str(output), "validation": metrics}, indent=2))


if __name__ == "__main__":
    main()
