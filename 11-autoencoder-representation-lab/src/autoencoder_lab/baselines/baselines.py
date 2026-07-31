from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA


@dataclass(slots=True)
class MeanImageBaseline:
    mean_image: NDArray[np.float32] | None = None

    def fit(self, images: NDArray[np.float32]) -> MeanImageBaseline:
        self.mean_image = images.mean(axis=0, dtype=np.float64).astype(np.float32)
        return self

    def reconstruct(self, images: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.mean_image is None:
            raise RuntimeError("baseline has not been fitted")
        return np.repeat(self.mean_image[None, ...], len(images), axis=0)


@dataclass(slots=True)
class PCABaseline:
    latent_dim: int
    pca: PCA | None = None

    def fit(self, images: NDArray[np.float32]) -> PCABaseline:
        if not 1 <= self.latent_dim <= min(len(images), 784):
            raise ValueError("latent_dim is incompatible with the training matrix")
        self.pca = PCA(n_components=self.latent_dim, random_state=42)
        self.pca.fit(images.reshape(len(images), -1))
        return self

    def encode(self, images: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.pca is None:
            raise RuntimeError("PCA has not been fitted")
        return np.asarray(
            self.pca.transform(images.reshape(len(images), -1)),
            dtype=np.float32,
        )

    def decode(self, latent: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.pca is None:
            raise RuntimeError("PCA has not been fitted")
        values = self.pca.inverse_transform(latent).reshape(-1, 1, 28, 28)
        return np.asarray(values, dtype=np.float32)

    def reconstruct(self, images: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.decode(self.encode(images))
