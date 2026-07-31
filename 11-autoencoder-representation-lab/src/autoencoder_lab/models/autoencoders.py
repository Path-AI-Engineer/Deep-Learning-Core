from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from autoencoder_lab.contracts import ModelConfig


class DenseAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def encode(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.encoder(inputs))

    def decode(self, latent: Tensor) -> Tensor:
        return cast(Tensor, self.decoder(latent)).reshape(-1, 1, 28, 28)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.decode(self.encode(inputs))


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.encoder_head = nn.Linear(32 * 7 * 7, latent_dim)
        self.decoder_head = nn.Linear(latent_dim, 32 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, inputs: Tensor) -> Tensor:
        features = self.encoder_conv(inputs)
        return cast(Tensor, self.encoder_head(features.flatten(start_dim=1)))

    def decode(self, latent: Tensor) -> Tensor:
        features = self.decoder_head(latent).reshape(-1, 32, 7, 7)
        return cast(Tensor, self.decoder_conv(features))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.decode(self.encode(inputs))


def build_model(config: ModelConfig) -> DenseAutoencoder | ConvolutionalAutoencoder:
    config.validate()
    if config.model_type == "dense-ae":
        return DenseAutoencoder(config.latent_dim)
    return ConvolutionalAutoencoder(config.latent_dim)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def assert_autoencoder_contract(model: nn.Module, batch_size: int = 3) -> None:
    inputs = torch.rand(batch_size, 1, 28, 28)
    output = model(inputs)
    if output.shape != inputs.shape:
        raise ValueError("autoencoder output must preserve [N, 1, 28, 28]")
    if not torch.isfinite(output).all() or output.min() < 0 or output.max() > 1:
        raise ValueError("autoencoder output must be finite and remain in [0, 1]")
