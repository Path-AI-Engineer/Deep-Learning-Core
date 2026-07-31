# Model card

## Purpose

Latent Representation Lab demonstrates deterministic compression, reconstruction,
denoising and representation inspection. It is an educational engineering system,
not a production vision model.

## Models

- Mean-image reconstruction baseline.
- PCA reconstruction baseline.
- Dense autoencoder.
- Convolutional autoencoder.
- Denoising convolutional autoencoder.
- Convolutional autoencoder with a two-dimensional bottleneck.

The active fixture bundle is `conv-ae:v1.0.0`. PCA currently has the lowest fixture
reconstruction MSE; this conflict is retained in comparison evidence.

## Training

Neural models optimize pixel MSE with AdamW, deterministic seeds and validation-based
early stopping. Denoising inputs are corrupted while targets remain clean. Labels are
excluded from optimization.

## Evaluation

Reconstruction uses MSE, MAE, PSNR and SSIM. Representation evidence uses a scaled
logistic-regression probe trained after embeddings are frozen. Robustness evaluates
matched corruptions. These signals are reported separately.

## Intended use

- Learn encoder-decoder and bottleneck mechanics.
- Compare neural reconstruction with simple baselines.
- Inspect local geometry and interpolation in a deterministic latent space.
- Review model errors and evidence limitations.

## Excluded use

Do not use this release for biometric inference, safety decisions, anomaly detection,
probabilistic generation, business automation or claims about official FashionMNIST
performance.
