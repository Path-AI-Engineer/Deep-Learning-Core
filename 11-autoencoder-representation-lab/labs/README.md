# Technical labs

These labs are guided investigations over the same product artifacts. They do not
introduce parallel implementations.

## 1. Basic autoencoder

Train `dense-ae`, inspect its learning history and verify that input and output retain
`[N, 1, 28, 28]`. Compare it with the mean-image baseline before drawing a conclusion.

## 2. Convolutional reconstruction

Train `conv-ae` under the same split and evaluation protocol. Compare MSE, MAE, PSNR,
SSIM, parameter count and elapsed training time with Dense AE and PCA.

## 3. Denoising

Apply one seeded corruption to both `conv-ae` and `denoising-ae`. Keep the original
clean image as target. Explain why a matched corruption is required for comparison.

## 4. Latent representation

Extract frozen embeddings and train the shared linear probe. Explain why macro F1 can
change independently from reconstruction MSE.

## 5. Two-dimensional bottleneck

Inspect the direct coordinates produced by `latent-2d`. Select neighbours and decode
an in-bounds coordinate. Do not describe the chart as a t-SNE or UMAP projection.

## 6. Interpolation and errors

Interpolate between two observed samples using three to twelve deterministic steps.
Then inspect the highest per-sample reconstruction errors and describe what those
errors do and do not establish.

## Reproducibility checklist

- Record model ID, version, seed and split.
- Preserve the untouched test partition.
- Confirm labels were excluded from autoencoder training.
- Distinguish fixture evidence from official FashionMNIST evidence.
- Keep baseline conflicts visible.
- Record limitations with every conclusion.
