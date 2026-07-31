# Evaluation protocol

## Reconstruction

MSE and MAE measure pixel error. PSNR expresses error on a logarithmic scale. SSIM
adds a structural comparison. No single metric is treated as sufficient.

## Representation

A linear probe is trained on frozen latent vectors using `StandardScaler` and
`LogisticRegression`. Accuracy and macro F1 indicate how linearly accessible labels
are in the learned representation. They do not prove that the representation is
causal, fair or generally useful.

## Robustness

The same held-out sample is corrupted with deterministic Gaussian or masking noise.
The denoising autoencoder receives corrupted inputs and always optimizes against clean
targets.

## Selection

Validation MSE drives early stopping. Test data is reserved for final evidence.
Comparison output keeps model size, elapsed training time and the three evidence
families visible. The fixture release is evaluated only as a software acceptance
artifact, never as an official benchmark.
