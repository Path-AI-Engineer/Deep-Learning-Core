# Limitations

- Fixture metrics are not FashionMNIST benchmarks.
- Reconstruction quality does not guarantee downstream usefulness.
- The linear probe measures label accessibility, not causality.
- A two-dimensional bottleneck sacrifices reconstruction capacity for inspection.
- Euclidean neighbours in latent space are local geometric evidence, not semantic truth.
- Linear interpolation may cross unsupported regions.
- The system is deterministic and does not model a latent probability distribution.
- Uploaded images may be out of distribution and are resized to a fixed grayscale input.
- No anomaly threshold is defined or exposed.
- No VAE, GAN, diffusion, t-SNE, UMAP, authentication or persistent user data exists.
