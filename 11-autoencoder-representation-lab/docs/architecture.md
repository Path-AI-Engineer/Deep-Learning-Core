# Architecture

## Runtime flow

```text
React laboratory
      |
      v
FastAPI /api/v1
      |
      v
LabRegistry
      |
      +-- versioned autoencoder bundles
      +-- deterministic sample gallery
      +-- baseline artifacts
      +-- latent index and evaluation evidence
```

Training and artifact creation are offline concerns. The API never trains a model
inside a request and the browser never receives model weights.

## Python boundaries

- `data`: official FashionMNIST preparation and deterministic fixture generation.
- `models`: dense and convolutional encoder-decoder definitions.
- `training`: deterministic optimization, validation and early stopping.
- `corruption`: bounded Gaussian and masking transformations.
- `baselines`: mean-image and PCA reference models.
- `evaluation`: reconstruction metrics.
- `representations`: embeddings, probe, neighbours and interpolation.
- `artifacts`: versioned save/load contract with state hash verification.
- `inference`: validated image conversion and response serialization.

The FastAPI backend is a delivery adapter over these modules. React depends only on
versioned HTTP contracts.

## Decisions

1. FashionMNIST labels are excluded from autoencoder optimization.
2. The official test set is never used for early stopping.
3. Model selection uses validation MSE, while the release reports all metric families.
4. The committed fixture supports reproducible software verification without claiming
   official benchmark status.
5. A dedicated 2D model powers the latent explorer; arbitrary projection is not hidden
   behind a chart.
6. Upload inference is transient and restricted to small image files.
