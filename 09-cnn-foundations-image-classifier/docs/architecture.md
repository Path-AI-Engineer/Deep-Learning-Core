# Architecture

CNN Vision Lab is an inference-only educational product. Training and artifact
approval run offline; the packaged application loads a CPU bundle once and
serves a React interface and a versioned FastAPI contract from one container.

```text
FashionMNIST
  -> deterministic train/validation/test contract
  -> fair MLP baseline + CNN experiments
  -> validation checkpoint selection
  -> isolated test evaluation
  -> immutable, hash-verified CPU bundle
  -> FastAPI registry
  -> prediction / activation / evaluation services
  -> React laboratory
```

## Boundaries

- `src/cnn_foundations/data`: official dataset, split and controlled gallery.
- `src/cnn_foundations/operations`: framework-independent convolution math.
- `src/cnn_foundations/models`: reconstructable MLP and CNN definitions.
- `src/cnn_foundations/training`: explicit train/evaluation modes and checkpoints.
- `src/cnn_foundations/evaluation`: metrics and reproducible error records.
- `src/cnn_foundations/artifacts`: versioned bundle writer, hashes and CPU loader.
- `src/cnn_foundations/inference`: shared preprocessing and prediction contract.
- `src/cnn_foundations/explanations`: whitelisted hooks with guaranteed cleanup.
- `backend/app`: thin HTTP interface and process-wide model registry.
- `frontend`: product interface; it never fabricates model outputs.

## Runtime policy

The API may boot in `degraded` state for local architecture work. Model-dependent
routes return HTTP 503 until both the official test set and an approved bundle are
present. The production Dockerfile is stricter: it refuses to build without them.

No database, authentication, remote storage, retraining endpoint or background
job exists in this project.
