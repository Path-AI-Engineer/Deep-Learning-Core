# Architecture

PyTorch Tabular Studio is a single deployable application with three explicit boundaries:

1. `src/pytorch_tabular` owns data preparation, training, evaluation, artifacts and inference.
2. `backend/app` owns HTTP contracts and delegates inference to the model registry.
3. `frontend` owns presentation and consumes only the versioned `/api/v1` contract.

Training is an offline operation. The API never accepts training jobs or user checkpoints.
The production image contains only approved CPU-compatible bundles.

## Dependency direction

```text
React UI -> FastAPI schemas/routes -> inference registry -> approved bundles
offline scripts -> experiment runner -> data/models/training/evaluation -> bundles
```

## Safety boundaries

- Preprocessing is fitted on training data only.
- Test data is evaluated once after model selection.
- A bundle is rejected when its manifest hash, schema or required file is invalid.
- Classification training consumes logits; softmax is reserved for inference and metrics.
- Batch inference is limited to 100 records.
