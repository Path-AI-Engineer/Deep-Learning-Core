# v1.0.0 — PyTorch Tabular Studio

Project 08 delivers an end-to-end PyTorch tabular inference product rather
than an isolated training notebook.

## Highlights

- Regression and multiclass classification MLPs with shared, tested training
  infrastructure.
- Deterministic train/validation/test splits and train-only preprocessing.
- Baseline-gated model acceptance with validation-selected checkpoints.
- Immutable, hash-validated, CPU-compatible inference bundles.
- Versioned FastAPI for health, task schemas, model cards, single prediction
  and batch prediction.
- Responsive React/TypeScript interface with guided forms, metrics, curves,
  probabilities, batch preview and downloadable results.
- Multi-stage Docker packaging and a reproducible smoke-test command.
- Technical contracts, model cards, demo guide and responsive screenshots.

## Evidence

- Regression MAE: `0.5097`, below the mean-regressor baseline.
- Classification macro F1: `0.9599`, above the prior-only baseline.
- Classification accuracy: `0.9630`.
- Python suite: `18 passed`.
- Static typing: `31 source files` clean.
- Frontend build and runtime tests: passed.
- Local API/UI smoke and browser QA: passed.

## Known limitation

The included regression artifact is a workflow-demonstration bundle trained on
the documented official-source reference sample. Retrain against the complete
California Housing CSV before publishing full-dataset benchmark claims.
