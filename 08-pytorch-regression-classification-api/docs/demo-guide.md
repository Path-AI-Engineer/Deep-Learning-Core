# Demo Guide

## Preparation

1. Start the complete application on port `8008`.
2. Open `/api/v1/health` and confirm both tasks are `true`.
3. Open the product root in a 1440 px browser.

## Product walkthrough

1. **Overview** — explain that two isolated PyTorch bundles share one runtime
   contract, while model weights and outputs remain task-specific.
2. **Regression Studio** — load the approved example, submit it and identify the
   predicted value, USD 100,000 unit, model version and limitations.
3. Compare MAE with the mean baseline and show the train/validation curve.
4. **Classification Studio** — load the approved example and submit it.
5. Explain the predicted Wine class, all three probability bars and why
   probability is not certainty.
6. Show macro F1 against the prior baseline and the confusion matrix.
7. **Batch Studio** — upload `data/samples/classification_batch.csv`, inspect the
   preview, run three observations and download the JSON result.
8. Demonstrate a controlled error using a CSV with a missing column.
9. **Model Metrics** — compare architecture, metrics, baselines and curves.
10. **About** — close with preprocessing, validation checkpoint selection,
    CPU-safe bundles and limitations.

## Technical evidence

```powershell
python -m pytest -q
python -m mypy src backend
python scripts\validate_project.py
python scripts\smoke_test.py --base-url http://127.0.0.1:8008
```

For the container segment, run the same smoke script against port `8080`.

## Captured evidence

- `docs/demo/overview-desktop-1440.png`
- `docs/demo/regression-result-1440.png`
- `docs/demo/classification-result-1440.png`
- `docs/demo/batch-result-1440.png`
- `docs/demo/experiments-desktop-1440.png`
- `docs/demo/overview-tablet-768.png`
- `docs/demo/overview-mobile-390.png`

## Suggested closing statement

> PyTorch Tabular Studio is not just two neural networks. It is the complete
> path from leakage-safe data preparation and validation-selected checkpoints
> to hash-validated CPU inference, stable API contracts and an interface that
> communicates model evidence and limitations.
