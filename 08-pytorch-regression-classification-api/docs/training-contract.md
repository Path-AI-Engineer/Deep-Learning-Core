# Training Contract

1. Seed Python, NumPy and PyTorch before splitting or model construction.
2. Fit preprocessing only on training observations.
3. Use `model.train()` while optimizing and `model.eval()` plus
   `torch.inference_mode()` while validating.
4. Regression uses `MSELoss`; classification uses `CrossEntropyLoss` on raw logits.
5. Select the best checkpoint by validation loss with bounded early stopping.
6. Restore the selected checkpoint before the single final test evaluation.
7. Compare MAE against `DummyRegressor` and macro-F1 against `DummyClassifier`.
8. Record configuration, environment, history, metrics and dataset fingerprint per run.

Underfitting is poor train and validation performance. Overfitting is improving train loss
while validation degrades beyond configured patience.
