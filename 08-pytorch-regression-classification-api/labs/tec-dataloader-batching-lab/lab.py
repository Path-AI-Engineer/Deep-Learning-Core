from __future__ import annotations

from pytorch_tabular.data import load_task_data

prepared = load_task_data("classification", seed=42)
features, targets = next(iter(prepared.loaders(batch_size=16).train))
print({"features": list(features.shape), "targets": list(targets.shape)})
print(prepared.split_summary())
