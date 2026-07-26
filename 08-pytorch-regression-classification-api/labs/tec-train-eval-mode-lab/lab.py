from __future__ import annotations

import torch

from pytorch_tabular.models import RegressionMLP

torch.manual_seed(42)
model = RegressionMLP(8, dropout=0.5)
sample = torch.ones(1, 8)
model.train()
train_outputs = [float(model(sample)) for _ in range(3)]
model.eval()
with torch.inference_mode():
    eval_outputs = [float(model(sample)) for _ in range(3)]
print({"train_outputs": train_outputs, "eval_outputs": eval_outputs})
