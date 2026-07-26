from __future__ import annotations

import torch
from torch import nn

from pytorch_tabular.models import ClassificationMLP

model = ClassificationMLP(input_features=13, class_count=3)
features = torch.randn(4, 13)
target = torch.tensor([0, 1, 2, 1])
logits = model(features)
loss = nn.CrossEntropyLoss()(logits, target)
loss.backward()
print({"logits": list(logits.shape), "loss": float(loss)})
print({"probability_sums": torch.softmax(logits, dim=1).sum(dim=1).tolist()})
