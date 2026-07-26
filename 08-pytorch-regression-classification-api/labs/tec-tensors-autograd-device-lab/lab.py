from __future__ import annotations

import torch

x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
w = torch.tensor([[0.5], [-0.25]], dtype=torch.float32, requires_grad=True)
prediction = x @ w
loss = prediction.square().mean()
loss.backward()

print({"shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device)})
print({"prediction": prediction.item(), "gradient": w.grad.tolist()})
w.grad = None
print({"gradient_reset": w.grad is None})
