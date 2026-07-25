# NumPy and PyTorch Parity

PyTorch is an independent reference, not the implementation used by the visual
lab. The parity check creates an equivalent two-layer `torch.nn` model, copies
the exact NumPy parameters and compares:

- forward predictions;
- scalar loss;
- gradients for every weight and bias;
- gradients with respect to input features;
- one SGD update.

Both implementations use `float64` and a default absolute tolerance of `1e-8`.
The report records maximum absolute differences per parameter. A successful
comparison verifies numerical equivalence for the tested architecture and
inputs; it is not a claim that the manual engine implements every PyTorch
feature.

Run:

```powershell
python scripts/compare_with_pytorch.py
```
