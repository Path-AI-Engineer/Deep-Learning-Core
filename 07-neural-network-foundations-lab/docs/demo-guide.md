# Demonstration Guide

## Engine evidence

1. Run `python scripts/validate_project.py`.
2. Open `artifacts/comparisons/demo-validation-summary.json`.
3. Compare `artifacts/figures/demo-xor-before.svg` and
   `artifacts/figures/demo-xor-after.svg`.
4. Open `artifacts/traces/demo-forward-trace.json` and identify the selected
   neuron's inputs, weights, bias, `z`, activation and gradients.
5. Run `python scripts/compare_with_pytorch.py` and inspect the maximum
   differences.

## Official interface flow

Run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend\app.py
```

The standalone Neural Network Foundations Lab must demonstrate:

1. select XOR;
2. initialize `2 -> 4 -> 1` with tanh, Xavier and seed 7;
3. select one observation;
4. execute forward and inspect a hidden neuron's equation;
5. inspect prediction and Binary Cross-Entropy;
6. execute backward and inspect upstream/local/parameter gradients;
7. apply one SGD update;
8. train a bounded number of epochs;
9. compare initial/current decision boundaries and loss history;
10. run PyTorch parity;
11. reset and reproduce the initial state.

No screenshot, CLI output or Swagger page replaces this interactive flow.
