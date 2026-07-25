# Experiment Contract

## Configuration

```json
{
  "dataset": "xor",
  "network": {
    "input_features": 2,
    "hidden_units": 4,
    "output_units": 1,
    "hidden_activation": "tanh",
    "output_activation": "sigmoid",
    "initialization": "xavier"
  },
  "loss": "binary_cross_entropy",
  "learning_rate": 0.5,
  "epochs": 3000,
  "seed": 7,
  "grid_resolution": 40
}
```

Invalid architecture, activation, objective or limit values are rejected before
parameters are allocated.

## Reproducibility

An experiment records the complete validated configuration, package version,
dataset metadata, seed, run identifier and generated artifact paths. A repeated
configuration and seed must reproduce initial parameters and metric history
within the documented floating-point tolerance.

## Planned output

- initial and final parameters;
- predictions, loss and binary accuracy;
- training histories;
- selected forward/backward trace;
- decision-grid predictions;
- checkpoint path;
- optional PyTorch parity report;
- diagnostic warnings.
