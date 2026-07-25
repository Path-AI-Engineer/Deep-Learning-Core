# Architecture

## Package boundaries

```text
contracts       validated dataset, network and experiment configuration
datasets        deterministic 2D scenarios and metadata
layers          dense transforms and parameter gradients
activations     nonlinear functions and local derivatives
losses          stable scalar objectives and prediction gradients
models          MLP composition and explicit forward/backward passes
optimizers      parameter updates
training        bounded deterministic training
evaluation      metrics, gradient checking and parity
tracing         versioned visual execution contract
serialization   checkpoints and JSON artifacts
experiments     reproducible runner
```

Dependencies point inward toward numerical primitives. The engine never imports
web-framework or application-session state. The Streamlit entry point imports
the public engine package and retains only bounded experiment state.

## Numerical conventions

- Public arrays use shape `(samples, features_or_units)`.
- Targets and binary predictions use `(samples, 1)`.
- Dense weights use `(input_units, output_units)`.
- Dense biases use `(1, output_units)` and broadcast across samples.
- Engine calculations use `numpy.float64`.
- The loss derivative owns batch averaging; dense parameter gradients therefore
  sum the received upstream values without a second averaging step.

## Interface

`frontend/app.py` is an independent presentation entry point inside this
project. It imports `neural_network_foundations`, creates validated
`ExperimentConfig` values and renders engine-owned predictions, traces,
gradients, histories, boundaries and parity results. Numerical formulas remain
in `src`; the interface does not implement a second model.

## Safety

The public configuration limits hidden width, epochs, learning rate, samples
and decision-grid resolution. Arrays are checked for shape, NaN and infinity.
No user-provided Python code is executed.
