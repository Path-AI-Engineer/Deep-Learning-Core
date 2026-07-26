# PyTorch Tabular Studio — Interface Contract v1

## Navigation

- Overview: model readiness, task comparison and system boundaries.
- Regression: schema-driven form, prediction, metrics and loss curves.
- Classification: schema-driven form, predicted class, probabilities and confusion matrix.
- Experiments: approved run evidence, model cards and optimizer comparison.
- About: architecture, reproducibility and limitations.

## Required states

Every API-backed view renders loading, success, empty and recoverable error states. Forms
explain units, bounds and feature meaning. Keyboard focus is visible and motion respects
`prefers-reduced-motion`.

## Language and visual direction

Predictions are estimates, never facts. Probabilities are model outputs, not certainty. The
visual language is an engineering studio: graphite surfaces, warm PyTorch orange for actions,
cyan for evidence, restrained typography and lightweight data graphics.
