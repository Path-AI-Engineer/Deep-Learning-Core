# Engineering Decisions

## ADR-001 — Two tasks, one learning system

Regression and classification share infrastructure but retain task-specific models, losses,
metrics and responses.

## ADR-002 — Offline training

Training is reproducible through scripts and configuration. The public API performs inference
only.

## ADR-003 — State dictionaries over serialized modules

Bundles store `state_dict`, architecture metadata and preprocessing separately so loading is
explicit, inspectable and CPU-compatible.

## ADR-004 — Validation selects; test reports

Early stopping and checkpoint selection use validation loss. Test never influences selection.

## ADR-005 — Small, evidence-oriented MLPs

Architectures are deliberately compact. The project demonstrates a correct lifecycle, not an
exhaustive hyperparameter search.
