# Artifact Contract

Each `artifacts/models/<model>/<version>/` directory is immutable after promotion and contains:

- `model_state.pt`;
- `model_config.json`;
- `preprocessing.json`;
- `metrics.json`;
- `training_history.json`;
- `manifest.json`.

The manifest identifies task, model, version, dataset status, input shape, class mapping, seed, device, and selection metric. `state_sha256` must match the weight file before loading. Missing, incompatible, or modified files cause bundle loading to fail closed.

Comparison artifacts name the selection metric and approved model. Fixture and official-UCI comparisons must never be merged into the same evidence table.

