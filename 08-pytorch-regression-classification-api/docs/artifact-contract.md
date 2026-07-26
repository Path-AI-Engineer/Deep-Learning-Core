# Model Bundle Contract v1

Each approved task bundle contains:

- `model_state.pt`: PyTorch `state_dict` only;
- `model_config.json`: known architecture used to reconstruct the model;
- `preprocessing.json`: fitted training-only `StandardScaler` statistics;
- `feature_schema.json`: ordered names, ranges, descriptions and examples;
- `metrics.json`: test metrics, baseline and primary metric;
- `training_history.json`: train and validation loss curves;
- `metadata.json`: runtime catalog, examples and limitations;
- `manifest.json`: task, version and SHA-256 digest for every payload file.

Classification additionally contains `class_mapping.json` and
`confusion_matrix.json`.

The loader validates completeness, hashes and schema ordering before
constructing the known architecture. It reconstructs preprocessing from
numeric JSON instead of loading an arbitrary pickle, uses
`map_location="cpu"`, switches to evaluation mode and predicts under
`torch.inference_mode()`. User-supplied bundles are not accepted.
