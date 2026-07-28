# Reproducibility

## Environment

- Python 3.12
- PyTorch 2.9.0
- torchvision 0.24.0
- CPU is the default and release compatibility target.

Create a virtual environment and install `requirements.txt` plus
`requirements-dev.txt`. The exact project and model configuration lives under
`configs/`.

## Build the evidence

```powershell
python scripts/prepare_data.py
python scripts/train_mlp_baseline.py
python scripts/train_cnn.py
```

Each training command prints its immutable run directory. Use both paths:

```powershell
python scripts/build_model_bundle.py `
  --mlp-run artifacts/runs/mlp-baseline/<run-id> `
  --cnn-run artifacts/runs/cnn-base/<run-id> `
  --version v1.0.0
```

Then run:

```powershell
python scripts/smoke_test.py
```

The experiment record includes the seed, split indices, runtime environment,
configuration, Git commit, history and isolated test output. The release bundle
adds hashes for every serving document and weight file.

## Non-determinism note

Seeds and deterministic algorithms reduce variation, but exact floating-point
results can still depend on PyTorch, hardware and underlying libraries. Record
the environment instead of claiming bit-for-bit portability across platforms.
