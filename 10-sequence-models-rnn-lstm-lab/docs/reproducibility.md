# Reproducibility

- Python: 3.12
- PyTorch: 2.9.0 CPU
- random seed: 42
- deterministic algorithms: enabled
- dependency versions: pinned in requirements files and frontend lockfile
- bundles: versioned and SHA-256 verified

For an official run, retain the download and preparation manifests alongside the bundle. Run all quality gates, rebuild the frontend, and record the comparison artifact. CPU and accelerator kernels can differ; a new environment or dependency set requires new evidence.

