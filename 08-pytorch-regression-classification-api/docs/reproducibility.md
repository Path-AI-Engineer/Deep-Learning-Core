# Reproducibility

Runs record an identifier, UTC timestamp, seed, Git revision when available, dependency
versions, experiment configuration, histories and final metrics. Run directories are never
overwritten.

PyTorch deterministic algorithms are requested when supported. Exact floating-point identity
across operating systems is not promised; acceptance comparisons use documented tolerances.
