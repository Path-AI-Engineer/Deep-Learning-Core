# Threat model

## Assets

- approved model weights and manifest;
- split and task definitions;
- validation metrics and prediction records;
- bounded trace outputs.

## Main threats and controls

| Threat | Control |
|---|---|
| artifact substitution | SHA-256 verification on every required bundle file |
| train/evaluation overlap | canonical-hash disjointness assertion |
| future-token leakage | causal mask tests and explicit mask semantics |
| unknown or malformed input | fixed vocabulary and Pydantic bounds |
| trace payload amplification | batch, layer, head and matrix-size limits |
| fabricated evidence | degraded state when bundle is unavailable |
| overclaiming | evidence status and limits rendered in product UI |

The application does not accept model uploads or arbitrary code execution. Cloud Run
uses an isolated project service account and scales the stateless API independently.
