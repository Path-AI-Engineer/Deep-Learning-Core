# Architecture Decisions

## ADR-001 — UCI HAR as the target dataset

Accepted. It provides a clear multivariate sequence contract, subject-aware partitions, and six interpretable activities.

## ADR-002 — Offline training and online inference

Accepted. Training scripts create immutable bundles; FastAPI serves only validated artifacts. This keeps request latency and evidence ownership predictable.

## ADR-003 — Fixture evidence is explicit

Accepted. A deterministic fixture makes the repository runnable without network access, but every surface labels it as non-benchmark evidence.

## ADR-004 — RNN, LSTM, and GRU remain separate

Accepted. Shared infrastructure is reused while model identity, metrics, and artifacts remain independently inspectable.

