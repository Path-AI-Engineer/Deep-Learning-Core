# Architecture and Learning Decisions

## ADR-001 — NumPy before PyTorch

**Decision:** the official learning engine is implemented and mathematically
validated with NumPy before the equivalent PyTorch model is introduced.

**Reason:** framework automation must verify understanding, not replace it.

## ADR-002 — One official interface

**Decision:** Project 07 owns one standalone Streamlit interface in
`frontend/app.py`. It consumes the same importable package exercised by tests
and command-line experiments.

**Reason:** the AI Engineer project must remain independently demonstrable
without duplicating neural mathematics or depending on another repository.

## ADR-003 — Small deterministic MLP

**Decision:** the primary architecture has two inputs, one bounded hidden layer
and one sigmoid output. Every experiment records a seed and uses `float64`.

**Reason:** small tensors are inspectable and `float64` provides a reliable
reference for finite-difference and PyTorch parity checks.

## ADR-004 — Trace as a public contract

**Decision:** forward and backward values cross the presentation boundary
through a versioned, JSON-serializable trace.

**Reason:** the interface can evolve without importing private engine objects
or recomputing mathematics.

## ADR-005 — Explicit safety limits

**Decision:** architecture, epochs, learning rate, dataset size and grid
resolution are bounded.

**Reason:** the lab is an explanatory CPU application, not an arbitrary model
training service.

## ADR-006 — Evidence beyond decreasing loss

**Decision:** project closure requires finite-difference gradient checks,
checkpoint reconstruction and PyTorch parity.

**Reason:** a decreasing loss can coexist with incorrect gradients,
non-reproducible behavior or an implementation that only works accidentally.
