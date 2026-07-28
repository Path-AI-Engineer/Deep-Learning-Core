# Engineering decisions

## ADR-001 — FashionMNIST is the only dataset

One stable dataset keeps the project focused on CNN fundamentals instead of
turning it into an ingestion platform. General image classification is out of
scope.

## ADR-002 — Cross-correlation terminology

PyTorch `Conv2d` performs cross-correlation because it does not flip the kernel.
The manual lab names the operation precisely and verifies numerical parity.

## ADR-003 — Fair MLP comparison

The MLP and CNN share the same dataset contract, split seed, batch size,
optimizer family, stopping policy and test isolation. This does not prove that
one architecture is universally superior; it is one controlled comparison.

## ADR-004 — Immutable serving bundle

Serving never imports a training checkpoint blindly. A bundle contains
`state_dict`, reconstruction configuration, preprocessing, metrics, split
manifest, error analysis and SHA-256 hashes.

## ADR-005 — Honest degraded mode

Local UI and the convolution lab remain inspectable before training, but
model-dependent routes explicitly return 503. Production packaging fails if the
approved bundle or official gallery is missing.

## ADR-006 — One purposeful 3D interaction

The overview includes a CSS 3D tensor stack to communicate spatial contraction
and channel growth. It is user-controlled, has semantic buttons, does not
auto-play, disappears on constrained mobile layouts and respects reduced motion.
