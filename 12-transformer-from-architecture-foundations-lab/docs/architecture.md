# Architecture

## Decision

The project uses a small encoder-decoder Transformer because every relevant operation
must remain inspectable. The model is assembled from project-local PyTorch modules:

1. separate source and target embeddings;
2. sinusoidal, learned or disabled position modules;
3. projected multi-head scaled dot-product attention;
4. causal and key-padding mask composition;
5. educational LayerNorm with Pre-LN and Post-LN support;
6. position-wise feed-forward blocks;
7. encoder memory and masked autoregressive decoder;
8. vocabulary projection and EOS-bounded greedy decoding.

High-level `torch.nn` Transformer components are excluded by a source-contract test.

## Runtime boundaries

`transformer_lab` owns the ML domain. `backend/app` adapts that domain into HTTP
resources. `frontend` consumes versioned API contracts and never reads artifact files
directly.

The `ModelRegistry` loads one immutable bundle during FastAPI lifespan. Each artifact
file is verified against the SHA-256 values in `manifest.json` before inference.
Incomplete or modified bundles leave the health endpoint degraded and do not silently
fall back to random weights.

## Trace boundary

Traces are opt-in and bounded to batch size one, configured layers and heads. Returned
payloads carry token axes, matrix shape, entropy and an interpretation warning.
Intermediate tensors are detached from the graph and never written as training data.

## Deployment

One multi-stage Docker image builds React, installs the pinned Python runtime, copies
the approved model and sample catalog, and serves both SPA and API through Uvicorn.
This avoids cross-origin configuration and keeps the deployed demonstration
consistent with the verified bundle.
