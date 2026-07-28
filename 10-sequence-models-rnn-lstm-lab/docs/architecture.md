# Architecture

Sequence Memory Lab separates scientific computation from transport and presentation.

```text
React SPA -> FastAPI routes -> artifact registry -> immutable model bundles
                               |
                               +-> inference, metrics, traces, cell labs

scripts -> data preparation -> training/evaluation -> versioned bundles
```

`src/sequence_models` owns reusable ML behavior. `backend/app` translates HTTP contracts without training or redefining domain rules. `frontend` consumes only `/api/v1`. Training scripts are offline operations and promotion occurs by creating a new bundle version.

The registry verifies weight hashes and loads PyTorch state dictionaries using `weights_only=True`. Requests are stateless and the committed fixture allows local operation without external services.

