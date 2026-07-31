# API reference

All product endpoints are versioned under `/api/v1`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Runtime and bundle readiness |
| GET | `/model-card` | Scope, model and limitations |
| GET | `/models` | Model comparison rows |
| GET | `/classes` | Class mapping |
| GET | `/samples` | Filterable deterministic gallery |
| GET | `/samples/{sample_id}` | Sample evidence |
| POST | `/reconstruct/sample` | Reconstruct a gallery sample |
| POST | `/reconstruct/upload` | Reconstruct an ephemeral upload |
| POST | `/denoise` | Apply corruption and reconstruct |
| GET | `/latent/points` | Dedicated 2D latent coordinates |
| GET | `/latent/sample/{sample_id}` | Point, reconstruction and neighbours |
| POST | `/latent/decode` | Decode a bounded 2D coordinate |
| POST | `/latent/interpolate` | Interpolate between gallery samples |
| GET | `/evaluation/summary` | Cross-model evidence |
| GET | `/evaluation/model/{model_id}` | One model's metrics |
| GET | `/evaluation/errors` | Highest reconstruction errors |

Validation failures return HTTP 422 with a bounded detail message. Unknown resources
return HTTP 404. Uploaded files are validated by media type, byte limit and decodable
image content and are not persisted.
