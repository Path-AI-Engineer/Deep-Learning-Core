# API Contract v1

Base path: `/api/v1`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Aggregate API version and model readiness |
| GET | `/health/live` | Process liveness |
| GET | `/health/ready` | Approved bundle readiness |
| GET | `/tasks` | Task catalog and batch limit |
| GET | `/tasks/{task}/schema` | Ordered feature schema and examples |
| GET | `/tasks/{task}/model-card` | Metrics, architecture and limitations |
| POST | `/predictions/{task}` | One prediction |
| POST | `/predictions/{task}/batch` | 1–100 predictions |

Unknown tasks return 404, invalid features return 422, oversized batches
return 413 and unavailable models return 503. No response includes local paths,
secrets or internal stack details.
