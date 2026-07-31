# API contract

All public routes are versioned under `/api/v1`.

| Method | Route | Responsibility |
|---|---|---|
| GET | `/health` | registry and active model status |
| GET | `/model-card` | purpose, architecture, metrics and limits |
| GET | `/architecture` | components, tensor shapes and trace bounds |
| GET | `/tasks` | controlled task contracts |
| GET | `/tokens` | fixed vocabulary and input rules |
| GET | `/models` | active immutable model |
| GET | `/samples` | bounded, filterable demonstration catalog |
| POST | `/predict` | greedy EOS-bounded transduction |
| POST | `/trace` | one bounded attention trace |
| POST | `/attention/compute` | educational Q/K/V calculation |
| GET | `/experiments` | registered validation runs |
| GET | `/evaluation/summary` | ID/OOD and cost evidence |
| GET | `/evaluation/by-length` | disaggregated length evidence |
| GET | `/evaluation/errors` | recorded sequence mismatches |
| GET | `/research` | research questions and validity threats |

Unknown tokens, incompatible recall examples, oversized matrices and unsupported trace
coordinates are rejected. The API never invents a default prediction when the bundle
is absent.
