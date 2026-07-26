# Project 08 Closeout

## Delivery status

| Global day | Delivery | Evidence |
|---:|---|---|
| 211 | PyTorch lifecycle and contracts | `docs/`, tensor/autograd lab |
| 212 | Dataset contracts and leakage-safe splits | `src/pytorch_tabular/data/` |
| 213 | Dataset and DataLoader integration | data tests and batching lab |
| 214 | Shared tensor/device foundations | utilities and tensor lab |
| 215 | Regression MLP | regression model and tests |
| 216 | Multiclass logits model | classification model and logits lab |
| 217 | Shared trainer and validation loop | training module and integration tests |
| 218 | Generalization protocol | training and reproducibility contracts |
| 219 | Regression experiment | approved regression bundle and model card |
| 220 | Classification experiment | approved classification bundle and model card |
| 221 | Optimizer and regularization comparison | optimizer lab and report |
| 222 | Reproducible experiment runner | immutable run IDs and evidence |
| 223 | CPU-safe inference bundles | JSON preprocessing, hashes and reload tests |
| 224 | Versioned FastAPI | health, catalog, schema, model card and inference |
| 225 | Interface contract | `docs/interface-contract.md` |
| 226 | React shell and typed API client | frontend shell, router and states |
| 227 | Regression Studio | schema form, result, metrics and curve |
| 228 | Classification Studio | class probabilities, metrics and curve |
| 229 | Batch and experiments | CSV validation, 100-row limit and download |
| 230 | Integrated quality and Docker | production build, tests and Dockerfile |
| 231 | Demo and release preparation | screenshots, smoke test and release notes |

## Verified gates

- Python: `18 passed`.
- Static typing: `mypy` reports no issues in 31 source files.
- Frontend: TypeScript/Vite production build passed.
- Frontend runtime tests: CSV validation and accessible error state passed.
- Bundles: regression and classification manifests and hashes passed.
- API smoke: health, both single predictions and batch inference passed.
- Browser: no console warnings/errors and no failed API requests.
- Responsive: no horizontal overflow at 1440, 768 or 390 px.
- Keyboard: skip link is the first focus target.
- Reduced motion: CSS disables animation, transition and smooth scrolling.

## Environment-specific limits

- Docker source is complete, but the local Docker Desktop Linux engine was not
  available during final validation, so no container result is claimed.
- The complete California Housing download was unavailable; the checked-in
  regression bundle transparently identifies its official-source reference
  sample and must not be presented as a full benchmark.
- Git metadata is outside this sandbox's writable boundary. No commit or tag is
  claimed in this closeout.
