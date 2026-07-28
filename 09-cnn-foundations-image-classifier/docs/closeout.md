# Closeout checklist

## Implemented

- [x] Strict dataset, model and experiment contracts.
- [x] Manual cross-correlation and PyTorch parity endpoint.
- [x] Output-shape and receptive-field utilities.
- [x] MLP baseline and two-block CNN.
- [x] Train-only augmentation and deterministic validation split.
- [x] Validation checkpoint selection and isolated test path.
- [x] Accuracy, macro F1, per-class metrics, confusion matrix and errors.
- [x] Hash-verified CPU bundle contract.
- [x] Memory-only PNG/JPEG preprocessing.
- [x] Whitelisted activation capture with hook cleanup.
- [x] Versioned FastAPI routes and honest degraded mode.
- [x] Responsive React laboratory with six routes and purposeful 3D tensor view.
- [x] Offline unit and contract tests.

## Pending environment evidence

- [ ] Official FashionMNIST download is available locally.
- [ ] MLP and CNN were trained in this checkout.
- [ ] Numeric model evidence was generated from the isolated test set.
- [ ] `v1.0.0` serving bundle and controlled gallery are present.
- [ ] Frontend dependencies were installed and production build passed.
- [ ] Production Docker image passed health and inference smoke tests.
- [ ] Desktop, tablet and mobile browser QA passed against the real app.

Project 09 must not be called fully completed while any pending item remains.

## Validation evidence — 2026-07-28

- `python -m pytest -q -p no:cacheprovider`: **33 passed**.
- `python -m ruff check . --no-cache`: **passed**.
- `python -m mypy src backend scripts --cache-dir=NUL`: **49 files passed**.
- `python scripts/validate_project.py`: **12 OpenAPI paths validated**.
- React source: TypeScript check passed against the already-installed compatible
  Project 08 toolchain without modifying Project 08.
- React source: a production-style minified bundle was generated successfully
  with esbuild for local runtime inspection.
- FastAPI + SPA: started on port 8009; `/api/v1/health` returned the expected
  honest `degraded` state.

## Exact blockers

- The official FashionMNIST files are not cached in this environment and network
  download was unavailable. Therefore no honest MLP/CNN run, metric, gallery or
  serving bundle can be generated here.
- The Project 09 npm dependency tree is not installed. The source was checked
  with a compatible existing toolchain, but `npm install` and the native Vite
  production command remain to be executed from a network-enabled terminal.
- Docker Desktop is not currently available to this execution environment, and
  the release Dockerfile correctly requires the missing official assets.
- In-app browser policy rejected localhost:8009, so responsive visual screenshots
  could not be captured in this run. No alternate browser mechanism was used.
