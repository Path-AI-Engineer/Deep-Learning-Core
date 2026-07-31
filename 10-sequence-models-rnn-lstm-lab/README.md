# Sequence Memory Lab

An inspectable sequence-classification workspace for comparing vanilla RNN, LSTM, and GRU models on multivariate sensor windows. It joins reproducible PyTorch experiments, educational recurrent-cell labs, versioned inference bundles, a FastAPI service, and a six-view React interface.

> **Evidence boundary:** the committed `v1.0.0` bundles use a deterministic, HAR-shaped educational fixture so the complete system works offline. They are not UCI HAR benchmark results. Run the official data workflow before making dataset-performance claims.

## What the project demonstrates

- many-to-one classification over tensors shaped `[batch, 128, 9]`;
- grouped train/validation splitting and the untouched official UCI test split;
- normalization fitted only on the training partition;
- temporal-statistics MLP, RNN, LSTM, and GRU comparison;
- BPTT gradient inspection, clipping, padding, and packed sequences;
- educational recurrent-cell parity against PyTorch primitives;
- immutable bundles with configuration, metrics, preprocessing, weights, and hashes;
- a 14-operation REST contract and a responsive product interface.

## Product routes

| Route | Purpose |
|---|---|
| `/` | Project status, evidence boundary, and active model |
| `/classify` | Classify an inspectable 128-step sensor sequence |
| `/sequence-lab` | Explore hidden states, gates, and gradient flow |
| `/compare` | Compare RNN, LSTM, and GRU evidence |
| `/evaluation` | Inspect confusion matrix and class-level metrics |
| `/about` | Review architecture, dataset contract, and limitations |

## Run locally

```powershell
Set-Location "C:\JeanLoa\Path-AI-Engineer\Deep-Learning-Core\10-sequence-models-rnn-lstm-lab"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
Set-Location frontend
npm install
npm run build
Set-Location ..
python scripts\bootstrap_fixture_bundles.py --force
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8010
```

Open `http://127.0.0.1:8010`. FastAPI serves both `/api/v1/*` and the compiled SPA.

## Official UCI HAR workflow

```powershell
python scripts\download_data.py
python scripts\prepare_data.py
python scripts\train_baseline.py
python scripts\train_model.py --model rnn
python scripts\train_model.py --model lstm
python scripts\train_model.py --model gru
python scripts\compare_models.py
python scripts\evaluate_model.py --model gru
```

The preparation command records provenance, subjects, split sizes, and training-only normalization statistics. See `docs/data-contract.md` before producing benchmark evidence.

## Quality gates

```powershell
$env:RUFF_NO_CACHE = "true"
python -m ruff check src backend scripts tests
python -m mypy src backend
python -m pytest -q
python scripts\validate_project.py
Set-Location frontend
npm run build
```

## Architecture

```text
frontend/                    React product interface
backend/app/                 FastAPI transport and artifact registry
src/sequence_models/         Data, models, cells, training, evaluation, inference
artifacts/models/            Immutable, versioned inference bundles
artifacts/comparisons/       Reproducible model-selection evidence
configs/                     Dataset, model, and experiment configuration
scripts/                     Download, preparation, training, evaluation, validation
labs/                        Guided technical experiments
docs/                        Contracts, decisions, limitations, and model evidence
tests/                       ML, API, and OpenAPI verification
```

The API never trains a model during a request. It loads validated bundles with `weights_only=True`, verifies SHA-256 hashes, and exposes deterministic inference and inspection endpoints.

## Docker

```powershell
docker build --file infra/docker/production.Dockerfile -t sequence-memory-lab:v1.0.0 .
docker run --rm -p 8080:8080 sequence-memory-lab:v1.0.0
```

### Cloud Run release package

```powershell
.\infra\gcp\release.ps1 -ProjectId "jeanloa-ai-engineer"
```

The script is non-mutating without `-Apply`. It builds the immutable image with
Cloud Build, publishes it to Artifact Registry `plan-02`, deploys
`ai-02-p10-sequence-memory-lab` with scale-to-zero limits and verifies the
versioned API health contract.

## Documentation

- [Architecture](docs/architecture.md)
- [Dataset contract](docs/data-contract.md)
- [Sequence contract](docs/sequence-contract.md)
- [Training contract](docs/training-contract.md)
- [Artifact contract](docs/artifact-contract.md)
- [API contract](docs/api-contract.md)
- [Model card](docs/model-card.md)
- [Reproducibility](docs/reproducibility.md)
- [Error analysis](docs/error-analysis.md)

## Release

`v1.0.0` is the first portfolio release of Sequence Memory Lab. An official UCI-trained release must use a new artifact version and retain the original split and provenance evidence.
