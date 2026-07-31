# Operations

## Local fixture release

```powershell
python scripts\bootstrap_fixture_bundles.py --force
Set-Location frontend
npm install
npm run build
Set-Location ..
$env:PYTHONPATH = "$PWD\src;$PWD\backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8011
```

## Official-data pipeline

```powershell
python scripts\download_data.py
python scripts\prepare_data.py
python scripts\train_model.py --model dense-ae
python scripts\train_model.py --model conv-ae
python scripts\train_model.py --model denoising-ae
python scripts\train_model.py --model latent-2d
python scripts\evaluate_models.py
```

The official workflow writes under `data/processed/fashion-mnist` and
`artifacts/official`; those outputs are ignored by Git until they are deliberately
reviewed and versioned.

## Validation

```powershell
python scripts\validate_project.py
docker build --file infra/docker/production.Dockerfile -t latent-representation-lab:v1.0.0 .
docker run --rm -p 8011:8080 latent-representation-lab:v1.0.0
```

Then request `/api/v1/health` and confirm `status=ready`. Docker commands require an
available Docker engine.
