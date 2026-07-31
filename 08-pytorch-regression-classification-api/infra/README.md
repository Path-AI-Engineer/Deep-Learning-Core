# Infrastructure

Project 08 keeps all delivery assets under this directory:

- `docker/production.Dockerfile`: reproducible application image.
- `gcp/cloudbuild.yaml`: remote image build.
- `gcp/release.ps1`: project-specific Cloud Run release wrapper.

```powershell
docker build --file infra/docker/production.Dockerfile --tag pytorch-tabular-studio:local .
```

A live deployment is only asserted after the release smoke check passes.
