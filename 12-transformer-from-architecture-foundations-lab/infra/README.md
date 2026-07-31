# Infrastructure

Project 12 keeps all delivery assets under this directory:

- `docker/production.Dockerfile`: reproducible application image.
- `gcp/cloudbuild.yaml`: remote image build.
- `gcp/release.ps1`: project-specific Cloud Run release wrapper.

```powershell
docker build --file infra/docker/production.Dockerfile --tag transformer-architecture-lab:local .
```

A live deployment is only asserted after the release smoke check passes.
