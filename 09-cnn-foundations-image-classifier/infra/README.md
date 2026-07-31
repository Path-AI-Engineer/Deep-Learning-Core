# Infrastructure

Project 09 keeps all delivery assets under this directory:

- `docker/production.Dockerfile`: reproducible application image.
- `gcp/cloudbuild.yaml`: remote image build.
- `gcp/release.ps1`: project-specific Cloud Run release wrapper.

```powershell
docker build --file infra/docker/production.Dockerfile --tag cnn-vision-lab:local .
```

The production build intentionally requires the approved model bundle.
