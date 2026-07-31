[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [string]$ImageTag = "v1.0.0",
    [string]$Region = "us-central1",
    [switch]$SkipBuild,
    [switch]$SmokeOnly,
    [switch]$Apply
)

& (Join-Path $PSScriptRoot "..\..\..\infra\gcp\deploy-cloud-run.ps1") `
    -ProjectId $ProjectId `
    -ProjectDirectory "09-cnn-foundations-image-classifier" `
    -ImageName "cnn-vision-lab" `
    -ServiceName "ai-02-p09-cnn-vision-lab" `
    -ServiceAccountName "p02-cnn-vision" `
    -ProjectNumber "09" `
    -ImageTag $ImageTag `
    -Region $Region `
    -Memory "2Gi" `
    -Concurrency 4 `
    -HealthPath "/api/v1/health" `
    -PreflightPaths @(
        "frontend/package-lock.json",
        "artifacts/models/cnn/v1.0.0/model_state.pt",
        "artifacts/models/cnn/v1.0.0/manifest.json",
        "data/raw/FashionMNIST"
    ) `
    -SkipBuild:$SkipBuild `
    -SmokeOnly:$SmokeOnly `
    -Apply:$Apply
