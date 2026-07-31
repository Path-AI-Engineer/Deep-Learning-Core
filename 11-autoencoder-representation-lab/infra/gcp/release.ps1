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
    -ProjectDirectory "11-autoencoder-representation-lab" `
    -ImageName "latent-representation-lab" `
    -ServiceName "ai-02-p11-latent-representation-lab" `
    -ServiceAccountName "p02-latent-representation" `
    -ProjectNumber "11" `
    -ImageTag $ImageTag `
    -Region $Region `
    -Memory "2Gi" `
    -Concurrency 4 `
    -HealthPath "/api/v1/health" `
    -PreflightPaths @("frontend/package-lock.json", "artifacts", "data/samples") `
    -SkipBuild:$SkipBuild `
    -SmokeOnly:$SmokeOnly `
    -Apply:$Apply
