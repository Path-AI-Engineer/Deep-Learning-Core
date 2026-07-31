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
    -ProjectDirectory "08-pytorch-regression-classification-api" `
    -ImageName "pytorch-tabular-studio" `
    -ServiceName "ai-02-p08-pytorch-tabular-studio" `
    -ServiceAccountName "p02-pytorch-tabular" `
    -ProjectNumber "08" `
    -ImageTag $ImageTag `
    -Region $Region `
    -Memory "2Gi" `
    -HealthPath "/api/v1/health/ready" `
    -PreflightPaths @("frontend/package-lock.json", "artifacts/models") `
    -SkipBuild:$SkipBuild `
    -SmokeOnly:$SmokeOnly `
    -Apply:$Apply
