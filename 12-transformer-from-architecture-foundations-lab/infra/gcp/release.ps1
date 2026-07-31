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
    -ProjectDirectory "12-transformer-from-architecture-foundations-lab" `
    -ImageName "transformer-architecture-lab" `
    -ServiceName "ai-02-p12-transformer-architecture-lab" `
    -ServiceAccountName "p02-transformer-lab" `
    -ProjectNumber "12" `
    -ImageTag $ImageTag `
    -Region $Region `
    -Memory "2Gi" `
    -Concurrency 4 `
    -HealthPath "/api/v1/health" `
    -PreflightPaths @("frontend/package-lock.json", "artifacts/models/transformer/v1.0.0-reference", "data/samples/demo_catalog.json") `
    -SkipBuild:$SkipBuild `
    -SmokeOnly:$SmokeOnly `
    -Apply:$Apply
