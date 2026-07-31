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
    -ProjectDirectory "10-sequence-models-rnn-lstm-lab" `
    -ImageName "sequence-memory-lab" `
    -ServiceName "ai-02-p10-sequence-memory-lab" `
    -ServiceAccountName "p02-sequence-memory" `
    -ProjectNumber "10" `
    -ImageTag $ImageTag `
    -Region $Region `
    -Memory "2Gi" `
    -HealthPath "/api/v1/health" `
    -PreflightPaths @("frontend/package-lock.json", "artifacts") `
    -SkipBuild:$SkipBuild `
    -SmokeOnly:$SmokeOnly `
    -Apply:$Apply
