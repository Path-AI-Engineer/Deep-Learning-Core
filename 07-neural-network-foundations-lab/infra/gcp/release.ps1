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
    -ProjectDirectory "07-neural-network-foundations-lab" `
    -ImageName "neural-foundations-lab" `
    -ServiceName "ai-02-p07-neural-foundations-lab" `
    -ServiceAccountName "p02-neural-foundations" `
    -ProjectNumber "07" `
    -ImageTag $ImageTag `
    -Region $Region `
    -Port 8080 `
    -Memory "1Gi" `
    -Concurrency 16 `
    -HealthPath "/_stcore/health" `
    -SkipBuild:$SkipBuild `
    -SmokeOnly:$SmokeOnly `
    -Apply:$Apply
