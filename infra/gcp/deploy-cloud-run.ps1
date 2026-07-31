[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[a-z][a-z0-9-]{4,28}[a-z0-9]$")]
    [string]$ProjectId,

    [Parameter(Mandatory = $true)]
    [string]$ProjectDirectory,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[a-z][a-z0-9-]{2,62}$")]
    [string]$ImageName,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[a-z][a-z0-9-]{2,62}$")]
    [string]$ServiceName,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[a-z][a-z0-9-]{2,28}$")]
    [string]$ServiceAccountName,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^\d{2}$")]
    [string]$ProjectNumber,

    [string]$ImageTag = "v1.0.0",
    [string]$Repository = "plan-02",
    [string]$Region = "us-central1",
    [int]$Port = 8080,
    [string]$Memory = "2Gi",
    [int]$Concurrency = 8,
    [int]$TimeoutSeconds = 300,
    [string]$HealthPath = "/api/v1/health",
    [string[]]$PreflightPaths = @(),
    [switch]$SkipBuild,
    [switch]$SmokeOnly,
    [switch]$Apply
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$projectRoot = Join-Path $repositoryRoot $ProjectDirectory
$dockerfile = Join-Path $projectRoot "infra\docker\production.Dockerfile"
$cloudBuildConfig = Join-Path $projectRoot "infra\gcp\cloudbuild.yaml"
$remoteImage = "$Region-docker.pkg.dev/$ProjectId/$Repository/$ImageName`:$ImageTag"
$serviceAccount = "$ServiceAccountName@$ProjectId.iam.gserviceaccount.com"

function Resolve-Tool {
    param([Parameter(Mandatory = $true)][string[]]$Candidates)
    foreach ($candidate in $Candidates) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }
    throw "$($Candidates -join ' or ') was not found."
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Command $($Arguments -join ' ')"
    }
}

function Read-GCloudValue {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $value = & $script:GCloud @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "gcloud query failed: gcloud $($Arguments -join ' ')"
    }
    return ($value | Out-String).Trim()
}

function Wait-Http {
    param([Parameter(Mandatory = $true)][string]$Uri)
    for ($attempt = 1; $attempt -le 30; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 20
            if ($response.StatusCode -eq 200) {
                Write-Host "Smoke passed: $Uri"
                return
            }
        }
        catch {
            if ($attempt -eq 30) {
                throw "Service did not become healthy at $Uri. $($_.Exception.Message)"
            }
        }
        Start-Sleep -Seconds 5
    }
}

if (-not (Test-Path $projectRoot -PathType Container)) {
    throw "Project directory was not found: $projectRoot"
}

$script:GCloud = Resolve-Tool @("gcloud.cmd", "gcloud")

if ($SmokeOnly) {
    $serviceUrl = Read-GCloudValue @(
        "run", "services", "describe", $ServiceName,
        "--project=$ProjectId",
        "--region=$Region",
        "--format=value(status.url)"
    )
    Wait-Http -Uri "$serviceUrl$HealthPath"
    Write-Host "Service URL: $serviceUrl"
    exit 0
}

if (-not $Apply) {
    Write-Host "Plan only:"
    Write-Host "  Build context: $projectRoot"
    Write-Host "  Dockerfile:    $dockerfile"
    Write-Host "  Image:         $remoteImage"
    Write-Host "  Service:       $ServiceName"
    Write-Host "  Health:        $HealthPath"
    Write-Host "No GCP resources were changed. Re-run with -Apply."
    exit 0
}

if (-not (Test-Path $dockerfile -PathType Leaf)) {
    throw "Production Dockerfile is missing from $dockerfile"
}
if (-not (Test-Path $cloudBuildConfig -PathType Leaf)) {
    throw "Cloud Build configuration is missing from $cloudBuildConfig"
}
foreach ($relativePath in $PreflightPaths) {
    $requiredPath = Join-Path $projectRoot $relativePath
    if (-not (Test-Path $requiredPath)) {
        throw "Required release input is missing: $relativePath"
    }
}

Invoke-Checked -Command $script:GCloud -Arguments @(
    "services", "enable",
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "iam.googleapis.com",
    "--project=$ProjectId",
    "--quiet"
)

$repositoryExists = & $script:GCloud artifacts repositories describe $Repository `
    "--project=$ProjectId" "--location=$Region" "--format=value(name)" 2>$null
if ($LASTEXITCODE -ne 0 -or -not $repositoryExists) {
    Invoke-Checked -Command $script:GCloud -Arguments @(
        "artifacts", "repositories", "create", $Repository,
        "--project=$ProjectId",
        "--location=$Region",
        "--repository-format=docker",
        "--description=AI Engineer Plan 02 release images",
        "--labels=path=ai-engineer,plan=02,environment=production",
        "--quiet"
    )
}

$accountExists = & $script:GCloud iam service-accounts describe $serviceAccount `
    "--project=$ProjectId" "--format=value(email)" 2>$null
if ($LASTEXITCODE -ne 0 -or -not $accountExists) {
    Invoke-Checked -Command $script:GCloud -Arguments @(
        "iam", "service-accounts", "create", $ServiceAccountName,
        "--project=$ProjectId",
        "--display-name=AI Plan 02 Project $ProjectNumber runtime",
        "--quiet"
    )
}

if (-not $SkipBuild) {
    Push-Location $projectRoot
    try {
        Invoke-Checked -Command $script:GCloud -Arguments @(
            "builds", "submit",
            "--project=$ProjectId",
            "--region=$Region",
            "--config=$cloudBuildConfig",
            "--substitutions=_IMAGE=$remoteImage",
            "--quiet",
            "."
        )
    }
    finally {
        Pop-Location
    }
}

Invoke-Checked -Command $script:GCloud -Arguments @(
    "run", "deploy", $ServiceName,
    "--project=$ProjectId",
    "--region=$Region",
    "--image=$remoteImage",
    "--service-account=$serviceAccount",
    "--allow-unauthenticated",
    "--port=$Port",
    "--cpu=1",
    "--memory=$Memory",
    "--concurrency=$Concurrency",
    "--timeout=$TimeoutSeconds",
    "--min=0",
    "--max=1",
    "--cpu-throttling",
    "--no-cpu-boost",
    "--labels=path=ai-engineer,plan=02,project=$ProjectNumber,environment=production",
    "--quiet"
)

$url = Read-GCloudValue @(
    "run", "services", "describe", $ServiceName,
    "--project=$ProjectId",
    "--region=$Region",
    "--format=value(status.url)"
)
Wait-Http -Uri "$url$HealthPath"

Write-Host "Deployment verified."
Write-Host "Service URL: $url"
Write-Host "Image:       $remoteImage"
