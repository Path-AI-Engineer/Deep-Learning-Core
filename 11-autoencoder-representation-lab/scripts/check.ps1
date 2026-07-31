$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
python scripts\validate_project.py
if ($LASTEXITCODE -ne 0) {
    throw "Project 11 quality gate failed."
}
