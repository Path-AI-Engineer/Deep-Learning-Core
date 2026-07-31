$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
& ".\.venv\Scripts\python.exe" "scripts\validate_project.py"
if ($LASTEXITCODE -ne 0) {
    throw "Project 12 quality gate failed."
}
