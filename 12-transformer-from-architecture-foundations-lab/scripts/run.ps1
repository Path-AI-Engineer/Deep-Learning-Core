[CmdletBinding()]
param(
    [int]$Port = 8012,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src;$ProjectRoot\backend"
$Python = ".\.venv\Scripts\python.exe"
$VenvReady = (Test-Path $Python) -and
    (Test-Path "$ProjectRoot\.venv\Lib\site-packages\uvicorn")
if (-not $VenvReady) {
    $Python = "python"
}
$arguments = @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$Port")
if ($Reload) {
    $arguments += "--reload"
}
& $Python @arguments
