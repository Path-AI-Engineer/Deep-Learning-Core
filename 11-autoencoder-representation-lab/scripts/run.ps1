$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path "frontend\node_modules")) {
    throw "Frontend dependencies are missing. Run 'npm install' inside frontend first."
}

Push-Location "frontend"
try {
    npm run build
    if ($LASTEXITCODE -ne 0) {
        throw "Frontend build failed."
    }
}
finally {
    Pop-Location
}

$env:PYTHONPATH = "$ProjectRoot\src;$ProjectRoot\backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8011
