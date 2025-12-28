# Test ML Service Locally
Write-Host "🚀 Starting ML Service on Port 5000..." -ForegroundColor Green
$env:PORT = "5000"
Set-Location "$PSScriptRoot"
python -m uvicorn app.main:app --host 0.0.0.0 --port 5000
