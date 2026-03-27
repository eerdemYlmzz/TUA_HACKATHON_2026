# PowerShell script to run Streamlit app

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Change to script directory
Set-Location $scriptDir

# Activate virtual environment
& ".\env\Scripts\Activate.ps1"

# Run Streamlit
Write-Host "🚀 Starting Streamlit app..." -ForegroundColor Green
Write-Host "📱 Open browser: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""

streamlit run app.py

# Keep window open on error
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error occurred. Press any key to exit..." -ForegroundColor Red
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
