# Build the AirDesk Windows release (onedir) and a zip in release\.
#   powershell -ExecutionPolicy Bypass -File scripts\build.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".venv\Scripts\python.exe"
& $py -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "tests failed" }
& $py -m PyInstaller AirDesk.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
$ver = & $py -c "import airdesk; print(airdesk.__version__)"
New-Item -ItemType Directory -Force release | Out-Null
$zip = "release\AirDesk-$ver-win64.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path dist\AirDesk -DestinationPath $zip
Write-Host "Built dist\AirDesk\AirDesk.exe and $zip"
