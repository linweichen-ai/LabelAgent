param(
    [string]$Python = "py",
    [string]$TempDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "resolve_python312.ps1")
$Venv = Join-Path $Root ".repro\public-browser-venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$ResolvedTemp = if ($TempDir) { [IO.Path]::GetFullPath($TempDir) } else { Join-Path $Root ".repro\public-browser-temp" }

Set-Location $Root
New-Item -ItemType Directory -Force -Path $ResolvedTemp | Out-Null
$env:TEMP = $ResolvedTemp
$env:TMP = $ResolvedTemp
$env:PYTHONUTF8 = "1"
$free = (Get-Item $ResolvedTemp).PSDrive.Free
if ($free -lt 512MB) {
    throw "At least 512 MiB free space is required for the public browser check: $ResolvedTemp"
}

if (-not (Test-Path $VenvPython)) {
    $BootstrapPython = Resolve-LabelAgentPython312 -Requested $Python
    & $BootstrapPython -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "Python 3.12 browser-check virtual environment creation failed" }
}

& $VenvPython -m pip install -r tools\requirements-public-browser-qa.txt
if ($LASTEXITCODE -ne 0) { throw "Public browser-check dependency installation failed" }
& $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Public browser-check dependency consistency check failed" }
& $VenvPython tools\verify_synthetic_signoff_browser.py --ephemeral
if ($LASTEXITCODE -ne 0) { throw "Public synthetic signoff browser check failed" }

Write-Host "Public synthetic browser check passed. No model weights, data, Node.js or GPU were used."
