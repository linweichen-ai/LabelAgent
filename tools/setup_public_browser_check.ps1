param(
    [string]$Python = "py",
    [string]$TempDir = "",
    [string]$VenvDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "resolve_python312.ps1")
$Requirements = Join-Path $Root "tools\requirements-public-browser-qa.txt"
$RequirementsHash = (Get-FileHash -LiteralPath $Requirements -Algorithm SHA256).Hash.Substring(0, 12).ToLowerInvariant()
$CacheRoot = if ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA "LabelAgent" } else { Join-Path ([IO.Path]::GetTempPath()) "LabelAgent" }
$Venv = if ($VenvDir) { [IO.Path]::GetFullPath($VenvDir) } else { Join-Path $CacheRoot "public-browser-$RequirementsHash" }
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$ResolvedTemp = if ($TempDir) { [IO.Path]::GetFullPath($TempDir) } else { Join-Path ([IO.Path]::GetTempPath()) "LabelAgent-public-browser-$RequirementsHash" }

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

Write-Host "Using short browser-check environment: $Venv"
& $VenvPython -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) { throw "Public browser-check dependency installation failed" }
& $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Public browser-check dependency consistency check failed" }
& $VenvPython tools\verify_synthetic_signoff_browser.py --ephemeral
if ($LASTEXITCODE -ne 0) { throw "Public synthetic signoff browser check failed" }

Write-Host "Public synthetic browser check passed. No model weights, data, Node.js or GPU were used."
