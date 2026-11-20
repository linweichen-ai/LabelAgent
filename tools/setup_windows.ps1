param(
    [string]$Python = "py",
    [string]$TempDir = "",
    [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$ResolvedTemp = if ($TempDir) { [IO.Path]::GetFullPath($TempDir) } else { Join-Path $Root ".repro\setup-temp" }

Set-Location $Root
New-Item -ItemType Directory -Force -Path $ResolvedTemp | Out-Null
$env:TEMP = $ResolvedTemp
$env:TMP = $ResolvedTemp
$env:PYTHONUTF8 = "1"
if (-not (Test-Path $VenvPython)) {
    if ($Python -eq "py") {
        & py -3.12 -m venv $Venv
    } else {
        & $Python -m venv $Venv
    }
    if ($LASTEXITCODE -ne 0) { throw "Python 3.12 virtual environment creation failed" }
}

& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

if (-not $SkipTorch) {
    & $VenvPython -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed" }
}

& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "LabelAgent dependency installation failed" }
& $VenvPython -m pip install -e .\source\sam3
if ($LASTEXITCODE -ne 0) { throw "SAM3 editable installation failed" }

& $VenvPython tools\repro_doctor.py --require-cuda
if ($LASTEXITCODE -ne 0) { throw "Environment self-check failed" }

Write-Host "Environment ready. Place sam3.pt in source\sam3\weights, then run tools\start_local.ps1."
