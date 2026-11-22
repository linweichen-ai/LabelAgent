param(
    [string]$Python = "py",
    [string]$TempDir = "",
    [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "resolve_python312.ps1")
$Venv = Join-Path $Root ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$ResolvedTemp = if ($TempDir) { [IO.Path]::GetFullPath($TempDir) } else { Join-Path $Root ".repro\setup-temp" }

Set-Location $Root
New-Item -ItemType Directory -Force -Path $ResolvedTemp | Out-Null
$env:TEMP = $ResolvedTemp
$env:TMP = $ResolvedTemp
$env:PYTHONUTF8 = "1"
$free = (Get-Item $ResolvedTemp).PSDrive.Free
$freshVenv = -not (Test-Path $VenvPython)
if ($freshVenv -and -not $SkipTorch -and $free -lt 8GB) {
    throw "A fresh CUDA environment needs at least 8 GiB free space: $ResolvedTemp"
}
if ($freshVenv -and -not $SkipTorch) {
    Write-Host "Preparing the full CUDA environment: about 2.9 GB will be downloaded and the final venv is about 5.1 GiB."
    Write-Host "The PyTorch download can be quiet for several minutes. For a quick no-GPU check, run tools\setup_public_browser_check.ps1 instead."
}
if (-not (Test-Path $VenvPython)) {
    $BootstrapPython = Resolve-LabelAgentPython312 -Requested $Python
    & $BootstrapPython -m venv $Venv
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
& $VenvPython -m pip install -r tools\requirements-browser-qa.txt
if ($LASTEXITCODE -ne 0) { throw "Browser acceptance dependency installation failed" }
& $VenvPython -m pip install -e .\source\sam3
if ($LASTEXITCODE -ne 0) { throw "SAM3 editable installation failed" }

& $VenvPython tools\repro_doctor.py --require-cuda
if ($LASTEXITCODE -ne 0) { throw "Environment self-check failed" }

Write-Host "Environment ready. Place sam3.pt in source\sam3\weights, then run tools\start_local.ps1."
