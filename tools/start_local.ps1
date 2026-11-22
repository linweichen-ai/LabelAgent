param(
    [string]$SamUrl = "http://127.0.0.1:8766",
    [string]$WorkDir = "",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Runtime = Join-Path $Root ".repro"
$Logs = Join-Path $Runtime "logs"
$Record = Join-Path $Runtime "processes.json"

if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run: powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1"
}
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$RuntimeTemp = Join-Path $Runtime "temp"
New-Item -ItemType Directory -Force -Path $RuntimeTemp | Out-Null

if (Test-Path $Record) {
    $old = Get-Content -Raw $Record | ConvertFrom-Json
    $alive = @($old.pids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
    if ($alive.Count -gt 0) { throw "LabelAgent is already managed by this checkout. Run tools\stop_local.ps1 first." }
}

$env:PYTHONUTF8 = "1"
$env:TEMP = $RuntimeTemp
$env:TMP = $RuntimeTemp
if ($WorkDir) { $env:LABELAGENT_WORK_DIR = [IO.Path]::GetFullPath($WorkDir) }
$env:SAM3_URL = $SamUrl
$pids = @()

$doctorArgs = @("tools\repro_doctor.py", "--require-web-port")
if ($SamUrl -eq "http://127.0.0.1:8766") {
    $doctorArgs += @("--require-cuda", "--require-weight", "--require-sam-port")
}
& $Python @doctorArgs | Write-Host
if ($LASTEXITCODE -ne 0) { throw "LabelAgent startup prerequisites are incomplete" }

if ($SamUrl -eq "http://127.0.0.1:8766") {
    $sam = Start-Process -FilePath $Python -ArgumentList "-m", "source.sam3_service.server" `
        -WorkingDirectory $Root -RedirectStandardOutput (Join-Path $Logs "sam3.out.log") `
        -RedirectStandardError (Join-Path $Logs "sam3.err.log") -PassThru -WindowStyle Hidden
    $pids += $sam.Id
} else {
    try {
        $null = Invoke-RestMethod -Uri ($SamUrl.TrimEnd('/') + "/health") -TimeoutSec 5
    } catch {
        throw "Configured SAM3 service is unavailable: $SamUrl"
    }
}

$web = Start-Process -FilePath $Python -ArgumentList "-m", "source.agent.server" `
    -WorkingDirectory $Root -RedirectStandardOutput (Join-Path $Logs "workbench.out.log") `
    -RedirectStandardError (Join-Path $Logs "workbench.err.log") -PassThru -WindowStyle Hidden
$pids += $web.Id

@{ pids = $pids; started_at = (Get-Date).ToString("o"); sam_url = $SamUrl; root = $Root } |
    ConvertTo-Json | Set-Content -Encoding UTF8 $Record

$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/run/status" -TimeoutSec 2
        $ready = $true
        break
    } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $ready) {
    foreach ($processId in $pids) {
        Stop-Process -Id $processId -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $Record -Force -ErrorAction SilentlyContinue
    throw "Workbench did not become ready. Check .repro\logs\workbench.err.log"
}

& $Python tools\repro_doctor.py --require-services --sam-url $SamUrl
if ($LASTEXITCODE -ne 0) {
    throw "Services started but failed the read-back check"
}

if (-not $NoBrowser) { Start-Process "http://127.0.0.1:8765" }
Write-Host "LabelAgent ready: http://127.0.0.1:8765"
Write-Host "Logs: $Logs"
