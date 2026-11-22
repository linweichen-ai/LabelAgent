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
    if ($alive.Count -gt 0) { throw "A live PID is present in this checkout's process record. Run tools\stop_local.ps1; if it reports a stale identity, re-run it with -ArchiveStaleRecord." }
}

$env:PYTHONUTF8 = "1"
$env:TEMP = $RuntimeTemp
$env:TMP = $RuntimeTemp
if ($WorkDir) { $env:LABELAGENT_WORK_DIR = [IO.Path]::GetFullPath($WorkDir) }
$env:SAM3_URL = $SamUrl
$pids = @()
$managedProcesses = @()
$startupId = [Guid]::NewGuid().ToString("N")
$recordInstalled = $false

$doctorArgs = @("tools\repro_doctor.py", "--require-web-port")
if ($SamUrl -eq "http://127.0.0.1:8766") {
    $doctorArgs += @("--require-cuda", "--require-weight", "--require-sam-port")
}
& $Python @doctorArgs | Write-Host
if ($LASTEXITCODE -ne 0) { throw "LabelAgent startup prerequisites are incomplete" }

try {
    if ($SamUrl -eq "http://127.0.0.1:8766") {
        $sam = Start-Process -FilePath $Python -ArgumentList "-m", "source.sam3_service.server" `
            -WorkingDirectory $Root -RedirectStandardOutput (Join-Path $Logs "sam3.out.log") `
            -RedirectStandardError (Join-Path $Logs "sam3.err.log") -PassThru -WindowStyle Hidden
        $pids += $sam.Id
        $managedProcesses += @{
            pid = $sam.Id
            role = "sam3"
            started_at_utc = $sam.StartTime.ToUniversalTime().ToString("o")
            executable_path = [IO.Path]::GetFullPath($sam.Path)
        }
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
    $managedProcesses += @{
        pid = $web.Id
        role = "workbench"
        started_at_utc = $web.StartTime.ToUniversalTime().ToString("o")
        executable_path = [IO.Path]::GetFullPath($web.Path)
    }

    $recordJson = @{
        schema_version = "labelagent.local-process-record.v2"
        startup_id = $startupId
        processes = $managedProcesses
        pids = $pids
        started_at = (Get-Date).ToUniversalTime().ToString("o")
        sam_url = $SamUrl
        root = [IO.Path]::GetFullPath($Root)
    } | ConvertTo-Json -Depth 4
    & (Join-Path $PSScriptRoot "write_process_record_atomic.ps1") -Record $Record -Json $recordJson
    $recordInstalled = $true

    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        if ($web.HasExited -or ($SamUrl -eq "http://127.0.0.1:8766" -and $sam.HasExited)) {
            break
        }
        try {
            $null = Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/run/status" -TimeoutSec 2
            $ready = $true
            break
        } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $ready) {
        throw "Workbench did not become ready. Check .repro\logs\workbench.err.log"
    }

    & $Python tools\repro_doctor.py --require-services --sam-url $SamUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Services started but failed the read-back check"
    }
} catch {
    foreach ($processId in $pids) {
        Stop-Process -Id $processId -ErrorAction SilentlyContinue
    }
    if ($recordInstalled -and (Test-Path -LiteralPath $Record)) {
        try {
            $currentRecord = Get-Content -Raw -LiteralPath $Record | ConvertFrom-Json
            if ($currentRecord.startup_id -eq $startupId) {
                Remove-Item -LiteralPath $Record -Force
            }
        } catch {
            Write-Warning "The failed startup could not safely remove its process record; it was preserved for inspection."
        }
    }
    throw
}

if (-not $NoBrowser) { Start-Process "http://127.0.0.1:8765" }
Write-Host "LabelAgent ready: http://127.0.0.1:8765"
Write-Host "Logs: $Logs"
