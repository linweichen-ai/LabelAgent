param(
    [switch]$ArchiveStaleRecord
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Record = Join-Path $Root ".repro\processes.json"

. (Join-Path $PSScriptRoot "lifecycle_mutex.ps1")
$lifecycleMutex = Enter-LabelAgentLifecycleMutex -Root $Root
try {
if (-not (Test-Path $Record)) {
    Write-Host "No locally managed LabelAgent process record was found."
    return
}

$data = Get-Content -Raw $Record | ConvertFrom-Json
$expectedRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\')
$recordRoot = if ($data.root) { [IO.Path]::GetFullPath([string]$data.root).TrimEnd('\') } else { "" }
if (-not [string]::Equals($expectedRoot, $recordRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to stop processes: the process record belongs to another checkout."
}

$expectedPython = [IO.Path]::GetFullPath($Python)
$identities = @()
if ($data.schema_version -eq "labelagent.local-process-record.v2" -and $data.processes) {
    $identities = @($data.processes)
} elseif ($data.pids -and $data.started_at) {
    $legacyStarted = [DateTimeOffset]::Parse([string]$data.started_at).UtcDateTime
    foreach ($processId in @($data.pids)) {
        $identities += @{
            pid = [int]$processId
            role = "legacy"
            legacy_min_utc = $legacyStarted.AddSeconds(-60)
            legacy_max_utc = $legacyStarted.AddSeconds(5)
            executable_path = $expectedPython
        }
    }
} else {
    throw "Refusing to stop processes: the process record has no verifiable identity."
}

$verified = @()
$stale = @()
$seen = @{}
foreach ($identity in $identities) {
    $processId = [int]$identity.pid
    if ($seen.ContainsKey($processId)) {
        throw "Refusing to stop processes: duplicate PID in the process record."
    }
    $seen[$processId] = $true
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if (-not $process) { continue }
    try {
        $actualPath = [IO.Path]::GetFullPath($process.Path)
        $actualStarted = $process.StartTime.ToUniversalTime()
    } catch {
        throw "Refusing to stop PID $processId because its identity cannot be read."
    }
    if (-not [string]::Equals($actualPath, $expectedPython, [StringComparison]::OrdinalIgnoreCase)) {
        $stale += "PID $processId executable mismatch"
        continue
    }
    if ($identity.role -eq "legacy") {
        if ($actualStarted -lt $identity.legacy_min_utc -or $actualStarted -gt $identity.legacy_max_utc) {
            $stale += "PID $processId legacy creation-time mismatch"
            continue
        }
    } else {
        if (@("sam3", "workbench") -notcontains [string]$identity.role) {
            throw "Refusing to stop PID $processId because its managed role is invalid."
        }
        if (-not [string]::Equals(
                [IO.Path]::GetFullPath([string]$identity.executable_path),
                $expectedPython,
                [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to stop PID $processId because the recorded executable is invalid."
        }
        $recordedStarted = [DateTimeOffset]::Parse([string]$identity.started_at_utc).UtcDateTime
        if ([Math]::Abs(($actualStarted - $recordedStarted).TotalSeconds) -gt 1) {
            $stale += "PID $processId creation-time mismatch"
            continue
        }
    }
    $verified += $process
}

if ($stale.Count -gt 0) {
    if ($verified.Count -gt 0) {
        throw "Refusing to stop processes: the record mixes matching and stale process identities."
    }
    if (-not $ArchiveStaleRecord) {
        throw "Refusing to stop processes: the record is stale. Re-run with -ArchiveStaleRecord to move the record aside without stopping those unrelated processes."
    }
    $archiveDirectory = Join-Path $Root ".repro\stale-process-records"
    New-Item -ItemType Directory -Force -Path $archiveDirectory | Out-Null
    $archiveName = "processes-" + (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmss-fff") + ".json"
    $archivePath = Join-Path $archiveDirectory $archiveName
    if (Test-Path $archivePath) {
        throw "Refusing to archive the stale record because the destination already exists."
    }
    Move-Item -LiteralPath $Record -Destination $archivePath
    Write-Host "Stale process record archived without stopping unrelated processes: $archivePath"
    return
}

foreach ($process in $verified) {
    Stop-Process -Id $process.Id
    if (-not $process.WaitForExit(10000)) {
        throw "Managed process $($process.Id) did not exit; the process record was preserved."
    }
}
Remove-Item -LiteralPath $Record -Force
Write-Host "Locally managed LabelAgent processes stopped."
} finally {
    Exit-LabelAgentLifecycleMutex -Mutex $lifecycleMutex
}
