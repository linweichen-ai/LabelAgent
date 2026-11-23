param(
    [Parameter(Mandatory = $true)]
    [string]$Record,
    [Parameter(Mandatory = $true)]
    [string]$Json
)

$ErrorActionPreference = "Stop"
$recordPath = [IO.Path]::GetFullPath($Record)
$recordDirectory = [IO.Path]::GetDirectoryName($recordPath)
if ([IO.Path]::GetFileName($recordPath) -ne "processes.json" -or
        [IO.Path]::GetFileName($recordDirectory) -ne ".repro") {
    throw "Process records may only be written as .repro\processes.json."
}
if (-not (Test-Path -LiteralPath $recordDirectory -PathType Container)) {
    throw "Process record directory does not exist."
}

$decoded = $Json | ConvertFrom-Json
if ($decoded.schema_version -ne "labelagent.local-process-record.v2" -or
        -not $decoded.startup_id -or
        @($decoded.processes).Count -lt 1 -or
        @($decoded.pids).Count -lt 1 -or
        -not $decoded.root) {
    throw "Process record payload is incomplete."
}

$pendingPath = Join-Path $recordDirectory ("processes.pending-" + [Guid]::NewGuid().ToString("N") + ".json")
$backupPath = Join-Path $recordDirectory ("processes.backup-" + [Guid]::NewGuid().ToString("N") + ".json")
try {
    [IO.File]::WriteAllText($pendingPath, $Json, [Text.UTF8Encoding]::new($false))
    $readBack = [IO.File]::ReadAllText($pendingPath, [Text.Encoding]::UTF8)
    if ($readBack -cne $Json) {
        throw "Pending process record did not read back exactly."
    }
    $roundTrip = $readBack | ConvertFrom-Json
    if ($roundTrip.startup_id -ne $decoded.startup_id -or
            $roundTrip.schema_version -ne $decoded.schema_version) {
        throw "Pending process record identity changed during read-back."
    }
    if (Test-Path -LiteralPath $recordPath) {
        [IO.File]::Replace($pendingPath, $recordPath, $backupPath, $true)
    } else {
        [IO.File]::Move($pendingPath, $recordPath)
    }
} finally {
    Remove-Item -LiteralPath $pendingPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
}
