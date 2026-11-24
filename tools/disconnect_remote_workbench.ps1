[CmdletBinding()]
param(
    [string]$RuntimeDirectory = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$runtime = if ([string]::IsNullOrWhiteSpace($RuntimeDirectory)) {
    Join-Path $root ".repro"
} else {
    [System.IO.Path]::GetFullPath($RuntimeDirectory)
}
$recordPath = Join-Path $runtime "remote-workbench-tunnel.json"
if (-not (Test-Path -LiteralPath $recordPath)) {
    Write-Output "No managed remote LabelAgent tunnel."
    exit 0
}

$record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$($record.pid)" -ErrorAction SilentlyContinue
if ($process) {
    $expectedExecutable = [System.IO.Path]::GetFullPath([string]$record.executable)
    $actualExecutable = [System.IO.Path]::GetFullPath([string]$process.ExecutablePath)
    if ($actualExecutable -ne $expectedExecutable -or $process.CommandLine -notlike "*$($record.marker)*") {
        throw "Refusing to stop PID $($record.pid): process identity does not match the managed SSH tunnel."
    }
    Stop-Process -Id $record.pid
    Wait-Process -Id $record.pid -Timeout 5 -ErrorAction SilentlyContinue
}
Remove-Item -LiteralPath $recordPath -Force
Write-Output "Managed remote LabelAgent tunnel stopped."
