[CmdletBinding()]
param(
    [string]$Target = "factory-5090",
    [ValidateRange(1024, 65535)][int]$RemotePort = 18765,
    [ValidateRange(1024, 65535)][int]$LocalPort = 18765,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$runtime = Join-Path $root ".repro"
$recordPath = Join-Path $runtime "remote-workbench-tunnel.json"
$marker = "${LocalPort}:127.0.0.1:${RemotePort}"
$url = "http://127.0.0.1:${LocalPort}"
New-Item -ItemType Directory -Force -Path $runtime | Out-Null

$ssh = (Get-Command ssh.exe -ErrorAction Stop).Source
if (Test-Path -LiteralPath $recordPath) {
    $existing = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($existing.pid)" -ErrorAction SilentlyContinue
    if ($process -and $process.ExecutablePath -eq $ssh -and $process.CommandLine -like "*${marker}*") {
        Write-Output "Remote LabelAgent tunnel is already running: $url"
        if (-not $NoBrowser) { Start-Process $url }
        exit 0
    }
    Move-Item -LiteralPath $recordPath -Destination "$recordPath.stale.$([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))"
}

if (Get-NetTCPConnection -State Listen -LocalPort $LocalPort -ErrorAction SilentlyContinue) {
    throw "Local port $LocalPort is already occupied. Choose another -LocalPort."
}

& $ssh -o BatchMode=yes -o ConnectTimeout=10 $Target `
    "curl -fsS --max-time 5 http://127.0.0.1:${RemotePort}/api/run/status >/dev/null"
if ($LASTEXITCODE -ne 0) {
    throw "The remote LabelAgent workbench is not ready on $Target port $RemotePort."
}

$arguments = @(
    "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
    "-o", "TCPKeepAlive=yes", "-L", $marker, $Target
)
$process = Start-Process -FilePath $ssh -ArgumentList $arguments -PassThru -WindowStyle Hidden
try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        if ($process.HasExited) { break }
        try {
            Invoke-WebRequest "$url/api/run/status" -UseBasicParsing -TimeoutSec 2 | Out-Null
            $ready = $true
            break
        } catch {
            Start-Sleep -Milliseconds 250
        }
    }
    if (-not $ready) { throw "The local tunnel did not become ready." }
    [ordered]@{
        schema_version = "labelagent.remote_workbench_tunnel.v1"
        pid = $process.Id
        executable = $ssh
        target = $Target
        local_port = $LocalPort
        remote_port = $RemotePort
        marker = $marker
        created_utc = [DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $recordPath -Encoding utf8
} catch {
    if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
    throw
}

Write-Output "Remote LabelAgent ready: $url"
Write-Output "Tunnel PID: $($process.Id)"
if (-not $NoBrowser) { Start-Process $url }
