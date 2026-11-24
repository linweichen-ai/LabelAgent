[CmdletBinding()]
param(
    [string]$Target = "factory-5090",
    [ValidateRange(1024, 65535)][int]$RemotePort = 18765,
    [ValidateRange(1024, 65535)][int]$LocalPort = 18765,
    [string]$RuntimeDirectory = "",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$runtime = if ([string]::IsNullOrWhiteSpace($RuntimeDirectory)) {
    Join-Path $root ".repro"
} else {
    [System.IO.Path]::GetFullPath($RuntimeDirectory)
}
$recordPath = Join-Path $runtime "remote-workbench-tunnel.json"
$marker = "${LocalPort}:127.0.0.1:${RemotePort}"
$url = "http://127.0.0.1:${LocalPort}"
New-Item -ItemType Directory -Force -Path $runtime | Out-Null

$ssh = (Get-Command ssh.exe -ErrorAction Stop).Source

function Get-WorkbenchReadiness([string]$BaseUrl) {
    try {
        $run = Invoke-RestMethod "$BaseUrl/api/run/status" -TimeoutSec 3
        $models = Invoke-RestMethod "$BaseUrl/api/model_readiness" -TimeoutSec 5
        return [pscustomobject]@{
            reachable = $true
            ready = ($models.ready -eq $true -and $models.sam.ready -eq $true -and $models.vlm.ready -eq $true)
            run_status = [string]$run.status
            sam_ready = ($models.sam.ready -eq $true)
            vlm_ready = ($models.vlm.ready -eq $true)
        }
    } catch {
        return [pscustomobject]@{
            reachable = $false
            ready = $false
            run_status = "unreachable"
            sam_ready = $false
            vlm_ready = $false
        }
    }
}

if (Test-Path -LiteralPath $recordPath) {
    $existing = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($existing.pid)" -ErrorAction SilentlyContinue
    if ($process -and $process.ExecutablePath -eq $ssh -and $process.CommandLine -like "*${marker}*") {
        $existingReadiness = Get-WorkbenchReadiness $url
        if ($existingReadiness.ready) {
            Write-Output "Remote LabelAgent is ready: $url (SAM3=true, Qwen8B=true, run=$($existingReadiness.run_status))"
            if (-not $NoBrowser) { Start-Process $url }
            exit 0
        }
        if ($existingReadiness.reachable) {
            throw "Remote LabelAgent page is connected, but its models are not ready (SAM3=$($existingReadiness.sam_ready), Qwen8B=$($existingReadiness.vlm_ready)). The managed tunnel was kept unchanged."
        }
        # The recorded SSH process still exists but no longer forwards a usable
        # page. It is safe to recycle only this identity-checked managed tunnel.
        Stop-Process -Id $existing.pid
        Wait-Process -Id $existing.pid -Timeout 5 -ErrorAction SilentlyContinue
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
        $observed = Get-WorkbenchReadiness $url
        if ($observed.reachable) { $ready = $true; break }
        Start-Sleep -Milliseconds 250
    }
    if (-not $ready) { throw "The local tunnel did not become ready." }
    if (-not $observed.ready) {
        throw "The remote page is reachable, but its models are not ready (SAM3=$($observed.sam_ready), Qwen8B=$($observed.vlm_ready))."
    }
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

Write-Output "Remote LabelAgent ready: $url (SAM3=true, Qwen8B=true, run=$($observed.run_status))"
Write-Output "Tunnel PID: $($process.Id)"
if (-not $NoBrowser) { Start-Process $url }
