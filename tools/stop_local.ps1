$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Record = Join-Path $Root ".repro\processes.json"

if (-not (Test-Path $Record)) {
    Write-Host "No locally managed LabelAgent process record was found."
    exit 0
}

$data = Get-Content -Raw $Record | ConvertFrom-Json
foreach ($processId in @($data.pids)) {
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($process) {
        Stop-Process -Id $processId
        $process.WaitForExit(10000)
    }
}
Remove-Item -LiteralPath $Record -Force
Write-Host "Locally managed LabelAgent processes stopped."
