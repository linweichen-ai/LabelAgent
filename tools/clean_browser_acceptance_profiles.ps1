param(
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$artifactRoot = (Resolve-Path -LiteralPath (Join-Path $repoRoot 'artifacts')).Path
$prefix = $artifactRoot.TrimEnd('\') + '\'

$targets = @(
    Get-ChildItem -LiteralPath $artifactRoot -Directory |
        ForEach-Object {
            $candidate = Join-Path $_.FullName 'edge-profile'
            if (Test-Path -LiteralPath $candidate -PathType Container) {
                Get-Item -LiteralPath $candidate
            }
        }
)

$rows = foreach ($target in $targets) {
    $resolved = $target.FullName
    if (-not $resolved.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing target outside artifacts: $resolved"
    }
    if ($target.Name -ne 'edge-profile' -or $target.Attributes.HasFlag(
            [System.IO.FileAttributes]::ReparsePoint)) {
        throw "Refusing unexpected or linked target: $resolved"
    }
    $files = @(Get-ChildItem -LiteralPath $resolved -File -Recurse)
    $bytes = ($files | Measure-Object Length -Sum).Sum
    [pscustomobject]@{
        Path = $resolved
        Files = $files.Count
        Bytes = [int64]$bytes
    }
}

$summary = [pscustomobject]@{
    Mode = if ($Execute) { 'execute' } else { 'dry-run' }
    TargetCount = $rows.Count
    FileCount = ($rows | Measure-Object Files -Sum).Sum
    Bytes = ($rows | Measure-Object Bytes -Sum).Sum
    Targets = $rows
}
$summary | ConvertTo-Json -Depth 4

if ($Execute) {
    foreach ($row in $rows) {
        Remove-Item -LiteralPath $row.Path -Recurse -Force
    }
    foreach ($row in $rows) {
        if (Test-Path -LiteralPath $row.Path) {
            throw "Cleanup readback failed: $($row.Path)"
        }
    }
}
