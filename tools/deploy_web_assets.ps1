[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)]
    [string]$TargetRoot,
    [string]$SourceRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$SourceRef = "HEAD",
    [Parameter(Mandatory = $true)]
    [string]$ExpectedTargetRef,
    [string]$BackupRoot = ""
)

$ErrorActionPreference = "Stop"
$Assets = @(
    "source/web/app.js",
    "source/web/guide.js",
    "source/web/index.html"
)

function Resolve-ExistingDirectory([string]$Path, [string]$Label) {
    $resolved = Resolve-Path -LiteralPath $Path -ErrorAction Stop
    if (-not (Get-Item -LiteralPath $resolved.Path).PSIsContainer) {
        throw "$Label is not a directory: $Path"
    }
    return $resolved.Path
}

function Read-GitBlob([string]$Repository, [string]$Ref, [string]$RelativePath) {
    $value = & git -C $Repository rev-parse "$Ref`:$RelativePath" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $value) {
        throw "Cannot resolve $Ref`:$RelativePath in $Repository"
    }
    return ([string]$value).Trim()
}

function Read-FileBlob([string]$Path) {
    $value = & git hash-object -- $Path
    if ($LASTEXITCODE -ne 0 -or -not $value) {
        throw "Cannot hash file: $Path"
    }
    return ([string]$value).Trim()
}

function Assert-ContainedPath([string]$Root, [string]$Path) {
    $rootPrefix = [IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    $fullPath = [IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Asset escaped the declared root: $Path"
    }
}

$SourceRoot = Resolve-ExistingDirectory $SourceRoot "SourceRoot"
$TargetRoot = Resolve-ExistingDirectory $TargetRoot "TargetRoot"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is required to verify immutable asset blobs"
}

$plan = @()
foreach ($relative in $Assets) {
    $sourcePath = Join-Path $SourceRoot ($relative -replace '/', '\')
    $targetPath = Join-Path $TargetRoot ($relative -replace '/', '\')
    Assert-ContainedPath $SourceRoot $sourcePath
    Assert-ContainedPath $TargetRoot $targetPath
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "Missing source asset: $relative"
    }
    if (-not (Test-Path -LiteralPath $targetPath -PathType Leaf)) {
        throw "Missing target asset: $relative"
    }
    $sourceBlob = Read-GitBlob $SourceRoot $SourceRef $relative
    $expectedTargetBlob = Read-GitBlob $SourceRoot $ExpectedTargetRef $relative
    $sourceFileBlob = Read-FileBlob $sourcePath
    $targetBlob = Read-FileBlob $targetPath
    if ($sourceFileBlob -ne $sourceBlob) {
        throw "Source working file does not match $SourceRef`: $relative"
    }
    if ($targetBlob -ne $expectedTargetBlob -and $targetBlob -ne $sourceBlob) {
        throw "Target asset has unrecognized local changes and was not touched: $relative"
    }
    $plan += [pscustomobject]@{
        relative = $relative
        source = $sourcePath
        target = $targetPath
        source_blob = $sourceBlob
        expected_target_blob = $expectedTargetBlob
        target_blob = $targetBlob
        needs_update = $targetBlob -ne $sourceBlob
    }
}

$pending = @($plan | Where-Object needs_update)
if (-not $pending.Count) {
    [pscustomobject]@{
        schema_version = "labelagent.web-asset-deploy.v1"
        status = "already_current"
        source_ref = $SourceRef
        expected_target_ref = $ExpectedTargetRef
        deployed_files = @()
        backup_root = $null
    } | ConvertTo-Json -Depth 5 -Compress
    exit 0
}

if (-not $BackupRoot) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $BackupRoot = Join-Path $TargetRoot ".local\deploy-backups\web-$stamp"
} else {
    $BackupRoot = [IO.Path]::GetFullPath($BackupRoot)
}
Assert-ContainedPath $TargetRoot $BackupRoot

if (-not $PSCmdlet.ShouldProcess($TargetRoot, "Deploy verified LabelAgent web assets")) {
    [pscustomobject]@{
        schema_version = "labelagent.web-asset-deploy.v1"
        status = "what_if"
        source_ref = $SourceRef
        expected_target_ref = $ExpectedTargetRef
        pending_files = @($pending.relative)
        backup_root = $BackupRoot
    } | ConvertTo-Json -Depth 5 -Compress
    exit 0
}

New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
$prepared = @()
$deployed = [Collections.Generic.List[object]]::new()
try {
    foreach ($item in $pending) {
        $targetDirectory = Split-Path -Parent $item.target
        $temporary = Join-Path $targetDirectory ("." + [IO.Path]::GetFileName($item.target) + "." + [guid]::NewGuid().ToString("N") + ".tmp")
        Copy-Item -LiteralPath $item.source -Destination $temporary
        if ((Read-FileBlob $temporary) -ne $item.source_blob) {
            Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
            throw "Prepared asset failed hash readback: $($item.relative)"
        }
        $backup = Join-Path $BackupRoot ($item.relative -replace '/', '\')
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $backup) | Out-Null
        if (Test-Path -LiteralPath $backup) {
            Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
            throw "Backup target already exists: $backup"
        }
        $prepared += [pscustomobject]@{item = $item; temporary = $temporary; backup = $backup}
    }

    foreach ($entry in $prepared) {
        [IO.File]::Replace($entry.temporary, $entry.item.target, $entry.backup, $true)
        if ((Read-FileBlob $entry.item.target) -ne $entry.item.source_blob) {
            throw "Deployed asset failed hash readback: $($entry.item.relative)"
        }
        if ((Read-FileBlob $entry.backup) -ne $entry.item.expected_target_blob) {
            throw "Backup asset failed hash readback: $($entry.item.relative)"
        }
        $deployed.Add($entry)
    }
} catch {
    foreach ($entry in @($deployed) | Select-Object -Reverse) {
        $restore = Join-Path (Split-Path -Parent $entry.item.target) ("." + [IO.Path]::GetFileName($entry.item.target) + ".restore." + [guid]::NewGuid().ToString("N") + ".tmp")
        Copy-Item -LiteralPath $entry.backup -Destination $restore
        $replaced = "$($entry.item.target).failed-$([guid]::NewGuid().ToString('N')).bak"
        [IO.File]::Replace($restore, $entry.item.target, $replaced, $true)
        if ((Read-FileBlob $entry.item.target) -ne $entry.item.expected_target_blob) {
            throw "Deployment failed and rollback readback also failed: $($entry.item.relative)"
        }
        Remove-Item -LiteralPath $replaced -Force -ErrorAction SilentlyContinue
    }
    foreach ($entry in $prepared) {
        Remove-Item -LiteralPath $entry.temporary -Force -ErrorAction SilentlyContinue
    }
    throw
}

[pscustomobject]@{
    schema_version = "labelagent.web-asset-deploy.v1"
    status = "deployed"
    source_ref = $SourceRef
    expected_target_ref = $ExpectedTargetRef
    deployed_files = @($deployed | ForEach-Object {
        [pscustomobject]@{
            file = $_.item.relative
            old_blob = $_.item.expected_target_blob
            new_blob = $_.item.source_blob
            backup = $_.backup
        }
    })
    backup_root = $BackupRoot
} | ConvertTo-Json -Depth 6 -Compress
