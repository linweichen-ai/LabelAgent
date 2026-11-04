param([Parameter(Mandatory=$true)][string]$Payload)
$ErrorActionPreference = 'Stop'
$notice = Get-Content -LiteralPath $Payload -Raw -Encoding UTF8 | ConvertFrom-Json
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$icon = New-Object System.Windows.Forms.NotifyIcon
try {
    $icon.Icon = [System.Drawing.SystemIcons]::Information
    $icon.Text = 'LabelAgent'
    $icon.BalloonTipTitle = [string]$notice.title
    $icon.BalloonTipText = [string]$notice.body
    $icon.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::Info
    $icon.Visible = $true
    $icon.ShowBalloonTip(8000)
    $until = [DateTime]::UtcNow.AddSeconds(8)
    while ([DateTime]::UtcNow -lt $until) {
        [System.Windows.Forms.Application]::DoEvents()
        Start-Sleep -Milliseconds 100
    }
    Write-Output 'submitted'
} finally {
    $icon.Visible = $false
    $icon.Dispose()
}
