function Resolve-LabelAgentPython312 {
    param([string]$Requested = "py")

    $candidates = @()
    if ($Requested -ne "py") {
        $command = Get-Command $Requested -ErrorAction SilentlyContinue
        if ($command) { $candidates += $command.Source }
        elseif (Test-Path -LiteralPath $Requested) { $candidates += (Resolve-Path -LiteralPath $Requested).Path }
    } else {
        $previousPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            $standard = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
            $standardExit = $LASTEXITCODE
            $registered = & py -0p 2>$null
        } finally {
            $ErrorActionPreference = $previousPreference
        }
        if ($standardExit -eq 0 -and $standard) { $candidates += @($standard)[-1].Trim() }

        foreach ($line in $registered) {
            if ($line -match '(?i)3\.12[^\r\n]*?\s+([A-Z]:\\.*python\.exe)\s*$') {
                $candidates += $Matches[1]
            }
        }
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        $version = & $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0 -and @($version)[-1].Trim() -eq "3.12") {
            return $candidate
        }
    }

    throw "Python 3.12 was not found. Install it or rerun with -Python C:\path\to\python.exe"
}
