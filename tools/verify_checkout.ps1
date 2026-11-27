param(
    [switch]$SkipPythonTests,
    [switch]$SkipWebTests,
    [switch]$RequireWeight,
    [switch]$RequireCuda
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$VenvScripts = Split-Path -Parent $Python
$TestTemp = Join-Path $Root ".repro\test-temp"
Set-Location $Root

if (-not (Test-Path $Python)) { throw "Missing .venv; run tools\setup_windows.ps1 first" }
New-Item -ItemType Directory -Force -Path $TestTemp | Out-Null
$env:TEMP = $TestTemp
$env:TMP = $TestTemp
$env:PYTHONUTF8 = "1"
$env:PATH = $VenvScripts + [IO.Path]::PathSeparator + $env:PATH
$free = (Get-Item $TestTemp).PSDrive.Free
if ($free -lt 512MB) {
    throw "At least 512 MiB free space is required for isolated regression tests: $TestTemp"
}

# Documentation is part of the release contract.  Fail before the long model
# and browser suites when a tutorial points to a file that is not shipped.
& $Python tools\check_release_doc_refs.py --json
if ($LASTEXITCODE -ne 0) { throw "Release documentation reference check failed" }

$doctorArgs = @("tools\repro_doctor.py")
if ($RequireWeight) { $doctorArgs += "--require-weight" }
if ($RequireCuda) { $doctorArgs += "--require-cuda" }
& $Python @doctorArgs
if ($LASTEXITCODE -ne 0) { throw "Reproducibility self-check failed" }

& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw "Python dependency consistency check failed" }

if (-not $SkipWebTests) {
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
        throw "Node.js is required for the browser logic tests; use -SkipWebTests only when intentionally unavailable"
    }
    & node --version
    if ($LASTEXITCODE -ne 0) { throw "Node.js preflight failed" }

    # Run the small real-browser gate before the long Python suite so a missing
    # Edge/Playwright dependency fails in seconds instead of after the full run.
    & $Python tools\verify_synthetic_signoff_browser.py --ephemeral
    if ($LASTEXITCODE -ne 0) { throw "Synthetic offline signoff browser preflight failed" }
}

if (-not $SkipPythonTests) {
    & $Python -m unittest discover -s tests -p "test_*.py"
    if ($LASTEXITCODE -ne 0) { throw "Python regression tests failed" }
}

if (-not $SkipWebTests) {
    & node --test tests\test_review_packet_core.js tests\test_web_candidate_audit.js `
        tests\test_web_polling.js tests\test_browser_main_run_acceptance.cjs `
        tests\test_candidate_audit_web.cjs tests\test_browser_cvat_human_signoff_draft_acceptance.cjs `
        tests\test_cvat_human_signoff_drafts.js
    if ($LASTEXITCODE -ne 0) { throw "Browser logic tests failed" }
}

Write-Host "LabelAgent checkout verification passed."
