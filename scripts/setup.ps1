<#
One-command setup for suspkin (Windows PowerShell 5.1+).
Usage: powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 [-Minimal] [-Test] [-NoFrontend]
#>
param(
    [switch]$Minimal,
    [switch]$Test,
    [switch]$NoFrontend
)

$ErrorActionPreference = 'Stop'

function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }
function Has($name) { return [bool](Get-Command $name -ErrorAction SilentlyContinue) }
# Run a native command (array: exe + args); fail fast on nonzero exit.
function Run([string[]]$cmd, [string]$what) {
    $exe = $cmd[0]
    $rest = @()
    if ($cmd.Length -gt 1) { $rest = $cmd[1..($cmd.Length - 1)] }
    & $exe @rest
    if ($LASTEXITCODE -ne 0) { Fail "$what failed (exit $LASTEXITCODE)" }
}
# Quiet probe: returns $true if the command exits 0.
function Probe([string[]]$cmd) {
    $exe = $cmd[0]
    $rest = @()
    if ($cmd.Length -gt 1) { $rest = $cmd[1..($cmd.Length - 1)] }
    try {
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $exe @rest *> $null
        $ok = ($LASTEXITCODE -eq 0)
        $ErrorActionPreference = $prev
        return $ok
    } catch { return $false }
}

Set-Location (Join-Path $PSScriptRoot '..')

# 1. Python >= 3.11
Step 'Checking Python >= 3.11'
$verCheck = 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'
$py = $null
$candidates = @(
    @('py', '-3'),
    @('python'),
    @('python3')
)
foreach ($c in $candidates) {
    if (-not (Has $c[0])) { continue }
    if (Probe ($c + @('-c', $verCheck))) { $py = $c; break }
}
if ($py) { Write-Host ("Using: " + ($py -join ' ')) }

# 2. uv
Step 'Checking uv'
$uv = $null
if (Has 'uv') {
    $uv = @('uv')
} elseif ($py -and (Probe ($py + @('-m', 'uv', '--version')))) {
    $uv = $py + @('-m', 'uv')
} elseif ($py) {
    Write-Host 'uv not found, installing with pip (--user)'
    Run ($py + @('-m', 'pip', 'install', '--user', 'uv')) 'pip install --user uv'
    if (-not (Probe ($py + @('-m', 'uv', '--version')))) { Fail 'uv installed but not importable' }
    $uv = $py + @('-m', 'uv')
} else {
    Fail 'Python 3.11+ not found and uv not available. Install Python 3.11+ (https://www.python.org/downloads/ or "winget install Python.Python.3.12") or uv (https://docs.astral.sh/uv/), then re-run.'
}
Write-Host ("Using: " + ($uv -join ' '))

if (-not $py) {
    Step 'No suitable Python found, fetching one with uv'
    Run ($uv + @('python', 'install', '3.12')) 'uv python install 3.12'
}

# 3. Dependencies
if ($Minimal) {
    Step 'uv sync (minimal)'
    Run ($uv + @('sync')) 'uv sync'
} else {
    Step 'uv sync --all-extras'
    Run ($uv + @('sync', '--all-extras')) 'uv sync --all-extras'
}

# 4. pre-commit hooks
if ((Test-Path '.git') -and (Test-Path '.pre-commit-config.yaml')) {
    Step 'Installing pre-commit hooks'
    Run ($uv + @('run', 'pre-commit', 'install')) 'pre-commit install'
}

# 5. Frontend (optional, needs Node 22.12+)
if ($NoFrontend) {
    Step 'Skipping frontend build (-NoFrontend)'
} elseif (-not (Test-Path 'frontend/package.json')) {
    Write-Host 'No frontend/package.json, skipping frontend build'
} elseif (-not ((Has 'node') -and (Has 'npm'))) {
    Write-Host 'node/npm not found, skipping frontend build (the web app UI needs Node 22.12+)'
} else {
    Step 'Building the frontend'
    Push-Location 'frontend'
    try {
        Run @('npm', 'ci') 'npm ci'
        Run @('npm', 'run', 'build') 'npm run build'
        Run @('npm', 'run', 'build:single') 'npm run build:single'
    } finally { Pop-Location }
}

# 6. Smoke check
Step 'Smoke check'
Run ($uv + @('run', 'suspkin', 'version')) 'suspkin version'

# 7. Tests
if ($Test) {
    Step 'Running tests'
    Run ($uv + @('run', 'pytest', '-q')) 'pytest'
}

$u = $uv -join ' '
Write-Host ""
Write-Host "Setup complete. Next steps:"
Write-Host "  $u run suspkin --help"
Write-Host "  $u run suspkin check <config.yaml>"
Write-Host "  $u run suspkin optimize configs/studies/fs_reference_optimize.yaml"
Write-Host "  $u run pytest -q"
