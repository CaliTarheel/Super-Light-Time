param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$appUrl = 'http://127.0.0.1:8766'
try {
    $health = Invoke-RestMethod -Uri "$appUrl/api/health" -TimeoutSec 2
    if ($health.app -eq 'deep-time' -and $health.root -eq $projectRoot) {
        if ($health.shutting_down) {
            Write-Host 'The previous server is saving its pause checkpoint. Waiting for shutdown...'
            for ($attempt = 0; $attempt -lt 30; $attempt++) {
                Start-Sleep -Milliseconds 1000
                try { $health = Invoke-RestMethod -Uri "$appUrl/api/health" -TimeoutSec 1 }
                catch { $health = $null; break }
            }
            if ($health) {
                Write-Host 'Checkpoint saving is still in progress. Run Start.cmd again after the server closes.'
                exit 1
            }
        } else {
            if (-not $NoBrowser) { Start-Process $appUrl }
            exit 0
        }
    }
} catch { }
$candidates = @(
    (Join-Path $projectRoot '.venv\Scripts\python.exe'),
    (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
)
$pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
if ($pythonCommand) { $candidates += $pythonCommand.Source }
$runtime = $null
foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate) {
        & $candidate -c 'import numpy, PIL' 2>$null
        if ($LASTEXITCODE -eq 0) { $runtime = $candidate; break }
    }
}
if (-not $runtime) {
    Write-Host 'Python with NumPy and Pillow is needed. See README.md for installation.'
    exit 1
}
$logPath = Join-Path $projectRoot 'output'
New-Item -ItemType Directory -Path $logPath -Force | Out-Null
$arguments = @('"' + (Join-Path $projectRoot 'server.py') + '"')
$process = Start-Process -FilePath $runtime -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logPath 'server.log') -RedirectStandardError (Join-Path $logPath 'server-error.log')
for ($attempt = 0; $attempt -lt 40; $attempt++) {
    Start-Sleep -Milliseconds 250
    try {
        $health = Invoke-RestMethod -Uri "$appUrl/api/health" -TimeoutSec 1
        if ($health.app -eq 'deep-time' -and $health.root -eq $projectRoot) {
            if (-not $NoBrowser) { Start-Process $appUrl }
            exit 0
        }
    } catch { }
    if ($process.HasExited) { break }
}
Write-Host 'The server could not start. See output\server-error.log; port 8766 may already be in use.'
exit 1
