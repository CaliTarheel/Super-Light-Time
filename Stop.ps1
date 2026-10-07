$appUrl = 'http://127.0.0.1:8766'
try {
    $health = Invoke-RestMethod -Uri "$appUrl/api/health" -TimeoutSec 2
    if ($health.app -eq 'deep-time' -and $health.root -eq $PSScriptRoot) {
        Invoke-RestMethod -Uri "$appUrl/api/shutdown" -Method Post -ContentType 'application/json' -Body '{}' | Out-Null
        Write-Host 'Deep Time is saving a resumable pause before closing...'
        $stopDeadline = (Get-Date).AddSeconds(60)
        do {
            Start-Sleep -Milliseconds 500
            try {
                $stopHealth = Invoke-RestMethod -Uri "$appUrl/api/health" -TimeoutSec 2
            } catch {
                Write-Host 'Deep Time is closed. Paused worlds can resume when you reopen it.'
                exit 0
            }
        } while ((Get-Date) -lt $stopDeadline)
        Write-Host 'Deep Time is still finishing its current step and saving the pause. Leave it running until this completes; do not force-close it.'
    }
} catch {
    Write-Host 'Deep Time is not running at its default address.'
}
