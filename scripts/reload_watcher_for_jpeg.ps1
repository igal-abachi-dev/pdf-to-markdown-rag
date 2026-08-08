param(
    [string]$TaskName = "RagPdfIngestWatcher"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run .\scripts\setup_windows.ps1 first."
}

$Task = Get-ScheduledTask -TaskName $TaskName
Write-Host "Stopping scheduled task: $TaskName"
Stop-ScheduledTask -TaskName $TaskName

try {
    $Deadline = (Get-Date).AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 500
        $State = (Get-ScheduledTask -TaskName $TaskName).State
    } while ($State -eq "Running" -and (Get-Date) -lt $Deadline)
    if ($State -eq "Running") {
        throw "Watcher did not stop within 30 seconds."
    }

    & $Python -m rag_pdf_ingest --root $ProjectRoot convert-page-images
    if ($LASTEXITCODE -ne 0) {
        throw "Page-image conversion failed with exit code $LASTEXITCODE."
    }
}
finally {
    Write-Host "Starting scheduled task: $TaskName"
    Start-ScheduledTask -TaskName $TaskName
}

Write-Host "Watcher reloaded. Future page renders use JPEG settings from .env."
