param(
    [string]$TaskName = "RagPdfIngestWatcher",
    [switch]$AtStartup
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunScript = Join-Path $ProjectRoot "scripts\run_watcher.ps1"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run .\scripts\setup_windows.ps1 first."
}

$PowerShell = (Get-Command powershell.exe).Source
$Arguments = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$RunScript`""
$Action = New-ScheduledTaskAction -Execute $PowerShell -Argument $Arguments -WorkingDirectory $ProjectRoot
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)
if ($AtStartup) {
    $Trigger = New-ScheduledTaskTrigger -AtStartup -RandomDelay (New-TimeSpan -Minutes 1)
    $Principal = New-ScheduledTaskPrincipal `
        -UserId "SYSTEM" `
        -LogonType ServiceAccount `
        -RunLevel Highest
} else {
    $Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $UserId = if ($env:USERDOMAIN) { "$env:USERDOMAIN\$env:USERNAME" } else { $env:USERNAME }
    $Principal = New-ScheduledTaskPrincipal -UserId $UserId -LogonType Interactive -RunLevel Limited
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Watches $ProjectRoot\inbox and converts PDFs with PyMuPDF and Gemini." `
    -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Write-Host "Installed and started scheduled task: $TaskName"
Write-Host "Trigger: $(if ($AtStartup) { 'Windows startup (SYSTEM)' } else { 'user logon' })"
Write-Host "Drop PDFs into: $(Join-Path $ProjectRoot 'inbox')"
Write-Host "Log file: $(Join-Path $ProjectRoot 'logs\ingest.log')"
