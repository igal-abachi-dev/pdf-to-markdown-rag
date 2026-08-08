param(
    [string]$PythonVersion = "3.12"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$EnvFile = Join-Path $ProjectRoot ".env"
$EnvExample = Join-Path $ProjectRoot ".env.example"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install 64-bit Python 3.12, then rerun this script."
}

& py "-$PythonVersion" -c "import sys; print(sys.version)"
if ($LASTEXITCODE -ne 0) {
    throw "Python $PythonVersion was not found. Install it with: winget install -e --id Python.Python.3.12"
}

if (-not (Test-Path $VenvPython)) {
    & py "-$PythonVersion" -m venv (Join-Path $ProjectRoot ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Creating the Python virtual environment failed."
    }
}

& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Upgrading pip failed. Check your network connection and rerun the script."
}
& $VenvPython -m pip install --editable $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    throw "Installing the PDF ingestion dependencies failed."
}

if (-not (Test-Path $EnvFile)) {
    Copy-Item -LiteralPath $EnvExample -Destination $EnvFile
    Write-Host "Created $EnvFile"
}

foreach ($Folder in @("inbox", "raw", "md", "metadata", "chunks", "pages", "processed", "failed", "logs")) {
    New-Item -ItemType Directory -Path (Join-Path $ProjectRoot $Folder) -Force | Out-Null
}

Write-Host ""
Write-Host "Installation complete."
Write-Host "1. Open $EnvFile and paste your GEMINI_API_KEY."
Write-Host "2. Run: .\scripts\run_once.ps1"
Write-Host "3. After the test succeeds, run: .\scripts\install_watcher_task.ps1"
Write-Host ""
& $VenvPython -m rag_pdf_ingest --root $ProjectRoot doctor
if ($LASTEXITCODE -ne 0) {
    Write-Host "The failed API-key check is expected until you edit .env. Resolve any other failed checks before continuing."
}
exit 0
