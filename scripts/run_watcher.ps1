$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run setup_windows.ps1 first."
}

& $Python -m rag_pdf_ingest --root $ProjectRoot watch
exit $LASTEXITCODE

