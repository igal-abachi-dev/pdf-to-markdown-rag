# PDF ingestion instructions

Use the repository CLI instead of writing ad-hoc PDF conversion scripts:

```powershell
.venv\Scripts\python.exe -m rag_pdf_ingest --root . scan
.venv\Scripts\python.exe -m rag_pdf_ingest --root . status
```



PDFs enter through `inbox/`.

# RAG Retrieval:
 Generated content is placed under `raw/`, `md/`, `metadata/`, `pages/`, and `chunks/`. Search the local corpus directly;

For a quick terminal/app search in codex:
  rg -n -i -C 3 "<terms derived from the user's question>" .\md


 `chunks/` is the hosted File Search export. Do not print `.env` or the Gemini API key. 
 
 Preserve generated citation metadata and page boundaries. If a manifest is `partial`, report both `failed_pages` and `fallback_pages`. 
 
 Do not cite failed pages; identify `native_text_fallback` evidence as lacking verified visual layout/image descriptions.


@AGENTS.md
  ## Claude Code
  Follow the repository retrieval instructions above for questions about the PDF corpus.