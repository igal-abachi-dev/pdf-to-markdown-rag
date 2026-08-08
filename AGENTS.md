# PDF ingestion instructions for coding agents

This repository contains a deterministic PDF ingestion service. Do not manually recreate its generated files.

When the user asks to ingest PDFs:

1. Put PDF files in `inbox/` if they are not already there.
2. Run `.venv\Scripts\python.exe -m rag_pdf_ingest --root . scan`.
3. Check `.venv\Scripts\python.exe -m rag_pdf_ingest --root . status` and `logs/ingest.log`.
4. Report the generated `raw/<book-id>/book.txt`, `md/<book-id>/book.md`, `metadata/<book-id>/pages.jsonl`, and `chunks/<book-id>/chunks.jsonl` paths.

For local retrieval, search `md/`, `raw/`, and `metadata/` directly. Use `chunks/` when preparing a hosted File Search corpus. Treat a `partial` manifest as processed but report its `failed_pages` and `fallback_pages`; never cite a failed page as converted evidence. A `native_text_fallback` page is citable text but must be identified as lacking verified visual layout/image descriptions.

Never expose or print `GEMINI_API_KEY`. Never commit `.env` or generated book content. Do not overwrite a completed book directory; content-addressed book IDs provide resume and deduplication.
