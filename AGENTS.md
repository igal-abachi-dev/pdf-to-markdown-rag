# PDF ingestion instructions for coding agents

This repository contains a deterministic PDF ingestion service. Do not manually recreate its generated files.

When the user asks to ingest PDFs:

1. Put PDF files in `inbox/` if they are not already there.
2. Run `.venv\Scripts\python.exe -m rag_pdf_ingest --root . scan`.
3. Check `.venv\Scripts\python.exe -m rag_pdf_ingest --root . status` and `logs/ingest.log`.
4. Report the generated `raw/<book-id>/book.txt`, `md/<book-id>/book.md`, `metadata/<book-id>/pages.jsonl`, and `chunks/<book-id>/chunks.jsonl` paths.

# RAG Retrieval:
For local retrieval, search `md/`, `raw/`, and `metadata/` directly. 

For a quick terminal/app search in codex:
  rg -n -i -C 3 "<terms derived from the user's question>" .\md
  
Use `chunks/` when preparing a hosted File Search corpus. 

Treat a `partial` manifest as processed but report its `failed_pages` and `fallback_pages`; 
never cite a failed page as converted evidence.

 A `native_text_fallback` page is citable text but must be identified as lacking verified visual layout/image descriptions.


Never expose or print `GEMINI_API_KEY`. Never commit `.env` or generated book content. Do not overwrite a completed book directory; content-addressed book IDs provide resume and deduplication.

## Local RAG retrieval - more info

  When the user asks a question about the ingested PDFs:

  1. Check the relevant `metadata/<book-id>/manifest.json`.
  2. Search `md/` first using ripgrep:
     `rg -n -i -C 3 "<search terms>" .\md`
  3. Generate useful synonyms or related terminology and run additional searches when needed(because rg is lexical, not semantic).
  4. Search `raw/` when exact transcription or additional context is needed.
  5. Use `metadata/` to verify book identity, PDF pages, page status, source quality, and citation fields.
  6. Synthesize an answer from the retrieved passages and cite the book plus PDF page numbers.
  7. Do not treat search matches as sufficient evidence without opening and reading the surrounding passage.
  8. If a manifest is `processing`, state that the corpus is incomplete and only cite pages whose page metadata has
  `"status": "complete"`.
  9. For a `partial` manifest, report `failed_pages` and `fallback_pages`. Never cite failed pages.
  10. A `native_text_fallback` page is citable, but identify it as lacking verified visual layout and image
  descriptions.