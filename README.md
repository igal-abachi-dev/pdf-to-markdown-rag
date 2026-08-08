# pdf-to-markdown-rag
ingest PDF →  Markdown + retrieval chunks + optional SQLite corpus   > for RAG agents retrieval

# Automatic PDF-to-RAG corpus builder for Windows 11

> Citation-preserving Markdown, auditable retrieval chunks, and an optional SQLite hybrid-search corpus for serious RAG agents.

[![Python 3.11–3.13](https://img.shields.io/badge/Python-3.11--3.13-blue.svg)](https://www.python.org/)
[![Windows 11](https://img.shields.io/badge/Windows-11-0078D4.svg)](https://www.microsoft.com/windows/windows-11)

## What this is

This is a production-oriented **pre-deployment corpus builder**, not a vector database, agent runtime, or real-time upload endpoint. It prepares difficult PDF books for local tools such as Codex and Claude Code and for hosted agents built with Fastify, the Vercel AI SDK, Gemini File Search, or a custom SQLite retrieval layer.

Drop a PDF into `inbox/` or run a one-shot command. The pipeline:
when you Drop a PDF into `inbox/`. The watcher extracts exact native text with PyMuPDF, renders every page, sends the page image plus native text to Gemini 3.6 Flash, and writes citation-preserving Markdown and JSON metadata.

1. Extracts exact native text, fonts, spans, geometry, outlines, and printed page labels with PyMuPDF.
2. Renders a high-resolution image of each physical PDF page.
3. Routes the page image plus native-text evidence to Gemini 3.6 Flash for structure-aware Markdown, unless the optional deterministic native route passes every gate.
4. Writes per-page Markdown, combined books, typed Page IR, diagnostics, provenance, and heading-aware chunks.
5. Optionally embeds and packs the corpus into one read-only SQLite file containing FTS5 lexical search and vectors.

The current visual backend is **Gemini 3.6 Flash with HIGH thinking and HIGH media resolution**. A conservative opt-in router can skip Gemini for pages whose deterministic native rendering passes every quality and visual-complexity gate. A local MinerU backend is planned behind the existing provider-neutral `PageConverter` boundary, but is not implemented yet.

## Who this is for

- Developers building citation-aware RAG agents from textbooks, technical manuals, and reference books.
- Corpora containing tables, columns, diagrams, scans, English, or Hebrew where flat PDF text is not enough.
- Workflows that need claims traceable to a physical PDF page, not merely a plausible retrieved paragraph.
- Small, mostly static libraries prepared before an agent is deployed.

## Why 

Many PDF-to-RAG scripts flatten reading order, split tables, discard page identity, or leave citation text for the answering model to invent. This pipeline treats the original physical PDF page as the stable evidence boundary and keeps every transformation auditable. It favors fidelity, provenance, resumability, and deterministic recovery over real-time throughput.

## Key features

- Hybrid extraction: exact PyMuPDF evidence plus visual layout understanding.
- Canonical physical-page citations with supplementary printed labels.
- Content-addressed book and chunk IDs that remain stable across unrelated repairs.
- Obsidian-friendly page notes and portable JSONL for other retrieval backends.
- Atomic Markdown/HTML tables and fenced blocks that are never silently split or truncated.
- Deterministic structured native fallback with explicit coverage and visual-review flags.
- English/Hebrew-aware text handling and conservative multi-column ordering.
- Bibliography, license, source-quality, parser-route, and API provenance.
- Resume, PID locks, rate-limit governance, targeted retry, and offline rebuilds without Gemini calls.
- Optional verified SQLite artifact with exact page Markdown, FTS5, cached embeddings, and vector-integrity hashes.

## Architecture at a glance

```text
PDF
 ├─→ PyMuPDF exact text + spans + geometry + document facts
 └─→ high-resolution page render
          ↓
   Gemini 3.6 Flash (or gated deterministic native route)
          ↓
page Markdown + Page IR + diagnostics + provenance
          ↓
heading-aware chunking with physical-page citations
          ↓
chunks.jsonl ──→ local filesystem search / Gemini File Search
          └────→ optional SQLite pack: FTS5 + vectors
```
This is intentionally a corpus builder, not a local vector database. Codex and Claude Code can search the files directly. A hosted agent can upload the prepared retrieval chunks to Gemini File Search.

The filesystem is the contract between ingestion and retrieval. The consuming agent does not need to know whether a page came from Gemini, deterministic native rendering, or a future local parser; that route remains explicit in metadata.

## Quick start

Requirements: Windows 11, 64-bit Python 3.12 recommended, and a Gemini API key.

```powershell
Set-Location D:\rag
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
notepad .env                         # set GEMINI_API_KEY
.venv\Scripts\python.exe -m rag_pdf_ingest --root . doctor

# Process one PDF immediately
.venv\Scripts\python.exe -m rag_pdf_ingest --root . ingest "D:\books\one-book.pdf"

# Inspect job status and prepared outputs
.venv\Scripts\python.exe -m rag_pdf_ingest --root . status
Get-ChildItem .\md
Get-ChildItem .\chunks
```

For continuous ingestion, install the watcher after the small manual test described below.

## Output layout

For `Biomechanics.pdf`, the content-addressed ID looks like `Biomechanics-a1b2c3d4e5`:

```text
inbox/                         PDFs waiting to be processed
raw/<book-id>/
  page-0001.txt                Native PyMuPDF text
  page-0001.blocks.json        Text spans, fonts and bounding boxes
  page-0001.native.md          Deterministic structured native rendering
  page-0001.ir.json            Provider-neutral typed page blocks
  page-0001.native-diagnostics.json  Coverage, complexity, configuration and review signals
  book.txt                     All raw pages with page markers
md/<book-id>/
  page-0001.md                 Page Markdown/Gemini Markdown with YAML citation/route metadata
  book.md                      Combined Markdown with visible page markers
metadata/<book-id>/
  document.json                PDF metadata, outline, labels, permissions and rotations
  document.xmp.xml             Original XMP packet when present
  document-profile.json        Font profile and repeated marginalia signals
  page-0001.json               Page-level citation and API metadata
  pages.jsonl                  One metadata record per page
  manifest.json                Resume/status/provenance manifest
chunks/<book-id>/
  chunk-<content-hash>-pp-0001-0003.md  Heading-aware retrieval chunk
  chunks.jsonl                 Stable IDs, bodies, page ranges and hashes
pages/<book-id>/page-0001.jpg  Compressed high-quality rendered source pages
debug/<book-id>/               Optional suspect-page layout overlays
artifacts/corpus.sqlite3      Optional static hosted-agent corpus (not committed)
processed/                     Complete or partially processed PDFs
failed/                        PDFs with book-level failures
logs/ingest.log                Watcher log
```

The canonical citation is the physical PDF page index, for example `Biomechanics.pdf, PDF p. 142`. When the PDF defines a different printed label, citations retain both, for example `PDF p. 18 (printed p. 3)` or `PDF p. 4 (printed p. iv)`.

The Markdown is Obsidian-friendly: YAML properties, tags, callouts, wikilinks and stable page block IDs are included. Open `D:\rag` as the vault root so links between `chunks/` and `md/` resolve. Visible `[Source: ..., PDF page ...]` markers also survive ripgrep and remote re-chunking.

Chunk IDs are content-addressed. `chunk_index` is only presentation order, so inserting or repairing an earlier chunk does not make an unchanged later chunk acquire a different identity. Incremental uploaders should upsert by `chunk_id` and remove remote IDs no longer present in `chunks.jsonl`.

## Detailed Windows setup

### 1. Install Python 3.12

Use 64-bit Python 3.12:

```powershell
winget install -e --id Python.Python.3.12
```

Close and reopen PowerShell, then confirm:

```powershell
py -3.12 --version
```

### 2. Create a Gemini API key

Create a key in [Google AI Studio](https://aistudio.google.com/app/apikey). Never paste the key into chat, source files, `AGENTS.md`, or `CLAUDE.md`.

### 3. Install the project

```powershell
Set-Location D:\rag
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
notepad .env
```

Paste the key after `GEMINI_API_KEY=`. Important defaults are:

```dotenv
GEMINI_MODEL=gemini-3.6-flash
GEMINI_THINKING_LEVEL=HIGH
GEMINI_MEDIA_RESOLUTION=HIGH
GEMINI_MAX_OUTPUT_TOKENS=65536
GEMINI_REQUESTS_PER_DAY=18
GEMINI_MIN_REQUEST_INTERVAL_SECONDS=15
GEMINI_PAUSE_ON_RATE_LIMIT=true
GEMINI_RATE_LIMIT_RETRY_SECONDS=300
MAX_CONSECUTIVE_PAGE_FAILURES=5
RENDER_DPI=240
POLL_SECONDS=5
FILE_STABLE_SECONDS=8
PAGE_IMAGE_FORMAT=JPEG
PAGE_JPEG_QUALITY=90
NATIVE_DEBUG_OVERLAYS=false
NATIVE_PAGE_ROUTER_ENABLED=false
CHUNK_TARGET_TOKENS=750
CHUNK_MAX_TOKENS=900
CHUNK_OVERLAP_TOKENS=120
```

High thinking is expensive for bulk transcription. After testing quality, consider `MEDIUM` for ordinary pages.

`NATIVE_PAGE_ROUTER_ENABLED` is deliberately off by default, so every page continues to use the configured visual parser. When enabled, a page skips the LLM only if native extraction succeeded, text is non-empty and usable, structured rendering passed its coverage gate, and `needs_visual_parser` is false. Routed pages remain `status: complete` but record `conversion_route: native_structured`, `parser_backend: pymupdf_native`, and `llm_called: false`. Rejected routes record every failed gate, including excessive-space or visual-content reasons, so thresholds can be evaluated on a real book before becoming a default.

The coverage score is a lexical completeness check against the same PyMuPDF evidence layer, not an independent proof that reading order or layout is correct. Layout, table, column, image and text-quality gates remain separate. The final manifest and `status` output summarize pages that were eligible, actually routed, and sent to an LLM, plus per-gate rejection counts; use those counts to watch `native_text_excessive_spaces` before enabling routing broadly.

#### Free-tier governor

Rate limits are project-specific and can change. Check the project in Google AI Studio's **Dashboard > Rate Limit**, then set `GEMINI_REQUESTS_PER_DAY` below the displayed daily allowance. Gemini 3.6 Flash currently shows 20 RPD for this project's free tier, so the supplied value is 18 and leaves two requests for testing or other applications sharing the project.

The watcher persists request timestamps under `.state/gemini_request_limit.json`, spaces call starts by at least 15 seconds (four RPM), and limits attempts to 18 per Gemini calendar day. Google's RPD window resets at midnight Pacific; the governor follows that fixed reset instead of using a rolling 24-hour approximation. API retries count as requests. A 429 logs Google's quota details; a daily-quota response or three consecutive 429s pauses until the next Pacific reset instead of retrying every five minutes forever.

Five consecutive page failures stop the book as a systemic failure. Completed pages remain resumable, preventing a bad key, incompatible model setting or outage from creating hundreds of failed placeholders.

In strict one-page-per-request mode, a 399-page book requires at least 20 Gemini quota days at the full 20 RPD, or about 23 days at the safer local cap of 18. Twenty 400-page books require about 8,000 successful requests and roughly 445 quota days at 18 RPD. This is the cost of retaining the highest-isolation page workflow on this free-tier quota; billing or a separately evaluated multi-page batching mode is required for materially shorter completion time.

The free Gemini API tier may use submitted content to improve Google's products. Confirm that each book's license permits that processing; a grant limited to strictly private processing may require a paid tier or a local model.

### 4. Run a small manual test

Start with a non-sensitive PDF of 2-5 pages. Copy it into `inbox/`, wait at least eight seconds, then run:

```powershell
.\scripts\run_once.ps1
```

Inspect `raw/<book-id>/book.txt`, `md/<book-id>/book.md`, `metadata/<book-id>/pages.jsonl`, `chunks/<book-id>/chunks.jsonl`, and `logs/ingest.log`. Compare numbers, Hebrew, tables, diagrams and reading order against the rendered JPEG pages.

Page renders default to 240-DPI JPEG quality 90, which preserves small text and diagrams while using substantially less space on image-heavy pages. To convert existing PNG page renders safely and update their page metadata:

```powershell
.venv\Scripts\python.exe -m rag_pdf_ingest --root . convert-page-images
```

The converter acquires each book's PID lock, skips a book that is currently processing, validates JPEG dimensions before writing, updates `page_image_path`, MIME type, byte size and conversion provenance atomically, then removes the PNG. Use `--book <book-id>`, `--quality 85`, or `--keep-png` when needed.

If the watcher is installed as the `SYSTEM` startup task, its already-running Python process must be reloaded once to pick up the new JPEG default. Open PowerShell as Administrator and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\reload_watcher_for_jpeg.ps1
```

This stops only the watcher task, converts unlocked/stale page images, and starts the watcher again. Completed pages remain resumable; no PDF pages are resent merely because their stored image format changed.

Useful commands:

```powershell
.venv\Scripts\python.exe -m rag_pdf_ingest --root . doctor
.venv\Scripts\python.exe -m rag_pdf_ingest --root . status
.venv\Scripts\python.exe -m rag_pdf_ingest --root . ingest "D:\books\one-book.pdf"
```

An interrupted job reuses successful page pairs. Failed page records are retried on the next explicit ingestion attempt.

To retry only failed pages from an existing book, pass either the original PDF or its renamed copy under `processed/`. The source SHA-256 resolves both paths to the same canonical book ID:

```powershell
.venv\Scripts\python.exe -m rag_pdf_ingest --root . retry `
  ".\processed\<book-id>.pdf"
```

Restrict the retry to selected physical PDF pages with `--pages 4` or `--pages 4,7-9`. Successful pages are never sent again. Native-text fallback pages are retained unless `--include-fallback` is supplied.

#### Optional license provenance

Place `Book.license.json` beside `Book.pdf` before ingestion. Copy [license.example.json](license.example.json) and record the grant, scope, rights holder and evidence reference appropriate to that title. The same sidecar may contain canonical `bibliography` fields: `title`, `authors`, `edition`, `publication_year`, and `language`. Bibliography resolution is explicit sidecar, then sane PDF metadata, then the PDF filename stem. `source_file` always remains separate provenance metadata. The pipeline copies the record and its SHA-256 into `metadata/<book-id>/license.json` and `manifest.json`; pages and chunks carry the resolved bibliography, its field sources, and `license_status`. This installation assumes only permitted books enter the inbox, so a missing or legacy `unspecified` record normalizes to `authorized`. Use `public_domain` only when that is the title's actual copyright status.

You can attach or correct a license later and regenerate only derived artifacts—no PDF rendering and no Gemini requests:

```powershell
.venv\Scripts\python.exe -m rag_pdf_ingest --root . rebuild-book `
  <book-id> --license "D:\licenses\Book.license.json"
```

`rebuild-book` acquires the same PID lock as ingestion, rejects a currently processing book, refreshes page frontmatter/citations, recomputes `book.md`, `pages.jsonl`, and `chunks.jsonl`, and refreshes page-status counts in the manifest. It makes no Gemini conversion calls. Correcting bibliography does not change content-addressed chunk IDs, but changed embedding input metadata invalidates only the affected embedding-cache entries. Each chunk records `included_pages`, `fallback_pages`, `has_native_text_fallback`, `source_quality`, `structural_block_type`, and `oversized_reason`.

After upgrading an older completed book, refresh PDF facts and deterministic native fallback pages without using Gemini:

```powershell
.venv\Scripts\python.exe -m rag_pdf_ingest --root . rebuild-book `
  <book-id> --refresh-pdf-facts --refresh-native-fallbacks
```

This records `/Info` and XMP metadata, outline paths, printed page labels, permissions, coordinate spaces and rotation. It regenerates only native artifacts and pages already marked `native_text_fallback`; successful Gemini page bodies remain unchanged.

The native renderer uses a small typed Page IR with deterministic content-hash block IDs. It records text quality separately from page complexity (text/image/vector density, table candidates and decorative vector rules), preserves suppressed source blocks with explicit reasons, hashes its processor configuration, and falls back to exact flat native text if its structured output fails the coverage gate. Simple tables use Markdown; confidently reconstructed merged cells use unchanged HTML. Low-confidence tables remain exact text and are flagged for a later visual parser rather than being invented.

`pack --preflight` requires these schema-v3 page diagnostics and stores them in the SQLite `pages` table. An older book must therefore run the two refresh flags above before it can be packed with the new schema.

An already-running watcher keeps the Python code it loaded at startup. Let an active book finish, then restart the scheduled watcher before adding another book. Afterward, run the offline refresh command for books created by the older watcher; the shared PID lock prevents that rebuild from racing ingestion.

### 5. Start automatically

The normal installation runs after your user logs on:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_watcher_task.ps1
```

To ingest before login, open PowerShell **as Administrator** and install a startup task running under `SYSTEM`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_watcher_task.ps1 -AtStartup
```

Verify or remove the task:

```powershell
Get-ScheduledTask -TaskName RagPdfIngestWatcher
Get-ScheduledTaskInfo -TaskName RagPdfIngestWatcher
Get-Content .\logs\ingest.log -Tail 50
powershell -ExecutionPolicy Bypass -File .\scripts\uninstall_watcher_task.ps1
```

The installer starts `RagPdfIngestWatcher` immediately; a reboot is not required. The Windows scheduled task—not Codex or Claude—owns inbox monitoring, so ingestion continues when those agents and their terminals are closed. Follow the live log, or manually start/restart the watcher, with:

```powershell
Get-Content D:\rag\logs\ingest.log -Tail 50 -Wait
Start-ScheduledTask -TaskName RagPdfIngestWatcher

# Restart after changing code or .env
Stop-ScheduledTask -TaskName RagPdfIngestWatcher
Start-ScheduledTask -TaskName RagPdfIngestWatcher
```

For multi-day ingestion, keep the machine powered and prevent sleep or hibernation. The task can run on battery, but Windows cannot process pages while the computer is asleep.

#### Inbox queue behavior

- The watcher polls every five seconds by default and accepts only non-empty `*.pdf` files whose modification time is at least eight seconds old.
- Multiple PDFs may be queued together. Each scan processes stable files sequentially in filename order; only one book is ingested at a time.
- A PDF added while another book is running is discovered on the next scan. Completed or partial books move to `processed/`; book-level failures move to `failed/`.
- Interrupted books remain recoverable through their content hash, page metadata, and PID lock. After restart, completed pages are reused instead of being sent again.

For a future trusted LAN inbox, share only `D:\rag\inbox`—never the project root, which contains `.env` and the Gemini API key. Restrict write access to trusted accounts. A client should copy `Book.pdf.uploading` and rename it to `Book.pdf` only after transfer completes; the watcher ignores the temporary extension. Avoid simultaneous uploads to the same filename, keep PyMuPDF updated, and remember that anyone allowed to enqueue PDFs can consume parser quota. The watcher still uses the local `D:\rag\inbox` path; clients can access the share as `\\YOUR-PC-NAME\RagInbox`.

## Failure behavior

The watcher retries temporary failures such as timeouts, rate limits and server errors. Permanent bad requests and completed responses with `MAX_TOKENS`, `SAFETY`, or another non-`STOP` finish reason are not repeatedly retried.

If one page fails, it gets an explicit Obsidian warning note with the error type, finish reason, API message and metadata path. The complete error response snapshot is stored in the page JSON rather than bloating retrieval Markdown.

For `RECITATION`, a page with native PDF text becomes `native_text_fallback`. The deterministic native renderer preserves headings, inline styling, conservative column order, lists, footnotes, simple Markdown tables and unambiguous merged-cell HTML tables. Low-confidence structures fall back to exact native text and carry reason codes instead of inventing layout. Figures receive an explicit placeholder but no fabricated semantic description. The page remains searchable and citable while flagged for visual review. A scanned page without native text remains failed and requires OCR or another parser. The remaining pages continue and the book finishes as `partial`.

The default output ceiling is 65,536 because thinking tokens share the output budget. A failed page can be retried by explicitly ingesting its processed PDF again; successful pages are reused.

Stale locks recover automatically. A lock is removed only when its recorded PID is no longer running, and a process never deletes a lock it failed to acquire.

## Retrieval: local and hosted agents

For Codex or Claude Code on this machine, search `md/`, `raw/` and `metadata/` directly. File search or ripgrep is especially reliable for exact numbers, units, names and Hebrew. Do not add a local vector database unless evaluation shows a real retrieval gap.

For a hosted agent, the same chunks can be packed into one read-only SQLite file. It contains books, exact per-page Markdown, chunks, an FTS5 lexical index, an explicit `vector_index -> chunk_id` mapping, and a contiguous little-endian Float32 vector matrix. One file is appropriate for this small, static corpus; keep mutable user/session data in the application's normal database.

A retrieval adapter should return the chunk body together with stable evidence—not just text. At minimum expose `chunk_id`, `book_id`, `citation`, `page_start`, `page_end`, `included_pages`, `source_quality`, and the visual-review flags. The answering agent can then validate that every emitted citation refers to evidence actually retrieved for that request.

Preflight one or all finalized books before spending embedding requests:

```powershell
.venv\Scripts\python.exe -m rag_pdf_ingest --root . pack --preflight
.venv\Scripts\python.exe -m rag_pdf_ingest --root . pack --preflight --book <book-id>
```

Production packing writes schema-v5 SQLite artifacts with bibliography, printed-page labels, outline paths, visual-review reasons, native processor hashes, page-complexity/quality JSON, and structural-block diagnostics. Complete HTML tables (including internal blank lines), Markdown tables, and fenced blocks remain atomic during chunking. An atomic structural block may exceed the preferred 900-token chunk size and records the reason; it is never silently truncated or flattened. Packing accepts `licensed`, `authorized`, `owned`, or public-domain license statuses and refuses failed-page holes, hash mismatches, missing exact page Markdown, stale bibliography/chunk schemas, and conservatively estimated inputs above Gemini Embedding 2's 8,192-token limit. Native-text fallback pages remain packable but are reported as warnings. `auto_truncate` is an Enterprise-only option and is intentionally omitted from Gemini Developer API requests; the local preflight keeps normal 500-900-token chunks far below the service limit.

For a local integration artifact that sends nothing to Gemini, use deterministic fake vectors:

```powershell
.venv\Scripts\python.exe -m rag_pdf_ingest --root . pack `
  --book test-46fc864fda --fake-embeddings `
  --output .\artifacts\corpus-dev.sqlite3
```

For the real corpus, set the embedding defaults in `.env` and omit `--fake-embeddings`:

```dotenv
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIMENSIONS=768
GEMINI_EMBEDDING_BATCH_SIZE=32
```

```powershell
.venv\Scripts\python.exe -m rag_pdf_ingest --root . pack `
  --output .\artifacts\corpus.sqlite3
```

Gemini Embedding 2 receives each chunk as a separate `Content` object. The packer asserts response cardinality and dimensions, uses `RETRIEVAL_DOCUMENT`, and caches each vector by embedding model, dimensions, task type, and the exact embedding-input SHA-256. Configuration and response-shape errors fail immediately; transient 408/409/429/5xx and network failures are retried. Repacking unchanged chunks reuses the cache. The SQLite sidecar manifest records the database SHA-256, corpus hash, ordered chunk-ID hash, vector-blob hash, dimensions, model, and cache hits/misses.

FTS5 indexes only `body` and `section` with `unicode61 remove_diacritics 2`; filenames and citations remain returnable metadata but cannot make every book chunk match. Runtime BM25 weights are `10.0, 3.0`. The query helper accepts user-question text, removes generic prose/schema words and standalone digits, normalizes multiplication and RPE forms, deduplicates terms, and stops at eight. Shared input/output fixtures live in `fixtures/fts-query-normalization.json`; the future TypeScript agent should run the same fixture file in its tests.

Every packed artifact carries `artifact_mode`, `semantic_vectors`, and `production_ready`. Fake-vector artifacts are marked `development/false/false`; a hosted agent must refuse to serve them. Real Gemini packs are marked `production/true/true`.

The SQLite file and its full-text book content are ignored by Git. Publish it only to private storage allowed by each title's recorded license, verify the sidecar SHA-256 before startup, and authenticate any API that can return licensed passages.

Gemini File Search remains another valid target. Each chunk is heading-aware, approximately 500-900 tokens, carries `page_start`/`page_end`, and contains visible page/source markers. The portable source artifact is still `chunks.jsonl`.

`chunks.jsonl` also contains the plain `body`, stable `chunk_id`, citation fields, source quality, and license status, so it can be loaded into another retrieval backend later without repeating PDF conversion.

Current integrations use a File Search store and the File Search tool. In the Vercel AI SDK, use `google.tools.fileSearch(...)`; `providerOptions.google.fileSearchStore` is not the current interface. See the official [Gemini File Search guide](https://ai.google.dev/gemini-api/docs/file-search) and [Vercel Google provider guide](https://ai-sdk.dev/providers/ai-sdk-providers/google-generative-ai).

## Why page images and native text are both supplied

- PyMuPDF provides exact embedded text, fonts, spans and bounding boxes.
- The rendered page gives Gemini columns, headings, figures, labels and reading order.
- Image-only scans simply have an empty native-text reference and are transcribed visually.
- Gemini is instructed to treat page content as untrusted data, not as instructions.

This can improve complex textbook pages over native extraction alone, but no parser wins on every page. Build a 50-100 page evaluation set from your own books before processing the entire collection.

For fitness and health books, include plain and Hebrew prose, columns, scans, nutrition tables, units/dosages, anatomy labels, exercise sequences, charts and references. Medical or training claims must remain traceable to the original PDF page; generated image descriptions are not independent clinical evidence.

## Limitations and responsible use

- With `NATIVE_PAGE_ROUTER_ENABLED=false`, one PDF page normally requires one Gemini generation request. Large libraries can take days or months on restrictive free-tier quotas. The opt-in native router can reduce requests, but should be enabled only after evaluating its gate statistics and output on representative pages.
- This is a batch preparation pipeline for mostly static corpora. It is not designed for user-facing, on-demand PDF uploads or real-time ingestion inside an API request.
- Gemini can refuse pages with outcomes such as `RECITATION` or safety filtering. Native-text pages remain recoverable and citable through the deterministic fallback; image-only pages require a successful visual parser.
- Native coverage measures lexical completeness against PyMuPDF extraction. It does not independently prove correct layout, reading order, table structure, or image interpretation.
- The current visual backend requires an external Gemini API connection. Local MinerU support is planned but not currently available.
- The SQLite pack is a static retrieval artifact, not a mutable user database or complete agent implementation. Authentication, authorization, retrieval orchestration, citation validation, and answer generation belong in the consuming application.

> [!IMPORTANT]
> **Process only material you own, material in the public domain, or material you are explicitly licensed or authorized to process. Do not ingest unauthorized books.** Keep corpus artifacts private when their permissions require private use, and confirm that the selected API tier and hosting destination are allowed by the applicable rights.

## Development

The repository currently uses Python's built-in `unittest` suite and does not define a separate development extra:

```powershell
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m compileall -q src tests
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use synthetic fixtures rather than copyrighted book pages. They cover bibliography precedence, rotations, Hebrew handling, tables, structural chunking, native routing, RECITATION fallback, rebuild locking, packing preflight, FTS5, embeddings, and SQLite retrieval/integrity invariants.

## Troubleshooting

If `doctor` reports a missing key, open `.env` and set:

```dotenv
GEMINI_API_KEY=your-key-here
```

If a PDF remains in `inbox/`, check `FILE_STABLE_SECONDS`, `logs/ingest.log`, and Task Scheduler history. The `doctor` command performs real write probes for the inbox and output directories.

In `Get-ScheduledTaskInfo`, `LastTaskResult` value `267009` (`0x41301`) means the long-running watcher task is currently running; it is not a failure code. Confirm with `Get-ScheduledTask -TaskName RagPdfIngestWatcher` and look for `State: Running`.

If costs are too high, change `GEMINI_THINKING_LEVEL=MEDIUM`, lower `RENDER_DPI` to 200, or use Gemini only for scanned and visually complex pages.

For the first quality test, keep `KEEP_PAGE_IMAGES=true`. Before the full 8,000-page run, decide whether you need every JPEG for auditing. Setting it to `false` still sends rendered pages to Gemini and keeps the original PDFs, but can save substantial local disk space.
