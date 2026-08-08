from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import Settings


def _parse_pages(value: str) -> set[int]:
    pages: set[int] = set()
    try:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start, end = int(start_text), int(end_text)
                if start <= 0 or end < start:
                    raise ValueError
                pages.update(range(start, end + 1))
            else:
                page = int(part)
                if page <= 0:
                    raise ValueError
                pages.add(page)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "pages must look like 4 or 4,7-9 using positive PDF page numbers"
        ) from exc
    if not pages:
        raise argparse.ArgumentTypeError("at least one page number is required")
    return pages


def _configure_logging(settings: Settings, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    file_handler = RotatingFileHandler(
        settings.logs_dir / "ingest.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def _doctor(settings: Settings) -> int:
    checks: list[tuple[str, bool, str]] = []
    supported_python = (3, 11) <= sys.version_info[:2] < (3, 14)
    checks.append(("Python version", supported_python, sys.version.split()[0]))
    try:
        import pymupdf

        checks.append(("PyMuPDF", True, getattr(pymupdf, "VersionBind", "installed")))
    except Exception as exc:
        checks.append(("PyMuPDF", False, str(exc)))
    try:
        import google.genai  # noqa: F401

        checks.append(("google-genai", True, "installed"))
    except Exception as exc:
        checks.append(("google-genai", False, str(exc)))
    checks.append(("Gemini API key", bool(settings.gemini_api_key), "configured" if settings.gemini_api_key else "missing"))
    checks.append(("Prompt", settings.prompt_path.exists(), str(settings.prompt_path)))
    checks.append(
        (
            "Request governor",
            True,
            f"{settings.requests_per_day}/Pacific calendar day; "
            f"{settings.min_request_interval_seconds:g}s minimum interval; "
            f"pause on 429={settings.pause_on_rate_limit}",
        )
    )
    writable_paths = [
        settings.inbox_dir,
        settings.raw_dir,
        settings.md_dir,
        settings.metadata_dir,
        settings.chunks_dir,
        settings.state_dir,
    ]
    for path in writable_paths:
        probe = path / f".doctor-write-{os.getpid()}"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            checks.append((f"Writable {path.name}", True, str(path)))
        except OSError as exc:
            checks.append((f"Writable {path.name}", False, str(exc)))

    for name, ok, detail in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


def _status(settings: Settings) -> int:
    manifests = sorted(settings.metadata_dir.glob("*/manifest.json"))
    if not manifests:
        print("No ingestion jobs found.")
        return 0
    for path in manifests:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"{data.get('status', 'unknown'):10} "
            f"{data.get('completed_pages', 0):>4}/{data.get('page_count', '?'):<4} "
            f"{data.get('book_id', path.parent.name)}"
        )
        if data.get("failed_pages"):
            print(f"  failed pages: {', '.join(map(str, data['failed_pages']))}")
        if data.get("fallback_pages"):
            print(f"  fallback pages: {', '.join(map(str, data['fallback_pages']))}")
        if data.get("native_routed_pages"):
            print(
                f"  native-routed pages: {len(data['native_routed_pages'])} "
                "(no LLM request)"
            )
        router_summary = data.get("native_router_summary") or {}
        if router_summary:
            print(
                "  native router: "
                f"{router_summary.get('eligible_if_enabled', 0)} eligible, "
                f"{router_summary.get('native_routed', 0)} routed, "
                f"{router_summary.get('llm_invoked', 0)} LLM-invoked"
            )
        if data.get("visual_review_pages"):
            print(
                "  visual review pages: "
                + ", ".join(map(str, data["visual_review_pages"]))
            )
        if data.get("error"):
            print(f"  error: {data['error']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyMuPDF + Gemini PDF ingestion")
    parser.add_argument("--root", type=Path, help="Project root (defaults to RAG_ROOT or package root)")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Check local setup without calling Gemini")
    subparsers.add_parser("scan", help="Process stable PDFs currently in inbox")
    subparsers.add_parser("watch", help="Continuously watch inbox")
    subparsers.add_parser("status", help="Show job manifests")
    ingest_parser = subparsers.add_parser("ingest", help="Process one PDF immediately")
    ingest_parser.add_argument("pdf", type=Path)
    retry_parser = subparsers.add_parser(
        "retry", help="Retry only failed pages from an existing book"
    )
    retry_parser.add_argument("pdf", type=Path)
    retry_parser.add_argument(
        "--pages",
        type=_parse_pages,
        help="Optional PDF pages, for example 4 or 4,7-9",
    )
    retry_parser.add_argument(
        "--include-fallback",
        action="store_true",
        help="Also retry native-text fallback pages to seek full Gemini Markdown",
    )
    rebuild_parser = subparsers.add_parser(
        "rebuild-book",
        help="Rebuild book.md and chunks from saved pages without Gemini calls",
    )
    rebuild_parser.add_argument("book_id")
    rebuild_parser.add_argument(
        "--license",
        type=Path,
        help="Optional license JSON to attach before rebuilding",
    )
    rebuild_parser.add_argument(
        "--refresh-pdf-facts",
        action="store_true",
        help="Reopen the stored PDF and refresh metadata, outline, labels, and rotations",
    )
    rebuild_parser.add_argument(
        "--refresh-native-fallbacks",
        action="store_true",
        help="Regenerate deterministic native artifacts and fallback pages without Gemini",
    )
    pack_parser = subparsers.add_parser(
        "pack", help="Preflight or build a static hybrid-search SQLite corpus"
    )
    pack_parser.add_argument(
        "--preflight", action="store_true", help="Validate inputs without embedding or writing SQLite"
    )
    pack_parser.add_argument(
        "--book",
        dest="book_ids",
        action="append",
        help="Restrict to a book_id; repeat for multiple books",
    )
    pack_parser.add_argument(
        "--include-unspecified",
        action="store_true",
        help="Allow unspecified license records for local development only",
    )
    pack_parser.add_argument(
        "--fake-embeddings",
        action="store_true",
        help="Use deterministic fake vectors for local integration tests",
    )
    pack_parser.add_argument(
        "--fake-dimensions", type=int, default=32, help=argparse.SUPPRESS
    )
    pack_parser.add_argument(
        "--output", type=Path, help="SQLite output path (defaults under artifacts/)"
    )
    images_parser = subparsers.add_parser(
        "convert-page-images",
        aliases=["compress-page-images"],
        help="Convert stored page PNGs to JPEG and update metadata",
    )
    images_parser.add_argument(
        "--book",
        dest="book_ids",
        action="append",
        help="Restrict to a book_id; repeat for multiple books",
    )
    images_parser.add_argument(
        "--quality", type=int, help="JPEG quality 1-100 (defaults to PAGE_JPEG_QUALITY)"
    )
    images_parser.add_argument(
        "--keep-png", action="store_true", help="Keep original PNGs after conversion"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.load(args.root)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _configure_logging(settings, args.verbose)

    try:
        if args.command == "doctor":
            return _doctor(settings)
        if args.command == "status":
            return _status(settings)

        if args.command == "rebuild-book":
            from .pipeline import rebuild_book

            manifest = rebuild_book(
                settings,
                args.book_id,
                license_sidecar=args.license,
                refresh_pdf_facts=args.refresh_pdf_facts,
                refresh_native_fallbacks=args.refresh_native_fallbacks,
            )
            print(
                f"Rebuilt {manifest['book_id']}: "
                f"{manifest.get('outputs', {}).get('chunk_count', 0)} chunks"
            )
            return 0
        if args.command == "pack":
            from .packing import (
                DeterministicFakeEmbedder,
                GeminiEmbedder,
                pack_corpus,
                preflight_corpus,
            )

            report = preflight_corpus(
                settings,
                book_ids=set(args.book_ids) if args.book_ids else None,
                include_unspecified=args.include_unspecified,
            )
            for warning in report.warnings:
                print(f"WARNING: {warning}")
            for error in report.errors:
                print(f"ERROR: {error}", file=sys.stderr)
            print(
                f"Preflight: {len(report.books)} books, {report.page_count} pages, "
                f"{report.chunk_count} chunks, {len(report.errors)} errors"
            )
            if report.errors:
                return 1
            if args.preflight:
                return 0
            if args.fake_dimensions <= 0:
                raise ValueError("--fake-dimensions must be positive")
            output_path = args.output
            if output_path is None:
                filename = "corpus-dev.sqlite3" if args.fake_embeddings else "corpus.sqlite3"
                output_path = settings.corpus_dir / filename
            elif not output_path.is_absolute():
                output_path = settings.root / output_path
            embedder = (
                DeterministicFakeEmbedder(args.fake_dimensions)
                if args.fake_embeddings
                else GeminiEmbedder(
                    api_key=settings.gemini_api_key,
                    model=settings.embedding_model,
                    dimensions=settings.embedding_dimensions,
                    max_retries=settings.max_retries,
                )
            )
            if args.fake_embeddings:
                print(
                    "WARNING: deterministic fake vectors are development-only; "
                    "the artifact will be marked production_ready=false"
                )
            try:
                result = pack_corpus(
                    settings,
                    report,
                    output_path=output_path,
                    embedder=embedder,
                    batch_size=settings.embedding_batch_size,
                )
            finally:
                embedder.close()
            print(
                f"Packed {result['chunk_count']} chunks into {result['path']} "
                f"(sha256 {result['sha256']})"
            )
            print(
                f"Embedding cache: {result['cache_hits']} hits, "
                f"{result['cache_misses']} misses"
            )
            return 0
        if args.command in {"convert-page-images", "compress-page-images"}:
            from .page_images import convert_page_images_to_jpeg

            result = convert_page_images_to_jpeg(
                settings,
                book_ids=set(args.book_ids) if args.book_ids else None,
                quality=args.quality,
                keep_png=args.keep_png,
            )
            for book_id in result["skipped_locked"]:
                print(f"WARNING: skipped locked/processing book {book_id}")
            for page_path in result["skipped_missing_metadata"]:
                print(
                    f"WARNING: retained unfinished PNG without page metadata: {page_path}"
                )
            for error in result["errors"]:
                print(f"ERROR: {error}", file=sys.stderr)
            print(
                f"Converted {result['converted']} PNG page images to JPEG quality "
                f"{result['jpeg_quality']}; saved {result['bytes_saved']} bytes"
            )
            return 1 if result["errors"] else 0

        from .pipeline import ingest_pdf, scan_once, watch

        settings.validate_for_ingestion()
        if args.command == "ingest":
            ingest_pdf(args.pdf, settings)
            return 0
        if args.command == "retry":
            ingest_pdf(
                args.pdf,
                settings,
                retry_pages=args.pages,
                retry_fallback_pages=args.include_fallback,
                retry_existing=True,
            )
            return 0
        if args.command == "scan":
            completed, failed = scan_once(settings)
            print(f"Completed: {completed}; failed: {failed}")
            return 1 if failed else 0
        if args.command == "watch":
            try:
                watch(settings)
            except KeyboardInterrupt:
                print("Watcher stopped.")
            return 0
    except Exception as exc:
        logging.getLogger(__name__).exception("Command failed")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 2
