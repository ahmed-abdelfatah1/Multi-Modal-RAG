"""Run the complete ingestion pipeline for secondary corpus."""

import json
import sys
from pathlib import Path

from src.config import settings
from src.ingestion.chunk import chunk_document, save_chunks
from src.ingestion.models import DocumentManifest
from src.ingestion.parse import parse_pdf
from src.ingestion.render import render_pdf_pages


def process_pdf(pdf_path: Path) -> DocumentManifest | None:
    """Process a single PDF through the full pipeline.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        DocumentManifest if successful, None if already processed.
    """
    doc_id = pdf_path.stem
    parsed_dir = settings.corpus_secondary_dir / "parsed" / doc_id
    manifest_path = parsed_dir / "manifest.json"

    # Check if already processed
    if manifest_path.exists():
        print(f"  {doc_id}: Already processed, skipping...", file=sys.stderr)
        with open(manifest_path) as f:
            data = json.load(f)
        return DocumentManifest(**data)

    print(f"  {doc_id}: Parsing...", file=sys.stderr)
    doc_data = parse_pdf(pdf_path)
    n_pages = len(doc_data.get("pages", {}))
    assert n_pages > 0, f"Docling produced 0 pages for {pdf_path}"

    print(f"  {doc_id}: Rendering {n_pages} pages...", file=sys.stderr)
    render_pdf_pages(pdf_path)

    print(f"  {doc_id}: Chunking...", file=sys.stderr)
    chunks = chunk_document(doc_id)
    save_chunks(doc_id, chunks)

    # Count chunks by type
    chunks_by_type: dict[str, int] = {}
    for chunk in chunks:
        chunks_by_type[chunk.chunk_type] = chunks_by_type.get(chunk.chunk_type, 0) + 1

    manifest = DocumentManifest(
        doc_id=doc_id,
        n_pages=n_pages,
        n_chunks=len(chunks),
        chunks_by_type=chunks_by_type,
    )

    # Save manifest
    with open(manifest_path, "w") as f:
        f.write(manifest.model_dump_json(indent=2))

    print(f"  {doc_id}: Done ({n_pages} pages, {len(chunks)} chunks)", file=sys.stderr)
    return manifest


def main() -> None:
    """Run ingestion on all secondary corpus PDFs."""
    raw_dir = settings.corpus_secondary_dir / "raw"

    if not raw_dir.exists():
        print("No secondary PDFs to ingest, skipping", file=sys.stderr)
        return

    pdfs = list(raw_dir.glob("*.pdf"))
    if not pdfs:
        print("No secondary PDFs to ingest, skipping", file=sys.stderr)
        return

    print(f"Processing {len(pdfs)} PDFs...", file=sys.stderr)

    manifests: list[DocumentManifest] = []
    for pdf_path in pdfs:
        manifest = process_pdf(pdf_path)
        if manifest:
            manifests.append(manifest)

    # Summary
    print("\n" + "=" * 60, file=sys.stderr)
    print("Ingestion Summary:", file=sys.stderr)
    total_pages = sum(m.n_pages for m in manifests)
    total_chunks = sum(m.n_chunks for m in manifests)
    print(f"  Documents: {len(manifests)}", file=sys.stderr)
    print(f"  Total pages: {total_pages}", file=sys.stderr)
    print(f"  Total chunks: {total_chunks}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
