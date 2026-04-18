"""Chunk documents using Docling's HybridChunker with chunk_type taxonomy."""

import hashlib
import json
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument

from src.config import settings
from src.ingestion.models import Chunk


def generate_chunk_id(doc_id: str, page_number: int, content: str) -> str:
    """Generate a unique chunk ID."""
    data = f"{doc_id}:{page_number}:{content}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _classify_chunk_type(meta) -> str:
    """Map Docling chunk metadata to our chunk_type taxonomy.

    Docling items carry labels like 'table', 'caption', 'footnote', 'text',
    'section_header', 'list_item', etc. We collapse these to our four-value
    enum: text / table / caption / footnote.
    """
    try:
        doc_items = getattr(meta, "doc_items", None) or []
        labels = [str(getattr(it, "label", "")).lower() for it in doc_items]
    except Exception:
        labels = []

    if any("table" in lbl for lbl in labels):
        return "table"
    if any("caption" in lbl for lbl in labels):
        return "caption"
    if any("footnote" in lbl for lbl in labels):
        return "footnote"
    return "text"


def _get_page_number(meta) -> int:
    """Extract the first page number from chunk metadata."""
    try:
        doc_items = getattr(meta, "doc_items", None) or []
        for item in doc_items:
            prov = getattr(item, "prov", None) or []
            for p in prov:
                page_no = getattr(p, "page_no", None)
                if page_no is not None:
                    return int(page_no)
    except Exception:
        pass
    return 1


def _get_section_path(meta) -> list[str]:
    """Extract section heading path from chunk metadata."""
    headings = getattr(meta, "headings", None)
    if headings:
        return [str(h) for h in headings]
    return []


def chunk_document(doc_id: str) -> list[Chunk]:
    """Extract chunks from a parsed DoclingDocument.

    Args:
        doc_id: Document identifier.

    Returns:
        List of Chunk objects.
    """
    parsed_dir = settings.corpus_secondary_dir / "parsed" / doc_id
    document_path = parsed_dir / "document.json"

    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    with open(document_path, encoding="utf-8") as f:
        doc_data = json.load(f)

    docling_doc = DoclingDocument.model_validate(doc_data)
    chunker = HybridChunker()

    chunks: list[Chunk] = []
    for docling_chunk in chunker.chunk(dl_doc=docling_doc):
        content = (docling_chunk.text or "").strip()
        if not content or len(content) < 10:
            continue

        meta = docling_chunk.meta
        chunk_type = _classify_chunk_type(meta)
        page_number = _get_page_number(meta)
        section_path = _get_section_path(meta)

        chunks.append(
            Chunk(
                chunk_id=generate_chunk_id(doc_id, page_number, content),
                doc_id=doc_id,
                page_number=page_number,
                section_path=section_path,
                chunk_type=chunk_type,
                content=content,
                bbox=None,
            )
        )

    return chunks


def save_chunks(doc_id: str, chunks: list[Chunk]) -> Path:
    """Save chunks to JSONL file."""
    output_dir = settings.corpus_secondary_dir / "parsed" / doc_id
    output_path = output_dir / "chunks.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")

    return output_path


if __name__ == "__main__":
    import sys

    parsed_dir = settings.corpus_secondary_dir / "parsed"
    if not parsed_dir.exists():
        print("No parsed documents found")
        sys.exit(0)

    doc_dirs = [d for d in parsed_dir.iterdir() if d.is_dir()]
    if not doc_dirs:
        print("No document directories found")
        sys.exit(0)

    doc_id = doc_dirs[0].name
    print(f"Chunking {doc_id}...")
    chunks = chunk_document(doc_id)
    print(f"  Generated {len(chunks)} chunks")

    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
    print(f"  By type: {by_type}")
