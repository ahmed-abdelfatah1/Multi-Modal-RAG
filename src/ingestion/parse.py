"""PDF parsing with Docling (table structure + layout analysis)."""

import json
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.config import settings


def get_converter() -> DocumentConverter:
    """Build a Docling DocumentConverter with table structure (no OCR)."""
    pipeline_options = PdfPipelineOptions()
    # Disabled because the secondary corpus is born-digital. Re-enable
    # per-document via a flag if we ever ingest scanned material.
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )


def parse_pdf(pdf_path: Path) -> dict:
    """Parse a PDF and produce a DoclingDocument JSON export.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dictionary with DoclingDocument data (from export_to_dict).
    """
    doc_id = pdf_path.stem
    output_dir = settings.corpus_secondary_dir / "parsed" / doc_id
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = get_converter()
    result = converter.convert(str(pdf_path))
    docling_doc = result.document

    # Serialize DoclingDocument as JSON
    doc_data = docling_doc.export_to_dict()

    output_path = output_dir / "document.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc_data, f, indent=2, ensure_ascii=False)

    return doc_data


if __name__ == "__main__":
    import sys

    raw_dir = settings.corpus_secondary_dir / "raw"
    pdfs = list(raw_dir.glob("*.pdf"))

    if not pdfs:
        print("No PDFs found in", raw_dir)
        sys.exit(0)

    pdf = pdfs[0]
    print(f"Parsing {pdf.name} with Docling...")
    result = parse_pdf(pdf)
    n_pages = len(result.get("pages", {}))
    assert n_pages > 0, f"Docling produced 0 pages for {pdf}"
    print(f"  Exported {len(json.dumps(result))} bytes of JSON ({n_pages} pages)")
