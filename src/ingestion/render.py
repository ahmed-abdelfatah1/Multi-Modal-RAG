"""Render PDF pages to PNG images using pdf2image."""

from pathlib import Path

from pdf2image import convert_from_path

from src.config import settings

# DPI for rendering (200 is good balance of quality and size)
RENDER_DPI = 200


def render_pdf_pages(pdf_path: Path) -> list[Path]:
    """Render all pages of a PDF to PNG images.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of paths to rendered PNG images.
    """
    doc_id = pdf_path.stem
    output_dir = settings.corpus_secondary_dir / "parsed" / doc_id / "pages"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already rendered (fast early-out)
    existing = sorted(output_dir.glob("page_*.png"))
    if existing:
        return existing

    # Render all pages at specified DPI
    poppler_path = str(settings.poppler_bin) if settings.poppler_bin else None
    images = convert_from_path(
        str(pdf_path), dpi=RENDER_DPI, fmt="png", poppler_path=poppler_path
    )

    rendered_paths: list[Path] = []
    for page_num, image in enumerate(images, start=1):
        output_path = output_dir / f"page_{page_num:03d}.png"
        image.save(output_path, "PNG")
        rendered_paths.append(output_path)

    return rendered_paths


if __name__ == "__main__":
    import sys

    raw_dir = settings.corpus_secondary_dir / "raw"
    pdfs = list(raw_dir.glob("*.pdf"))

    if not pdfs:
        print("No PDFs found in", raw_dir)
        sys.exit(0)

    pdf = pdfs[0]
    print(f"Rendering {pdf.name}...")
    paths = render_pdf_pages(pdf)
    print(f"  Rendered {len(paths)} pages")
    if paths:
        print(f"  First page: {paths[0]}")
