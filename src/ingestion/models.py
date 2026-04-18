"""Pydantic models for ingestion pipeline."""

from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A text chunk extracted from a document."""

    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    doc_id: str = Field(..., description="Document identifier (PDF filename stem)")
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    section_path: list[str] = Field(
        default_factory=list,
        description="Hierarchy of section headings",
    )
    chunk_type: Literal["text", "table", "caption", "footnote"] = Field(
        ...,
        description="Type of content",
    )
    content: str = Field(..., min_length=1, description="The actual text content")
    bbox: list[float] | None = Field(
        default=None,
        description="Bounding box [x0, y0, x1, y1] if available",
    )


class DocumentManifest(BaseModel):
    """Manifest for a processed document."""

    doc_id: str
    n_pages: int
    n_chunks: int
    chunks_by_type: dict[str, int]
