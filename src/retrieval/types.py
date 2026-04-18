"""Type definitions for retrieval."""

from typing import Literal

from pydantic import BaseModel, Field


class Retrieved(BaseModel):
    """A single retrieved item."""

    id: str = Field(..., description="Unique identifier")
    source_type: Literal["text", "page"] = Field(
        ...,
        description="Whether this is from text or page index",
    )
    doc_id: str = Field(..., description="Document identifier")
    page_number: int = Field(..., description="Page number")
    score: float = Field(..., description="Retrieval score")
    payload: dict = Field(default_factory=dict, description="Additional metadata")


class RetrievalResult(BaseModel):
    """Result of a retrieval operation."""

    query: str = Field(..., description="Original query")
    items: list[Retrieved] = Field(default_factory=list, description="Retrieved items")
    mode: str = Field(default="hybrid", description="Retrieval mode used")

    def get_unique_pages(self) -> list[tuple[str, int]]:
        """Get unique (doc_id, page_number) pairs."""
        seen: set[tuple[str, int]] = set()
        unique = []
        for item in self.items:
            key = (item.doc_id, item.page_number)
            if key not in seen:
                seen.add(key)
                unique.append(key)
        return unique
