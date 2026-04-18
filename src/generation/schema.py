"""Pydantic models for generation."""

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A citation to a source document."""

    doc_id: str = Field(..., description="Document identifier")
    page_number: int = Field(..., description="Page number")
    # Changed from required verbatim quote to short paraphrased snippet.
    # Verbatim long quotes trigger Gemini's RECITATION safety filter on
    # canonical documents like government reports. A short paraphrase
    # preserves the citation's purpose (pointing to source) without
    # reproducing training data.
    snippet: str = Field(
        ...,
        max_length=400,
        description=(
            "A short paraphrased snippet (≤30 words) describing what this "
            "source says, in your own words. Do NOT quote verbatim."
        ),
    )


class Answer(BaseModel):
    """Generated answer with citations."""

    answer: str = Field(..., description="The generated answer")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations supporting the answer",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Confidence level in the answer",
    )
    missing_info: str | None = Field(
        default=None,
        description="Information that was missing to fully answer",
    )
