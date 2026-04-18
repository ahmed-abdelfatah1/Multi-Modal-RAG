"""Chat rendering helpers for the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from src.config import settings
from src.generation.schema import Answer, Citation
from src.retrieval.types import RetrievalResult, Retrieved

# Keep UI thumbnails reasonable; four columns fit most screens.
MAX_SOURCE_COLUMNS = 4
SNIPPET_PREVIEW_CHARS = 240


def find_image_path(image_filename: str, source_corpus: str | None) -> Path | None:
    """Locate an image on disk given filename and source corpus.

    Mirrors the pattern in src/generation/generator.py::_find_image.
    """
    if not image_filename:
        return None

    if source_corpus == "primary":
        candidate = settings.corpus_primary_dir / "pages" / image_filename
        if candidate.exists():
            return candidate

    if source_corpus == "secondary":
        parsed_dir = settings.corpus_secondary_dir / "parsed"
        if parsed_dir.exists():
            for doc_dir in parsed_dir.iterdir():
                candidate = doc_dir / "pages" / image_filename
                if candidate.exists():
                    return candidate

    # Last-ditch fallback search across both corpora
    for corpus_dir in (settings.corpus_primary_dir, settings.corpus_secondary_dir):
        if corpus_dir.exists():
            for candidate in corpus_dir.rglob(image_filename):
                return candidate
    return None


def _find_retrieved_for_citation(
    retrieved: RetrievalResult | None,
    citation: Citation,
) -> Retrieved | None:
    if retrieved is None:
        return None
    for item in retrieved.items:
        if item.doc_id == citation.doc_id and item.page_number == citation.page_number:
            return item
    return None


def render_sources(answer: Answer, retrieved: RetrievalResult | None) -> None:
    """Render the 'Sources' expander with one column per citation."""
    if not answer.citations:
        st.info("No citations returned with this answer.")
        return

    with st.expander("Sources", expanded=True):
        citations = answer.citations
        n_cols = min(len(citations), MAX_SOURCE_COLUMNS)
        cols = st.columns(n_cols)

        for i, citation in enumerate(citations):
            col = cols[i % n_cols]
            with col:
                st.caption(f"**{citation.doc_id}** — p.{citation.page_number}")

                item = _find_retrieved_for_citation(retrieved, citation)
                if item is not None:
                    image_filename = item.payload.get("image_filename")
                    source_corpus = item.payload.get("source_corpus")
                    path = find_image_path(image_filename, source_corpus) if image_filename else None
                    if path is not None:
                        try:
                            img = Image.open(path)
                            st.image(img, use_container_width=True)
                        except Exception as exc:  # noqa: BLE001
                            st.warning(f"Could not render image: {exc}")

                snippet = (citation.snippet or "").strip()
                if snippet:
                    shown = snippet if len(snippet) <= SNIPPET_PREVIEW_CHARS else snippet[: SNIPPET_PREVIEW_CHARS] + "..."
                    st.markdown(f"> _{shown}_")


def render_also_considered(
    retrieved: RetrievalResult,
    answer: Answer,
) -> None:
    """Render the 'Also considered' expander with non-cited retrieved items."""
    cited: set[tuple[str, int]] = {(c.doc_id, c.page_number) for c in answer.citations}
    non_cited = [
        item
        for item in retrieved.items
        if (item.doc_id, item.page_number) not in cited
    ]
    if not non_cited:
        return

    with st.expander("Also considered", expanded=False):
        for item in non_cited:
            st.markdown(
                f"- `[{item.source_type}]` **{item.doc_id}** p.{item.page_number}"
                f" — score `{item.score:.3f}`"
            )


def render_assistant_turn(
    answer: Answer,
    retrieved: RetrievalResult | None,
    mode: str,
    cache_hit: bool,
) -> None:
    """Render a full assistant turn: answer markdown, cache badge, sources."""
    st.markdown(answer.answer)

    confidence_colors = {"high": "green", "medium": "orange", "low": "red"}
    color = confidence_colors.get(answer.confidence, "gray")
    badge_cols = st.columns([1, 1, 6])
    with badge_cols[0]:
        st.markdown(f":{color}[confidence: **{answer.confidence}**]")
    with badge_cols[1]:
        if cache_hit:
            st.markdown(":violet[**cache hit**]")
    if answer.missing_info:
        st.caption(f"Missing info: {answer.missing_info}")

    render_sources(answer, retrieved)

    if mode == "hybrid" and retrieved is not None:
        render_also_considered(retrieved, answer)
