"""Streamlit chat UI for the Multi-Modal RAG QA system.

Layout and state orchestration only; rendering helpers live in chat.py and
sidebar controls in sidebar.py. Keeps this module under the 200-line cap.
"""

from __future__ import annotations

import contextlib
import io
from typing import TypedDict

import streamlit as st

from src.app.chat import render_assistant_turn
from src.app.sidebar import SidebarState, render_sidebar
from src.generation.schema import Answer
from src.graph.qa_graph import get_qa_graph
from src.retrieval.hybrid import HybridRetriever, RetrieverMode, get_hybrid_retriever
from src.retrieval.types import RetrievalResult, Retrieved

PAGE_TITLE = "Multi-Modal RAG QA — DSAI 413"


class ChatTurn(TypedDict, total=False):
    """One entry in st.session_state.messages."""

    role: str
    content: str
    answer: Answer
    retrieved: RetrievalResult | None
    mode: str
    cache_hit: bool


# Page config must be the first Streamlit call.
st.set_page_config(page_title=PAGE_TITLE, layout="wide")


@st.cache_resource(show_spinner="Loading retrievers (ColSmol + bge-small-en)...")
def _load_retriever(mode: str) -> HybridRetriever:
    """Warm up and cache the retriever for a given mode.

    Uses ``st.cache_resource`` so ColSmol weights load once per app session.
    """
    return get_hybrid_retriever(RetrieverMode(mode))


def _filter_by_corpora(
    retrieved: RetrievalResult | None,
    corpora: tuple[str, ...],
) -> RetrievalResult | None:
    """Drop retrieved items whose source_corpus is not in the allowed set."""
    if retrieved is None or not corpora:
        return retrieved
    allowed = set(corpora)
    filtered: list[Retrieved] = [
        item for item in retrieved.items
        if item.payload.get("source_corpus", "primary") in allowed
    ]
    return RetrievalResult(query=retrieved.query, items=filtered, mode=retrieved.mode)


def _run_pipeline(
    query: str,
    state: SidebarState,
) -> tuple[Answer, RetrievalResult | None, bool]:
    """Run the QA graph and report (answer, retrieved, cache_hit).

    Cache-hit detection: the Generator prints "cache hit" / "cache miss" to
    stderr on every call. Capturing stderr lets us surface that to the UI
    without modifying the Generator API.
    """
    # Ensure retrievers are warm before the graph runs.
    _load_retriever(state.mode)

    graph = get_qa_graph()  # type: ignore[no-untyped-call]  # upstream fn is untyped
    initial_state = {
        "query": query,
        "mode": state.mode,
        "retrieved": None,
        "answer": None,
        "regen_count": 0,
        "needs_regen": False,
    }

    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        result = graph.invoke(initial_state)
    cache_hit = "cache hit" in buffer.getvalue()

    answer: Answer = result["answer"]
    retrieved: RetrievalResult | None = result.get("retrieved")
    retrieved = _filter_by_corpora(retrieved, state.corpora)
    return answer, retrieved, cache_hit


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _replay_history() -> None:
    """Re-render prior chat turns from session state."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "answer" in msg:
                render_assistant_turn(
                    msg["answer"],
                    msg.get("retrieved"),
                    msg.get("mode", "hybrid"),
                    bool(msg.get("cache_hit", False)),
                )
            else:
                st.markdown(msg.get("content", ""))


def _handle_new_question(prompt: str, state: SidebarState) -> None:
    """Append user turn, run pipeline, append assistant turn."""
    user_turn: ChatTurn = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_turn)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not state.corpora:
            st.error("Select at least one corpus in the sidebar before asking.")
            return

        with st.spinner("Retrieving and generating..."):
            try:
                answer, retrieved, cache_hit = _run_pipeline(prompt, state)
            except Exception as exc:  # noqa: BLE001 - friendly banner, do not leak key
                msg = str(exc)
                st.error(f"Pipeline failed: {msg}")
                st.caption(
                    "Common causes: Qdrant not running (docker compose up -d), "
                    "Gemini rate limit (wait ~7s), or missing GEMINI_API_KEY."
                )
                error_turn: ChatTurn = {
                    "role": "assistant",
                    "content": f"Error: {msg}",
                }
                st.session_state.messages.append(error_turn)
                return

        render_assistant_turn(answer, retrieved, state.mode, cache_hit)
        assistant_turn: ChatTurn = {
            "role": "assistant",
            "content": answer.answer,
            "answer": answer,
            "retrieved": retrieved,
            "mode": state.mode,
            "cache_hit": cache_hit,
        }
        st.session_state.messages.append(assistant_turn)


# ---- top-level layout ------------------------------------------------------
_init_session_state()
_sidebar_state = render_sidebar()

st.title(PAGE_TITLE)
st.caption(
    "Ask questions about the indexed policy and financial documents. "
    "Retrieval uses ColSmol-256M (vision) + bge-small-en (text) with RRF; "
    "answers come from Gemini 2.5 Flash."
)

_replay_history()

_prompt = st.chat_input("Ask a question about the documents...")
if _prompt:
    _handle_new_question(_prompt, _sidebar_state)
