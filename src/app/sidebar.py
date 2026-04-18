"""Sidebar rendering and status checks for the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from src.config import settings


@dataclass(frozen=True)
class SidebarState:
    """Snapshot of sidebar controls for one render pass."""

    mode: str
    top_k: int
    corpora: tuple[str, ...]
    show_heatmap: bool


@st.cache_data(ttl=30)
def check_qdrant() -> tuple[bool, str]:
    """Return (ok, message) for Qdrant connectivity. Cached briefly."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            host=settings.qdrant_url.replace("http://", "").replace("https://", ""),
            port=settings.qdrant_port,
            timeout=2,
        )
        collections = client.get_collections()
        return True, f"{len(collections.collections)} collections"
    except Exception as exc:  # noqa: BLE001 - surface any failure mode
        return False, str(exc)[:60]


@st.cache_data(ttl=300)
def check_gemini() -> tuple[bool, str]:
    """Return (ok, message) for Gemini API key presence. Never reveals the key."""
    if not settings.gemini_api_key:
        return False, "GEMINI_API_KEY not set"
    # Do not display or log the key itself
    return True, "API key configured"


def _device_label() -> str:
    """Human-readable device label for ColSmol."""
    return settings.device


def _badge(ok: bool, label: str, detail: str) -> None:
    """Render a colored status line."""
    color = "green" if ok else "red"
    icon = "OK" if ok else "DOWN"
    st.markdown(f":{color}[**{icon}**] {label}: {detail}")


def render_sidebar() -> SidebarState:
    """Render the sidebar and return the current control state."""
    with st.sidebar:
        st.title("Settings")

        st.subheader("Corpus filter")
        use_primary = st.checkbox("primary (HF dataset)", value=True, key="corpus_primary")
        use_secondary = st.checkbox("secondary (PDFs)", value=True, key="corpus_secondary")
        corpora: list[str] = []
        if use_primary:
            corpora.append("primary")
        if use_secondary:
            corpora.append("secondary")

        st.subheader("Retriever")
        mode = st.radio(
            "Mode",
            options=["hybrid", "vision_only", "text_only"],
            index=0,
            key="retriever_mode",
            help="Hybrid fuses text + vision with Reciprocal Rank Fusion",
        )
        top_k = st.slider(
            "Top-k",
            min_value=3,
            max_value=10,
            value=6,
            key="top_k_slider",
        )
        show_heatmap = st.checkbox(
            "Show similarity heatmap",
            value=False,
            key="show_heatmap",
            help="Toggle only; visualization not implemented in this step",
        )

        st.divider()
        st.subheader("Status")
        _badge(True, "ColSmol device", _device_label())
        q_ok, q_msg = check_qdrant()
        _badge(q_ok, "Qdrant", q_msg)
        g_ok, g_msg = check_gemini()
        _badge(g_ok, "Gemini API", g_msg)

        st.divider()
        if st.button("Clear chat", key="clear_chat_btn"):
            st.session_state.messages = []
            st.rerun()

    return SidebarState(
        mode=str(mode),
        top_k=int(top_k),
        corpora=tuple(corpora),
        show_heatmap=bool(show_heatmap),
    )
