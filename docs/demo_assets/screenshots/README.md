# Screenshots

The three demo queries should be captured by running them through the live Streamlit
app at `http://localhost:8501` (or `:8502` if `:8501` is already bound) and saving the
full assistant turn (question → answer panel with citations + retrieved pages).

Expected files:

| File | Source query |
|---|---|
| `query_1_answer.png` | "What types of revenue should MMS account for as a custodial activity?" |
| `query_2_answer.png` | "What does the BIS say about non-bank financial intermediation risks in 2024?" |
| `query_3_answer.png` | "How have communities adapted their recycling programs?" |

## How to produce them

1. Start Qdrant if it is not already running: `docker compose up -d`.
2. Launch the app: `uv run streamlit run src/app/main.py`.
3. Open the browser at the URL Streamlit prints, ask each question in order, wait for
   the answer panel to finish rendering, and use the OS screenshot tool to capture the
   full chat card (question + answer + citations + retrieved page thumbnails).
4. Save the PNGs into this directory with the filenames above.

## Why not automated

Headless-browser automation of Streamlit (Playwright, Selenium) requires installing a
browser binary plus a ~200 MB dependency; this sits outside the locked dependency set
and the assignment's lightweight-stack constraint. Manual screenshots are fine for the
demo video — the underlying answer + retrieval data is already captured programmatically
in `docs/demo_assets/transcript/*.json` via `scripts/capture_demo.py`.
