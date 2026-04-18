"""Capture raw answer + retrieval data for the three demo queries.

Runs each query through the hybrid retriever and Gemini 2.5 Flash generator and
writes a JSON dump per query under ``docs/demo_assets/transcript/``. Also emits
an aggregated ``docs/demo_assets/demo_transcript.md`` suitable for the video
narration script.

This is a read-only script with respect to the indices (no writes to Qdrant)
and benefits from the Gemini cache: re-running it costs zero API calls if the
queries are already cached.

Run::

    uv run python scripts/capture_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.generation.generator import get_generator
from src.retrieval.hybrid import RetrieverMode, get_hybrid_retriever

QUERIES: list[dict[str, str]] = [
    {
        "id": "query_1",
        "label": "Primary corpus, both channels hit",
        "query": "What types of revenue should MMS account for as a custodial activity?",
    },
    {
        "id": "query_2",
        "label": "Secondary corpus (BIS), table/chart dominant",
        "query": "What does the BIS say about non-bank financial intermediation risks in 2024?",
    },
    {
        "id": "query_3",
        "label": "Primary corpus, hybrid recovers vision miss",
        "query": "How have communities adapted their recycling programs?",
    },
]

TOP_K = 6
OUT_DIR = Path("docs/demo_assets/transcript")
MD_PATH = Path("docs/demo_assets/demo_transcript.md")


def capture(query_spec: dict[str, str]) -> dict:
    """Run retrieval (hybrid) and generation for one query, return a dict."""
    retriever = get_hybrid_retriever(RetrieverMode.HYBRID)
    retrieved = retriever.retrieve(query_spec["query"], top_k=TOP_K)

    retrieved_dump = [
        {
            "rank": i,
            "source_type": item.source_type,
            "doc_id": item.doc_id,
            "page_number": int(item.page_number),
            "score": float(item.score),
        }
        for i, item in enumerate(retrieved.items, start=1)
    ]

    generator = get_generator()
    answer = generator.generate(query_spec["query"], retrieved)

    return {
        "id": query_spec["id"],
        "label": query_spec["label"],
        "query": query_spec["query"],
        "retrieved": retrieved_dump,
        "answer": answer.model_dump(),
    }


def write_markdown(results: list[dict]) -> None:
    """Render a narration-friendly markdown transcript from the JSON results."""
    lines: list[str] = [
        "# Demo transcript",
        "",
        "Raw output from `scripts/capture_demo.py`. Each section is one demo query.",
        "",
    ]
    for res in results:
        lines.append(f"## {res['id']} — {res['label']}")
        lines.append("")
        lines.append(f"**Query.** {res['query']}")
        lines.append("")
        lines.append("### Retrieved (top-6)")
        lines.append("")
        lines.append("| Rank | Source | doc_id | Page | Score |")
        lines.append("|---|---|---|---|---|")
        for r in res["retrieved"]:
            lines.append(
                f"| {r['rank']} | `{r['source_type']}` | `{r['doc_id']}` | "
                f"{r['page_number']} | {r['score']:.4f} |"
            )
        lines.append("")
        lines.append("### Generated answer")
        lines.append("")
        ans = res["answer"]
        lines.append(f"**Confidence.** `{ans.get('confidence')}`")
        lines.append("")
        lines.append(f"> {ans.get('answer', '').strip()}")
        lines.append("")
        citations = ans.get("citations") or []
        if citations:
            lines.append("**Citations.**")
            lines.append("")
            for i, c in enumerate(citations, start=1):
                snippet = c.get("snippet") or c.get("quote") or ""
                lines.append(
                    f"{i}. `{c.get('doc_id')}` p.{c.get('page_number')} — {snippet}"
                )
            lines.append("")
        missing = ans.get("missing_info")
        if missing:
            lines.append(f"**Missing info.** {missing}")
            lines.append("")
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for spec in QUERIES:
        print(f"[capture] {spec['id']}: {spec['query']}")
        try:
            res = capture(spec)
        except Exception as exc:  # noqa: BLE001 - surface any pipeline failure
            print(f"  ERROR: {exc}")
            res = {
                "id": spec["id"],
                "label": spec["label"],
                "query": spec["query"],
                "error": str(exc),
            }
        results.append(res)
        out_path = OUT_DIR / f"{spec['id']}.json"
        out_path.write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  wrote {out_path}")
    write_markdown(results)
    print(f"wrote {MD_PATH}")


if __name__ == "__main__":
    main()
