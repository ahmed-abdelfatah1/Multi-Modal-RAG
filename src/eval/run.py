"""Run retrieval (3 modes) + generation on the QA set and write JSONL results.

For each query in ``data/corpus_primary/qa.jsonl``: retrieve in text_only,
vision_only, hybrid; generate on HYBRID only (saves Gemini quota); emit
per-query JSONL records to ``data/eval/runs/{mode}.jsonl``. Idempotent unless
``--recreate`` is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from src.config import settings
from src.generation.generator import Generator, get_generator
from src.retrieval.hybrid import HybridRetriever, RetrieverMode, get_hybrid_retriever
from src.retrieval.types import RetrievalResult

# Rate-limit: Gemini free tier ~10 RPM for 2.5 Flash.
MIN_SECONDS_BETWEEN_GEMINI_CALLS = 7
TOP_K = 5


def load_qa(path: Path, n: int | None) -> list[dict[str, object]]:
    """Load QA records, casting page numbers to int."""
    records: list[dict[str, object]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Page numbers are strings in qa.jsonl; cast to int.
            try:
                rec["page"] = int(rec["page"])
            except (KeyError, ValueError, TypeError):
                rec["page"] = -1
            records.append(rec)
    if n is not None:
        records = records[:n]
    return records


def _retrieve_safely(
    retriever: HybridRetriever,
    query: str,
    top_k: int,
) -> tuple[RetrievalResult, float]:
    """Run retrieval and return (result, latency_ms). Returns empty on error."""
    start = time.perf_counter()
    try:
        result = retriever.retrieve(query, top_k=top_k)
    except Exception as e:
        print(f"  retrieval error: {e}", file=sys.stderr)
        result = RetrievalResult(query=query, items=[], mode=retriever.mode.value)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return result, latency_ms


def _generate_safely(
    generator: Generator, query: str, retrieved: RetrievalResult
) -> tuple[str | None, bool]:
    """Generate an answer and detect a cache hit. On error returns (None, cache_hit)."""
    cache_hit = False
    try:
        contents, image_hashes = generator._build_contents(query, retrieved, regen=False)
        config_str = f"temp={settings.temperature}|max={settings.max_new_tokens}"
        prompt_text = "".join(str(c) for c in contents if isinstance(c, str))
        cache_hit = generator.cache.get(prompt_text, image_hashes, config_str) is not None
    except Exception as e:
        print(f"  cache lookup error: {e}", file=sys.stderr)
    try:
        return generator.generate(query, retrieved).answer, cache_hit
    except Exception as e:
        print(f"  generation error: {e}", file=sys.stderr)
        return None, cache_hit


def _retrieved_pages(result: RetrievalResult, limit: int = 5) -> list[int]:
    """Extract ordered, deduplicated top-N page numbers from a result."""
    seen: set[int] = set()
    pages: list[int] = []
    for item in result.items:
        p = int(item.page_number)
        if p in seen:
            continue
        seen.add(p)
        pages.append(p)
        if len(pages) >= limit:
            break
    return pages


def run_all(
    qa_path: Path,
    output_dir: Path,
    n: int | None = None,
    recreate: bool = False,
) -> dict[str, Path]:
    """Run retrieval in all 3 modes + generation on hybrid. Returns mode->path."""
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = [RetrieverMode.TEXT_ONLY, RetrieverMode.VISION_ONLY, RetrieverMode.HYBRID]
    out_paths = {m.value: output_dir / f"{m.value}.jsonl" for m in modes}

    # Idempotency check.
    if not recreate and all(p.exists() for p in out_paths.values()):
        print(f"All run files exist in {output_dir}. Use --recreate to overwrite.")
        return out_paths

    qa_records = load_qa(qa_path, n)
    print(f"Loaded {len(qa_records)} QA records from {qa_path}")

    retrievers = {m: get_hybrid_retriever(m) for m in modes}
    generator = get_generator(min_seconds_between_calls=MIN_SECONDS_BETWEEN_GEMINI_CALLS)

    # Open all output files inside an ExitStack so they all close together.
    with ExitStack() as stack:
        files = {
            m.value: stack.enter_context(open(out_paths[m.value], "w", encoding="utf-8"))
            for m in modes
        }
        summary: dict[str, dict[str, int]] = {m.value: {"ok": 0, "miss": 0} for m in modes}

        for i, qa in enumerate(qa_records, start=1):
            query = str(qa.get("query", ""))
            # "page" was normalized to int in load_qa, but we cast defensively.
            page_val = qa.get("page", -1)
            expected_page = int(page_val) if isinstance(page_val, int) else -1
            expected_answer = str(qa.get("answer", ""))

            print(f"[{i}/{len(qa_records)}] {query[:70]}")

            # Retrieve in all 3 modes.
            results: dict[str, tuple[RetrievalResult, float]] = {}
            for mode in modes:
                results[mode.value] = _retrieve_safely(retrievers[mode], query, TOP_K)

            # Generate once on hybrid.
            hybrid_result, _ = results[RetrieverMode.HYBRID.value]
            gen_answer, cache_hit = _generate_safely(generator, query, hybrid_result)

            for mode in modes:
                result, latency_ms = results[mode.value]
                pages = _retrieved_pages(result, limit=5)
                record = {
                    "query": query,
                    "expected_page": expected_page,
                    "expected_answer": expected_answer,
                    "retrieved_pages": pages,
                    "generated_answer": gen_answer if mode == RetrieverMode.HYBRID else None,
                    "latency_ms": round(latency_ms, 2),
                    "cache_hit": cache_hit if mode == RetrieverMode.HYBRID else None,
                }
                files[mode.value].write(json.dumps(record, ensure_ascii=False) + "\n")
                files[mode.value].flush()

                if expected_page in pages:
                    summary[mode.value]["ok"] += 1
                else:
                    summary[mode.value]["miss"] += 1

    # Summary table.
    print("\n=== Summary (top-5 hit counts) ===")
    print(f"{'mode':<14} {'ok':>6} {'miss':>6}")
    for mode in modes:
        s = summary[mode.value]
        print(f"{mode.value:<14} {s['ok']:>6} {s['miss']:>6}")

    return out_paths


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run retrieval + generation eval")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N queries")
    parser.add_argument("--recreate", action="store_true", help="Overwrite existing run files")
    args = parser.parse_args()

    qa_path = settings.corpus_primary_dir / "qa.jsonl"
    output_dir = settings.eval_dir / "runs"

    if not qa_path.exists():
        print(f"QA file not found: {qa_path}", file=sys.stderr)
        sys.exit(1)

    run_all(qa_path, output_dir, n=args.n, recreate=args.recreate)


if __name__ == "__main__":
    main()
