"""Retrieval and generation metrics for the evaluation suite.

Computes Hit@k, MRR, latency statistics, and cache-hit rate per mode.
Ragas-style faithfulness/answer-relevancy are intentionally out of scope
for v1 (future work) to avoid pulling heavyweight dependencies.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import TypedDict


class EvalRecord(TypedDict, total=False):
    """One line from ``data/eval/runs/{mode}.jsonl``."""

    query: str
    expected_page: int
    expected_answer: str
    retrieved_pages: list[int]
    generated_answer: str | None
    latency_ms: float
    cache_hit: bool


# Typed dict describing the per-mode metrics we expose.
class ModeMetrics(TypedDict):
    """Aggregate metrics for a single retrieval mode."""

    n: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    cache_hit_rate: float


def load_records(path: Path) -> list[EvalRecord]:
    """Load eval records from a JSONL file. Returns [] if missing."""
    if not path.exists():
        return []
    records: list[EvalRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_hit_at_k(records: list[EvalRecord], k: int) -> float:
    """Fraction of records where expected_page is in retrieved_pages[:k]."""
    if not records:
        return 0.0
    hits = 0
    for r in records:
        retrieved = r.get("retrieved_pages", []) or []
        expected = r.get("expected_page")
        if expected is None:
            continue
        if expected in retrieved[:k]:
            hits += 1
    return hits / len(records)


def compute_mrr(records: list[EvalRecord]) -> float:
    """Mean reciprocal rank of expected_page within retrieved_pages."""
    if not records:
        return 0.0
    total = 0.0
    for r in records:
        retrieved = r.get("retrieved_pages", []) or []
        expected = r.get("expected_page")
        if expected is None:
            continue
        try:
            rank = retrieved.index(expected) + 1
            total += 1.0 / rank
        except ValueError:
            # Not found: reciprocal rank = 0
            continue
    return total / len(records)


def _percentile(values: list[float], pct: float) -> float:
    """Compute a percentile using nearest-rank (no numpy dependency)."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    # Nearest-rank method; clamp index to valid range.
    idx = max(0, min(len(sorted_v) - 1, int(round(pct / 100.0 * (len(sorted_v) - 1)))))
    return sorted_v[idx]


def compute_latency_stats(records: list[EvalRecord]) -> dict[str, float]:
    """Return mean, p50, p95 of latency_ms across records."""
    latencies = [float(r["latency_ms"]) for r in records if "latency_ms" in r]
    if not latencies:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": statistics.fmean(latencies),
        "p50": _percentile(latencies, 50),
        "p95": _percentile(latencies, 95),
    }


def compute_cache_hit_rate(records: list[EvalRecord]) -> float:
    """Fraction of records with cache_hit == True (ignores None/missing)."""
    candidates = [r for r in records if r.get("cache_hit") is not None]
    if not candidates:
        return 0.0
    hits = sum(1 for r in candidates if r.get("cache_hit"))
    return hits / len(candidates)


def compute_mode_metrics(records: list[EvalRecord]) -> ModeMetrics:
    """Compute all aggregate metrics for a single mode's records."""
    latency = compute_latency_stats(records)
    return ModeMetrics(
        n=len(records),
        hit_at_1=compute_hit_at_k(records, 1),
        hit_at_3=compute_hit_at_k(records, 3),
        hit_at_5=compute_hit_at_k(records, 5),
        mrr=compute_mrr(records),
        mean_latency_ms=latency["mean"],
        p50_latency_ms=latency["p50"],
        p95_latency_ms=latency["p95"],
        cache_hit_rate=compute_cache_hit_rate(records),
    )


def per_mode_metrics(runs_dir: Path) -> dict[str, ModeMetrics]:
    """Load all per-mode JSONL files and compute metrics for each mode."""
    result: dict[str, ModeMetrics] = {}
    if not runs_dir.exists():
        return result
    for path in sorted(runs_dir.glob("*.jsonl")):
        mode = path.stem
        records = load_records(path)
        result[mode] = compute_mode_metrics(records)
    return result


if __name__ == "__main__":
    # Smoke test on synthetic data (no indices required).
    sample: list[EvalRecord] = [
        {
            "query": "q1",
            "expected_page": 5,
            "expected_answer": "a",
            "retrieved_pages": [5, 2, 3, 4, 1],
            "generated_answer": None,
            "latency_ms": 120.0,
            "cache_hit": False,
        },
        {
            "query": "q2",
            "expected_page": 7,
            "expected_answer": "b",
            "retrieved_pages": [1, 2, 7, 4, 5],
            "generated_answer": None,
            "latency_ms": 240.0,
            "cache_hit": True,
        },
    ]
    print("hit@1:", compute_hit_at_k(sample, 1))
    print("hit@3:", compute_hit_at_k(sample, 3))
    print("mrr:", compute_mrr(sample))
    print("latency:", compute_latency_stats(sample))
    print("cache_hit_rate:", compute_cache_hit_rate(sample))
