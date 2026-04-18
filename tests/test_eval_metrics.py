"""Unit tests for the eval metrics module. No Qdrant or network required."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.metrics import (
    EvalRecord,
    compute_cache_hit_rate,
    compute_hit_at_k,
    compute_latency_stats,
    compute_mode_metrics,
    compute_mrr,
    load_records,
    per_mode_metrics,
)


def _make_record(
    expected: int,
    retrieved: list[int],
    latency: float = 100.0,
    cache_hit: bool | None = False,
) -> EvalRecord:
    """Build a minimal synthetic record for testing."""
    return EvalRecord(
        query="q",
        expected_page=expected,
        expected_answer="a",
        retrieved_pages=retrieved,
        generated_answer=None,
        latency_ms=latency,
        cache_hit=cache_hit if cache_hit is not None else False,
    )


def test_hit_at_k_basic() -> None:
    records = [
        _make_record(1, [1, 2, 3, 4, 5]),
        _make_record(3, [1, 2, 3, 4, 5]),
        _make_record(9, [1, 2, 3, 4, 5]),
    ]
    assert compute_hit_at_k(records, 1) == pytest.approx(1 / 3)
    assert compute_hit_at_k(records, 3) == pytest.approx(2 / 3)
    assert compute_hit_at_k(records, 5) == pytest.approx(2 / 3)


def test_hit_at_k_empty() -> None:
    assert compute_hit_at_k([], 1) == 0.0
    assert compute_hit_at_k([], 5) == 0.0


def test_hit_at_k_short_retrieval() -> None:
    records = [_make_record(7, [7, 8])]
    assert compute_hit_at_k(records, 5) == 1.0
    assert compute_hit_at_k(records, 1) == 1.0


def test_mrr_basic() -> None:
    records = [
        _make_record(1, [1, 2, 3]),
        _make_record(3, [1, 2, 3]),
        _make_record(9, [1, 2, 3]),
    ]
    expected = (1.0 + 1 / 3 + 0.0) / 3
    assert compute_mrr(records) == pytest.approx(expected)


def test_mrr_empty() -> None:
    assert compute_mrr([]) == 0.0


def test_latency_stats() -> None:
    records = [_make_record(1, [1], latency=x) for x in [100.0, 200.0, 300.0, 400.0, 500.0]]
    stats = compute_latency_stats(records)
    assert stats["mean"] == pytest.approx(300.0)
    assert stats["p50"] == pytest.approx(300.0)
    assert stats["p95"] == pytest.approx(500.0)


def test_latency_stats_empty() -> None:
    stats = compute_latency_stats([])
    assert stats == {"mean": 0.0, "p50": 0.0, "p95": 0.0}


def test_cache_hit_rate() -> None:
    records = [
        _make_record(1, [1], cache_hit=True),
        _make_record(2, [2], cache_hit=False),
        _make_record(3, [3], cache_hit=True),
        _make_record(4, [4], cache_hit=True),
    ]
    assert compute_cache_hit_rate(records) == pytest.approx(3 / 4)


def test_cache_hit_rate_all_none() -> None:
    records = [
        EvalRecord(
            query="q",
            expected_page=1,
            expected_answer="a",
            retrieved_pages=[1],
            generated_answer=None,
            latency_ms=1.0,
            cache_hit=None,  # type: ignore[typeddict-item]
        ),
    ]
    assert compute_cache_hit_rate(records) == 0.0


def test_compute_mode_metrics_shape() -> None:
    # Record 1: expected=1 at rank 1 in [1,2,3,4,5] -> hit@1
    # Record 2: expected=2 at rank 2 in [1,2,3,4,5] -> hit@3 but not hit@1
    records = [
        _make_record(1, [1, 2, 3, 4, 5], latency=50.0, cache_hit=True),
        _make_record(2, [1, 2, 3, 4, 5], latency=150.0, cache_hit=False),
    ]
    m = compute_mode_metrics(records)
    assert m["n"] == 2
    assert m["hit_at_1"] == pytest.approx(0.5)
    assert m["hit_at_3"] == pytest.approx(1.0)
    assert m["hit_at_5"] == pytest.approx(1.0)
    assert m["mrr"] == pytest.approx((1.0 + 0.5) / 2)
    assert m["mean_latency_ms"] == pytest.approx(100.0)
    assert m["cache_hit_rate"] == pytest.approx(0.5)


def test_load_records_missing(tmp_path: Path) -> None:
    assert load_records(tmp_path / "nope.jsonl") == []


def test_load_records_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "mode.jsonl"
    records = [_make_record(1, [1, 2]), _make_record(5, [1, 2, 3, 4, 5])]
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    loaded = load_records(path)
    assert len(loaded) == 2
    assert loaded[0]["expected_page"] == 1
    assert loaded[1]["retrieved_pages"] == [1, 2, 3, 4, 5]


def test_per_mode_metrics(tmp_path: Path) -> None:
    for mode, pages in [("text_only", [1, 2]), ("hybrid", [5, 1])]:
        path = tmp_path / f"{mode}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(_make_record(pages[0], pages, latency=10.0)) + "\n")

    metrics = per_mode_metrics(tmp_path)
    assert set(metrics.keys()) == {"text_only", "hybrid"}
    assert metrics["text_only"]["n"] == 1
    assert metrics["text_only"]["hit_at_1"] == 1.0


def test_per_mode_metrics_missing_dir(tmp_path: Path) -> None:
    result = per_mode_metrics(tmp_path / "does_not_exist")
    assert result == {}
