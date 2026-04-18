"""Tests for data download module."""

import json
from pathlib import Path

import pytest

from src.config import settings


@pytest.fixture
def qa_jsonl_path() -> Path:
    """Path to the QA JSONL file."""
    return settings.corpus_primary_dir / "qa.jsonl"


def test_qa_file_exists(qa_jsonl_path: Path) -> None:
    """Test that qa.jsonl exists after download."""
    if not qa_jsonl_path.exists():
        pytest.skip("qa.jsonl not found - run download first")
    assert qa_jsonl_path.exists()


def test_qa_has_enough_records(qa_jsonl_path: Path) -> None:
    """Test that qa.jsonl has at least 90 records."""
    if not qa_jsonl_path.exists():
        pytest.skip("qa.jsonl not found - run download first")

    records = []
    with open(qa_jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))

    assert len(records) >= 90, f"Expected at least 90 records, got {len(records)}"


def test_qa_records_have_required_fields(qa_jsonl_path: Path) -> None:
    """Test that each QA record has required fields."""
    if not qa_jsonl_path.exists():
        pytest.skip("qa.jsonl not found - run download first")

    required_fields = {"query", "answer", "page", "image_filename", "source"}

    with open(qa_jsonl_path) as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            missing = required_fields - set(record.keys())
            assert not missing, f"Record {i} missing fields: {missing}"
            assert record["query"], f"Record {i} has empty query"
            assert record["page"] is not None, f"Record {i} has no page number"


def test_pages_directory_has_images() -> None:
    """Test that pages directory contains PNG images."""
    pages_dir = settings.corpus_primary_dir / "pages"
    if not pages_dir.exists():
        pytest.skip("pages directory not found - run download first")

    png_files = list(pages_dir.glob("*.png"))
    assert len(png_files) >= 900, f"Expected at least 900 PNGs, got {len(png_files)}"
