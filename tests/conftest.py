"""Pytest configuration and fixtures."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (may require API keys)",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow running",
    )
