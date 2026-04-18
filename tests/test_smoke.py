"""Smoke tests to verify all packages can be imported."""


def test_import_src() -> None:
    """Test that the main src package can be imported."""
    import src  # noqa: F401


def test_import_config() -> None:
    """Test that config module can be imported."""
    from src.config import settings
    assert settings is not None


def test_import_data() -> None:
    """Test that data package can be imported."""
    import src.data  # noqa: F401


def test_import_ingestion() -> None:
    """Test that ingestion package can be imported."""
    import src.ingestion  # noqa: F401


def test_import_indexing() -> None:
    """Test that indexing package can be imported."""
    import src.indexing  # noqa: F401


def test_import_retrieval() -> None:
    """Test that retrieval package can be imported."""
    import src.retrieval  # noqa: F401


def test_import_generation() -> None:
    """Test that generation package can be imported."""
    import src.generation  # noqa: F401


def test_import_graph() -> None:
    """Test that graph package can be imported."""
    import src.graph  # noqa: F401


def test_import_app() -> None:
    """Test that app package can be imported."""
    import src.app  # noqa: F401


def test_import_evaluation() -> None:
    """Test that evaluation package can be imported."""
    from src import eval as evaluation  # noqa: F401


def test_import_utils() -> None:
    """Test that utils package can be imported."""
    import src.utils  # noqa: F401
