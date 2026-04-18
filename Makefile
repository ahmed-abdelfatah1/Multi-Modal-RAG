.PHONY: setup download ingest index eval app test lint typecheck clean

# Install dependencies with uv
setup:
	uv sync

# Download datasets (HuggingFace + secondary PDFs)
download:
	uv run python -m src.data.download

# Run ingestion pipeline on secondary corpus
ingest:
	uv run python -m src.ingestion.run_all

# Build all indices (text + vision)
index:
	uv run python -m src.indexing.build_all

# Run evaluation suite
eval:
	uv run python -m src.eval.run

# Launch Streamlit app
app:
	uv run streamlit run src/app/main.py

# Run tests (excluding integration tests)
test:
	uv run pytest -q -m "not integration"

# Run all tests including integration
test-all:
	uv run pytest -q

# Run linter
lint:
	uv run ruff check src

# Run type checker
typecheck:
	uv run mypy src

# Clean generated files
clean:
	rm -rf data/corpus_primary
	rm -rf data/corpus_secondary
	rm -rf data/indices
	rm -rf data/eval
	rm -rf data/gemini_cache
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Start Qdrant via Docker
qdrant-up:
	docker compose up -d

# Stop Qdrant
qdrant-down:
	docker compose down
