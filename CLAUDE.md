# Project: dsai-413-multimodal-rag

## What this is
A multi-modal Retrieval-Augmented Generation QA system for policy and financial
documents. Vision-first retrieval via ColSmol-256M over page images, with a text
channel via bge-small-en for lexical queries, fused with reciprocal rank fusion.
Answer generation by Gemini 2.5 Flash over retrieved page images + text chunks.

## Hard constraints
- Retrieval stack is fully open-source, Hugging Face only.
- Generator is Gemini 2.5 Flash via the google-genai SDK. API key in env only.
- Vector store: Qdrant self-hosted in Docker. Not Pinecone, not Weaviate.
- Python 3.11, uv for deps, ruff + mypy, pytest.

## Stack
- **Vision retrieval**: `vidore/colSmol-256M` via `colpali-engine`
  (`ColIdefics3` / `ColIdefics3Processor`). 80.1 ViDoRe nDCG@5. Apache 2.0. ~512 MB.
- **Text retrieval**: `BAAI/bge-small-en-v1.5` via `sentence-transformers`. ~133 MB.
- **PDF parsing**: Docling (`DocumentConverter` + `HybridChunker`) with OCR and
  table structure recognition enabled. pdf2image for page rendering.
- **Generation**: Gemini 2.5 Flash API (no local inference).

## Gemini SDK — use ONLY these patterns
- Package: `google-genai` on PyPI. Import: `from google import genai`.
- Client: `genai.Client()` — auto-reads GEMINI_API_KEY from env.
- Model string: `gemini-2.5-flash` (stable).
- Call: `client.models.generate_content(model=..., contents=[...], config=types.GenerateContentConfig(...))`
- Pass PIL Images directly in `contents`, or use `types.Part.from_bytes(data=b, mime_type='image/png')`.
- Structured output: `response_mime_type='application/json'` + `response_schema=PydanticModel`, then read `response.parsed`.
- DO NOT use `google.generativeai` (deprecated legacy package).
- DO NOT use `genai.GenerationConfig` (old pattern).
- Always wrap calls with tenacity retry. Always cache by SHA-256 of prompt+images+config.

## Canonical commands
- Install deps: `uv sync`
- Start Qdrant: `docker compose up -d`
- Download data: `uv run python -m src.data.download`
- Ingest corpus: `uv run python -m src.ingestion.run_all`
- Build indices: `uv run python -m src.indexing.build_all`
- Run eval: `uv run python -m src.eval.run`
- Launch app: `uv run streamlit run src/app/main.py`
- Tests: `uv run pytest -q`
- Type check: `uv run mypy src`
- Lint: `uv run ruff check src`

## Code style
- 200-line max per module. If a file grows past that, split it.
- Every module has a __main__ smoke test under 10 seconds.
- Pydantic for all config and data models that cross module boundaries.
- No magic numbers. Constants live in src/config.py.
- Type hints everywhere. No `Any` unless commented why.

## Workflow
- Every commit message ends with a `Verify:` line showing the command that proves the change works.
- When verification fails twice, stop and ask; don't guess a third fix.
- When in doubt about a parameter (batch size, top-k, dimensions), look at the existing tests or ask — don't invent a value.
- GPU ops must be batched. Never call a model one-item-at-a-time.
- Gemini calls must be cached. Check the cache before hitting the API.
- Stack changes require explicit approval from the user before modification.

## Data notes
- Primary corpus: vidore/syntheticDocQA_government_reports_test on Hugging Face.
  1000 pages, 100 QA pairs with ground-truth page numbers. MIT.
- Secondary corpus: 2 real PDFs downloaded in step 2 for the ingestion demo.
- Never commit data/ to git. Add to .gitignore immediately.

## Common gotchas
- **Running on 4 GB VRAM — do not try to load ColQwen2-v1.0 under any
  circumstance, it will OOM. ColSmol-256M is the chosen model.** It uses
  `ColIdefics3` / `ColIdefics3Processor` classes from `colpali-engine`.
- ColSmol-256M returns multi-vector embeddings (one vector per patch). Qdrant
  needs MultiVectorConfig with MaxSim comparator. Vector size is 128.
- Vision indexing default batch_size is 2 on 4 GB VRAM. Drop to 1 if OOM.
- pdf2image needs Poppler installed at OS level.
- bfloat16 on CUDA, float32 on CPU. Check `torch.cuda.is_available()` before loading.
- Gemini 2.5 Flash has a ~1M token context but request payload size cap is ~20MB.
  Cap retrieved page images at 4 per call to stay safe.
- Gemini free tier: ~10 RPM for Flash. For eval runs, add a 7-second sleep between
  calls or the whole run fails at query 11.
