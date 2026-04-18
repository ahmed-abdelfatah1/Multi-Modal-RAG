# Claude Code prompt for DSAI 413 Assignment 1

Multi-Modal RAG QA system with Gemini 2.5 Flash as the generator, ColQwen2 + bge-small-en as open-source retrievers. Built following Anthropic's official Claude Code best practices: Explore → Plan → Code, specific context, verification criteria at every step, CLAUDE.md for persistent context.

---

## How to use this document

You will paste three things into Claude Code across the life of the project:

1. The bootstrap prompt (section A) — paste this as the very first message when you open Claude Code in an empty directory. This triggers Plan Mode exploration and produces a complete plan.
2. The CLAUDE.md seed (section B) — Claude Code will write this to disk during step 1. It is persistent context for every future session.
3. The step prompts (section C) — one per implementation step. Paste them in order, verifying each step before moving on.

Everything below is copy-paste ready. Square-bracketed placeholders like [your GPU] you fill in once.

Before you start: get a free Gemini API key at https://aistudio.google.com/apikey. The free tier is enough for all development and eval runs for this assignment.

---

## Section A — Bootstrap prompt (paste first, in Plan Mode)

Copy everything between the lines below into Claude Code. Activate Plan Mode first (Shift+Tab until you see "plan mode on").

---

I'm building a Multi-Modal RAG QA system for my DSAI 413 university assignment. The assignment rubric weights are: Accuracy and Faithfulness 25 percent, Multi-modal Coverage 20 percent, System Design and Architecture 20 percent, Innovation and Tooling Choice 15 percent, Code Quality 10 percent, Presentation 10 percent. My submission must process text, tables, images, charts, and footnotes, answer questions with citations, and come with an evaluation suite. Deliverables are a GitHub repo, a demo app, a 2-page technical report, and a 2-to-5 minute video.

Design choice — hybrid stack: open-source vision retrieval with Gemini 2.5 Flash for answer generation. Rationale I will put in the report: ColQwen2 handles the hard part (finding the right page from charts and tables without OCR), and Gemini 2.5 Flash handles the other hard part (reasoning over retrieved page images to produce faithful, cited answers). Using a frontier VLM for generation removes latency and VRAM bottlenecks from the demo while letting us show off the interesting retrieval innovation. This is a defensible architecture decision that I will write up as such.

Here is the exact stack I have decided on. Do not suggest alternatives unless a library is deprecated or broken.

Retrieval layer (open source, Hugging Face):
- Vision retriever: vidore/colqwen2-v1.0 (Apache 2.0, 89.3 nDCG@5 on ViDoRe)
- Text retriever: BAAI/bge-small-en-v1.5 (MIT, 384-dim dense embeddings)
- Vector database: Qdrant (self-hosted via Docker), with multi-vector support for ColQwen2 and a separate text collection
- Hybrid fusion: reciprocal rank fusion of both channels

Generation layer (Gemini API):
- Model: gemini-2.5-flash (stable, free tier is generous)
- SDK: the new google-genai Python package (NOT the legacy google-generativeai)
- Client pattern: genai.Client() reads GEMINI_API_KEY from env automatically
- Native multi-image input, structured JSON output via response_schema

Parsing layer:
- Docling from IBM Research (MIT, best open-source complex-table accuracy)
- pdf2image for rendering original PDFs to 200 DPI page images

Orchestration and UI:
- LangGraph for the QA pipeline graph (retrieve, generate, validate nodes)
- Streamlit for the chat UI with thumbnail-citation panel
- Ragas for retrieval and answer-quality metrics (using Gemini as the judge for consistency)

Data source — this is important, do not download raw PDFs from the web for the primary corpus:
- Primary corpus: the Hugging Face dataset vidore/syntheticDocQA_government_reports_test. This is 1000 pre-rendered pages from government reports, English, with 100 pre-authored question-answer-page triplets. One load_dataset call and you have everything: images, queries, answers, ground-truth pages. MIT license.
- Secondary corpus for the ingestion-pipeline demo: 2 real PDFs downloaded in step 2, so the ingestion code path is actually exercised against raw PDFs (not just pre-rendered images). This is critical for the Multi-modal Coverage rubric score.

Environment I am running on:
- OS: [fill in: macOS / Ubuntu / Windows WSL2]
- GPU: [fill in your GPU, e.g. "NVIDIA RTX 4090 with 24GB VRAM" or "none, CPU only"]
- Python: 3.11
- Package manager: uv (install from https://docs.astral.sh/uv/ if not present)
- Docker is available for running Qdrant

Note on Gemini SDK — the API has changed. Use these modern patterns and nothing else:

```python
from google import genai
from google.genai import types

client = genai.Client()  # picks up GEMINI_API_KEY from env

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=['What is in this image?', pil_image],  # pass PIL Images directly
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema=MyPydanticModel,  # structured output
        temperature=0.1,
    ),
)
parsed: MyPydanticModel = response.parsed
```

Do NOT use `google.generativeai` (legacy, deprecated). Do NOT use `genai.GenerationConfig(...)` (old pattern). Do NOT use `google-ai-generativelanguage`. The only correct package is `google-genai` on PyPI.

Your task right now, in Plan Mode:

1. Do not write any code yet. Use Plan Mode only.
2. Produce two outputs in this order:
   a. A proposed directory layout, with one-sentence purpose for each top-level folder.
   b. A step-by-step implementation plan with exactly 11 steps matching the structure I will give you in the step prompts below. Each step must have: title, definition of done, verification command(s) I can run to prove the step works.
3. Then STOP and wait for me to review the plan. Do not exit Plan Mode.

Key constraints you must honor throughout:
- Every commit must have a verification command in its message that I can run to prove the behavior.
- No step is "done" until its verification command passes.
- If a step's verification fails more than twice, stop and ask me for clarification instead of guessing.
- Prefer tiny, focused modules over god-files. Target 200 lines max per Python file.
- Every Python module has a __main__ smoke-test under `if __name__ == "__main__":` that runs in under 10 seconds.
- All secrets live in a .env file (never committed). Ship .env.example with every variable documented.
- GEMINI_API_KEY is the single secret the project needs. All other config is non-secret.
- Code style: ruff + mypy strict, pytest for tests.
- Python dependencies are managed with uv. The project uses pyproject.toml, not requirements.txt.
- Gemini API calls must be wrapped with retry + exponential backoff (use tenacity). Free-tier rate limits are real.
- Every Gemini call is cached to disk by SHA-256 of (prompt + image_bytes + config) so re-running eval does not re-bill quota.

---

End of bootstrap prompt. Wait for Claude Code to produce the plan, review it in your text editor with Ctrl+G, then say "approved, proceed to step 1" or correct specifics.

---

## Section B — CLAUDE.md seed

Tell Claude Code during step 1 to write the following into CLAUDE.md. This file gets loaded at the start of every future session and keeps persistent context small and relevant.

```markdown
# Project: dsai-413-multimodal-rag

## What this is
A multi-modal Retrieval-Augmented Generation QA system for policy and financial
documents. Vision-first retrieval via ColQwen2 over page images, with a text
channel via bge-small-en for lexical queries, fused with reciprocal rank fusion.
Answer generation by Gemini 2.5 Flash over retrieved page images + text chunks.

## Hard constraints
- Retrieval stack is fully open-source, Hugging Face only.
- Generator is Gemini 2.5 Flash via the google-genai SDK. API key in env only.
- Vector store: Qdrant self-hosted in Docker. Not Pinecone, not Weaviate.
- Python 3.11, uv for deps, ruff + mypy, pytest.

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

## Data notes
- Primary corpus: vidore/syntheticDocQA_government_reports_test on Hugging Face.
  1000 page images, 100 QA pairs with ground-truth page numbers. MIT.
- Secondary corpus: 2 real PDFs downloaded in step 2 for the ingestion demo.
- Never commit data/ to git. Add to .gitignore immediately.

## Common gotchas
- ColQwen2 returns multi-vector embeddings (one vector per patch). Qdrant needs
  MultiVectorConfig with MaxSim comparator.
- pdf2image needs Poppler installed at OS level.
- bfloat16 on GPU; float32 on CPU. Check torch.cuda.is_available() before loading.
- Gemini 2.5 Flash has a ~1M token context but request payload size cap is ~20MB.
  Cap retrieved page images at 4 per call to stay safe.
- Gemini free tier: ~10 RPM for Flash. For eval runs, add a 7-second sleep between
  calls or the whole run fails at query 11.
```

---

## Section C — Eleven step prompts

Paste one at a time after the previous step's verification passes.

---

### Step 1 — Scaffold and environment

```
Now execute step 1 of the plan: scaffold the repository.

Create:
- pyproject.toml with uv, declaring these groups:
  * core: pdf2image, pillow, pydantic, pydantic-settings, python-dotenv, tenacity
  * retrieval: qdrant-client, colpali-engine, sentence-transformers, transformers, torch, accelerate
  * parsing: docling
  * generation: google-genai
  * ui: streamlit
  * orchestration: langgraph
  * eval: ragas, datasets
  * dev: pytest, ruff, mypy, ipykernel
- docker-compose.yml with Qdrant service, volume mounted to data/indices/qdrant.
- .env.example documenting: GEMINI_API_KEY (required, get free at
  https://aistudio.google.com/apikey), QDRANT_URL, QDRANT_PORT, HF_HOME (for model
  cache), DEVICE (cuda or cpu or mps), DATA_DIR, INDEX_DIR, GEMINI_CACHE_DIR.
- .gitignore excluding .env, data/, __pycache__/, .venv/, *.egg-info/, .mypy_cache/,
  .pytest_cache/, .ruff_cache/.
- Makefile with targets: setup, download, ingest, index, eval, app, test, lint,
  typecheck, clean.
- CLAUDE.md exactly as specified in Section B of the prompt doc I gave you.
- README.md skeleton with sections: Overview, Architecture diagram placeholder,
  Quickstart (4 commands), How it works, Evaluation results placeholder.
- src/ with empty __init__.py files and these package folders: data, ingestion,
  indexing, retrieval, generation, graph, app, eval, utils, config.py (Pydantic
  Settings for all env variables).
- tests/ with a single test_smoke.py that imports every src package and asserts
  nothing. Also a test_gemini_smoke.py that ONLY runs if GEMINI_API_KEY is set —
  it makes a single "say ok" call to confirm the SDK works. Mark it with
  @pytest.mark.integration and skip if no key.

Verification:
- `uv sync` completes.
- `docker compose up -d` brings Qdrant up on port 6333.
- `curl http://localhost:6333/healthz` returns ok.
- `uv run pytest -q -m "not integration"` passes.
- With GEMINI_API_KEY set: `uv run pytest -q -m integration` passes (proves the
  Gemini SDK is wired correctly).
- `uv run ruff check src` passes.
- `uv run mypy src` passes.

Do not proceed past verification failures. Report back when all checks pass.
```

---

### Step 2 — Data download (the easy part — pre-built dataset)

```
Execute step 2: data download.

The primary corpus comes from Hugging Face in one call, already pre-rendered and
pre-annotated. Do NOT crawl the web or download any IMF or government PDFs by URL
for the primary corpus.

Write src/data/download.py that does exactly three things:

1. Download the primary corpus:
   `ds = load_dataset("vidore/syntheticDocQA_government_reports_test", split="test")`
   Save each unique page image to data/corpus_primary/pages/{image_filename}.png
   (use the existing image_filename field, flattened to a single folder).
   Save the QA pairs to data/corpus_primary/qa.jsonl with fields:
   query, answer, page, image_filename, source.
   Also emit data/corpus_primary/manifest.json with: n_pages, n_queries,
   unique_source_pdfs (list).

2. Download the secondary mini-corpus (2 real PDFs for the ingestion-pipeline demo):
   Download these two publicly-hosted PDFs using requests with timeout and retry:
   - https://www.imf.org/-/media/Files/Publications/CR/2024/English/1USAEA2024001.ashx
     save as data/corpus_secondary/raw/us_2024_article_iv.pdf
   - https://www.imf.org/-/media/Files/Publications/CR/2025/English/1GBREA2025001.ashx
     save as data/corpus_secondary/raw/uk_2025_article_iv.pdf
   If a URL returns a non-200 or fails after 3 retries, skip it, log a warning,
   and continue — do not crash. The pipeline works with either 0, 1, or 2 secondary
   PDFs.

3. Print a summary at the end: "Primary: X pages, Y queries. Secondary: Z PDFs."

Requirements:
- Use the `datasets` library for #1 and `requests` for #2.
- Respect HF_HOME from the environment for caching.
- Make the script idempotent: if files already exist, skip the download.
- Add `if __name__ == "__main__": main()` so it runs as a module.

Verification:
- `uv run python -m src.data.download` runs to completion.
- data/corpus_primary/pages/ contains at least 900 PNG files.
- data/corpus_primary/qa.jsonl has at least 90 lines, each valid JSON with the
  expected fields.
- data/corpus_secondary/raw/ contains 0, 1, or 2 PDFs and the script didn't crash.
- Re-running the script a second time is fast (no re-download).

Add a small test in tests/test_data.py that asserts: after download, loading
qa.jsonl yields at least 90 records, each with non-empty query and page fields.
Run `uv run pytest -q`.
```

---

### Step 3 — Ingestion pipeline (proves multi-modal parsing)

```
Execute step 3: ingestion pipeline for the SECONDARY corpus only. The primary
corpus arrives pre-rendered, so it skips this step — but we must demonstrate real
PDF parsing on the secondary PDFs for the Multi-modal Coverage rubric score.

Build src/ingestion/ with these modules (each under 200 lines):

- parse.py: wrap Docling. Input: a PDF path. Output: a DoclingDocument plus a
  structured JSON export saved to data/corpus_secondary/parsed/{doc_id}/document.json.
  Configure Docling with OCR enabled and table structure recognition enabled.
- render.py: use pdf2image at 200 DPI to write page PNGs to
  data/corpus_secondary/parsed/{doc_id}/pages/page_{NNN}.png.
- chunk.py: use Docling's HybridChunker to produce text chunks. Each chunk is a
  dict with: chunk_id, doc_id, page_number, section_path (list of headings),
  chunk_type (one of: text, table, caption, footnote), content (str),
  bbox (optional). Write one JSONL line per chunk to
  data/corpus_secondary/parsed/{doc_id}/chunks.jsonl.
- run_all.py: for each PDF in data/corpus_secondary/raw/, run parse + render +
  chunk. Idempotent — skip docs already processed. Log progress to stderr.

Requirements:
- Handle "no PDFs present" gracefully: log "No secondary PDFs to ingest, skipping"
  and exit 0. The pipeline must not fail when step 2 downloaded zero PDFs.
- Every chunk's chunk_type must be one of the four allowed values. Add a Pydantic
  model Chunk in src/ingestion/models.py to enforce this.
- Derive doc_id from the PDF filename stem.
- The manifest.json for each doc records: n_pages, n_chunks, chunks_by_type.

Verification:
- If 0 secondary PDFs were downloaded: `uv run python -m src.ingestion.run_all`
  prints the skip message and exits 0.
- If at least 1 PDF was downloaded: after running, data/corpus_secondary/parsed/
  contains a subfolder per doc, each with pages/ (PNG files), chunks.jsonl
  (at least 50 chunks for a real Article IV report), and document.json.
- Spot-check chunks.jsonl by eye: confirm at least one chunk_type == "table"
  exists for each real PDF.

Add tests/test_ingestion.py with a tiny fixture PDF (you can generate one with
reportlab — 2 pages, one with a table) and verify parse + chunk both run and
produce the expected chunk types. Run `uv run pytest tests/test_ingestion.py -q`.
```

---

### Step 4 — Text indexing

```
Execute step 4: text indexing into Qdrant.

Build src/indexing/text_index.py:

- Connect to Qdrant at QDRANT_URL using qdrant-client.
- Create collection "text" (recreate if --recreate flag passed) with:
  vector size 384, distance Cosine, on-disk storage enabled.
- Add payload indexes on: doc_id (keyword), source (keyword), chunk_type (keyword),
  page_number (integer).
- Load BAAI/bge-small-en-v1.5 via sentence-transformers once, encode all chunks
  from BOTH corpora in batches of 64.
  - For the primary corpus: each QA row in qa.jsonl already points to a page
    image, but for the text channel we need text. Generate short pseudo-text
    per page by embedding the question-answer pair itself as a placeholder "text"
    (this is a known ViDoRe dataset limitation — synthetic docs don't ship with
    OCR text). Mark chunk_type as "synthetic_qa_anchor" and include both query
    and answer in content. Flag this in a comment.
  - For the secondary corpus: use the actual text chunks from step 3.
- Upsert points with id = sha256(content + doc_id + str(page_number))[:16] and
  full payload.

Verification:
- `uv run python -m src.indexing.text_index --recreate` completes without error.
- `curl http://localhost:6333/collections/text` returns status green with
  points_count > 100.
- Write tests/test_text_index.py that runs a sample query "government budget"
  through the index and asserts top-5 returns at least one result with score > 0.3.

Commit with message "feat(index): build text index with bge-small-en-v1.5 —
Verify: `uv run python -m src.indexing.text_index` and tests/test_text_index.py".
```

---

### Step 5 — Vision indexing (the key innovation component)

```
Execute step 5: vision indexing with ColQwen2 multi-vector embeddings. This is
the highest-leverage step for the Innovation rubric score.

Build src/indexing/vision_index.py:

- Load vidore/colqwen2-v1.0 once at module level with:
  torch_dtype=torch.bfloat16 if cuda else torch.float32,
  device_map=DEVICE (from env), attn_implementation="flash_attention_2" if
  available else None, eval mode.
- Load the matching ColQwen2Processor.
- Create Qdrant collection "pages" with:
    VectorParams(size=128, distance=Distance.COSINE,
                 multivector_config=MultiVectorConfig(
                     comparator=MultiVectorComparator.MAX_SIM))
  and quantization_config=BinaryQuantization(...) to cut memory 32x with small
  recall loss.
- Iterate over every page image from both corpora in batches of 2 (safe on
  12GB VRAM; bump to 4 on 24GB+). For each batch:
    batch = processor.process_images(images).to(model.device)
    with torch.no_grad():
        embeds = model(**batch)  # (B, n_patches, 128)
  Then apply HierarchicalTokenPooler with pool_factor=3 to reduce patch count
  67% with <3% quality loss.
- Upsert each page as ONE point with multi-vector embedding. Payload includes:
  doc_id, page_number, image_filename (for rendering in UI), source_corpus
  (primary | secondary).

Requirements:
- Make batch_size a CLI flag that defaults to 2. Document that 24GB VRAM can use 4,
  and CPU should use 1.
- Progress bar with tqdm.
- Idempotency: on re-run, skip pages already in Qdrant by checking the point id.
  Use image_filename hashed to 16 chars as the id.
- Time the full run; log pages/second.

Verification:
- `uv run python -m src.indexing.vision_index` runs to completion.
- `curl http://localhost:6333/collections/pages` returns points_count equal to
  total page images (at least 1000 from primary, plus any from secondary).
- A manual sanity check: write a throwaway script that queries the collection
  with a query embedding from "budget chart" and prints the top-3 image_filenames.
  Eyeball them to confirm they look relevant.

If the run OOMs, reduce batch_size to 1 and retry. If CPU-only, note that this
step will take multiple hours on 1000 pages and consider subsampling to 200 pages
for development. Document the CPU-subset path in the README.
```

---

### Step 6 — Retrieval (text, vision, hybrid)

```
Execute step 6: the three retrievers.

Build src/retrieval/:

- types.py: Pydantic models Retrieved (fields: id, source_type ["text"|"page"],
  doc_id, page_number, score, payload: dict). Also a RetrievalResult wrapping
  a list of Retrieved plus the original query.
- text_retriever.py: TextRetriever class with .retrieve(query, top_k=8) using
  bge-small-en + Qdrant "text" collection.
- vision_retriever.py: VisionRetriever class with .retrieve(query, top_k=8) using
  ColQwen2.process_queries + Qdrant "pages" collection multi-vector search via
  query_points with prefetch.
- hybrid.py: HybridRetriever that runs both in parallel (asyncio or
  concurrent.futures), then applies reciprocal rank fusion:
    rrf_score(item) = sum over retrievers of 1 / (k + rank_in_retriever)
    with k=60 (standard).
  Deduplicate: when a text chunk and a page image map to the same
  (doc_id, page_number), keep the page image (it carries more info).
  Return top-8 fused.

Requirements:
- Each retriever class is instantiated once (lazy model loading) and reused.
- Add a `retriever_mode` enum: text_only | vision_only | hybrid. Expose it via
  the HybridRetriever constructor so the UI can flip modes for the demo.
- Handle empty results gracefully (return empty RetrievalResult).

Verification:
- tests/test_retrieval.py: 3 tests (one per retriever mode), each with a hand-
  picked query from qa.jsonl, asserting the ground-truth page_number appears in
  the top-5 results. This catches regressions.
- Run `uv run pytest tests/test_retrieval.py -q` and confirm all three pass.

Commit with message "feat(retrieval): text/vision/hybrid retrievers with RRF —
Verify: pytest tests/test_retrieval.py".
```

---

### Step 7 — Generation with Gemini 2.5 Flash

```
Execute step 7: answer generation using Gemini 2.5 Flash.

First, read this reference API usage carefully. The google-genai SDK has replaced
the old google-generativeai package. Use ONLY the patterns below.

Correct imports:
    from google import genai
    from google.genai import types

Client creation — picks up GEMINI_API_KEY from env automatically:
    client = genai.Client()

Basic multimodal call with PIL images:
    from PIL import Image
    img = Image.open("page.png")
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=['Question here', img],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type='application/json',
            response_schema=Answer,  # Pydantic model
        ),
    )
    parsed: Answer = response.parsed

Alternative with raw bytes (if you don't want PIL):
    with open("page.png", "rb") as f:
        img_bytes = f.read()
    contents = [
        'Question here',
        types.Part.from_bytes(data=img_bytes, mime_type='image/png'),
    ]

Now build src/generation/:

- schema.py: Pydantic model Answer with fields: answer (str),
  citations (list of Citation with doc_id, page_number, quote), confidence
  (Literal["high","medium","low"]), missing_info (str | None).
- prompts.py: SYSTEM_PROMPT constant instructing the model to answer ONLY from the
  provided sources, cite every factual claim with [doc_id p.N], and say
  "I cannot answer from the provided sources" with confidence="low" when evidence
  is insufficient.
- cache.py: a disk-backed cache. Key = sha256(prompt + concatenated image bytes
  + model + temperature). Value = Answer JSON. Store under GEMINI_CACHE_DIR as
  {key}.json. Wrap with a simple Cache class: get(key), set(key, val).
- generator.py: Generator class.
  * Constructor: creates genai.Client() once.
  * .generate(question, retrieved: RetrievalResult) -> Answer.
    Build contents list:
      1. SYSTEM_PROMPT as leading text.
      2. "Question: {question}\n\nSources:"
      3. For each retrieved text chunk: "[Source: {doc_id} p.{page}]\n{content}"
      4. For each retrieved page image (cap at 4 to stay under payload limits):
         "[Page image: {doc_id} p.{page}]" followed by the PIL image.
    Call client.models.generate_content with response_schema=Answer and
    response_mime_type='application/json'.
    Use tenacity: @retry(stop=stop_after_attempt(3),
                         wait=wait_exponential(min=2, max=30),
                         retry=retry_if_exception_type(APIError)).
    Check cache first; on miss, call API, then cache.

Requirements:
- Cap retrieved page images at 4. Log a warning if more were passed in.
- Every call is cached. Print "cache hit" or "cache miss" to stderr for visibility
  during eval runs.
- If response.parsed is None (schema violation), raise ValueError with the raw
  text for debugging — do not try to parse manually.
- Respect free-tier rate limit (~10 RPM for Flash): add an optional
  min_seconds_between_calls parameter defaulting to 0. Eval harness will set it to 7.

Add tests/test_generation.py:
- One unit test with a mocked client that returns a fixed JSON — asserts parsing
  and cache behavior.
- One integration test (mark @pytest.mark.integration, skip if no API key) that
  takes the first QA from qa.jsonl, retrieves with hybrid, generates, and asserts
  the answer is non-empty and contains at least one citation.

Verification:
- `uv run pytest tests/test_generation.py -q -m "not integration"` passes (unit).
- With API key set: `uv run pytest tests/test_generation.py -q -m integration`
  passes (real API call, ~10 seconds).
- Manual smoke: `uv run python -m src.generation.generator
  --query "What is the main topic of the report?"` prints a valid Answer JSON
  with at least one citation.
- Cache check: run the same manual smoke a second time — it should be instant
  (cache hit).
```

---

### Step 8 — LangGraph pipeline

```
Execute step 8: wire everything with LangGraph.

Build src/graph/qa_graph.py with exactly these nodes:

1. retrieve: calls HybridRetriever, writes to state["retrieved"].
2. generate: calls Generator, writes to state["answer"].
3. validate: checks that every citation in state["answer"].citations refers to
   a (doc_id, page_number) pair that appears in state["retrieved"]. If not,
   set state["needs_regen"] = True.
4. Conditional edge from validate: if needs_regen is True AND regen_count < 1,
   go back to generate with a stricter prompt suffix. Else END.

State schema is a TypedDict with: query, retrieved, answer, regen_count, needs_regen.

Expose a single callable `ask(query: str, mode: str = "hybrid") -> Answer` that
builds and invokes the graph.

Verification:
- `uv run python -c "from src.graph.qa_graph import ask; print(ask('What is the
  main topic?').model_dump_json(indent=2))"` prints a valid answer JSON with
  citations that all exist in the retrieved set (add a print of retrieved
  doc/page pairs if helpful).
- tests/test_graph.py exercises the validate-then-regenerate path by monkey-
  patching the generator to first return a bad citation, then a good one, and
  asserts regen_count ends at 1.
```

---

### Step 9 — Streamlit demo app (the demo video's star)

```
Execute step 9: the Streamlit chat UI. This is what the grader sees in the video.

Build src/app/main.py with this exact layout:

- Page config: wide layout, title "Multi-Modal RAG QA — DSAI 413".
- Sidebar:
  * Corpus filter: checkbox per source_corpus (primary, secondary), default both on.
  * Retriever mode radio: hybrid (default) | vision-only | text-only.
  * Top-k slider: 3 to 10, default 6.
  * "Show similarity heatmap" checkbox (stretch goal, leave toggle even if not
    implemented this step).
  * Status badge: shows ColQwen2 device, Qdrant status, Gemini API connectivity
    (green if last call succeeded, red if rate limited).

- Main area:
  * Chat history (st.chat_message).
  * User input at the bottom (st.chat_input).
  * For each assistant turn, render:
    - The answer as markdown.
    - An expander labeled "Sources" containing ONE column per cited page,
      showing: doc_id + page number as caption, the page image thumbnail
      (clickable to open fullsize in a modal via st.dialog), and the cited
      quote below.
    - If retriever_mode was hybrid, also show "Also considered" expander with
      the non-cited top-k results and their scores.

- On first load, call a cached model-loader function so the ColQwen2 weights only
  load once per Streamlit session. Gemini client is cheap — create fresh per request
  via the Generator class's internal client.

Requirements:
- The app MUST work end-to-end on the primary corpus alone (no secondary PDFs
  needed).
- The app MUST NOT crash if Qdrant is down or Gemini is rate-limited; show a
  friendly error banner with the specific error.
- Use st.cache_resource for ColQwen2 loading, st.cache_data for Qdrant client.
- Never log or display the API key.
- Show a "cache hit" indicator next to cached answers so the grader sees that
  the system has a cost-aware cache layer.

Verification:
- `uv run streamlit run src/app/main.py` launches, opens in browser.
- Three demo queries all return answers with visible page thumbnails:
  1. "What is the main conclusion of the government report?"
  2. (Take a query directly from qa.jsonl that asks about a chart or table)
  3. (Take a cross-page query from qa.jsonl)
- Toggle retriever_mode between hybrid and text-only for query 2 and confirm
  that vision retrieval beats text for chart-heavy queries.

Screenshot the working UI and paste it into README.md under Architecture.
```

---

### Step 10 — Evaluation suite

```
Execute step 10: evaluation against the ground-truth Q&A set.

Build src/eval/:

- run.py: for each query in data/corpus_primary/qa.jsonl, run retrieval in all
  3 modes (text_only, vision_only, hybrid), then run generation on the hybrid
  result. Save per-query outputs to data/eval/runs/{mode}.jsonl with fields:
  query, expected_page, expected_answer, retrieved_pages (top-5), generated_answer,
  latency_ms, cache_hit.

- metrics.py: compute, per mode:
  * Hit@1, Hit@3, Hit@5: is expected_page in the top-k retrieved_pages?
  * MRR of expected_page.
  * Per-query-type breakdown if qa.jsonl has tags; otherwise overall only.
  * For hybrid mode, also compute Ragas faithfulness and answer_relevancy.
    Configure Ragas to use Gemini 2.5 Flash as the judge LLM and Gemini's
    text-embedding-004 as the embedder (keeps eval fully on one provider for
    consistency, and the judge results are comparable to a production setup).

- report.py: render data/eval/report.md with:
  * A comparison table of Hit@1/3/5 and MRR across the three modes.
  * A second table of per-chunk-type breakdown (where possible).
  * Ragas faithfulness and relevance for hybrid mode.
  * Operational metrics: mean and P95 end-to-end latency per mode, cache hit rate.
  * A matplotlib bar chart saved as data/eval/charts/hit_at_k.png.

Requirements:
- Make the eval subsample size a flag (--n 20 for quick iteration, default full 100).
- Cache embeddings to disk so re-runs only re-embed new queries.
- Respect Gemini free-tier rate limits: pass min_seconds_between_calls=7 to the
  generator during eval. The cache means re-runs are fast after the first pass.
- Print a summary table to stdout at the end.

Verification:
- `uv run python -m src.eval.run --n 20` completes in under 15 minutes on first
  run (Gemini rate-limited). Second run is under 1 minute (all cache hits).
- `uv run python -m src.eval.run --n 20` generates data/eval/report.md with
  non-placeholder numbers.
- The hybrid mode Hit@5 is >= max(text_only Hit@5, vision_only Hit@5) — if it's
  worse, the RRF is broken and we need to debug.
- Run the FULL eval once (`uv run python -m src.eval.run`) and commit the
  report.md + chart.png. With cache warm, the full 100-query run takes under
  20 minutes.
```

---

### Step 11 — Technical report and video prep

```
Execute step 11: the non-code deliverables.

Part A — technical report (docs/report.md, strict 2-page limit).

Structure:
1. Title line, author, course, 1-sentence problem statement.
2. Architecture diagram (Mermaid) showing: HF dataset + raw PDFs → Docling +
   pdf2image → ColQwen2 + bge-small → Qdrant (text + pages) → Hybrid RRF →
   Gemini 2.5 Flash → UI.
3. Design decisions table (5 rows):
   * ColQwen2 for vision retrieval (innovation, state-of-the-art on ViDoRe).
   * Qdrant multi-vector with binary quantization (scalability).
   * RRF fusion of text + vision (recall + precision).
   * Gemini 2.5 Flash for generation (best multi-image VLM, removes local
     inference bottleneck — justify as a deliberate pragmatic choice).
   * Response caching (reproducibility + cost).
   One sentence rationale each.
4. Evaluation table: Hit@1/3/5 + MRR across three modes, pulled from eval/report.md.
5. Observations paragraph: where vision beats text, where text beats vision,
   failure modes.
6. Limitations: English only, 1000-page scale tested, dependence on external API.
7. Future work: local VLM (Qwen2.5-VL) as a drop-in generator for offline use,
   cross-doc reasoning, fine-tuning ColQwen2 on IMF corpora.
8. References: ColPali paper arxiv:2407.01449, ViDoRe V2 arxiv:2505.17166,
   Docling arxiv:2501.17887, Gemini 2.5 technical report.

Export to docs/report.pdf via pandoc or (if pandoc unavailable) via LibreOffice
headless. Must fit on 2 pages at 11pt Arial, 1-inch margins.

Part B — video demo script (docs/demo_script.md):

- 0:00–0:20 — Problem statement. One sentence, show a page with a complex
  table. "Text-only RAG loses this."
- 0:20–0:50 — Architecture flyover. Show the report's Mermaid diagram.
- 0:50–2:30 — Live demo. Three queries:
  * Query 1 (text-dominant): "Summarize the main findings of the report."
    Hybrid mode. Show retrieved pages and citations.
  * Query 2 (chart-dominant): pick a chart question from qa.jsonl. Toggle
    retriever_mode between text_only and vision_only to show that vision
    retrieves the correct page while text misses it.
  * Query 3 (multi-page): a question whose answer spans 2 pages. Show both
    citation thumbnails.
- 2:30–3:30 — Show eval/report.md: the Hit@k comparison chart. Call out
  the delta between vision-only and hybrid modes.
- 3:30–4:00 — Limitations and next steps.

Part C — final README.md polish:

Update README.md with:
- One-paragraph overview.
- The architecture Mermaid diagram.
- Quickstart (4 commands: `make setup && make download && make index && make app`).
- Prerequisites section: mentions needing a free Gemini API key with a link to
  https://aistudio.google.com/apikey.
- Link to docs/report.pdf and data/eval/report.md.
- Model and service list with links and licenses:
  * vidore/colqwen2-v1.0 — Apache 2.0
  * BAAI/bge-small-en-v1.5 — MIT
  * Gemini 2.5 Flash — Google API (free tier)
- Acknowledgements to ViDoRe benchmark and Illuin Tech.

Verification:
- docs/report.pdf exists and is exactly 2 pages.
- docs/demo_script.md exists.
- README.md quickstart is literally runnable on a fresh clone with only a
  GEMINI_API_KEY as external dependency (test with
  `rm -rf .venv data/indices && make setup && make download && make app`
  on a friend's machine or a Docker container).

This completes the project. Create a final commit "chore: docs, report, demo
script — Verify: open docs/report.pdf and confirm 2 pages".
```

---

## Section D — Recovery and debugging prompts

Keep these in your back pocket for when things go wrong.

### When a step's verification fails

```
The verification for step [N] failed with this output:

[paste the full error or failing test output]

Do NOT try a different fix immediately. First, investigate by:
1. Reading the relevant module end-to-end.
2. Running the component in isolation (e.g. `uv run python -m src.[module]`).
3. Looking at actual data (e.g. `head data/corpus_primary/qa.jsonl`).
4. Proposing exactly one hypothesis for what's wrong and one surgical fix.

Then apply the fix and re-run verification.
```

### When Gemini rate-limits you

```
The Gemini API is returning 429 rate-limit errors during [context]. Do NOT try
to "fix" this by removing retries or the cache. Instead:

1. Confirm tenacity retry decorator is on the call.
2. Confirm the cache is actually being checked (grep for "cache hit").
3. If running eval, confirm min_seconds_between_calls is set to 7.
4. Consider running eval with --n 20 first to warm the cache, then the full run.
```

### When context is getting cluttered

```
/clear

Resuming step [N]. The state of the repo is: [one-sentence summary].
Current issue: [one-sentence description].
Continue from where the previous session left off. Start by running
`git status` and `git log -5 --oneline` to orient yourself.
```

### For hard debugging

```
Use a subagent to investigate [specific question]. Report back with:
- The root cause in one sentence.
- The relevant file and line numbers.
- The minimal fix.
Do not make changes — just investigate.
```

---

## Section E — Sanity check before submission

Before you submit, run this final verification sequence manually:

```
# Fresh-clone test
rm -rf /tmp/rag-test && git clone [your-repo-url] /tmp/rag-test
cd /tmp/rag-test
cp .env.example .env  # fill in GEMINI_API_KEY
make setup
make download
docker compose up -d
make index
make eval
make app
# Then open browser, run 3 test queries, record video
```

If that completes on a friend's laptop with just their own GEMINI_API_KEY, you are ready to submit.

---

End of prompt document.
