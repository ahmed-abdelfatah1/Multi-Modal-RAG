# Report Data — handoff for technical report

Every number, table, chart path, and fact needed to write the 2-page technical report.
Do not edit; regenerate by re-running the commands referenced under each section.

## Section 1 — Corpus statistics

### Primary corpus

| Field | Value |
|---|---|
| Source | `vidore/syntheticDocQA_government_reports_test` (Hugging Face) |
| Total pages | 972 |
| Total queries (QA pairs) | 100 |
| Unique source documents | 88 PDFs (US government reports, synthetic QA) |
| License | MIT |
| On-disk size | 1018 MB (rendered PNGs at 200 DPI) |

### Secondary corpus

| Doc ID | Pages | Total chunks | text | table | footnote |
|---|---|---|---|---|---|
| `bis_annual_report_2024` | 150 | 93 | 86 | 7 | 0 |
| `bis_quarterly_review_2024q3` | 94 | 89 | 68 | 12 | 9 |
| **Total** | **244** | **182** | **154** | **19** | **9** |

- Parsed via Docling with `do_ocr=False` (born-digital PDFs) and `do_table_structure=True`.
- Page images rendered via `pdf2image` at 200 DPI. Total on-disk: 106 MB.
- Both documents downloaded from bis.org (permissive public access).

## Section 2 — Primary corpus retrieval results

**Top-k = 5. N = 100 queries.**

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| vision_only | 0.860 | 0.940 | 0.990 | 0.910 |
| hybrid (RRF k=60) | 0.990 | 1.000 | 1.000 | 0.995 |

### Methodology note — why `text_only` is excluded from this table

The primary corpus is image-only (`vidore/syntheticDocQA_government_reports_test` ships PNG
page images without OCR text). To populate the text channel for this corpus,
`src/indexing/text_index.py:load_primary_corpus_texts()` synthesises one text chunk per
page that has an associated QA pair, built from `"Q: <query>\nA: <answer>"`.

At eval time, the text retriever is queried with `<query>` — which is the same string that
was concatenated into the synthetic chunk indexed for the ground-truth page. Cosine
similarity approaches 1.0, so **`text_only` Hit@k collapses to a near-tautology** and
reports 100% at every k.

`text_only` numbers therefore describe index integrity, not retrieval capability, and are
excluded from the primary-corpus comparison. The meaningful comparison is
`vision_only` (ColSmol-256M multi-vector retrieval over page images) versus `hybrid` (RRF
of vision + the biased text channel).

For the raw, un-redacted numbers see `data/eval/report.md`.

### Appendix 2.A — sample queries (seed=42, 5 random)

Shown for transparency of the text_only tautology and to illustrate vision/hybrid behaviour.

| Query | Expected | text top-3 | vision top-3 | hybrid top-3 |
|---|---|---|---|---|
| Why is a collaborative approach preferred over traditional approaches for c… | 18 | [18, 17, 6] | [18, 21, 33] | [18, 17, 21] |
| What actions did the government of Honduras take in response to the COVID-1… | 31 | [31, 5, 40] | [31, 21, 15] | [31, 5, 21] |
| How have communities adapted their recycling programs? | 18 | [18, 41, 108] | [136, 37, 101] | [18, 136, 41] |
| What are some considerations for developing an employee Internet usage poli… | 3 | [3, 5, 49] | [3, 166, 17] | [3, 5, 166] |
| What is the role of the Steering Committee in relation to the Program Manag… | 29 | [29, 7, 8] | [29, 88, 28] | [29, 7, 88] |

Generator: `python -c "..." ` seeded sample over `data/eval/runs/text_only.jsonl`.

## Section 3 — Secondary corpus retrieval results

**NOT MEASURED.** The BIS secondary corpus was ingested and indexed (244 pages, 182
chunks, 150 vision vectors for the annual report + 94 for the quarterly review), but no
QA ground-truth set was constructed for it. `data/eval/secondary_queries.jsonl` does not
exist.

### What would be needed

- 15–30 hand-authored questions across query types: `text_lookup`, `table`, `chart`,
  `cross_page`, `footnote`, each annotated with the expected `(doc_id, page_number)`.
- A variant of `src/eval/run.py` (or a `--qa-path` flag) that points at the secondary QA
  file and a corpus filter so retrievers only return secondary-corpus hits.
- ~2 hours of authoring + ~10 min eval runtime (retrieval is fast; skip generation to
  avoid Gemini quota).

### What this means for the report

The three-way (text/vision/hybrid) comparison cannot be made honestly on this corpus. The
paper can still describe the secondary corpus as **qualitative evidence of multi-modal
coverage** (tables and footnotes parsed, born-digital text channel populated) without
claiming numeric superiority of any mode.

## Section 4 — Generation quality

### Ragas scores

**NOT MEASURED.** `ragas` is not installed (see `pyproject.toml` dependency list) and
`data/eval/report.md` explicitly defers Ragas-style faithfulness and answer-relevancy
metrics to "future work" to stay within the lightweight stack.

Adding them would require: `uv add ragas datasets` + a judge LLM (Gemini or OpenAI) +
~100 additional LLM judge calls per metric per mode. Free-tier Gemini quota (see
Section 5) makes this expensive to run today.

### RECITATION failure rate

The 100-query eval ran **before** the schema fix (citations required a verbatim `quote`
field). Gemini's RECITATION safety filter blocked `finish_reason=RECITATION` on queries
whose answer closely matched training data — these are US government reports, which
Gemini has almost certainly seen.

| Window | Prompt schema | Success | RECITATION failures | Rate |
|---|---|---|---|---|
| 100-query eval (pre-fix) | `Citation.quote` required | 89/100 | 11/100 | **11.0%** |
| Post-fix sanity (2 queries via app) | `Citation.snippet`, paraphrase prompt | 2/2 | 0/2 | 0% (n too small) |

**A re-run of all 100 queries under the fixed schema has not been executed** (would cost
100 Gemini calls). The 0% post-fix rate is anecdotal.

### Citation and confidence statistics

Computed over the **94 cached Gemini responses** that parsed successfully (mix of pre-fix
eval + post-fix app interactions). The eval `hybrid.jsonl` only stores the `.answer`
string, so citations and confidence must be read from `data/gemini_cache/*.json`.

| Metric | Value |
|---|---|
| Cached successful answers | 94 |
| — pre-fix (`quote` field) | 89 |
| — post-fix (`snippet` field) | 5 |
| Mean citations per answer | 1.54 |
| Median citations per answer | 1 |
| Min / max citations | 1 / 6 |
| Confidence = `high` | 94 / 94 (100%) |
| Confidence = `medium` | 0 |
| Confidence = `low` | 0 |

**Observation.** The model self-reports `confidence: high` on every successful response.
This is diagnostic of the schema rather than of calibration — the prompt does not force
the model to consider uncertainty, and `high` is the natural default when cited sources
exist. For the report this should be flagged as uncalibrated self-assessment.

## Section 5 — Operational metrics

### End-to-end retrieval latency (N=100)

| Mode | Mean (ms) | P50 (ms) | P95 (ms) |
|---|---|---|---|
| text_only | 109.5 | 45.6 | 60.9 |
| vision_only | 257.7 | 192.1 | 282.1 |
| hybrid | 199.8 | 198.3 | 252.2 |

Hybrid < vision because the two channels run in parallel (`ThreadPoolExecutor`, see
`src/retrieval/hybrid.py:121-123`); wall time is dominated by the slower branch.
Generation latency is **not included** in these numbers — eval measures retrieval only
per `_retrieve_safely` in `src/eval/run.py:48-61`.

### Gemini cache hit rate during eval

- Hybrid cache hit rate: **1 / 100 = 0.010**. The eval ran fresh, so a single accidental
  duplicate accounts for the one hit. This is an eval-time metric, not a production
  steady-state.

### Gemini API calls (eval)

| Quantity | Value |
|---|---|
| Queries with cache-miss generation attempt | 100 |
| Calls that parsed successfully (response.parsed ≠ None) | 89 |
| Calls that hit RECITATION (blocked) | 11 |
| Total actual API calls (including retries) | up to 100 + 33 (3 retries × 11 failures) = **≤ 133** |

Retries occur inside tenacity at `src/generation/generator.py:145-150` (`stop_after_attempt=3`). The upper bound above assumes every
retry consumed a fresh call; some were likely short-circuited by the 429 back-off wait.

### Indexing throughput

| Phase | Items | Wall time | Rate |
|---|---|---|---|
| Vision (ColSmol-256M, batch=1, CUDA bf16, RTX 3050 Ti 4 GB) | 1216 pages | 1684 s (~28 min) | **0.72 pages/s** |
| Text (bge-small-en-v1.5, batch=64, CUDA) | 254 chunks | ~5 s | **~50 chunks/s** |

Vision source: second rebuild log (`Indexed 1216 pages in 1684.0s`). Text source: from
the same log — embedding + 3 upsert batches completed in the order of seconds.

### Index size on disk

| Collection | Points | Vector dim | Disk bytes | Human |
|---|---|---|---|---|
| `pages` (multi-vector, binary-quantised) | 1216 | 128 per patch, ~1024 patches per page | 1,822,566,603 | 1.70 GiB |
| `text` (single-vector) | 254 | 384 | 865,629,303 | 825 MiB |
| **Total Qdrant on-disk** | | | **2,688,195,906** | **2.50 GiB** |

The text collection size is disproportionate to its point count because Qdrant allocates
fixed segment overhead regardless of row count; 8 segments × ~100 MB baseline each
dominates the 254 rows themselves.

## Section 6 — Chart inventory

| Path (repo-relative) | Description |
|---|---|
| `data/eval/charts/hit_at_k.png` | Grouped bar chart: Hit@1 / Hit@3 / Hit@5 per retrieval mode (text_only, vision_only, hybrid). Autogenerated by `src/eval/report.py`. |

No other charts were produced. Latency, cache, and confidence charts do not exist; the
report will either cite the tables directly or render fresh charts from the
`data/eval/runs/*.jsonl` raw data.

## Section 7 — Design decisions actually made

Rationale is kept to one sentence each; file:line pointers are in
`docs/report_references.md`.

- **ColSmol-256M over ColQwen2-v1.0.** The laptop has 4 GB of VRAM; ColQwen2 OOMs at
  load, ColSmol fits at bfloat16 with batch=1 and still scores 80.1 on ViDoRe.
- **Gemini 2.5 Flash over a self-hosted VLM.** No local VLM fits alongside ColSmol on 4
  GB; Gemini's 1 M-token context and free tier let the project stay open-source for
  retrieval while outsourcing generation.
- **Qdrant with multi-vector `MaxSim` + binary quantisation.** ColSmol emits ~1 000 patch
  vectors per page, so a single-vector store would destroy the late-interaction signal;
  binary quantisation keeps the 1 216 pages under 2 GB.
- **RRF (k=60) over learned fusion.** RRF is training-free, requires no calibration of
  vision-vs-text scores (which live on different scales), and k=60 is the Cormack/Clarke
  default.
- **Paraphrased `snippet` over verbatim `quote`.** Gemini's RECITATION filter blocks
  verbatim reproductions of government reports, so forcing the schema to require a quote
  produced an 11 % failure rate; paraphrasing bypasses the filter.
- **Docling with OCR disabled for born-digital PDFs.** BIS reports have a real text
  layer, so OCR adds only memory pressure (we hit `std::bad_alloc` with OCR on during
  ingestion) and no information.
- **Separate primary / secondary corpora.** ViDoRe gives image-only ground truth; BIS
  gives digital text + tables — keeping them separate avoids contaminating retrieval
  evaluation with Docling-extracted text for the primary corpus.
- **Gemini 2.5 Flash with `thinking_budget=0`.** The default thinking budget eats
  `max_output_tokens` before JSON is emitted, so `response.parsed` comes back `None`;
  setting it to 0 guarantees the full budget is spent on the structured answer.
- **`doc_id` in vision point UUID.** The secondary corpus uses `page_NNN.png` filenames
  per document, so `uuid.uuid5(NAMESPACE_OID, filename)` collided across the two BIS PDFs
  (94 quarterly pages overwrote 94 annual pages); keying on `f"{doc_id}:{filename}"`
  eliminates the collision.
- **Regex doc-id extraction from primary filenames.** The primary QA file only covers 72
  of 972 pages, so the other 900 payloads were stamped `"unknown"`; parsing the UUID out
  of the flattened filename gives every page a real doc_id.

## Section 8 — Limitations discovered during implementation

- **RECITATION filter on government text.** 11 % of eval generations blocked before the
  schema fix; even with the fix the failure mode is latent on any query whose answer
  overlaps training data.
- **text_only is a tautology on primary corpus.** The text channel indexes synthesised
  `"Q: … A: …"` strings and is then queried with `Q`, so Hit@5 is 100 % by construction.
  This is why Section 2 excludes `text_only`.
- **Secondary corpus has no QA ground truth.** We can narrate qualitative multi-modal
  coverage (tables, footnotes) but cannot report a text-vs-vision numeric comparison on
  this corpus.
- **`std::bad_alloc` during Docling ingestion with OCR on.** The EasyOCR model plus the
  layout model exceed CPU/GPU memory on long PDFs; resolved by disabling OCR, but a
  scanned-document corpus would need OCR re-enabled with smaller batches.
- **Gemini 2.5 Flash thinking budget + structured output.** Silent failure mode:
  `response.parsed is None` with no actionable error until the raw `response.text` is
  logged. Set `thinking_budget=0` when using `response_schema`.
- **Free-tier Gemini quota.** 20 requests/day on one account blocked the first eval run;
  200 RPD on the current account is enough for one eval but not for iterative prompt
  tuning.
- **UUID collision between same-named pages across docs.** Discovered because 94
  secondary-corpus pages were silently overwritten; a generic fix (`doc_id` in the UUID
  seed) now guards all future indexers in this codebase.
- **Confidence field is uncalibrated.** 94/94 cached answers self-report
  `confidence: high`; the prompt does not instruct the model to reserve `medium` / `low`
  for weak evidence.
- **Ragas faithfulness / answer-relevancy not measured.** Deferred in favour of the
  lightweight-stack constraint; table headers in the report should be explicit about
  this.

## Section 9 — Model and dataset provenance

### Models

| Role | HF ID / Source | License | On-disk (HF cache) |
|---|---|---|---|
| Vision retriever | `vidore/colSmol-256M` (ColIdefics3 via `colpali-engine>=0.3.10`) | Apache 2.0 | ~512 MB |
| Text retriever | `BAAI/bge-small-en-v1.5` (Sentence-Transformers) | MIT | ~133 MB |
| PDF layout | `ds4sd/docling-models` + `docling-project/docling-layout-heron` | MIT / Apache 2.0 | ~200 MB |
| PDF OCR backend (disabled at runtime) | Bundled with Docling (`rapidocr-onnxruntime`) | Apache 2.0 | not used |
| Generator | `gemini-2.5-flash` (Google AI Studio API) | Google API ToS — free tier | API-only (no local weights) |
| PyTorch runtime | `torch 2.5.1+cu121`, `torchvision 0.20.1+cu121` | BSD-3 | CUDA 12.1 wheels |

### Datasets

| Role | Source | License | N |
|---|---|---|---|
| Primary corpus | `vidore/syntheticDocQA_government_reports_test` (HF) | MIT | 972 pages / 100 QA |
| Secondary #1 | `https://www.bis.org/publ/arpdf/ar2024e.pdf` | BIS terms (public report) | 150 pages |
| Secondary #2 | `https://www.bis.org/publ/qtrpdf/r_qt2409.pdf` | BIS terms (public report) | 94 pages |

### Runtime

| Component | Version |
|---|---|
| Python | 3.11 (uv-managed) |
| uv | project-managed; wheels pinned via `pyproject.toml` + `uv.lock` |
| Qdrant (Docker) | 1.17.1 (`qdrant-multimodal-rag` container) |
| GPU | NVIDIA RTX 3050 Ti Laptop, 4 GB VRAM, CUDA 12.1 |
| OS | Windows 11 Home, WSL-style bash shell via Claude Code |

## Section 10 — Sanity check

Run against the repo state at the end of the handoff task (2026-04-18).

| Check | Result |
|---|---|
| `uv run pytest -q` | **PASS** — 29 passed, 1 skipped, 0.25 s. |
| `uv run ruff check src` | **13 pre-existing errors** (mostly I001 import ordering, 10 auto-fixable with `--fix`). None introduced by this handoff task (no `src/` edits). See below. |
| `uv run mypy src` | **49 pre-existing errors** across 11 files — mostly `Missing type arguments for generic type "dict"` / `"list"` and a couple of assignment types. None introduced by this handoff task. |
| Streamlit launch (import-level smoke) | **PASS** — all modules import cleanly; `Citation` has `snippet`, no `quote`. Full launch was not re-run since port 8501/8502 were in use, but the import smoke confirms no new breakage. |
| `data/eval/report.md` numbers match §2 / §5 | **PASS with 1–6 ms P95 latency drift.** Retrieval Hit@k and MRR match exactly. P95 latency differs slightly (e.g. vision_only 282.1 vs 275.4 ms) because this document uses `sorted_list[int(0.95 * N)]` while `src/eval/report.py` computes P95 differently. Trust `data/eval/report.md` for the report — it is the artifact generated by the eval pipeline. Flagged; not a data integrity issue. |

### Ruff/mypy backlog

These are pre-existing technical-debt items in `src/`. The handoff spec explicitly says
"don't modify any source code in `src/`", so they are **reported but not fixed**.

```text
ruff: 13 errors — 10 auto-fixable with `uv run ruff check src --fix`
  I001 import ordering: src/retrieval/vision_retriever.py, src/indexing/vision_index.py,
       src/generation/generator.py, ... (10 of 13)
  (other rules hidden without --unsafe-fixes)

mypy: 49 errors in 11 files
  src/generation/generator.py: 5 errors (Client assignment, list[Any] type args, Any return)
  src/graph/qa_graph.py: 8 errors (dict[...] type args, StateGraph type arg, untyped call)
  ... (full report: uv run mypy src)
```

If the instructor grades on static-analysis cleanliness, run `uv run ruff check src --fix`
and add targeted `# type: ignore[...]` comments to the mypy findings in a follow-up PR.

## Missing-data flags

1. **Secondary-corpus QA set** — required for Section 3; suggest authoring 15–30 queries
   with `(doc_id, page_number, query_type)` annotations.
2. **Ragas metrics** — required to populate "faithfulness / answer_relevancy" columns in
   Section 4; requires `uv add ragas` and a judge LLM.
3. **Post-fix eval** — required to quote a real RECITATION-rate improvement; costs 100
   Gemini calls.
4. **Per-query-type breakdown** — blocked by (1); infrastructure exists once the QA set
   is written.
5. **Charts beyond `hit_at_k.png`** — latency box plots, confidence histograms, RECITATION
   timeline could all be generated from `data/eval/runs/*.jsonl` + `data/gemini_cache/`.
