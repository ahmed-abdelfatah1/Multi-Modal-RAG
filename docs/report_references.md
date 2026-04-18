# Report references

Code pointers for the design decisions in `docs/report_data.md` §7, plus verbatim copies
of the system prompt, the `Answer` schema, and the RRF formula. Line numbers refer to
the state of the repo at the time of writing; re-check after any refactor.

## Design-decision → code map

| Decision | File:lines | What to look at |
|---|---|---|
| ColSmol-256M over ColQwen2-v1.0 | `src/indexing/vision_index.py:30-35` | `MODEL_NAME = "vidore/colSmol-256M"` with the comment naming the 4 GB VRAM constraint. |
| Gemini 2.5 Flash via `google-genai` | `src/generation/generator.py:146-168` | `_call_api` builds the request; model string is literal `"gemini-2.5-flash"`. |
| Qdrant multi-vector + binary quantisation | `src/indexing/vision_index.py:167-180` | `create_collection(...)` with `MultiVectorConfig(MAX_SIM)` and `BinaryQuantization`. |
| RRF fusion, k=60 | `src/retrieval/hybrid.py:19-74` | `RRF_K = 60`; `reciprocal_rank_fusion()` implements the formula. |
| Paraphrased `snippet` over verbatim quote | `src/generation/schema.py:13-25` | Comment above `snippet` field explains the RECITATION fix. |
| Docling with OCR disabled | `src/ingestion/parse.py:15-23` | `do_ocr = False` with comment on why; `do_table_structure = True` retained. |
| Separate primary / secondary corpora | `src/data/download.py` + `src/ingestion/run_all.py` | Primary loaded from HF, secondary downloaded + parsed separately. |
| `thinking_budget=0` for Gemini 2.5 Flash | `src/generation/generator.py:161-164` | `types.ThinkingConfig(thinking_budget=0)` with comment on the structured-output failure mode. |
| `doc_id` in vision point UUID | `src/indexing/vision_index.py:57-66` | `generate_point_id` seeds UUID with `f"{doc_id}:{image_filename}"`. |
| Regex doc-id extraction | `src/indexing/vision_index.py:46-55` | `_PRIMARY_DOC_ID_RE` + `_extract_primary_doc_id` plus usage at `:78-82`. |
| Tenacity retry with `reraise=True` | `src/generation/generator.py:141-146` | Decorated `_call_api` retries 3×; `reraise` surfaces the underlying `ClientError`. |
| Retrieval top_k = 5 in eval | `src/eval/run.py:24-26` | `TOP_K = 5` and `MIN_SECONDS_BETWEEN_GEMINI_CALLS = 7`. |
| Generation image cap at 4 pages | `src/generation/generator.py:20` | `MAX_IMAGES = 4` — kept under Gemini's ~20 MB payload cap. |
| Streamlit stderr capture for cache-hit flag | `src/app/main.py:64-90` | `contextlib.redirect_stderr` around `graph.invoke` to detect `"cache hit"` string. |

## Verbatim `SYSTEM_PROMPT`

Source: `src/generation/prompts.py:3-25`.

```text
You are a precise document QA assistant. Your task is to answer questions based ONLY on the provided source documents.

CRITICAL RULES:
1. ONLY use information from the provided sources. Do not use prior knowledge.
2. Cite every factual claim with [doc_id p.N]. For each citation, write a short snippet in your own words (≤30 words) describing what the source says. Do NOT reproduce sentences verbatim from the sources — paraphrase and synthesize. Multiple citations supporting the same claim are encouraged.
3. If the sources do not contain enough information to answer, say "I cannot answer from the provided sources" with confidence="low".
4. Be concise but complete.

You will receive:
- A question
- Text chunks from documents (with source identifiers)
- Page images from documents (with source identifiers)

Respond with a JSON object matching this schema:
{
    "answer": "Your answer with inline citations like [source p.1], in your own words",
    "citations": [
        {"doc_id": "source_name", "page_number": 1, "snippet": "short paraphrase of what this source says (≤30 words)"}
    ],
    "confidence": "high" | "medium" | "low",
    "missing_info": "What information was missing, if any" | null
}
```

### `REGEN_SUFFIX` (appended when validator requests a regeneration)

```text
IMPORTANT: Your previous response contained citations that don't match the provided sources.
Please review the sources again and ensure EVERY citation refers to a document and page number that was actually provided.
Only cite sources you can see in the context above.
```

## Verbatim `Answer` Pydantic schema

Source: `src/generation/schema.py`.

```python
from typing import Literal
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A citation to a source document."""

    doc_id: str = Field(..., description="Document identifier")
    page_number: int = Field(..., description="Page number")
    # Changed from required verbatim quote to short paraphrased snippet.
    # Verbatim long quotes trigger Gemini's RECITATION safety filter on
    # canonical documents like government reports. A short paraphrase
    # preserves the citation's purpose (pointing to source) without
    # reproducing training data.
    snippet: str = Field(
        ...,
        max_length=400,
        description=(
            "A short paraphrased snippet (≤30 words) describing what this "
            "source says, in your own words. Do NOT quote verbatim."
        ),
    )


class Answer(BaseModel):
    """Generated answer with citations."""

    answer: str = Field(..., description="The generated answer")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations supporting the answer",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Confidence level in the answer",
    )
    missing_info: str | None = Field(
        default=None,
        description="Information that was missing to fully answer",
    )
```

## RRF formula as implemented

Source: `src/retrieval/hybrid.py:23-74`.

The fusion score for a document `d` appearing in any input ranked list is

```
score(d) = Σ_i   1 / (k + rank_i(d))
          i ∈ lists containing d
```

with `k = 60` (`src/retrieval/hybrid.py:20`), matching the Cormack/Clarke default
(TREC 2009 proceedings). Documents are deduplicated by `f"{doc_id}:{page_number}"`
(`src/retrieval/hybrid.py:43`). When the same page is returned both as a `page` hit
(from ColSmol) and as a `text` hit (from bge-small-en), the fused record keeps the
`page` payload (`src/retrieval/hybrid.py:52-54`) because it carries the rendered image
for downstream generation.

### Minimal Python expression of the formula

```python
RRF_K = 60  # src/retrieval/hybrid.py:20
scores = defaultdict(float)
for results in results_list:               # one list per retriever
    for rank, item in enumerate(results, 1):
        key = f"{item.doc_id}:{item.page_number}"
        scores[key] += 1 / (RRF_K + rank)
ranked = sorted(scores, key=scores.get, reverse=True)
```

### Retriever fan-out

`src/retrieval/hybrid.py:120-131` runs the two retrievers in parallel via
`ThreadPoolExecutor(max_workers=2)` so hybrid wall-clock latency is `max(text, vision)`
rather than the sum — this is visible in §5 of `report_data.md` where `hybrid.mean`
(199.8 ms) is below `vision_only.mean` (257.7 ms).
