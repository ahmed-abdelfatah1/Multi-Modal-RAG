# Demo queries

Three queries for the demo video. Selection constraints from the handoff spec could not
be satisfied exactly because **there is no secondary-corpus QA eval** (see
`docs/report_data.md` §3). Two of the three queries are therefore **hand-authored
against the indexed BIS corpus** rather than drawn from eval ground truth; this is
flagged per query below.

## Query 1 — primary-corpus, text-dominant, both channels hit

```
What types of revenue should MMS account for as a custodial activity?
```

| Field | Value |
|---|---|
| Source | `data/corpus_primary/qa.jsonl` (eval ground truth) |
| Expected doc_id | `18b0d914-cdcf-4cb9-b638-ad03478c44a4.pdf` |
| Expected page | 132 |
| Expected answer (from dataset) | `['rents', 'royalties', 'bonuses']` |
| text_only top-3 | `[132, 21, 50]` → Hit@1 ✓ |
| vision_only top-3 | `[132, 40, 6]` → Hit@1 ✓ |
| hybrid top-3 | `[132, 21, 40]` → Hit@1 ✓ |

**Why this query.** Clean case where the text channel wins because the synthesised
QA-anchor makes the match trivial, vision wins because the source page visually centres
the revenue table, and hybrid keeps both signals. Good opening demo — short answer,
well-cited, no RECITATION risk (already confirmed in cache).

## Query 2 — secondary-corpus, table/chart dominant (hand-authored)

```
What does the BIS say about non-bank financial intermediation risks in 2024?
```

| Field | Value |
|---|---|
| Source | **Hand-authored** — not in any eval ground-truth file |
| Target doc_id | `bis_annual_report_2024` |
| Target page | Unknown — verified at demo time by retrieval output |
| Query type (intended) | `table` / `chart` — BIS annual reports contain NBFI risk heatmaps |

**Why this query.** The BIS Annual Report 2024 dedicates multiple pages to non-bank
financial intermediation (NBFI) — a policy-heavy topic whose canonical expression is a
risk heatmap and supporting tables. Vision retrieval should surface the chart page;
text retrieval should surface a narrative chunk from the report body.

**Caveat.** Because no ground-truth page has been annotated for this query, the demo
cannot claim "vision_only hit, text_only missed" numerically — narrate the retrieved
pages qualitatively instead.

## Query 3 — cross-page, hybrid beats both singles (primary, approximated)

```
How have communities adapted their recycling programs?
```

| Field | Value |
|---|---|
| Source | `data/corpus_primary/qa.jsonl` (eval ground truth) |
| Expected page | 18 |
| text_only top-3 | `[18, 41, 108]` → Hit@1 ✓ (tautology) |
| vision_only top-3 | `[136, 37, 101]` → Hit@1 ✗, Hit@5 ✗ |
| hybrid top-3 | `[18, 136, 41]` → Hit@1 ✓ |

**Why this query.** A real eval case where vision_only **missed** within top-5 but the
hybrid RRF lifted the correct page to rank 1 by combining the biased-text signal with
vision. This is the tightest evidence in the existing eval data that fusion adds value.
Narrate it as "vision alone puts an unrelated infographic on top; hybrid recovers the
right page because the text channel knows which synthesised QA-anchor matches."

**Caveat.** The text channel's contribution here is from the synthesised QA-anchor, not
OCR'd document text — explain this honestly in the video narration so viewers understand
why hybrid ≠ "text-retrieval-as-in-production-RAG".

## Recording order recommendation

1. **Query 1** as the system's "happy path" baseline (30 s).
2. **Query 2** to show multi-modal coverage of the secondary corpus (60 s of retrieval
   walk-through + generation).
3. **Query 3** as the motivation for the hybrid pipeline (45 s, with the single-mode
   versus hybrid comparison on screen).
