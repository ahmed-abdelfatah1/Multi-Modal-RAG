# Demo transcript

Raw output from `scripts/capture_demo.py`. Each section is one demo query.

## query_1 — Primary corpus, both channels hit

**Query.** What types of revenue should MMS account for as a custodial activity?

### Retrieved (top-6)

| Rank | Source | doc_id | Page | Score |
|---|---|---|---|---|
| 1 | `page` | `18b0d914-cdcf-4cb9-b638-ad03478c44a4.pdf` | 132 | 0.0328 |
| 2 | `text` | `68349c6d-3980-467c-a518-05b509218d54.pdf` | 21 | 0.0161 |
| 3 | `page` | `26a0b975-3e05-44a8-a0d9-6741f6bfda08.pdf` | 40 | 0.0161 |
| 4 | `text` | `0adf903b-924b-414c-b7e1-14b7fc3ccdbb.pdf` | 50 | 0.0159 |
| 5 | `page` | `ef6bc11c-6fc0-413e-8195-979067747a58.pdf` | 6 | 0.0159 |
| 6 | `text` | `68349c6d-3980-467c-a518-05b509218d54.pdf` | 77 | 0.0156 |

### Generated answer

**Confidence.** `high`

> The Minerals Management Service (MMS) should account for exchange revenue as a custodial activity. This includes rents, royalties, and bonuses that are collected on behalf of the General Fund, states, Indian tribes, and allottees. The revenue amounts should be recognized and measured according to the exchange revenue standards when they are due based on contractual agreements [18b0d914-cdcf-4cb9-b638-ad03478c44a4.pdf p.132]. Additionally, proceeds from the auctioning of radio spectrum, where the Federal Communications Commission (FCC) collects revenue on behalf of the U.S. Government, should also be treated as exchange revenue for a custodial activity [18b0d914-cdcf-4cb9-b638-ad03478c44a4.pdf p.132].

**Citations.**

1. `18b0d914-cdcf-4cb9-b638-ad03478c44a4.pdf` p.132 — MMS should consider exchange revenue as a custodial activity, including rents, royalties, and bonuses collected for the General Fund, states, Indian tribes, and allottees, and recognize these amounts based on contractual agreements.
2. `18b0d914-cdcf-4cb9-b638-ad03478c44a4.pdf` p.132 — The FCC should also treat proceeds from radio spectrum auctions as exchange revenue for a custodial activity, as it collects these funds for the U.S. Government.

## query_2 — Secondary corpus (BIS), table/chart dominant

**Query.** What does the BIS say about non-bank financial intermediation risks in 2024?

**Source.** Captured from the live Streamlit UI (top_k=8, per `src/graph/qa_graph.py:29`).
An earlier `scripts/capture_demo.py` run used top_k=6 and returned _"I cannot answer
from the provided sources"_ because the `bis_annual_report_2024` p.17 chunk fell outside
the cut-off. Keeping the UI output as the canonical demo response.

### Retrieved (top-8, from UI)

| Rank | Source | doc_id | Page | Score |
|---|---|---|---|---|
| 1 | `page` | `bis_annual_report_2024` | 17 | _(primary cited)_ |
| 2 | `text` | `bis_quarterly_review_2024q3` | 4 | 0.032 |
| 3 | `text` | `bis_annual_report_2024` | 13 | 0.030 |
| 4 | `page` | `bis_quarterly_review_2024q3` | 91 | 0.016 |
| 5 | `text` | `bis_annual_report_2024` | 1 | 0.016 |
| 6 | `page` | `bis_annual_report_2024` | 37 | 0.016 |
| 7 | `text` | `bis_quarterly_review_2024q3` | 1 | 0.016 |
| 8 | `page` | `bis_quarterly_review_2024q3` | 77 | 0.016 |

### Generated answer

**Confidence.** `high`

> The BIS emphasizes that the current macroprudential framework, developed after the Great Financial Crisis, needs to be strengthened to fully address risks in the non-bank financial intermediation sector. This requires clear policy mandates ... _(truncated in UI capture)_

**Citations.**

1. `bis_annual_report_2024` p.17 — strengthening the post-GFC macroprudential framework
   for NBFI risks; requires clear policy mandates.

## query_3 — Primary corpus, hybrid recovers vision miss

**Query.** How have communities adapted their recycling programs?

### Retrieved (top-6)

| Rank | Source | doc_id | Page | Score |
|---|---|---|---|---|
| 1 | `page` | `4bb3a383-85e5-4cd8-9c7e-3f9fb7853f81.pdf` | 18 | 0.0318 |
| 2 | `page` | `875fa820-331a-4720-9d38-b13d5b8cf5e8.pdf` | 136 | 0.0164 |
| 3 | `text` | `875fa820-331a-4720-9d38-b13d5b8cf5e8.pdf` | 41 | 0.0161 |
| 4 | `page` | `4bb3a383-85e5-4cd8-9c7e-3f9fb7853f81.pdf` | 37 | 0.0161 |
| 5 | `text` | `875fa820-331a-4720-9d38-b13d5b8cf5e8.pdf` | 108 | 0.0159 |
| 6 | `page` | `875fa820-331a-4720-9d38-b13d5b8cf5e8.pdf` | 101 | 0.0159 |

### Generated answer

**Confidence.** `high`

> Communities have adapted their recycling programs in various ways, including changing collection frequencies and methods. For instance, Austin, Texas, switched to weekly collection of recyclables and moved from manual to semi-automated rearload vehicles. Central Virginia Waste Management Authority also transitioned to weekly collection of recyclables. Edmond, Oklahoma, reduced its recycling collection to once per week and replaced manual rearload collection vehicles with fully automated sideloaders. Greensboro, North Carolina, also reduced its recycling collection frequency to once per week. Houston, Texas, implemented pilot studies to reduce collection frequency and added yard trimmings collection. Indianapolis, Indiana, reduced recycling collection to once per week and adopted automated sideloaders for both recyclables and yard trimmings [4bb3a383-85e5-4cd8-9c7e-3f9fb7853f81.pdf p.18].

**Citations.**

1. `4bb3a383-85e5-4cd8-9c7e-3f9fb7853f81.pdf` p.18 — Several communities, including Austin, Central Virginia, Edmond, Greensboro, Houston, and Indianapolis, have modified their recycling programs by altering collection frequency and adopting automated collection vehicles.
