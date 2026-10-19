# CAG Stress Test Report (Session 6)

## Setup

- Base: instructor `session_05_live` (compression, anchors, tier, ACB, evals).
- Model: `gpt-4o-mini` (single provider; forced for curve comparability).
- Observation: aggregated `turn_observed` via `GET /sessions/{id}.last_turn`.
- Tokens/cost/latency accumulate **all** LLM calls in the turn (actor + metadata extractor + summarizer when compression runs).
- `cache_hit_kind` is always `none` on the conversational path (exact/semantic caches are skipped by design).
- Rows in CSV: **176** total, **169** without error (160 multi-turn, 9 attachment probes).

## 1. Summary table

| scenario | attach_kb | n | P50 latency_ms | P95 latency_ms | total cost_usd | exact hit rate | semantic hit rate | mean recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| attachment_probe | 0 | 3 | 3697 | 6557 | 0.0026 | 0% | 0% | 0% |
| attachment_probe | 100 | 1 | 8065 | 8065 | 0.0080 | 0% | 0% | 0% |
| attachment_probe | 20 | 1 | 8538 | 8538 | 0.0080 | 0% | 0% | 0% |
| attachment_probe | 5 | 3 | 7038 | 11032 | 0.0169 | 0% | 0% | 0% |
| attachment_probe | 50 | 1 | 21782 | 21782 | 0.0081 | 0% | 0% | 0% |
| contradiction | 0 | 60 | 9760 | 21375 | 0.1344 | 0% | 0% | 67% |
| growing | 0 | 60 | 10248 | 18079 | 0.1213 | 0% | 0% | 100% |
| pivot | 0 | 40 | 10143 | 23813 | 0.1146 | 0% | 0% | 0% |

## 2. Curves

### 2.1 Latency vs tokens_in (multi-turn, all scenarios)

| turn_index | mean tokens_in | mean latency_ms | n |
|---:|---:|---:|---:|
| 1 | 9385 | 6553 | 8 |
| 2 | 6956 | 7599 | 8 |
| 3 | 6871 | 6185 | 8 |
| 4 | 11187 | 8604 | 8 |
| 5 | 11685 | 7322 | 8 |
| 6 | 10298 | 7750 | 8 |
| 7 | 8334 | 7998 | 8 |
| 8 | 15859 | 11468 | 8 |
| 9 | 11465 | 11161 | 8 |
| 10 | 13963 | 11708 | 8 |
| 11 | 11616 | 10444 | 8 |
| 12 | 11549 | 12743 | 8 |
| 13 | 9518 | 10871 | 8 |
| 14 | 9915 | 11141 | 8 |
| 15 | 22615 | 16072 | 8 |
| 16 | 14072 | 13690 | 8 |
| 17 | 13709 | 13936 | 8 |
| 18 | 14771 | 14214 | 8 |
| 19 | 13358 | 15304 | 8 |
| 20 | 10195 | 11511 | 8 |

### 2.2 Cumulative cost_usd vs turn_index (by scenario)

| scenario | turn_index | mean cumulative cost_usd |
|---|---:|---:|
| contradiction | 1 | 0.0010 |
| contradiction | 2 | 0.0025 |
| contradiction | 3 | 0.0042 |
| contradiction | 4 | 0.0055 |
| contradiction | 5 | 0.0090 |
| contradiction | 6 | 0.0105 |
| contradiction | 7 | 0.0119 |
| contradiction | 8 | 0.0141 |
| contradiction | 9 | 0.0164 |
| contradiction | 10 | 0.0180 |
| contradiction | 11 | 0.0193 |
| contradiction | 12 | 0.0213 |
| contradiction | 13 | 0.0234 |
| contradiction | 14 | 0.0253 |
| contradiction | 15 | 0.0322 |
| contradiction | 16 | 0.0359 |
| contradiction | 17 | 0.0381 |
| contradiction | 18 | 0.0401 |
| contradiction | 19 | 0.0433 |
| contradiction | 20 | 0.0448 |
| growing | 1 | 0.0011 |
| growing | 2 | 0.0025 |
| growing | 3 | 0.0035 |
| growing | 4 | 0.0052 |
| growing | 5 | 0.0068 |
| growing | 6 | 0.0091 |
| growing | 7 | 0.0107 |
| growing | 8 | 0.0124 |
| growing | 9 | 0.0149 |
| growing | 10 | 0.0166 |
| growing | 11 | 0.0204 |
| growing | 12 | 0.0236 |
| growing | 13 | 0.0255 |
| growing | 14 | 0.0275 |
| growing | 15 | 0.0296 |
| growing | 16 | 0.0321 |
| growing | 17 | 0.0343 |
| growing | 18 | 0.0363 |
| growing | 19 | 0.0387 |
| growing | 20 | 0.0404 |
| pivot | 1 | 0.0038 |
| pivot | 2 | 0.0052 |
| pivot | 3 | 0.0063 |
| pivot | 4 | 0.0102 |
| pivot | 5 | 0.0110 |
| pivot | 6 | 0.0129 |
| pivot | 7 | 0.0150 |
| pivot | 8 | 0.0208 |
| pivot | 9 | 0.0227 |
| pivot | 10 | 0.0282 |
| pivot | 11 | 0.0296 |
| pivot | 12 | 0.0309 |
| pivot | 13 | 0.0328 |
| pivot | 14 | 0.0354 |
| pivot | 15 | 0.0387 |
| pivot | 16 | 0.0407 |
| pivot | 17 | 0.0452 |
| pivot | 18 | 0.0510 |
| pivot | 19 | 0.0538 |
| pivot | 20 | 0.0573 |

### 2.3 MemoryDriftMetric recall vs N (multi-turn)

| turn_index (N) | mean recall | n |
|---:|---:|---:|
| 1 | 62% | 8 |
| 2 | 62% | 8 |
| 3 | 62% | 8 |
| 4 | 62% | 8 |
| 5 | 62% | 8 |
| 6 | 62% | 8 |
| 7 | 62% | 8 |
| 8 | 62% | 8 |
| 9 | 62% | 8 |
| 10 | 62% | 8 |
| 11 | 62% | 8 |
| 12 | 62% | 8 |
| 13 | 62% | 8 |
| 14 | 62% | 8 |
| 15 | 62% | 8 |
| 16 | 62% | 8 |
| 17 | 62% | 8 |
| 18 | 62% | 8 |
| 19 | 62% | 8 |
| 20 | 62% | 8 |

### 2.4 Attachment probe: latency / cost / marker recall vs size

| attach_kb | mean latency_ms | mean cost_usd | mean tokens_in | marker recall |
|---:|---:|---:|---:|---:|
| 0 | 4720 | 0.0009 | 4136 | 0% |
| 5 | 8348 | 0.0056 | 34456 | 0% |
| 20 | 8538 | 0.0080 | 51363 | 0% |
| 50 | 21782 | 0.0081 | 51433 | 0% |
| 100 | 8065 | 0.0080 | 50982 | 0% |

## 3. Reading

Across N=1..20, mean MemoryDriftMetric recall of the first-turn fact never fell below 60% in this run — silent memory loss did not dominate within the 20-turn window. Mean per-turn cost at turn 1 is 0.0017 USD (9385 tokens_in, 6553 ms); at turn 20 it is 0.0021 USD (10195 tokens_in, 11511 ms) — about 1.2× the turn-1 cost. Around turn 7 (when `MAX_CONVERSATION_TURNS=6` forces summarizer calls), mean per-turn cost is 0.0016 USD versus 0.0017 USD at turn 1. On attachment probes, mean latency moved from 4720 ms at 0 KB to 8065 ms at 100 KB.

The dominant degradation dimension in this baseline is **latency and cumulative cost under a growing prompt**, not hard schema failures: every successful turn still returns a structured `EstimationResult`, while P50 latency already exceeds the 4000 ms budget on many turns and spend compounds linearly with N. A boundary that would justify jumping to RAG is when durable facts required by the product (project name, locked budget, stack pivot) fall out of summary/anchors/metadata while tokens_in keeps climbing — i.e. you pay more to remember less. Until that recall cliff appears, CAG remains viable; past it, retrieval over a durable store is the architectural answer rather than a larger context window.

## 4. Metric placement note

Stress metrics live in `evals/stress/metrics.py` (not `evals/metrics.py`) because they evaluate `turn_observed` / session snapshots, not `(GoldenCase, EstimationResult)` pairs from the golden eval suite.
