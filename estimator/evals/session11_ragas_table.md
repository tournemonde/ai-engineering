# Session 11 — RAGAS generation-quality baseline

Judge: OpenAI `gpt-4o-mini` + `text-embedding-3-small`. Contexts = relevant budget components from `data/budgets_sample.json`. Answers = structured `Estimate` via Instructor / `LLMWrapper` (`gpt-4o-mini` harness; production default remains `GENERATION_MODEL`).

| Query | faithfulness | answer_relevancy | context_precision | context_recall |
| --- | ---: | ---: | ---: | ---: |
| Q1 | 0.529 | nan | 1.000 | 0.600 |
| Q2 | nan | nan | 0.806 | 0.400 |
| Q3 | 0.333 | nan | 1.000 | 0.500 |
| Q4 | nan | nan | 1.000 | 0.500 |
| Q5 | nan | nan | 0.893 | 0.600 |
| **average** | 0.431* | nan | 0.940 | 0.520 |

\*faithfulness average over non-NaN rows only (Q1, Q3).

## Note (2–3 sentences)

Lo más chirriante es que **answer_relevancy sale NaN en las cinco queries** (el juez no devolvió score usable sobre respuestas estructuradas module/task) y faithfulness también falla a NaN en Q2/Q4/Q5. Context_precision está alto (~0.94) porque los contextos son exactamente los presupuestos relevantes, pero context_recall ronda 0.52 — el ground_truth pide matices (p.ej. no inflar Q2 con marketplace) que el generador no verbaliza. Además, en Q1 el modelo devolvió `total_engineer_days=680` (casi las horas crudas del corpus): clásico conflation hours→days que la citación estructural no atrapa.

## Accompanying artifacts

- `evals/golden_generation_s11.json` — Q1–Q5 + `ground_truth`
- `evals/citation_report_sample.json` — dangling id=999 detected on purpose
- `evals/citation_report_from_generation.json` — verify_citations on a real Q1 generation (0 dangling)
- `scripts/eval_generation_ragas_s11.py` / `scripts/demo_citation_verification_s11.py`
