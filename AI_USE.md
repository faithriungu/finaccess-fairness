# AI use disclosure


This file discloses how AI assistance was used in this project, as required by
the DSA 8401 course policy. `[CONFIRM: cite the exact policy section, if one is named.]`

## Tools used

| Tool | Used for |
| --- | --- |
| Claude (Anthropic) | Code review, debugging, and documentation, explaining methods |
 |

## What AI assistance was used for

**Code**

- Reviewing the scripted pipeline in `src/` step by step. I ran
  every step locally, checked its outputs, and asked for fixes when results
  looked wrong.
- Restructuring the repository for reproducibility: the single-command runner
  (`src/run_pipeline.py`), the ingestion and validation step, batch prediction
  (`src/predict.py`), model packaging (`src/package.py`), threshold robustness
  checks (`src/robustness.py`), calibration and segment reporting in
  `src/evaluate.py`, and the `pytest` suite.
- Diagnosing problems in earlier versions of the work, including a data-leakage
  problem that inflated the earlier results, and preprocessing choices that
  recoded survey non-responses as zeros.

**Writing**

- Drafting `README.md`, `DECISIONS.md` and `TECHNIQUES.md`, and reviewing a
  summary of results against the actual output files.


## What I did myself

- Chose the problem, the FinAccess 2024 data and the research question.
- Made the modelling and fairness decisions recorded in `DECISIONS.md`,
  including the target definition, the adult-only population, the predictor
  set, the champion model and the decision-threshold rule.
- Created and run the pipeline, checked every result against the output files, and
  interpreted the findings.


## How AI output was checked

- Every number in the report comes from files in `reports/` produced by the
  pipeline, not from AI-written text.
- The full pipeline was re-run from scratch to confirm the code works and
  reproduces the reported results, and the `pytest` suite passes.


## Data and privacy

No raw survey data was shared with AI tools beyond `[CONFIRM: e.g. "aggregated
output tables and a project archive containing the processed 48-column extract"]`.
The extract contains no names or contact details.
