# AI use disclosure


This file discloses how AI assistance was used in this project, as required by
the DSA 8401 course policy. 

## Tools used

| Tool | Used for |
| --- | --- |
| Claude (Anthropic) | Code review, debugging, drafting pipeline code and documentation, explaining methods |


## What AI assistance was used for

**Code Development**

- Reviewing the scripted pipeline in `src/`. I ran
  every step locally, checked its outputs, and asked for fixes when results
  looked wrong.
- Restructuring the repository for reproducibility: the single-command runner
  (`src/run_pipeline.py`), the ingestion and validation step, batch prediction
  (`src/predict.py`), model packaging (`src/package.py`), threshold robustness
  checks (`src/robustness.py`), calibration and segment reporting in
  `src/evaluate.py`, and the `pytest` suite.
- Drafting the FastAPI service (`api/`), its generated request schema and
  contract tests, the `Dockerfile` and `docker-compose.yml`. I built and ran
  the container myself.
- Diagnosing problems in earlier versions of the work, including a data-leakage
  problem that inflated the earlier results, and preprocessing choices that
  recoded survey non-responses as zeros.

**Writing**

- Reviewing `README.md`, `DECISIONS.md` and `TECHNIQUES.md` against
 the actual output files.
- AI tools were also used to explain technical concepts and help organise findings
  into language suitable for project documentation and presentation.

## What I did myself

- Reviewed AI-assisted outputs and independently evaluated which suggestions to adopt, modify or reject.
- Conducted exploratory data analysis (EDA) on the FinAccess 2024 dataset, examining data distributions, missing values and inconsistencies.
- Identified and selected relevant variables and prepared the dataset for machine learning.
- Developed and reviewed descriptive statistics and visualisations to understand patterns in the data.
- Examined relationships between selected predictors and traditional bank use.
- Evaluated and compared the performance of the candidate machine learning models.
- Analysed gender-based differences in model performance and fairness metrics.
- Interpreted the findings, identified project limitations and prepared the final project outputs and presentation materials.


## Declaration

I acknowledge the use of the AI tools listed above in the development of this project. Their use was intended to support technical implementation, troubleshooting and documentation. 

I remain responsible for the final code, methodological decisions, interpretation of results, accuracy of the submitted work and compliance with the university's academic integrity requirements.
