# Archived exploration notebooks

These notebooks record the exploratory data analysis and data-quality audits
done **before** the scripted pipeline in `src/` was written. They are kept as a
record of how the cleaning decisions were reached.

They are **not** part of the reproducible pipeline and are not run by
`python -m src.run_pipeline`. Some cells use variables that the final pipeline
deliberately does not create (for example `income_volatility`, dropped because
`1 - lowest/highest` measures a relative income range, not volatility over time;
see `DECISIONS.md`). Where a notebook and `src/` disagree, `src/` is authoritative.
