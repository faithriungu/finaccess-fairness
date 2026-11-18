# Decision log

Each entry records what was decided, what else was considered, and why. Numbers
come from files in `reports/`. Methodological reasoning for the choice of
techniques is in `TECHNIQUES.md`.

## Data and population

### D1. Study population: adults aged 18+
- **Decision:** Keep respondents aged 18 and over (19,750 of 20,871; 1,121 excluded).
- **Why:** Bank-account ownership is an adult outcome; minors can't hold accounts in their own name, so including them would mix two populations.
- **Evidence:** `reports/data_cleaning_audit.xlsx`, sheet *Sample flow*.

### D2. Target: traditional bank use
- **Decision:** `target_bank_account` from FinAccess `Traditional_bank` (Usage = 1, Non-usage = 0).
- **Definition:** The FinAccess variable dictionary describes this as commercial bank use, excluding M-Shwari and KCB M-Pesa. The pipeline does not assume microfinance banks are included. `[CONFIRM against the FinAccess 2024 codebook.]`
- **Rejected alternative:** `target_formal_access`. Every bank user is also formally included (5,187 of 5,187), so formal access is a broader outcome that contains the target. Using it as a predictor would leak the answer.

### D3. Variables excluded from the predictors
| Variable | Reason |
| --- | --- |
| `gender` | Sensitive attribute. Kept only for evaluation and for the threshold comparison (D11). |
| `hh_head_gender` | Sex of the household **head**, not the respondent. 94% of spouses are coded male, so it would act as a gender stand-in. |
| `target_formal_access`, `mobile_money_access` | Other financial-inclusion outcomes. Using them would predict one kind of financial product use from another. |
| `num_children`, `num_children_under5`, `num_indigenous_cattle` | Missingness or skip logic not established (cattle missing for more than 80% of adults). Missing counts are not set to zero. |
| `owns_house`, `land_ownership_document` | Replaced by the derived `land_title_status` (D6). |
| `survey_weight`, `cluster_id`, `respondent_id` | Survey design fields and identifiers, not respondent characteristics. |

### D4. Missing values are not treated as zero or "no"
- **Decision:** `#NULL!`, blanks, "Refused" and "Don't know" stay missing, or are kept as their own named category. They are never recoded to 0, "No" or "No schooling".
- **Why:** In a survey, a skipped question isn't a "no". Recoding skips as zero invents data and can create spurious patterns.
- **How:** Missing values are filled in inside each training fold: median plus a missing-indicator for numeric columns, a constant `__MISSING__` category for categorical ones.

### D5. Income
- **Decision:** Income amounts are converted to numbers (KSh) and log-transformed inside the pipeline. 71 adults whose reported lowest monthly income exceeds their highest have both values set to missing.
- **Rejected:** An `income_volatility = 1 - lowest/highest` feature. It measures a relative income range at one point in time, not volatility over time, so the name would mislead.

### D6. Ordered survey answers kept as categories
- **Decision:** Wealth quintile, financial-literacy score, distance and cost bands, and education are one-hot encoded, not converted to numbers.
- **Why:** The gaps between bands aren't equal; for example, "1-2 km" to "2-5 km" isn't the same step as "5-10 km" to "10+ km". Numeric codes would impose a spacing the survey doesn't support. The official wealth-index and literacy-score formulas aren't documented in the extract.

## Validation design

### D7. Splits keep whole clusters together; the test set is locked
- **Decision:** 20% of **clusters** (3,942 adults) are held out as the locked test set. Model selection uses nested cross-validation that also keeps clusters together: 5 outer folds, 3 inner folds.
- **Why:** Respondents in the same enumeration area share local conditions such as bank-branch access. A random split would put neighbours in both training and test, and overstate performance.
- **Check:** `tests/test_pipeline_outputs.py` confirms that no cluster appears in both training and test.

### D8. Survey weights used for fitting and reporting
- **Decision:** KNBS survey weights (scaled to mean 1) are used when fitting, and weighted metrics are reported alongside unweighted ones.
- **Why:** Weighted results describe Kenyan adults; unweighted results describe the sample. Both are shown so neither is hidden.

### D9. Selection metric: weighted PR-AUC
- **Decision:** Models are ranked by mean survey-weighted PR-AUC across the outer folds.
- **Why:** Only about 30% of adults (weighted) use a bank, and the task is finding users. PR-AUC concentrates on the positive class; ROC-AUC can look good while precision is poor. A no-skill model's PR-AUC equals the base rate, so a dummy baseline is included to show it.

## Modelling

### D10. Champion: gradient boosting (HistGradientBoostingClassifier)
- **Model ladder (nested CV, weighted PR-AUC, mean ± SD):**

  | Rung | Model | PR-AUC | Lead over next rung |
  | --- | --- | --- | --- |
  | 0 | Dummy (prior) | 0.286 ± 0.006 | - |
  | 1 | Logistic regression | 0.694 ± 0.035 | +0.408 |
  | 2 | Random forest | 0.703 ± 0.034 | +0.009 |
  | 3 | Gradient boosting | 0.712 ± 0.029 | +0.009 |

- **Honest reading:** Every real model is far above chance. Gradient boosting has the highest mean, but each step up from logistic regression (about +0.009) is smaller than the fold-to-fold SD (about 0.03). The three models are statistically indistinguishable on this criterion. `reports/model_cv_summary.csv`, column `lead_exceeds_1sd`.
- **Why gradient boosting anyway:** It was best under the criterion fixed in advance (D9), has the lowest Brier score (best-calibrated probabilities), handles mixed numeric and categorical inputs with interactions without manual feature engineering, and gets exact, fast SHAP explanations (TreeSHAP). It also serialises and serves as a small file (about 170 KB).
- **Credible alternative:** Logistic regression performs about as well and is easier to explain. If simplicity were the overriding goal, it would be a defensible choice. Its results are kept in `reports/model_cv_fold_results.csv`.
- **Rejected:** Stacking. With base models this close, a stack can gain little, and it would make SHAP explanations and deployment harder. See `TECHNIQUES.md`.

### D11. Decision threshold: one gender-blind threshold, 0.320
- **Decision:** The deployed model labels someone a likely bank user when the predicted probability is at least 0.320. This is the common threshold that maximises weighted F1 on the **training** out-of-fold predictions; the test set was not used.
- **Compared on the test set** (`reports/mitigation_comparison.csv`):

  | Rule | Weighted F1 | Female TPR | Male TPR | TPR gap (M - F) [95% CI] |
  | --- | --- | --- | --- | --- |
  | Common 0.50 | 0.668 | 0.519 | 0.675 | +0.156 [0.072, 0.231] |
  | **Common 0.320 (deployed)** | 0.690 | 0.703 | 0.794 | +0.091 [0.022, 0.155] |
  | Gender-specific (F 0.245 / M 0.380) | 0.691 | 0.806 | 0.754 | -0.052 [-0.131, 0.025] |

- **Why not gender-specific thresholds:** They close the gap, and in fact reverse it, but they require reading each person's gender at decision time. That is direct differential treatment by sex. Article 27 of the Constitution of Kenya prohibits sex discrimination; Article 27(6) permits affirmative action, but an automated rule that uses sex directly would need explicit legal and policy justification. It would also sit uneasily with the non-discrimination expectations in the CBK Digital Credit Providers Regulations 2022 and the automated-decision provisions of the Data Protection Act 2019. They are reported as a fairness benchmark, not used.
- **Stability:** The threshold-selection rule was re-run on 200 cluster-bootstrap resamples of the training predictions (`reports/threshold_stability.csv`). The common threshold varied from 0.305 to 0.423 (95% interval). The F1 curve is flat between about 0.25 and 0.40 (`reports/figures/threshold_sweep_test.png`), so F1 alone barely separates thresholds in that range. Lower thresholds in that range also give smaller TPR gaps. Choosing the threshold to target a gap tolerance, using training data only, is listed as future work.

### D12. Fairness criterion: equal opportunity
- **Decision:** The main fairness measure is the gap in true positive rate (recall) between men and women, with cluster-bootstrap 95% CIs. Selection-rate and FPR gaps and per-group calibration are reported alongside.
- **Why not demographic parity:** Actual bank use differs by gender (weighted 23.4% of women vs 38.0% of men in the test set). Forcing equal selection rates would require mislabelling people whose real status is known. When base rates differ, a model can't be both calibrated and have demographic parity. The model is calibrated by gender (ECE 0.027 for women, 0.038 for men; `reports/test_calibration_summary.csv`), and that is kept.

## Repository and process

### D13. Exploration notebooks archived
- `notebooks/archive/` holds the EDA and audit notebooks that came before the scripted pipeline. They are not run by the pipeline, and some use the rejected `income_volatility` feature (D5). `src/` is authoritative.

### D14. Model package contents
- `models/model_package.joblib` holds the fitted pipeline, the 0.320 threshold, an input schema learnt from the training partition, and metadata: version, git commit, SHA-256 hashes of both data files, library versions, and test metrics. `src/package.py` saves it, reloads it, and fails unless the reloaded package reproduces the predictions exactly.

### D15. Serving design
- **Decision:** FastAPI service in Docker. The request schema is generated from the model package, so the API validates exactly what the model was trained on; unknown fields and categories are rejected with 422. Endpoints are synchronous `def`, because scoring is CPU-bound. The model is loaded once at startup, and errors fail closed (503/500, never a default decision).
- **Reason codes:** Each prediction returns its top 3 SHAP contributions. This supports the right to an explanation of automated decisions under the Data Protection Act 2019, and lets a reviewer see why a person was scored as they were.
- **Data minimisation:** Gender is not an API field. Request inputs are not logged; only the request ID, decision and latency are.
- **Rejected:** Building the schema at runtime from the package. A generated file is readable by reviewers, and lets requests be validated even if the model fails to load.

## Known limitations
- **Gender can be inferred from other inputs.** Leaving gender out doesn't make the model gender-blind. During review, a gradient-boosting classifier predicted respondent gender from the 35 features with AUC about 0.92, mainly through relationship to household head, marital status and household decision-making. That check isn't part of the pipeline.
- **Gaps are larger within subgroups.** At 0.50, recall is 0.49 for rural women and 0.74 for urban men (`reports/test_segment_metrics.csv`). The gender-and-location gap is larger than the gender gap alone.
- **Cross-sectional data.** FinAccess 2024 is one survey round. Nothing here shows how well the model would hold up in a later year.
- **Prediction, not causation.** SHAP values explain the model, not cause and effect.
