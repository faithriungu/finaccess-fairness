# Technique justification

The capstone brief asks why each technique was chosen **and** why plausible
alternatives were left out. This file covers both. Individual data and design
decisions are logged in `DECISIONS.md`.

## The problem shapes the method

- **Tabular survey data.** 19,750 adults and 35 mixed numeric and categorical predictors. No images, text or sequences.
- **Moderate imbalance.** About 30% of adults use a bank (weighted).
- **Complex survey design.** Respondents are clustered in enumeration areas and carry unequal sampling weights.
- **Real-world messiness.** Skip patterns and `#NULL!` codes, impossible income pairs, and missing education values.
- **A fairness question.** Does the model work equally well for women and men?

That points to classical supervised learning with honest validation, subgroup
evaluation and explainability. It does not point to more complex models.

## What was used, and why

| Technique | Why it fits |
| --- | --- |
| **Model ladder: dummy → logistic regression → random forest → gradient boosting** | Each rung must beat the one below. The dummy sets the chance PR-AUC (0.286), so gains are judged against no skill, not against zero. |
| **Gradient boosting (sklearn `HistGradientBoostingClassifier`)** | Strong default for tabular data; captures nonlinearities and interactions; best mean PR-AUC and best Brier score. See `DECISIONS.md` D10 for why its narrow lead is treated with caution. |
| **Nested CV that keeps clusters together** | Inner folds tune, outer folds estimate, and the test set is used once. Keeping clusters together prevents neighbours from sharing information across splits. |
| **Fold-internal preprocessing (`Pipeline` + `ColumnTransformer`)** | Imputers and encoders are fitted on training folds only, so no information leaks from validation data. |
| **Survey-weighted fitting and metrics** | Results describe Kenyan adults, not just the sample. Unweighted results are shown too. |
| **PR-AUC as the headline metric; ROC-AUC, Brier score and reliability diagram as diagnostics** | PR-AUC focuses on the minority positive class; the diagnostics check ranking and calibration. |
| **Cluster bootstrap CIs (1,000 resamples)** | Gender gaps carry uncertainty; resampling whole clusters respects the survey design. |
| **Equal opportunity (TPR gap) as the fairness criterion** | Asks whether actual bank users are found equally often among women and men. It is compatible with calibration when base rates differ (D12). |
| **Post-processing thresholds** | Fairness levers that don't require retraining. Comparing common and gender-specific thresholds shows the trade-off openly. |
| **TreeSHAP** | Exact, fast attributions for tree models. Used globally, by gender, and on errors to explain *why* recall differs, not only *that* it differs. |
| **Model package + batch prediction + `pytest`** | One versioned artefact (model, threshold, schema, metadata) with a reload check and contract tests, so that what was evaluated is exactly what gets served. |

## What was considered and left out

**Stacking.** Logistic regression, random forest and gradient boosting are within one fold-SD of each other (0.694 / 0.703 / 0.712 ± ~0.03). Stacking pays off when base models make different errors; here their gains are small and overlapping, so a meta-learner has little to combine. It would also add a second training layer, multiply serving cost, and make SHAP attributions harder to interpret. Not worth the operational cost.

**Deep learning (MLP, entity embeddings).** On small-to-medium tabular data, tree ensembles usually match or beat neural networks with far less tuning. An MLP would need scaling, embedding choices for 47 counties and other high-cardinality categories, and careful regularisation. It would also lose exact TreeSHAP. Nothing in the data (no images, text or sequences) needs representation learning.

**XGBoost / LightGBM.** sklearn's histogram gradient boosting uses the same algorithm family, works with survey sample weights inside an sklearn `Pipeline`, and adds no dependency to the API container. The expected difference in accuracy is small compared with the fold-to-fold variation seen here.

**SVM, k-NN.** Neither gives well-calibrated probabilities out of the box, and both scale poorly with many one-hot categorical columns. Calibrated probabilities matter here because the decision threshold is applied to them.

**SMOTE and class weights.** The imbalance is moderate (about 30% positive), not extreme. The course's recommended order of remedies is threshold moving, then class weights, then SMOTE. Threshold moving is enough here and keeps probabilities calibrated. Class weights and SMOTE both distort predicted probabilities and would require recalibration, and SMOTE would generate synthetic "respondents" whose survey weights mean nothing.

**Recalibration (Platt / isotonic).** Not applied, because the model is already well calibrated on the locked test set: weighted ECE is 0.019 overall, 0.027 for women and 0.038 for men (`reports/test_calibration_summary.csv`). Adding a calibrator would need its own held-out data and add complexity for little gain.

**Fairness-constrained training (e.g. fairlearn `ExponentiatedGradient`).** Possible, but it builds the fairness constraint into the model and usually needs the sensitive attribute during training. Post-processing was preferred because it leaves the calibrated scores unchanged and makes the accuracy-fairness trade-off visible and easy to adjust. In-processing is listed as future work for comparison.

**Demographic parity.** Rejected as the fairness target, because actual bank use differs by gender (23% vs 38%). Equal selection rates would require deliberately mislabelling people, and would conflict with calibration (D12). The selection-rate gap is still reported.

**Unsupervised methods (clustering, PCA).** The task has a clear label. PCA would make SHAP explanations unreadable in terms of original survey questions.

**Drift monitoring (Evidently AI), MLflow, CI/CD.** FinAccess 2024 is a single cross-sectional survey round, so there is no time dimension to monitor drift over until the next round. Experiments are tracked through versioned outputs in `reports/`, the data hashes in the model package, and the decision log. These tools would be the natural next step for a deployed service, but would add infrastructure without new evidence for this project.
