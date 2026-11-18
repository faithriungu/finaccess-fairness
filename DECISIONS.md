# Project decisions

This file summarises the main choices made during the project and why they were made. The reported figures are taken from the project output files.

## 1. Data and study population

### 1.1 Study population

The analysis includes adults aged 18 and above. Of the 20,871 respondents in the selected dataset, 19,750 met this age requirement and 1,121 were excluded.

The study focuses on adult bank use and adult financial decision-making. The sample restriction is therefore applied before modelling.

**Source:** `reports/data_cleaning_audit.xlsx`, sheet `Sample flow`.

### 1.2 Target variable

The outcome is `target_bank_account`, created from the FinAccess `Traditional_bank` variable:

- `Usage` = 1
- `Non-usage` = 0

The project uses this outcome to predict traditional bank use among adults. The report will state the precise product definition given in the FinAccess 2024 documentation, including the treatment of mobile-based products.

`target_formal_access` was not used as a predictor. It is a broader financial-inclusion outcome and overlaps with the target: all 5,187 bank users in the selected data are also classified as formally included. Including it would give the model information closely tied to the answer it is meant to predict.

### 1.3 Predictor selection

The model uses 35 respondent characteristics covering demographic, household, economic, digital and geographic information.

The following variables are not included as predictors:

| Variable | Reason |
|---|---|
| `gender` | Kept for the fairness analysis, not used by the main prediction model. |
| `hh_head_gender` | Describes the household head, not necessarily the respondent. |
| `target_formal_access`, `mobile_money_access` | Other financial-inclusion outcomes that could introduce target leakage. |
| `num_children`, `num_children_under5`, `num_indigenous_cattle` | Not used because their missingness or survey skip patterns were not sufficiently clear. |
| `owns_house`, `land_ownership_document` | Replaced by the derived `land_title_status` feature. |
| `survey_weight`, `cluster_id`, `respondent_id` | Used for survey analysis, data splitting or identification, not as model predictors. |

Gender is therefore not a direct input to the main model. This does not mean that the remaining variables contain no information associated with gender. Fairness is assessed separately by comparing model performance for women and men.

### 1.4 Missing values and income

Survey blanks, `#NULL!`, refusals and “Don't know” responses are not automatically treated as “No” or zero. Where appropriate, missing values are retained and handled within the training process.

Numeric variables are imputed using the training data, with missing-value indicators. Missing categorical values are assigned a separate category.

Income amounts are converted to numeric KSh values and log-transformed in the modelling pipeline. For 71 adults whose reported lowest monthly income exceeded their highest monthly income, both values were set to missing.

### 1.5 Categorical survey responses

Wealth quintile, financial-literacy score, distance bands, cost bands and education are treated as categories. They are one-hot encoded rather than treated as equally spaced numeric values.

## 2. Validation and model assessment

### 2.1 Cluster-based data splitting

The data are split by survey cluster rather than by individual respondent. Twenty per cent of clusters are held out as a test set, containing 3,942 adults.

Model selection uses nested cross-validation with five outer folds and three inner folds. Clusters are kept together during these splits so that respondents from the same enumeration area do not appear in both sides of a split.

The test set is kept separate from model selection and threshold selection.

**Check:** `tests/test_pipeline_outputs.py`.

### 2.2 Survey weights

The individual survey weight supplied with the source data is used in model fitting and weighted evaluation. Unweighted results are also reported to show performance within the analysed sample.

### 2.3 Model selection metric

Models are compared using mean survey-weighted PR-AUC across the outer cross-validation folds. PR-AUC is used because the study focuses on identifying bank users, who make up about 30% of adults on a weighted basis.

A dummy classifier is included as a baseline.

## 3. Model development

### 3.1 Models compared

The project compares a dummy classifier, Logistic Regression, Random Forest and Gradient Boosting (`HistGradientBoostingClassifier`).

| Model | Mean weighted PR-AUC | Standard deviation |
|---|---:|---:|
| Dummy classifier | 0.286 | 0.006 |
| Logistic Regression | 0.694 | 0.035 |
| Random Forest | 0.703 | 0.034 |
| Gradient Boosting | **0.712** | 0.029 |

Gradient Boosting had the highest mean PR-AUC and was selected as the final model. The differences between the three machine-learning models were small relative to the variation across folds. The choice therefore reflects the best mean result under the selected metric, not a claim that Gradient Boosting is substantially or statistically proven to outperform the other two models.

Logistic Regression remains a reasonable simpler alternative. Stacking was not added because the three models performed similarly and a more complex model would make training, interpretation and deployment harder.

**Sources:** `reports/model_cv_summary.csv` and `reports/model_cv_fold_results.csv`.

### 3.2 Prediction threshold

The final model uses a common threshold of 0.320. A respondent is classified as a likely bank user when the predicted probability is at least 0.320.

This threshold was selected by maximising weighted F1 on training out-of-fold predictions. The locked test set was not used to choose it.

The test-set comparison was:

| Threshold rule | Weighted F1 | Female TPR | Male TPR | TPR gap (M − F) |
|---|---:|---:|---:|---:|
| Common 0.50 | 0.668 | 0.519 | 0.675 | +0.156 |
| Common 0.320 | 0.690 | 0.703 | 0.794 | +0.091 |
| Gender-specific: F 0.245, M 0.380 | 0.691 | 0.806 | 0.754 | −0.052 |

The common 0.320 threshold improves weighted F1 and reduces the observed TPR gap compared with 0.50. It does not remove the gap.

Gender-specific thresholds bring the observed TPRs closer together in this test set, but use gender when making the classification. They are included as a comparison, not selected for the model package. The final package uses the common threshold.

The threshold stability analysis also shows that the selected threshold varies across bootstrap samples. It should be understood as a practical operating point, not as the one uniquely correct threshold.

**Sources:** `reports/mitigation_comparison.csv`, `reports/threshold_stability.csv`.

## 4. Fairness assessment

### 4.1 Main fairness measure

The main fairness measure is the difference in True Positive Rate (TPR) between women and men. TPR measures the proportion of actual bank users correctly identified by the model.

The study also reports False Negative Rate, False Positive Rate, selection rate, precision and calibration by gender. Cluster-bootstrap confidence intervals are used when reporting gender gaps.

The focus on TPR reflects the study's interest in whether the model identifies actual bank users at similar rates among women and men.

### 4.2 Why other measures are also reported

The weighted rate of actual bank use differs between women and men in the test set. For that reason, equal selection rates are not treated as the only definition of fairness.

No single metric captures every aspect of fairness. The project therefore reports several measures and discusses the trade-offs rather than claiming that one result proves the model is fair.

### 4.3 Gender is excluded from the main model, but fairness is still checked

Gender is not included as a predictor in the main model. However, other variables may contain patterns associated with gender. Excluding gender alone does not guarantee that predictions will be free from gender-related differences.

The study checks this by comparing prediction performance and error rates for women and men.

## 5. Explainability and deployment

### 5.1 SHAP

TreeSHAP is used to examine the features that contribute to model predictions. The analysis includes overall feature importance, comparisons by gender, and selected false-positive and false-negative cases.

SHAP explains how the fitted model uses the available features. It does not show that those features cause bank use.

### 5.2 Model package and API

The fitted model, threshold, input schema and supporting metadata are saved in a model package. The package is reloaded and checked to confirm that it reproduces the original predictions.

A FastAPI service provides single and batch prediction endpoints. Gender is not part of the prediction request. The API returns prediction information and SHAP-based reason codes; these are intended to help users inspect model behaviour and should not be treated as a guarantee of legal compliance or a complete explanation of a real-world decision.

## 6. Project limitations

- The data come from one cross-sectional survey round. The results do not show how the model would perform on a later year's data.
- The model identifies predictive relationships, not causal effects.
- Gender-related patterns may still be present in the other predictors.
- Performance differs between women and men and may also differ across combined groups such as gender and location.
- The gender-specific threshold comparison is a test of a possible mitigation approach, not the threshold used in the final model.
