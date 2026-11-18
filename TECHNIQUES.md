# Techniques used in the project

This project uses machine learning to predict traditional bank use among Kenyan adults and to examine whether prediction performance differs between women and men.

The data are survey-based, with numeric and categorical variables, missing responses, unequal survey weights and respondents grouped in survey clusters. The methods were selected to suit these features of the data and the project's research objective.

## 1. Models compared

Four models were used:

- **Dummy classifier:** provides a simple baseline. It shows how the model performs compared with a prediction that does not learn relationships from the respondent characteristics.
- **Logistic Regression:** provides a familiar statistical baseline and is relatively easy to interpret.
- **Random Forest:** can capture non-linear relationships and interactions between variables.
- **Gradient Boosting:** builds trees in sequence, with each stage improving on earlier predictions. It performed best on the project's main model-selection measure.

Gradient Boosting was selected because it had the highest mean survey-weighted PR-AUC across the outer cross-validation folds. Its lead over Random Forest and Logistic Regression was small compared with the variation across folds, so the result is not presented as a large or statistically established advantage.

## 2. Validation and preprocessing

**Cluster-based nested cross-validation** was used to compare models and tune them. Respondents from the same survey cluster were kept together during splitting. This reduces the risk of the model being tested on respondents from clusters it has already seen during training.

A separate cluster-held-out test set was reserved for final evaluation. It was not used to choose the model or its decision threshold.

Preprocessing was placed inside the modelling pipeline. This means that imputers and encoders are fitted using the training portion of each split, rather than using information from the validation or test data.

Survey weights were used in model fitting and weighted evaluation. Unweighted results were also retained to show how the model performed on the analysed sample.

## 3. Performance measures

Weighted PR-AUC was the main measure used to rank models. It focuses on the model's ability to identify the positive class, which in this study is adults who use traditional banks.

Other measures were reported to give a fuller picture:

- **ROC-AUC:** how well the model ranks bank users above non-users across possible thresholds.
- **Precision:** how many respondents classified as bank users are actual bank users.
- **Recall / TPR:** how many actual bank users the model correctly identifies.
- **F1-score:** combines precision and recall.
- **Brier score and calibration results:** assess how close predicted probabilities are to observed outcomes.

The dummy classifier provides a baseline for interpreting the PR-AUC results.

## 4. Fairness assessment

The main fairness comparison is the difference in recall (TPR) between women and men. This asks whether the model identifies actual bank users at similar rates in the two groups.

The analysis also reports false negative rates, false positive rates, selection rates, precision and calibration by gender. Cluster-bootstrap confidence intervals are used to show uncertainty around group differences.

The project does not rely on a single fairness measure. Actual bank-use rates differ between women and men, so equal selection rates alone would not describe whether the model identifies actual bank users equally well. The different measures are reported together to make the trade-offs visible.

Gender is not included as a predictor in the main model. This does not guarantee that the model is free from gender-related patterns, because other respondent characteristics may be associated with gender. The group-based evaluation is therefore an important part of the study.

## 5. Threshold comparison

The model produces probabilities. A threshold is needed to convert those probabilities into a classification of bank user or non-user.

Three approaches were compared:

1. A common threshold of 0.50.
2. A common threshold of 0.320, selected using training out-of-fold predictions.
3. Separate thresholds for women and men, used as a fairness comparison.

The common 0.320 threshold improved weighted F1 and reduced the observed TPR gap compared with 0.50. Gender-specific thresholds brought the group TPRs closer in the test results, but they require gender to be used at decision time. The final model package therefore uses the common 0.320 threshold.

Threshold adjustment changes the final classification rule; it does not change the model's underlying predicted probabilities.

## 6. Explainability

TreeSHAP was used because the selected model is tree-based. It provides feature contributions for individual predictions and summary measures across groups.

The project uses SHAP to examine overall feature importance, compare patterns for women and men, and inspect selected false-positive and false-negative predictions.

These explanations describe how the model arrived at its predictions. They should not be interpreted as evidence that a feature causes traditional bank use.

## 7. Techniques not used

A number of other methods were considered but were not needed for the main project.

- **Stacking:** the three machine-learning models performed relatively similarly. Combining them would add complexity without a clear expected improvement.
- **Deep learning:** the dataset is structured survey data with a moderate sample size. A neural network would require additional tuning and would make interpretation more difficult.
- **XGBoost and LightGBM:** another boosting library was not necessary to answer the research question. The selected scikit-learn model provided a suitable comparison and fitted the existing pipeline.
- **SVM and k-nearest neighbours:** these were not prioritised because the project needs useful probability estimates and contains many encoded categorical variables.
- **SMOTE:** the positive class is not so rare that synthetic observations are an obvious requirement. Synthetic records would also be difficult to reconcile with the survey weights. Threshold adjustment was tested instead.
- **Fairness-constrained training:** the project uses post-processing thresholds to compare the performance-fairness trade-off without retraining the model under a fairness constraint. In-processing methods could be examined in future work.
- **Unsupervised learning:** the study has a defined outcome to predict, so supervised classification is the more direct approach.
- **Drift monitoring and experiment-management platforms:** the dataset contains one survey round. These tools may be useful if the model is maintained over time, but they were not needed to answer the current research question.

## 8. Summary

The project uses established classification methods, cluster-aware validation, survey-weighted evaluation, gender-disaggregated fairness measures, threshold comparison and SHAP explanations.

The emphasis is on a model that can be evaluated carefully and explained clearly, rather than using the largest possible number of algorithms.
