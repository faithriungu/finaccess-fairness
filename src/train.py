"""Group-aware model comparison and final training.

Run from project root: python -m src.train
The final test clusters are locked and are not used for model selection.
"""
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from src.config import (
    CLEAN_FILE, FEATURES_FILE, MODELS_DIR, REPORTS_DIR, TRAINED_MODEL_FILE,
    TARGET, GENDER, WEIGHT, CLUSTER, RESPONDENT_ID,
    RANDOM_STATE, TEST_FRACTION, OUTER_FOLDS, INNER_FOLDS, N_JOBS,
)

INCOME = ["income_last_month", "income_highest_month", "income_lowest_month"]


def make_pipeline(model, numeric_cols, categorical_cols, scale_numeric=False):
    """All imputation, transforms and encoding are fitted inside each CV fold."""
    income_cols = [c for c in numeric_cols if c in INCOME]
    other_numeric = [c for c in numeric_cols if c not in INCOME]

    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    income_steps = [
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
        income_steps.append(("scale", StandardScaler()))

    categorical_steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=25,
            sparse_output=False,
        )),
    ]

    transformers = []
    if other_numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), other_numeric))
    if income_cols:
        transformers.append(("income", Pipeline(income_steps), income_cols))
    if categorical_cols:
        transformers.append(("categorical", Pipeline(categorical_steps), categorical_cols))

    preprocessor = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def model_spaces(numeric_cols, categorical_cols):
    """Small, defensible search spaces; every inner split is cluster-aware."""
    return {
        # Rung 0: predicts the weighted base rate for everyone. Its PR-AUC is
        # the chance baseline every real model must beat.
        "Dummy (prior)": (
            make_pipeline(DummyClassifier(strategy="prior"), numeric_cols, categorical_cols),
            {},
            1,
        ),
        "Logistic regression": (
            make_pipeline(LogisticRegression(max_iter=3000, solver="lbfgs"), numeric_cols, categorical_cols, True),
            {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
            5,
        ),
        "Random forest": (
            make_pipeline(RandomForestClassifier(
                n_estimators=250, random_state=RANDOM_STATE, n_jobs=1,
                class_weight=None,
            ), numeric_cols, categorical_cols),
            {
                "model__max_depth": [None, 12, 20],
                "model__min_samples_leaf": [5, 10, 20],
                "model__max_features": ["sqrt", 0.5, 0.8],
            },
            5,
        ),
        "Gradient boosting": (
            make_pipeline(HistGradientBoostingClassifier(
                max_iter=200, random_state=RANDOM_STATE,
            ), numeric_cols, categorical_cols),
            {
                "model__learning_rate": [0.03, 0.06, 0.1],
                "model__max_leaf_nodes": [15, 31],
                "model__l2_regularization": [0.0, 0.5, 1.0],
                "model__min_samples_leaf": [20, 50],
            },
            5,
        ),
    }


def weighted_group_search(pipeline, param_space, n_iter, X, y, groups, weights, n_splits, seed):
    """Manual randomized search so inner validation PR-AUC uses survey weights.

    All preprocessing is inside the estimator pipeline. Each validation fold is
    cluster-disjoint from its corresponding fit fold.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if not param_space:
        candidates = [{}]  # nothing to tune (dummy baseline)
    else:
        candidates = list(ParameterSampler(param_space, n_iter=n_iter, random_state=seed))
    best_score = -np.inf
    best_params = None

    for candidate_no, params in enumerate(candidates, start=1):
        fold_scores = []
        for fit_idx, val_idx in cv.split(X, y, groups=groups):
            estimator = clone(pipeline).set_params(**params)
            fit_w = np.asarray(weights[fit_idx], dtype=float)
            fit_w = fit_w / fit_w.mean()
            estimator.fit(
                X.iloc[fit_idx], y.iloc[fit_idx],
                model__sample_weight=fit_w,
            )
            val_prob = estimator.predict_proba(X.iloc[val_idx])[:, 1]
            score = average_precision_score(
                y.iloc[val_idx], val_prob,
                sample_weight=np.asarray(weights[val_idx], dtype=float),
            )
            fold_scores.append(score)
        candidate_score = float(np.mean(fold_scores))
        if candidate_score > best_score:
            best_score = candidate_score
            best_params = params
        print(f"      candidate {candidate_no}/{len(candidates)} weighted inner PR-AUC={candidate_score:.4f}")

    best_estimator = clone(pipeline).set_params(**best_params)
    fit_w = np.asarray(weights, dtype=float)
    fit_w = fit_w / fit_w.mean()
    best_estimator.fit(X, y, model__sample_weight=fit_w)
    return best_estimator, best_params, best_score


def evaluate_fold(y, probability, weight):
    return {
        "pr_auc_unweighted": average_precision_score(y, probability),
        "pr_auc_weighted": average_precision_score(y, probability, sample_weight=weight),
        "roc_auc_unweighted": roc_auc_score(y, probability),
        "roc_auc_weighted": roc_auc_score(y, probability, sample_weight=weight),
        "brier_unweighted": brier_score_loss(y, probability),
        "brier_weighted": brier_score_loss(y, probability, sample_weight=weight),
    }


def main():
    if not CLEAN_FILE.exists():
        raise FileNotFoundError("Run python -m src.clean before training.")

    df = pd.read_csv(CLEAN_FILE, low_memory=False)
    manifest = json.loads(FEATURES_FILE.read_text(encoding="utf-8"))
    features = manifest["final_features"]
    required = features + [TARGET, GENDER, WEIGHT, CLUSTER, RESPONDENT_ID]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Columns missing from model-ready data: {missing}")

    forbidden = {GENDER, "hh_head_gender", WEIGHT, CLUSTER, RESPONDENT_ID,
                 TARGET, "target_formal_access", "mobile_money_access"}
    if set(features) & forbidden:
        raise ValueError(f"Forbidden predictors found: {set(features) & forbidden}")

    numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in features if c not in numeric_cols]
    X = df[features].copy()
    for col in categorical_cols:
        X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
    y = df[TARGET].astype(int)
    groups = df[CLUSTER].to_numpy()
    weights = df[WEIGHT].astype(float).to_numpy()

    # Reserve entire clusters for the final test set.
    holdout = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(holdout.split(X, y, groups=groups))
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([
        train[[RESPONDENT_ID]].assign(split="train"),
        test[[RESPONDENT_ID]].assign(split="test"),
    ]).to_csv(MODELS_DIR / "split.csv", index=False)

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    g_train = groups[train_idx]
    w_train = weights[train_idx]

    print(f"Adults: {len(df):,} | Train: {len(train):,} | Locked test: {len(test):,}")
    print(f"Clusters: {df[CLUSTER].nunique():,} | Features: {len(features)}")
    print(f"Weighted bank-use prevalence (full adult data): {np.average(y, weights=weights):.3f}")
    print("Sensitive attribute gender is not included in X.")

    spaces = model_spaces(numeric_cols, categorical_cols)
    outer_cv = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    oof_rows = []

    for model_name, (pipeline, params, n_iter) in spaces.items():
        print(f"\nNested group CV: {model_name}")
        for fold, (fit_idx, val_idx) in enumerate(
            outer_cv.split(X_train, y_train, groups=g_train), start=1
        ):
            estimator, best_params, inner_score = weighted_group_search(
                pipeline, params, n_iter,
                X_train.iloc[fit_idx], y_train.iloc[fit_idx],
                g_train[fit_idx], w_train[fit_idx],
                n_splits=INNER_FOLDS, seed=RANDOM_STATE + fold,
            )
            prob = estimator.predict_proba(X_train.iloc[val_idx])[:, 1]
            val_weights = w_train[val_idx]
            metrics = evaluate_fold(y_train.iloc[val_idx], prob, val_weights)
            fold_rows.append({
                "model": model_name, "fold": fold,
                "best_inner_pr_auc_weighted": inner_score,
                **metrics,
                "best_params": json.dumps(best_params, default=str),
            })
            for j, p in zip(val_idx, prob):
                row = train.iloc[j]
                oof_rows.append({
                    "model": model_name,
                    RESPONDENT_ID: row[RESPONDENT_ID],
                    "y": int(row[TARGET]), "prob": float(p),
                    "weight": float(row[WEIGHT]),
                    GENDER: row[GENDER],
                    "location_type": row["location_type"],
                    CLUSTER: row[CLUSTER],
                    "fold": fold,
                })
            print(
                f"  fold {fold}/{OUTER_FOLDS}: weighted PR-AUC="
                f"{metrics['pr_auc_weighted']:.3f}; unweighted PR-AUC="
                f"{metrics['pr_auc_unweighted']:.3f}"
            )

    fold_results = pd.DataFrame(fold_rows)
    fold_results.to_csv(REPORTS_DIR / "model_cv_fold_results.csv", index=False)
    oof = pd.DataFrame(oof_rows)
    oof.to_csv(MODELS_DIR / "oof_train_predictions.csv", index=False)

    summary = fold_results.groupby("model").agg(
        pr_auc_weighted_mean=("pr_auc_weighted", "mean"),
        pr_auc_weighted_sd=("pr_auc_weighted", "std"),
        pr_auc_unweighted_mean=("pr_auc_unweighted", "mean"),
        pr_auc_unweighted_sd=("pr_auc_unweighted", "std"),
        roc_auc_weighted_mean=("roc_auc_weighted", "mean"),
        roc_auc_unweighted_mean=("roc_auc_unweighted", "mean"),
        brier_weighted_mean=("brier_weighted", "mean"),
        brier_unweighted_mean=("brier_unweighted", "mean"),
    ).sort_values("pr_auc_weighted_mean", ascending=False)
    # Model ladder: how far each model sits above the chance baseline, and
    # whether its lead over the next model exceeds the fold-to-fold spread.
    dummy_pr = summary.loc["Dummy (prior)", "pr_auc_weighted_mean"]
    summary["gain_over_dummy"] = summary["pr_auc_weighted_mean"] - dummy_pr
    summary["lead_over_next"] = summary["pr_auc_weighted_mean"].diff(-1)
    summary["lead_exceeds_1sd"] = summary["lead_over_next"] > summary["pr_auc_weighted_sd"]
    summary.to_csv(REPORTS_DIR / "model_cv_summary.csv")
    print("\nNested group-CV summary:")
    print(summary.round(4).to_string())

    champion = summary.drop(index="Dummy (prior)").index[0]
    print(f"\nSelected champion by mean weighted PR-AUC: {champion}")

    # Tune the selected model on the complete training partition using weighted,
    # cluster-aware inner CV. The locked test clusters are not involved.
    pipeline, params, n_iter = spaces[champion]
    final_model, final_params, final_inner_score = weighted_group_search(
        pipeline, params, n_iter,
        X.iloc[train_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        groups[train_idx], weights[train_idx],
        n_splits=5, seed=RANDOM_STATE + 100,
    )

    artifact = {
        "model": final_model,
        "features": features,
        "target": TARGET,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "metadata": {
            "model_name": champion,
            "best_params": final_params,
            "final_inner_weighted_pr_auc": final_inner_score,
            "selected_by": "highest mean weighted PR-AUC across nested cluster-aware outer folds",
            "test_set_locked": True,
            "n_adults": int(len(df)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_features": int(len(features)),
            "training_weights": "survey_weight normalised to mean 1 within each fit fold",
            "inner_search_metric": "survey-weighted average precision",
        },
    }
    joblib.dump(artifact, TRAINED_MODEL_FILE)
    print(f"Saved final model: {TRAINED_MODEL_FILE}")
    print("Saved split, nested-CV results and OOF training predictions.")


if __name__ == "__main__":
    main()
