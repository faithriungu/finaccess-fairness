
"""
SHAP analysis for the FinAccess traditional bank-use model.

Run from the project root:
    python -m src.explain
By default, false negatives/positives are defined at the deployed decision
threshold (common_threshold_weighted_f1 in reports/decision_thresholds.json),
so the error analysis matches what the model package and API decide.
Override with:
    python -m src.explain --threshold 0.50

Analyses:
1. Global SHAP importance across the 35 original predictors.
2. SHAP magnitude and direction comparison by gender.
3. SHAP analysis of false negatives and false positives.

Uses the saved model and locked test set. Does not retrain the model.
"""

import argparse
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import (
    CLEAN_FILE,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    TARGET,
    GENDER,
    WEIGHT,
    CLUSTER,
    RESPONDENT_ID,
)


# ---------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=("Threshold used to identify false negatives and false positives. "
              "Default: the deployed threshold from reports/decision_thresholds.json.")
    )

    return parser.parse_args()


# ---------------------------------------------------------
# 2. LOAD MODEL AND LOCKED TEST DATA
# ---------------------------------------------------------

def load_test_data(artifact):

    split_path = MODELS_DIR / "split.csv"

    if not CLEAN_FILE.exists():
        raise FileNotFoundError(
            f"Clean data not found: {CLEAN_FILE}"
        )

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}"
        )

    features = artifact["features"]

    df = pd.read_csv(CLEAN_FILE, low_memory=False)

    split = pd.read_csv(
        split_path,
        dtype={RESPONDENT_ID: str}
    )

    df[RESPONDENT_ID] = df[RESPONDENT_ID].astype(str)

    df = df.merge(
        split,
        on=RESPONDENT_ID,
        how="inner",
        validate="one_to_one"
    )

    test = df[df["split"] == "test"].copy()

    if test.empty:
        raise ValueError("No test observations found.")

    required = features + [
        TARGET,
        GENDER,
        WEIGHT,
        CLUSTER,
        RESPONDENT_ID
    ]

    missing = sorted(set(required) - set(test.columns))

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    X_test = test[features].copy()

    return test, X_test


# ---------------------------------------------------------
# 3. MAP TRANSFORMED FEATURES TO ORIGINAL FEATURES
# ---------------------------------------------------------


def map_features(preprocessor, original_features):
    """
    Map actual transformed columns back to their original predictors.

    This handles:
    - One-hot encoded categorical variables.
    - Grouped infrequent categories.
    - Numeric imputation indicators.
    - Features that produce no transformed columns.
    """

    transformed_names = np.asarray(
        preprocessor.get_feature_names_out(),
        dtype=object
    )

    mapping = np.empty(len(transformed_names), dtype=object)

    for i, name in enumerate(transformed_names):
        name = str(name)

        # Numeric missing-value indicators
        if name.startswith("missingindicator_"):
            source = name.replace("missingindicator_", "", 1)

            if source in original_features:
                mapping[i] = source
                continue

        # Exact matches, including ordinary numeric features
        if name in original_features:
            mapping[i] = name
            continue

        # One-hot encoded columns, including grouped infrequent categories.
        # Longest-prefix matching avoids confusion between similarly
        # named original predictors.
        matches = [
            feature for feature in original_features
            if name.startswith(str(feature) + "_")
        ]

        if matches:
            mapping[i] = max(matches, key=len)
        else:
            raise ValueError(
                f"Cannot map transformed column '{name}' "
                f"to an original predictor. "
                f"Check the fitted preprocessing pipeline."
            )

    # Report predictors that produced no transformed columns.
    missing_features = [
        feature for feature in original_features
        if not np.any(mapping == feature)
    ]

    if missing_features:
        print(
            "\nWARNING: These original predictors produced no "
            "transformed columns:"
        )
        for feature in missing_features:
            print(f"  - {feature}")

    return transformed_names, mapping



# ---------------------------------------------------------
# 4. NORMALISE SHAP OUTPUT
# ---------------------------------------------------------

def get_positive_class_shap(values):

    if isinstance(values, list):

        values = values[1] if len(values) == 2 else values[0]

    values = np.asarray(values)

    if values.ndim == 2:
        return values

    if values.ndim == 3:

        if values.shape[-1] == 2:
            return values[:, :, 1]

        if values.shape[0] == 2:
            return values[1, :, :]

    raise ValueError(
        f"Unexpected SHAP output shape: {values.shape}"
    )


# ---------------------------------------------------------
# 5. AGGREGATE ENCODED FEATURES INTO ORIGINAL PREDICTORS
# ---------------------------------------------------------

def aggregate_shap(shap_values, original_features, mapping):
    """
    Sum encoded-column SHAP values back to their original predictors.

    If a predictor produced no transformed columns during training,
    its SHAP contribution is zero because the fitted model has no
    corresponding input column.
    """

    result = np.zeros(
        (shap_values.shape[0], len(original_features))
    )

    missing_features = []

    for j, feature in enumerate(original_features):

        positions = np.where(mapping == feature)[0]

        if len(positions) == 0:
            missing_features.append(feature)
            continue

        result[:, j] = shap_values[:, positions].sum(axis=1)

    if missing_features:
        print(
            "\nPredictors with zero SHAP contribution "
            "because they have no transformed columns:"
        )
        for feature in missing_features:
            print(f"  - {feature}")

    return result



# ---------------------------------------------------------
# 6. WEIGHTED AVERAGE
# ---------------------------------------------------------

def weighted_mean(values, weights):

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if not valid.any():
        return np.nan

    return np.average(
        values[valid],
        weights=weights[valid]
    )


# ---------------------------------------------------------
# 7. ANALYSIS 1: GLOBAL SHAP IMPORTANCE
# ---------------------------------------------------------

def global_shap_analysis(
    shap_values,
    features,
    weights
):

    rows = []

    for j, feature in enumerate(features):

        values = shap_values[:, j]

        rows.append({

            "feature": feature,

            "mean_abs_shap_unweighted":
                np.mean(np.abs(values)),

            "mean_abs_shap_weighted":
                weighted_mean(
                    np.abs(values),
                    weights
                ),

            "mean_signed_shap_unweighted":
                np.mean(values),

            "mean_signed_shap_weighted":
                weighted_mean(
                    values,
                    weights
                )
        })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "mean_abs_shap_weighted",
        ascending=False
    )

    result["rank"] = range(1, len(result) + 1)

    result.to_csv(
        REPORTS_DIR / "shap_global_importance.csv",
        index=False
    )

    return result


# ---------------------------------------------------------
# 8. ANALYSIS 2: GENDER COMPARISON
# ---------------------------------------------------------

def gender_shap_analysis(
    shap_values,
    features,
    gender,
    weights
):

    rows = []

    gender = np.asarray(gender)

    for sex in ["Female", "Male"]:

        mask = gender == sex

        if not mask.any():
            warnings.warn(
                f"No observations found for {sex}"
            )
            continue

        for j, feature in enumerate(features):

            values = shap_values[mask, j]

            group_weights = weights[mask]

            rows.append({

                "gender": sex,

                "feature": feature,

                "n": mask.sum(),

                "mean_abs_shap_unweighted":
                    np.mean(np.abs(values)),

                "mean_abs_shap_weighted":
                    weighted_mean(
                        np.abs(values),
                        group_weights
                    ),

                "mean_signed_shap_unweighted":
                    np.mean(values),

                "mean_signed_shap_weighted":
                    weighted_mean(
                        values,
                        group_weights
                    )
            })

    result = pd.DataFrame(rows)

    result.to_csv(
        REPORTS_DIR / "shap_gender_comparison.csv",
        index=False
    )

    return result


# ---------------------------------------------------------
# 9. ANALYSIS 3: FALSE NEGATIVES AND FALSE POSITIVES
# ---------------------------------------------------------

def error_shap_analysis(
    shap_values,
    features,
    y_true,
    y_pred,
    weights
):

    false_negative = (
        (y_true == 1)
        & (y_pred == 0)
    )

    false_positive = (
        (y_true == 0)
        & (y_pred == 1)
    )

    rows = []

    for error_name, mask in [
        ("False negative", false_negative),
        ("False positive", false_positive)
    ]:

        n_errors = int(mask.sum())

        if n_errors == 0:
            warnings.warn(
                f"No {error_name} observations found."
            )
            continue

        for j, feature in enumerate(features):

            values = shap_values[mask, j]

            group_weights = weights[mask]

            rows.append({

                "error_type": error_name,

                "n_errors": n_errors,

                "feature": feature,

                "mean_abs_shap_unweighted":
                    np.mean(np.abs(values)),

                "mean_abs_shap_weighted":
                    weighted_mean(
                        np.abs(values),
                        group_weights
                    ),

                "mean_signed_shap_unweighted":
                    np.mean(values),

                "mean_signed_shap_weighted":
                    weighted_mean(
                        values,
                        group_weights
                    )
            })

    result = pd.DataFrame(rows)

    result.to_csv(
        REPORTS_DIR / "shap_error_analysis.csv",
        index=False
    )

    return result, false_negative, false_positive


# ---------------------------------------------------------
# 10. SAVE VISUALISATIONS
# ---------------------------------------------------------

def save_plots(
    global_results,
    gender_results,
    error_results
):

    # Global SHAP importance

    top = global_results.head(20).sort_values(
        "mean_abs_shap_weighted"
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(
        top["feature"],
        top["mean_abs_shap_weighted"]
    )

    ax.set_xlabel(
        "Survey-weighted mean absolute SHAP value"
    )

    ax.set_title(
        "Global SHAP Importance: Traditional Bank Usage"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "shap_global_importance.png",
        dpi=300
    )

    plt.close(fig)

    # Gender comparison

    top_features = global_results.head(15)["feature"]

    gender_plot = gender_results[
        gender_results["feature"].isin(top_features)
    ]

    gender_plot = gender_plot.pivot(
        index="feature",
        columns="gender",
        values="mean_abs_shap_weighted"
    )

    gender_plot = gender_plot.reindex(
        top_features.tolist()
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    gender_plot.plot(
        kind="barh",
        ax=ax
    )

    ax.set_xlabel(
        "Survey-weighted mean absolute SHAP value"
    )

    ax.set_title(
        "SHAP Contribution Magnitude by Gender"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "shap_gender_magnitude.png",
        dpi=300
    )

    plt.close(fig)

    # False negatives and false positives

    if error_results.empty:
        return

    for error_name, filename in [
        ("False negative", "shap_false_negatives.png"),
        ("False positive", "shap_false_positives.png")
    ]:

        subset = error_results[
            error_results["error_type"] == error_name
        ]

        if subset.empty:
            continue

        subset = subset.nlargest(
            15,
            "mean_abs_shap_weighted"
        )

        subset = subset.sort_values(
            "mean_signed_shap_weighted"
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.barh(
            subset["feature"],
            subset["mean_signed_shap_weighted"]
        )

        ax.axvline(0, linewidth=1)

        ax.set_xlabel(
            "Mean signed SHAP value"
        )

        ax.set_title(
            f"{error_name}: Strongest SHAP Contributions"
        )

        fig.tight_layout()

        fig.savefig(
            FIGURES_DIR / filename,
            dpi=300
        )

        plt.close(fig)


# ---------------------------------------------------------
# 11. MAIN EXECUTION
# ---------------------------------------------------------

def main():

    args = parse_args()

    if args.threshold is None:
        thresholds_path = REPORTS_DIR / "decision_thresholds.json"
        if not thresholds_path.exists():
            raise FileNotFoundError(
                f"{thresholds_path} not found. Run python -m src.mitigate first, "
                "or pass --threshold explicitly."
            )
        thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
        args.threshold = round(float(thresholds["common_threshold_weighted_f1"]), 3)

    if not 0 < args.threshold < 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = MODELS_DIR / "bank_model.joblib"

    artifact = joblib.load(model_path)

    model = artifact["model"]

    features = list(artifact["features"])

    print("Loading locked test data...")

    test, X_test = load_test_data(artifact)

    y_true = test[TARGET].to_numpy(dtype=int)

    weights = test[WEIGHT].to_numpy(dtype=float)

    gender = test[GENDER].astype(str).to_numpy()

    # Confirm expected pipeline structure.

    if not {"preprocess", "model"}.issubset(
        set(model.named_steps)
    ):
        raise TypeError(
            "Expected pipeline steps named 'preprocess' and 'model'."
        )

    preprocessor = model.named_steps["preprocess"]

    tree_model = model.named_steps["model"]

    # Generate probabilities using the complete saved pipeline.

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= args.threshold
    ).astype(int)

    # Transform data using the already fitted preprocessing pipeline.

    print("Transforming test predictors...")

    X_transformed = preprocessor.transform(X_test)

    # Identify which transformed columns belong to each original predictor.

    _, mapping = map_features(
        preprocessor,
        features
    )

    # TreeSHAP.

    print("Calculating SHAP values...")

    explainer = shap.TreeExplainer(
        tree_model,
        feature_perturbation="tree_path_dependent",
        model_output="raw"
    )

    shap_raw = explainer.shap_values(
        X_transformed,
        check_additivity=False
    )

    shap_transformed = get_positive_class_shap(
        shap_raw
    )

    # Aggregate encoded variables back to the 35 original predictors.

    shap_original = aggregate_shap(
        shap_transformed,
        features,
        mapping
    )

    # -----------------------------------------------------
    # ANALYSIS 1
    # -----------------------------------------------------

    print("\nRunning global SHAP analysis...")

    global_results = global_shap_analysis(
        shap_original,
        features,
        weights
    )

    print("\nTop 15 global predictors:")

    print(
        global_results[
            [
                "rank",
                "feature",
                "mean_abs_shap_weighted",
                "mean_signed_shap_weighted"
            ]
        ].head(15).round(5).to_string(index=False)
    )

    # -----------------------------------------------------
    # ANALYSIS 2
    # -----------------------------------------------------

    print("\nRunning gender SHAP comparison...")

    gender_results = gender_shap_analysis(
        shap_original,
        features,
        gender,
        weights
    )

    # -----------------------------------------------------
    # ANALYSIS 3
    # -----------------------------------------------------

    print("\nRunning error SHAP analysis...")

    error_results, false_negative, false_positive = (
        error_shap_analysis(
            shap_original,
            features,
            y_true,
            predictions,
            weights
        )
    )

    # Save respondent-level error information.

    error_cases = test[
        [
            RESPONDENT_ID,
            GENDER,
            WEIGHT,
            CLUSTER
        ]
    ].copy()

    error_cases["actual"] = y_true

    error_cases["predicted"] = predictions

    error_cases["predicted_probability"] = probabilities

    error_cases["threshold"] = args.threshold

    error_cases["error_type"] = np.select(
        [
            false_negative,
            false_positive
        ],
        [
            "False negative",
            "False positive"
        ],
        default="Correct prediction"
    )

    error_cases.to_csv(
        REPORTS_DIR / "shap_test_error_cases.csv",
        index=False
    )

    # Save counts.

    counts = pd.DataFrame([{

        "threshold": args.threshold,

        "test_observations": len(test),

        "false_negatives": int(false_negative.sum()),

        "false_positives": int(false_positive.sum()),

        "true_positives": int(
            ((y_true == 1) & (predictions == 1)).sum()
        ),

        "true_negatives": int(
            ((y_true == 0) & (predictions == 0)).sum()
        )

    }])

    counts.to_csv(
        REPORTS_DIR / "shap_error_counts.csv",
        index=False
    )

    # -----------------------------------------------------
    # VISUALISATIONS
    # -----------------------------------------------------

    print("\nCreating SHAP plots...")

    save_plots(
        global_results,
        gender_results,
        error_results
    )

    print("\nSHAP ANALYSIS COMPLETED")

    print(f"Test observations: {len(test):,}")

    print(f"Predictors explained: {len(features)}")

    print(f"Decision threshold: {args.threshold:.3f}")

    print("\nError counts:")

    print(counts.to_string(index=False))

    print("\nResults saved in reports/.")

    print("1. shap_global_importance.csv")

    print("2. shap_gender_comparison.csv")

    print("3. shap_error_analysis.csv")

    print("4. shap_test_error_cases.csv")

    print("5. shap_error_counts.csv")

    print("6. shap_global_importance.png")

    print("7. shap_gender_magnitude.png")

    print("8. shap_false_negatives.png")

    print("9. shap_false_positives.png")


if __name__ == "__main__":
    main()
