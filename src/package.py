"""Build the deployable model package and check it reloads identically.

Run after: python -m src.mitigate
Run:       python -m src.package

The package bundles everything needed to score a new respondent:
  - the fitted pipeline (preprocessing + gradient boosting),
  - the decision threshold (common, gender-blind, chosen on training OOF),
  - the input schema (feature names, types, allowed categories, numeric ranges),
  - metadata (model version, git commit, data hashes, library versions,
    validation metrics).

It is saved, reloaded and checked: the reloaded package must reproduce the
in-memory predictions exactly, or the step fails.
"""
import json
import platform
import subprocess
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config import (
    ROOT, CLEAN_FILE, SELECTED_FILE, DATA_MANIFEST, MODELS_DIR, REPORTS_DIR, SAMPLE_DIR,
    TRAINED_MODEL_FILE, PACKAGE_FILE, MODEL_VERSION, RESPONDENT_ID,
)
from src.ingest import sha256_file

SAMPLE_INPUT = SAMPLE_DIR / "sample_input.csv"
SAMPLE_EXPECTED = SAMPLE_DIR / "sample_expected_predictions.csv"
N_SAMPLE = 25


def git_commit():
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown (not a git repository)"


def build_schema(train_frame, numeric, categorical):
    """Describe valid inputs from the TRAINING partition only."""
    schema = {"numeric": {}, "categorical": {}}
    for col in numeric:
        s = pd.to_numeric(train_frame[col], errors="coerce")
        schema["numeric"][col] = {
            "min": float(s.min()), "max": float(s.max()),
            "nullable": True,
        }
    for col in categorical:
        levels = sorted(train_frame[col].dropna().astype(str).unique().tolist())
        schema["categorical"][col] = {"allowed": levels, "nullable": True}
    return schema


def prepare(frame, features, categorical):
    """Same column handling as training: categorical NaN as np.nan objects."""
    x = frame[features].copy()
    for col in categorical:
        x[col] = x[col].astype(object).where(x[col].notna(), np.nan)
    return x


def main():
    for path in [TRAINED_MODEL_FILE, REPORTS_DIR / "decision_thresholds.json",
                 REPORTS_DIR / "test_overall_metrics.csv", CLEAN_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}. Run the earlier steps first.")

    trained = joblib.load(TRAINED_MODEL_FILE)
    thresholds = json.loads((REPORTS_DIR / "decision_thresholds.json").read_text(encoding="utf-8"))
    features = trained["features"]
    numeric = trained["numeric_features"]
    categorical = trained["categorical_features"]

    df = pd.read_csv(CLEAN_FILE, low_memory=False)
    df[RESPONDENT_ID] = df[RESPONDENT_ID].astype(str)
    split = pd.read_csv(MODELS_DIR / "split.csv", dtype={RESPONDENT_ID: str})
    df = df.merge(split, on=RESPONDENT_ID, how="inner", validate="one_to_one")
    train_frame = df[df["split"] == "train"]
    test_frame = df[df["split"] == "test"]

    test_metrics = pd.read_csv(REPORTS_DIR / "test_overall_metrics.csv", index_col=0)
    weighted = test_metrics.loc["Survey weighted"]
    mitigation = pd.read_csv(REPORTS_DIR / "mitigation_comparison.csv", index_col=0)
    at_threshold = mitigation.loc["Common threshold (OOF weighted F1)"]

    manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8")) if DATA_MANIFEST.exists() else {}
    package = {
        "model": trained["model"],
        "features": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "threshold": round(float(thresholds["common_threshold_weighted_f1"]), 3),
        "threshold_rule": (
            "Common (gender-blind) threshold maximising survey-weighted F1 on "
            "training out-of-fold predictions. Gender-specific thresholds were "
            "evaluated as a comparison only and are not used for decisions."
        ),
        "input_schema": build_schema(train_frame, numeric, categorical),
        "output_schema": {
            "probability": "float in [0, 1]: predicted probability of traditional bank use",
            "decision": "int, 1 if probability >= threshold else 0",
        },
        "metadata": {
            "model_version": MODEL_VERSION,
            "model_name": trained["metadata"]["model_name"],
            "best_params": trained["metadata"]["best_params"],
            "target": trained["target"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "selected_data_sha256": manifest.get("sha256", sha256_file(SELECTED_FILE)),
            "model_ready_data_sha256": sha256_file(CLEAN_FILE),
            "n_train": trained["metadata"]["n_train"],
            "n_test": trained["metadata"]["n_test"],
            "test_metrics_weighted_at_0_50": {
                k: float(weighted[k]) for k in ["PR_AUC", "ROC_AUC", "Brier", "F1"]
            },
            "test_metrics_weighted_at_deployed_threshold": {
                "precision": float(at_threshold["precision_weighted"]),
                "recall": float(at_threshold["recall_weighted"]),
                "F1": float(at_threshold["F1_weighted"]),
                "TPR_gap_male_minus_female": float(at_threshold["TPR_gap_M_minus_F_weighted"]),
            },
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sensitive_attribute_used_as_input": False,
        },
    }

    # Score a small fixed sample before saving, so the reload check has a reference.
    sample = test_frame.sort_values(RESPONDENT_ID).head(N_SAMPLE)
    sample_x = prepare(sample, features, categorical)
    before_full = package["model"].predict_proba(prepare(test_frame, features, categorical))[:, 1]
    before_sample = package["model"].predict_proba(sample_x)[:, 1]

    joblib.dump(package, PACKAGE_FILE)
    reloaded = joblib.load(PACKAGE_FILE)
    after_full = reloaded["model"].predict_proba(prepare(test_frame, features, categorical))[:, 1]
    if not np.array_equal(before_full, after_full):
        raise AssertionError("Reloaded package does not reproduce the saved predictions.")

    # Sample input (features only - no ID, gender, weight or target) for predict.py,
    # tests and API examples, plus the expected output.
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    sample_x.to_csv(SAMPLE_INPUT, index=False)
    pd.DataFrame({
        "probability": before_sample,
        "decision": (before_sample >= package["threshold"]).astype(int),
    }).to_csv(SAMPLE_EXPECTED, index=False)

    print("PACKAGE COMPLETED")
    print(f"Saved: {PACKAGE_FILE}")
    print(f"Model version: {MODEL_VERSION} | git commit: {package['metadata']['git_commit']}")
    print(f"Decision threshold: {package['threshold']:.3f}")
    print(f"Reload check: identical predictions on {len(test_frame):,} test rows - PASSED")
    print(f"Saved sample input ({N_SAMPLE} rows): {SAMPLE_INPUT}")


if __name__ == "__main__":
    main()
