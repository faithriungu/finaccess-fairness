"""Evaluate the locked test set once, with weighted and unweighted results.

Run after: python -m src.clean && python -m src.train
Run:       python -m src.evaluate
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_score, recall_score, f1_score,
)

from src.config import (
    CLEAN_FILE, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, TARGET, GENDER, WEIGHT,
    CLUSTER, RESPONDENT_ID, LOCATION, TRAINED_MODEL_FILE, N_BOOT, RANDOM_STATE,
)

CALIBRATION_BINS = np.linspace(0, 1, 11)


def safe_div(num, den):
    return float(num / den) if den > 0 else np.nan


def group_metrics(frame, threshold=0.5, weighted=True):
    y = frame["y"].to_numpy(dtype=int)
    p = frame["prob"].to_numpy(dtype=float)
    w = frame["weight"].to_numpy(dtype=float) if weighted else None
    pred = (p >= threshold).astype(int)
    kwargs = {"sample_weight": w} if weighted else {}
    positives = y == 1
    negatives = y == 0
    tp = float(np.sum((w if weighted else 1) * ((pred == 1) & positives)))
    actual_pos = float(np.sum((w if weighted else 1) * positives))
    selected = float(np.sum((w if weighted else 1) * (pred == 1)))
    fp = float(np.sum((w if weighted else 1) * ((pred == 1) & negatives)))
    actual_neg = float(np.sum((w if weighted else 1) * negatives))
    return {
        "n": len(frame),
        "base_rate": float(np.average(y, weights=w)) if weighted else float(y.mean()),
        "selection_rate": float(np.average(pred, weights=w)) if weighted else float(pred.mean()),
        "precision": safe_div(tp, selected),
        "recall_TPR": safe_div(tp, actual_pos),
        "FNR": 1 - safe_div(tp, actual_pos) if actual_pos else np.nan,
        "FPR": safe_div(fp, actual_neg),
        "F1": f1_score(y, pred, sample_weight=w, zero_division=0),
        "PR_AUC": average_precision_score(y, p, sample_weight=w),
        "ROC_AUC": roc_auc_score(y, p, sample_weight=w) if len(np.unique(y)) == 2 else np.nan,
        "Brier": brier_score_loss(y, p, sample_weight=w),
    }


def cluster_bootstrap_gap(frame, metric_name="recall_TPR", n_boot=1000, seed=42):
    """Cluster bootstrap CI for Male minus Female weighted group metric."""
    rng = np.random.default_rng(seed)
    groups = {key: idx for key, idx in frame.groupby(CLUSTER).indices.items()}
    cluster_ids = np.array(list(groups.keys()))
    gaps = []
    for _ in range(n_boot):
        chosen = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        idx = np.concatenate([groups[c] for c in chosen])
        sample = frame.iloc[idx]
        female = sample[sample[GENDER] == "Female"]
        male = sample[sample[GENDER] == "Male"]
        if len(female) == 0 or len(male) == 0:
            continue
        fm = group_metrics(female, weighted=True)[metric_name]
        mm = group_metrics(male, weighted=True)[metric_name]
        if np.isfinite(fm) and np.isfinite(mm):
            gaps.append(mm - fm)
    if not gaps:
        return np.nan, np.nan
    return tuple(np.quantile(gaps, [0.025, 0.975]))


def cluster_bootstrap_metric(frame, metric_name, n_boot=N_BOOT, seed=RANDOM_STATE):
    """Cluster bootstrap 95% CI for one weighted metric within a subset."""
    rng = np.random.default_rng(seed)
    groups = frame.groupby(CLUSTER).indices
    cluster_ids = np.array(list(groups.keys()))
    values = []
    for _ in range(n_boot):
        chosen = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sample = frame.iloc[np.concatenate([groups[c] for c in chosen])]
        if sample["y"].nunique() < 2:
            continue
        value = group_metrics(sample, weighted=True)[metric_name]
        if np.isfinite(value):
            values.append(value)
    return tuple(np.quantile(values, [0.025, 0.975])) if values else (np.nan, np.nan)


def calibration_table(d):
    """Weighted mean predicted vs observed bank use, by gender and probability bin."""
    rows = []
    for sex, g in d.groupby(GENDER):
        bins = pd.cut(g["prob"], CALIBRATION_BINS, include_lowest=True)
        for interval, b in g.groupby(bins, observed=True):
            rows.append({
                "gender": sex,
                "bin": str(interval),
                "n": len(b),
                "mean_predicted_weighted": np.average(b["prob"], weights=b["weight"]),
                "observed_rate_weighted": np.average(b["y"], weights=b["weight"]),
            })
    table = pd.DataFrame(rows)
    table["gap_predicted_minus_observed"] = (
        table["mean_predicted_weighted"] - table["observed_rate_weighted"]
    )
    return table


def calibration_summary(d):
    """Calibration-in-the-large and weighted expected calibration error by gender."""
    rows = []
    for sex, g in [("All", d)] + list(d.groupby(GENDER)):
        bins = pd.cut(g["prob"], CALIBRATION_BINS, include_lowest=True)
        total_w = g["weight"].sum()
        ece = 0.0
        for _, b in g.groupby(bins, observed=True):
            share = b["weight"].sum() / total_w
            ece += share * abs(
                np.average(b["prob"], weights=b["weight"])
                - np.average(b["y"], weights=b["weight"])
            )
        rows.append({
            "group": sex,
            "mean_predicted_weighted": np.average(g["prob"], weights=g["weight"]),
            "observed_rate_weighted": np.average(g["y"], weights=g["weight"]),
            "calibration_in_large": np.average(g["prob"], weights=g["weight"])
            - np.average(g["y"], weights=g["weight"]),
            "ECE_weighted": ece,
        })
    return pd.DataFrame(rows)


def plot_calibration(table, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Perfect calibration")
    for sex, marker in [("Female", "o"), ("Male", "s")]:
        t = table[table["gender"] == sex]
        ax.plot(t["mean_predicted_weighted"], t["observed_rate_weighted"],
                marker=marker, label=sex)
    ax.set_xlabel("Mean predicted probability (weighted)")
    ax.set_ylabel("Observed bank-use rate (weighted)")
    ax.set_title("Reliability diagram by gender - locked test set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def segment_metrics(d, threshold=0.5):
    """Gender x location results with cluster-bootstrap CIs for TPR."""
    rows = []
    for (location, sex), g in d.groupby([LOCATION, GENDER]):
        m = group_metrics(g, threshold=threshold, weighted=True)
        lo, hi = cluster_bootstrap_metric(g, "recall_TPR")
        rows.append({
            "location_type": location, "gender": sex, **m,
            "TPR_ci_low": lo, "TPR_ci_high": hi,
        })
    return pd.DataFrame(rows)


def main():
    artifact_path = TRAINED_MODEL_FILE
    split_path = MODELS_DIR / "split.csv"
    if not artifact_path.exists() or not split_path.exists():
        raise FileNotFoundError("Run python -m src.train before evaluation.")

    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    features = artifact["features"]
    df = pd.read_csv(CLEAN_FILE, low_memory=False)
    split = pd.read_csv(split_path, dtype={RESPONDENT_ID: str})
    df[RESPONDENT_ID] = df[RESPONDENT_ID].astype(str)
    df = df.merge(split, on=RESPONDENT_ID, how="inner", validate="one_to_one")
    test = df[df["split"] == "test"].copy()
    if test.empty:
        raise ValueError("No test observations found. Check split.csv and respondent IDs.")

    prob = model.predict_proba(test[features])[:, 1]
    d = pd.DataFrame({
        RESPONDENT_ID: test[RESPONDENT_ID].to_numpy(),
        "y": test[TARGET].to_numpy(dtype=int),
        "prob": prob,
        "weight": test[WEIGHT].to_numpy(dtype=float),
        GENDER: test[GENDER].to_numpy(),
        "location_type": test["location_type"].to_numpy(),
        CLUSTER: test[CLUSTER].to_numpy(),
    })

    # Overall results, explicitly separating weighted from unweighted estimates.
    overall_rows = []
    for weighted in [False, True]:
        overall_rows.append({
            "estimation": "Survey weighted" if weighted else "Unweighted sample",
            **group_metrics(d, threshold=0.5, weighted=weighted),
        })
    overall = pd.DataFrame(overall_rows).set_index("estimation")
    # A no-skill model's PR-AUC equals the base rate; report it alongside.
    overall["PR_AUC_chance_baseline"] = overall["base_rate"]
    overall["PR_AUC_lift_over_chance"] = overall["PR_AUC"] - overall["base_rate"]

    # Gender-specific metrics using the same metric definition and threshold.
    gender_rows = []
    for sex in ["Female", "Male"]:
        subset = d[d[GENDER] == sex]
        for weighted in [False, True]:
            gender_rows.append({
                "gender": sex,
                "estimation": "Survey weighted" if weighted else "Unweighted sample",
                **group_metrics(subset, threshold=0.5, weighted=weighted),
            })
    gender_table = pd.DataFrame(gender_rows)

    # Weighted group gaps and cluster-bootstrap confidence intervals.
    weighted_group = {
        sex: group_metrics(d[d[GENDER] == sex], weighted=True)
        for sex in ["Female", "Male"]
    }
    gap_rows = []
    for metric in ["base_rate", "selection_rate", "recall_TPR", "FPR", "precision", "PR_AUC", "ROC_AUC", "Brier"]:
        female = weighted_group["Female"][metric]
        male = weighted_group["Male"][metric]
        lo, hi = (np.nan, np.nan)
        if metric in ["recall_TPR", "FPR", "selection_rate"]:
            lo, hi = cluster_bootstrap_gap(d, metric_name=metric)
        gap_rows.append({
            "metric": metric,
            "female_weighted": female,
            "male_weighted": male,
            "gap_male_minus_female": male - female,
            "ci_low": lo,
            "ci_high": hi,
        })
    gaps = pd.DataFrame(gap_rows)

    calibration = calibration_table(d)
    calibration_overall = calibration_summary(d)
    segments = segment_metrics(d)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(REPORTS_DIR / "test_calibration_by_gender.csv", index=False)
    calibration_overall.to_csv(REPORTS_DIR / "test_calibration_summary.csv", index=False)
    plot_calibration(calibration, FIGURES_DIR / "calibration_by_gender.png")
    segments.to_csv(REPORTS_DIR / "test_segment_metrics.csv", index=False)
    d.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)
    overall.to_csv(REPORTS_DIR / "test_overall_metrics.csv")
    gender_table.to_csv(REPORTS_DIR / "test_gender_metrics.csv", index=False)
    gaps.to_csv(REPORTS_DIR / "test_gender_gaps.csv", index=False)

    print(f"Champion: {artifact['metadata']['model_name']}")
    print(f"Locked test observations: {len(d):,}; clusters: {d[CLUSTER].nunique():,}")
    print("\nOverall test metrics at threshold 0.50:")
    print(overall.round(4).to_string())
    print("\nGender metrics:")
    print(gender_table.round(4).to_string(index=False))
    print("\nWeighted gender gaps (Male minus Female):")
    print(gaps.round(4).to_string(index=False))
    print("\nCalibration by gender (weighted):")
    print(calibration_overall.round(4).to_string(index=False))
    print("\nGender x location at threshold 0.50 (weighted):")
    print(segments[["location_type", "gender", "n", "base_rate", "selection_rate",
                    "recall_TPR", "TPR_ci_low", "TPR_ci_high", "FPR", "ROC_AUC"]]
          .round(4).to_string(index=False))
    print("\nSaved test predictions, calibration, segment and gap tables in reports/.")


if __name__ == "__main__":
    main()
