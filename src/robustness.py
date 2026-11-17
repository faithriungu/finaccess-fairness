"""Threshold robustness checks.

Run after: python -m src.mitigate
Run:       python -m src.robustness

Two questions:
1. Stability - how much would the chosen thresholds move if the training
   sample were slightly different? The threshold-selection rules from
   src/mitigate.py are re-run on cluster-bootstrap resamples of the training
   out-of-fold (OOF) predictions. Only training data is used.
2. Sensitivity - how do precision, recall and the gender TPR gap on the
   locked test set change across a range of common thresholds? This is a
   descriptive sweep; it is NOT used to choose the threshold.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR, GENDER, CLUSTER, RANDOM_STATE,
    TRAINED_MODEL_FILE,
)
from src.mitigate import (
    choose_global_threshold, threshold_for_target_tpr, metric_bundle, tpr,
)

N_BOOT_THRESHOLDS = 200
SWEEP = np.round(np.arange(0.15, 0.651, 0.025), 3)


def bootstrap_thresholds(oof, n_boot=N_BOOT_THRESHOLDS, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    by_cluster = oof.groupby(CLUSTER).indices
    clusters = np.array(list(by_cluster.keys()))
    rows = []
    for b in range(n_boot):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        sample = oof.iloc[np.concatenate([by_cluster[c] for c in chosen])].reset_index(drop=True)
        common = choose_global_threshold(sample)
        pred = (sample["prob"].to_numpy() >= common).astype(int)
        target = tpr(sample["y"], pred, sample["weight"])
        rows.append({
            "replicate": b + 1,
            "common_threshold": common,
            "target_tpr": target,
            "female_threshold": threshold_for_target_tpr(sample[sample[GENDER] == "Female"], target),
            "male_threshold": threshold_for_target_tpr(sample[sample[GENDER] == "Male"], target),
        })
    return pd.DataFrame(rows)


def summarise(replicates, chosen):
    rows = []
    for col, key in [
        ("common_threshold", "common_threshold_weighted_f1"),
        ("female_threshold", "Female"),
        ("male_threshold", "Male"),
    ]:
        value = chosen[key] if key in chosen else chosen["gender_specific_thresholds"][key]
        series = replicates[col]
        rows.append({
            "threshold": col,
            "chosen_on_full_oof": value,
            "bootstrap_mean": series.mean(),
            "bootstrap_sd": series.std(),
            "ci_low": series.quantile(0.025),
            "ci_high": series.quantile(0.975),
        })
    return pd.DataFrame(rows)


def test_sweep(test):
    rows = []
    prob = test["prob"].to_numpy()
    for threshold in SWEEP:
        pred = (prob >= threshold).astype(int)
        overall = metric_bundle(test, pred)
        row = {
            "threshold": threshold,
            "precision_weighted": overall["precision_weighted"],
            "recall_weighted": overall["recall_weighted"],
            "F1_weighted": overall["F1_weighted"],
            "selection_rate_weighted": overall["selection_rate_weighted"],
        }
        for sex in ["Female", "Male"]:
            mask = (test[GENDER] == sex).to_numpy()
            m = metric_bundle(test.loc[mask].reset_index(drop=True), pred[mask])
            row[f"{sex}_TPR_weighted"] = m["TPR_weighted"]
            row[f"{sex}_FPR_weighted"] = m["FPR_weighted"]
        row["TPR_gap_M_minus_F"] = row["Male_TPR_weighted"] - row["Female_TPR_weighted"]
        row["FPR_gap_M_minus_F"] = row["Male_FPR_weighted"] - row["Female_FPR_weighted"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_sweep(sweep, chosen_threshold, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(sweep["threshold"], sweep["precision_weighted"], label="Precision")
    ax.plot(sweep["threshold"], sweep["recall_weighted"], label="Recall")
    ax.plot(sweep["threshold"], sweep["F1_weighted"], label="F1")
    ax.set_title("Overall (weighted)")
    ax = axes[1]
    ax.plot(sweep["threshold"], sweep["Female_TPR_weighted"], label="Female TPR")
    ax.plot(sweep["threshold"], sweep["Male_TPR_weighted"], label="Male TPR")
    ax.plot(sweep["threshold"], sweep["TPR_gap_M_minus_F"], linestyle="--", label="TPR gap (M - F)")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_title("Recall by gender (weighted)")
    for ax in axes:
        ax.axvline(chosen_threshold, color="black", linestyle=":", linewidth=1,
                   label=f"Chosen threshold {chosen_threshold:.3f}")
        ax.set_xlabel("Common decision threshold")
        ax.legend(fontsize=8)
    fig.suptitle("Locked test set - threshold sensitivity (descriptive only)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    import joblib
    thresholds_path = REPORTS_DIR / "decision_thresholds.json"
    for path in [MODELS_DIR / "oof_train_predictions.csv",
                 REPORTS_DIR / "test_predictions.csv", thresholds_path, TRAINED_MODEL_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}. Run the earlier steps first.")

    champion = joblib.load(TRAINED_MODEL_FILE)["metadata"]["model_name"]
    oof_all = pd.read_csv(MODELS_DIR / "oof_train_predictions.csv")
    oof = oof_all[oof_all["model"] == champion].reset_index(drop=True)
    test = pd.read_csv(REPORTS_DIR / "test_predictions.csv").reset_index(drop=True)
    chosen = json.loads(thresholds_path.read_text(encoding="utf-8"))

    print(f"Bootstrapping threshold selection on training OOF ({N_BOOT_THRESHOLDS} replicates)...")
    replicates = bootstrap_thresholds(oof)
    stability = summarise(replicates, chosen)

    sweep = test_sweep(test)

    replicates.to_csv(REPORTS_DIR / "threshold_bootstrap_replicates.csv", index=False)
    stability.to_csv(REPORTS_DIR / "threshold_stability.csv", index=False)
    sweep.to_csv(REPORTS_DIR / "threshold_sweep_test.csv", index=False)
    plot_sweep(sweep, chosen["common_threshold_weighted_f1"], FIGURES_DIR / "threshold_sweep_test.png")

    print("\nThreshold stability (training OOF cluster bootstrap):")
    print(stability.round(4).to_string(index=False))
    print("\nTest-set sensitivity sweep (descriptive):")
    print(sweep[["threshold", "precision_weighted", "recall_weighted", "F1_weighted",
                 "Female_TPR_weighted", "Male_TPR_weighted", "TPR_gap_M_minus_F"]]
          .round(4).to_string(index=False))
    print("\nSaved threshold_stability.csv, threshold_sweep_test.csv and figure.")


if __name__ == "__main__":
    main()
