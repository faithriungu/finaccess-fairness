"""Compare a common threshold with gender-specific TPR post-processing.

All thresholds are chosen from nested group-CV out-of-fold training scores.
The locked test set is used only to evaluate the frozen thresholds.
Run after: python -m src.evaluate
Run:       python -m src.mitigate
"""
import json
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, REPORTS_DIR, GENDER, CLUSTER, RANDOM_STATE


def weighted_rate(y, pred, w, condition):
    mask = np.asarray(condition, dtype=bool)
    denom = np.sum(w[mask])
    if denom <= 0:
        return np.nan
    return float(np.sum(w[mask] * pred[mask]) / denom)


def metric_bundle(d, pred):
    y = d["y"].to_numpy(dtype=int)
    w = d["weight"].to_numpy(dtype=float)
    pred = np.asarray(pred, dtype=int)
    result = {}
    for weighted in [False, True]:
        ww = w if weighted else np.ones(len(d), dtype=float)
        label = "weighted" if weighted else "unweighted"
        pos, neg = y == 1, y == 0
        tp = np.sum(ww * (pred == 1) * pos)
        fp = np.sum(ww * (pred == 1) * neg)
        actual_pos = np.sum(ww * pos)
        actual_neg = np.sum(ww * neg)
        selected = np.sum(ww * (pred == 1))
        precision = tp / selected if selected else np.nan
        recall = tp / actual_pos if actual_pos else np.nan
        fpr = fp / actual_neg if actual_neg else np.nan
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        result[f"precision_{label}"] = precision
        result[f"recall_{label}"] = recall
        result[f"F1_{label}"] = f1
        result[f"selection_rate_{label}"] = np.sum(ww * pred) / np.sum(ww)
        result[f"TPR_{label}"] = recall
        result[f"FPR_{label}"] = fpr
    return result


def tpr(y, pred, w):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    w = np.asarray(w, dtype=float)
    denom = np.sum(w * (y == 1))
    return float(np.sum(w * pred * (y == 1)) / denom) if denom > 0 else np.nan


def choose_global_threshold(oof):
    p = oof["prob"].to_numpy()
    y = oof["y"].to_numpy(dtype=int)
    w = oof["weight"].to_numpy(dtype=float)
    grid = np.unique(np.r_[np.linspace(0.05, 0.95, 181), 0.5, np.quantile(p, np.linspace(0.01, 0.99, 99))])
    scores = []
    for threshold in grid:
        pred = (p >= threshold).astype(int)
        bundle = metric_bundle(oof, pred)
        scores.append(bundle["F1_weighted"])
    return float(grid[int(np.nanargmax(scores))])


def threshold_for_target_tpr(group, target_tpr):
    p = group["prob"].to_numpy()
    y = group["y"].to_numpy(dtype=int)
    w = group["weight"].to_numpy(dtype=float)
    grid = np.unique(np.r_[np.linspace(0.01, 0.99, 197), np.quantile(p, np.linspace(0, 1, 101))])
    achieved = np.array([tpr(y, (p >= threshold).astype(int), w) for threshold in grid])
    return float(grid[np.nanargmin(np.abs(achieved - target_tpr))])


def group_gap(d, pred, metric="TPR"):
    out = {}
    for sex in ["Female", "Male"]:
        g = d[d[GENDER] == sex]
        out[sex] = metric_bundle(g, pred[g.index.to_numpy()])
    key = f"{metric}_weighted"
    return out["Male"][key] - out["Female"][key]


def cluster_bootstrap_gap(d, pred, n_boot=1000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    by_cluster = {c: idx for c, idx in d.groupby(CLUSTER).indices.items()}
    clusters = np.array(list(by_cluster.keys()))
    values = []
    for _ in range(n_boot):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([by_cluster[c] for c in chosen])
        sample = d.iloc[idx].reset_index(drop=True)
        p = pred[idx]
        female = sample[GENDER].eq("Female").to_numpy()
        male = sample[GENDER].eq("Male").to_numpy()
        gap = tpr(sample.loc[male, "y"], p[male], sample.loc[male, "weight"]) - \
              tpr(sample.loc[female, "y"], p[female], sample.loc[female, "weight"])
        if np.isfinite(gap):
            values.append(gap)
    return tuple(np.quantile(values, [0.025, 0.975])) if values else (np.nan, np.nan)


def main():
    oof_path = MODELS_DIR / "oof_train_predictions.csv"
    test_path = REPORTS_DIR / "test_predictions.csv"
    artifact_path = MODELS_DIR / "bank_model.joblib"
    for path in [oof_path, test_path, artifact_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}. Run clean, train and evaluate first.")

    import joblib
    artifact = joblib.load(artifact_path)
    champion = artifact["metadata"]["model_name"]
    oof_all = pd.read_csv(oof_path)
    oof = oof_all[oof_all["model"] == champion].copy().reset_index(drop=True)
    test = pd.read_csv(test_path).reset_index(drop=True)
    if oof.empty:
        raise ValueError(f"No OOF predictions found for champion {champion}.")
    if not set(oof[GENDER].unique()) >= {"Female", "Male"}:
        raise ValueError("Both Female and Male groups must be present in OOF data.")

    # Baseline common threshold is selected using weighted F1 on training OOF predictions.
    global_threshold = choose_global_threshold(oof)
    oof_global = (oof["prob"].to_numpy() >= global_threshold).astype(int)
    target_tpr = tpr(oof["y"], oof_global, oof["weight"])

    # Gender-specific thresholds target the baseline training TPR, using OOF data only.
    group_thresholds = {
        sex: threshold_for_target_tpr(oof[oof[GENDER] == sex], target_tpr)
        for sex in ["Female", "Male"]
    }

    test_prob = test["prob"].to_numpy()
    test_gender = test[GENDER].to_numpy()
    decisions = {
        "Common threshold 0.50": (test_prob >= 0.50).astype(int),
        "Common threshold (OOF weighted F1)": (test_prob >= global_threshold).astype(int),
        "Gender-specific thresholds (OOF TPR target)": np.array([
            int(prob >= group_thresholds[sex])
            for prob, sex in zip(test_prob, test_gender)
        ]),
    }

    rows = []
    for name, pred in decisions.items():
        row = {"method": name, **metric_bundle(test, pred)}
        female = test_gender == "Female"
        male = test_gender == "Male"
        for sex, mask in [("Female", female), ("Male", male)]:
            sub = test.loc[mask].reset_index(drop=True)
            sub_pred = pred[mask]
            m = metric_bundle(sub, sub_pred)
            for key, value in m.items():
                row[f"{sex}_{key}"] = value
        row["TPR_gap_M_minus_F_weighted"] = (
            row["Male_TPR_weighted"] - row["Female_TPR_weighted"]
        )
        row["FPR_gap_M_minus_F_weighted"] = (
            row["Male_FPR_weighted"] - row["Female_FPR_weighted"]
        )
        row["selection_gap_M_minus_F_weighted"] = (
            row["Male_selection_rate_weighted"] - row["Female_selection_rate_weighted"]
        )
        ci_low, ci_high = cluster_bootstrap_gap(test, pred)
        row["TPR_gap_CI_low"] = ci_low
        row["TPR_gap_CI_high"] = ci_high
        row["point_gap_within_5pp"] = abs(row["TPR_gap_M_minus_F_weighted"]) <= 0.05
        row["CI_entirely_within_5pp"] = ci_low >= -0.05 and ci_high <= 0.05
        rows.append(row)

    comparison = pd.DataFrame(rows).set_index("method")
    thresholds = {
        "champion": champion,
        "common_threshold_0_50": 0.50,
        "common_threshold_weighted_f1": global_threshold,
        "training_oof_target_tpr": target_tpr,
        "gender_specific_thresholds": group_thresholds,
        "note": "Gender-specific thresholds use gender at decision time and are a post-processing comparison, not an automatic deployment recommendation.",
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(REPORTS_DIR / "mitigation_comparison.csv")
    (REPORTS_DIR / "decision_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    print(f"Champion: {champion}")
    print(f"Common threshold selected on OOF weighted F1: {global_threshold:.3f}")
    print(f"Training OOF target TPR: {target_tpr:.3f}")
    print(f"Gender-specific thresholds: {group_thresholds}")
    print("\nTest-set comparison (weighted and unweighted metrics reported separately):")
    cols = [
        "precision_weighted", "recall_weighted", "F1_weighted",
        "Female_TPR_weighted", "Male_TPR_weighted",
        "TPR_gap_M_minus_F_weighted", "TPR_gap_CI_low", "TPR_gap_CI_high",
        "Female_FPR_weighted", "Male_FPR_weighted",
        "point_gap_within_5pp", "CI_entirely_within_5pp",
    ]
    print(comparison[cols].round(4).to_string())
    print("\nSaved mitigation comparison and threshold settings.")


if __name__ == "__main__":
    main()
