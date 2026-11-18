"""Gender-proxy check: how much gender information do the 35 predictors carry?

Run after: python -m src.train
Run:       python -m src.proxy_check

Gender is excluded from the model inputs. That does not make the model
gender-blind if other inputs reveal gender. This step measures it directly:
a classifier with the same architecture and settings as the champion is
trained to predict respondent gender (not bank use) from the predictors.

- Fitted on the training clusters only; scored on the locked test clusters.
- AUC 0.5 = predictors carry no gender information; 1.0 = gender fully recoverable.
- Repeated with groups of household-role variables removed, to show where the
  information comes from, plus each predictor on its own.

The bank-use model is not changed by this step.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.config import (
    CLEAN_FILE, FEATURES_FILE, MODELS_DIR, REPORTS_DIR, GENDER, WEIGHT,
    RESPONDENT_ID, RANDOM_STATE, TRAINED_MODEL_FILE,
)
from src.train import make_pipeline

HOUSEHOLD_ROLE = ["relationship_to_hh_head", "marital_status",
                  "decides_daily_spending", "decides_major_spending"]
ABLATIONS = {
    "All 35 predictors": [],
    "Without relationship_to_hh_head": ["relationship_to_hh_head"],
    "Without all household-role variables": HOUSEHOLD_ROLE,
}


def gender_auc(train, test, features, numeric, params):
    num = [c for c in numeric if c in features]
    cat = [c for c in features if c not in num]
    pipe = make_pipeline(HistGradientBoostingClassifier(max_iter=200, random_state=RANDOM_STATE),
                         num, cat)
    pipe.set_params(**params)
    pipe = clone(pipe)

    def prep(frame):
        x = frame[features].copy()
        for col in cat:
            x[col] = x[col].astype(object).where(x[col].notna(), np.nan)
        return x

    y_train = (train[GENDER] == "Female").astype(int)
    y_test = (test[GENDER] == "Female").astype(int)
    pipe.fit(prep(train), y_train, model__sample_weight=train[WEIGHT].to_numpy())
    prob = pipe.predict_proba(prep(test))[:, 1]
    return (roc_auc_score(y_test, prob, sample_weight=test[WEIGHT]),
            roc_auc_score(y_test, prob))


def main():
    if not TRAINED_MODEL_FILE.exists():
        raise FileNotFoundError("Run python -m src.train first.")
    trained = joblib.load(TRAINED_MODEL_FILE)
    params = trained["metadata"]["best_params"]
    numeric = trained["numeric_features"]
    features = json.loads(FEATURES_FILE.read_text(encoding="utf-8"))["final_features"]

    df = pd.read_csv(CLEAN_FILE, low_memory=False)
    df[RESPONDENT_ID] = df[RESPONDENT_ID].astype(str)
    split = pd.read_csv(MODELS_DIR / "split.csv", dtype={RESPONDENT_ID: str})
    df = df.merge(split, on=RESPONDENT_ID, how="inner", validate="one_to_one")
    df = df[df[GENDER].isin(["Female", "Male"])]
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]

    rows = []
    for label, dropped in ABLATIONS.items():
        kept = [f for f in features if f not in dropped]
        weighted, unweighted = gender_auc(train, test, kept, numeric, params)
        rows.append({"predictor_set": label, "n_predictors": len(kept),
                     "gender_AUC_weighted": weighted, "gender_AUC_unweighted": unweighted})
    summary = pd.DataFrame(rows)

    single = []
    for feature in features:
        weighted, unweighted = gender_auc(train, test, [feature], numeric, params)
        single.append({"predictor": feature, "gender_AUC_weighted": weighted,
                       "gender_AUC_unweighted": unweighted})
    single = (pd.DataFrame(single)
              .sort_values("gender_AUC_weighted", ascending=False)
              .reset_index(drop=True))

    summary.to_csv(REPORTS_DIR / "gender_proxy_check.csv", index=False)
    single.to_csv(REPORTS_DIR / "gender_proxy_single_predictors.csv", index=False)

    print("GENDER-PROXY CHECK (locked test set; 0.5 = no gender information)")
    print(summary.round(3).to_string(index=False))
    print("\nStrongest single predictors of gender:")
    print(single.head(8).round(3).to_string(index=False))
    print("\nSaved gender_proxy_check.csv and gender_proxy_single_predictors.csv in reports/.")


if __name__ == "__main__":
    main()