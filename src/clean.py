"""Create an adult, model-ready FinAccess file without unsupported recoding.

Run from the project root: python -m src.clean
The source finaccess_selected.csv is never overwritten.
"""
import json
import numpy as np
import pandas as pd

from src.config import (
    SELECTED_FILE, CLEAN_FILE, FEATURES_FILE, REPORTS_DIR,
    TARGET, GENDER, WEIGHT, CLUSTER, RESPONDENT_ID,
)

# Main-model predictors. Gender is deliberately held out for fairness auditing.
# Direct financial-inclusion outcomes and survey identifiers are not predictors.
FEATURES = [
    "age", "household_size", "num_income_sources",
    "income_last_month", "income_highest_month", "income_lowest_month",
    "education_level", "wealth_quintile", "internet_use_frequency",
    "financial_literacy_score", "cost_to_nearest_provider",
    "distance_to_nearest_provider", "disability_status",
    "income_from_farming", "income_from_regular_job",
    "income_from_casual_work", "income_from_own_business",
    "relies_on_family_support", "has_any_id", "owns_phone",
    "phone_internet_capable", "can_read_sms", "skipped_meals_last_month",
    "experienced_shock", "location_type", "county", "livelihood_category",
    "marital_status", "relationship_to_hh_head", "decides_daily_spending",
    "decides_major_spending", "phone_access_type", "nearest_provider_type",
    "network_quality", "land_title_status",
]

NUMERIC_FEATURES = [
    "age", "household_size", "num_income_sources",
    "income_last_month", "income_highest_month", "income_lowest_month",
    "income_from_farming", "income_from_regular_job",
    "income_from_casual_work", "income_from_own_business",
    "relies_on_family_support",
]

INCOME_COLUMNS = ["income_last_month", "income_highest_month", "income_lowest_month"]
BINARY_COLUMNS = [
    "owns_house", "owns_land", "owns_phone", "skipped_meals_last_month",
    "experienced_shock",
]


def text_series(series: pd.Series) -> pd.Series:
    """Trim text while retaining actual missing values."""
    out = series.astype("string").str.strip()
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return out


def yes_no(series: pd.Series) -> pd.Series:
    """Map explicit Yes/No responses; all other responses remain missing."""
    s = text_series(series).str.lower()
    return s.map({"yes": 1, "no": 0}).astype("float")


def simplify_relationship(value):
    if pd.isna(value):
        return np.nan
    s = str(value).lower()
    if "head of household" in s:
        return "Head"
    if "spouse" in s:
        return "Spouse"
    if "son" in s or "daughter" in s or "child" in s:
        return "Child"
    if "mother" in s or "father" in s or "parent" in s:
        return "Parent"
    if "other relative" in s:
        return "Other relative"
    if "non-relative" in s or "domestic worker" in s or "friend" in s:
        return "Other non-relative"
    return str(value).strip()


def simplify_decision(value):
    if pd.isna(value):
        return np.nan
    s = str(value).lower()
    if s == "self":
        return "Self"
    if "jointly" in s:
        return "Joint decision"
    if "don't know" in s or "dont know" in s:
        return "Don't know"
    if "refused" in s:
        return "Refused"
    if "not stated" in s:
        return "Not stated"
    return "Other person / no one"


def simplify_phone_access(value):
    if pd.isna(value):
        return np.nan
    s = str(value).lower()
    if "only i use" in s:
        return "Own phone - sole use"
    if "both me and others" in s:
        return "Own phone - shared use"
    if "able to use someone else" in s:
        return "Uses another person's phone"
    if "don't own or use" in s or "sina wala situmii" in s:
        return "Does not own or use a phone"
    if "#null!" in s:
        return "Not recorded / skipped"
    if "refused" in s:
        return "Refused"
    return str(value).strip()


def simplify_provider(value):
    if pd.isna(value):
        return np.nan
    s = str(value).lower()
    if "mobile money agent" in s:
        return "Mobile money agent"
    if "bank agent" in s or "post bank agent" in s:
        return "Bank / Postbank agent"
    if "commercial bank" in s or "post bank branch" in s or "atm" in s:
        return "Commercial / microfinance / Postbank branch or ATM"
    if "sacco" in s:
        return "SACCO"
    if "microfinance institution" in s:
        return "Microfinance institution"
    if "insurance" in s:
        return "Insurance provider"
    if "don't know" in s or "dont know" in s:
        return "Don't know"
    if "refused" in s:
        return "Refused"
    if "#null!" in s:
        return "Not recorded / skipped"
    return str(value).strip()


def simplify_network(value):
    if pd.isna(value):
        return np.nan
    s = str(value).lower()
    if "strong" in s:
        return "Strong"
    if "okay" in s or "fair" in s or "average" in s:
        return "Fair / average"
    if "weak" in s or "poor" in s:
        return "Weak / poor"
    if "don't know" in s or "dont know" in s:
        return "Don't know"
    if "refused" in s:
        return "Refused"
    if "#null!" in s:
        return "Not recorded / skipped"
    return str(value).strip()


def simplify_frequency(value):
    if pd.isna(value):
        return np.nan
    s = str(value).lower()
    if "never" in s:
        return "Never"
    if "daily" in s or "kila siku" in s:
        return "Daily"
    if "weekly" in s or "kila wiki" in s:
        return "Weekly"
    if "monthly" in s or "mwezi" in s:
        return "Monthly"
    if "less often" in s or "less often" in s or "mara kwa mara" in s:
        return "Less often"
    if "refused" in s:
        return "Refused"
    if "#null!" in s:
        return "Not recorded / skipped"
    return str(value).strip()


def simplify_sms(value):
    if pd.isna(value):
        return np.nan
    s = str(value).lower()
    if "then correct" in s:
        return "Correct response"
    if "visually impaired" in s or "visually" in s:
        return "Visually impaired"
    if "then incorrect" in s or "incorrect" in s:
        return "Incorrect response"
    if "don't know" in s or "dont know" in s:
        return "Don't know"
    if "refused" in s:
        return "Refused"
    if "#null!" in s:
        return "Not recorded / skipped"
    return str(value).strip()


def derive_land_title(row):
    land = row.get("owns_land")
    doc = row.get("land_ownership_document")
    if land == 0:
        return "Does not own land"
    if pd.isna(land):
        return "Land ownership unknown"
    if pd.isna(doc):
        return "Ownership document unknown"
    s = str(doc).lower()
    if "no ownership document" in s:
        return "No ownership document"
    if any(term in s for term in ["title deed", "certificate of lease", "sectional title", "transfer deed"]):
        return "Formal ownership document"
    if any(term in s for term in ["refused", "#null!", "98"]):
        return "Ownership document unknown"
    return "Other ownership document"


def main():
    if not SELECTED_FILE.exists():
        raise FileNotFoundError(
            f"Selected data not found: {SELECTED_FILE}\n"
            "Place finaccess_selected.csv in data/processed/ before running this script."
        )

    df = pd.read_csv(SELECTED_FILE, low_memory=False)
    required = [RESPONDENT_ID, CLUSTER, WEIGHT, GENDER, "age", TARGET]
    missing_required = sorted(set(required) - set(df.columns))
    if missing_required:
        raise ValueError(f"Required columns missing from selected data: {missing_required}")

    start_n = len(df)
    if df[RESPONDENT_ID].duplicated().any():
        raise ValueError("Duplicate respondent_id values found. Resolve them before modelling.")

    # Standardise text columns without treating survey response codes as zeros.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = text_series(df[col])

    # Convert identifiers and survey design fields.
    df[RESPONDENT_ID] = df[RESPONDENT_ID].astype("string")
    df[CLUSTER] = pd.to_numeric(df[CLUSTER], errors="coerce")
    df[WEIGHT] = pd.to_numeric(df[WEIGHT], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Restrict the modelling population to adults, as specified in the study design.
    before_adults = len(df)
    df = df.loc[df["age"].ge(18)].copy()
    adult_n = len(df)

    # Target: retain only the two documented FinAccess categories.
    target_map = {"Usage": 1, "Non-usage": 0}
    df[TARGET] = df[TARGET].map(target_map)
    if df[TARGET].isna().any():
        bad = df.loc[df[TARGET].isna(), "target_bank_account"].value_counts(dropna=False).to_dict()
        raise ValueError(f"Unmapped or missing target values after adult filter: {bad}")
    df[TARGET] = df[TARGET].astype(int)

    # Gender is not a model feature, but it is required for group fairness analysis.
    df[GENDER] = text_series(df[GENDER])
    if not set(df[GENDER].dropna().unique()).issubset({"Female", "Male"}):
        raise ValueError(f"Unexpected gender categories: {df[GENDER].unique().tolist()}")
    if df[GENDER].isna().any():
        raise ValueError("Missing gender values must be investigated before fairness evaluation.")

    # Survey design integrity checks.
    if df[CLUSTER].isna().any():
        raise ValueError("Missing cluster_id values found.")
    if df[WEIGHT].isna().any() or (df[WEIGHT] <= 0).any():
        raise ValueError("Survey weights must be present and strictly positive.")

    # Convert known Yes/No indicators. Non-substantive responses remain missing.
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = yes_no(df[col])

    if "disability_status" in df.columns:
        s = text_series(df["disability_status"])
        df["disability_status"] = s.map({
            "With Disability": 1, "Without Disability": 0
        }).astype("float")

    if "has_any_id" in df.columns:
        s = text_series(df["has_any_id"])
        df["has_any_id"] = np.select(
            [s.eq("No ID").fillna(False), s.str.startswith("Has ID", na=False)],
            [0, 1], default=np.nan
        )

    # Official formal-access outcome retained only for auditing, never used as a feature.
    if "target_formal_access" in df.columns:
        df["target_formal_access"] = yes_no(df["target_formal_access"])
    if "mobile_money_access" in df.columns:
        df["mobile_money_access"] = yes_no(df["mobile_money_access"])

    # Income fields are numeric KSh amounts. '#NULL!' and other non-numeric
    # responses become NaN and are imputed inside each training fold.
    for col in INCOME_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype("string").str.replace(",", "", regex=False),
                errors="coerce"
            )
            df.loc[df[col] < 0, col] = np.nan

    # Flag impossible high/low pairs. Do not create an income-volatility ratio.
    inconsistent_income = (
        df["income_lowest_month"].notna() &
        df["income_highest_month"].notna() &
        (df["income_lowest_month"] > df["income_highest_month"])
    )
    n_inconsistent_income = int(inconsistent_income.sum())
    df.loc[inconsistent_income, ["income_lowest_month", "income_highest_month"]] = np.nan

    # Convert count variables to numeric but do not assume missing means zero.
    for col in ["num_children", "num_children_under5", "num_indigenous_cattle"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace({"#NULL!": np.nan}), errors="coerce")

    # Existing binary source indicators should contain only 0/1.
    source_binary = [
        "income_from_farming", "income_from_regular_job",
        "income_from_casual_work", "income_from_own_business",
        "relies_on_family_support",
    ]
    for col in source_binary:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            invalid = ~df[col].isin([0, 1]) & df[col].notna()
            if invalid.any():
                raise ValueError(f"Unexpected values in {col}: {df.loc[invalid, col].unique()[:10]}")

    for col in ["age", "household_size", "num_income_sources"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Explicitly retain 'Other' education. Missing education remains missing;
    # it is not recoded to 'no schooling'.
    if "education_level" in df.columns:
        df["education_level"] = text_series(df["education_level"])

    # Keep survey response levels categorical unless the source supports a true numeric scale.
    if "phone_internet_capable" in df.columns:
        s = text_series(df["phone_internet_capable"])
        s = s.map({"0": "No", "1": "Yes", "#NULL!": "Not recorded / skipped"})
        no_owned_phone = df.get("owns_phone", pd.Series(index=df.index, dtype=float)).eq(0)
        s = s.mask(s.eq("Not recorded / skipped") & no_owned_phone, "Not asked - no personally owned phone")
        s = s.fillna("Not recorded / skipped")
        df["phone_internet_capable"] = s.astype("object")

    if "phone_access_type" in df.columns:
        df["phone_access_type"] = df["phone_access_type"].map(simplify_phone_access)
    if "internet_use_frequency" in df.columns:
        df["internet_use_frequency"] = df["internet_use_frequency"].map(simplify_frequency)
    if "network_quality" in df.columns:
        df["network_quality"] = df["network_quality"].map(simplify_network)
    if "nearest_provider_type" in df.columns:
        df["nearest_provider_type"] = df["nearest_provider_type"].map(simplify_provider)
    if "can_read_sms" in df.columns:
        df["can_read_sms"] = df["can_read_sms"].map(simplify_sms)
    if "relationship_to_hh_head" in df.columns:
        df["relationship_to_hh_head"] = df["relationship_to_hh_head"].map(simplify_relationship)
    for col in ["decides_daily_spending", "decides_major_spending"]:
        if col in df.columns:
            df[col] = df[col].map(simplify_decision)

    # Land title status: only classify as formal where the response explicitly names
    # a title deed, certificate of lease, sectional title or transfer deed.
    if "owns_land" in df.columns and "land_ownership_document" in df.columns:
        df["land_title_status"] = df.apply(derive_land_title, axis=1)

    # Treat '#NULL!' as a named not-recorded response in selected categorical fields.
    # Genuine blank values (e.g. education) remain NaN and are imputed in-fold.
    for col in [
        "wealth_quintile", "financial_literacy_score", "cost_to_nearest_provider",
        "distance_to_nearest_provider", "marital_status", "livelihood_category",
        "county", "location_type",
    ]:
        if col in df.columns:
            s = text_series(df[col])
            df[col] = s.replace({"#NULL!": "Not recorded / skipped"})

    # Validate the intended model feature list.
    absent = sorted(set(FEATURES) - set(df.columns))
    if absent:
        raise ValueError(f"Expected model features missing from selected data: {absent}")

    # Confirm no leakage or identifiers are included in model predictors.
    forbidden = {
        RESPONDENT_ID, CLUSTER, WEIGHT, GENDER, "hh_head_gender",
        "target_formal_access", TARGET, "mobile_money_access",
        "income_volatility", "num_children", "num_children_under5",
        "num_indigenous_cattle", "owns_house", "land_ownership_document",
    }
    if set(FEATURES) & forbidden:
        raise AssertionError(f"Forbidden feature(s) present: {set(FEATURES) & forbidden}")

    # Store the cleaned adult data. No global imputation/encoding is performed here.
    df.to_csv(CLEAN_FILE, index=False)
    feature_types = {
        "numeric": [c for c in FEATURES if c in NUMERIC_FEATURES],
        "categorical": [c for c in FEATURES if c not in NUMERIC_FEATURES],
    }
    FEATURES_FILE.write_text(json.dumps({
        "target": TARGET,
        "sensitive_attribute": GENDER,
        "final_features": FEATURES,
        "feature_types": feature_types,
        "excluded_from_primary_model": sorted(forbidden),
        "notes": [
            "Gender is retained for fairness evaluation and excluded from model inputs.",
            "target_formal_access and mobile_money_access are excluded from the primary predictor set.",
            "Education missing values are not recoded as no schooling; Other is retained.",
            "Income values are numeric; impossible lowest-greater-than-highest pairs are set missing.",
            "No income_volatility ratio is created because it is not a validated volatility measure.",
            "Children and cattle count variables are retained for audit only and excluded from the model because their missingness/skip pattern is unresolved or extensive.",
            "All learned imputation and encoding occurs inside model pipelines after data splitting.",
        ],
    }, indent=2), encoding="utf-8")

    # Create a compact audit workbook for reproducibility.
    missing_overall = pd.DataFrame({
        "missing_n": df.isna().sum(),
        "missing_pct": df.isna().mean() * 100,
    }).sort_values("missing_pct", ascending=False)
    missing_gender = []
    for col in FEATURES:
        for sex, group in df.groupby(GENDER):
            missing_gender.append({
                "variable": col, "gender": sex,
                "n": len(group), "missing_n": group[col].isna().sum(),
                "missing_pct": group[col].isna().mean() * 100,
            })
    missing_gender = pd.DataFrame(missing_gender)

    target_gender = pd.crosstab(df[GENDER], df[TARGET], normalize="index") * 100
    target_gender.columns = [f"target_{c}_pct" for c in target_gender.columns]
    flow = pd.DataFrame({
        "stage": ["Selected rows", "Adults age 18+"],
        "n": [start_n, adult_n],
    })
    checks = pd.DataFrame([
        {"check": "Rows excluded as under 18", "value": before_adults - adult_n},
        {"check": "Adult analysis rows", "value": adult_n},
        {"check": "Duplicate respondent IDs", "value": int(df[RESPONDENT_ID].duplicated().sum())},
        {"check": "Income lowest > highest (set to missing)", "value": n_inconsistent_income},
        {"check": "Unique clusters", "value": df[CLUSTER].nunique()},
        {"check": "Missing survey weights", "value": int(df[WEIGHT].isna().sum())},
        {"check": "Non-positive survey weights", "value": int((df[WEIGHT] <= 0).sum())},
    ])

    feature_decisions = pd.DataFrame([
        {"variable": "target_bank_account", "decision": "Target; Usage=1, Non-usage=0", "reason": "FinAccess Traditional_bank variable; dictionary describes commercial bank usage excluding Mshwari/KCB M-Pesa."},
        {"variable": "target_formal_access", "decision": "Exclude from predictors; retain for audit", "reason": "Broader formal-access outcome overlaps directly with the bank-use target."},
        {"variable": "mobile_money_access", "decision": "Exclude from primary predictors; retain for sensitivity analysis", "reason": "Another financial-product-use outcome; avoids relying on adjacent financial inclusion behaviour in the primary model."},
        {"variable": "gender", "decision": "Exclude from base predictor set; retain for fairness analysis", "reason": "Sensitive attribute used for group-specific evaluation and post-processing comparison."},
        {"variable": "education_level", "decision": "Categorical; blank remains missing; Other retained", "reason": "No evidence that blank means no schooling; imputation occurs inside each training fold."},
        {"variable": "income_last_month / income_highest_month / income_lowest_month", "decision": "Convert to numeric KSh; inconsistent high/low pairs set missing", "reason": "Avoid string comparison and invalid values; impute inside model folds."},
        {"variable": "income_volatility", "decision": "Do not create", "reason": "The old 1-lowest/highest formula is a relative income range, not a validated time-volatility measure."},
        {"variable": "num_children / num_children_under5", "decision": "Retain for audit; exclude from primary model", "reason": "Missing/skip patterns are not established; do not convert missing to zero."},
        {"variable": "num_indigenous_cattle", "decision": "Retain for audit; exclude from primary model", "reason": "Very high missingness (over 80% in adult sample); #NULL! semantics are not verified."},
        {"variable": "wealth_quintile", "decision": "Categorical; not-recorded retained as category", "reason": "Avoid assuming equal numeric spacing; official wealth-index construction still needs verification."},
        {"variable": "financial_literacy_score", "decision": "Categorical; retain provisionally", "reason": "Dictionary identifies the score but does not provide its component items or construction formula."},
        {"variable": "distance_to_nearest_provider / cost_to_nearest_provider", "decision": "Categorical bands", "reason": "Survey responses are ranges; do not impose equal numeric distances or costs."},
        {"variable": "phone_internet_capable", "decision": "Categorical; distinguish not asked from not recorded", "reason": "#NULL! is not automatically interpreted as No; question is described as ability to access internet."},
        {"variable": "land_title_status", "decision": "Derived categorical feature", "reason": "No land, no ownership document, named formal document, other document and unknown are kept distinct."},
        {"variable": "survey_weight / cluster_id", "decision": "Metadata, not predictors", "reason": "Weights used for model fitting and weighted estimates; clusters used for grouped splits and bootstrap."},
    ])

    audit_path = REPORTS_DIR / "data_cleaning_audit.xlsx"
    with pd.ExcelWriter(audit_path, engine="openpyxl") as writer:
        flow.to_excel(writer, sheet_name="Sample flow", index=False)
        checks.to_excel(writer, sheet_name="Quality checks", index=False)
        missing_overall.to_excel(writer, sheet_name="Missing overall")
        missing_gender.to_excel(writer, sheet_name="Missing by gender", index=False)
        target_gender.to_excel(writer, sheet_name="Target by gender")
        pd.DataFrame({"feature": FEATURES}).to_excel(writer, sheet_name="Model features", index=False)
        df[INCOME_COLUMNS].describe().T.to_excel(writer, sheet_name="Income summary")
        feature_decisions.to_excel(writer, sheet_name="Feature decisions", index=False)

    print("CLEANING COMPLETED")
    print(f"Selected rows: {start_n:,}")
    print(f"Adults retained: {adult_n:,}")
    print(f"Under-18 rows excluded: {before_adults - adult_n:,}")
    print(f"Income inconsistent pairs set missing: {n_inconsistent_income:,}")
    print(f"Final model features: {len(FEATURES)}")
    print("Target by gender (% within gender):")
    print(target_gender.round(2).to_string())
    print(f"Saved model-ready data: {CLEAN_FILE}")
    print(f"Saved feature manifest: {FEATURES_FILE}")
    print(f"Saved audit workbook: {audit_path}")


if __name__ == "__main__":
    main()
