"""Batch prediction with the deployable model package.

Run from the project root:
    python -m src.predict --input data/sample/sample_input.csv --output reports/sample_predictions.csv

The input CSV needs the 35 model features (column names as in
reports/selected_features.json). Extra columns are ignored, except that an
identifier column (default respondent_id) is carried through to the output.
Gender, survey weight and the target are never used for scoring.
"""
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd

from src.config import PACKAGE_FILE, RESPONDENT_ID


def load_package(path=PACKAGE_FILE):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run python -m src.run_pipeline first.")
    return joblib.load(path)


def check_inputs(frame, package):
    """Fail on missing columns; warn on values the training data never contained."""
    schema = package["input_schema"]
    missing = [c for c in package["features"] if c not in frame.columns]
    if missing:
        raise ValueError(f"Input is missing required feature columns: {missing}")

    issues = []
    for col, spec in schema["numeric"].items():
        values = pd.to_numeric(frame[col], errors="coerce")
        bad_type = frame[col].notna() & values.isna()
        if bad_type.any():
            raise ValueError(f"Non-numeric values in numeric column '{col}'.")
        outside = values.notna() & ((values < spec["min"]) | (values > spec["max"]))
        if outside.any():
            issues.append(f"{col}: {int(outside.sum())} value(s) outside training range "
                          f"[{spec['min']:g}, {spec['max']:g}]")
    for col, spec in schema["categorical"].items():
        observed = frame[col].dropna().astype(str)
        unknown = sorted(set(observed) - set(spec["allowed"]))
        if unknown:
            issues.append(f"{col}: unseen categories {unknown[:5]} (scored as infrequent)")
    for message in issues:
        warnings.warn(message, stacklevel=2)
    return issues


def predict_frame(frame, package):
    check_inputs(frame, package)
    x = frame[package["features"]].copy()
    for col in package["numeric_features"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    for col in package["categorical_features"]:
        x[col] = x[col].astype(object).where(x[col].notna(), np.nan)
    probability = package["model"].predict_proba(x)[:, 1]
    return pd.DataFrame({
        "probability": probability,
        "decision": (probability >= package["threshold"]).astype(int),
        "threshold": package["threshold"],
        "model_version": package["metadata"]["model_version"],
    }, index=frame.index)


def main():
    parser = argparse.ArgumentParser(description="Score respondents with the model package.")
    parser.add_argument("--input", required=True, help="CSV of respondents to score")
    parser.add_argument("--output", required=True, help="Where to write predictions (CSV)")
    parser.add_argument("--id-column", default=RESPONDENT_ID,
                        help="Identifier column to carry through, if present")
    args = parser.parse_args()

    package = load_package()
    frame = pd.read_csv(args.input, low_memory=False)
    result = predict_frame(frame, package)
    if args.id_column in frame.columns:
        result.insert(0, args.id_column, frame[args.id_column].to_numpy())
    result.to_csv(args.output, index=False)

    print(f"Scored {len(result):,} rows with model {package['metadata']['model_version']} "
          f"(threshold {package['threshold']:.3f}).")
    print(f"Predicted bank users: {int(result['decision'].sum()):,} of {len(result):,}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
