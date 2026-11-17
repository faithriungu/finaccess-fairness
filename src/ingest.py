"""Step 1 - ingestion: produce and validate the 48-column selected extract.

Run from the project root:
    python -m src.ingest                  # validate data/processed/finaccess_selected.csv
    python -m src.ingest --raw PATH.xlsx  # rebuild the extract from the raw workbook

Two routes, because the raw FinAccess 2024 workbook is not redistributed
with this repository:

1. Raw route (--raw). Reads the original FinAccess 2024 household workbook,
   keeps the 48 study variables listed in config/raw_variable_map.csv, renames
   them to the project names and writes data/processed/finaccess_selected.csv.
   Every row of the map must have its raw_name filled in.
2. Validation route (default). If the extract already exists, checks it
   against the expected schema and row count.

Both routes finish by writing reports/data_manifest.json with the SHA-256
hash, shape and column list of the extract. The hash is copied into the model
package so a model can always be traced to the exact data that trained it.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from src.config import (
    SELECTED_FILE, DATA_MANIFEST, RAW_VARIABLE_MAP,
    EXPECTED_SELECTED_COLUMNS, EXPECTED_SELECTED_ROWS, RESPONDENT_ID,
)


def sha256_file(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_from_raw(raw_path, force=False):
    """Select and rename the study variables from the raw workbook."""
    if not RAW_VARIABLE_MAP.exists():
        raise FileNotFoundError(f"Variable map not found: {RAW_VARIABLE_MAP}")
    mapping = pd.read_csv(RAW_VARIABLE_MAP, dtype=str).fillna("")
    unmapped = mapping.loc[mapping["raw_name"].str.strip().eq(""), "project_name"].tolist()
    if unmapped:
        raise ValueError(
            "config/raw_variable_map.csv has no raw_name for: "
            + ", ".join(unmapped)
            + ". Fill these in from the FinAccess 2024 codebook before using --raw."
        )
    if SELECTED_FILE.exists() and not force:
        raise FileExistsError(
            f"{SELECTED_FILE} already exists. Re-run with --force to overwrite it."
        )

    raw_names = mapping["raw_name"].str.strip().tolist()
    print(f"Reading raw workbook (this can take a few minutes): {raw_path}")
    raw = pd.read_excel(raw_path, usecols=raw_names, dtype=str)
    rename = dict(zip(raw_names, mapping["project_name"].str.strip()))
    selected = raw.rename(columns=rename)[EXPECTED_SELECTED_COLUMNS]
    SELECTED_FILE.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(SELECTED_FILE, index=False)
    print(f"Wrote {SELECTED_FILE} with shape {selected.shape}")


def validate_selected():
    if not SELECTED_FILE.exists():
        raise FileNotFoundError(
            f"{SELECTED_FILE} not found.\n"
            "Either place the 48-column extract there, or rebuild it with\n"
            "    python -m src.ingest --raw <path to FinAccess 2024 workbook>"
        )
    df = pd.read_csv(SELECTED_FILE, low_memory=False, encoding="utf-8-sig")

    missing = [c for c in EXPECTED_SELECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_SELECTED_COLUMNS]
    if missing or extra:
        raise ValueError(f"Schema mismatch. Missing: {missing}; unexpected: {extra}")
    if len(df) != EXPECTED_SELECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SELECTED_ROWS:,} rows, found {len(df):,}. "
            "The extract differs from the one this project was built on."
        )
    if df[RESPONDENT_ID].duplicated().any():
        raise ValueError("Duplicate respondent IDs found.")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", help="Path to the raw FinAccess 2024 workbook (.xlsx)")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing extract")
    args = parser.parse_args()

    if args.raw:
        build_from_raw(args.raw, force=args.force)

    df = validate_selected()
    manifest = {
        "file": str(SELECTED_FILE.relative_to(SELECTED_FILE.parents[2])),
        "sha256": sha256_file(SELECTED_FILE),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "source": "raw workbook via --raw" if args.raw else "existing extract",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    DATA_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("INGESTION / VALIDATION COMPLETED")
    print(f"Extract: {SELECTED_FILE}")
    print(f"Shape: {df.shape}")
    print(f"SHA-256: {manifest['sha256']}")
    print(f"Saved data manifest: {DATA_MANIFEST}")


if __name__ == "__main__":
    main()
