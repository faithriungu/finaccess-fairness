"""Integrity checks on pipeline outputs. Skipped until the pipeline has run."""
import json

import pandas as pd
import pytest

from src.config import (
    CLEAN_FILE, MODELS_DIR, FEATURES_FILE, CLUSTER, RESPONDENT_ID, TARGET,
)

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "split.csv").exists() or not CLEAN_FILE.exists(),
    reason="Run python -m src.run_pipeline first.",
)


@pytest.fixture(scope="module")
def data_with_split():
    df = pd.read_csv(CLEAN_FILE, low_memory=False)
    df[RESPONDENT_ID] = df[RESPONDENT_ID].astype(str)
    split = pd.read_csv(MODELS_DIR / "split.csv", dtype={RESPONDENT_ID: str})
    return df.merge(split, on=RESPONDENT_ID, how="inner", validate="one_to_one")


def test_adults_only_and_binary_target(data_with_split):
    assert data_with_split["age"].min() >= 18
    assert set(data_with_split[TARGET].unique()) == {0, 1}


def test_train_and_test_clusters_are_disjoint(data_with_split):
    train = set(data_with_split.loc[data_with_split["split"] == "train", CLUSTER])
    test = set(data_with_split.loc[data_with_split["split"] == "test", CLUSTER])
    assert train and test and not (train & test)


def test_every_adult_is_in_exactly_one_split(data_with_split):
    clean = pd.read_csv(CLEAN_FILE, usecols=[RESPONDENT_ID])
    assert len(data_with_split) == len(clean)


def test_feature_manifest_matches_cleaning_code():
    from src.clean import FEATURES
    manifest = json.loads(FEATURES_FILE.read_text(encoding="utf-8"))
    assert manifest["final_features"] == FEATURES
