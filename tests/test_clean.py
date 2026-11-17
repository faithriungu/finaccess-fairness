"""Unit tests for the cleaning rules in src/clean.py (no data needed)."""
import numpy as np
import pandas as pd

from src import clean
from src.config import GENDER, TARGET, WEIGHT, CLUSTER, RESPONDENT_ID


def test_yes_no_keeps_non_answers_missing():
    out = clean.yes_no(pd.Series(["Yes", "no", "Refused", None, "#NULL!"]))
    assert out.iloc[0] == 1 and out.iloc[1] == 0
    assert out.iloc[2:].isna().all()


def test_null_codes_are_not_recoded_to_zero():
    assert clean.simplify_network("#NULL!") == "Not recorded / skipped"
    assert clean.simplify_frequency("#NULL!") == "Not recorded / skipped"
    assert clean.simplify_provider("#NULL!") == "Not recorded / skipped"


def test_land_title_keeps_states_distinct():
    assert clean.derive_land_title({"owns_land": 0, "land_ownership_document": None}) == "Does not own land"
    assert clean.derive_land_title({"owns_land": np.nan, "land_ownership_document": None}) == "Land ownership unknown"
    assert clean.derive_land_title({"owns_land": 1, "land_ownership_document": "Title deed"}) == "Formal ownership document"
    assert clean.derive_land_title({"owns_land": 1, "land_ownership_document": "No ownership document"}) == "No ownership document"


def test_feature_list_excludes_sensitive_and_leaky_columns():
    forbidden = {GENDER, "hh_head_gender", TARGET, "target_formal_access",
                 "mobile_money_access", WEIGHT, CLUSTER, RESPONDENT_ID, "income_volatility"}
    assert not set(clean.FEATURES) & forbidden
    assert len(clean.FEATURES) == len(set(clean.FEATURES)) == 35
