"""Shared project paths and modelling settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATA_DIR = ROOT / "data" / "processed"
SAMPLE_DIR = ROOT / "data" / "sample"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Optional raw-source route (see src/ingest.py). The raw FinAccess workbook is
# not redistributed with this repository.
RAW_VARIABLE_MAP = ROOT / "config" / "raw_variable_map.csv"

SELECTED_FILE = DATA_DIR / "finaccess_selected.csv"
CLEAN_FILE = DATA_DIR / "finaccess_model_ready.csv"
FEATURES_FILE = REPORTS_DIR / "selected_features.json"
DATA_MANIFEST = REPORTS_DIR / "data_manifest.json"

# Training output (model only) and the deployable package (model + threshold +
# schema + metadata) written by src/package.py.
TRAINED_MODEL_FILE = MODELS_DIR / "bank_model.joblib"
PACKAGE_FILE = MODELS_DIR / "model_package.joblib"
MODEL_VERSION = "1.0.0"

TARGET = "target_bank_account"
GENDER = "gender"
WEIGHT = "survey_weight"
CLUSTER = "cluster_id"
RESPONDENT_ID = "respondent_id"
LOCATION = "location_type"
RANDOM_STATE = 42
TEST_FRACTION = 0.20
OUTER_FOLDS = 5
INNER_FOLDS = 3
N_JOBS = 2
N_BOOT = 1000

# Expected shape of the selected extract (validated by src/ingest.py).
EXPECTED_SELECTED_ROWS = 20871
EXPECTED_SELECTED_COLUMNS = [
    "respondent_id", "county", "cluster_id", "location_type", "hh_head_gender",
    "age", "relationship_to_hh_head", "gender", "household_size", "num_children",
    "num_children_under5", "decides_daily_spending", "decides_major_spending",
    "skipped_meals_last_month", "can_read_sms", "income_from_farming",
    "income_from_regular_job", "income_from_casual_work",
    "income_from_own_business", "relies_on_family_support", "income_last_month",
    "income_highest_month", "income_lowest_month", "phone_access_type",
    "phone_internet_capable", "internet_use_frequency", "network_quality",
    "nearest_provider_type", "cost_to_nearest_provider",
    "distance_to_nearest_provider", "num_indigenous_cattle", "owns_house",
    "owns_land", "land_ownership_document", "survey_weight",
    "livelihood_category", "wealth_quintile", "education_level",
    "marital_status", "has_any_id", "owns_phone", "mobile_money_access",
    "num_income_sources", "target_formal_access", "target_bank_account",
    "experienced_shock", "financial_literacy_score", "disability_status",
]

for folder in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, SAMPLE_DIR):
    folder.mkdir(parents=True, exist_ok=True)
