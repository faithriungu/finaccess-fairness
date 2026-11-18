"""API contract tests (course ch. 10) using FastAPI's TestClient.

Skipped until the model package exists (python -m src.run_pipeline).
"""
import copy
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import PACKAGE_FILE, SAMPLE_DIR

pytestmark = pytest.mark.skipif(not PACKAGE_FILE.exists(),
                                reason="Run python -m src.run_pipeline first.")

EXAMPLES = PACKAGE_FILE.parents[1] / "api" / "examples"


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as test_client:  # runs the lifespan (model load)
        yield test_client


@pytest.fixture
def record():
    return json.loads((EXAMPLES / "predict_request.json").read_text(encoding="utf-8"))


# --- schema stays in step with the model package -----------------------------

def test_generated_schema_matches_package():
    from api.generate_schema import render
    from api import schema
    package = joblib.load(PACKAGE_FILE)
    on_disk = (PACKAGE_FILE.parents[1] / "api" / "schema.py").read_text(encoding="utf-8")
    assert on_disk == render(package), "Run: python -m api.generate_schema"
    assert schema.FEATURES == package["features"]


def test_gender_is_not_an_api_field():
    from api.schema import Respondent
    assert "gender" not in Respondent.model_fields


# --- ops endpoints -------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_ready(client):
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["status"] == "ready"


def test_metadata(client):
    body = client.get("/metadata").json()
    for key in ["model_version", "threshold", "git_commit", "selected_data_sha256", "features"]:
        assert key in body
    assert body["sensitive_attribute_used_as_input"] is False


# --- /predict happy path ------------------------------------------------------

def test_predict_response_contract(client, record):
    r = client.post("/predict", json=record)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"request_id", "model_version", "probability", "decision",
                         "decision_label", "threshold", "reason_codes", "warnings"}
    assert 0 <= body["probability"] <= 1
    assert body["decision"] == int(body["probability"] >= body["threshold"])
    assert len(body["reason_codes"]) == 3


def test_api_matches_packaged_predictions(client):
    """Same respondents, same probabilities as the offline package."""
    batch = json.loads((EXAMPLES / "batch_request.json").read_text(encoding="utf-8"))
    expected = pd.read_csv(SAMPLE_DIR / "sample_expected_predictions.csv").head(len(batch["records"]))
    body = client.post("/predict/batch", json=batch).json()
    got = np.array([p["probability"] for p in body["predictions"]])
    np.testing.assert_allclose(got, expected["probability"], atol=1e-6)
    assert [p["decision"] for p in body["predictions"]] == expected["decision"].tolist()


def test_reason_codes_add_up_to_the_score(client, record):
    """SHAP contributions of all 35 predictors + base value = model log-odds."""
    from api.main import STATE, _score
    from api.schema import Respondent
    package = STATE["package"]
    pred = _score([Respondent(**record)], "test")[0]
    frame = pd.DataFrame([record])
    encoded = package["model"].named_steps["preprocess"].transform(frame[package["features"]])
    contributions = np.asarray(STATE["explainer"].shap_values(encoded, check_additivity=False)).sum()
    log_odds = np.log(pred.probability / (1 - pred.probability))
    assert abs(STATE["explainer"].expected_value + contributions - log_odds) < 1e-3


def test_nulls_are_accepted(client, record):
    record = {k: None for k in record}
    r = client.post("/predict", json=record)
    assert r.status_code == 200


def test_out_of_training_range_value_warns(client, record):
    record["income_last_month"] = 5_000_000
    body = client.post("/predict", json=record).json()
    assert any("income_last_month" in w for w in body["warnings"])


# --- invalid input is rejected with 422 ---------------------------------------

def test_missing_field_rejected(client, record):
    del record["age"]
    assert client.post("/predict", json=record).status_code == 422


def test_unknown_field_rejected(client, record):
    record["gender"] = "Female"
    assert client.post("/predict", json=record).status_code == 422


def test_unknown_category_rejected(client, record):
    record["county"] = "Atlantis"
    assert client.post("/predict", json=record).status_code == 422


@pytest.mark.parametrize("field,value", [("age", 12), ("age", 150),
                                         ("income_last_month", -5), ("has_any_id", 2)])
def test_out_of_bounds_rejected(client, record, field, value):
    record[field] = value
    assert client.post("/predict", json=record).status_code == 422


def test_wrong_type_rejected(client, record):
    record["age"] = "twenty"
    assert client.post("/predict", json=record).status_code == 422


def test_batch_limits(client, record):
    assert client.post("/predict/batch", json={"records": []}).status_code == 422
    too_many = {"records": [copy.deepcopy(record) for _ in range(501)]}
    assert client.post("/predict/batch", json=too_many).status_code == 422


# --- fail closed -------------------------------------------------------------

def test_fails_closed_when_model_missing(monkeypatch, tmp_path, record):
    from fastapi.testclient import TestClient
    import api.main as main
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    try:
        with TestClient(main.app) as broken:
            assert broken.get("/health").status_code == 200
            assert broken.get("/ready").status_code == 503
            r = broken.post("/predict", json=record)
            assert r.status_code == 503 and "probability" not in r.json()
    finally:
        main.load_model(main.DEFAULT_MODEL_PATH)  # restore for any later tests
