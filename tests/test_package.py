"""Model-package contract tests (course ch. 10). Skipped until packaged."""
import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import PACKAGE_FILE, SAMPLE_DIR, MODEL_VERSION, GENDER
from src.predict import predict_frame

pytestmark = pytest.mark.skipif(
    not PACKAGE_FILE.exists(), reason="Run python -m src.run_pipeline first."
)


@pytest.fixture(scope="module")
def package():
    return joblib.load(PACKAGE_FILE)


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(SAMPLE_DIR / "sample_input.csv")


def test_package_has_required_parts(package):
    for key in ["model", "features", "threshold", "input_schema", "metadata"]:
        assert key in package
    meta = package["metadata"]
    for key in ["model_version", "git_commit", "selected_data_sha256",
                "model_ready_data_sha256", "created_at_utc", "scikit_learn"]:
        assert meta.get(key)
    assert meta["model_version"] == MODEL_VERSION
    assert 0 < package["threshold"] < 1


def test_gender_is_not_a_model_input(package):
    assert GENDER not in package["features"]
    assert package["metadata"]["sensitive_attribute_used_as_input"] is False


def test_schema_covers_every_feature(package):
    schema = package["input_schema"]
    assert set(schema["numeric"]) | set(schema["categorical"]) == set(package["features"])


def test_reloaded_package_reproduces_expected_predictions(package, sample):
    expected = pd.read_csv(SAMPLE_DIR / "sample_expected_predictions.csv")
    result = predict_frame(sample, package)
    np.testing.assert_allclose(result["probability"], expected["probability"], rtol=0, atol=1e-12)
    assert (result["decision"].to_numpy() == expected["decision"].to_numpy()).all()


def test_outputs_are_valid_probabilities(package, sample):
    result = predict_frame(sample, package)
    assert result["probability"].between(0, 1).all()
    assert set(result["decision"].unique()) <= {0, 1}


def test_missing_feature_is_rejected(package, sample):
    with pytest.raises(ValueError, match="missing required feature"):
        predict_frame(sample.drop(columns=["age"]), package)


def test_predictions_do_not_depend_on_row_order(package, sample):
    forward = predict_frame(sample, package)["probability"].to_numpy()
    backward = predict_frame(sample.iloc[::-1], package)["probability"].to_numpy()[::-1]
    np.testing.assert_allclose(forward, backward, atol=1e-12)


def test_unseen_category_warns_but_still_scores(package, sample):
    changed = sample.copy()
    changed.loc[changed.index[0], "county"] = "Not a real county"
    with pytest.warns(UserWarning, match="unseen categories"):
        result = predict_frame(changed, package)
    assert result["probability"].notna().all()
