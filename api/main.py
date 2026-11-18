"""FastAPI service for the traditional-bank-use model.

Run locally (from the project root, venv active):
    uvicorn api.main:app --reload

Endpoints
    GET  /health         process is up (liveness)
    GET  /ready          model package loaded and usable (readiness)
    GET  /metadata       model version, threshold, data hashes, test metrics
    POST /predict        score one respondent
    POST /predict/batch  score up to 500 respondents

Design notes (course ch. 10):
- The request schema (api/schema.py) is generated from the model package and
  forbids unknown fields; invalid input is rejected with 422.
- The package is loaded once, at startup, in the lifespan handler.
- Scoring is CPU-bound, so endpoints are plain `def`, run in a thread pool.
- Errors fail closed: if the model is not loaded or scoring fails, the API
  returns 503/500 and never a default decision.
- Each response has a request_id and reason codes (top SHAP contributions),
  supporting the right to an explanation of automated decisions under the
  Kenya Data Protection Act 2019. Inputs are not logged.
"""
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.schema import FEATURES, MODEL_VERSION, TRAINING_RANGES, BatchRequest, Respondent

logger = logging.getLogger("bank_use_api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model_package.joblib"
N_REASONS = 3

STATE: dict = {"package": None, "explainer": None, "feature_map": None, "error": None}


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ReasonCode(BaseModel):
    feature: str
    effect: Literal["raises", "lowers"]
    contribution: float  # SHAP value on the model's log-odds scale


class Prediction(BaseModel):
    request_id: str
    model_version: str
    probability: float
    decision: int
    decision_label: Literal["likely bank user", "unlikely bank user"]
    threshold: float
    reason_codes: list[ReasonCode]
    warnings: list[str]


class BatchPrediction(BaseModel):
    request_id: str
    model_version: str
    threshold: float
    n: int
    predictions: list[Prediction]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _map_encoded_to_original(preprocessor, features):
    """Map each encoded column back to the original predictor it came from."""
    names = [str(n) for n in preprocessor.get_feature_names_out()]
    mapping = []
    for name in names:
        if name.startswith("missingindicator_"):
            name = name.replace("missingindicator_", "", 1)
        if name in features:
            mapping.append(name)
            continue
        matches = [f for f in features if name.startswith(f + "_")]
        if not matches:
            raise ValueError(f"Cannot map encoded column '{name}' to a predictor.")
        mapping.append(max(matches, key=len))
    return np.array(mapping, dtype=object)


def load_model(path: Path) -> None:
    """Load the package and SHAP explainer once. Raises on any problem."""
    package = joblib.load(path)
    if package["features"] != FEATURES:
        raise RuntimeError("api/schema.py does not match the model package. "
                           "Run: python -m api.generate_schema")
    if package["metadata"]["model_version"] != MODEL_VERSION:
        raise RuntimeError("Schema and package model versions differ.")

    import shap  # imported here so /health works even if SHAP is unavailable

    preprocessor = package["model"].named_steps["preprocess"]
    tree_model = package["model"].named_steps["model"]
    explainer = shap.TreeExplainer(tree_model, feature_perturbation="tree_path_dependent",
                                   model_output="raw")
    STATE.update(package=package, explainer=explainer,
                 feature_map=_map_encoded_to_original(preprocessor, FEATURES), error=None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    try:
        load_model(path)
        meta = STATE["package"]["metadata"]
        logger.info("Model loaded: version=%s threshold=%.3f path=%s",
                    meta["model_version"], STATE["package"]["threshold"], path)
    except Exception as exc:  # fail closed: stay up but not ready
        STATE.update(package=None, explainer=None, feature_map=None, error=str(exc))
        logger.error("Model failed to load from %s: %s", path, exc)
    yield
    STATE.update(package=None, explainer=None, feature_map=None)


app = FastAPI(
    title="FinAccess traditional bank-use API",
    description=("Predicts whether a Kenyan adult uses a traditional bank, from 35 survey "
                 "characteristics. Gender is not an input. DSA 8401 capstone."),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.exception_handler(Exception)
def unhandled_error(request: Request, exc: Exception):
    error_id = str(uuid.uuid4())
    logger.exception("Unhandled error %s on %s", error_id, request.url.path)
    return JSONResponse(status_code=500,
                        content={"detail": "Internal error; no prediction was made.",
                                 "error_id": error_id})


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _require_model():
    if STATE["package"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded; no prediction was made.")
    return STATE["package"]


def _range_warnings(record: dict) -> list[str]:
    warnings = []
    for col, (lo, hi) in TRAINING_RANGES.items():
        value = record.get(col)
        if value is not None and not lo <= value <= hi:
            warnings.append(f"{col}={value:g} is outside the training range [{lo:g}, {hi:g}]; "
                            "treat this prediction with extra caution.")
    return warnings


def _score(records: list[Respondent], request_id: str) -> list[Prediction]:
    package = _require_model()
    rows = [r.model_dump() for r in records]
    frame = pd.DataFrame(rows, columns=FEATURES)
    for col in package["numeric_features"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in package["categorical_features"]:
        frame[col] = frame[col].astype(object).where(frame[col].notna(), np.nan)

    probability = package["model"].predict_proba(frame)[:, 1]
    if not np.all(np.isfinite(probability)):
        raise RuntimeError("Model returned a non-finite probability.")

    # Reason codes: SHAP on the encoded inputs, summed back to the 35 predictors.
    encoded = package["model"].named_steps["preprocess"].transform(frame)
    shap_values = np.asarray(STATE["explainer"].shap_values(encoded, check_additivity=False))
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 1]
    per_feature = np.zeros((len(frame), len(FEATURES)))
    for j, feature in enumerate(FEATURES):
        cols = STATE["feature_map"] == feature
        if cols.any():
            per_feature[:, j] = shap_values[:, cols].sum(axis=1)

    threshold = float(package["threshold"])
    version = package["metadata"]["model_version"]
    results = []
    for i, row in enumerate(rows):
        top = np.argsort(-np.abs(per_feature[i]))[:N_REASONS]
        reasons = [ReasonCode(feature=FEATURES[j],
                              effect="raises" if per_feature[i, j] > 0 else "lowers",
                              contribution=round(float(per_feature[i, j]), 4))
                   for j in top]
        decision = int(probability[i] >= threshold)
        results.append(Prediction(
            request_id=request_id, model_version=version,
            probability=round(float(probability[i]), 6), decision=decision,
            decision_label="likely bank user" if decision else "unlikely bank user",
            threshold=threshold, reason_codes=reasons, warnings=_range_warnings(row),
        ))
    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health():
    """Liveness: the process is running."""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready():
    """Readiness: the model is loaded and the service can score requests."""
    if STATE["package"] is None:
        raise HTTPException(status_code=503,
                            detail=f"Model not loaded: {STATE['error'] or 'starting up'}")
    return {"status": "ready", "model_version": STATE["package"]["metadata"]["model_version"]}


@app.get("/metadata", tags=["ops"])
def metadata():
    """Model card essentials: version, threshold rule, data hashes, metrics."""
    package = _require_model()
    return {
        **package["metadata"],
        "threshold": package["threshold"],
        "threshold_rule": package["threshold_rule"],
        "features": package["features"],
        "output": package["output_schema"],
        "reason_codes": "Top 3 SHAP contributions (log-odds scale) per prediction; "
                        "they explain the model, not causes.",
    }


@app.post("/predict", response_model=Prediction, tags=["scoring"])
def predict(respondent: Respondent):
    """Score one respondent."""
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    result = _score([respondent], request_id)[0]
    logger.info("predict request_id=%s decision=%d latency_ms=%.1f",
                request_id, result.decision, (time.perf_counter() - start) * 1000)
    return result


@app.post("/predict/batch", response_model=BatchPrediction, tags=["scoring"])
def predict_batch(batch: BatchRequest):
    """Score up to 500 respondents in one call."""
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    results = _score(batch.records, request_id)
    logger.info("predict_batch request_id=%s n=%d latency_ms=%.1f",
                request_id, len(results), (time.perf_counter() - start) * 1000)
    package = STATE["package"]
    return BatchPrediction(request_id=request_id,
                           model_version=package["metadata"]["model_version"],
                           threshold=float(package["threshold"]),
                           n=len(results), predictions=results)
