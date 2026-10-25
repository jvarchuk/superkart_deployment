"""SuperKart sales-prediction API (Flask).

Endpoints
  GET  /                  -> welcome message
  GET  /health            -> liveness / model-loaded check
  POST /v1/predict        -> single prediction (JSON body)
  POST /v1/predictbatch   -> batch prediction (multipart CSV upload, field name "file");
                             rows that fail validation are returned as null and listed in the X-Skipped-Rows header
"""
import logging
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("superkart")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "superkart_model.joblib"))
# Must match the reference year used in training (Store_Age_Years = 2025 - establishment year)
REFERENCE_YEAR = int(os.environ.get("REFERENCE_YEAR", "2025"))
MAX_BATCH_ROWS = int(os.environ.get("MAX_BATCH_ROWS", "50000"))

# Exact feature set (and order) the model pipeline was trained on
FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]

# numeric column -> (lower bound, lower bound inclusive?, upper bound or None)
NUMERIC_RULES = {
    "Product_Weight": (0, False, None),
    "Product_Allocated_Area": (0, False, 1),
    "Product_MRP": (0, False, None),
    "Store_Age_Years": (0, True, None),
}

# Only categories seen in training are accepted (the encoder would otherwise silently ignore them)
ALLOWED = {
    "Product_Sugar_Content": ["Low Sugar", "Regular", "No Sugar"],
    "Store_Size": ["Small", "Medium", "High"],
    "Store_Location_City_Type": ["Tier 1", "Tier 2", "Tier 3"],
    "Store_Type": ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"],
    "Product_Id_char": ["FD", "DR", "NC"],
    "Product_Type_Category": ["Perishables", "Non Perishables"],
}

PERISHABLES = {"Dairy", "Meat", "Fruits and Vegetables", "Breakfast", "Breads", "Seafood"}


class ValidationError(Exception):
    """Raised for bad client input (HTTP 400)."""


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering used in training and validate the result."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # --- same cleaning / feature engineering as the notebook ---
    if "Product_Sugar_Content" in df:
        df["Product_Sugar_Content"] = (
            df["Product_Sugar_Content"].astype(str).str.strip().replace({"reg": "Regular"})
        )
    if "Product_Id_char" not in df and "Product_Id" in df:
        df["Product_Id_char"] = df["Product_Id"].astype(str).str.strip().str[:2].str.upper()
    if "Product_Type_Category" not in df and "Product_Type" in df:
        df["Product_Type_Category"] = df["Product_Type"].apply(
            lambda x: "Perishables" if str(x).strip() in PERISHABLES else "Non Perishables"
        )
    if "Store_Age_Years" not in df and "Store_Establishment_Year" in df:
        df["Store_Age_Years"] = REFERENCE_YEAR - pd.to_numeric(
            df["Store_Establishment_Year"], errors="coerce"
        )

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing required field(s): {missing}")
    df = df[FEATURES].copy()

    problems = []
    for col, (lo, lo_inclusive, hi) in NUMERIC_RULES.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = df[col].isna() | ((df[col] < lo) if lo_inclusive else (df[col] <= lo))
        if hi is not None:
            bad |= df[col] > hi
        if bad.any():
            rows = df.index[bad].tolist()[:5]
            problems.append(f"{col} has missing/out-of-range values (rows {rows})")

    for col, allowed in ALLOWED.items():
        df[col] = df[col].astype(str).str.strip()
        bad = ~df[col].isin(allowed)
        if bad.any():
            values = df.loc[bad, col].unique().tolist()[:5]
            problems.append(f"{col} has unsupported value(s) {values}; allowed: {allowed}")

    if problems:
        raise ValidationError("; ".join(problems))
    return df


# ------------------------------------------------------------------ app
superkart_api = Flask("SuperKart")
superkart_api.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload cap

model = joblib.load(MODEL_PATH)
log.info("Model loaded from %s", MODEL_PATH)


@superkart_api.errorhandler(ValidationError)
def handle_validation_error(err):
    return jsonify({"error": str(err)}), 400


@superkart_api.errorhandler(HTTPException)
def handle_http_error(err):
    return jsonify({"error": err.description}), err.code


@superkart_api.errorhandler(Exception)
def handle_unexpected_error(err):
    log.exception("Unhandled error")
    return jsonify({"error": "Internal server error"}), 500


@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart System"


@superkart_api.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@superkart_api.post("/v1/predict")
def predict_sales():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValidationError("Request body must be a JSON object.")
    input_data = prepare_features(pd.DataFrame([payload]))
    prediction = float(model.predict(input_data)[0])
    return jsonify({"Sales": round(prediction, 2)})


def score_batch(raw: pd.DataFrame):
    """Score a DataFrame row-safely.

    Returns (results, skipped): results maps row index -> predicted sales (None if the
    row was skipped); skipped lists the row indexes rejected by validation.
    A bad row (e.g. a store type never seen in training) does not block the other rows;
    a file-level problem (e.g. a missing column) still raises ValidationError.
    """
    n = len(raw)
    try:
        valid = list(range(n))
        input_data = prepare_features(raw)
    except ValidationError as first_error:
        valid = []
        for i in range(n):
            try:
                prepare_features(raw.iloc[[i]])
                valid.append(i)
            except ValidationError:
                pass
        if not valid:
            raise first_error
        input_data = prepare_features(raw.iloc[valid])
    predictions = model.predict(input_data).tolist()
    results = {str(i): None for i in range(n)}
    for i, p in zip(valid, predictions):
        results[str(i)] = round(float(p), 2)
    skipped = [i for i in range(n) if results[str(i)] is None]
    return results, skipped


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    if "file" not in request.files:
        raise ValidationError("Upload a CSV file using the form field name 'file'.")
    try:
        raw = pd.read_csv(request.files["file"])
    except Exception:
        raise ValidationError("Could not read the uploaded file as CSV.")
    if raw.empty:
        raise ValidationError("The uploaded CSV has no rows.")
    if len(raw) > MAX_BATCH_ROWS:
        raise ValidationError(f"Too many rows ({len(raw)}); the limit is {MAX_BATCH_ROWS}.")

    results, skipped = score_batch(raw)
    # {"0": 3297.8, "4": null, ...}  row index -> predicted sales (null = row skipped)
    response = jsonify(results)
    if skipped:
        response.headers["X-Skipped-Rows"] = ",".join(str(i) for i in skipped[:200])
    return response


# Local development only. In Docker the app is served by Gunicorn.
if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
