from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "regression" / "flight_price_xgboost_model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "flight_ml_data.csv"


def _resolve_input_dataframe(df=None, **kwargs):
    """Return a DataFrame from a DataFrame, a dictionary, or the processed dataset."""
    if df is None and not kwargs:
        return pd.read_csv(DATA_PATH)

    if isinstance(df, pd.DataFrame):
        return df.copy()

    if df is None:
        record = {}
    elif isinstance(df, dict):
        record = df.copy()
    else:
        raise TypeError("df must be a pandas DataFrame, dict, or None.")

    record.update(kwargs)

    field_map = {
        "from_city": "from",
        "to_city": "to",
        "flight_type": "flightType",
        "travel_code": "travelCode",
        "user_code": "userCode",
        "price": "price",
        "time": "time",
        "distance": "distance",
        "agency": "agency",
        "year": "year",
        "month": "month",
        "day": "day",
        "day_of_week": "day_of_week",
        "is_weekend": "is_weekend",
    }

    normalized = {}
    for key, value in record.items():
        target_key = field_map.get(key, key)
        normalized[target_key] = value

    return pd.DataFrame([normalized])


def predict_price(df=None, model_path: str | Path = MODEL_PATH, **kwargs):
    """Load a flight price model and predict prices from a DataFrame or direct row inputs."""
    features = _resolve_input_dataframe(df=df, **kwargs)
    model = joblib.load(model_path)

    feature_columns = [
        column for column in features.columns if column not in {"price", "date"}
    ]
    feature_frame = features[feature_columns].copy()

    predictions = model.predict(feature_frame)
    return predictions[0] if len(predictions) == 1 else predictions


__all__ = ["predict_price"]
