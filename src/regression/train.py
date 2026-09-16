"""
Train the production flight-price regression model.

The training pipeline mirrors the validated notebook workflow:
- GroupShuffleSplit for configuration-aware train/test separation
- ColumnTransformer for numeric/categorical preprocessing
- XGBoost regression
"""

from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "flight_ml_data.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "regression"
MODEL_PATH = MODEL_DIR / "flight_price_xgboost_model.pkl"


NUMERIC_FEATURES = [
    "time",
    "distance",
    "year",
    "month",
    "day",
    "day_of_week",
    "is_weekend",
]

CATEGORICAL_FEATURES = [
    "from",
    "to",
    "flightType",
    "agency",
]


def train_model():
    """Train, evaluate, and save the flight-price model."""

    df = pd.read_csv(DATA_PATH)

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X = df[feature_columns]
    y = df["price"]

    # Configuration-aware grouping used by the validated notebook.
    groups = (
        df[
            [
                "from",
                "to",
                "flightType",
                "time",
                "distance",
                "agency",
                "year",
                "month",
                "day",
                "day_of_week",
                "is_weekend",
            ]
        ]
        .astype(str)
        .agg("_".join, axis=1)
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print("=" * 60)
    print("Flight Price Regression Training Complete")
    print("=" * 60)
    print(f"Training rows : {len(X_train)}")
    print(f"Testing rows  : {len(X_test)}")
    print(f"MAE           : {mae:.4f}")
    print(f"RMSE          : {rmse:.4f}")
    print(f"R2            : {r2:.6f}")
    print(f"Model saved   : {MODEL_PATH}")
    print("=" * 60)

    return pipeline, {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


if __name__ == "__main__":
    train_model()