from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
USERS_PATH = PROJECT_ROOT / "data" / "processed" / "users_clean.csv"
USER_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "user_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "classification" / "final_gender_classifier.pkl"


def _load_training_data():
    """Build the classifier training dataset from cleaned user + feature data."""
    users = pd.read_csv(USERS_PATH)
    features = pd.read_csv(USER_FEATURES_PATH)

    train_df = features.copy()

    if "gender" not in train_df.columns:
        user_target = users[["code", "gender"]].drop_duplicates(subset="code")
        train_df = train_df.merge(user_target, on="code", how="inner")

    duplicate_target_cols = [
        column for column in train_df.columns if column.startswith("gender_")
    ]
    if duplicate_target_cols:
        train_df = train_df.drop(columns=duplicate_target_cols)

    return train_df


def train_classifier(train_df=None, target_col="gender", model_path: str | Path = MODEL_PATH):
    """Train a lightweight gender-classification model and save it to disk."""
    if train_df is None:
        train_df = _load_training_data()

    model_df = train_df.copy()

    duplicate_target_cols = [
        column for column in model_df.columns if column.startswith(f"{target_col}_")
    ]
    if duplicate_target_cols:
        model_df = model_df.drop(columns=duplicate_target_cols)

    if target_col not in model_df.columns:
        for candidate in (f"{target_col}_x", f"{target_col}_y", f"{target_col}_user"):
            if candidate in model_df.columns:
                model_df = model_df.rename(columns={candidate: target_col})
                break
        else:
            raise KeyError(target_col)

    excluded_cols = {target_col, "code", "name", "company"}
    X = model_df.drop(columns=[col for col in excluded_cols if col in model_df.columns], errors="ignore")
    y = model_df[target_col]

    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = [col for col in X.columns if col not in categorical_columns]

    transformer = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", transformer),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    _ = pipeline.score(X_valid, y_valid)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    return pipeline


__all__ = ["train_classifier"]
