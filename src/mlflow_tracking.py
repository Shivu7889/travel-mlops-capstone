from pathlib import Path

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGRESSION_MODEL = (
    PROJECT_ROOT / "models" / "regression" / "flight_price_xgboost_model.pkl"
)
CLASSIFICATION_MODEL = (
    PROJECT_ROOT / "models" / "classification" / "final_gender_classifier.pkl"
)
RECOMMENDATION_MODEL = (
    PROJECT_ROOT / "models" / "recommendation" / "hybrid_hotel_recommendation.pkl"
)

MLRUNS_URI = f"file://{PROJECT_ROOT / 'mlruns'}"
mlflow.set_tracking_uri(MLRUNS_URI)


def _log_model_artifact(model_path: Path, experiment_name: str, run_name: str, params: dict, metrics: dict):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "project_root": str(PROJECT_ROOT),
            "model_file": model_path.name,
            "model_family": experiment_name,
            "logging_mode": "existing_artifact_only",
            **params,
        })

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))
            else:
                mlflow.log_param(key, str(value))

        mlflow.log_artifact(str(model_path), artifact_path="artifacts")


def log_regression_model():

    _log_model_artifact(

        REGRESSION_MODEL,

        "travel-flight-price-regression",

        "xgboost-final",

        {

            "model_type": "XGBoost",

            "feature_pipeline": "ColumnTransformer",

        },

        {

            "mae": 6.8542,

            "rmse": 9.0214,

            "r2": 0.999380,

        },

    )

    print("Logged flight price model artifact")
def log_classification_model():

    _log_model_artifact(

        CLASSIFICATION_MODEL,

        "travel-user-classification",

        "gradient-boosting-final",

        {

            "model_type": "GradientBoostingClassifier",

            "feature_pipeline": "ColumnTransformer",

        },

        {

            "accuracy": 0.3433,

            "cv_mean_accuracy": 0.3351,

            "cv_std_accuracy": 0.025,

        },

    )

    print("Logged gender classification model artifact")

def log_recommendation_model():

    _log_model_artifact(

        RECOMMENDATION_MODEL,

        "travel-hotel-recommendation",

        "hybrid-final",

        {
            "model_type": "HybridHotelRecommendation",
            "collaborative_weight": 0.60,
            "content_weight": 0.30,
            "popularity_weight": 0.10,
            "n_neighbors": 20,
            "top_n": 5,
        },

        {
            "hit_rate_at_1": 0.6475,
            "hit_rate_at_3": 0.9121,
            "hit_rate_at_5": 0.9759,
        },

    )

    print("Logged hotel recommendation model artifact")

if __name__ == "__main__":
    log_regression_model()
    log_classification_model()
    log_recommendation_model()
    print("All MLflow runs completed successfully")