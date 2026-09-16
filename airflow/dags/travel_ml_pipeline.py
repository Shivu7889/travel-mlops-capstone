from datetime import datetime

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

from src.data.preprocessing import preprocess_all_data
from src.regression.train import train_model
from src.classification.train import train_classifier
from src.recommendation.train import validate_recommendation_model


def run_preprocessing():
    preprocess_all_data(save=True)


def run_regression_training():
    train_model()


def run_classification_training():
    train_classifier()


def run_recommendation_validation():
    validate_recommendation_model()


with DAG(
    dag_id="travel_ml_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["travel", "mlops"],
) as dag:

    preprocess_data = PythonOperator(
        task_id="preprocess_data",
        python_callable=run_preprocessing,
    )

    train_flight_price_model = PythonOperator(
        task_id="train_flight_price_model",
        python_callable=run_regression_training,
    )

    train_gender_model = PythonOperator(
        task_id="train_gender_model",
        python_callable=run_classification_training,
    )

    validate_hotel_recommendation = PythonOperator(
        task_id="validate_hotel_recommendation",
        python_callable=run_recommendation_validation,
    )

    preprocess_data >> train_flight_price_model >> train_gender_model
    train_gender_model >> validate_hotel_recommendation