import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from api.schemas import (
    FlightPriceRequest,
    GenderClassificationRequest,
    HotelRecommendationRequest,
)
from src.recommendation.recommend import HotelRecommender


REGRESSION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "regression" / "flight_price_xgboost_model.pkl"
)

CLASSIFICATION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "classification" / "final_gender_classifier.pkl"
)

RECOMMENDATION_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
    / "hybrid_hotel_recommendation.pkl"
)


app = Flask(__name__)


# Load models once when the API starts
flight_model = joblib.load(REGRESSION_MODEL_PATH)
gender_model = joblib.load(CLASSIFICATION_MODEL_PATH)
hotel_recommender = HotelRecommender(str(RECOMMENDATION_MODEL_PATH))


GENDER_LABELS = {
    0: "female",
    1: "male",
    2: "none",
}
GENDER_CODES = {label: code for code, label in GENDER_LABELS.items()}


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "models": {
                "flight_price": "loaded",
                "gender_classifier": "loaded",
                "hotel_recommender": "loaded",
            },
        }
    )


@app.post("/predict/flight-price")
def predict_flight_price():
    try:
        data = FlightPriceRequest(**request.get_json())

        features = pd.DataFrame(
            [
                {
                    "from": data.from_city,
                    "to": data.to_city,
                    "flightType": data.flight_type,
                    "time": data.time,
                    "distance": data.distance,
                    "agency": data.agency,
                    "year": data.year,
                    "month": data.month,
                    "day": data.day,
                    "day_of_week": data.day_of_week,
                    "is_weekend": data.is_weekend,
                }
            ]
        )

        prediction = float(flight_model.predict(features)[0])

        return jsonify(
            {
                "predicted_price": prediction
            }
        )

    except Exception as exc:
        return jsonify(
            {
                "error": str(exc)
            }
        ), 400


@app.post("/predict/gender")
def predict_gender():
    try:
        data = GenderClassificationRequest(**request.get_json())

        features = pd.DataFrame(
            [
                {
                    "company": data.company,
                    "age": data.age,
                    "userCode": data.user_code,
                    "flight_count": data.flight_count,
                    "total_flight_spend": data.total_flight_spend,
                    "average_flight_price": data.average_flight_price,
                    "average_flight_distance": data.average_flight_distance,
                    "total_flight_distance": data.total_flight_distance,
                    "unique_destinations": data.unique_destinations,
                    "unique_origins": data.unique_origins,
                    "unique_flight_types": data.unique_flight_types,
                    "unique_agencies": data.unique_agencies,
                    "average_flight_time": data.average_flight_time,
                    "hotel_booking_count": data.hotel_booking_count,
                    "total_hotel_spend": data.total_hotel_spend,
                    "average_hotel_spend": data.average_hotel_spend,
                    "average_hotel_price": data.average_hotel_price,
                    "average_stay_days": data.average_stay_days,
                    "total_stay_days": data.total_stay_days,
                    "unique_hotels": data.unique_hotels,
                    "unique_hotel_locations": data.unique_hotel_locations,
                }
            ]
        )

        raw_prediction = gender_model.predict(features)[0]
        if isinstance(raw_prediction, str):
            predicted_gender = raw_prediction
            encoded_prediction = GENDER_CODES.get(predicted_gender, 2)
        else:
            encoded_prediction = int(raw_prediction)
            predicted_gender = GENDER_LABELS.get(encoded_prediction, "unknown")

        return jsonify(
            {
                "predicted_gender": predicted_gender,
                "encoded_prediction": encoded_prediction,
            }
        )

    except Exception as exc:
        return jsonify(
            {
                "error": str(exc)
            }
        ), 400


@app.post("/recommend/hotels")
def recommend_hotels():
    try:
        data = HotelRecommendationRequest(**request.get_json())

        recommendations = hotel_recommender.recommend(
            user_id=data.user_id,
            top_n=data.top_n,
        )

        return jsonify(
            {
                "user_id": data.user_id,
                "recommendations": recommendations,
            }
        )

    except Exception as exc:
        return jsonify(
            {
                "error": str(exc)
            }
        ), 400


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )