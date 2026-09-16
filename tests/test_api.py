from api.app import app


def test_health():
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "healthy"


def test_flight_price_prediction():
    client = app.test_client()

    payload = {
        "from_city": "Rio de Janeiro",
        "to_city": "Sao Paulo",
        "flight_type": "economic",
        "time": 90,
        "distance": 360,
        "agency": "Netfif",
        "year": 2025,
        "month": 1,
        "day": 15,
        "day_of_week": 2,
        "is_weekend": 0,
    }

    response = client.post(
        "/predict/flight-price",
        json=payload,
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)
    assert data["predicted_price"] > 0


def test_gender_prediction():
    client = app.test_client()

    payload = {
        "company": "Air France",
        "age": 35,
        "flight_count": 10,
        "total_flight_spend": 3000,
        "average_flight_price": 300,
        "average_flight_distance": 500,
        "total_flight_distance": 5000,
        "unique_destinations": 3,
        "unique_flight_types": 2,
        "unique_agencies": 2,
        "hotel_booking_count": 5,
        "total_hotel_spend": 1500,
        "average_hotel_spend": 300,
        "average_hotel_price": 250,
        "average_stay_days": 3,
        "total_stay_days": 15,
        "unique_hotels": 3,
    }

    response = client.post(
        "/predict/gender",
        json=payload,
    )

    assert response.status_code == 200

    data = response.get_json()

    assert data["encoded_prediction"] in [0, 1, 2]
    assert data["predicted_gender"] in [
        "female",
        "male",
        "none",
    ]


def test_hotel_recommendation():
    client = app.test_client()

    payload = {
        "user_id": 0,
        "top_n": 5,
    }

    response = client.post(
        "/recommend/hotels",
        json=payload,
    )

    assert response.status_code == 200

    data = response.get_json()

    assert data["user_id"] == 0
    assert "recommendations" in data
    assert len(data["recommendations"]) <= 5