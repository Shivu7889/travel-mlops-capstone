import requests
import streamlit as st

API_URL = "http://127.0.0.1:5000"

st.set_page_config(
    page_title="Travel ML Dashboard",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Travel ML Dashboard")
st.markdown("Machine Learning & MLOps Travel Analytics Platform")

# ---------------------------------------------------------
# API Health
# ---------------------------------------------------------
st.subheader("API Status")

try:
    response = requests.get(f"{API_URL}/health", timeout=5)

    if response.status_code == 200:
        health = response.json()
        st.success("API is healthy")
        st.json(health)
    else:
        st.error(f"API returned status code {response.status_code}")

except requests.exceptions.RequestException:
    st.error("Could not connect to Flask API. Make sure it is running on port 5000.")


# ---------------------------------------------------------
# Navigation
# ---------------------------------------------------------
st.divider()

option = st.selectbox(
    "Select ML Service",
    [
        "Flight Price Prediction",
        "Gender Classification",
        "Hotel Recommendation",
    ],
)


# ---------------------------------------------------------
# Flight Price Prediction
# ---------------------------------------------------------
if option == "Flight Price Prediction":

    st.header("✈️ Flight Price Prediction")

    col1, col2 = st.columns(2)

    with col1:
        from_city = st.text_input("From", "Madrid")
        to_city = st.text_input("To", "Paris")
        flight_type = st.text_input("Flight Type", "firstClass")
        agency = st.text_input("Agency", "FlyingDrops")
        time = st.number_input("Time", min_value=0.0, value=2.0)
        distance = st.number_input("Distance", min_value=0.0, value=1000.0)

    with col2:
        year = st.number_input("Year", min_value=2000, value=2026)
        month = st.number_input("Month", min_value=1, max_value=12, value=1)
        day = st.number_input("Day", min_value=1, max_value=31, value=15)
        day_of_week = st.number_input(
            "Day of Week",
            min_value=0,
            max_value=6,
            value=3,
        )
        is_weekend = st.selectbox(
            "Weekend?",
            [0, 1],
            format_func=lambda x: "Yes" if x else "No",
        )

    if st.button("Predict Flight Price"):

        payload = {
            "from_city": from_city,
            "to_city": to_city,
            "flight_type": flight_type,
            "time": time,
            "distance": distance,
            "agency": agency,
            "year": year,
            "month": month,
            "day": day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
        }

        try:
            response = requests.post(
                f"{API_URL}/predict/flight-price",
                json=payload,
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()
                st.success(
                    f"Predicted Flight Price: {result['predicted_price']:.2f}"
                )
            else:
                st.error(response.text)

        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")


# ---------------------------------------------------------
# Gender Classification
# ---------------------------------------------------------
elif option == "Gender Classification":

    st.header("👤 User Classification")

    st.info(
        "This model predicts the labeled class present in the training dataset "
        "based on user demographic and travel behavior features."
    )

    col1, col2 = st.columns(2)

    with col1:
        company = st.text_input("Company", "Air France")
        age = st.number_input("Age", min_value=0.0, value=30.0)
        flight_count = st.number_input("Flight Count", min_value=0.0, value=10.0)
        total_flight_spend = st.number_input(
            "Total Flight Spend",
            min_value=0.0,
            value=1000.0,
        )
        average_flight_price = st.number_input(
            "Average Flight Price",
            min_value=0.0,
            value=100.0,
        )
        average_flight_distance = st.number_input(
            "Average Flight Distance",
            min_value=0.0,
            value=1000.0,
        )
        total_flight_distance = st.number_input(
            "Total Flight Distance",
            min_value=0.0,
            value=10000.0,
        )
        unique_destinations = st.number_input(
            "Unique Destinations",
            min_value=0.0,
            value=3.0,
        )
        unique_flight_types = st.number_input(
            "Unique Flight Types",
            min_value=0.0,
            value=2.0,
        )

    with col2:
        unique_agencies = st.number_input(
            "Unique Agencies",
            min_value=0.0,
            value=2.0,
        )
        hotel_booking_count = st.number_input(
            "Hotel Booking Count",
            min_value=0.0,
            value=5.0,
        )
        total_hotel_spend = st.number_input(
            "Total Hotel Spend",
            min_value=0.0,
            value=1000.0,
        )
        average_hotel_spend = st.number_input(
            "Average Hotel Spend",
            min_value=0.0,
            value=200.0,
        )
        average_hotel_price = st.number_input(
            "Average Hotel Price",
            min_value=0.0,
            value=200.0,
        )
        average_stay_days = st.number_input(
            "Average Stay Days",
            min_value=0.0,
            value=3.0,
        )
        total_stay_days = st.number_input(
            "Total Stay Days",
            min_value=0.0,
            value=15.0,
        )
        unique_hotels = st.number_input(
            "Unique Hotels",
            min_value=0.0,
            value=3.0,
        )

    if st.button("Classify User"):

        payload = {
            "company": company,
            "age": age,
            "flight_count": flight_count,
            "total_flight_spend": total_flight_spend,
            "average_flight_price": average_flight_price,
            "average_flight_distance": average_flight_distance,
            "total_flight_distance": total_flight_distance,
            "unique_destinations": unique_destinations,
            "unique_flight_types": unique_flight_types,
            "unique_agencies": unique_agencies,
            "hotel_booking_count": hotel_booking_count,
            "total_hotel_spend": total_hotel_spend,
            "average_hotel_spend": average_hotel_spend,
            "average_hotel_price": average_hotel_price,
            "average_stay_days": average_stay_days,
            "total_stay_days": total_stay_days,
            "unique_hotels": unique_hotels,
        }

        try:
            response = requests.post(
                f"{API_URL}/predict/gender",
                json=payload,
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()

                st.success(
                    f"Predicted Class: {result['predicted_gender']}"
                )
                st.write(
                    f"Encoded prediction: {result['encoded_prediction']}"
                )

            else:
                st.error(response.text)

        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")


# ---------------------------------------------------------
# Hotel Recommendation
# ---------------------------------------------------------
else:

    st.header("🏨 Hotel Recommendation")

    st.write(
        "Get personalized hotel recommendations using the hybrid "
        "recommendation model."
    )

    user_id = st.number_input(
        "User ID",
        min_value=0,
        value=0,
        step=1,
    )

    top_n = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=9,
        value=5,
    )

    if st.button("Recommend Hotels"):

        payload = {
            "user_id": int(user_id),
            "top_n": int(top_n),
        }

        try:
            response = requests.post(
                f"{API_URL}/recommend/hotels",
                json=payload,
                timeout=10,
            )

            if response.status_code == 200:

                result = response.json()

                st.success("Recommendations generated!")

                for i, hotel in enumerate(
                    result["recommendations"],
                    start=1,
                ):
                    st.write(f"### {i}. {hotel}")

            else:
                st.error(response.text)

        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")