"""
Production data preprocessing utilities for the Travel ML project.

The functions in this module provide the reusable data-loading,
cleaning, date processing, and feature-engineering logic required
by the project's ML pipelines.
"""

from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def load_raw_data():
    """
    Load the three raw travel datasets.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        users, flights, hotels
    """

    users = pd.read_csv(RAW_DATA_DIR / "users.csv")
    flights = pd.read_csv(RAW_DATA_DIR / "flights.csv")
    hotels = pd.read_csv(RAW_DATA_DIR / "hotels.csv")

    return users, flights, hotels


# -------------------------------------------------------------------
# Basic cleaning
# -------------------------------------------------------------------

def clean_users(users):
    """
    Clean the users dataset.
    """

    users = users.copy()

    users = users.drop_duplicates().reset_index(drop=True)

    return users


def clean_flights(flights):
    """
    Clean the flights dataset and prepare date/numeric fields.
    """

    flights = flights.copy()

    flights = flights.drop_duplicates().reset_index(drop=True)

    # Convert date
    flights["date"] = pd.to_datetime(
        flights["date"],
        errors="coerce"
    )

    # Numeric columns
    numeric_columns = [
        "travelCode",
        "userCode",
        "price",
        "time",
        "distance",
    ]

    for column in numeric_columns:
        flights[column] = pd.to_numeric(
            flights[column],
            errors="coerce"
        )

    return flights


def clean_hotels(hotels):
    """
    Clean the hotels dataset and prepare date/numeric fields.
    """

    hotels = hotels.copy()

    hotels = hotels.drop_duplicates().reset_index(drop=True)

    # Convert date
    hotels["date"] = pd.to_datetime(
        hotels["date"],
        errors="coerce"
    )

    # Numeric columns
    numeric_columns = [
        "travelCode",
        "userCode",
        "days",
        "price",
        "total",
    ]

    for column in numeric_columns:
        hotels[column] = pd.to_numeric(
            hotels[column],
            errors="coerce"
        )

    return hotels


# -------------------------------------------------------------------
# Date feature engineering
# -------------------------------------------------------------------

def add_date_features(df, date_column="date"):
    """
    Add calendar-based features to a DataFrame.

    Generated features:
        year
        month
        day
        day_of_week
        is_weekend
    """

    df = df.copy()

    if date_column not in df.columns:
        raise ValueError(
            f"Column '{date_column}' not found in DataFrame."
        )

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    df["year"] = df[date_column].dt.year
    df["month"] = df[date_column].dt.month
    df["day"] = df[date_column].dt.day
    df["day_of_week"] = df[date_column].dt.dayofweek
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    return df


# -------------------------------------------------------------------
# User behavioral features
# -------------------------------------------------------------------

def create_user_features(users, flights, hotels):
    """
    Create user-level behavioral features from flight and hotel data.

    This produces the aggregated user feature dataset used by the
    classification workflow.
    """

    users = users.copy()
    flights = flights.copy()
    hotels = hotels.copy()

    # ---------------------------------------------------------------
    # Flight behavior
    # ---------------------------------------------------------------

    flight_features = flights.groupby("userCode").agg(
        flight_count=("travelCode", "count"),
        total_flight_spend=("price", "sum"),
        average_flight_price=("price", "mean"),
        average_flight_distance=("distance", "mean"),
        total_flight_distance=("distance", "sum"),
        unique_destinations=("to", "nunique"),
        unique_origins=("from", "nunique"),
        unique_flight_types=("flightType", "nunique"),
        unique_agencies=("agency", "nunique"),
        average_flight_time=("time", "mean"),
    ).reset_index()

    # ---------------------------------------------------------------
    # Hotel behavior
    # ---------------------------------------------------------------

    hotel_features = hotels.groupby("userCode").agg(
        hotel_booking_count=("travelCode", "count"),
        total_hotel_spend=("total", "sum"),
        average_hotel_spend=("total", "mean"),
        average_hotel_price=("price", "mean"),
        average_stay_days=("days", "mean"),
        total_stay_days=("days", "sum"),
        unique_hotels=("name", "nunique"),
        unique_hotel_locations=("place", "nunique"),
    ).reset_index()

    # ---------------------------------------------------------------
    # Merge behavioral features with user information
    # ---------------------------------------------------------------

    user_metadata = users[["code", "company", "name", "gender", "age"]].copy()

    user_features = user_metadata.merge(
        flight_features,
        left_on="code",
        right_on="userCode",
        how="left",
    )

    user_features = user_features.merge(
        hotel_features,
        on="userCode",
        how="left",
    )

    # Users without flights/hotels receive zero behavioral values
    behavioral_columns = [
        "flight_count",
        "total_flight_spend",
        "average_flight_price",
        "average_flight_distance",
        "total_flight_distance",
        "unique_destinations",
        "unique_origins",
        "unique_flight_types",
        "unique_agencies",
        "average_flight_time",
        "hotel_booking_count",
        "total_hotel_spend",
        "average_hotel_spend",
        "average_hotel_price",
        "average_stay_days",
        "total_stay_days",
        "unique_hotels",
        "unique_hotel_locations",
    ]

    for column in behavioral_columns:
        if column in user_features.columns:
            user_features[column] = (
                user_features[column].fillna(0)
            )

    return user_features


# -------------------------------------------------------------------
# Flight ML dataset
# -------------------------------------------------------------------

def create_flight_ml_data(flights):
    """
    Create the flight ML dataset with engineered date features.
    """

    flight_ml_data = flights.copy()

    flight_ml_data = add_date_features(
        flight_ml_data,
        date_column="date",
    )

    return flight_ml_data


# -------------------------------------------------------------------
# Full preprocessing pipeline
# -------------------------------------------------------------------

def preprocess_all_data(save=True):
    """
    Execute the complete preprocessing workflow.

    Returns
    -------
    tuple
        users_clean, flights_clean, hotels_clean,
        user_features, flight_ml_data
    """

    users, flights, hotels = load_raw_data()

    users_clean = clean_users(users)
    flights_clean = clean_flights(flights)
    hotels_clean = clean_hotels(hotels)

    # Date features used by downstream ML workflows
    flights_clean = add_date_features(
        flights_clean,
        date_column="date",
    )

    hotels_clean = add_date_features(
        hotels_clean,
        date_column="date",
    )

    user_features = create_user_features(
        users_clean,
        flights_clean,
        hotels_clean,
    )

    flight_ml_data = create_flight_ml_data(
        flights_clean
    )

    if save:
        PROCESSED_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        users_clean.to_csv(
            PROCESSED_DATA_DIR / "users_clean.csv",
            index=False,
        )

        flights_clean.to_csv(
            PROCESSED_DATA_DIR / "flights_clean.csv",
            index=False,
        )

        hotels_clean.to_csv(
            PROCESSED_DATA_DIR / "hotels_clean.csv",
            index=False,
        )

        user_features.to_csv(
            PROCESSED_DATA_DIR / "user_features.csv",
            index=False,
        )

        flight_ml_data.to_csv(
            PROCESSED_DATA_DIR / "flight_ml_data.csv",
            index=False,
        )

    return (
        users_clean,
        flights_clean,
        hotels_clean,
        user_features,
        flight_ml_data,
    )


# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------

def validate_raw_data(users, flights, hotels):
    """
    Validate required columns in the raw datasets.
    """

    required_columns = {
        "users": [
            "code",
            "company",
            "name",
            "gender",
            "age",
        ],
        "flights": [
            "travelCode",
            "userCode",
            "from",
            "to",
            "flightType",
            "price",
            "time",
            "distance",
            "agency",
            "date",
        ],
        "hotels": [
            "travelCode",
            "userCode",
            "name",
            "place",
            "days",
            "price",
            "total",
            "date",
        ],
    }

    datasets = {
        "users": users,
        "flights": flights,
        "hotels": hotels,
    }

    for dataset_name, columns in required_columns.items():

        missing = [
            column
            for column in columns
            if column not in datasets[dataset_name].columns
        ]

        if missing:
            raise ValueError(
                f"{dataset_name} is missing columns: {missing}"
            )

    return True


# -------------------------------------------------------------------
# Script execution
# -------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("TRAVEL ML — DATA PREPROCESSING")
    print("=" * 70)

    users, flights, hotels = load_raw_data()

    print("\nRaw datasets:")
    print(f"Users   : {users.shape}")
    print(f"Flights : {flights.shape}")
    print(f"Hotels  : {hotels.shape}")

    validate_raw_data(
        users,
        flights,
        hotels,
    )

    (
        users_clean,
        flights_clean,
        hotels_clean,
        user_features,
        flight_ml_data,
    ) = preprocess_all_data(save=True)

    print("\nProcessed datasets:")
    print(f"Users clean      : {users_clean.shape}")
    print(f"Flights clean    : {flights_clean.shape}")
    print(f"Hotels clean     : {hotels_clean.shape}")
    print(f"User features    : {user_features.shape}")
    print(f"Flight ML data   : {flight_ml_data.shape}")

    print("\nProcessed files saved to:")
    print(PROCESSED_DATA_DIR)

    print("\nPreprocessing completed successfully.")