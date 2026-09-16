from pydantic import BaseModel, Field


class FlightPriceRequest(BaseModel):
    from_city: str = Field(..., description="Departure city")
    to_city: str = Field(..., description="Destination city")
    flight_type: str = Field(..., description="Flight type")
    time: float = Field(..., ge=0, description="Flight time")
    distance: float = Field(..., ge=0, description="Flight distance")
    agency: str = Field(..., description="Flight agency")
    year: int = Field(..., description="Flight year")
    month: int = Field(..., ge=1, le=12)
    day: int = Field(..., ge=1, le=31)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)


class FlightPriceResponse(BaseModel):
    predicted_price: float


class GenderClassificationRequest(BaseModel):
    company: str
    age: float = Field(..., ge=0)
    user_code: float = Field(0, ge=0)
    flight_count: float = Field(..., ge=0)
    total_flight_spend: float = Field(..., ge=0)
    average_flight_price: float = Field(..., ge=0)
    average_flight_distance: float = Field(..., ge=0)
    total_flight_distance: float = Field(..., ge=0)
    unique_destinations: float = Field(..., ge=0)
    unique_origins: float = Field(0, ge=0)
    unique_flight_types: float = Field(..., ge=0)
    unique_agencies: float = Field(..., ge=0)
    average_flight_time: float = Field(0, ge=0)
    hotel_booking_count: float = Field(..., ge=0)
    total_hotel_spend: float = Field(..., ge=0)
    average_hotel_spend: float = Field(..., ge=0)
    average_hotel_price: float = Field(..., ge=0)
    average_stay_days: float = Field(..., ge=0)
    total_stay_days: float = Field(..., ge=0)
    unique_hotels: float = Field(..., ge=0)
    unique_hotel_locations: float = Field(0, ge=0)


class GenderClassificationResponse(BaseModel):
    predicted_gender: str
    encoded_prediction: int


class HotelRecommendationRequest(BaseModel):
    user_id: int = Field(..., ge=0)
    top_n: int = Field(5, ge=1, le=9)


class HotelRecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[str]