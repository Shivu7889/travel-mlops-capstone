from pathlib import Path

from src.recommendation.recommend import HotelRecommender


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
    / "hybrid_hotel_recommendation.pkl"
)


def validate_recommendation_model():
    """Validate the saved hotel recommendation model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Recommendation model not found: {MODEL_PATH}"
        )

    recommender = HotelRecommender(model_path=MODEL_PATH)

    test_user_id = 0
    recommendations = recommender.recommend(
        user_id=test_user_id,
        top_n=5,
    )

    if not recommendations:
        raise ValueError("Recommendation model returned no recommendations.")

    if len(recommendations) > 5:
        raise ValueError("Recommendation model returned more than top_n items.")

    print("=" * 60)
    print("Hotel Recommendation Validation Complete")
    print("=" * 60)
    print(f"Test user       : {test_user_id}")
    print(f"Recommendations : {recommendations}")
    print(f"Model path      : {MODEL_PATH}")
    print("=" * 60)

    return recommendations


if __name__ == "__main__":
    validate_recommendation_model()