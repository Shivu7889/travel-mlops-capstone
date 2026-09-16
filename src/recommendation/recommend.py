from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECOMMENDER_MODEL_PATH = PROJECT_ROOT / "models" / "recommendation" / "hybrid_hotel_recommendation.pkl"


class HotelRecommender:
    """Recommendation wrapper around the serialized hybrid hotel recommender dictionary."""

    def __init__(self, model_path: str | Path = RECOMMENDER_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Recommendation model not found: {self.model_path}")
        return joblib.load(self.model_path)

    def recommend(self, user_id, top_n=5):
        """Return hotel recommendations for an integer user index."""
        if not isinstance(user_id, int):
            try:
                user_id = int(user_id)
            except (TypeError, ValueError):
                raise ValueError("user_id must be an integer index.")

        matrix = self.model["user_hotel_matrix"]
        similarity = self.model["user_similarity"]
        popular = self.model["popular_hotels"]
        hotel_names = self.model["hotel_names"]

        if user_id < 0 or user_id >= len(matrix):
            raise IndexError(f"user_id {user_id} is out of range for the recommendation matrix.")

        user_vector = matrix.iloc[user_id]
        ranked_users = similarity.iloc[user_id].sort_values(ascending=False)
        neighbors = ranked_users[ranked_users.index != user_id].head(10)

        if neighbors.empty:
            candidate_hotels = popular.head(top_n).index.tolist()
        else:
            neighbor_scores = matrix.iloc[neighbors.index]
            candidate_scores = neighbor_scores.T.dot(neighbors.values)
            candidate_scores = candidate_scores / candidate_scores.sum()
            candidate_hotels = candidate_scores.sort_values(ascending=False).head(top_n).index.tolist()

        unseen = [name for name in candidate_hotels if user_vector.get(name, 0) == 0]
        if not unseen:
            unseen = [name for name in hotel_names if user_vector.get(name, 0) == 0]

        recommendations = unseen[:top_n]
        if not recommendations:
            recommendations = popular.head(top_n).index.tolist()

        return recommendations

    def __call__(self, user_id, top_n=5):
        return self.recommend(user_id=user_id, top_n=top_n)


def recommend_hotels(user_id, model_path: str | Path = RECOMMENDER_MODEL_PATH, top_n=5):
    """Convenience wrapper around the serialized hotel recommender."""
    recommender = HotelRecommender(model_path=model_path)
    return recommender.recommend(user_id=user_id, top_n=top_n)


__all__ = ["HotelRecommender", "recommend_hotels"]
