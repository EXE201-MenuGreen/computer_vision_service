"""
app.db — PostgREST database access layer.

Modules:
  client          — anon_client() / service_client() factory
  food_nutrition  — match_food(), upsert_food()
  meal_history    — store_meal(), query_history(), get_recent()
  user_profile    — get_user_cv_profile()
  visual_store    — store_visual_embedding()
"""
from app.db.client import anon_client, service_client
from app.db.food_nutrition import match_food, upsert_food
from app.db.meal_history import get_recent, query_history, store_meal
from app.db.user_profile import get_user_cv_profile

__all__ = [
    "anon_client",
    "service_client",
    "match_food",
    "upsert_food",
    "store_meal",
    "query_history",
    "get_recent",
    "get_user_cv_profile",
]
