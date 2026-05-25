"""
app.db — Supabase database access layer.

Modules:
  client          — anon_client() / service_client() factory
  food_nutrition  — match_food(), upsert_food()
  meal_history    — store_meal(), query_history(), get_recent()
  visual_store    — store_visual_embedding()
"""
from app.db.client import anon_client, service_client
from app.db.food_nutrition import match_food, upsert_food
from app.db.meal_history import get_recent, query_history, store_meal
from app.db.visual_store import store_visual_embedding

__all__ = [
    "anon_client",
    "service_client",
    "match_food",
    "upsert_food",
    "store_meal",
    "query_history",
    "get_recent",
    "store_visual_embedding",
]
