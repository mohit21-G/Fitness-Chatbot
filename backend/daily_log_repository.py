"""
Daily Log Repository — MongoDB version.
CRUD for food/exercise logs + summary aggregation.
"""
from __future__ import annotations

from datetime import date, datetime, timezone, timedelta

# IST = UTC + 5:30
IST = timezone(timedelta(hours=5, minutes=30))
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId


# Canonical chronological order for displaying meal sections in the Daily Log.
# Requirement: Breakfast → Morning → Lunch → Afternoon → Evening → Dinner.
# "snack" is a between-meals slot (afternoon-ish, before evening); any unknown
# meal type sorts last so it is never dropped. Only meals that actually have
# logged items are included by the caller — empty meals never appear.
_MEAL_CHRONOLOGICAL_ORDER = [
    "breakfast",
    "morning",
    "lunch",
    "afternoon",
    "snack",
    "evening",
    "dinner",
]
_MEAL_RANK = {m: i for i, m in enumerate(_MEAL_CHRONOLOGICAL_ORDER)}


def meal_sort_key(meal_type: str) -> int:
    """Chronological rank for a meal type. Unknown types sort after all known
    meals (kept, never dropped), preserving a stable, sensible display order."""
    return _MEAL_RANK.get((meal_type or "").strip().lower(), len(_MEAL_CHRONOLOGICAL_ORDER))


def order_meal_breakdown(meal_breakdown: dict) -> dict:
    """Return a new dict with the same meal entries re-ordered chronologically.
    Only reorders keys — the per-meal item lists are left exactly as-is (no item
    is lost, duplicated, or internally reordered)."""
    return {
        meal: meal_breakdown[meal]
        for meal in sorted(meal_breakdown.keys(), key=meal_sort_key)
    }


class DailyLogRepository:
    """Food and exercise log operations against MongoDB."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.food_logs = db["daily_food_logs"]
        self.exercise_logs = db["daily_exercise_logs"]

    # ------------------------------------------------------------------
    # Food Logs
    # ------------------------------------------------------------------

    async def create_food_log(self, **kwargs) -> dict:
        doc = {**kwargs, "created_at": datetime.now(IST)}
        # Convert date to string for storage
        if isinstance(doc.get("log_date"), date):
            doc["log_date"] = doc["log_date"].isoformat()
        result = await self.food_logs.insert_one(doc)
        doc["id"] = str(result.inserted_id)
        doc.pop("_id", None)
        return doc

    async def get_food_log(self, log_id: str) -> Optional[dict]:
        try:
            doc = await self.food_logs.find_one({"_id": ObjectId(log_id)})
            if doc:
                doc["id"] = str(doc.pop("_id"))
            return doc
        except Exception:
            return None

    async def get_food_logs_by_date(self, user_id: str, log_date: date) -> list[dict]:
        date_str = log_date.isoformat() if isinstance(log_date, date) else log_date
        cursor = self.food_logs.find(
            {"user_id": user_id, "log_date": date_str}
        ).sort("created_at", 1)
        docs = await cursor.to_list(length=500)
        for d in docs:
            d["id"] = str(d.pop("_id"))
        return docs

    async def get_food_logs_by_range(self, user_id: str, start_date: date, end_date: date) -> list[dict]:
        cursor = self.food_logs.find({
            "user_id": user_id,
            "log_date": {"$gte": start_date.isoformat(), "$lte": end_date.isoformat()}
        }).sort([("log_date", 1), ("created_at", 1)])
        docs = await cursor.to_list(length=1000)
        for d in docs:
            d["id"] = str(d.pop("_id"))
        return docs

    async def update_food_log(self, log_id: str, user_id: str, **kwargs) -> Optional[dict]:
        try:
            result = await self.food_logs.update_one(
                {"_id": ObjectId(log_id), "user_id": user_id},
                {"$set": kwargs}
            )
            if result.modified_count == 0:
                return None
            return await self.get_food_log(log_id)
        except Exception:
            return None

    async def delete_food_log(self, log_id: str, user_id: str) -> bool:
        try:
            result = await self.food_logs.delete_one({"_id": ObjectId(log_id), "user_id": user_id})
            return result.deleted_count > 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Exercise Logs
    # ------------------------------------------------------------------

    async def create_exercise_log(self, **kwargs) -> dict:
        doc = {**kwargs, "created_at": datetime.now(IST)}
        if isinstance(doc.get("log_date"), date):
            doc["log_date"] = doc["log_date"].isoformat()
        result = await self.exercise_logs.insert_one(doc)
        doc["id"] = str(result.inserted_id)
        doc.pop("_id", None)
        return doc

    async def get_exercise_log(self, log_id: str) -> Optional[dict]:
        try:
            doc = await self.exercise_logs.find_one({"_id": ObjectId(log_id)})
            if doc:
                doc["id"] = str(doc.pop("_id"))
            return doc
        except Exception:
            return None

    async def get_exercise_logs_by_date(self, user_id: str, log_date: date) -> list[dict]:
        date_str = log_date.isoformat() if isinstance(log_date, date) else log_date
        cursor = self.exercise_logs.find(
            {"user_id": user_id, "log_date": date_str}
        ).sort("created_at", 1)
        docs = await cursor.to_list(length=500)
        for d in docs:
            d["id"] = str(d.pop("_id"))
        return docs

    async def get_exercise_logs_by_range(self, user_id: str, start_date: date, end_date: date) -> list[dict]:
        cursor = self.exercise_logs.find({
            "user_id": user_id,
            "log_date": {"$gte": start_date.isoformat(), "$lte": end_date.isoformat()}
        }).sort([("log_date", 1), ("created_at", 1)])
        docs = await cursor.to_list(length=1000)
        for d in docs:
            d["id"] = str(d.pop("_id"))
        return docs

    async def update_exercise_log(self, log_id: str, user_id: str, **kwargs) -> Optional[dict]:
        try:
            result = await self.exercise_logs.update_one(
                {"_id": ObjectId(log_id), "user_id": user_id},
                {"$set": kwargs}
            )
            if result.modified_count == 0:
                return None
            return await self.get_exercise_log(log_id)
        except Exception:
            return None

    async def delete_exercise_log(self, log_id: str, user_id: str) -> bool:
        try:
            result = await self.exercise_logs.delete_one({"_id": ObjectId(log_id), "user_id": user_id})
            return result.deleted_count > 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Daily Summary Aggregation
    # ------------------------------------------------------------------

    async def get_daily_food_totals(self, user_id: str, log_date: date) -> dict:
        date_str = log_date.isoformat() if isinstance(log_date, date) else log_date
        docs = await self.get_food_logs_by_date(user_id, log_date)
        totals = {
            "total_calories_consumed": 0.0,
            "total_protein_g": 0.0,
            "total_carbs_g": 0.0,
            "total_fat_g": 0.0,
            "total_fiber_g": 0.0,
            "food_entries": len(docs),
            "meal_breakdown": {},
        }
        for log in docs:
            totals["total_calories_consumed"] += log.get("calories", 0)
            totals["total_protein_g"] += log.get("protein_g", 0)
            totals["total_carbs_g"] += log.get("carbs_g", 0)
            totals["total_fat_g"] += log.get("fat_g", 0)
            totals["total_fiber_g"] += log.get("fiber_g", 0)

            meal = log.get("meal_type", "snack")
            if meal not in totals["meal_breakdown"]:
                totals["meal_breakdown"][meal] = {"calories": 0.0, "items": [], "item_details": []}
            totals["meal_breakdown"][meal]["calories"] += log.get("calories", 0)
            totals["meal_breakdown"][meal]["items"].append(log.get("food_name_display", ""))
            totals["meal_breakdown"][meal]["item_details"].append({
                "food_name": log.get("food_name_display") or log.get("food_name", ""),
                "quantity": log.get("quantity_input", ""),
                "calories": round(log.get("calories", 0) or 0, 1),
                "protein_g": round(log.get("protein_g", 0) or 0, 1),
                "carbs_g": round(log.get("carbs_g", 0) or 0, 1),
                "fat_g": round(log.get("fat_g", 0) or 0, 1),
                "category": log.get("category", "")
            })

        for k in ["total_calories_consumed", "total_protein_g", "total_carbs_g", "total_fat_g", "total_fiber_g"]:
            totals[k] = round(totals[k], 1)
        for meal in totals["meal_breakdown"]:
            totals["meal_breakdown"][meal]["calories"] = round(totals["meal_breakdown"][meal]["calories"], 1)

        # Order meal sections chronologically for display (Breakfast → Morning →
        # Lunch → Afternoon → Evening → Dinner). Only meals that were actually
        # logged are present, so empty sections never show; a meal logged later
        # (e.g. Afternoon after Evening) lands in its correct position, not appended.
        totals["meal_breakdown"] = order_meal_breakdown(totals["meal_breakdown"])

        return totals

    async def get_daily_exercise_totals(self, user_id: str, log_date: date) -> dict:
        docs = await self.get_exercise_logs_by_date(user_id, log_date)
        total_burned = sum(d.get("calories_avg", 0) for d in docs)
        return {
            "total_calories_burned": round(total_burned, 1),
            "exercise_entries": len(docs),
        }
