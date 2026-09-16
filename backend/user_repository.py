"""
User Repository — MongoDB version.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase


class UserRepository:
    """User profile CRUD against MongoDB."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db["user_profiles"]

    async def get_by_id(self, user_id: str) -> Optional[dict]:
        return await self.collection.find_one({"user_id": user_id}, {"_id": 0})

    async def get_by_email(self, email: str) -> Optional[dict]:
        return await self.collection.find_one({"email": email}, {"_id": 0})

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[dict]:
        cursor = self.collection.find({}, {"_id": 0}).skip(offset).limit(limit)
        return await cursor.to_list(length=limit)

    async def count(self) -> int:
        return await self.collection.count_documents({})

    async def create(self, data: dict) -> dict:
        """Create a new user profile. Raises ValueError if user_id exists."""
        existing = await self.get_by_id(data["user_id"])
        if existing:
            raise ValueError(f"User '{data['user_id']}' already exists")

        # Hash password if provided
        password_hash = None
        if data.get("password"):
            from auth import hash_password
            password_hash = hash_password(data["password"])

        doc = {
            "user_id": data["user_id"],
            "name": data["name"],
            "email": data.get("email"),
            "password_hash": password_hash,
            "age": data["age"],
            "gender": data["gender"],
            "height_cm": data["height_cm"],
            "weight_kg": data["weight_kg"],
            "activity_level": data["activity_level"],
            "fitness_goal": data["fitness_goal"],
            "diet_type": data["diet_type"],
            "target_weight_kg": data.get("target_weight_kg"),
            "medical_conditions": data.get("medical_conditions"),
            "created_at": datetime.now(timezone.utc),
            "updated_at": None,
        }
        await self.collection.insert_one(doc)
        return await self.get_by_id(data["user_id"])

    async def update(self, user_id: str, data: dict) -> Optional[dict]:
        """Partial update — only modifies provided fields.

        Special case: ``custom_calorie_goal`` may be explicitly set to None to
        clear a previously saved custom target (revert to auto-computed TDEE).
        All other fields follow the usual "skip None" rule so callers can
        send only the fields they want to change.
        """
        user = await self.get_by_id(user_id)
        if not user:
            return None

        update_fields = {}
        for key, value in data.items():
            if key == "user_id":
                # user_id is the stable unique identifier — never overwrite it
                continue
            if key == "custom_calorie_goal":
                # Explicit None = clear the custom goal; keep other None fields as "not provided"
                if hasattr(value, "value"):
                    value = value.value
                update_fields[key] = value   # None is intentional here
            elif value is not None:
                if hasattr(value, "value"):
                    value = value.value
                update_fields[key] = value

        if update_fields:
            update_fields["updated_at"] = datetime.now(timezone.utc)
            await self.collection.update_one(
                {"user_id": user_id},
                {"$set": update_fields}
            )

        return await self.get_by_id(user_id)

    async def delete(self, user_id: str) -> bool:
        result = await self.collection.delete_one({"user_id": user_id})
        return result.deleted_count > 0
