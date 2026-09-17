"""
User Repository — MongoDB version.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

import bcrypt
from motor.motor_asyncio import AsyncIOMotorDatabase


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False


def slugify_username(raw: str) -> str:
    """Convert a raw username to a stable, URL-safe user_id slug.
    Keeps letters, digits and underscores; collapses everything else to '_'.
    Always lowercase.  e.g. 'Mohit Gangani' → 'mohit_gangani'.
    """
    slug = raw.strip().lower()
    slug = re.sub(r"[^a-z0-9_]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "user"


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
            # onboarding_complete=False is a valid falsy value — must pass through
            if key == "onboarding_complete" and value is False:
                update_fields[key] = False

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

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    async def get_by_username(self, username: str) -> Optional[dict]:
        """Look up a user by their username slug (case-insensitive)."""
        slug = slugify_username(username)
        return await self.collection.find_one({"user_id": slug}, {"_id": 0})

    async def signup(self, username: str, password: str) -> dict:
        """
        Register a new user. Safe to call even if a previous incomplete
        signup attempt left a partial record (onboarding_complete=False).

        Rules:
        - If username does not exist → create fresh record.
        - If username exists but onboarding_complete=False → overwrite
          the password hash and return the record so the caller can
          proceed to save the full profile. This handles the 'retry
          signup after a failed attempt' case.
        - If username exists and onboarding_complete=True → raise
          ValueError so the frontend tells the user to Sign In instead.

        Returns the user doc. Never raises on a brand-new username.
        """
        slug = slugify_username(username)
        existing = await self.get_by_username(username)

        if existing:
            if existing.get("onboarding_complete"):
                raise ValueError(
                    "Username already taken. Please Sign In or choose a different username."
                )
            # Incomplete record from a previous signup attempt — reset password
            # and let the caller complete onboarding.
            await self.collection.update_one(
                {"user_id": slug},
                {"$set": {
                    "password_hash": _hash_password(password),
                    "updated_at":    datetime.now(timezone.utc),
                }}
            )
            return await self.get_by_username(username)

        # Brand-new user
        doc = {
            "user_id":             slug,
            "username":            slug,
            "name":                username.strip(),
            "email":               None,
            "password_hash":       _hash_password(password),
            "age":                 None,
            "gender":              None,
            "height_cm":           None,
            "weight_kg":           None,
            "activity_level":      None,
            "fitness_goal":        None,
            "diet_type":           None,
            "target_weight_kg":    None,
            "medical_conditions":  None,
            "custom_calorie_goal": None,
            "onboarding_complete": False,
            "created_at":          datetime.now(timezone.utc),
            "updated_at":          None,
        }
        await self.collection.insert_one(doc)
        return await self.get_by_username(username)

    async def signin(self, username: str, password: str) -> dict:
        """
        Authenticate an existing, fully-onboarded user.

        Raises ValueError with a user-visible message on any failure:
        - username not found
        - wrong password
        - account exists but onboarding is incomplete (tell user to Sign Up)
        """
        existing = await self.get_by_username(username)

        if not existing:
            raise ValueError("Username not found. Please Sign Up to create an account.")

        stored_hash = existing.get("password_hash") or ""
        if stored_hash and not _verify_password(password, stored_hash):
            raise ValueError("Incorrect password. Please try again.")

        if not existing.get("onboarding_complete"):
            raise ValueError(
                "Your account setup is incomplete. Please use Sign Up to finish."
            )

        return existing

    async def login_or_register(self, username: str, password: str) -> tuple[dict, bool]:
        """
        Legacy combined method — kept for backward compatibility.
        New code should call signup() or signin() directly.
        """
        slug = slugify_username(username)
        existing = await self.get_by_username(username)

        if existing:
            stored_hash = existing.get("password_hash") or ""
            if stored_hash and not _verify_password(password, stored_hash):
                raise ValueError("Incorrect password.")
            return existing, False

        doc = {
            "user_id":             slug,
            "username":            slug,
            "name":                username.strip(),
            "email":               None,
            "password_hash":       _hash_password(password),
            "age":                 None,
            "gender":              None,
            "height_cm":           None,
            "weight_kg":           None,
            "activity_level":      None,
            "fitness_goal":        None,
            "diet_type":           None,
            "target_weight_kg":    None,
            "medical_conditions":  None,
            "custom_calorie_goal": None,
            "onboarding_complete": False,
            "created_at":          datetime.now(timezone.utc),
            "updated_at":          None,
        }
        await self.collection.insert_one(doc)
        return await self.get_by_username(username), True
