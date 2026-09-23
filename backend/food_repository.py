"""
Food Repository — MongoDB version.
All food + alias database operations.
"""
from __future__ import annotations

import re
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorDatabase


# ---------------------------------------------------------------------------
# Learned-alias hygiene
# ---------------------------------------------------------------------------
# Tokens that are never part of a real food name. If a would-be learned alias
# contains any of these, it is a raw *sentence fragment* (e.g. "aje bhakhari",
# "chai khadi") rather than a food name. Storing such fragments as aliases
# poisons future fuzzy lookups — e.g. the alias "aje bhakhari" pointing at an
# unrelated USDA pepper entry then out-scores the correct "bhakri".
# NOTE: deliberately EXCLUDES connectives like "and"/"with"/"aur"/"saath" —
# those legitimately appear in real dish names ("apple and honey sorbet",
# "nariyal ke saath phoolgobhi"). Only tokens that are never part of a dish
# name are listed: pronouns, eating/drinking verbs and time/meal context.
_ALIAS_STOP_TOKENS = {
    # pronouns / fillers (English + Romanised Indian languages)
    "i", "we", "you", "my", "me", "just", "some",
    "had", "have", "has", "was", "were", "am", "is",
    "today", "yesterday", "aaje", "aaj", "aje", "aj", "ajj",
    "kale", "kal", "gay kale", "gay kal",
    "maine", "mene", "mein", "main", "hu", "hoon",
    # eating / drinking verbs (incl. common transliteration spellings)
    "khadha", "khadhi", "khadhu", "khadho", "khadhni", "khadhne", "khadhna",
    "khadi", "khadni", "khadne", "khadna", "khada", "khadu",
    "khaya", "khayi", "khaye", "khai", "khavu", "khava", "khao", "khiya",
    "lidha", "lidhi", "lidhu", "lidho", "jamya", "jamyu",
    "li", "liya", "liye", "liyu", "leli",
    "pidhi", "pidhu", "pidha", "pidho", "pido", "piya", "piyo",
    "ate", "eaten", "eating", "drank", "drinking", "having",
    # meal / time context
    "savare", "savar", "bapore", "sanje",
    "breakfast", "lunch", "dinner", "nashta", "nasto",
}


def is_safe_learned_alias(alias: str) -> bool:
    """
    True when `alias` is safe to persist as a *learned* (runtime) food alias.

    Rejects raw sentence fragments so garbage aliases can never outrank a
    correct canonical food name in fuzzy search. For example the fragment
    "aje bhakhari" (from "me aje savare bhakhari ...") must never be stored
    pointing at an unrelated food, because it would then out-score "bhakri".

    Applies ONLY to learned aliases — curated primary/regional dataset aliases
    (which legitimately contain connectives like "and"/"aur"/"ke saath") are
    never filtered by this.
    """
    a = (alias or "").strip().lower()
    if len(a) < 3:
        return False
    tokens = re.findall(r"[^\W\d_]+", a)
    if not tokens:
        return False
    # Any stop token means this is a sentence fragment, not a food name.
    if any(t in _ALIAS_STOP_TOKENS for t in tokens):
        return False
    # Generic umbrella terms must NEVER be learned as aliases for specific foods.
    from food_specificity import ALL_GENERIC_TERMS
    if a in ALL_GENERIC_TERMS or (len(tokens) == 1 and tokens[0] in ALL_GENERIC_TERMS):
        return False
    # Real food names are short phrases; 5+ words indicates a sentence.
    if len(tokens) > 4:
        return False
    return True


class FoodRepository:
    """Food & alias queries against MongoDB."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.foods = db["foods"]
        self.aliases = db["food_aliases"]

    # ------------------------------------------------------------------
    # Single-record lookups
    # ------------------------------------------------------------------

    async def get_by_id(self, food_id: str):
        return await self.foods.find_one({"food_id": food_id}, {"_id": 0})

    async def get_by_exact_name(self, name: str):
        return await self.foods.find_one({"food_name": name.strip().lower()}, {"_id": 0})

    async def get_by_exact_alias(self, alias: str):
        alias_doc = await self.aliases.find_one({"alias": alias.strip().lower()})
        if alias_doc:
            return await self.get_by_id(alias_doc["food_id"])
        return None

    # ------------------------------------------------------------------
    # List / search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: Optional[str] = None,
        vegetarian_status: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        filter_q = {}
        if query:
            filter_q["food_name"] = {"$regex": query.lower(), "$options": "i"}
        if vegetarian_status:
            filter_q["vegetarian_status"] = vegetarian_status
        if category:
            filter_q["category"] = category

        cursor = self.foods.find(filter_q, {"_id": 0}).skip(offset).limit(limit)
        return await cursor.to_list(length=limit)

    async def count(
        self,
        query: Optional[str] = None,
        vegetarian_status: Optional[str] = None,
        category: Optional[str] = None,
    ) -> int:
        filter_q = {}
        if query:
            filter_q["food_name"] = {"$regex": query.lower(), "$options": "i"}
        if vegetarian_status:
            filter_q["vegetarian_status"] = vegetarian_status
        if category:
            filter_q["category"] = category
        return await self.foods.count_documents(filter_q)

    async def search_by_alias_substring(self, alias: str, limit: int = 10) -> list[dict]:
        cursor = self.aliases.find(
            {"alias": {"$regex": alias.lower(), "$options": "i"}}
        ).limit(limit)
        alias_docs = await cursor.to_list(length=limit)
        food_ids = list({a["food_id"] for a in alias_docs})
        if not food_ids:
            return []
        return await self.get_foods_by_ids(food_ids)

    # ------------------------------------------------------------------
    # Bulk lookups (fuzzy matching)
    # ------------------------------------------------------------------

    async def get_all_names_with_ids(self) -> list[tuple[str, str]]:
        cursor = self.foods.find({}, {"food_name": 1, "food_id": 1, "_id": 0})
        docs = await cursor.to_list(length=5000)
        return [(d["food_name"], d["food_id"]) for d in docs]

    async def get_all_aliases_with_ids(self) -> list[tuple[str, str]]:
        cursor = self.aliases.find({}, {"alias": 1, "food_id": 1, "_id": 0})
        docs = await cursor.to_list(length=5000)
        return [(d["alias"], d["food_id"]) for d in docs]

    async def get_foods_by_ids(self, food_ids: list[str]) -> list[dict]:
        if not food_ids:
            return []
        cursor = self.foods.find({"food_id": {"$in": food_ids}}, {"_id": 0})
        docs = await cursor.to_list(length=len(food_ids))
        id_map = {d["food_id"]: d for d in docs}
        return [id_map[fid] for fid in food_ids if fid in id_map]

    # ------------------------------------------------------------------
    # Alias helpers
    # ------------------------------------------------------------------

    async def get_exact_alias(self, alias: str):
        return await self.aliases.find_one({"alias": alias.lower()}, {"_id": 0})

    async def get_aliases_for_food(self, food_id: str) -> list[dict]:
        cursor = self.aliases.find({"food_id": food_id}, {"_id": 0})
        return await cursor.to_list(length=100)

    # ------------------------------------------------------------------
    # Save learned live foods (Open Food Facts / USDA / AI Fallback)
    # ------------------------------------------------------------------

    async def save_learned_food(self, query_normalized: str, food_dict: dict):
        """Async save live-learned food with source attribution & overwrite protection.

        Deduplication rules:
        - If a VERIFIED local record exists for this food_name → reuse its
          food_id and only add aliases; never overwrite curated data.
        - If an UNVERIFIED learned record exists with the same food_name →
          reuse its food_id to prevent duplicates; update nutrition if new
          data is available.
        - Otherwise → insert new record.
        """
        if not food_dict:
            return

        food_name_clean = food_dict.get("food_name", "").strip().lower() or query_normalized.strip().lower()
        if not food_name_clean:
            return

        cals = float(food_dict.get("calories_kcal", 0.0) or 0.0)
        if cals <= 0:
            return  # Invalid live data validation

        # Check existing by food_name (prevents duplicates across food_id variants)
        existing = await self.foods.find_one({"food_name": food_name_clean})
        if existing and existing.get("is_verified", False):
            # Never overwrite trusted local data; just add alias
            food_id = existing["food_id"]
        else:
            # Reuse existing unverified food_id to prevent duplicates
            food_id = (
                (existing or {}).get("food_id")
                or food_dict.get("food_id")
                or f"food_live_{re.sub(r'[^\w]', '_', food_name_clean)}"
            )
            source = food_dict.get("data_source") or food_dict.get("source", "Live API Import")
            from import_to_mongodb import generate_food_schema_fields
            schema_fields = generate_food_schema_fields(food_name_clean, food_dict.get("category", ""), "", food_dict.get("serving_unit", ""))

            doc = {
                "food_id": food_id,
                "food_name": food_name_clean,
                "food_name_display": food_dict.get("food_name_display", food_name_clean.title()),
                "category": food_dict.get("category", "Learned Food"),
                "vegetarian_status": food_dict.get("vegetarian_status", "Veg"),
                "serving_size_g": float(food_dict.get("serving_size_g", 100.0)),
                "calories_per_100g": cals,
                "calories_kcal": cals,
                "protein_g": max(0.0, float(food_dict.get("protein_g", 0.0) or 0.0)),
                "carbs_g": max(0.0, float(food_dict.get("carbs_g", 0.0) or 0.0)),
                "fat_g": max(0.0, float(food_dict.get("fat_g", 0.0) or 0.0)),
                "fiber_g": max(0.0, float(food_dict.get("fiber_g", 0.0) or 0.0)),
                "quantity_units": schema_fields["quantity_units"],
                "quantity_options": schema_fields["quantity_options"],
                "preparation_variants": schema_fields["preparation_variants"],
                "variant_nutrition": schema_fields["variant_nutrition"],
                "data_source": source,
                "source": source,
                "is_verified": False,
                "is_learned": True,
            }
            await self.foods.update_one({"food_id": food_id}, {"$set": doc}, upsert=True)

        aliases_to_add = {query_normalized.strip().lower(), food_name_clean}
        for alias in aliases_to_add:
            # Only persist clean food-name-like aliases (see is_safe_learned_alias).
            if is_safe_learned_alias(alias):
                await self.aliases.update_one(
                    {"alias": alias},
                    {"$set": {"food_id": food_id, "alias": alias, "alias_type": "learned", "language": "en"}},
                    upsert=True
                )


# ------------------------------------------------------------------
# Sync version for search_engine (used inside sync context)
# ------------------------------------------------------------------

class FoodRepositorySync:
    """Synchronous version using pymongo — for search_engine fuzzy matching."""

    _names_cache: Optional[list[tuple[str, str]]] = None
    _aliases_cache: Optional[list[tuple[str, str]]] = None

    def __init__(self, db):
        self.foods = db["foods"]
        self.aliases = db["food_aliases"]

    def get_by_id(self, food_id: str):
        return self.foods.find_one({"food_id": food_id}, {"_id": 0})

    def get_by_exact_name(self, name: str):
        return self.foods.find_one({"food_name": name.strip().lower()}, {"_id": 0})

    def get_by_exact_alias(self, alias: str):
        alias_doc = self.aliases.find_one({"alias": alias.strip().lower()})
        if alias_doc:
            return self.get_by_id(alias_doc["food_id"])
        return None

    def get_all_names_with_ids(self) -> list[tuple[str, str]]:
        if FoodRepositorySync._names_cache is None:
            docs = list(self.foods.find({}, {"food_name": 1, "food_id": 1, "_id": 0}))
            FoodRepositorySync._names_cache = [(d["food_name"], d["food_id"]) for d in docs]
        return FoodRepositorySync._names_cache

    def get_all_aliases_with_ids(self) -> list[tuple[str, str]]:
        if FoodRepositorySync._aliases_cache is None:
            docs = list(self.aliases.find({}, {"alias": 1, "food_id": 1, "_id": 0}))
            FoodRepositorySync._aliases_cache = [(d["alias"], d["food_id"]) for d in docs]
        return FoodRepositorySync._aliases_cache

    def get_foods_by_ids(self, food_ids: list[str]) -> list[dict]:
        if not food_ids:
            return []
        docs = list(self.foods.find({"food_id": {"$in": food_ids}}, {"_id": 0}))
        id_map = {d["food_id"]: d for d in docs}
        return [id_map[fid] for fid in food_ids if fid in id_map]

    def save_learned_food_sync(self, query_normalized: str, food_dict: dict):
        """
        Sync save live-learned food with validation, source attribution & overwrite protection.

        Deduplication rules:
        - Verified local record by food_name → reuse food_id, add aliases only.
        - Unverified record already exists with same food_name → reuse its
          food_id to prevent duplicates across different external food_id slugs.
        - Otherwise → insert new record.
        """
        if not food_dict:
            return

        food_name_clean = food_dict.get("food_name", "").strip().lower() or query_normalized.strip().lower()
        if not food_name_clean:
            return

        cals = float(food_dict.get("calories_kcal", 0.0) or 0.0)
        if cals <= 0:
            return  # Live validation check

        existing = self.foods.find_one({"food_name": food_name_clean})
        if existing and existing.get("is_verified", False):
            food_id = existing["food_id"]
        else:
            # Reuse existing unverified food_id to prevent duplicates
            food_id = (
                (existing or {}).get("food_id")
                or food_dict.get("food_id")
                or f"food_live_{re.sub(r'[^\w]', '_', food_name_clean)}"
            )
            source = food_dict.get("data_source") or food_dict.get("source", "Live API Import")
            from import_to_mongodb import generate_food_schema_fields
            schema_fields = generate_food_schema_fields(food_name_clean, food_dict.get("category", ""), "", food_dict.get("serving_unit", ""))

            doc = {
                "food_id": food_id,
                "food_name": food_name_clean,
                "food_name_display": food_dict.get("food_name_display", food_name_clean.title()),
                "category": food_dict.get("category", "Learned Food"),
                "vegetarian_status": food_dict.get("vegetarian_status", "Veg"),
                "serving_size_g": float(food_dict.get("serving_size_g", 100.0)),
                "calories_per_100g": cals,
                "calories_kcal": cals,
                "protein_g": max(0.0, float(food_dict.get("protein_g", 0.0) or 0.0)),
                "carbs_g": max(0.0, float(food_dict.get("carbs_g", 0.0) or 0.0)),
                "fat_g": max(0.0, float(food_dict.get("fat_g", 0.0) or 0.0)),
                "fiber_g": max(0.0, float(food_dict.get("fiber_g", 0.0) or 0.0)),
                "quantity_units": schema_fields["quantity_units"],
                "quantity_options": schema_fields["quantity_options"],
                "preparation_variants": schema_fields["preparation_variants"],
                "variant_nutrition": schema_fields["variant_nutrition"],
                "data_source": source,
                "source": source,
                "is_verified": False,
                "is_learned": True,
            }
            self.foods.update_one({"food_id": food_id}, {"$set": doc}, upsert=True)

        aliases_to_add = {query_normalized.strip().lower(), food_name_clean}
        for alias in aliases_to_add:
            # Only persist clean food-name-like aliases. Raw sentence fragments
            # ("aje bhakhari", "chai khadi") would poison future fuzzy lookups.
            if is_safe_learned_alias(alias):
                self.aliases.update_one(
                    {"alias": alias},
                    {"$set": {"food_id": food_id, "alias": alias, "alias_type": "learned", "language": "en"}},
                    upsert=True
                )
        FoodRepositorySync._names_cache = None
        FoodRepositorySync._aliases_cache = None
