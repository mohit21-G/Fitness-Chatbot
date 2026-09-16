"""
External Food API Service — Open Food Facts & USDA FoodData Central
Implements live online nutrition lookups with caching, 5s timeouts,
and graceful fallback.
"""
from __future__ import annotations

import os
import re
import time
import httpx
from typing import Optional

# In-memory search cache: (source, query_lower) -> (food_dict, timestamp)
_API_CACHE: dict[tuple[str, str], tuple[Optional[dict], float]] = {}
CACHE_TTL_SECONDS = 3600  # 1 hour

USER_AGENT = os.getenv("OPEN_FOOD_FACTS_USER_AGENT", "FitnessAIChatbot/1.0 (contact@fitnessai.local)")


class ExternalFoodAPIService:
    """
    Service for querying Open Food Facts (Tier 2) and USDA FoodData Central (Tier 3).
    """

    def __init__(self, usda_api_key: Optional[str] = None):
        from database import get_settings
        settings = get_settings()
        self.usda_api_key = (
            usda_api_key
            or getattr(settings, "USDA_API_KEY", None)
            or os.getenv("USDA_API_KEY")
        )

    # ------------------------------------------------------------------
    # Public Unified Search API
    # ------------------------------------------------------------------

    async def search_open_food_facts(self, query: str) -> Optional[dict]:
        """
        Tier 2: Search Open Food Facts API (Packaged & Branded Foods).
        Returns a standardized food dict or None if no match / API error.
        """
        q_norm = query.strip().lower()
        if not q_norm:
            return None

        # Check cache
        cache_key = ("open_food_facts", q_norm)
        if cache_key in _API_CACHE:
            cached_food, ts = _API_CACHE[cache_key]
            if time.time() - ts < CACHE_TTL_SECONDS:
                return cached_food

        url = "https://world.openfoodfacts.org/cgi/search.pl"
        params = {
            "search_terms": q_norm,
            "search_simple": "1",
            "action": "process",
            "json": "1",
            "page_size": "5",
        }
        headers = {"User-Agent": USER_AGENT}

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url, params=params, headers=headers)

            if response.status_code != 200:
                _API_CACHE[cache_key] = (None, time.time())
                return None

            data = response.json()
            products = data.get("products", [])
            if not products:
                _API_CACHE[cache_key] = (None, time.time())
                return None

            # Pick best product candidate with nutrition data
            for prod in products:
                nutr = prod.get("nutriments", {})
                if not nutr:
                    continue

                cals = (
                    nutr.get("energy-kcal_100g")
                    or nutr.get("energy-kcal_serving")
                    or (nutr.get("energy_100g", 0) / 4.184 if nutr.get("energy_100g") else 0)
                )
                if not cals or cals <= 0:
                    continue

                protein = float(nutr.get("proteins_100g") or nutr.get("proteins_serving") or 0.0)
                carbs = float(nutr.get("carbohydrates_100g") or nutr.get("carbohydrates_serving") or 0.0)
                fat = float(nutr.get("fat_100g") or nutr.get("fat_serving") or 0.0)
                fiber = float(nutr.get("fiber_100g") or nutr.get("fiber_serving") or 0.0)

                raw_name = prod.get("product_name_en") or prod.get("product_name") or q_norm
                serving_size = float(prod.get("serving_quantity") or 100.0)

                slug = re.sub(r"[^\w]", "_", raw_name.lower()).strip("_")
                food_dict = {
                    "food_id": f"off_{slug}",
                    "food_name": raw_name.lower(),
                    "food_name_display": raw_name.strip().title(),
                    "category": "Packaged Food",
                    "serving_size_g": 100.0,
                    "serving_unit": "g",
                    "calories_kcal": round(float(cals), 1),
                    "protein_g": round(protein, 1),
                    "carbs_g": round(carbs, 1),
                    "fat_g": round(fat, 1),
                    "fiber_g": round(fiber, 1),
                    "source": "Open Food Facts",
                    "is_ai_estimated": False,
                }
                _API_CACHE[cache_key] = (food_dict, time.time())
                return food_dict

        except Exception:
            pass  # Graceful fallback on timeout/error

        _API_CACHE[cache_key] = (None, time.time())
        return None

    async def search_usda(self, query: str) -> Optional[dict]:
        """
        Tier 3: Search USDA FoodData Central API (Generic / Raw Foods).
        Returns a standardized food dict or None if no match / API error.
        """
        q_norm = query.strip().lower()
        if not q_norm:
            return None

        # Check cache
        cache_key = ("usda", q_norm)
        if cache_key in _API_CACHE:
            cached_food, ts = _API_CACHE[cache_key]
            if time.time() - ts < CACHE_TTL_SECONDS:
                return cached_food

        url = "https://api.nal.usda.gov/fdc/v1/foods/search"
        params = {
            "query": q_norm,
            "pageSize": "5",
            "api_key": self.usda_api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url, params=params)

            if response.status_code != 200:
                _API_CACHE[cache_key] = (None, time.time())
                return None

            data = response.json()
            foods = data.get("foods", [])
            if not foods:
                _API_CACHE[cache_key] = (None, time.time())
                return None

            for f_item in foods:
                description = f_item.get("description", q_norm)
                nutrients = f_item.get("foodNutrients", [])

                cals, protein, carbs, fat, fiber = 0.0, 0.0, 0.0, 0.0, 0.0
                for n in nutrients:
                    name = n.get("nutrientName", "").lower()
                    val = float(n.get("value", 0.0))
                    if "energy" in name and ("kcal" in n.get("unitName", "").lower() or cals == 0):
                        cals = val
                    elif "protein" in name:
                        protein = val
                    elif "carbohydrate" in name:
                        carbs = val
                    elif "total lipid" in name or "fat" in name:
                        fat = val
                    elif "fiber" in name:
                        fiber = val

                if cals <= 0 and protein == 0 and carbs == 0 and fat == 0:
                    continue

                serving_size = float(f_item.get("servingSize") or 100.0)
                unit = f_item.get("servingSizeUnit") or "g"

                slug = re.sub(r"[^\w]", "_", description.lower()).strip("_")
                food_dict = {
                    "food_id": f"usda_{slug}",
                    "food_name": description.lower(),
                    "food_name_display": description.strip().title(),
                    "category": "USDA Generic",
                    "serving_size_g": 100.0,
                    "serving_unit": "g",
                    "calories_kcal": round(cals, 1),
                    "protein_g": round(protein, 1),
                    "carbs_g": round(carbs, 1),
                    "fat_g": round(fat, 1),
                    "fiber_g": round(fiber, 1),
                    "source": "USDA FoodData",
                    "is_ai_estimated": False,
                }
                _API_CACHE[cache_key] = (food_dict, time.time())
                return food_dict

        except Exception:
            pass  # Graceful fallback on timeout/error

        _API_CACHE[cache_key] = (None, time.time())
        return None
