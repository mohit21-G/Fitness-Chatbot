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

_OFF_CIRCUIT_BREAKER_UNTIL: float = 0.0
_USDA_CIRCUIT_BREAKER_UNTIL: float = 0.0

USER_AGENT = os.getenv("OPEN_FOOD_FACTS_USER_AGENT", "FitnessAIChatbot/1.0 (contact@fitnessai.local)")
TIMEOUT_CONFIG = httpx.Timeout(connect=1.5, read=2.5, write=1.5, pool=1.5)


class ExternalFoodAPIService:
    """
    Service for querying Open Food Facts (Tier 2) and USDA FoodData Central (Tier 3)
    with circuit-breakers, resilient timeouts, validation, and MongoDB learning.
    """

    def __init__(self, usda_api_key: Optional[str] = None):
        from database import get_settings
        settings = get_settings()
        self.usda_api_key = (
            usda_api_key
            or getattr(settings, "USDA_API_KEY", None)
            or os.getenv("USDA_API_KEY")
            or "DEMO_KEY"
        )

    # ------------------------------------------------------------------
    # Public Unified Search API
    # ------------------------------------------------------------------

    async def search_open_food_facts(self, query: str) -> Optional[dict]:
        """
        Tier 2: Search Open Food Facts API (Packaged & Branded Foods).
        Returns a standardized food dict or None if no match / API error / circuit broken.
        """
        global _OFF_CIRCUIT_BREAKER_UNTIL
        q_norm = query.strip().lower()
        if not q_norm:
            return None

        # Check circuit breaker
        if time.time() < _OFF_CIRCUIT_BREAKER_UNTIL:
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
            "page_size": "3",
        }
        headers = {"User-Agent": USER_AGENT}

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
                response = await client.get(url, params=params, headers=headers)

            if response.status_code == 503:
                _OFF_CIRCUIT_BREAKER_UNTIL = time.time() + 120.0
                _API_CACHE[cache_key] = (None, time.time())
                return None

            if response.status_code != 200:
                _API_CACHE[cache_key] = (None, time.time())
                return None

            data = response.json()
            products = data.get("products", [])
            if not products:
                _API_CACHE[cache_key] = (None, time.time())
                return None

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
                food_dict = {
                    "food_name": raw_name.lower(),
                    "food_name_display": raw_name.strip().title(),
                    "category": "Packaged Food",
                    "serving_size_g": 100.0,
                    "serving_unit": "g",
                    "calories_kcal": round(float(cals), 1),
                    "calories_per_100g": round(float(cals), 1),
                    "protein_g": round(protein, 1),
                    "carbs_g": round(carbs, 1),
                    "fat_g": round(fat, 1),
                    "fiber_g": round(fiber, 1),
                    "source": "Open Food Facts",
                    "data_source": "Open Food Facts",
                    "is_verified": False,
                }
                _API_CACHE[cache_key] = (food_dict, time.time())
                return food_dict

        except Exception:
            pass

        _API_CACHE[cache_key] = (None, time.time())
        return None

    async def search_usda(self, query: str) -> Optional[dict]:
        """
        Tier 3: Search USDA FoodData Central API (Generic / Raw Foods).
        Returns a standardized food dict or None if no match / API error / circuit broken.
        """
        global _USDA_CIRCUIT_BREAKER_UNTIL
        q_norm = query.strip().lower()
        if not q_norm:
            return None

        # Check circuit breaker
        if time.time() < _USDA_CIRCUIT_BREAKER_UNTIL:
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
            "pageSize": "3",
            "api_key": self.usda_api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
                response = await client.get(url, params=params)

            if response.status_code == 429:
                _USDA_CIRCUIT_BREAKER_UNTIL = time.time() + 180.0
                _API_CACHE[cache_key] = (None, time.time())
                return None

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

                food_dict = {
                    "food_name": description.lower(),
                    "food_name_display": description.strip().title(),
                    "category": "Generic Food",
                    "serving_size_g": 100.0,
                    "serving_unit": "g",
                    "calories_kcal": round(cals, 1),
                    "calories_per_100g": round(cals, 1),
                    "protein_g": round(protein, 1),
                    "carbs_g": round(carbs, 1),
                    "fat_g": round(fat, 1),
                    "fiber_g": round(fiber, 1),
                    "source": "USDA FoodData Central",
                    "data_source": "USDA FoodData Central",
                    "is_verified": False,
                }
                _API_CACHE[cache_key] = (food_dict, time.time())
                return food_dict

        except Exception:
            pass

        _API_CACHE[cache_key] = (None, time.time())
        return None

    def query_live_food_sync(self, term: str, repo=None) -> tuple[Optional[dict], Optional[str]]:
        """
        Synchronous live query with validation and optional MongoDB persistence.
        """
        global _USDA_CIRCUIT_BREAKER_UNTIL, _OFF_CIRCUIT_BREAKER_UNTIL
        term_clean = term.strip().lower()
        if not term_clean:
            return None, None

        # Check in-memory cache
        for src in ("usda", "open_food_facts"):
            if (src, term_clean) in _API_CACHE:
                cached_doc, ts = _API_CACHE[(src, term_clean)]
                if cached_doc and time.time() - ts < CACHE_TTL_SECONDS:
                    return cached_doc, cached_doc.get("source", src)

        # 1. USDA FoodData Central (sync)
        if time.time() >= _USDA_CIRCUIT_BREAKER_UNTIL:
            try:
                url = "https://api.nal.usda.gov/fdc/v1/foods/search"
                params = {"query": term_clean, "pageSize": "2", "api_key": self.usda_api_key}
                with httpx.Client(timeout=TIMEOUT_CONFIG) as client:
                    resp = client.get(url, params=params)
                if resp.status_code == 429:
                    _USDA_CIRCUIT_BREAKER_UNTIL = time.time() + 180.0
                elif resp.status_code == 200:
                    data = resp.json()
                    for f in data.get("foods", []):
                        nuts = {n.get("nutrientName", "").lower(): float(n.get("value", 0.0)) for n in f.get("foodNutrients", [])}
                        cals = float(nuts.get("energy", 0.0))
                        if cals > 0:
                            from nutrition_validator import GenericNutritionValidator
                            desc = (f.get("description") or term_clean).strip().lower()
                            if GenericNutritionValidator.is_nutritionally_plausible(desc, cals, "Generic Food"):
                                food_dict = {
                                    "food_name": desc,
                                    "food_name_display": desc.title(),
                                    "category": "Generic Food",
                                    "calories_kcal": round(cals, 1),
                                    "calories_per_100g": round(cals, 1),
                                    "protein_g": round(nuts.get("protein", 0.0), 1),
                                    "carbs_g": round(nuts.get("carbohydrate, by difference", 0.0), 1),
                                    "fat_g": round(nuts.get("total lipid (fat)", 0.0), 1),
                                    "fiber_g": round(nuts.get("fiber, total dietary", 0.0), 1),
                                    "data_source": "USDA FoodData Central",
                                    "source": "USDA FoodData Central",
                                }
                                _API_CACHE[("usda", term_clean)] = (food_dict, time.time())
                                if repo:
                                    try:
                                        repo.save_learned_food_sync(term_clean, food_dict)
                                    except Exception:
                                        pass
                                return food_dict, "USDA FoodData Central"
            except Exception:
                pass

        # 2. Open Food Facts (sync)
        if time.time() >= _OFF_CIRCUIT_BREAKER_UNTIL:
            try:
                url = "https://world.openfoodfacts.org/cgi/search.pl"
                params = {"search_terms": term_clean, "search_simple": "1", "action": "process", "json": "1", "page_size": "2"}
                headers = {"User-Agent": USER_AGENT}
                with httpx.Client(timeout=TIMEOUT_CONFIG) as client:
                    resp = client.get(url, params=params, headers=headers)
                if resp.status_code == 503:
                    _OFF_CIRCUIT_BREAKER_UNTIL = time.time() + 120.0
                elif resp.status_code == 200:
                    data = resp.json()
                    for p in data.get("products", []):
                        nut = p.get("nutriments", {})
                        cals = float(nut.get("energy-kcal_100g") or nut.get("energy-kcal") or 0.0)
                        if cals > 0:
                            from nutrition_validator import GenericNutritionValidator
                            pname = (p.get("product_name") or term_clean).strip().lower()
                            if GenericNutritionValidator.is_nutritionally_plausible(pname, cals, "Packaged Food"):
                                food_dict = {
                                    "food_name": pname,
                                    "food_name_display": pname.title(),
                                    "category": "Packaged Food",
                                    "calories_kcal": round(cals, 1),
                                    "calories_per_100g": round(cals, 1),
                                    "protein_g": round(float(nut.get("proteins_100g") or 0.0), 1),
                                    "carbs_g": round(float(nut.get("carbohydrates_100g") or 0.0), 1),
                                    "fat_g": round(float(nut.get("fat_100g") or 0.0), 1),
                                    "fiber_g": round(float(nut.get("fiber_100g") or 0.0), 1),
                                    "data_source": "Open Food Facts",
                                    "source": "Open Food Facts",
                                }
                                _API_CACHE[("open_food_facts", term_clean)] = (food_dict, time.time())
                                if repo:
                                    try:
                                        repo.save_learned_food_sync(term_clean, food_dict)
                                    except Exception:
                                        pass
                                return food_dict, "Open Food Facts"
            except Exception:
                pass

        _API_CACHE[("usda", term_clean)] = (None, time.time())
        _API_CACHE[("open_food_facts", term_clean)] = (None, time.time())
        return None, None
