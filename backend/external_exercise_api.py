"""
External Exercise API Service — ExerciseDB, wger & 2024 Compendium of Physical Activities.

Resolution role (called only when the local MongoDB `exercises` collection cannot
confidently resolve an exercise):

    MongoDB → ExerciseDB → wger → Compendium → (AI fallback in engine)

Design mirrors external_food_api.py:
  - In-memory TTL cache keyed by (source, query) — caches hits AND misses.
  - httpx.AsyncClient per call, 5s timeout, graceful try/except fallback.
  - Every source returns a *standardized exercise dict* shaped for
    exercise_calculator.ExerciseCalculator (calories_per_unit_min/avg/max,
    measurement_unit, typical_duration_min, category, display fields).
  - Deduplication of results before returning/merging.
  - No source/API details are ever placed into user-facing text; `source` is
    kept internal only.

MET → calories conversion (Compendium):
    kcal_per_minute = MET * 3.5 * weight_kg / 200
ExerciseDB / wger provide the exercise identity (name, category, equipment,
muscle group) but not calories, so those results are paired with a Compendium
MET lookup (falling back to a category-based MET) to produce a calorie rate.
"""
from __future__ import annotations

import os
import re
import time
import difflib
import httpx
from typing import Optional

# In-memory search cache: (source, query_lower) -> (exercise_dict|None, timestamp)
_EX_API_CACHE: dict[tuple[str, str], tuple[Optional[dict], float]] = {}
CACHE_TTL_SECONDS = 3600  # 1 hour

# Default body weight used for MET → kcal when the user's weight is unknown.
DEFAULT_WEIGHT_KG = 70.0


# ---------------------------------------------------------------------------
# 2024 Compendium of Physical Activities — embedded MET table
# ---------------------------------------------------------------------------
# A curated subset of the 2024 Compendium covering gym/strength, yoga, sports,
# cardio and general activities. MET values follow the published compendium.
# This lets the Compendium tier resolve common exercises fully offline (no
# network dependency), while the network ExerciseDB/wger tiers enrich identity
# metadata (equipment / muscle group) when reachable.
#
# Each entry: canonical_name -> (MET, category, measurement_unit)
#   measurement_unit is what the user most naturally reports the activity in.
COMPENDIUM_MET: dict[str, tuple[float, str, str]] = {
    # ── Cardio ────────────────────────────────────────────────────────────
    "running": (9.8, "Cardio", "minutes"),
    "jogging": (7.0, "Cardio", "minutes"),
    "walking": (3.5, "Cardio", "minutes"),
    "brisk walking": (4.3, "Cardio", "minutes"),
    "cycling": (7.5, "Cardio", "minutes"),
    "swimming": (7.0, "Cardio", "minutes"),
    "skipping": (11.0, "Cardio", "minutes"),
    "jump rope": (11.0, "Cardio", "minutes"),
    "rowing": (7.0, "Cardio", "minutes"),
    "elliptical": (5.0, "Cardio", "minutes"),
    "stair climbing": (8.8, "Cardio", "minutes"),
    "hiking": (6.0, "Cardio", "minutes"),
    "dancing": (5.0, "Cardio", "minutes"),
    "zumba": (6.5, "Cardio", "minutes"),
    "aerobics": (7.3, "Cardio", "minutes"),
    "treadmill": (8.0, "Cardio", "minutes"),
    # ── Strength / Gym ──────────────────────────────────────────────────────
    "push-ups": (8.0, "Strength", "reps"),
    "pushups": (8.0, "Strength", "reps"),
    "pull-ups": (8.0, "Strength", "reps"),
    "pullups": (8.0, "Strength", "reps"),
    "squats": (5.0, "Strength", "reps"),
    "lunges": (4.0, "Strength", "reps"),
    "burpees": (8.0, "Strength", "reps"),
    "crunches": (3.8, "Strength", "reps"),
    "sit-ups": (3.8, "Strength", "reps"),
    "situps": (3.8, "Strength", "reps"),
    "plank": (3.3, "Strength", "minutes"),
    "deadlifts": (6.0, "Strength", "reps"),
    "bench press": (5.0, "Strength", "reps"),
    "bicep curls": (3.5, "Strength", "reps"),
    "shoulder press": (5.0, "Strength", "reps"),
    "leg press": (5.0, "Strength", "reps"),
    "weight lifting": (6.0, "Strength", "minutes"),
    "resistance training": (5.0, "Strength", "minutes"),
    "mountain climbers": (8.0, "Strength", "reps"),
    "jumping jacks": (8.0, "Strength", "reps"),
    "calisthenics": (8.0, "Strength", "minutes"),
    # ── Yoga / Flexibility ──────────────────────────────────────────────────
    "yoga": (3.0, "Yoga", "minutes"),
    "hatha yoga": (2.5, "Yoga", "minutes"),
    "power yoga": (4.0, "Yoga", "minutes"),
    "surya namaskar": (4.0, "Yoga", "reps"),
    "sun salutation": (4.0, "Yoga", "reps"),
    "pranayama": (2.0, "Yoga", "minutes"),
    "meditation": (1.3, "Yoga", "minutes"),
    "stretching": (2.3, "Yoga", "minutes"),
    "pilates": (3.0, "Yoga", "minutes"),
    # ── Sports ────────────────────────────────────────────────────────────
    "cricket": (5.0, "Sports", "minutes"),
    "football": (7.0, "Sports", "minutes"),
    "soccer": (7.0, "Sports", "minutes"),
    "basketball": (6.5, "Sports", "minutes"),
    "badminton": (5.5, "Sports", "minutes"),
    "tennis": (7.3, "Sports", "minutes"),
    "table tennis": (4.0, "Sports", "minutes"),
    "volleyball": (4.0, "Sports", "minutes"),
    "kabaddi": (7.0, "Sports", "minutes"),
    "boxing": (7.8, "Sports", "minutes"),
    "martial arts": (10.3, "Sports", "minutes"),
    "kho kho": (7.0, "Sports", "minutes"),
    "hockey": (7.8, "Sports", "minutes"),
    "golf": (4.8, "Sports", "minutes"),
    # ── General activities ────────────────────────────────────────────────
    "gardening": (3.8, "Activity", "minutes"),
    "cleaning": (3.3, "Activity", "minutes"),
    "climbing stairs": (8.8, "Activity", "minutes"),
    "gym workout": (6.0, "Fitness", "minutes"),
    "workout": (6.0, "Fitness", "minutes"),
    "cardio": (7.0, "Cardio", "minutes"),
}

# Category-level MET fallback when an exercise identity is known (from
# ExerciseDB/wger) but no specific Compendium MET exists.
CATEGORY_MET_FALLBACK: dict[str, tuple[float, str]] = {
    "cardio": (7.0, "minutes"),
    "strength": (6.0, "reps"),
    "weight training": (6.0, "reps"),
    "waist": (5.0, "reps"),
    "chest": (5.0, "reps"),
    "back": (5.0, "reps"),
    "legs": (5.0, "reps"),
    "upper legs": (5.0, "reps"),
    "lower legs": (5.0, "reps"),
    "shoulders": (5.0, "reps"),
    "upper arms": (3.5, "reps"),
    "lower arms": (3.5, "reps"),
    "arms": (3.5, "reps"),
    "yoga": (3.0, "minutes"),
    "stretching": (2.3, "minutes"),
    "sports": (6.0, "minutes"),
    "plyometrics": (8.0, "reps"),
}

# Reps-based activities: assume this many reps per minute so a MET (kcal/min)
# can be re-expressed as kcal/rep.
REPS_PER_MINUTE = 25.0


def _slug(name: str) -> str:
    return re.sub(r"[^\w]", "_", name.strip().lower()).strip("_")


def met_to_calories_per_unit(
    met: float,
    measurement_unit: str,
    weight_kg: float = DEFAULT_WEIGHT_KG,
) -> tuple[float, float, float]:
    """
    Convert a MET value into (cal_per_unit_min, cal_per_unit_avg, cal_per_unit_max)
    for the given measurement unit.

      kcal/min = MET * 3.5 * weight_kg / 200
    For reps-based units we divide by REPS_PER_MINUTE to get kcal/rep.
    min/max follow the existing gym-CSV convention of avg ±20%.
    """
    kcal_per_min = met * 3.5 * float(weight_kg) / 200.0
    if measurement_unit == "reps":
        avg = kcal_per_min / REPS_PER_MINUTE
    else:
        # minutes / session / other duration-like units
        avg = kcal_per_min
    avg = round(max(avg, 0.01), 3)
    return round(avg * 0.8, 3), avg, round(avg * 1.2, 3)


def _build_exercise_doc(
    canonical_name: str,
    display_name: str,
    met: float,
    category: str,
    measurement_unit: str,
    source: str,
    weight_kg: float = DEFAULT_WEIGHT_KG,
    equipment: Optional[str] = None,
    muscle_group: Optional[str] = None,
    aliases: Optional[list[str]] = None,
) -> dict:
    """Assemble a standardized exercise document shaped for ExerciseCalculator."""
    cal_min, cal_avg, cal_max = met_to_calories_per_unit(met, measurement_unit, weight_kg)
    name_norm = canonical_name.strip().lower()
    typical = 30.0 if measurement_unit != "reps" else 15.0
    return {
        "exercise_id": f"{source}_{_slug(name_norm)}",
        "exercise_name": name_norm,
        "exercise_name_display": display_name.strip().title(),
        "category": category or "Fitness",
        "measurement_unit": measurement_unit,
        "calories_per_unit_min": cal_min,
        "calories_per_unit_avg": cal_avg,
        "calories_per_unit_max": cal_max,
        # The body weight these per-unit rates were baked at, so the calculator
        # can rescale burn to the logging user's ACTUAL weight without
        # double-counting the weight that was already applied here.
        "reference_weight_kg": round(float(weight_kg), 1),
        "typical_duration_min": typical,
        "met": round(float(met), 2),
        "equipment": equipment or "",
        "muscle_group": muscle_group or "",
        "aliases": aliases or [],
        # internal only — never shown to the user
        "source": source,
        "is_ai_estimated": False,
    }


def _compendium_lookup(query: str) -> Optional[tuple[str, float, str, str]]:
    """
    Resolve a query against the embedded Compendium MET table.
    Returns (canonical_name, met, category, measurement_unit) or None.
    Exact match first, then fuzzy (difflib) at a conservative cutoff.
    """
    q = query.strip().lower()
    if not q:
        return None
    if q in COMPENDIUM_MET:
        met, cat, unit = COMPENDIUM_MET[q]
        return q, met, cat, unit
    # substring / contains match (e.g. "morning yoga" -> "yoga")
    for name, (met, cat, unit) in COMPENDIUM_MET.items():
        if name in q or q in name:
            return name, met, cat, unit
    # fuzzy match
    close = difflib.get_close_matches(q, list(COMPENDIUM_MET.keys()), n=1, cutoff=0.84)
    if close:
        name = close[0]
        # For short queries (<= 5 chars), do not allow length differences > 1
        # (e.g. prevents 5-char "plate" from matching 7-char "pilates" while allowing real typos)
        if len(q) <= 5 and abs(len(q) - len(name)) > 1:
            return None
        met, cat, unit = COMPENDIUM_MET[name]
        return name, met, cat, unit
    return None


class ExternalExerciseAPIService:
    """
    Queries external exercise sources when the local DB cannot resolve an
    exercise. Order: ExerciseDB (identity) → wger (identity) → Compendium (MET).

    ExerciseDB/wger supply identity + muscle/equipment; the Compendium supplies
    the MET used for calories. Every method returns a calculator-ready dict or
    None, and caches results (hits + misses) to avoid repeat API calls.
    """

    def __init__(
        self,
        exercisedb_api_key: Optional[str] = None,
        exercisedb_host: Optional[str] = None,
        wger_base_url: Optional[str] = None,
    ):
        try:
            from database import get_settings
            settings = get_settings()
        except Exception:
            settings = None

        self.exercisedb_api_key = (
            exercisedb_api_key
            or getattr(settings, "EXERCISEDB_API_KEY", None)
            or os.getenv("EXERCISEDB_API_KEY", "")
        )
        self.exercisedb_host = (
            exercisedb_host
            or getattr(settings, "EXERCISEDB_HOST", None)
            or os.getenv("EXERCISEDB_HOST", "exercisedb.p.rapidapi.com")
        )
        self.wger_base_url = (
            wger_base_url
            or getattr(settings, "WGER_BASE_URL", None)
            or os.getenv("WGER_BASE_URL", "https://wger.de")
        )

    # ------------------------------------------------------------------
    # Tier 1: ExerciseDB (gym / strength identity)
    # ------------------------------------------------------------------
    async def search_exercisedb(self, query: str, weight_kg: float = DEFAULT_WEIGHT_KG) -> Optional[dict]:
        q_norm = query.strip().lower()
        if not q_norm:
            return None
        cache_key = ("exercisedb", q_norm)
        cached = self._cache_get(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        # ExerciseDB requires a RapidAPI key. Without one, skip gracefully.
        if not self.exercisedb_api_key:
            return self._cache_put(cache_key, None)

        url = f"https://{self.exercisedb_host}/exercises/name/{q_norm}"
        headers = {
            "X-RapidAPI-Key": self.exercisedb_api_key,
            "X-RapidAPI-Host": self.exercisedb_host,
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers=headers, params={"limit": "5"})
            if resp.status_code != 200:
                return self._cache_put(cache_key, None)
            results = resp.json()
            if not isinstance(results, list) or not results:
                return self._cache_put(cache_key, None)
            top = results[0]
            name = (top.get("name") or q_norm).strip()
            body_part = (top.get("bodyPart") or "").strip()
            target = (top.get("target") or "").strip()
            equipment = (top.get("equipment") or "").strip()
            met, category, unit = self._met_for_identity(name, body_part or target)
            doc = _build_exercise_doc(
                canonical_name=name, display_name=name, met=met,
                category=category, measurement_unit=unit, source="exercisedb",
                weight_kg=weight_kg, equipment=equipment,
                muscle_group=target or body_part,
            )
            return self._cache_put(cache_key, doc)
        except Exception:
            return self._cache_put(cache_key, None)

    # ------------------------------------------------------------------
    # Tier 2: wger (general fitness identity)
    # ------------------------------------------------------------------
    async def search_wger(self, query: str, weight_kg: float = DEFAULT_WEIGHT_KG) -> Optional[dict]:
        q_norm = query.strip().lower()
        if not q_norm:
            return None
        cache_key = ("wger", q_norm)
        cached = self._cache_get(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        url = f"{self.wger_base_url.rstrip('/')}/api/v2/exercise/search/"
        params = {"term": q_norm, "language": "english", "format": "json"}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, params=params)
            if resp.status_code != 200:
                return self._cache_put(cache_key, None)
            data = resp.json()
            suggestions = data.get("suggestions") or []
            if not suggestions:
                return self._cache_put(cache_key, None)
            top = suggestions[0].get("data", {}) or {}
            name = (top.get("name") or q_norm).strip()
            category = (top.get("category") or "").strip()
            met, cat, unit = self._met_for_identity(name, category)
            doc = _build_exercise_doc(
                canonical_name=name, display_name=name, met=met,
                category=cat, measurement_unit=unit, source="wger",
                weight_kg=weight_kg, muscle_group=category,
            )
            return self._cache_put(cache_key, doc)
        except Exception:
            return self._cache_put(cache_key, None)

    # ------------------------------------------------------------------
    # Tier 3: Compendium of Physical Activities (MET values — offline)
    # ------------------------------------------------------------------
    async def search_compendium(self, query: str, weight_kg: float = DEFAULT_WEIGHT_KG) -> Optional[dict]:
        q_norm = query.strip().lower()
        if not q_norm:
            return None
        cache_key = ("compendium", q_norm)
        cached = self._cache_get(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        hit = _compendium_lookup(q_norm)
        if not hit:
            return self._cache_put(cache_key, None)
        canonical, met, category, unit = hit
        doc = _build_exercise_doc(
            canonical_name=canonical, display_name=canonical, met=met,
            category=category, measurement_unit=unit, source="compendium",
            weight_kg=weight_kg,
        )
        return self._cache_put(cache_key, doc)

    # ------------------------------------------------------------------
    # Unified resolver: ExerciseDB → wger → Compendium (with dedupe/merge)
    # ------------------------------------------------------------------
    async def resolve(self, query: str, weight_kg: float = DEFAULT_WEIGHT_KG) -> Optional[dict]:
        """
        Try the external tiers in order and return a single best exercise doc,
        or None if none resolve. Identity from ExerciseDB/wger is enriched with
        a Compendium MET so the calorie rate is well-grounded; duplicate results
        are merged (identity + best MET) rather than returned twice.
        """
        q_norm = query.strip().lower()
        if not q_norm:
            return None

        candidates: list[dict] = []
        for fetch in (self.search_exercisedb, self.search_wger, self.search_compendium):
            try:
                doc = await fetch(q_norm, weight_kg)
            except Exception:
                doc = None
            if doc:
                candidates.append(doc)

        if not candidates:
            return None

        merged = self._dedupe_and_merge(candidates, q_norm, weight_kg)
        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _met_for_identity(self, name: str, group: str) -> tuple[float, str, str]:
        """
        Given an exercise name + body-part/category from an identity source,
        derive (met, category, measurement_unit). Prefers a specific Compendium
        entry; falls back to a category-level MET.
        """
        comp = _compendium_lookup(name)
        if comp:
            _, met, cat, unit = comp
            return met, cat, unit
        g = (group or "").strip().lower()
        if g in CATEGORY_MET_FALLBACK:
            met, unit = CATEGORY_MET_FALLBACK[g]
            cat = "Cardio" if unit == "minutes" and g == "cardio" else (
                "Yoga" if g in ("yoga", "stretching") else
                "Sports" if g == "sports" else "Strength"
            )
            return met, cat, unit
        # Generic strength default
        return 5.0, "Strength", "reps"

    def _dedupe_and_merge(self, candidates: list[dict], query: str, weight_kg: float) -> dict:
        """
        Deduplicate candidate docs (same normalized exercise name) and merge
        their best fields: prefer an identity source (ExerciseDB/wger) for the
        display name / equipment / muscle group, and the most specific MET
        (Compendium exact match) for calories.
        """
        # Group by normalized name.
        by_name: dict[str, list[dict]] = {}
        for d in candidates:
            by_name.setdefault(d["exercise_name"], []).append(d)

        # Pick the group whose name best matches the query.
        best_name = min(
            by_name.keys(),
            key=lambda n: (
                0 if n == query else
                1 if (n in query or query in n) else
                2
            ),
        )
        group = by_name[best_name]

        # Base: prefer identity source order exercisedb > wger > compendium for
        # metadata, but take MET/calories from a compendium entry if present.
        priority = {"exercisedb": 0, "wger": 1, "compendium": 2}
        group_sorted = sorted(group, key=lambda d: priority.get(d.get("source"), 9))
        base = dict(group_sorted[0])

        comp_entry = next((d for d in group if d.get("source") == "compendium"), None)
        if comp_entry:
            for f in ("calories_per_unit_min", "calories_per_unit_avg",
                      "calories_per_unit_max", "measurement_unit", "met",
                      "category", "typical_duration_min"):
                if comp_entry.get(f) not in (None, "", 0):
                    base[f] = comp_entry[f]

        # Collect aliases from all candidates (deduped).
        aliases = set(base.get("aliases") or [])
        for d in group:
            aliases.update(d.get("aliases") or [])
            if d.get("exercise_name"):
                aliases.add(d["exercise_name"])
        aliases.add(query)
        base["aliases"] = sorted(a for a in aliases if a)
        return base

    # -- cache primitives --
    @staticmethod
    def _cache_get(key: tuple[str, str]):
        entry = _EX_API_CACHE.get(key)
        if entry is None:
            return _CACHE_MISS
        value, ts = entry
        if time.time() - ts < CACHE_TTL_SECONDS:
            return value
        _EX_API_CACHE.pop(key, None)
        return _CACHE_MISS

    @staticmethod
    def _cache_put(key: tuple[str, str], value: Optional[dict]) -> Optional[dict]:
        _EX_API_CACHE[key] = (value, time.time())
        return value


# Sentinel distinguishing "cached None (miss)" from "not cached".
_CACHE_MISS = object()
