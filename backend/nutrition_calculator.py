"""
Nutrition Calculator
Converts (food, quantity, variant) → final nutrition breakdown.

Pipeline:
  1. Resolve grams from (amount, unit) using food-category-aware conversion table
  2. Apply variant multiplier (ghee, fried, boiled, etc.)
  3. Compute nutrition = base_nutrition × (grams / serving_size_g) × variant_multiplier

All macros in the DB are PER SERVING (serving_size_g).
  ratio = actual_grams / food.serving_size_g
  final_calories = food.calories_kcal × ratio × variant_multiplier
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from quantity_parser import ParsedQuantity


# ---------------------------------------------------------------------------
# Unit → Grams Conversion Tables
# ---------------------------------------------------------------------------

# Default gram equivalents per unit (generic, food-agnostic fallbacks)
DEFAULT_UNIT_GRAMS: dict[str, float] = {
    "g": 1.0,             # already grams, multiply amount × 1
    "ml": 1.0,            # approximate: 1 ml ≈ 1 g for most foods/liquids
    "piece": -1,          # sentinel: use food's serving_size_g
    "rotlo": -1,          # sentinel: use food's serving_size_g (flatbread piece)
    "rotla": -1,          # sentinel: use food's serving_size_g
    "bowl": 200.0,        # standard Indian katori bowl
    "katori": 150.0,      # smaller Indian katori
    "cup": 240.0,         # standard metric cup
    "plate": 300.0,       # full plate
    "glass": 250.0,       # standard glass
    "serving": -1,        # sentinel: use food's serving_size_g
    "tablespoon": 15.0,
    "tbsp": 15.0,
    "teaspoon": 5.0,
    "tsp": 5.0,
    "slice": 30.0,        # average bread slice
}

# Food-category-specific overrides (category → unit → grams)
# Applies ONLY to container/volume units: bowl, katori, cup, plate, glass, slice.
# "piece" and "serving" are NOT here — those always use food["serving_size_g"]
# directly in _resolve_grams() (Rule 2), making per-food data authoritative.
CATEGORY_UNIT_OVERRIDES: dict[str, dict[str, float]] = {
    "rice": {
        "bowl": 180.0,    # cooked rice in a standard Indian bowl
        "plate": 300.0,
        "cup": 200.0,
    },
    "dal": {
        "bowl": 200.0,
        "katori": 150.0,
        "cup": 200.0,
    },
    "curry": {
        "bowl": 200.0,
        "katori": 150.0,
    },
    "bread": {
        "slice": 30.0,
    },
    "roti": {
        # per-piece weight comes from serving_size_g (Rule 2) — no override needed
    },
    "paratha": {
        # per-piece weight comes from serving_size_g (Rule 2) — no override needed
    },
    "idli": {
        "piece": 40.0,     # 1 idli ≈ 40g (DB serving_size_g = multi-piece serving)
    },
    "dosa": {
        "piece": 100.0,    # 1 dosa ≈ 100g
    },
    "egg": {
        "piece": 50.0,     # 1 egg ≈ 50g
    },
    "fruit": {
        "bowl": 150.0,
    },
    "milk": {
        "glass": 200.0,
        "cup": 200.0,
    },
    "juice": {
        "glass": 200.0,
        "cup": 200.0,
    },
    "tea": {
        "cup": 150.0,
        "glass": 150.0,
    },
    "coffee": {
        "cup": 150.0,
    },
    "soup": {
        "bowl": 250.0,
        "cup": 250.0,
    },
    "salad": {
        "bowl": 150.0,
        "plate": 200.0,
    },
    "pizza": {
        "slice": 100.0,
    },
    "chaat": {
        "plate": 150.0,
    },
    "beverage": {
        "glass": 250.0,
        "cup": 150.0,
    },
    "snack": {
        "bowl": 100.0,
        "plate": 150.0,
    },
    "sweet": {
        "plate": 100.0,
    },
    "cake": {
        "slice": 80.0,
    },
    "biscuit": {
        # pieces handled by Rule 2; no container overrides needed
    },
}

# Food-name keyword → category hint (used when food.category is generic/empty)
# Only relevant for CONTAINER units (bowl, katori, cup, plate, glass).
# "piece" and "serving" are not affected by this — they always use serving_size_g.
NAME_TO_CATEGORY_HINTS: dict[str, str] = {
    "pizza": "pizza",
    "dabeli": "snack", "vada pav": "snack", "samosa": "snack", "kachori": "snack",
    "pani puri": "chaat", "sev puri": "chaat", "bhel": "chaat", "chaat": "chaat",
    "rice": "rice", "biryani": "rice", "pulao": "rice", "khichdi": "rice",
    "chawal": "rice", "fried rice": "rice",
    "dal": "dal", "daal": "dal", "sambar": "dal", "rasam": "dal",
    "lentil": "dal",
    "roti": "roti", "chapatti": "roti", "bhakri": "roti", "naan": "roti",
    "paratha": "paratha", "kulcha": "paratha",
    "thepla": "paratha", "methi thepla": "paratha", "bajri thepla": "paratha",
    "khakhra": "snack",  # khakhra is a cracker; "piece" → serving_size_g
    "puri": "bread", "poori": "bread",
    "bhatura": "bread",
    "idli": "idli",
    "dosa": "dosa", "uttapam": "dosa",
    "egg": "egg", "anda": "egg",
    "milk": "milk", "doodh": "milk", "lassi": "milk",
    "yogurt": "milk", "buttermilk": "milk", "chaas": "milk",
    "tea": "tea", "chai": "tea",
    "coffee": "coffee",
    "juice": "juice", "smoothie": "juice", "sharbat": "juice",
    "soup": "soup", "shorba": "soup",
    "salad": "salad", "raita": "salad",
    "cake": "cake", "pastry": "cake", "brownie": "cake",
    "biscuit": "biscuit", "cookie": "biscuit",
}


# ---------------------------------------------------------------------------
# Variant Multipliers
# ---------------------------------------------------------------------------
# These adjust calories and fat when a preparation method differs from "normal".
# Multipliers are applied to ALL macros proportionally (simplification).

@dataclass
class VariantMultiplier:
    """Multiplier applied to nutrition when a cooking variant is detected."""
    name: str
    calorie_multiplier: float
    fat_multiplier: float
    description: str


VARIANT_MULTIPLIERS: dict[str, VariantMultiplier] = {
    "ghee": VariantMultiplier(
        name="ghee",
        calorie_multiplier=1.25,
        fat_multiplier=1.50,
        description="Prepared with ghee (+25% calories, +50% fat)",
    ),
    "butter": VariantMultiplier(
        name="butter",
        calorie_multiplier=1.20,
        fat_multiplier=1.40,
        description="Prepared with butter (+20% calories, +40% fat)",
    ),
    "oil": VariantMultiplier(
        name="oil",
        calorie_multiplier=1.20,
        fat_multiplier=1.45,
        description="Cooked in oil (+20% calories, +45% fat)",
    ),
    "deep fried": VariantMultiplier(
        name="deep fried",
        calorie_multiplier=1.50,
        fat_multiplier=2.00,
        description="Deep fried (+50% calories, +100% fat)",
    ),
    "fried": VariantMultiplier(
        name="fried",
        calorie_multiplier=1.35,
        fat_multiplier=1.70,
        description="Pan/stir fried (+35% calories, +70% fat)",
    ),
    "grilled": VariantMultiplier(
        name="grilled",
        calorie_multiplier=0.90,
        fat_multiplier=0.75,
        description="Grilled (-10% calories, -25% fat)",
    ),
    "boiled": VariantMultiplier(
        name="boiled",
        calorie_multiplier=0.85,
        fat_multiplier=0.60,
        description="Boiled (-15% calories, -40% fat)",
    ),
    "steamed": VariantMultiplier(
        name="steamed",
        calorie_multiplier=0.85,
        fat_multiplier=0.60,
        description="Steamed (-15% calories, -40% fat)",
    ),
    "baked": VariantMultiplier(
        name="baked",
        calorie_multiplier=0.95,
        fat_multiplier=0.85,
        description="Baked (-5% calories, -15% fat)",
    ),
    "roasted": VariantMultiplier(
        name="roasted",
        calorie_multiplier=0.95,
        fat_multiplier=0.85,
        description="Roasted (-5% calories, -15% fat)",
    ),
    "raw": VariantMultiplier(
        name="raw",
        calorie_multiplier=1.0,
        fat_multiplier=1.0,
        description="Raw / no cooking (no change)",
    ),
    "sugar": VariantMultiplier(
        name="sugar",
        calorie_multiplier=1.20,
        fat_multiplier=1.00,
        description="Prepared with sugar (+20% calories)",
    ),
    "with sugar": VariantMultiplier(
        name="with sugar",
        calorie_multiplier=1.20,
        fat_multiplier=1.00,
        description="Prepared with sugar (+20% calories)",
    ),
    "without sugar": VariantMultiplier(
        name="without sugar",
        calorie_multiplier=0.70,
        fat_multiplier=1.00,
        description="Prepared without sugar (-30% calories)",
    ),
    "normal": VariantMultiplier(
        name="normal",
        calorie_multiplier=1.0,
        fat_multiplier=1.0,
        description="Standard preparation (no change)",
    ),
}


# ---------------------------------------------------------------------------
# Nutrition Result
# ---------------------------------------------------------------------------

@dataclass
class NutritionResult:
    """Final calculated nutrition for a food + quantity + variant."""
    food_id: str
    food_name: str
    food_name_display: str

    # Input echo
    quantity_amount: float
    quantity_unit: str
    quantity_grams: float        # resolved grams
    variant: str                 # "normal" if no variant
    variant_description: str

    # Calculated nutrition
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    sugar_g: Optional[float]
    sodium_mg: Optional[float]

    # Metadata
    serving_size_g: float        # food's original serving size
    calculation_note: str = ""   # e.g., "variant multiplier applied"
    confidence: float = 1.0      # overall confidence


# ---------------------------------------------------------------------------
# Variant normalization
# ---------------------------------------------------------------------------

_VARIANT_PREFIX_STRIP = re.compile(
    r"^(with|in|using|cooked in|made with|prepared with)\s+",
    re.IGNORECASE,
)


def _normalize_variant(raw: Optional[str]) -> str:
    """
    Normalize a variant string to match VARIANT_MULTIPLIERS keys.
    Handles: 'with ghee' → 'ghee', 'in oil' → 'oil', 'deep fried' → 'deep fried'
    """
    if not raw:
        return "normal"
    cleaned = raw.strip().lower()
    cleaned = _VARIANT_PREFIX_STRIP.sub("", cleaned)
    cleaned = cleaned.strip()
    # Check if it matches a known variant
    if cleaned in VARIANT_MULTIPLIERS:
        return cleaned
    # Try partial match: "pan fried" → "fried"
    for key in VARIANT_MULTIPLIERS:
        if key in cleaned:
            return key
    return cleaned if cleaned else "normal"


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class NutritionCalculator:
    """
    Computes nutrition for a food given quantity and variant.

    Usage:
        calc = NutritionCalculator()
        result = calc.calculate(food, parsed_quantity, variant="ghee")
    """

    def calculate(
        self,
        food: models.Food,
        quantity: ParsedQuantity,
        variant: Optional[str] = None,
    ) -> NutritionResult:
        """
        Main entry point.

        Args:
            food: SQLAlchemy Food model instance
            quantity: ParsedQuantity from quantity_parser
            variant: Optional cooking method (ghee, fried, boiled, etc.)

        Returns:
            NutritionResult with fully computed values.
        """
        # ── Step 1: Resolve grams ──────────────────────────────────────
        grams = self._resolve_grams(food, quantity)

        # ── Step 2: Base nutrition ratio ───────────────────────────────
        # All DB macros are per serving_size_g
        ratio = grams / food["serving_size_g"] if food["serving_size_g"] > 0 else 1.0

        # ── Step 3: Variant multiplier ─────────────────────────────────
        variant_key = _normalize_variant(variant)
        vm = None

        if isinstance(food, dict) and food.get("variant_nutrition"):
            vn_map = food.get("variant_nutrition", {})
            v_info = vn_map.get(variant_key) or (vn_map.get(variant.lower()) if variant else None)
            if v_info:
                c_mult = float(v_info.get("calorie_multiplier", 1.0))
                f_mult = float(v_info.get("fat_multiplier", 1.0))
                is_est = v_info.get("is_estimated", False)
                desc = f"Variant '{variant_key}': {c_mult:.2f}x cals, {f_mult:.2f}x fat"
                if is_est:
                    desc += " (estimated)"
                vm = VariantMultiplier(name=variant_key, calorie_multiplier=c_mult, fat_multiplier=f_mult, description=desc)

        if not vm:
            vm = VARIANT_MULTIPLIERS.get(variant_key, VARIANT_MULTIPLIERS["normal"])

        # ── Step 4: Compute final nutrition ────────────────────────────
        calories = food["calories_kcal"] * ratio * vm.calorie_multiplier
        protein = food["protein_g"] * ratio
        carbs = food["carbs_g"] * ratio
        fat = food["fat_g"] * ratio * vm.fat_multiplier
        fiber = food["fiber_g"] * ratio
        sugar = (food["sugar_g"] * ratio) if food.get("sugar_g") is not None else None
        sodium = (food["sodium_mg"] * ratio) if food.get("sodium_mg") is not None else None

        # Build note
        notes = []
        if variant_key != "normal":
            notes.append(f"Variant '{variant_key}': {vm.description}")
        if quantity.unit in ("g", "ml"):
            notes.append(f"Computed from {grams:.0f}g (DB serving: {food['serving_size_g']:.0f}g)")
        else:
            notes.append(
                f"{quantity.amount} {quantity.unit} = {grams:.0f}g "
                f"(DB serving: {food['serving_size_g']:.0f}g)"
            )

        # Validate calculation generically against nutritional science bounds
        try:
            from nutrition_validator import GenericNutritionValidator
            v_rep = GenericNutritionValidator.validate_calculated_nutrition(
                food, quantity.unit, quantity.amount, grams, calories
            )
            if v_rep and v_rep.note:
                notes.append(v_rep.note)
        except Exception:
            pass

        return NutritionResult(
            food_id=food["food_id"],
            food_name=food["food_name"],
            food_name_display=food["food_name_display"],
            quantity_amount=quantity.amount,
            quantity_unit=quantity.unit,
            quantity_grams=round(grams, 1),
            variant=variant_key,
            variant_description=vm.description,
            calories=round(calories, 1),
            protein_g=round(protein, 1),
            carbs_g=round(carbs, 1),
            fat_g=round(fat, 1),
            fiber_g=round(fiber, 1),
            sugar_g=round(sugar, 1) if sugar is not None else None,
            sodium_mg=round(sodium, 1) if sodium is not None else None,
            serving_size_g=food["serving_size_g"],
            calculation_note="; ".join(notes),
            confidence=quantity.confidence,
        )

    # ------------------------------------------------------------------
    # Gram resolution
    # ------------------------------------------------------------------

    def _resolve_grams(self, food: dict, quantity: ParsedQuantity) -> float:
        """
        Convert (amount, unit) → grams using food context.

        Unit-resolution precedence
        ─────────────────────────
        1. Direct weight/volume: g, ml → use the amount as-is.

        2. Piece-like / per-item units: "piece", "rotlo", "rotla".
           Strategy: use food["serving_size_g"] DIRECTLY when the food's stored
           serving itself represents one discrete piece (i.e. serving_unit is a
           count word: "piece", "rotlo", "rotla", "1 piece", etc.).
           When serving_unit is "serving" (= a multi-piece serving), fall back to
           the category-specific per-piece gram table, then to serving_size_g.
           This correctly handles:
             • thepla  serving_size_g=80  serving_unit=piece → 1 piece = 80 g ✓
             • roti    serving_size_g=40  serving_unit=piece → 1 piece = 40 g ✓
             • idli    serving_size_g=180 serving_unit=serving → use table (40 g) ✓

        3. Serving unit: always use food["serving_size_g"] × amount.

        4. Container/volume units (bowl, katori, cup, plate, glass, slice …):
           Look up category-specific gram values, fall back to DEFAULT_UNIT_GRAMS,
           then to food["serving_size_g"] as a last resort.
        """
        unit   = quantity.unit
        amount = quantity.amount

        # ── Rule 1: direct weight / volume ─────────────────────────────────────
        if unit in ("g", "ml"):
            return amount

        # ── Rule 3: explicit serving ────────────────────────────────────────────
        if unit == "serving":
            return amount * food["serving_size_g"]

        # ── Rule 2: discrete piece units ───────────────────────────────────────
        _PIECE_UNITS = {"piece", "rotlo", "rotla"}
        if unit in _PIECE_UNITS:
            category_hint      = self._infer_category(food)
            category_overrides = CATEGORY_UNIT_OVERRIDES.get(category_hint, {})
            cat_piece = category_overrides.get("piece", DEFAULT_UNIT_GRAMS.get("piece", -1))

            if cat_piece > 0 and cat_piece < food["serving_size_g"]:
                return amount * cat_piece

            # Generic piece weight resolution using serving & food context
            try:
                from nutrition_validator import GenericNutritionValidator
                piece_g = GenericNutritionValidator.get_piece_weight(
                    food.get("food_name", ""),
                    float(food.get("serving_size_g") or 0.0),
                    str(food.get("serving_unit") or "serving"),
                    food.get("quantity_options") or [],
                )
                if piece_g > 0:
                    return amount * piece_g
            except Exception:
                pass

            # Fallback to food's serving_size_g
            return amount * food["serving_size_g"]

        # ── Rule 4: container / volume units ───────────────────────────────────
        category_hint      = self._infer_category(food)
        category_overrides = CATEGORY_UNIT_OVERRIDES.get(category_hint, {})
        grams_per_unit     = category_overrides.get(unit, DEFAULT_UNIT_GRAMS.get(unit, -1))

        if grams_per_unit < 0:          # sentinel: fall back to food's serving_size_g
            grams_per_unit = food["serving_size_g"]

        return amount * grams_per_unit

    def _infer_category(self, food: dict) -> str:
        """Infer a category hint from food name or category field."""
        name = food["food_name"].lower()
        food_cat = (food.get("category") or "").lower()
        food_subcat = (food.get("subcategory") or "").lower()

        # Name keyword match FIRST (most specific)
        for keyword, cat in NAME_TO_CATEGORY_HINTS.items():
            if keyword in name:
                return cat

        # Then try the food's own category/subcategory
        for cat_key in CATEGORY_UNIT_OVERRIDES:
            if cat_key in food_cat or cat_key in food_subcat:
                return cat_key

        return ""  # no override, will use defaults
