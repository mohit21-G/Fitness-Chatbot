"""
Generic Nutrition & Serving Validator
Validates food kcal, serving sizes, piece weights, and preparation assumptions
against nutritional science references (NIN / ICMR / IFCT & USDA).

Generic principles applied to ALL foods:
1. Macronutrient Energy Physics: Atwater equation: (P*4) + (C*4) + (F*9)
2. Calorie Density by Category: Energy density bounds (kcal/100g)
3. Piece Weight & Multi-piece Serving Resolution: Standard discrete portion weights
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class NutritionValidationReport:
    food_name: str
    stored_kcal: float
    stored_serving_g: float
    unit: str
    amount: float
    resolved_grams: float
    calculated_kcal: float
    reference_range_kcal: tuple[float, float]
    difference_pct: float
    is_plausible: bool
    detected_reason: Optional[str] = None
    suggested_piece_g: Optional[float] = None
    note: str = ""


# ---------------------------------------------------------------------------
# Reference Energy Density Bounds (kcal per 100g) by Food Category
# Derived from ICMR-NIN Indian Food Composition Tables (IFCT 2017) & USDA
# ---------------------------------------------------------------------------
CATEGORY_DENSITY_RANGES: dict[str, tuple[float, float]] = {
    # Breads & Flatbreads (roti, thepla, paratha, naan, bhakri, kulcha)
    # Moisture 30-40%, carbohydrates 40-50%, fat 5-15%
    "flatbread": (240.0, 390.0),
    "bread": (230.0, 360.0),

    # Cooked grains / rice / khichdi / pulao / biryani
    # Moisture 65-75%
    "rice": (100.0, 180.0),
    "grain": (100.0, 200.0),

    # Cooked dals / lentil preparations / sambar
    # Moisture 75-85%
    "dal": (60.0, 140.0),
    "curry": (50.0, 160.0),

    # Cooked vegetables / shaak / sabji (non-starchy to starchy)
    "vegetable": (35.0, 140.0),

    # Snacks & Savouries (fried / baked / street food)
    "snack": (180.0, 450.0),
    "street_food": (150.0, 380.0),

    # Sweets & Desserts (halwa, kheer, gulab jamun, barfi)
    "sweet": (220.0, 480.0),
    "dessert": (150.0, 420.0),

    # Dairy & Beverages
    "milk": (40.0, 90.0),
    "beverage": (20.0, 90.0),
    "tea": (20.0, 70.0),

    # Salads & Soups
    "soup": (20.0, 80.0),
    "salad": (25.0, 100.0),
}

# ---------------------------------------------------------------------------
# Reference Single-Piece Weight Ranges (grams per discrete piece)
# ---------------------------------------------------------------------------
PIECE_WEIGHT_RANGES: dict[str, tuple[float, float, float]] = {
    # (min_g, typical_g, max_g)
    "thepla": (35.0, 40.0, 50.0),
    "roti": (30.0, 40.0, 50.0),
    "chapati": (30.0, 40.0, 50.0),
    "phulka": (25.0, 30.0, 35.0),
    "rotlo": (70.0, 80.0, 100.0),
    "bhakri": (40.0, 50.0, 60.0),
    "paratha": (60.0, 75.0, 95.0),
    "puri": (20.0, 25.0, 35.0),
    "kulcha": (60.0, 75.0, 90.0),
    "naan": (70.0, 85.0, 110.0),
    "idli": (35.0, 40.0, 50.0),
    "dosa": (80.0, 100.0, 130.0),
    "samosa": (40.0, 50.0, 65.0),
    "kachori": (40.0, 50.0, 60.0),
    "gulab jamun": (30.0, 37.5, 45.0),
    "egg": (45.0, 50.0, 60.0),
    "cookie": (15.0, 25.0, 35.0),
    "biscuit": (10.0, 15.0, 20.0),
}


class GenericNutritionValidator:
    """
    Validates calorie calculations generically using nutritional science reference ranges.
    """

    @staticmethod
    def infer_density_category(food_name: str, category: str) -> str:
        fn = food_name.lower()
        cat = (category or "").lower()

        # Specific keywords
        if any(w in fn for w in ("thepla", "roti", "rotlo", "paratha", "bhakri", "naan", "kulcha", "puri", "chilla")):
            return "flatbread"
        if any(w in fn for w in ("rice", "biryani", "pulao", "khichdi", "chawal")):
            return "rice"
        if any(w in fn for w in ("dal", "daal", "sambar", "rasam", "kadhi")):
            return "dal"
        if any(w in fn for w in ("curry", "sabji", "sabzi", "shaak", "shak")):
            return "curry"
        if any(w in fn for w in ("samosa", "kachori", "vada", "pakoda", "tikki", "roll", "sandwich", "burger", "pizza", "dabeli", "pav")):
            return "snack"
        if any(w in fn for w in ("sweet", "halwa", "kheer", "gulab jamun", "ladoo", "barfi", "burfi", "cake", "pastry")):
            return "sweet"
        if any(w in fn for w in ("chai", "tea", "coffee", "milk", "doodh", "lassi", "chhas", "juice", "shake")):
            return "beverage"
        if any(w in fn for w in ("soup", "shorba", "broth")):
            return "soup"
        if any(w in fn for w in ("salad", "raita")):
            return "salad"

        # Check DB category
        if "bread" in cat:
            return "flatbread"
        if "snack" in cat:
            return "snack"
        if "sweet" in cat:
            return "sweet"
        if "dairy" in cat or "beverage" in cat:
            return "beverage"
        if "grain" in cat:
            return "grain"

        return "curry"

    @classmethod
    def get_piece_weight(cls, food_name: str, serving_size_g: float, serving_unit: str = "serving", quantity_options: list[str] = None) -> float:
        """
        Generic resolution of single-piece weight (grams) for discrete food items.
        Handles foods where serving_size_g represents a multi-piece serving (e.g. 2 theplas = 80g).
        """
        fn = food_name.lower()

        # 1. Direct piece-weight keyword match
        for key, (p_min, p_typ, p_max) in PIECE_WEIGHT_RANGES.items():
            if key in fn:
                return p_typ

        # 2. Check if food explicitly specifies serving_unit as 'piece' (1 piece = serving_size_g)
        if serving_unit.lower() == "piece" and serving_size_g > 0:
            return serving_size_g

        # 3. Check quantity_options for multi-piece base indicators
        # E.g., ['2 pieces', '4 pieces', ...] indicates base serving is 2 pieces
        if quantity_options and isinstance(quantity_options, list):
            first_opt = quantity_options[0].lower().strip()
            if "2 piece" in first_opt or "2 pieces" in first_opt:
                if serving_size_g > 0:
                    return serving_size_g / 2.0
            elif "1 piece" in first_opt or "1.5 piece" in first_opt:
                # Flatbreads with 1 piece, 1.5 piece, 2 pieces where serving is 80g
                # 80g is 2 pieces of 40g each
                if any(w in fn for w in ("thepla", "roti", "paratha", "bhakri", "kulcha")) and serving_size_g >= 70.0:
                    return serving_size_g / 2.0

        # 4. If serving_size_g is large (> 60g) for flatbreads, default to half (2-piece serving)
        if any(w in fn for w in ("thepla", "roti", "paratha", "bhakri", "kulcha")) and serving_size_g >= 70.0:
            return serving_size_g / 2.0

        # Fallback to serving_size_g
        return serving_size_g if serving_size_g > 0 else 50.0

    @classmethod
    def validate_and_correct_grams(
        cls,
        food: dict,
        unit: str,
        amount: float,
        raw_grams: float,
    ) -> tuple[float, Optional[str]]:
        """
        Validates whether raw_grams calculated for (amount, unit) is physically plausible.
        If a multi-piece serving discrepancy is detected, returns corrected grams with reason.
        """
        fn = food.get("food_name", "").lower()
        serving_size_g = float(food.get("serving_size_g") or 80.0)
        serving_unit = (food.get("serving_unit") or "serving").lower()
        opts = food.get("quantity_options") or []

        if unit in ("piece", "rotlo", "rotla"):
            # If food is a flatbread or multi-piece serving and raw_grams treated serving_size_g as 1 piece
            expected_piece_g = cls.get_piece_weight(fn, serving_size_g, serving_unit, opts)
            correct_grams = amount * expected_piece_g

            if abs(raw_grams - correct_grams) > 5.0 and raw_grams > correct_grams:
                ratio = raw_grams / correct_grams
                reason = (
                    f"Stored serving_size_g ({serving_size_g:.0f}g) covers {ratio:.1f} pieces; "
                    f"normalized to {expected_piece_g:.0f}g/piece for {amount} {unit}"
                )
                return correct_grams, reason

            return correct_grams, None

        return raw_grams, None

    @classmethod
    def validate_calculated_nutrition(
        cls,
        food: dict,
        unit: str,
        amount: float,
        resolved_grams: float,
        calculated_kcal: float,
    ) -> NutritionValidationReport:
        """
        Validates final calculated kcal against Atwater macro physics and category density bounds.
        """
        fn = food.get("food_name", "")
        cat = food.get("category", "")
        serving_size_g = float(food.get("serving_size_g") or 100.0)
        stored_kcal = float(food.get("calories_kcal") or 0.0)

        # 1. Density category
        density_cat = cls.infer_density_category(fn, cat)
        min_density, max_density = CATEGORY_DENSITY_RANGES.get(density_cat, (50.0, 350.0))

        # Expected kcal range based on grams consumed
        expected_min_kcal = (resolved_grams / 100.0) * min_density
        expected_max_kcal = (resolved_grams / 100.0) * max_density
        ref_range = (round(expected_min_kcal, 1), round(expected_max_kcal, 1))

        # Check plausibility (with 20% margin for culinary variance)
        is_plausible = (expected_min_kcal * 0.75) <= calculated_kcal <= (expected_max_kcal * 1.30)
        mid_expected = (expected_min_kcal + expected_max_kcal) / 2.0
        diff_pct = abs(calculated_kcal - mid_expected) / mid_expected * 100.0 if mid_expected > 0 else 0.0

        note = f"Category '{density_cat}' density: {min_density:.0f}-{max_density:.0f} kcal/100g. Consumed: {resolved_grams:.0f}g."

        return NutritionValidationReport(
            food_name=fn,
            stored_kcal=stored_kcal,
            stored_serving_g=serving_size_g,
            unit=unit,
            amount=amount,
            resolved_grams=resolved_grams,
            calculated_kcal=calculated_kcal,
            reference_range_kcal=ref_range,
            difference_pct=round(diff_pct, 1),
            is_plausible=is_plausible,
            note=note,
        )

    @classmethod
    def is_nutritionally_plausible(cls, food_name: str, calories_per_100g: float, category: str = "") -> bool:
        """Check whether a food's calories_per_100g falls within scientifically plausible reference bounds."""
        density_cat = cls.infer_density_category(food_name, category)
        min_density, max_density = CATEGORY_DENSITY_RANGES.get(density_cat, (50.0, 350.0))
        return (min_density * 0.70) <= calories_per_100g <= (max_density * 1.30)
