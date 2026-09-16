"""
BMR & TDEE Calculator
Implements Mifflin-St Jeor equation for BMR, activity multipliers for TDEE,
and goal-based calorie/macro targets.

Formulas:
  BMR (Male)   = 10 × weight(kg) + 6.25 × height(cm) - 5 × age - 5 + 5
  BMR (Female) = 10 × weight(kg) + 6.25 × height(cm) - 5 × age - 5 - 161

  Actually Mifflin-St Jeor:
    Male:   BMR = 10W + 6.25H - 5A + 5
    Female: BMR = 10W + 6.25H - 5A - 161

  TDEE = BMR × activity_multiplier
  Calorie Target = TDEE + goal_adjustment
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Activity Multipliers (Harris-Benedict / Mifflin extension)
# ---------------------------------------------------------------------------

ACTIVITY_MULTIPLIERS: dict[str, tuple[float, str]] = {
    "sedentary":    (1.2,   "Little or no exercise, desk job"),
    "light":        (1.375, "Light exercise 1-3 days/week"),
    "moderate":     (1.55,  "Moderate exercise 3-5 days/week"),
    "active":       (1.725, "Hard exercise 6-7 days/week"),
    "very_active":  (1.9,   "Very hard exercise, physical job or 2x training"),
}

# ---------------------------------------------------------------------------
# Goal Adjustments (daily calorie delta)
# ---------------------------------------------------------------------------

GOAL_ADJUSTMENTS: dict[str, tuple[float, str]] = {
    "lose_weight":  (-500,  "Deficit of 500 kcal/day (~0.5 kg/week loss)"),
    "maintain":     (0,     "No adjustment (maintain weight)"),
    "gain_muscle":  (+300,  "Surplus of kcal/day (lean muscle gain)"),
    "gain_weight":  (+500,  "Surplus of 500 kcal/day (~0.5 kg/week gain)"),
}

# ---------------------------------------------------------------------------
# Macro split by goal (protein%, carbs%, fat% of calorie target)
# ---------------------------------------------------------------------------

MACRO_SPLITS: dict[str, tuple[float, float, float]] = {
    # (protein_pct, carbs_pct, fat_pct)
    "lose_weight":  (0.35, 0.35, 0.30),   # high protein, moderate carbs/fat
    "maintain":     (0.25, 0.45, 0.30),   # balanced
    "gain_muscle":  (0.35, 0.40, 0.25),   # high protein, high carbs
    "gain_weight":  (0.25, 0.50, 0.25),   # high carbs for surplus
}


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class CalorieTarget:
    """Computed BMR, TDEE, and personalized targets."""
    bmr: float
    tdee: float
    calorie_target: float
    protein_target_g: float
    carbs_target_g: float
    fat_target_g: float
    bmr_formula: str
    tdee_multiplier: float
    tdee_description: str
    calorie_adjustment: str


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

def calculate_bmr(gender: str, weight_kg: float, height_cm: float, age: int) -> float:
    """
    Mifflin-St Jeor BMR calculation.

    Male:   BMR = 10 × weight(kg) + 6.25 × height(cm) - 5 × age + 5
    Female: BMR = 10 × weight(kg) + 6.25 × height(cm) - 5 × age - 161
    """
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    if gender.lower() == "male":
        return round(base + 5, 1)
    else:
        return round(base - 161, 1)


def calculate_tdee(bmr: float, activity_level: str) -> tuple[float, float, str]:
    """
    TDEE = BMR × activity multiplier.
    Returns (tdee, multiplier, description).
    """
    multiplier, description = ACTIVITY_MULTIPLIERS.get(
        activity_level.lower(),
        ACTIVITY_MULTIPLIERS["moderate"]
    )
    tdee = round(bmr * multiplier, 1)
    return tdee, multiplier, description


def calculate_calorie_target(tdee: float, fitness_goal: str) -> tuple[float, str]:
    """
    Apply goal adjustment to TDEE.
    Returns (calorie_target, adjustment_description).
    """
    adjustment, description = GOAL_ADJUSTMENTS.get(
        fitness_goal.lower(),
        GOAL_ADJUSTMENTS["maintain"]
    )
    target = round(tdee + adjustment, 1)
    # Minimum floor: never go below 1200 kcal for safety
    target = max(target, 1200.0)
    return target, description


def calculate_macro_targets(
    calorie_target: float, fitness_goal: str
) -> tuple[float, float, float]:
    """
    Calculate daily protein, carbs, fat targets in grams.
    Based on calorie target and goal-specific macro split.

    Protein: 4 kcal/g, Carbs: 4 kcal/g, Fat: 9 kcal/g
    """
    protein_pct, carbs_pct, fat_pct = MACRO_SPLITS.get(
        fitness_goal.lower(),
        MACRO_SPLITS["maintain"]
    )
    protein_g = round((calorie_target * protein_pct) / 4, 1)
    carbs_g = round((calorie_target * carbs_pct) / 4, 1)
    fat_g = round((calorie_target * fat_pct) / 9, 1)
    return protein_g, carbs_g, fat_g


def compute_full_target(user: dict) -> CalorieTarget:
    """
    Full computation pipeline from a user document (dict).
    Returns a CalorieTarget dataclass with all values.
    """
    bmr = calculate_bmr(user["gender"], user["weight_kg"], user["height_cm"], user["age"])
    tdee, multiplier, tdee_desc = calculate_tdee(bmr, user["activity_level"])
    calorie_target, adjustment_desc = calculate_calorie_target(tdee, user["fitness_goal"])
    protein_g, carbs_g, fat_g = calculate_macro_targets(calorie_target, user["fitness_goal"])

    return CalorieTarget(
        bmr=bmr,
        tdee=tdee,
        calorie_target=calorie_target,
        protein_target_g=protein_g,
        carbs_target_g=carbs_g,
        fat_target_g=fat_g,
        bmr_formula="Mifflin-St Jeor",
        tdee_multiplier=multiplier,
        tdee_description=tdee_desc,
        calorie_adjustment=adjustment_desc,
    )


def compute_full_target_dict(user: dict) -> CalorieTarget:
    """
    Same as compute_full_target but accepts a dict (MongoDB document).
    """
    bmr = calculate_bmr(user["gender"], user["weight_kg"], user["height_cm"], user["age"])
    tdee, multiplier, tdee_desc = calculate_tdee(bmr, user["activity_level"])
    calorie_target, adjustment_desc = calculate_calorie_target(tdee, user["fitness_goal"])
    protein_g, carbs_g, fat_g = calculate_macro_targets(calorie_target, user["fitness_goal"])

    return CalorieTarget(
        bmr=bmr,
        tdee=tdee,
        calorie_target=calorie_target,
        protein_target_g=protein_g,
        carbs_target_g=carbs_g,
        fat_target_g=fat_g,
        bmr_formula="Mifflin-St Jeor",
        tdee_multiplier=multiplier,
        tdee_description=tdee_desc,
        calorie_adjustment=adjustment_desc,
    )
