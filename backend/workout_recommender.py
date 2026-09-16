"""
Workout Recommendation Engine
==============================
Produces a fully data-driven "tomorrow's workout" recommendation using:

  • Today's exercise log          — what was done, calories burned
  • 7-day exercise history        — which muscle groups need recovery
  • User goal + profile           — drives category priority order
  • Today's calorie / protein     — guards against under-fuelled training
  • Live exercise DB (MongoDB)    — all suggestions pulled from real exercises;
                                   zero hardcoded workout schedules or calorie values

Recovery model
--------------
Every DB exercise category maps to a recovery group with a mandatory rest window:

  Strength / Calisthenics   → "strength"  — 48 h recovery (micro-tear repair)
  Full body                 → "full_body" — 48 h recovery (compound fatigue)
  Pilates & Core            → "core"      — 24 h recovery (core muscles)
  Cardio / Swimming /
  Dance & Zumba / Sports /
  Martial Arts / Activity   → "cardio"    —  0 h (safe every day)
  Yoga                      → "yoga"      —  0 h (safe every day)
  Fitness (external cache)  → "general"   —  0 h (can't infer structure)

Goal → category priority
------------------------
  lose_weight  : Cardio, Yoga, Full body, Pilates & Core, Strength
  gain_muscle  : Strength, Full body, Calisthenics, Pilates & Core, Cardio
  maintain     : Cardio, Yoga, Strength, Full body, Sports, Calisthenics
  gain_weight  : Strength, Full body, Calisthenics, Yoga, Pilates & Core

All public; import and call `build_workout_recommendation(...)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

# IST = UTC+5:30
IST = timezone(timedelta(hours=5, minutes=30))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How many hours a recovery group needs before it can be trained again.
_RECOVERY_HOURS: dict[str, int] = {
    "strength":  48,
    "full_body": 48,
    "core":      24,
    "cardio":     0,
    "yoga":       0,
    "general":    0,
}

# DB category name → recovery group
_CATEGORY_TO_GROUP: dict[str, str] = {
    "strength":              "strength",
    "calisthenics":          "strength",   # bodyweight compound — same recovery
    "full body":             "full_body",
    "pilates & core":        "core",
    "cardio":                "cardio",
    "swimming":              "cardio",
    "dance & zumba":         "cardio",
    "sports":                "cardio",
    "martial arts/boxing":   "cardio",
    "activity":              "cardio",
    "yoga":                  "yoga",
    "fitness":               "general",    # externally-cached exercises
}

# Goal → ordered list of DB category names to consider (highest priority first)
_GOAL_CATEGORY_PRIORITY: dict[str, list[str]] = {
    "lose_weight":  ["Cardio", "Yoga", "Full body", "Pilates & Core", "Strength"],
    "gain_muscle":  ["Strength", "Full body", "Calisthenics", "Pilates & Core", "Cardio"],
    "maintain":     ["Cardio", "Yoga", "Strength", "Full body", "Sports", "Calisthenics"],
    "gain_weight":  ["Strength", "Full body", "Calisthenics", "Yoga", "Pilates & Core"],
}
# Fallback for unknown goals
_DEFAULT_CATEGORY_PRIORITY = ["Cardio", "Strength", "Yoga", "Full body"]

# How many exercises to show per recommended category
_EXERCISES_PER_CATEGORY = 3

# How many days back to look at exercise history for recovery checks
_HISTORY_DAYS = 7

# Calorie threshold below which we warn about under-fuelling (kcal remaining)
_UNDERFUEL_THRESHOLD = 200.0

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExerciseSuggestion:
    """One exercise pulled from the DB for display."""
    name_display:   str
    category:       str
    unit:           str
    typical_amount: float          # in the exercise's native unit
    est_calories:   float          # typical_amount × calories_per_unit_avg

@dataclass
class CategoryBlock:
    """A recommended training category with its exercise suggestions."""
    category:          str
    recovery_group:    str
    reason:            str         # human-readable justification
    suggestions:       list[ExerciseSuggestion] = field(default_factory=list)

@dataclass
class WorkoutRecommendation:
    """Full recommendation payload returned to the caller."""
    today_burned_cal:      float
    today_exercises:       list[str]           # display names done today
    recommended_blocks:    list[CategoryBlock] # 1–2 category blocks
    underfuelled:          bool                # calories too low to train hard
    protein_warning:       bool                # protein < 50 % of target
    protein_consumed_g:    float
    protein_target_g:      float
    calories_consumed:     float
    calorie_target:        float
    goal:                  str
    blocked_groups:        dict[str, float]    # group → hours_since_last
    no_history:            bool                # user has never logged exercise


# ---------------------------------------------------------------------------
# Core recommendation logic
# ---------------------------------------------------------------------------

def _group_of(category: str) -> str:
    """Map a DB category string to its recovery group."""
    return _CATEGORY_TO_GROUP.get(category.lower().strip(), "general")


def _hours_since(last_date_str: str) -> float:
    """Hours since a log_date string (YYYY-MM-DD), measured from IST now."""
    try:
        d = datetime.strptime(last_date_str[:10], "%Y-%m-%d").date()
        now_ist = datetime.now(IST).date()
        return (now_ist - d).days * 24.0     # conservative: whole-day granularity
    except Exception:
        return 999.0


def _select_exercises(db, category: str, weight_kg: float) -> list[ExerciseSuggestion]:
    """
    Pull up to _EXERCISES_PER_CATEGORY exercises from MongoDB for the given
    category, sorted by typical_duration_min (ascending — shortest first so
    suggestions feel approachable).  External-cached 'Fitness' docs often have
    garbled names so they are excluded.
    """
    try:
        col = db["exercises"]
        # Exclude the external-cache Fitness bucket (garbled names / low quality)
        docs = list(col.find(
            {"category": category, "is_verified": True},
            {"_id": 0, "exercise_name_display": 1, "category": 1,
             "measurement_unit": 1, "typical_duration_min": 1,
             "calories_per_unit_avg": 1},
        ).sort("typical_duration_min", 1).limit(_EXERCISES_PER_CATEGORY))

        suggestions = []
        for doc in docs:
            unit        = doc.get("measurement_unit", "minutes")
            duration    = float(doc.get("typical_duration_min") or 30)
            cal_per_u   = float(doc.get("calories_per_unit_avg") or 5.0)

            # Typical amount in native unit
            if unit == "minutes":
                typical_amount = duration
            elif unit in ("reps", "jumps", "laps", "rounds"):
                # Express as a round, sensible rep count
                typical_amount = duration
            elif unit == "session":
                typical_amount = 1.0
            else:
                typical_amount = duration

            # Scale calorie estimate by user weight (70 kg baseline in DB)
            weight_factor = weight_kg / 70.0
            est_cal = round(typical_amount * cal_per_u * weight_factor, 0)

            suggestions.append(ExerciseSuggestion(
                name_display   = doc.get("exercise_name_display", ""),
                category       = category,
                unit           = unit,
                typical_amount = typical_amount,
                est_calories   = est_cal,
            ))
        return suggestions
    except Exception:
        return []


def build_workout_recommendation(
    db,                        # pymongo sync Database
    user: dict,
    exercise_logs_7d: list[dict],   # from get_exercise_logs_by_range (7 days)
    today_food_totals: dict,        # from get_daily_food_totals (today)
    today_exercise_totals: dict,    # from get_daily_exercise_totals (today)
    calorie_target: "CalorieTarget",  # from compute_full_target(user)  # noqa: F821
) -> WorkoutRecommendation:
    """
    Build a WorkoutRecommendation from live DB data.  Pure logic — no I/O.

    Parameters
    ----------
    db                     pymongo sync Database (for exercise queries)
    user                   MongoDB user document dict
    exercise_logs_7d       all exercise logs for the past 7 days
    today_food_totals      today's food aggregate from DailyLogRepository
    today_exercise_totals  today's exercise aggregate from DailyLogRepository
    calorie_target         CalorieTarget dataclass from compute_full_target(user)
    """

    goal        = (user.get("fitness_goal") or "maintain").lower()
    weight_kg   = float(user.get("weight_kg") or 70)

    # ── Today's exercise snapshot ───────────────────────────────────────
    today_burned   = float(today_exercise_totals.get("total_calories_burned") or 0)
    today_entries  = int(today_exercise_totals.get("exercise_entries") or 0)

    # Collect display names done today from the 7-day log (same date)
    today_str = datetime.now(IST).date().isoformat()
    today_exercises = [
        log.get("exercise_name_display") or log.get("exercise_name", "")
        for log in exercise_logs_7d
        if log.get("log_date", "")[:10] == today_str
    ]

    # ── Nutrition snapshot ──────────────────────────────────────────────
    consumed_cal  = float(today_food_totals.get("total_calories_consumed") or 0)
    consumed_prot = float(today_food_totals.get("total_protein_g") or 0)
    prot_target   = float(calorie_target.protein_target_g or 100)
    cal_target_v  = float(calorie_target.calorie_target or 2000)

    net_cal         = consumed_cal - today_burned
    remaining_cal   = cal_target_v - net_cal
    underfuelled    = remaining_cal < _UNDERFUEL_THRESHOLD
    protein_warning = consumed_prot < (prot_target * 0.50)

    # ── Build recovery map ──────────────────────────────────────────────
    # {group → most-recent log_date string} from the past 7 days
    group_last_logged: dict[str, str] = {}
    for log in exercise_logs_7d:
        cat   = (log.get("category") or "").strip()
        grp   = _group_of(cat)
        d_str = (log.get("log_date") or "")[:10]
        if not d_str:
            continue
        if grp not in group_last_logged or d_str > group_last_logged[grp]:
            group_last_logged[grp] = d_str

    blocked_groups: dict[str, float] = {}
    for grp, last_d in group_last_logged.items():
        required = _RECOVERY_HOURS.get(grp, 0)
        if required == 0:
            continue
        hrs = _hours_since(last_d)
        if hrs < required:
            blocked_groups[grp] = hrs

    # ── Goal-driven category priority ──────────────────────────────────
    priority_cats = _GOAL_CATEGORY_PRIORITY.get(goal, _DEFAULT_CATEGORY_PRIORITY)

    # ── If underfuelled → force lightweight recovery options ────────────
    if underfuelled:
        priority_cats = ["Yoga", "Cardio"]   # only safe low-intensity options

    # ── Select recommended categories (skip blocked, pick 1-2) ─────────
    recommended_blocks: list[CategoryBlock] = []

    for cat in priority_cats:
        if len(recommended_blocks) >= 2:
            break

        grp      = _group_of(cat)
        required = _RECOVERY_HOURS.get(grp, 0)
        hours_s  = _hours_since(group_last_logged[grp]) if grp in group_last_logged else 999.0

        # Skip if this group still needs recovery
        if grp in blocked_groups:
            continue

        suggestions = _select_exercises(db, cat, weight_kg)
        if not suggestions:
            # No verified exercises in DB for this category — skip silently
            continue

        # Build the human-readable reason for this recommendation
        reason = _build_reason(cat, grp, hours_s, goal, underfuelled, blocked_groups)

        recommended_blocks.append(CategoryBlock(
            category       = cat,
            recovery_group = grp,
            reason         = reason,
            suggestions    = suggestions,
        ))

    # ── Fallback: if everything is blocked, recommend Yoga / light Cardio ──
    if not recommended_blocks:
        for fallback_cat in ["Yoga", "Cardio"]:
            suggestions = _select_exercises(db, fallback_cat, weight_kg)
            if suggestions:
                recommended_blocks.append(CategoryBlock(
                    category       = fallback_cat,
                    recovery_group = _group_of(fallback_cat),
                    reason         = _RECOVERY_FALLBACK_REASON,
                    suggestions    = suggestions,
                ))
                break

    return WorkoutRecommendation(
        today_burned_cal   = today_burned,
        today_exercises    = today_exercises,
        recommended_blocks = recommended_blocks,
        underfuelled       = underfuelled,
        protein_warning    = protein_warning,
        protein_consumed_g = consumed_prot,
        protein_target_g   = prot_target,
        calories_consumed  = consumed_cal,
        calorie_target     = cal_target_v,
        goal               = goal,
        blocked_groups     = blocked_groups,
        no_history         = len(exercise_logs_7d) == 0,
    )


# ---------------------------------------------------------------------------
# Reason string builders (English only — caller translates)
# ---------------------------------------------------------------------------

_RECOVERY_FALLBACK_REASON = (
    "All high-intensity groups need recovery today. "
    "Active recovery with yoga or light cardio is ideal."
)

_GOAL_REASONS: dict[str, str] = {
    "lose_weight": "Supports calorie deficit with steady fat-burning.",
    "gain_muscle": "Drives muscle protein synthesis with progressive overload.",
    "maintain":    "Keeps your fitness baseline and overall health.",
    "gain_weight": "Builds mass with compound strength movements.",
}

def _build_reason(
    cat: str,
    grp: str,
    hours_since_last: float,
    goal: str,
    underfuelled: bool,
    blocked_groups: dict,
) -> str:
    """Return a 1-sentence reason this category is recommended."""
    goal_note = _GOAL_REASONS.get(goal, "Matches your fitness goal.")

    if underfuelled:
        return "Calorie intake is low today — light activity keeps you active without taxing recovery."

    if hours_since_last >= 999:
        return f"No {cat} logged recently — {goal_note}"

    hrs_str = f"{hours_since_last:.0f}h" if hours_since_last < 48 else f"{int(hours_since_last // 24)}d"
    return f"Last {cat} session was {hrs_str} ago — full recovery complete. {goal_note}"


# ---------------------------------------------------------------------------
# Response formatter  (returns a markdown string like _build_nutrition_recommendation)
# ---------------------------------------------------------------------------

def format_recommendation(rec: WorkoutRecommendation, lang: str = "en") -> str:
    """
    Format a WorkoutRecommendation into a rich markdown string suitable for
    direct inclusion in the chatbot response.

    Supports English (en), Hindi (hi), Gujarati (gu) — same pattern as
    _build_nutrition_recommendation in chatbot_engine.
    """

    def L(en: str, hi: str, gu: str) -> str:
        if lang == "gu": return gu
        if lang == "hi": return hi
        return en

    lines: list[str] = []

    # ── Section 1: Today's training ─────────────────────────────────────
    h1 = L("🏋️ **Today's Training**", "🏋️ **आज की ट्रेनिंग**", "🏋️ **આજની ટ્રેનિંગ**")

    if rec.no_history or (not rec.today_exercises and rec.today_burned_cal == 0):
        today_line = L(
            "No exercise logged today.",
            "आज कोई exercise log नहीं किया।",
            "આજે કોઈ exercise log નથી.",
        )
    else:
        ex_list = ", ".join(rec.today_exercises) if rec.today_exercises else L(
            "Activity logged", "Activity log ki", "Activity log kari"
        )
        today_line = L(
            f"• {ex_list} — **{rec.today_burned_cal:.0f} kcal burned**",
            f"• {ex_list} — **{rec.today_burned_cal:.0f} kcal burn ki**",
            f"• {ex_list} — **{rec.today_burned_cal:.0f} kcal burn thi**",
        )

    lines.append(h1)
    lines.append(today_line)

    # Recovery summary (blocked groups)
    if rec.blocked_groups:
        blocked_names = []
        for grp, hrs in rec.blocked_groups.items():
            needed   = _RECOVERY_HOURS.get(grp, 0)
            hrs_left = needed - hrs
            blocked_names.append(L(
                f"{grp.replace('_',' ').title()} ({hrs_left:.0f}h rest remaining)",
                f"{grp.replace('_',' ').title()} ({hrs_left:.0f}h rest baaki)",
                f"{grp.replace('_',' ').title()} ({hrs_left:.0f}h rest baaki che)",
            ))
        rec_note = L(
            "⏸️ Recovery needed: " + ", ".join(blocked_names),
            "⏸️ Recovery chahiye: " + ", ".join(blocked_names),
            "⏸️ Recovery jaruri: " + ", ".join(blocked_names),
        )
        lines.append(rec_note)

    lines.append("")

    # ── Section 2: Tomorrow's recommendation ────────────────────────────
    h2 = L(
        "💡 **Recommended for Tomorrow**",
        "💡 **कल के लिए सुझाव**",
        "💡 **આવતીકાલ માટે સૂચન**",
    )
    lines.append(h2)

    if rec.underfuelled:
        fuel_warn = L(
            f"⚠️ Only **{rec.calories_consumed:.0f} kcal** consumed today vs target "
            f"**{rec.calorie_target:.0f} kcal** — avoid intense training. "
            "Stick to light activity.",
            f"⚠️ Aaj sirf **{rec.calories_consumed:.0f} kcal** khaaya, target "
            f"**{rec.calorie_target:.0f} kcal** hai — intense workout avoid karein. "
            "Halki activity karein.",
            f"⚠️ Aaj matr **{rec.calories_consumed:.0f} kcal** khadhu, target "
            f"**{rec.calorie_target:.0f} kcal** — intense workout avoid karo. "
            "Halki activity karo.",
        )
        lines.append(fuel_warn)
        lines.append("")

    if not rec.recommended_blocks:
        lines.append(L(
            "Rest day recommended — no suitable exercises found in the database.",
            "Aaj rest lena best hai — koi suitable exercise nahi mili.",
            "Aaj rest levo best che — koi suitable exercise maldi nathi.",
        ))
    else:
        for block in rec.recommended_blocks:
            # Category header + reason
            cat_header = L(
                f"**{block.category}** — {block.reason}",
                f"**{block.category}** — {block.reason}",
                f"**{block.category}** — {block.reason}",
            )
            lines.append(cat_header)

            # ── Section 3: Suggested exercises ──────────────────────────
            if block.suggestions:
                for s in block.suggestions:
                    unit_label = _unit_display(s.unit, s.typical_amount)
                    lines.append(L(
                        f"  • {s.name_display} — {s.typical_amount:.0f} {unit_label} (~{s.est_calories:.0f} kcal)",
                        f"  • {s.name_display} — {s.typical_amount:.0f} {unit_label} (~{s.est_calories:.0f} kcal)",
                        f"  • {s.name_display} — {s.typical_amount:.0f} {unit_label} (~{s.est_calories:.0f} kcal)",
                    ))
            lines.append("")

    # ── Section 4: Nutrition note ────────────────────────────────────────
    h4 = L("🥗 **Nutrition for Training**", "🥗 **Training ke liye Nutrition**", "🥗 **Training mate Nutrition**")
    lines.append(h4)

    if rec.protein_warning:
        prot_gap = max(0.0, rec.protein_target_g - rec.protein_consumed_g)
        lines.append(L(
            f"⚠️ Protein: **{rec.protein_consumed_g:.0f}g** consumed vs target **{rec.protein_target_g:.0f}g** "
            f"— need **{prot_gap:.0f}g more** today to support muscle recovery.",
            f"⚠️ Protein: **{rec.protein_consumed_g:.0f}g** liya vs target **{rec.protein_target_g:.0f}g** "
            f"— muscle recovery ke liye **{prot_gap:.0f}g aur** chahiye.",
            f"⚠️ Protein: **{rec.protein_consumed_g:.0f}g** lidhu vs target **{rec.protein_target_g:.0f}g** "
            f"— muscle recovery mate **{prot_gap:.0f}g vahu** jaruri.",
        ))
    else:
        lines.append(L(
            f"✅ Protein on track: **{rec.protein_consumed_g:.0f}g** / **{rec.protein_target_g:.0f}g**",
            f"✅ Protein sahi hai: **{rec.protein_consumed_g:.0f}g** / **{rec.protein_target_g:.0f}g**",
            f"✅ Protein sari rite: **{rec.protein_consumed_g:.0f}g** / **{rec.protein_target_g:.0f}g**",
        ))

    # Goal-specific training tip
    goal_tip = _goal_tip(rec.goal, lang)
    if goal_tip:
        lines.append(goal_tip)

    return "\n".join(lines)


def _unit_display(unit: str, amount: float) -> str:
    """Return a clean unit label (e.g. 'min' for minutes, 'reps')."""
    _map = {
        "minutes": "min",
        "reps":    "reps",
        "jumps":   "jumps",
        "laps":    "laps",
        "rounds":  "rounds",
        "session": "session",
    }
    return _map.get(unit, unit)


def _goal_tip(goal: str, lang: str) -> str:
    """Return a short, goal-specific training tip in the user's language."""
    tips = {
        "lose_weight": (
            "💧 Stay hydrated and keep intensity moderate to maximise fat oxidation.",
            "💧 Hydrated rahein aur moderate intensity se fat burn karein.",
            "💧 Hydrated raho ane moderate intensity thi fat burn karo.",
        ),
        "gain_muscle": (
            "🍗 Eat a protein-rich meal 1–2 h before training for best muscle gains.",
            "🍗 Training se 1-2 ghante pehle protein-rich khana khayein.",
            "🍗 Training thi 1-2 klaak pahela protein-rich khana khao.",
        ),
        "maintain": (
            "⚖️ Consistency beats intensity — aim for 4–5 sessions per week.",
            "⚖️ Consistency important hai — hafte mein 4-5 sessions karein.",
            "⚖️ Consistency important che — hapta ma 4-5 sessions karo.",
        ),
        "gain_weight": (
            "🍚 Calorie surplus matters — eat enough carbs before heavy lifts.",
            "🍚 Calorie surplus zaroori — heavy lifts se pehle carbs khayein.",
            "🍚 Calorie surplus jaruri — heavy lifts pahela carbs khao.",
        ),
    }
    t = tips.get(goal)
    if not t:
        return ""
    if lang == "gu": return t[2]
    if lang == "hi": return t[1]
    return t[0]
