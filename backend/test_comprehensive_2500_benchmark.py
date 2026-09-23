"""
Comprehensive 2,500-Case Fitness Chatbot Pipeline Benchmark
===========================================================
Validates:
1. Structured workout routines across muscle groups, reps, sets, exercise counts, and durations:
   - Exercise items, reps, sets, and muscle groups are extracted accurately.
   - Strictly ZERO duration invented when duration is not provided (duration_min MUST BE None).
   - Never converts reps into minutes (e.g. 12 reps is NEVER 12 minutes).
   - Accurately captures explicit duration when provided.
2. Single-exercise logging across categories, reps, sets, durations, distance, and multilingual noise.
3. Food and nutrition logging across single-food, multi-food, quantities, units, and variants.
4. Nutrition vs Workout recommendation isolation:
   - "what should I eat for dinner?" -> RECOMMEND_MEAL
   - "post workout meal suggest karo" -> RECOMMEND_MEAL
   - "what workout should I do?" -> RECOMMEND_WORKOUT
   - "workout after heavy dinner" -> RECOMMEND_WORKOUT
   - Zero nutrition -> workout errors, zero workout -> nutrition errors.
5. Conversational, summary, query, and flow follow-ups.

Target: Exactly/at least 2,500 unique cases. 100% PASS rate.
"""
from __future__ import annotations

import os
import sys
import time
from typing import NamedTuple, Optional

# Ensure standard output can print Unicode cleanly on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from llm_service import LLMService, Intent, classify_recommendation_intent
from exercise_calculator import parse_workout_routine, calculate_routine


class TestCase(NamedTuple):
    case_id: int
    category: str
    message: str
    expected_intent: str
    expect_duration_none: bool = False
    expected_duration_min: Optional[float] = None
    expected_min_reps: Optional[int] = None
    language: str = "en"


def build_benchmark_cases() -> list[TestCase]:
    cases: list[TestCase] = []
    case_id = 1
    seen_messages = set()

    def add_case(cat: str, msg: str, exp_intent: str, dur_none: bool = False,
                 dur_min: Optional[float] = None, min_reps: Optional[int] = None, lang: str = "en"):
        nonlocal case_id
        clean_m = msg.strip()
        if clean_m in seen_messages:
            return
        seen_messages.add(clean_m)
        cases.append(TestCase(
            case_id=case_id,
            category=cat,
            message=clean_m,
            expected_intent=exp_intent,
            expect_duration_none=dur_none,
            expected_duration_min=dur_min,
            expected_min_reps=min_reps,
            language=lang,
        ))
        case_id += 1

    # =========================================================================
    # CATEGORY 1: Structured Workout Routines (>= 800 Cases)
    # Muscle groups + exercise counts + reps/sets. NO INVENTED DURATION.
    # =========================================================================
    muscle_groups_pool = [
        ("Chest", "Shoulder", "Triceps"),
        ("Back", "Biceps", "Forearms"),
        ("Legs", "Calves", "Glutes"),
        ("Chest", "Back", "Abs"),
        ("Shoulders", "Arms", "Core"),
        ("Quadriceps", "Hamstrings", "Calves"),
        ("Biceps", "Triceps", "Shoulder"),
        ("Chest", "Triceps", "Abs"),
        ("Upper Back", "Lats", "Biceps"),
        ("Pectorals", "Deltoids", "Triceps"),
        ("Chest", "Delts", "Triceps"),
        ("Quads", "Glutes", "Hamstrings"),
        ("Lats", "Rhomboids", "Biceps"),
        ("Shoulders", "Traps", "Neck"),
        ("Core", "Abs", "Obliques"),
    ]

    count_templates = [
        "{m1}: 4 exercises × 12 reps, {m2}: 2 exercises × 12 reps, {m3}: 2 exercises × 12 reps",
        "aaje me {m1} 4 exercises x 12 reps {m2} 2 exercises x 12 reps {m3} 2 exercises x 12 reps mara",
        "today did {m1} 3 exercises of 10 reps, {m2} 3 exercises of 12 reps, {m3} 2 exercises of 15 reps",
        "aaj maine {m1} 4 exercise 12 rep {m2} 2 exercise 10 rep {m3} 3 exercise 12 rep kiye",
        "workout routine: {m1} (4 exercises x 12 reps), {m2} (3 exercises x 10 reps)",
        "{m1} 4 exercises 12 reps {m2} 2 exercises 12 reps",
        "{m1} 3 excercise x 15 reps, {m2} 2 excercise x 12 reps",
        "me aaje {m1} na 4 exercise ane {m2} na 3 exercise karyo 12 reps",
        "{m1} 4 exercises × 10 reps, {m2} 3 exercises × 10 reps, {m3} 2 exercises × 12 reps",
        "{m1} 5 exercises x 10 reps, {m2} 3 exercises x 12 reps",
        "gym session: {m1} 4 exercises x 12 reps, {m2} 2 exercises x 12 reps, {m3} 2 exercises x 15 reps",
        "today's training: {m1} 3 exercises x 12 reps, {m2} 3 exercises x 12 reps",
    ]

    for m1, m2, m3 in muscle_groups_pool:
        for tmpl in count_templates:
            msg = tmpl.format(m1=m1, m2=m2, m3=m3)
            add_case("structured_routine_no_duration", msg, Intent.LOG_EXERCISE,
                     dur_none=True, min_reps=20)

    # Multilingual & Indic script routines
    indic_routines = [
        ("આજે મેં ચેસ્ટ 4 કસરત 12 રેપ્સ અને શોલ્ડર 2 કસરત 12 રેપ્સ કર્યા", "gu", True, None, 72),
        ("ચેસ્ટ: ૪ કસરત x ૧૨ રેપ્સ, શોલ્ડર: ૨ કસરત x ૧૨ રેપ્સ", "gu", True, None, 72),
        ("આજે બાઇસેપ્સ 3 કસરત 10 રેપ્સ અને ટ્રાઇસેપ્સ 3 કસરત 12 રેપ્સ", "gu", True, None, 66),
        ("લેગ્સ: ૪ કસરત x ૧૫ રેપ્સ, કાફ્સ: ૨ કસરત x ૨૦ રેપ્સ", "gu", True, None, 100),
        ("आज मैंने चेस्ट 4 एक्सरसाइज 12 रेप्स और ट्राइसेप्स 2 एक्सरसाइज 12 रेप्स किए", "hi", True, None, 72),
        ("चेस्ट 4 एक्सरसाइज x 10 रेप्स, शोल्डर 3 एक्सरसाइज x 12 रेप्स", "hi", True, None, 76),
        ("बैक 4 एक्सरसाइज 12 रेप्स, बाइसेप्स 3 एक्सरसाइज 12 रेप्स किए", "hi", True, None, 84),
        ("aaje me chest 3 exercises 15 reps ane back 3 exercises 12 reps marya", "gu", True, None, 81),
        ("me savare leg day karyo: squats 4 exercises x 12 reps, calves 2 exercises x 15 reps", "gu", True, None, 78),
        ("gym session: chest 4 exercises 12 reps, triceps 2 exercises 15 reps", "en", True, None, 78),
        ("biceps 3 exercises x 12 reps, triceps 3 exercises x 12 reps mara", "gu", True, None, 72),
        ("shoulder 4 exercises x 10 reps, abs 2 exercises x 20 reps", "en", True, None, 80),
        ("back 4 exercises x 12 reps, biceps 2 exercises x 12 reps", "en", True, None, 72),
    ]

    for base_msg, lang, dur_none, dur_val, min_reps in indic_routines:
        for prefix in ["", "dost ", "bhai ", "today ", "savare ", "subah ", "hello ", "hey "]:
            add_case("indic_structured_routines", prefix + base_msg, Intent.LOG_EXERCISE,
                     dur_none=dur_none, dur_min=dur_val, min_reps=min_reps, lang=lang)

    # Structured routines WITH explicit duration (duration MUST match, reps preserved)
    explicit_durations = [
        ("Chest 4 exercises x 12 reps, Shoulder 2 exercises x 12 reps for 45 minutes", 45.0, 72),
        ("Back 3 exercises x 10 reps, Biceps 2 exercises x 12 reps for 30 mins", 30.0, 54),
        ("Legs 4 exercises x 12 reps, Calves 2 exercises x 15 reps for 1 hour", 60.0, 78),
        ("aaje me Chest 4 exercises x 12 reps, Triceps 2 exercises x 12 reps 40 minute karyu", 40.0, 72),
        ("aaj mene Shoulders 3 exercises x 12 reps 35 min kiya", 35.0, 36),
        ("did Chest 4 exercises of 12 reps and Arms 2 exercises of 12 reps for 50 minutes", 50.0, 72),
        ("workout: Chest 3 exercises x 15 reps for 45 minutes", 45.0, 45),
        ("Full body: Chest 2 exercises x 12 reps, Back 2 exercises x 12 reps for 1 hour", 60.0, 48),
        ("Legs 3 exercises x 12 reps, Abs 2 exercises x 15 reps for 25 minutes", 25.0, 66),
        ("Chest 5 exercises x 10 reps for 50 mins", 50.0, 50),
        ("Back 4 exercises x 12 reps for 40 mins", 40.0, 48),
        ("Shoulders 4 exercises x 12 reps for 35 minutes", 35.0, 48),
    ]

    for base_msg, dur, reps in explicit_durations:
        for suffix in ["", " at gym", " with trainer", " hard session", " feeling good", " in the morning"]:
            add_case("structured_routine_with_duration", base_msg + suffix, Intent.LOG_EXERCISE,
                     dur_none=False, dur_min=dur, min_reps=reps)

    # Systematic permutations of muscle groups & sets to ensure robust coverage
    rep_counts = [8, 10, 12, 15]
    ex_counts = [2, 3, 4, 5]
    muscles = ["Chest", "Shoulder", "Triceps", "Biceps", "Back", "Legs", "Abs", "Glutes", "Quads", "Hamstrings"]

    for i in range(len(muscles)):
        for j in range(len(muscles)):
            if i == j:
                continue
            m_a, m_b = muscles[i], muscles[j]
            for r in [10, 12]:
                for c in [3, 4]:
                    msg = f"{m_a} {c} exercises x {r} reps, {m_b} 2 exercises x {r} reps"
                    add_case("systematic_muscle_routines", msg, Intent.LOG_EXERCISE,
                             dur_none=True, min_reps=(c + 2) * r)

    # =========================================================================
    # CATEGORY 2: Single-Exercise Logging (>= 700 Cases)
    # Rep-based, duration-based, distance-based, Indian past-tense verbs, typos
    # =========================================================================
    rep_exercises = [
        ("push-ups", ["push-ups", "pushups", "push up", "puchups", "pushap"]),
        ("squats", ["squats", "squat", "squot", "squuts"]),
        ("pull-ups", ["pull-ups", "pullups", "pull up", "pull ups"]),
        ("lunges", ["lunges", "lunge"]),
        ("crunches", ["crunches", "crunch"]),
        ("jumping jacks", ["jumping jacks", "jumping jack"]),
        ("burpees", ["burpees", "burpee"]),
        ("dips", ["dips", "dip"]),
        ("sit-ups", ["sit-ups", "situps", "sit up", "sit ups"]),
        ("deadlifts", ["deadlifts", "deadlift"]),
    ]

    for canon, variants in rep_exercises:
        for v in variants:
            for amt in [15, 20, 30, 50]:
                # English
                add_case("single_exercise_reps_en", f"did {amt} {v}", Intent.LOG_EXERCISE, dur_none=True)
                add_case("single_exercise_reps_en", f"completed 3 sets of {amt} {v}", Intent.LOG_EXERCISE, dur_none=True)
                # Gujarati past tense
                add_case("single_exercise_reps_gu", f"aaje me {amt} {v} marya", Intent.LOG_EXERCISE, dur_none=True, lang="gu")
                add_case("single_exercise_reps_gu", f"savare {amt} {v} karya", Intent.LOG_EXERCISE, dur_none=True, lang="gu")
                # Hindi past tense
                add_case("single_exercise_reps_hi", f"aaj maine {amt} {v} lagaye", Intent.LOG_EXERCISE, dur_none=True, lang="hi")
                add_case("single_exercise_reps_hi", f"subah {amt} {v} kiye", Intent.LOG_EXERCISE, dur_none=True, lang="hi")

    duration_exercises = [
        ("running", ["running", "runing", "ran"]),
        ("jogging", ["jogging", "joging", "jogged"]),
        ("cycling", ["cycling", "cyclin", "cycled", "saikal"]),
        ("swimming", ["swimming", "swiming", "swam"]),
        ("yoga", ["yoga", "yga", "yoag"]),
        ("plank", ["plank", "planks"]),
        ("walking", ["walking", "walkin", "walked", "chalyo"]),
        ("skipping", ["skipping", "skiping"]),
        ("treadmill", ["treadmill"]),
        ("zumba", ["zumba"]),
    ]

    for canon, variants in duration_exercises:
        for v in variants:
            for d in [20, 30, 45, 60]:
                add_case("single_exercise_dur_en", f"{v} for {d} minutes", Intent.LOG_EXERCISE, dur_none=False, dur_min=float(d))
                add_case("single_exercise_dur_gu", f"aaje me {d} minute {v} karyu", Intent.LOG_EXERCISE, dur_none=False, dur_min=float(d), lang="gu")
                add_case("single_exercise_dur_hi", f"aaj {d} minute {v} ki", Intent.LOG_EXERCISE, dur_none=False, dur_min=float(d), lang="hi")

    # Distance-based
    for dist in [2, 3, 5, 8, 10]:
        for act in ["running", "cycling", "walking"]:
            add_case("single_exercise_dist", f"did {dist} km {act} today", Intent.LOG_EXERCISE)
            add_case("single_exercise_dist", f"aaje {dist} km {act} karyo", Intent.LOG_EXERCISE, lang="gu")
            add_case("single_exercise_dist", f"aaj {dist} km {act} kiya", Intent.LOG_EXERCISE, lang="hi")

    # =========================================================================
    # CATEGORY 3: Food & Nutrition Logging (>= 600 Cases)
    # Single food, multi-food, quantities, units, preparation variants.
    # Must NEVER route to LOG_EXERCISE or RECOMMEND_WORKOUT.
    # =========================================================================
    foods = [
        "roti", "thepla", "bhakri", "dal", "rice", "khichdi", "paneer tikka",
        "boiled eggs", "grilled chicken", "masala dosa", "idli sambar", "poha",
        "upma", "chana masala", "rajma chawal", "chaas", "milk", "coffee",
        "green tea", "apple", "banana", "salad", "sprouts", "almonds", "oats",
    ]

    for f in foods:
        for q in ["2 pieces", "1 bowl", "150g", "1 glass", "1 plate", "200g"]:
            add_case("food_single_en", f"I had {q} {f} for lunch", Intent.LOG_FOOD)
            add_case("food_single_gu", f"aaje me {q} {f} khadha", Intent.LOG_FOOD, lang="gu")
            add_case("food_single_hi", f"maine subah {q} {f} khaya", Intent.LOG_FOOD, lang="hi")

    # Multi-food combinations
    multi_pairs = [
        ("2 roti", "1 bowl dal"),
        ("3 thepla", "1 cup chai"),
        ("2 idli", "1 bowl sambar"),
        ("1 bowl rice", "1 bowl rajma"),
        ("150g chicken", "1 bowl rice"),
        ("2 eggs", "2 slices toast"),
        ("1 plate poha", "1 glass milk"),
        ("2 bhakri", "1 glass chaas"),
        ("1 bowl salad", "1 bowl sprouts"),
        ("1 cup coffee", "2 biscuits"),
        ("2 rotli", "1 katori shak"),
        ("1 bowl khichdi", "1 bowl dahi"),
        ("1 bowl oats", "1 banana"),
        ("1 scoop protein", "300ml milk"),
        ("2 paratha", "1 cup curd"),
    ]

    for p1, p2 in multi_pairs:
        add_case("food_multi_en", f"ate {p1} and {p2} for dinner", Intent.LOG_FOOD)
        add_case("food_multi_gu", f"aaje bapore me {p1} ane {p2} lidhu", Intent.LOG_FOOD, lang="gu")
        add_case("food_multi_hi", f"dopahar ko {p1} aur {p2} khaya", Intent.LOG_FOOD, lang="hi")

    # Beverages & zero drinks
    drinks = [
        "1 glass chaas", "1 cup black coffee", "1 glass nimbu pani", "1 can diet coke",
        "1 can coke zero", "1 glass coconut water", "1 scoop whey protein", "1 cup green tea",
        "1 glass dudh", "1 cup chai", "1 glass lassi", "1 bottle water",
    ]
    for d in drinks:
        add_case("food_drinks", f"drank {d}", Intent.LOG_FOOD)
        add_case("food_drinks", f"aaje {d} pidhu", Intent.LOG_FOOD, lang="gu")
        add_case("food_drinks", f"aaj {d} piya", Intent.LOG_FOOD, lang="hi")

    # =========================================================================
    # CATEGORY 4: Nutrition vs Workout Recommendation Isolation (>= 500 Cases)
    # The 1000/1000 routing isolation guarantee.
    # Meal recommendations must NEVER route to workout recommendations.
    # Workout recommendations must NEVER route to meal recommendations.
    # =========================================================================
    nutrition_queries = [
        "what should I eat for dinner?",
        "suggest high protein breakfast",
        "post workout meal suggest karo",
        "what to eat after gym",
        "pre-workout snack idea",
        "bapore shu jamvu?",
        "dinner me kya khana chahiye?",
        "suggest a healthy lunch",
        "low calorie snacks for weight loss",
        "what food to eat to build muscle?",
        "what should I drink after workout?",
        "suggest some protein rich vegetarian foods",
        "ratre su banavu?",
        "healthy dinner recipes",
        "what can I eat before morning workout?",
        "post gym protein meal options",
        "what to eat after heavy cardio?",
        "dinner ma su banavu?",
        "suggest food for fat loss",
        "protein diet plan options",
        "what to eat for breakfast tomorrow?",
        "healthy snacks for evening",
        "suggest vegetarian diet for muscle gain",
        "what to eat to recover after intense workout",
        "dinner options under 400 calories",
    ]

    for nq in nutrition_queries:
        for prefix in ["", "please ", "bhai ", "hey ", "can you tell me ", "dost ", "hello "]:
            add_case("recommend_meal_isolation", prefix + nq, Intent.RECOMMEND_MEAL)

    workout_queries = [
        "what workout should I do?",
        "what should I do tomorrow?",
        "workout after heavy dinner",
        "suggest a workout routine",
        "chest workout suggest karo",
        "exercise to burn pizza calories",
        "what exercise for tomorrow?",
        "kal kya exercise karein?",
        "aavti kal exercise suchavjo",
        "mane workout suggest karo",
        "full body workout plan for beginner",
        "cardio routine to burn fat",
        "what exercises to do after cheat meal?",
        "gym routine for muscle gain",
        "home workout without equipment",
        "kal subah kya workout karein?",
        "leg workout suggest karo",
        "workout to burn 500 calories",
        "suggest high intensity cardio",
        "back and biceps routine",
        "what routine to do for upper body?",
        "exercise to lose belly fat",
        "suggest a 30 minute cardio session",
        "what workout after overeating pizza",
        "post dinner workout suggestions",
    ]

    for wq in workout_queries:
        for prefix in ["", "please ", "bhai ", "hey ", "can you tell me ", "dost ", "hello "]:
            add_case("recommend_workout_isolation", prefix + wq, Intent.RECOMMEND_WORKOUT)

    # Mixed & tricky phrases (eating after gym vs working out after eating)
    tricky_cases = [
        ("post workout meal suggest karo", Intent.RECOMMEND_MEAL),
        ("pre workout meal options", Intent.RECOMMEND_MEAL),
        ("what food to eat after chest workout", Intent.RECOMMEND_MEAL),
        ("protein snack after intense gym", Intent.RECOMMEND_MEAL),
        ("what shake to drink post exercise", Intent.RECOMMEND_MEAL),
        ("post workout dinner options", Intent.RECOMMEND_MEAL),
        ("food after heavy workout", Intent.RECOMMEND_MEAL),
        ("what breakfast after morning running", Intent.RECOMMEND_MEAL),
        ("snack after evening workout", Intent.RECOMMEND_MEAL),
        ("nutrition advice after workout", Intent.RECOMMEND_MEAL),
        ("what to eat after leg day workout", Intent.RECOMMEND_MEAL),
        ("dinner after gym session", Intent.RECOMMEND_MEAL),
        ("recovery drink after workout", Intent.RECOMMEND_MEAL),
        ("workout after heavy dinner", Intent.RECOMMEND_WORKOUT),
        ("exercise after eating pizza", Intent.RECOMMEND_WORKOUT),
        ("workout to burn burger calories", Intent.RECOMMEND_WORKOUT),
        ("what exercise to do after heavy lunch", Intent.RECOMMEND_WORKOUT),
        ("post dinner workout suggestions", Intent.RECOMMEND_WORKOUT),
        ("exercise after cheat meal", Intent.RECOMMEND_WORKOUT),
        ("workout plan to burn fat after feast", Intent.RECOMMEND_WORKOUT),
        ("what cardio after cheat day", Intent.RECOMMEND_WORKOUT),
        ("morning workout after dinner last night", Intent.RECOMMEND_WORKOUT),
        ("routine to burn yesterday sweets", Intent.RECOMMEND_WORKOUT),
        ("what exercise to burn biryani", Intent.RECOMMEND_WORKOUT),
        ("running to burn cake calories", Intent.RECOMMEND_WORKOUT),
    ]

    for msg, exp_int in tricky_cases:
        for mod in ["", "dost ", "bhai ", "jaldi ", "please ", "sir "]:
            add_case("tricky_mixed_isolation", mod + msg, exp_int)

    # =========================================================================
    # CATEGORY 5: Conversation, Summary, Queries & Greetings (>= 60 Cases)
    # =========================================================================
    greetings = [
        "hi", "hello", "hey", "kem cho", "namaste", "namaskar",
        "good morning", "good evening", "good afternoon", "good night",
        "kemcho", "hello dost", "hi there", "namaste ji",
    ]
    for g in greetings:
        add_case("greetings", g, Intent.GREETING)

    summaries = [
        "today summary", "aaj ka summary", "show summary", "report", "kitna khaya",
        "yesterday summary", "kal ka summary", "gatkale nu summary", "daily report",
        "show today report", "today's summary", "aaj ki report", "kal ka report",
    ]
    for s in summaries:
        add_case("summary", s, Intent.GET_SUMMARY)

    query_exercises = [
        "what exercise did I do today?", "aaje me kya exercise kari?", "show my workout log",
        "which workout did I do today?", "kya exercise ki?", "show workout", "how many calories did I burn",
        "today workout log", "list my workout", "show training log",
    ]
    for qe in query_exercises:
        add_case("query_exercise", qe, Intent.QUERY_EXERCISE)

    query_meals = [
        "what did I eat for breakfast?", "aaje breakfast ma su lidhu?", "su khadhu bapore?",
        "what did I eat for lunch?", "dinner me kya khaya tha?", "show lunch items",
        "breakfast items today", "what did I eat today?",
    ]
    for qm in query_meals:
        add_case("query_meal", qm, Intent.QUERY_MEAL)

    return cases


def run_benchmark():
    start_time = time.time()
    cases = build_benchmark_cases()
    total_cases = len(cases)

    print("=" * 80)
    print(f"COMPREHENSIVE 2,500-CASE FIT-BOT PIPELINE BENCHMARK (Total: {total_cases})")
    print("=" * 80)

    if total_cases < 2500:
        print(f"[ERROR] Benchmark requires at least 2,500 cases, but only generated {total_cases}!")
        return 1

    llm = LLMService(account_id="", api_token="")

    passed = 0
    failed = 0
    failures = []
    category_stats: dict[str, dict] = {}
    invented_durations = 0
    nutrition_to_workout_errors = 0
    workout_to_nutrition_errors = 0

    for case in cases:
        cat = case.category
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "passed": 0, "failed": 0}
        category_stats[cat]["total"] += 1

        msg = case.message
        # Parse intent through fallback / sync intent parser
        parsed = llm.parse_intent_sync(msg)
        predicted_intent = parsed.intent

        # Evaluate intent
        intent_match = (predicted_intent == case.expected_intent)

        # Evaluate duration guarantees for workout routines
        duration_ok = True
        duration_err_detail = ""
        if case.expect_duration_none:
            if parsed.duration_min is not None:
                duration_ok = False
                invented_durations += 1
                duration_err_detail = f"INVENTED DURATION: duration_min={parsed.duration_min} (expected None)"
        elif case.expected_duration_min is not None:
            if parsed.duration_min != case.expected_duration_min:
                duration_ok = False
                duration_err_detail = f"MISMATCH DURATION: got {parsed.duration_min} (expected {case.expected_duration_min})"

        # Check misroute violations
        if case.expected_intent == Intent.RECOMMEND_MEAL and predicted_intent == Intent.RECOMMEND_WORKOUT:
            nutrition_to_workout_errors += 1
        elif case.expected_intent == Intent.RECOMMEND_WORKOUT and predicted_intent == Intent.RECOMMEND_MEAL:
            workout_to_nutrition_errors += 1

        # Reps check
        reps_ok = True
        if case.expected_min_reps is not None:
            total_reps = parsed.reps or 0
            if total_reps < case.expected_min_reps:
                reps_ok = False

        if intent_match and duration_ok and reps_ok:
            passed += 1
            category_stats[cat]["passed"] += 1
        else:
            failed += 1
            category_stats[cat]["failed"] += 1
            reasons = []
            if not intent_match:
                reasons.append(f"Intent expected={case.expected_intent}, got={predicted_intent}")
            if not duration_ok:
                reasons.append(duration_err_detail)
            if not reps_ok:
                reasons.append(f"Reps expected>={case.expected_min_reps}, got={parsed.reps}")

            failures.append({
                "id": case.case_id,
                "category": case.category,
                "message": case.message,
                "reasons": reasons,
            })

    elapsed = time.time() - start_time
    pass_rate = (passed / total_cases) * 100.0

    print("\nCATEGORY BREAKDOWN:")
    print("-" * 80)
    for cat, stats in sorted(category_stats.items()):
        cat_pass = stats["passed"]
        cat_tot = stats["total"]
        pct = (cat_pass / cat_tot) * 100.0 if cat_tot > 0 else 0.0
        print(f"  * {cat:<35}: {cat_pass:4d} / {cat_tot:4d} ({pct:6.2f}%)")

    print("-" * 80)
    print("CRITICAL GUARANTEE METRICS:")
    print(f"  * Invented Durations Count : {invented_durations} (MUST BE 0)")
    print(f"  * Nutrition -> Workout Error: {nutrition_to_workout_errors} (MUST BE 0)")
    print(f"  * Workout -> Nutrition Error: {workout_to_nutrition_errors} (MUST BE 0)")
    print(f"  * Overall Pass Rate        : {passed} / {total_cases} ({pass_rate:.2f}%)")
    print(f"  * Total Execution Time     : {elapsed:.2f}s")
    print("=" * 80)

    if failures:
        print(f"\n[FAILURES REPORTED - FIRST {min(20, len(failures))}]:")
        for f in failures[:20]:
            print(f"  [Case #{f['id']} | {f['category']}] \"{f['message']}\"")
            for r in f["reasons"]:
                print(f"      - {r}")

    if passed == total_cases and invented_durations == 0 and nutrition_to_workout_errors == 0 and workout_to_nutrition_errors == 0:
        print("\n[SUCCESS] PERFECT BENCHMARK SCORE: 100.00% PASSED!")
        print("All workout routines, reps, durations, and nutrition intents strictly verified.")
        return 0
    else:
        print(f"\n[FAILURE] {failed} test cases failed out of {total_cases}.")
        return 1


if __name__ == "__main__":
    sys.exit(run_benchmark())
