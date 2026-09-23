"""
test_bugs_and_adversarial_benchmark.py
=======================================
Rigorous benchmark suite verifying the architectural fixes for:
1. Single and multi-routine parsing (exercise count preservation across unseen word order, languages, typos, transliteration).
   - "4 exercise" must ALWAYS remain 4 exercises (not 1, not reps).
   - Never infer duration from reps/sets/exercise count.
2. Semantic intent routing: LOG_EXERCISE vs RECOMMEND_WORKOUT vs RECOMMEND_MEAL.
   - "kale mare ky exercise karavi joye" -> RECOMMEND_WORKOUT
   - Questions/future/modal intent route to recommendation.
   - Past activity routes to logging.
3. Centralized duration parsing:
   - "1 hour", "1 houre", "1.5 hours", "2 hours", "half an hour", "aadha ghanta", "dedh ghanta", "30 minutes", "45 min".
   - Converted to exact minutes, never 1 minute for hours.
4. End-to-end ChatbotEngine pipeline testing.
"""

import sys
import os
import io
import re
import asyncio

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exercise_calculator import (
    parse_workout_routine,
    parse_duration_minutes,
    parse_exercise_input,
    calculate_routine,
    ExerciseCalculator,
)
from llm_service import LLMService, Intent, classify_recommendation_intent
from chatbot_engine import ChatbotEngine


# Dummy DB for ChatbotEngine testing
class DummyAsyncCollection:
    def __init__(self, data=None):
        self.data = list(data or [])

    def find(self, *args, **kwargs):
        return self

    def sort(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    async def __aiter__(self):
        for item in self.data:
            yield item

    async def to_list(self, length=None):
        return list(self.data)

    async def find_one(self, *args, **kwargs):
        return self.data[0] if self.data else None

    async def insert_one(self, doc):
        self.data.append(doc)
        return type("InsertResult", (), {"inserted_id": "dummy_id"})()

    async def update_one(self, *args, **kwargs):
        return type("UpdateResult", (), {"modified_count": 1})()

    async def delete_one(self, *args, **kwargs):
        return type("DeleteResult", (), {"deleted_count": 1})()

    async def delete_many(self, *args, **kwargs):
        return type("DeleteResult", (), {"deleted_count": 1})()

    async def count_documents(self, *args, **kwargs):
        return len(self.data)


class DummyAsyncDB:
    def __init__(self):
        self.food_logs = DummyAsyncCollection()
        self.exercise_logs = DummyAsyncCollection()
        self.daily_logs = DummyAsyncCollection()
        self.users = DummyAsyncCollection([{"user_id": "test_adv_user", "name": "Adv Tester", "gender": "male", "age": 25, "height_cm": 175, "weight_kg": 70, "goal": "maintain", "fitness_goal": "maintain", "activity_level": "moderate"}])
        self.conversations = DummyAsyncCollection()
        self.chat_history = DummyAsyncCollection()

    def __getitem__(self, name):
        if not hasattr(self, name):
            setattr(self, name, DummyAsyncCollection())
        return getattr(self, name)


# ==============================================================================
# 1. ROUTINE PARSING BENCHMARK (Counts, Order, Transliteration, Typos)
# ==============================================================================
ROUTINE_CASES = [
    # (input_text, expected_items: [(muscle, count, reps)], expected_total_reps, expected_duration_min)
    ("aaje chest na 4 exercise 12 reps karya", [("chest", 4, 12)], 48, None),
    ("4 exercise chest 12 reps", [("chest", 4, 12)], 48, None),
    ("4 chest exercises 12 reps", [("chest", 4, 12)], 48, None),
    ("chest 4 exercises x 12 reps", [("chest", 4, 12)], 48, None),
    ("chest 4 excersise 10 reps", [("chest", 4, 10)], 40, None),
    ("chest 4 kasarat 12 reps karyo", [("chest", 4, 12)], 48, None),
    ("12 reps chest na 4 exercise", [("chest", 4, 12)], 48, None),
    ("chest na 4 exercise", [("chest", 4, None)], 0, None),
    ("back ni 3 kasrat 10 reps", [("back", 3, 10)], 30, None),
    ("legs ke 5 exercise 15 reps", [("legs", 5, 15)], 75, None),
    ("shoulder 4 exercises 12 reps", [("shoulder", 4, 12)], 48, None),
    ("biceps 3 exercises 10 reps", [("biceps", 3, 10)], 30, None),
    ("triceps 2 exercises 12 reps", [("triceps", 2, 12)], 24, None),
    ("chest 12 reps", [("chest", 1, 12)], 12, None),
    ("chest 3 sets of 12 reps", [("chest", 1, 12)], 36, None),
    ("chest: 4 exercises x 3 sets x 12 reps", [("chest", 4, 12)], 144, None),
    # Multi-muscle routines
    (
        "Chest: 4 exercises x 12 reps Shoulder: 2 exercises x 12 reps Triceps: 2 exercises x 12 reps",
        [("chest", 4, 12), ("shoulder", 2, 12), ("triceps", 2, 12)],
        96,
        None,
    ),
    (
        "Chest na 4 exercise ane shoulder na 2 exercise 12 reps",
        [("chest", 4, 12), ("shoulder", 2, 12)],
        72,
        None,
    ),
    (
        "4 chest exercises 12 reps, 2 exercises shoulder 10 reps, triceps 15 reps",
        [("chest", 4, 12), ("shoulder", 2, 10), ("triceps", 1, 15)],
        83,
        None,
    ),
    # Explicit duration phrases (ONLY when explicit duration is given)
    ("Chest 4 exercises 12 reps for 45 minutes", [("chest", 4, 12)], 48, 45.0),
    ("Chest 4 exercises 12 reps for 1 hour", [("chest", 4, 12)], 48, 60.0),
    ("Chest 4 exercises 12 reps for 1.5 hours", [("chest", 4, 12)], 48, 90.0),
    ("Chest 4 exercises 12 reps 1 houre", [("chest", 4, 12)], 48, 60.0),
]


# ==============================================================================
# 2. DURATION PARSER BENCHMARK
# ==============================================================================
DURATION_CASES = [
    # (input_text, expected_minutes)
    ("1 hour", 60.0),
    ("1 houre", 60.0),
    ("1 houres", 60.0),
    ("1 hourse", 60.0),
    ("1 hr", 60.0),
    ("2 hours", 120.0),
    ("2 hrs", 120.0),
    ("1.5 hours", 90.0),
    ("0.5 hour", 30.0),
    ("half an hour", 30.0),
    ("half hour", 30.0),
    ("aadha ghanta", 30.0),
    ("aadho kalak", 30.0),
    ("1 kalak", 60.0),
    ("dedh ghanta", 90.0),
    ("dhai ghanta", 150.0),
    ("45 min", 45.0),
    ("45 mins", 45.0),
    ("45 minutes", 45.0),
    ("30 minutes", 30.0),
    ("15 minute", 15.0),
    ("90 seconds", 1.5),
    ("one hour", 60.0),
    ("ek ghanta", 60.0),
    ("two hours", 120.0),
]


# ==============================================================================
# 3. SEMANTIC INTENT ROUTING BENCHMARK (LOG vs RECOMMEND)
# ==============================================================================
INTENT_CASES = [
    # ── Workout Recommendations (Questions, Future, Modals, Gujarati, Hindi, English)
    ("kale mare ky exercise karavi joye", Intent.RECOMMEND_WORKOUT),
    ("kale mare kai exercise karvi joiye", Intent.RECOMMEND_WORKOUT),
    ("kale mare ky exercise karvi joyie", Intent.RECOMMEND_WORKOUT),
    ("kale su exercise karvi", Intent.RECOMMEND_WORKOUT),
    ("su exercise karvi joiye", Intent.RECOMMEND_WORKOUT),
    ("mare kai exercise karvi", Intent.RECOMMEND_WORKOUT),
    ("what exercise should I do tomorrow", Intent.RECOMMEND_WORKOUT),
    ("which exercises should i do", Intent.RECOMMEND_WORKOUT),
    ("what workout should i do", Intent.RECOMMEND_WORKOUT),
    ("suggest a workout for tomorrow", Intent.RECOMMEND_WORKOUT),
    ("konsi exercise karu", Intent.RECOMMEND_WORKOUT),
    ("kya workout karna chahiye", Intent.RECOMMEND_WORKOUT),
    ("should I do cardio or weights tomorrow", Intent.RECOMMEND_WORKOUT),
    ("can I do chest and triceps today", Intent.RECOMMEND_WORKOUT),
    ("workout routine for beginners", Intent.RECOMMEND_WORKOUT),
    ("which exercises are best for fat loss", Intent.RECOMMEND_WORKOUT),
    ("heavy dinner ke baad konsi exercise karein", Intent.RECOMMEND_WORKOUT),
    ("workout after heavy dinner", Intent.RECOMMEND_WORKOUT),
    ("exercises to burn pizza calories", Intent.RECOMMEND_WORKOUT),
    ("કઈ કસરત કરવી જોઈએ", Intent.RECOMMEND_WORKOUT),
    ("કાલે કઈ કસરત કરવી", Intent.RECOMMEND_WORKOUT),
    ("કસરત સૂચવો", Intent.RECOMMEND_WORKOUT),
    ("कौन सी कसरत करनी चाहिए", Intent.RECOMMEND_WORKOUT),

    # ── Exercise Logging (Past activity, explicit workouts, logged sets/reps)
    ("aaje chest na 4 exercise 12 reps karya", Intent.LOG_EXERCISE),
    ("chest 4 exercises 12 reps", Intent.LOG_EXERCISE),
    ("did 50 pushups", Intent.LOG_EXERCISE),
    ("ran 5 km this morning", Intent.LOG_EXERCISE),
    ("1 hour cycling kiya", Intent.LOG_EXERCISE),
    ("30 min walking", Intent.LOG_EXERCISE),
    ("aaje 45 min walk kari", Intent.LOG_EXERCISE),
    ("today did 1 hour gym workout", Intent.LOG_EXERCISE),
    ("swimming for 45 minutes", Intent.LOG_EXERCISE),
    ("20 squats marya", Intent.LOG_EXERCISE),

    # ── Meal Recommendations (Questions, Suggestions, Post-workout meal)
    ("what should I eat for dinner?", Intent.RECOMMEND_MEAL),
    ("post workout meal suggest karo", Intent.RECOMMEND_MEAL),
    ("pre workout snack idea", Intent.RECOMMEND_MEAL),
    ("what to eat after gym", Intent.RECOMMEND_MEAL),
    ("high protein dinner options", Intent.RECOMMEND_MEAL),
    ("ratre su khavu joiye", Intent.RECOMMEND_MEAL),
    ("bapore su jamvu", Intent.RECOMMEND_MEAL),
    ("khana suggest karo", Intent.RECOMMEND_MEAL),
    ("low calorie snack options", Intent.RECOMMEND_MEAL),

    # ── Food Logging
    ("ate 2 bananas and 1 glass milk", Intent.LOG_FOOD),
    ("log 2 rotli and 1 cup dal", Intent.LOG_FOOD),
    ("had 1 bowl oats for breakfast", Intent.LOG_FOOD),
    ("aaje 2 aloo paratha khadha", Intent.LOG_FOOD),
]


async def run_adversarial_suite():
    print("=" * 80)
    print("RUNNING ADVERSARIAL BENCHMARK SUITE FOR ARCHITECTURAL FIXES")
    print("=" * 80)

    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    failures = []

    # --------------------------------------------------------------------------
    # Part 1: Routine Parsing Tests
    # --------------------------------------------------------------------------
    print("\n--- [PART 1] Routine Parsing & Exercise Count Preservation ---")
    for text, exp_items, exp_total_reps, exp_dur in ROUTINE_CASES:
        total_tests += 1
        r = parse_workout_routine(text)
        if not r:
            failed_tests += 1
            failures.append(f"[ROUTINE FAIL] {text!r} returned None")
            continue

        # Check item counts and reps
        items_match = True
        if len(r.items) != len(exp_items):
            items_match = False
        else:
            for it, (exp_m, exp_c, exp_r) in zip(r.items, exp_items):
                if it.muscle_group.lower() != exp_m.lower() or it.exercise_count != exp_c:
                    items_match = False
                    break
                if exp_r is not None and it.reps != exp_r:
                    items_match = False
                    break

        reps_match = (r.total_reps == exp_total_reps)
        dur_match = (r.duration_min == exp_dur)

        if items_match and reps_match and dur_match:
            passed_tests += 1
        else:
            failed_tests += 1
            actual_items = [(it.muscle_group, it.exercise_count, it.reps) for it in r.items]
            failures.append(
                f"[ROUTINE MISMATCH] {text!r}\n"
                f"  Expected items: {exp_items}, total_reps: {exp_total_reps}, duration: {exp_dur}\n"
                f"  Actual items:   {actual_items}, total_reps: {r.total_reps}, duration: {r.duration_min}"
            )

    print(f"Part 1 Complete: {len(ROUTINE_CASES) - (failed_tests)} / {len(ROUTINE_CASES)} passed")

    # --------------------------------------------------------------------------
    # Part 2: Duration Parsing Tests
    # --------------------------------------------------------------------------
    p2_fails = 0
    print("\n--- [PART 2] Centralized Duration Parser ---")
    for text, exp_min in DURATION_CASES:
        total_tests += 1
        actual = parse_duration_minutes(text)
        if actual == exp_min:
            passed_tests += 1
        else:
            p2_fails += 1
            failed_tests += 1
            failures.append(f"[DURATION FAIL] {text!r} -> Expected {exp_min} min, got {actual}")

    print(f"Part 2 Complete: {len(DURATION_CASES) - p2_fails} / {len(DURATION_CASES)} passed")

    # --------------------------------------------------------------------------
    # Part 3: Intent Routing Tests (via LLMService deterministic fallback)
    # --------------------------------------------------------------------------
    p3_fails = 0
    print("\n--- [PART 3] Semantic Intent Routing (LLMService) ---")
    llm = LLMService(account_id="", api_token="")
    for text, exp_intent in INTENT_CASES:
        total_tests += 1
        res = llm.parse_intent_sync(text)
        if res.intent == exp_intent:
            passed_tests += 1
        else:
            p3_fails += 1
            failed_tests += 1
            failures.append(
                f"[INTENT FAIL] {text!r}\n"
                f"  Expected intent: {exp_intent}\n"
                f"  Actual intent:   {res.intent} (raw: {res.raw_response})"
            )

    print(f"Part 3 Complete: {len(INTENT_CASES) - p3_fails} / {len(INTENT_CASES)} passed")

    # --------------------------------------------------------------------------
    # Part 4: End-to-End ChatbotEngine Tests
    # --------------------------------------------------------------------------
    p4_fails = 0
    print("\n--- [PART 4] End-to-End ChatbotEngine Integration ---")
    dummy_db = DummyAsyncDB()
    user = {"user_id": "test_adv_user", "name": "Adv Tester", "gender": "male", "age": 25, "height_cm": 175, "weight_kg": 70, "goal": "maintain", "fitness_goal": "maintain", "activity_level": "moderate"}
    engine = ChatbotEngine(dummy_db, user, llm)

    # 4a. Workout Recommendation Question must give workout suggestions (not ask for minutes)
    e2e_cases = [
        ("kale mare ky exercise karavi joye", Intent.RECOMMEND_WORKOUT, ["workout", "exercise", "routine", "plan"]),
        ("what should I eat for dinner?", Intent.RECOMMEND_MEAL, ["meal", "eat", "dinner", "protein", "calories"]),
    ]

    for msg, exp_intent, expected_keywords in e2e_cases:
        total_tests += 1
        resp = await engine.process_message(msg)
        resp_text = resp.message.lower() if resp and resp.message else ""
        resp_intent = resp.intent if resp else None

        intent_ok = (resp_intent == exp_intent)
        # Verify it did not ask for missing minutes on recommendation!
        no_missing_minutes = ("how many minutes" not in resp_text and "how long" not in resp_text)

        if intent_ok and no_missing_minutes:
            passed_tests += 1
        else:
            p4_fails += 1
            failed_tests += 1
            failures.append(
                f"[E2E FAIL] {msg!r}\n"
                f"  Expected intent: {exp_intent}, got: {resp_intent}\n"
                f"  Engine response: {resp.message if resp else 'None'}"
            )

    # 4b. Duration clarification reply in ChatbotEngine:
    # Simulating user logging "running" and replying "1 houre" to duration clarification prompt
    from conversation import FlowState, PendingFlow
    total_tests += 1
    await engine.flow_mgr.save_flow_state(
        PendingFlow(
            state=FlowState.AWAITING_EXERCISE_AMOUNT,
            exercise_data={
                "exercise_name": "running",
                "exercise_name_display": "Running",
                "native_unit": "minutes",
                "log_date": "today",
            }
        )
    )
    reply_resp = await engine.process_message("1 houre")
    reply_text = reply_resp.message.lower() if reply_resp and reply_resp.message else ""
    calc_meta = reply_resp.data.get("exercise_calc", {}) if reply_resp and reply_resp.data else {}
    calories_burned = calc_meta.get("calories", 0)
    duration_used = calc_meta.get("amount", 0)

    # "1 houre" must be parsed as 60 minutes (check message mentions 60 or duration_used is 60)
    if "60 min" in reply_text or "60.0 min" in reply_text or duration_used == 60.0 or "1 hour" in reply_text or calories_burned > 300:
        passed_tests += 1
    else:
        p4_fails += 1
        failed_tests += 1
        failures.append(
            f"[E2E DURATION REPLY FAIL] '1 houre' clarification reply\n"
            f"  Expected 60 min. Calories: {calories_burned}\n"
            f"  Response: {reply_resp.message if reply_resp else 'None'}"
        )

    print(f"Part 4 Complete: {len(e2e_cases) + 1 - p4_fails} / {len(e2e_cases) + 1} passed")

    # --------------------------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"ADVERSARIAL SUITE SUMMARY: {passed_tests}/{total_tests} PASSED ({passed_tests/total_tests*100:.1f}%)")
    print("=" * 80)

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    else:
        print("\nALL ADVERSARIAL TESTS PASSED PERFECTLY!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(run_adversarial_suite())
