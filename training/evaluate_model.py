"""
evaluate_model.py
==================
A-to-Z evaluation benchmark for the Fitness Chatbot intent parser.
Measures the CURRENT Qwen3 model quality across all intent categories.

Usage:
    cd backend && python ../training/evaluate_model.py

What it tests:
  • Food logging — English, Gujarati, Hindi, Hinglish, typos
  • Exercise logging — duration, reps, sets, distance
  • Nutrition queries (get_calories)
  • Daily-log reads (get_summary / query_meal)
  • Multi-food extraction — all items must be present
  • Date extraction — today / yesterday
  • Variant extraction — ghee, fried, boiled …
  • Clarification — vague / junk must NOT enter food flow
  • Food vs Exercise disambiguation
  • Missing-detail detection
  • Profile and exercise-read queries
  • Safety: unsupported / off-topic inputs

Grading is behavioural:
  • Intent match          (required for every case)
  • Food/exercise field   (for log_food / log_exercise / get_calories)
  • Quantity field        (when specified in the case)
  • Meal type             (when specified)
  • Date                  (when specified)
  • Missing detail        (when specified)
  • Variant               (when specified)
  • Multi-food completeness (all expected foods present)

Output:
  • Per-case PASS/FAIL
  • Category accuracy
  • Overall score (before → after prompt changes)
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

# When output is redirected to a file, force line-buffering so results
# appear progressively rather than all at once on process exit.
import functools
_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

# Add backend to path when running from project root or training/
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_backend = os.path.join(_root, "backend")
if _backend not in sys.path:
    sys.path.insert(0, _backend)

# Pre-load .env so LLMService can find CF credentials without MongoDB.
_env_file = os.path.join(_backend, ".env")
if os.path.exists(_env_file):
    with open(_env_file) as _ef:
        for _line in _ef:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

from llm_service import LLMService

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation cases — completely separate from training/generate_dataset.py
# ──────────────────────────────────────────────────────────────────────────────
# Schema per case:
#   (category, user_input, expected_intent, checks_dict)
# checks_dict keys (all optional):
#   food        — expected food_query substring (case-insensitive)
#   foods       — list of substrings, ALL must appear in the foods[] array
#   quantity    — expected quantity substring
#   meal        — expected meal_type
#   date        — "today" | "yesterday"
#   variant     — expected variant substring
#   exercise    — expected exercise_query substring
#   missing     — expected missing_detail value
#   no_food     — True: assert that no food entered the flow (junk/clarify)

CASES: list[tuple[str, str, str, dict]] = [

    # ── FOOD LOGGING — English ────────────────────────────────────────────────
    ("food_en", "I had 3 idli for breakfast",
     "log_food", {"food": "idli", "quantity": "3", "meal": "breakfast"}),
    ("food_en", "ate 1 plate of chole bhature",
     "log_food", {"food": "chole bhature", "quantity": "1 plate"}),
    ("food_en", "had 150g grilled chicken breast",
     "log_food", {"food": "chicken", "quantity": "150g", "variant": "grilled"}),
    ("food_en", "drank 1 glass of mango lassi",
     "log_food", {"food": "lassi", "quantity": "1 glass"}),
    ("food_en", "ate 2 parathas with ghee for breakfast",
     "log_food", {"food": "paratha", "quantity": "2", "variant": "ghee", "meal": "breakfast"}),
    ("food_en", "had a bowl of vegetable soup for dinner",
     "log_food", {"food": "soup", "quantity": "1 bowl", "meal": "dinner"}),
    ("food_en", "ate 100g almonds as an afternoon snack",
     "log_food", {"food": "almond", "quantity": "100g", "meal": "snack"}),
    ("food_en", "had 1 cup of coffee with sugar",
     "log_food", {"food": "coffee", "quantity": "1 cup"}),
    ("food_en", "ate 2 samosa at tea time",
     "log_food", {"food": "samosa", "quantity": "2", "meal": "snack"}),
    ("food_en", "had 1 bowl of oatmeal for breakfast",
     "log_food", {"food": "oat", "quantity": "1 bowl", "meal": "breakfast"}),
    ("food_en", "ate gulab jamun after dinner",
     "log_food", {"food": "gulab jamun", "meal": "snack"}),
    ("food_en", "had 250ml whole milk",
     "log_food", {"food": "milk", "quantity": "250ml"}),

    # ── FOOD LOGGING — Gujarati / Roman-Gu ────────────────────────────────────
    ("food_gu", "aaje me 2 thepla khadha",
     "log_food", {"food": "thepla", "quantity": "2"}),
    ("food_gu", "savare bhakri ane chhas lidha",
     "log_food", {"foods": ["bhakri", "chhas"]}),
    ("food_gu", "me aaje lunch ma khichdi khadhi",
     "log_food", {"food": "khichdi", "meal": "lunch"}),
    ("food_gu", "hu sanje 1 samosa khadho",
     "log_food", {"food": "samosa", "quantity": "1", "meal": "snack"}),
    ("food_gu", "raatre me 2 rotli ane dal khadha",
     "log_food", {"foods": ["roti", "dal"], "meal": "dinner"}),
    ("food_gu", "aaje breakfast ma poha khadha",
     "log_food", {"food": "poha", "meal": "breakfast"}),

    # ── FOOD LOGGING — Hindi / Hinglish ───────────────────────────────────────
    ("food_hi", "maine subah 2 ande khaaye",
     "log_food", {"food": "egg", "quantity": "2", "meal": "breakfast"}),
    ("food_hi", "lunch me dal chawal khayi",
     "log_food", {"foods": ["dal", "rice"], "meal": "lunch"}),
    ("food_hi", "shaam ko chai ke saath 2 biscuit khayi",
     "log_food", {"foods": ["chai", "biscuit"], "meal": "snack"}),
    ("food_hi", "raat ko 3 roti aur sabzi khayi",
     "log_food", {"food": "roti", "quantity": "3", "meal": "dinner"}),
    ("food_hi", "subah nashte me aloo paratha ghee ke saath",
     "log_food", {"food": "aloo paratha", "variant": "ghee", "meal": "breakfast"}),
    ("food_hi", "dopahar ko rajma chawal khaya",
     "log_food", {"foods": ["rajma"], "meal": "lunch"}),

    # ── FOOD LOGGING — Typos / Phonetic ───────────────────────────────────────
    ("food_typo", "ate palak paneer",
     "log_food", {"food": "palak paneer"}),
    ("food_typo", "had samoosa for snacks",
     "log_food", {"food": "samosa", "meal": "snack"}),
    ("food_typo", "aaje methi thaepla khadha",
     "log_food", {"food": "thepla"}),
    ("food_typo", "had dahi wada",
     "log_food", {"food": "dahi vada"}),
    ("food_typo", "ate 1 bowel rajmah",
     "log_food", {"food": "rajma", "quantity": "1 bowl"}),
    ("food_typo", "had masla dosa for breakfast",
     "log_food", {"food": "dosa", "meal": "breakfast"}),

    # ── MULTI-FOOD ─────────────────────────────────────────────────────────────
    ("multi_food", "had 2 roti, dal and salad for lunch",
     "log_food", {"foods": ["roti", "dal", "salad"], "meal": "lunch"}),
    ("multi_food", "Breakfast was idli, sambar and coconut chutney",
     "log_food", {"foods": ["idli", "sambar"]}),
    ("multi_food", "ate paneer tikka and butter naan for dinner",
     "log_food", {"foods": ["paneer tikka", "naan"], "meal": "dinner"}),
    ("multi_food", "aaje me breakfast ma thepla, chundo ane chai lidha",
     "log_food", {"foods": ["thepla", "chai"]}),
    ("multi_food", "dinner was grilled chicken, rice and steamed vegetables",
     "log_food", {"foods": ["chicken", "rice"]}),
    ("multi_food", "maine subah doodh, anda aur toast khayi",
     "log_food", {"foods": ["milk", "egg", "toast"]}),

    # ── EXERCISE LOGGING ──────────────────────────────────────────────────────
    ("exercise", "did 30 minutes of jogging",
     "log_exercise", {"exercise": "jog", "duration": 30.0}),
    ("exercise", "ran 5 km this morning",
     "log_exercise", {"exercise": "running", "distance": 5.0}),
    ("exercise", "completed 3 sets of 12 squats",
     "log_exercise", {"exercise": "squat", "sets": 3.0, "reps": 12.0}),
    ("exercise", "did 45 minutes of yoga",
     "log_exercise", {"exercise": "yoga", "duration": 45.0}),
    ("exercise", "aaj 40 minute cycling ki",
     "log_exercise", {"exercise": "cycling", "duration": 40.0}),
    ("exercise", "me aaje 50 push-ups kara",
     "log_exercise", {"exercise": "push", "reps": 50.0}),
    ("exercise", "walked 10000 steps today",
     "log_exercise", {"exercise": "walk"}),
    ("exercise", "aaj mene 4 sets of 10 bench press kiya",
     "log_exercise", {"exercise": "bench press", "sets": 4.0, "reps": 10.0}),
    ("exercise", "swam 1 km in the pool",
     "log_exercise", {"exercise": "swim", "distance": 1.0}),
    ("exercise", "did 20 minutes of vigorous HIIT",
     "log_exercise", {"exercise": "HIIT", "duration": 20.0}),

    # ── NUTRITION QUERIES ─────────────────────────────────────────────────────
    ("nutrition", "how many calories in 1 roti",
     "get_calories", {"food": "roti"}),
    ("nutrition", "how much protein is in 100g paneer",
     "get_calories", {"food": "paneer", "quantity": "100g"}),
    ("nutrition", "kitni calories hoti hai ek samosa me",
     "get_calories", {"food": "samosa"}),
    ("nutrition", "how much fat in 1 tablespoon ghee",
     "get_calories", {"food": "ghee"}),
    ("nutrition", "calories in biryani 1 plate",
     "get_calories", {"food": "biryani", "quantity": "1 plate"}),
    ("nutrition", "how many carbs in 1 chapati",
     "get_calories", {"food": "chapati"}),
    ("nutrition", "1 banana mein kitni calories hai",
     "get_calories", {"food": "banana"}),

    # ── DAILY LOG READS ───────────────────────────────────────────────────────
    ("daily_log", "show me today's summary", "get_summary", {"date": "today"}),
    ("daily_log", "aaj ka summary dikhao", "get_summary", {"date": "today"}),
    ("daily_log", "how many calories did I eat today", "get_summary", {"date": "today"}),
    ("daily_log", "total calories today", "get_summary", {"date": "today"}),
    ("daily_log", "gatkale ketli calorie thai", "get_summary", {"date": "yesterday"}),
    ("daily_log", "show yesterday's summary", "get_summary", {"date": "yesterday"}),
    ("daily_log", "what did I eat for breakfast today", "query_meal", {"meal": "breakfast"}),
    ("daily_log", "aaje lunch ma su khadhu", "query_meal", {"meal": "lunch"}),
    ("daily_log", "dinner items today dikhao", "query_meal", {"meal": "dinner"}),

    # ── DATE EXTRACTION ───────────────────────────────────────────────────────
    ("date", "yesterday I had dal chawal for lunch",
     "log_food", {"food": "dal", "meal": "lunch", "date": "yesterday"}),
    ("date", "gatkale me thepla khadhi",
     "log_food", {"food": "thepla", "date": "yesterday"}),
    ("date", "kal subah paratha khaya",
     "log_food", {"food": "paratha", "date": "yesterday", "meal": "breakfast"}),
    ("date", "had biryani yesterday for dinner",
     "log_food", {"food": "biryani", "date": "yesterday", "meal": "dinner"}),
    ("date", "aaj lunch me dal khayi",
     "log_food", {"food": "dal", "date": "today", "meal": "lunch"}),

    # ── MISSING DETAIL ────────────────────────────────────────────────────────
    ("missing", "ate paneer tikka for dinner",
     "log_food", {"food": "paneer tikka", "meal": "dinner", "missing": "quantity"}),
    ("missing", "had 2 roti",
     "log_food", {"food": "roti", "quantity": "2", "missing": "meal_type"}),
    ("missing", "drank lassi",
     "log_food", {"food": "lassi", "missing": "quantity"}),
    ("missing", "aaje me biryani khadhi",
     "log_food", {"food": "biryani", "missing": "quantity"}),
    ("missing", "aaj 50 push-ups kiye",
     "log_exercise", {}),   # reps given, nothing missing

    # ── CLARIFICATION / JUNK ─────────────────────────────────────────────────
    ("junk", "abcdef", "clarification_needed", {"no_food": True}),
    ("junk", "qwerty", "clarification_needed", {"no_food": True}),
    ("junk", "what is the weather today", "clarification_needed", {"no_food": True}),
    ("junk", "tell me a joke", "clarification_needed", {"no_food": True}),
    ("junk", "book me a cab", "clarification_needed", {"no_food": True}),
    ("junk", "I ate something", "clarification_needed", {}),
    ("junk", "had a bit of that thing", "clarification_needed", {}),
    ("junk", "kuch khaya", "clarification_needed", {}),

    # ── FOOD vs EXERCISE DISAMBIGUATION ───────────────────────────────────────
    ("disambiguation", "jogging 30 minutes",
     "log_exercise", {"exercise": "jog", "duration": 30.0}),
    ("disambiguation", "ate 4 dates after workout",
     "log_food", {"food": "date"}),
    ("disambiguation", "had a protein bar before gym",
     "log_food", {"food": "protein bar"}),
    ("disambiguation", "did 20 push-ups before breakfast",
     "log_exercise", {"exercise": "push"}),
    ("disambiguation", "cycling 10 km",
     "log_exercise", {"exercise": "cycling", "distance": 10.0}),

    # ── PROFILE / GREETING ────────────────────────────────────────────────────
    ("profile", "show my profile", "get_profile", {}),
    ("profile", "mera profile dikhao", "get_profile", {}),
    ("greeting", "hi", "greeting", {}),
    ("greeting", "namaste", "greeting", {}),
    ("greeting", "hello how are you", "greeting", {}),

    # ── EXERCISE READS ────────────────────────────────────────────────────────
    ("exercise_read", "what exercise did I do today",
     "query_exercise", {"date": "today"}),
    ("exercise_read", "aaj mene kya exercise ki",
     "query_exercise", {"date": "today"}),
    ("exercise_read", "how many calories did I burn today",
     "query_exercise", {"date": "today"}),

    # ── SKIP MEAL ─────────────────────────────────────────────────────────────
    ("skip_meal", "I skipped breakfast today",
     "skip_meal", {"meal": "breakfast"}),
    ("skip_meal", "aaj lunch nahi khaya",
     "skip_meal", {"meal": "lunch"}),
]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()


def _check(result, checks: dict) -> tuple[bool, list[str]]:
    """Return (passed, list_of_failures)."""
    failures = []

    # food substring
    if "food" in checks:
        fq = _norm(result.food_query or "")
        if _norm(checks["food"]) not in fq:
            failures.append(f"food_query={result.food_query!r} expected ~{checks['food']!r}")

    # foods[] — ALL substrings must appear across the foods array
    if "foods" in checks:
        foods_str = " | ".join(_norm(f.get("food_query", "")) for f in (result.foods or []))
        for want in checks["foods"]:
            if _norm(want) not in foods_str:
                failures.append(f"foods missing {want!r} in {foods_str!r}")

    # quantity
    if "quantity" in checks:
        qty = _norm(result.quantity or "")
        if _norm(checks["quantity"]) not in qty:
            failures.append(f"quantity={result.quantity!r} expected ~{checks['quantity']!r}")

    # meal
    if "meal" in checks:
        if _norm(result.meal_type or "") != _norm(checks["meal"]):
            failures.append(f"meal_type={result.meal_type!r} expected {checks['meal']!r}")

    # date
    if "date" in checks:
        if _norm(result.date or "") != _norm(checks["date"]):
            failures.append(f"date={result.date!r} expected {checks['date']!r}")

    # variant
    if "variant" in checks:
        var = _norm(result.variant or "")
        if _norm(checks["variant"]) not in var:
            failures.append(f"variant={result.variant!r} expected ~{checks['variant']!r}")

    # exercise
    if "exercise" in checks:
        eq = _norm(result.exercise_query or "")
        if _norm(checks["exercise"]) not in eq:
            failures.append(f"exercise_query={result.exercise_query!r} expected ~{checks['exercise']!r}")

    # duration
    if "duration" in checks:
        if result.duration_min != checks["duration"]:
            failures.append(f"duration_min={result.duration_min} expected {checks['duration']}")

    # distance
    if "distance" in checks:
        if result.distance != checks["distance"]:
            failures.append(f"distance={result.distance} expected {checks['distance']}")

    # sets / reps
    if "sets" in checks and result.sets != checks["sets"]:
        failures.append(f"sets={result.sets} expected {checks['sets']}")
    if "reps" in checks and result.reps != checks["reps"]:
        failures.append(f"reps={result.reps} expected {checks['reps']}")

    # missing_detail
    if "missing" in checks:
        if result.missing_detail != checks["missing"]:
            failures.append(f"missing_detail={result.missing_detail!r} expected {checks['missing']!r}")

    # no_food — junk must NOT have a resolved food
    if checks.get("no_food"):
        if result.food_query and result.food_query.strip():
            failures.append(f"food_query={result.food_query!r} should be empty for junk")

    return len(failures) == 0, failures


async def run_eval():
    llm = LLMService()

    print("=" * 96)
    print("  FITNESS CHATBOT — Intent Parser Evaluation Benchmark")
    print("  Model:", os.getenv("CF_MODEL", "@cf/qwen/qwen3-30b-a3b-fp8"))
    print("=" * 96)
    print(f"{'#':>3}  {'CAT':<16} {'INPUT':<44} {'EXP':<16} {'GOT':<16} RESULT")
    print("-" * 96)

    results: list[dict] = []
    for i, (cat, user_input, expected_intent, checks) in enumerate(CASES, 1):
        try:
            result = await llm.parse_intent(user_input)
        except Exception as e:
            result = type("R", (), {"intent": "error", "food_query": None,
                                    "foods": [], "quantity": None, "meal_type": None,
                                    "date": "today", "variant": None, "exercise_query": None,
                                    "duration_min": None, "distance": None, "sets": None,
                                    "reps": None, "missing_detail": None})()
            result.error = str(e)

        intent_ok = _norm(result.intent) == _norm(expected_intent)
        if intent_ok:
            passed, field_failures = _check(result, checks)
        else:
            passed = False
            field_failures = [f"intent={result.intent!r}"]

        mark = "PASS" if passed else "FAIL"
        got_intent = (result.intent or "")[:16]
        print(f"{i:>3}  {cat:<16} {user_input[:43]:<44} {expected_intent:<16} {got_intent:<16} {mark}")

        results.append({
            "n": i, "cat": cat, "input": user_input,
            "expected_intent": expected_intent, "got_intent": result.intent,
            "passed": passed, "failures": field_failures,
        })
        await asyncio.sleep(0.1)  # avoid rate limiting

    # Summary
    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    acc    = passed / total * 100 if total else 0

    cats: dict[str, list[int]] = {}
    for r in results:
        cats.setdefault(r["cat"], [0, 0])
        cats[r["cat"]][0] += 1
        if r["passed"]:
            cats[r["cat"]][1] += 1

    print("\n" + "=" * 96)
    print("  SUMMARY")
    print("=" * 96)
    print(f"  Total  : {total}")
    print(f"  Passed : {passed}")
    print(f"  Failed : {failed}")
    print(f"  Score  : {acc:.1f}%")
    print("\n  Category accuracy:")
    for c in sorted(cats):
        tot, pas = cats[c]
        print(f"    {c:<20} {pas}/{tot}  ({pas/tot*100:.0f}%)")

    fails = [r for r in results if not r["passed"]]
    if fails:
        print("\n  FAILURES:")
        by_cat: dict[str, list[dict]] = {}
        for r in fails:
            by_cat.setdefault(r["cat"], []).append(r)
        for c in sorted(by_cat):
            print(f"  [{c}]")
            for r in by_cat[c]:
                print(f"    #{r['n']} {r['input']!r}")
                for f in r["failures"]:
                    print(f"       {f}")
    else:
        print("\n  No failures.")

    print("=" * 96)
    print()
    print("To compare before/after a prompt change, run this script twice")
    print("and compare the Score: line.")

    return acc, results


if __name__ == "__main__":
    asyncio.run(run_eval())
