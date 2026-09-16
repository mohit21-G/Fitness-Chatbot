"""
Deterministic benchmark for Daily Log vs Food Logging ROUTING.

Purpose
-------
Verify that the CURRENT production pipeline routes messages correctly:

  • Daily Log queries  (view/read logged data — totals, a meal's foods, a day's
    summary) MUST go straight to Daily Log RETRIEVAL and MUST NEVER trigger
    food/quantity/variant detection or the food-logging flow.
  • Food Logging queries (the user reporting what they ate, with/without a
    quantity) MUST remain Food Logging.

The benchmark drives the EXACT production entry point
`ChatbotEngine.process_message(...)` — the same call the chatbot API makes —
using the production Qwen intent parser, DB, and routing. It does NOT modify
production code, prompts, the model, the DB, or routing.

Intent taxonomy (from backend/llm_service.py Intent)
----------------------------------------------------
  DAILY-LOG RETRIEVAL intents (read-only, no food detection):
      get_summary   -> whole-day totals / summary (today or yesterday)
      query_meal    -> a specific meal's foods, or a day's logged items
  FOOD-LOGGING intent (food/quantity/variant detection expected):
      log_food      -> user is recording eaten food

Outcome extraction (production-faithful)
----------------------------------------
For each message we read the resulting ChatResponse + flow state and classify
the ROUTE the pipeline actually took:
  • "daily_log"     -> intent in {get_summary, query_meal} and NO food flow was
                       started (no items, no confirmation, no awaiting-variant/
                       quantity, no multi_choice).
  • "food_logging"  -> intent == log_food OR a food flow was started
                       (needs_confirmation / pending_action / flow items /
                       intent contains "flow:awaiting" or "multi_choice").
  • "other"         -> anything else (greeting/unknown/clarification/exercise).

We ALSO independently compute `food_detection_triggered` (bool): True if the
message caused ANY food/quantity/variant detection. For a Daily Log query this
MUST be False — that is the critical guarantee under test.

Grading
-------
A case PASSES when:
  • route matches expected_route, AND
  • for daily_log cases: food_detection_triggered is False, AND
  • date matches expected_date (today / yesterday) when specified, AND
  • meal matches expected_meal when specified.
  • for food_logging cases: food_detection_triggered is True.

Usage:  python test_daily_log_routing_benchmark.py   (needs MongoDB; live Qwen)
"""
import asyncio
import io
import sys
from datetime import datetime, timezone, timedelta

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

from database import async_db
from chatbot_engine import ChatbotEngine
from llm_service import LLMService
from conversation import PendingFlow

IST_TZ = timezone(timedelta(hours=5, minutes=30))


def _today_iso() -> str:
    return datetime.now(IST_TZ).date().isoformat()


def _yesterday_iso() -> str:
    return (datetime.now(IST_TZ).date() - timedelta(days=1)).isoformat()


# ---------------------------------------------------------------------------
# Test cases: (category, message, expected_route, expected_date, expected_meal)
#   expected_route : "daily_log" | "food_logging"
#   expected_date  : "today" | "yesterday" | None (don't-care)
#   expected_meal  : "breakfast"|"lunch"|"dinner"|"snack"|"evening" | None
# ---------------------------------------------------------------------------
DL = "daily_log"
FL = "food_logging"

CASES: list[tuple[str, str, str, str | None, str | None]] = [

    # ── TODAY: whole-day total / summary (English) ──────────────────────
    ("today_total", "show today's summary",                 DL, "today", None),
    ("today_total", "today summary",                        DL, "today", None),
    ("today_total", "how many calories did I eat today",    DL, "today", None),
    ("today_total", "total calories today",                 DL, "today", None),
    ("today_total", "what's my calorie count for today",    DL, "today", None),
    ("today_total", "show me today's daily log",            DL, "today", None),
    ("today_total", "give me today's report",               DL, "today", None),

    # ── TODAY total — Hinglish / Hindi / Gujarati / Roman Gujarati ──────
    ("today_total", "aaj ka summary dikhao",                DL, "today", None),
    ("today_total", "aaj kitni calories khayi",             DL, "today", None),
    ("today_total", "aaj ka total kitna hua",               DL, "today", None),
    ("today_total", "aaje ketli calorie thai",              DL, "today", None),
    ("today_total", "aaje no total batao",                  DL, "today", None),
    ("today_total", "aaje nu summary batavo",               DL, "today", None),
    ("today_total", "mera aaj ka report dikhao",            DL, "today", None),

    # ── YESTERDAY: whole-day total / summary ────────────────────────────
    ("yesterday_total", "show yesterday's summary",         DL, "yesterday", None),
    ("yesterday_total", "yesterday total calories",         DL, "yesterday", None),
    ("yesterday_total", "how many calories yesterday",      DL, "yesterday", None),
    ("yesterday_total", "kal kitni calories thi",           DL, "yesterday", None),
    ("yesterday_total", "kal ka summary dikhao",            DL, "yesterday", None),
    ("yesterday_total", "gay kale ketli calorie hati",       DL, "yesterday", None),
    ("yesterday_total", "gay kal no total batao",            DL, "yesterday", None),
    ("yesterday_total", "kal ka daily log dikhao",          DL, "yesterday", None),

    # ── TODAY breakfast foods ──────────────────────────────────────────
    ("today_breakfast", "what did I eat for breakfast today",   DL, "today", "breakfast"),
    ("today_breakfast", "show my breakfast",                    DL, "today", "breakfast"),
    ("today_breakfast", "aaje breakfast ma su khadhu",          DL, "today", "breakfast"),
    ("today_breakfast", "aaj nashte me kya khaya",              DL, "today", "breakfast"),
    ("today_breakfast", "breakfast ma su lidhu aaje",           DL, "today", "breakfast"),
    ("today_breakfast", "aaj subah kya khaya tha",              DL, "today", "breakfast"),

    # ── TODAY lunch foods ──────────────────────────────────────────────
    ("today_lunch", "what did I have for lunch today",      DL, "today", "lunch"),
    ("today_lunch", "show my lunch",                        DL, "today", "lunch"),
    ("today_lunch", "aaje lunch ma su khadhu",              DL, "today", "lunch"),
    ("today_lunch", "aaj lunch me kya khaya",               DL, "today", "lunch"),
    ("today_lunch", "bapore su jamyu aaje",                 DL, "today", "lunch"),
    ("today_lunch", "aaj dopahar ka khana dikhao",          DL, "today", "lunch"),

    # ── TODAY dinner foods ─────────────────────────────────────────────
    ("today_dinner", "what did I eat for dinner today",     DL, "today", "dinner"),
    ("today_dinner", "show my dinner",                      DL, "today", "dinner"),
    ("today_dinner", "aaje dinner ma su khadhu",            DL, "today", "dinner"),
    ("today_dinner", "aaj raat ko kya khaya",               DL, "today", "dinner"),
    ("today_dinner", "raatre su jamyu",                     DL, "today", "dinner"),
    ("today_dinner", "aaj dinner me kya tha",               DL, "today", "dinner"),

    # ── YESTERDAY meal foods (breakfast/lunch/dinner) ──────────────────
    ("yesterday_meal", "what did I eat for breakfast yesterday", DL, "yesterday", "breakfast"),
    ("yesterday_meal", "kal breakfast me kya khaya",             DL, "yesterday", "breakfast"),
    ("yesterday_meal", "gay kale lunch ma su khadhu",             DL, "yesterday", "lunch"),
    ("yesterday_meal", "kal dinner me kya khaya tha",            DL, "yesterday", "dinner"),
    ("yesterday_meal", "gay kal raatre su jamyu",                 DL, "yesterday", "dinner"),
    ("yesterday_meal", "yesterday's lunch foods",                DL, "yesterday", "lunch"),

    # ── snack / evening reads ──────────────────────────────────────────
    ("today_snack", "what snacks did I have today",         DL, "today", "snack"),
    ("today_snack", "aaje nasta ma su khadhu",              DL, "today", "snack"),
    ("today_snack", "aaj snack me kya khaya",               DL, "today", "snack"),
    ("today_evening", "aaje sanje su khadhu",               DL, "today", "evening"),
    ("today_evening", "show my evening snack today",        DL, "today", "evening"),

    # ── repeated / rephrased versions of the SAME query ────────────────
    ("rephrase_total", "today total calories",              DL, "today", None),
    ("rephrase_total", "calories eaten today total",        DL, "today", None),
    ("rephrase_total", "how much did I eat in total today", DL, "today", None),
    ("rephrase_total", "aaj total calorie kitni hui",       DL, "today", None),
    ("rephrase_total", "aaje total ketli calorie",          DL, "today", None),
    ("rephrase_bfast", "today breakfast foods",             DL, "today", "breakfast"),
    ("rephrase_bfast", "show breakfast items today",        DL, "today", "breakfast"),
    ("rephrase_bfast", "aaje savar ma su khadhu",           DL, "today", "breakfast"),
    ("rephrase_bfast", "aaj nashta kya tha",                DL, "today", "breakfast"),

    # ── mixed Daily Log queries (whole log, no meal filter) ────────────
    ("mixed_log", "show my full day log",                   DL, "today", None),
    ("mixed_log", "aaj ka pura log dikhao",                 DL, "today", None),
    ("mixed_log", "aaje nu aakho log batavo",               DL, "today", None),
    ("mixed_log", "what all did I eat today",               DL, "today", None),
    ("mixed_log", "aaje me su su khadhu",                   DL, "today", None),
    ("mixed_log", "kal ka pura log dikhao",                 DL, "yesterday", None),

    # ══════════════════════════════════════════════════════════════════
    #  FOOD LOGGING — these MUST stay food logging (detection expected)
    # ══════════════════════════════════════════════════════════════════
    ("log_qty", "I had 2 roti for lunch",                   FL, None, None),
    ("log_qty", "aaje breakfast ma 3 idli khadhi",          FL, None, None),
    ("log_qty", "maine 1 bowl dal tadka khaya",             FL, None, None),
    ("log_qty", "ate 2 samosa",                             FL, None, None),
    ("log_qty", "1 plate biryani for dinner",               FL, None, None),
    ("log_qty", "aaj lunch me 2 chapati aur sabzi khayi",   FL, None, None),
    ("log_qty", "me aaje 1 glass chhas pidhi",              FL, None, None),
    ("log_qty", "2 bhakri ghee sathe khadhi",               FL, None, None),
    ("log_qty", "had a pizza",                              FL, None, None),
    ("log_qty", "dinner me 1 katori khichdi khadhi",        FL, None, None),

    ("log_noqty", "I ate paneer",                           FL, None, None),
    ("log_noqty", "aaje me dosa khadho",                    FL, None, None),
    ("log_noqty", "chocolate ice cream khadhu",             FL, None, None),
    ("log_noqty", "lassi pidhi",                            FL, None, None),
    ("log_noqty", "breakfast me poha khaya",                FL, None, None),
    ("log_noqty", "aaje me thepla khadha",                  FL, None, None),
    ("log_noqty", "jeera rice and dal makhani for lunch",   FL, None, None),
    ("log_noqty", "samosa khaya aaj",                       FL, None, None),
]


def _flow_food_detection(resp, flow) -> bool:
    """Return True if the message triggered ANY food/quantity/variant detection.

    Markers (any one indicates the food-logging flow was engaged):
      • response intent == log_food
      • intent contains 'flow:awaiting' (awaiting_variant / awaiting_quantity)
      • intent contains 'multi_choice'
      • needs_confirmation OR pending_action set (asking to save / confirm food)
      • flow gained food items / pending_items / a resolved food_name
    """
    intent = (resp.intent or "").lower()
    if intent == "log_food":
        return True
    if "flow:awaiting" in intent or "multi_choice" in intent or "awaiting" in intent:
        return True
    if getattr(resp, "needs_confirmation", False):
        return True
    if getattr(resp, "pending_action", None):
        # A pending_action of type log_food is food detection; ignore others.
        pa = resp.pending_action or {}
        if isinstance(pa, dict) and pa.get("type") in (None, "log_food"):
            return True
        return True
    try:
        if flow is not None:
            if flow.items or flow.pending_items:
                return True
            if getattr(flow, "food_name", None) or getattr(flow, "food_query", None):
                return True
            st = (getattr(flow, "state", "") or "").lower()
            if "awaiting" in st and st not in ("", "idle"):
                return True
    except Exception:
        pass
    return False


async def _route(engine: ChatbotEngine, text: str):
    """Run one message through production and classify the route it took."""
    await engine.flow_mgr.save_flow_state(PendingFlow())
    resp = await engine.process_message(text, auto_log=False)
    intent = (resp.intent or "").lower()

    try:
        flow = await engine.flow_mgr.get_pending_flow()
    except Exception:
        flow = None

    food_triggered = _flow_food_detection(resp, flow)

    if intent in ("get_summary", "query_meal") and not food_triggered:
        route = "daily_log"
    elif intent == "log_food" or food_triggered:
        route = "food_logging"
    else:
        route = "other"

    # Date + meal actually resolved (read from response data where available).
    data = resp.data if isinstance(resp.data, dict) else {}
    got_date = data.get("date")
    got_meal = data.get("meal_type")
    return route, got_date, got_meal, food_triggered, intent


def _date_matches(expected: str | None, got_iso: str | None) -> bool:
    if expected is None:
        return True
    if not got_iso:
        return False
    want = _today_iso() if expected == "today" else _yesterday_iso()
    return got_iso == want


def _grade(exp_route, exp_date, exp_meal, route, got_date, got_meal, food_triggered):
    if route != exp_route:
        return False
    if exp_route == "daily_log":
        if food_triggered:
            return False
        if not _date_matches(exp_date, got_date):
            return False
        if exp_meal is not None and (got_meal or "").lower() != exp_meal:
            return False
        return True
    # food_logging
    return food_triggered


async def main():
    user = {
        "user_id": "benchmark_routing_user",
        "name": "Routing Benchmark User",
        "age": 30, "gender": "male", "height_cm": 175, "weight_kg": 70,
        "activity_level": "moderate", "fitness_goal": "maintain",
    }
    engine = ChatbotEngine(async_db, user, LLMService())
    await engine.flow_mgr.save_flow_state(PendingFlow())

    print("=" * 100)
    print("  Daily Log vs Food Logging — ROUTING Benchmark (production config, UNCHANGED)")
    print("=" * 100)
    print(f"{'#':>3}  {'TYPE':<17} {'MESSAGE':<38} {'EXP':<11} {'GOT':<11} {'FOODdet':<8} RESULT")
    print("-" * 100)

    results = []
    for i, (cat, text, exp_route, exp_date, exp_meal) in enumerate(CASES, 1):
        try:
            route, got_date, got_meal, food_triggered, intent = await _route(engine, text)
        except Exception as e:
            route, got_date, got_meal, food_triggered, intent = f"error:{e}", None, None, False, "error"
        passed = _grade(exp_route, exp_date, exp_meal, route, got_date, got_meal, food_triggered)
        results.append({
            "n": i, "type": cat, "input": text,
            "exp_route": exp_route, "exp_date": exp_date, "exp_meal": exp_meal,
            "route": route, "got_date": got_date, "got_meal": got_meal,
            "food_triggered": food_triggered, "intent": intent, "passed": passed,
        })
        mark = "PASS" if passed else "FAIL"
        fd = "YES" if food_triggered else "no"
        print(f"{i:>3}  {cat:<17} {text[:37]:<38} {exp_route:<11} {route[:11]:<11} {fd:<8} {mark}")
        await asyncio.sleep(0.05)

    await engine.flow_mgr.save_flow_state(PendingFlow())

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    acc = (passed / total * 100.0) if total else 0.0

    # Critical guarantee: no Daily Log query triggered food detection.
    dl_cases = [r for r in results if r["exp_route"] == "daily_log"]
    dl_leaks = [r for r in dl_cases if r["food_triggered"]]

    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    print(f"  Total tests : {total}")
    print(f"  Passed      : {passed}")
    print(f"  Failed      : {failed}")
    print(f"  Accuracy    : {acc:.1f}%")
    print(f"  Daily Log queries that WRONGLY triggered food detection: "
          f"{len(dl_leaks)} / {len(dl_cases)}")

    print("\n  Accuracy by type:")
    types: dict[str, list[int]] = {}
    for r in results:
        types.setdefault(r["type"], [0, 0])
        types[r["type"]][0] += 1
        if r["passed"]:
            types[r["type"]][1] += 1
    for t in sorted(types):
        tot, pas = types[t]
        print(f"    {t:<18} {pas}/{tot}  ({pas / tot * 100:.0f}%)")

    fails = [r for r in results if not r["passed"]]
    if fails:
        print("\n  FAILED examples (grouped by type):")
        by_type: dict[str, list[dict]] = {}
        for r in fails:
            by_type.setdefault(r["type"], []).append(r)
        for t in sorted(by_type):
            print(f"    [{t}]")
            for r in by_type[t]:
                print(f"       input={r['input']!r}")
                print(f"         expected: route={r['exp_route']} date={r['exp_date']} meal={r['exp_meal']}")
                print(f"         actual  : route={r['route']} intent={r['intent']!r} "
                      f"date={r['got_date']} meal={r['got_meal']} food_detection={r['food_triggered']}")
    else:
        print("\n  No failures.")

    if dl_leaks:
        print("\n  !! CRITICAL — Daily Log queries that triggered food detection:")
        for r in dl_leaks:
            print(f"       input={r['input']!r}  intent={r['intent']!r}  route={r['route']}")

    print("=" * 100)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
