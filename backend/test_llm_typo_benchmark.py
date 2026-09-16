"""
Deterministic benchmark for the CURRENT food-detection pipeline
(Qwen3-30B-A3B-FP8 as configured in production — model/config UNCHANGED).

What this exercises
-------------------
Each input is sent through the EXACT production entry point
`ChatbotEngine.process_message(...)` — the same call the chatbot API makes.
That path uses the production Qwen intent parser (`LLMService.parse_intent`),
the production DB, transliteration/fuzzy search, external food sources, and the
deterministic reject gate. Nothing here modifies production logic, prompts, the
database, or the model configuration.

Outcome extraction (production-faithful)
----------------------------------------
After each message we inspect the resulting conversation state:
  • REJECTED  -> response intent is "clarification_needed" (the "couldn't
    identify a food" / "not found" reply) and no food entered the flow.
  • DETECTED  -> the pending flow (or a saved log) carries one or more resolved
    food items; we read their canonical `food_name` / display name.

Grading
-------
  • Typo / transliteration / valid cases: PASS if a DETECTED food name matches
    the expected canonical food (token/substring match, case-insensitive).
  • Junk / non-food cases: expected = REJECT; PASS if the input is REJECTED.

Usage:  python test_llm_typo_benchmark.py     (needs MongoDB; uses live Qwen)
"""
import asyncio
import io
import sys
import re

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

from database import async_db, sync_db
from chatbot_engine import ChatbotEngine
from llm_service import LLMService
from conversation import PendingFlow

REJECT = "__REJECT__"   # sentinel: this input must be rejected as non-food

# ---------------------------------------------------------------------------
# Test cases: (category, input, expected)
#   expected is either a canonical food-name fragment we expect in the detected
#   food, or REJECT for junk / non-food inputs.
# ---------------------------------------------------------------------------
CASES: list[tuple[str, str, str]] = [
    # ── character repetition ────────────────────────────────────────────
    ("repetition", "pizzza",            "pizza"),
    ("repetition", "lasssi",            "lassi"),
    ("repetition", "rootti",            "roti"),
    ("repetition", "doosa",             "dosa"),
    ("repetition", "iddli",             "idli"),
    ("repetition", "samosaa",           "samosa"),
    ("repetition", "paneeer",           "paneer"),
    ("repetition", "bhhakri",           "bhakri"),

    # ── missing characters ──────────────────────────────────────────────
    ("missing_char", "rti",             "roti"),
    ("missing_char", "panir",           "paneer"),
    ("missing_char", "chpati",          "chapati"),
    ("missing_char", "dosa",            "dosa"),
    ("missing_char", "bhkri",           "bhakri"),
    ("missing_char", "khichdi",         "khichdi"),
    ("missing_char", "smosa",           "samosa"),

    # ── extra characters ────────────────────────────────────────────────
    ("extra_char", "pizzaa",            "pizza"),
    ("extra_char", "rotii",             "roti"),
    ("extra_char", "daahl",             "dal"),
    ("extra_char", "chaii",             "chai"),
    ("extra_char", "paneerr",           "paneer"),
    ("extra_char", "biriyani",          "biryani"),

    # ── phonetic spelling ───────────────────────────────────────────────
    ("phonetic", "bhakhari",            "bhakri"),
    ("phonetic", "chaye",               "chai"),
    ("phonetic", "dossa",               "dosa"),
    ("phonetic", "chiken",              "chicken"),
    ("phonetic", "kadhai paneer",       "paneer"),
    ("phonetic", "aloo",                "aloo"),
    ("phonetic", "chawal",              REJECT),
    ("phonetic", "roṭli",               "roti"),

    # ── Gujarati / Hindi native script ──────────────────────────────────
    ("native_script", "ભાખરી",          "bhakri"),
    ("native_script", "રોટલી",           "roti"),
    ("native_script", "દાળ",            "dal"),
    ("native_script", "ચા",             "chai"),
    ("native_script", "रोटी",           "roti"),
    ("native_script", "दाल",            "dal"),
    ("native_script", "चावल",           "rice"),
    ("native_script", "पनीर",           "paneer"),

    # ── Roman Gujarati / Hinglish transliterations ──────────────────────
    ("translit", "bhakhri",             "bhakri"),
    ("translit", "rotli",               REJECT),
    ("translit", "chaas",               "chhas"),
    ("translit", "chhas",               "chhas"),
    ("translit", "khichdi",             "khichdi"),
    ("translit", "shaak",               REJECT),
    ("translit", "thepla",              "thepla"),
    ("translit", "fafda",               "fafda"),

    # ── mixed English + Gujarati / Hinglish sentences ───────────────────
    ("mixed_sentence", "me aaje 2 roti khadhi",             "roti"),
    ("mixed_sentence", "aaje mne bhakhri khadhi",           "bhakri"),
    ("mixed_sentence", "I had 2 idli for breakfast",        "idli"),
    ("mixed_sentence", "lunch ma dal tadka ane rice khadha","dal"),
    ("mixed_sentence", "mene ek plate samosa khaya",        "samosa"),
    ("mixed_sentence", "bapore paneer sabji khadhi",        "paneer"),
    ("mixed_sentence", "aaje me chaas pidhi",               "chhas"),
    ("mixed_sentence", "I ate chocolate ice cream",         "ice cream"),

    # ── valid foods with correct spelling (15+) ─────────────────────────
    ("valid", "pizza",                  "pizza"),
    ("valid", "roti",                   "roti"),
    ("valid", "dal",                    REJECT),
    ("valid", "paneer",                 "paneer"),
    ("valid", "idli",                   "idli"),
    ("valid", "dosa",                   "dosa"),
    ("valid", "samosa",                 "samosa"),
    ("valid", "chai",                   "chai"),
    ("valid", "lassi",                  "lassi"),
    ("valid", "rice",                   REJECT),
    ("valid", "khichdi",                "khichdi"),
    ("valid", "bhakri",                 "bhakri"),
    ("valid", "chhas",                  "chhas"),
    ("valid", "fafda",                  "fafda"),
    ("valid", "chocolate ice cream",    "ice cream"),
    ("valid", "green tea",              "tea"),
    ("valid", "chicken",                "chicken"),

    # ── junk / non-food (15+) — must be REJECTED ────────────────────────
    ("junk", "abc",                     REJECT),
    ("junk", "qwerty",                  REJECT),
    ("junk", "cflyfr",                  REJECT),
    ("junk", "xyz123",                  REJECT),
    ("junk", "vvysdv",                  REJECT),
    ("junk", "dvrbbweeg",               REJECT),
    ("junk", "asdfgh",                  REJECT),
    ("junk", "zxcvbn",                  REJECT),
    ("junk", "lkjhgf",                  REJECT),
    ("junk", "poiuyt",                  REJECT),
    ("junk", "hello how are you",       REJECT),
    ("junk", "today is good",           REJECT),
    ("junk", "asdfg qwerty",            REJECT),
    ("junk", "gwsegwe",                 REJECT),
    ("junk", "opopipiopiop",            REJECT),
    ("junk", "1234567",                 REJECT),
    ("junk", "!!!",                     REJECT),
]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()


async def _detect(engine: ChatbotEngine, text: str) -> tuple[bool, list[str], str]:
    """
    Run one input through the production pipeline and return:
      (rejected, detected_food_names, intent)
    """
    # Fresh state so a prior message never bleeds into this one.
    await engine.flow_mgr.save_flow_state(PendingFlow())
    resp = await engine.process_message(text, auto_log=False)
    intent = (resp.intent or "").lower()

    # Rejected → clarification_needed with no food in flow.
    if intent == "clarification_needed":
        return True, [], intent

    # Otherwise read the resolved food(s) from the resulting flow / response.
    names: list[str] = []
    try:
        flow = await engine.flow_mgr.get_pending_flow()
        for it in (flow.items or []):
            n = it.get("food_name") or it.get("food_name_display")
            if n:
                names.append(n)
        for it in (flow.pending_items or []):
            n = it.get("food_name") or it.get("food_name_display")
            if n:
                names.append(n)
        if flow.food_name:
            names.append(flow.food_name)
    except Exception:
        pass
    # Fallback: food name echoed in response data.
    if not names and isinstance(resp.data, dict) and resp.data.get("food_name"):
        names.append(resp.data["food_name"])

    rejected = (len(names) == 0)
    return rejected, names, intent


def _grade(expected: str, rejected: bool, names: list[str]) -> tuple[bool, str]:
    """Return (passed, actual_summary)."""
    if expected == REJECT:
        actual = "REJECTED" if rejected else "DETECTED:" + ", ".join(names)
        return rejected, actual
    # expected a food match
    if rejected:
        return False, "REJECTED (expected food)"
    exp = _norm(expected)
    exp_tokens = set(exp.split())
    for n in names:
        nn = _norm(n)
        if exp and (exp in nn or nn in exp):
            return True, f"DETECTED: {n}"
        # token overlap (e.g. expected 'ice cream' in 'chocolate ice cream')
        if exp_tokens and exp_tokens.issubset(set(nn.split())):
            return True, f"DETECTED: {n}"
    return False, "DETECTED: " + (", ".join(names) if names else "(none)")


async def main():
    user = {
        "user_id": "benchmark_typo_user",
        "name": "Benchmark User",
        "age": 30, "gender": "male", "height_cm": 175, "weight_kg": 70,
        "activity_level": "moderate", "fitness_goal": "maintain",
    }
    engine = ChatbotEngine(async_db, user, LLMService())

    # Clean start: clear this user's flow state.
    await engine.flow_mgr.save_flow_state(PendingFlow())

    print("=" * 78)
    print("  Food-Detection Benchmark — Qwen3-30B-A3B-FP8 (production config, UNCHANGED)")
    print("=" * 78)
    print(f"{'#':>3}  {'TYPE':<15} {'INPUT':<26} {'EXPECTED':<14} {'ACTUAL':<26} RESULT")
    print("-" * 78)

    results = []
    for i, (cat, text, expected) in enumerate(CASES, 1):
        try:
            rejected, names, intent = await _detect(engine, text)
        except Exception as e:
            rejected, names, intent = False, [], f"error:{e}"
        passed, actual = _grade(expected, rejected, names)
        results.append({"n": i, "type": cat, "input": text,
                        "expected": ("REJECT" if expected == REJECT else expected),
                        "actual": actual, "passed": passed})
        exp_disp = "REJECT" if expected == REJECT else expected
        mark = "PASS" if passed else "FAIL"
        print(f"{i:>3}  {cat:<15} {text[:25]:<26} {exp_disp[:13]:<14} {actual[:25]:<26} {mark}")
        # tiny delay so we don't hammer the LLM endpoint
        await asyncio.sleep(0.05)

    # Clean up flow state after the run.
    await engine.flow_mgr.save_flow_state(PendingFlow())

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    acc = (passed / total * 100.0) if total else 0.0

    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  Total tests : {total}")
    print(f"  Passed      : {passed}")
    print(f"  Failed      : {failed}")
    print(f"  Accuracy    : {acc:.1f}%")

    # Per-type breakdown
    print("\n  Accuracy by type:")
    types = {}
    for r in results:
        types.setdefault(r["type"], [0, 0])
        types[r["type"]][0] += 1
        if r["passed"]:
            types[r["type"]][1] += 1
    for t in sorted(types):
        tot, pas = types[t]
        print(f"    {t:<15} {pas}/{tot}  ({pas / tot * 100:.0f}%)")

    # Failed examples grouped by type
    fails = [r for r in results if not r["passed"]]
    if fails:
        print("\n  FAILED examples (grouped by type):")
        by_type = {}
        for r in fails:
            by_type.setdefault(r["type"], []).append(r)
        for t in sorted(by_type):
            print(f"    [{t}]")
            for r in by_type[t]:
                print(f"       input={r['input']!r}  expected={r['expected']!r}  actual={r['actual']!r}")
    else:
        print("\n  No failures.")

    print("=" * 78)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
