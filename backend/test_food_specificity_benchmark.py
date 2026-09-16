"""
Food-specificity & clarification benchmark (100 NEW questions).

Goal
----
Verify how the production pipeline handles GENERIC vs SPECIFIC foods:

  • A SPECIFIC food ("bajra no rotlo", "paneer tikka", "gulab jamun") must be
    detected and logged normally.
  • A GENERIC food ("shak", "sabji", "dal", "curry", "rice", "soup", "fruit",
    "sweet", "snack", "juice", "vegetable", …) SHOULD trigger a clarification
    that asks the user for the specific dish — instead of silently logging a
    guessed/branded item.
  • When a sentence mixes one specific + one generic food, the specific item is
    detected and the generic item is clarified.
  • When EVERYTHING is specific, NO clarification should happen.

The benchmark drives the exact production entry point
`ChatbotEngine.process_message(...)`. It does NOT modify production code, the
model/config, or the DB.

Outcome model
-------------
For each message we read the resulting ChatResponse + flow and compute:
  • detected_foods      — canonical food names that entered the food flow
  • clarification_raised — True if the pipeline asked the user to identify /
    specify a food (intent == clarification_needed) OR left a generic item
    unresolved while proceeding.

Grading (against the DESIRED specificity behaviour)
---------------------------------------------------
  • expected_clarify=True  → PASS if clarification_raised is True.
  • expected_clarify=False → PASS if clarification_raised is False AND every
    expected specific food was detected.

Usage:  python test_food_specificity_benchmark.py   (needs MongoDB; live Qwen)
"""
import asyncio
import io
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

from database import async_db
from chatbot_engine import ChatbotEngine
from llm_service import LLMService
from conversation import PendingFlow

# ---------------------------------------------------------------------------
# Cases: (category, message, expected_specific_foods, expected_clarify)
#   expected_specific_foods : list of specific food fragments that MUST be
#                             detected (token/substring match, case-insensitive)
#   expected_clarify        : True if a generic item should trigger clarification
# ---------------------------------------------------------------------------
CASES: list[tuple[str, str, list[str], bool]] = [

    # ===================================================================
    #  A. SPECIFIC + GENERIC in one sentence  → detect specific, clarify generic
    # ===================================================================
    ("mix_gu",   "me aaje bajra no rotlo ane shak khadhu",            ["bajra", "rotlo"], True),
    ("mix_gu",   "aaje me thepla ane sabji khadhi",                   ["thepla"],         True),
    ("mix_gu",   "me sanje dhokla ane chutney lidha",                 ["dhokla"],         True),
    ("mix_gu",   "aaje bajri no rotlo ne shaak jamyu",                ["bajri", "rotlo"], True),
    ("mix_gu",   "me methi na thepla ane dal khadha",                 ["thepla"],         True),
    ("mix_gu",   "aaje khichdi ane kadhi khadhi",                     ["khichdi"],        True),
    ("mix_gu",   "me handvo ane sweet khadhu",                        ["handvo"],         True),
    ("mix_gu",   "aaje undhiyu ane rotli jamya",                      ["undhiyu"],        True),
    ("mix_gu",   "me dhosa ane sambhar lidhu",                        ["dosa"],           True),
    ("mix_gu",   "aaje fafda ane curry khadha",                       ["fafda"],          True),

    ("mix_hi",   "maine samosa aur sabzi khaya",                      ["samosa"],         True),
    ("mix_hi",   "aaj maine paratha aur dal khayi",                   ["paratha"],        True),
    ("mix_hi",   "maine idli aur curry khayi",                        ["idli"],           True),
    ("mix_hi",   "aaj poha aur juice liya",                           ["poha"],           True),
    ("mix_hi",   "maine kachori aur sweet khaya",                     ["kachori"],        True),
    ("mix_hi",   "aaj chole aur rice khaya",                          ["chole"],          True),
    ("mix_hi",   "maine dhokla aur soup liya",                        ["dhokla"],         True),
    ("mix_hi",   "aaj tikki aur fruit khaya",                         ["tikki"],          True),

    ("mix_hing", "I ate 2 bhakri and some sabji",                     ["bhakri"],         True),
    ("mix_hing", "had gulab jamun and a bit of curry",                ["gulab jamun"],    True),
    ("mix_hing", "ate paneer tikka with some sabzi",                  ["paneer tikka"],   True),
    ("mix_hing", "had a masala dosa and some soup",                   ["dosa"],           True),
    ("mix_hing", "ate 3 momos and a fruit",                           ["momos"],          True),
    ("mix_hing", "had jalebi and some sweet after lunch",             ["jalebi"],         True),
    ("mix_hing", "ate vada pav and a juice",                          ["vada pav"],       True),
    ("mix_hing", "had spring rolls with some curry",                  ["spring roll"],    True),

    ("mix_en",   "I ate grilled chicken and some vegetable",          ["chicken"],        True),
    ("mix_en",   "had a cheese sandwich and some soup",               ["sandwich"],       True),
    ("mix_en",   "ate scrambled eggs and a fruit",                    ["egg"],            True),
    ("mix_en",   "had a chicken burger and some snack",               ["burger"],         True),
    ("mix_en",   "ate peanut butter toast and a sweet",               ["toast"],          True),

    # ===================================================================
    #  B. GENERIC-only  → should clarify (no specific dish named)
    # ===================================================================
    ("gen_only", "aaje me shak khadhu",                               [], True),
    ("gen_only", "me sabji khadhi",                                   [], True),
    ("gen_only", "maine sabzi khayi",                                 [], True),
    ("gen_only", "I ate some curry",                                  [], True),
    ("gen_only", "had a fruit",                                       [], True),
    ("gen_only", "ate a sweet",                                       [], True),
    ("gen_only", "had some snack",                                    [], True),
    ("gen_only", "drank a juice",                                     [], True),
    ("gen_only", "ate some vegetable",                                [], True),
    ("gen_only", "had a soup",                                        [], True),
    ("gen_only", "me shaak lidhu",                                    [], True),
    ("gen_only", "aaj sabzi khayi",                                   [], True),
    ("gen_only", "kuch sabji khaya",                                  [], True),
    ("gen_only", "had a bowl of curry",                               [], True),
    ("gen_only", "ate a plate of sabji",                              [], True),
    ("gen_only", "me thodu shaak khadhu",                             [], True),
    ("gen_only", "aaj koi fruit khaya",                               [], True),
    ("gen_only", "had some mithai",                                   [], True),
    ("gen_only", "ate a dessert",                                     [], True),
    ("gen_only", "had a drink",                                       [], True),

    # ── same intent, many wordings ("gol gol" variations) ─────────────
    ("gen_vary", "kai shak banavu chhu te khadhu",                    [], True),
    ("gen_vary", "aaje sabji jevu kaik khadhu",                       [], True),
    ("gen_vary", "just had some curry type thing",                    [], True),
    ("gen_vary", "ate that vegetable dish",                           [], True),
    ("gen_vary", "had a sweet something",                             [], True),
    ("gen_vary", "some fruit types khaya",                            [], True),
    ("gen_vary", "koi sabzi thi wo khayi",                            [], True),
    ("gen_vary", "aaje sanje ek shaak hatu te lidhu",                 [], True),
    ("gen_vary", "had my usual curry",                                [], True),
    ("gen_vary", "ate a random sweet",                                [], True),
    ("gen_vary", "some juice pi li",                                  [], True),
    ("gen_vary", "thodi si sabzi thi",                                [], True),
    ("gen_vary", "aaj koi mithai type khaya",                         [], True),
    ("gen_vary", "had a light soup",                                  [], True),
    ("gen_vary", "ate a quick snack",                                 [], True),

    # ===================================================================
    #  C. GENERIC + quantity / meal context  → still clarify the dish
    # ===================================================================
    ("gen_qty",  "had 2 bowls of curry for lunch",                    [], True),
    ("gen_qty",  "ate 1 plate of sabji at dinner",                    [], True),
    ("gen_qty",  "me 1 vatki shak khadhu",                            [], True),
    ("gen_qty",  "maine ek katori sabzi khayi",                       [], True),
    ("gen_qty",  "had 2 fruits in the morning",                       [], True),
    ("gen_qty",  "ate 3 sweets after dinner",                         [], True),
    ("gen_qty",  "drank 1 glass of juice",                            [], True),
    ("gen_qty",  "had a bowl of soup at night",                       [], True),
    ("gen_qty",  "aaje 2 vatki dal jevu khadhu",                      [], True),
    ("gen_qty",  "ate some snacks in the evening",                    [], True),

    # ===================================================================
    #  D. ALL-SPECIFIC  → NO clarification, detect the specific food(s)
    # ===================================================================
    ("spec_single", "aaje me bajra no rotlo khadho",                  ["bajra", "rotlo"], False),
    ("spec_single", "had gulab jamun",                                ["gulab jamun"],    False),
    ("spec_single", "ate paneer tikka",                               ["paneer tikka"],   False),
    ("spec_single", "maine masala dosa khaya",                        ["dosa"],           False),
    ("spec_single", "had a plate of pav bhaji",                       ["pav bhaji"],      False),
    ("spec_single", "ate chole bhature",                              ["chole"],          False),
    ("spec_single", "me thepla khadha",                               ["thepla"],         False),
    ("spec_single", "had aloo paratha",                               ["paratha"],        False),
    ("spec_single", "ate rajma chawal",                               ["rajma"],          False),
    ("spec_single", "maine dhokla khaya",                             ["dhokla"],         False),
    ("spec_single", "had methi thepla",                               ["thepla"],         False),
    ("spec_single", "ate palak paneer",                               ["paneer"],         False),
    ("spec_single", "had a veg biryani",                              ["biryani"],        False),
    ("spec_single", "me handvo khadho",                               ["handvo"],         False),
    ("spec_single", "ate khaman dhokla",                              ["khaman"],         False),

    ("spec_multi", "aaje me bajra no rotlo ane bhindi nu shaak khadhu", ["bajra", "bhindi"], False),
    ("spec_multi", "had 2 aloo paratha and dahi",                     ["paratha"],        False),
    ("spec_multi", "ate masala dosa and filter coffee",              ["dosa"],           False),
    ("spec_multi", "maine chole bhature aur lassi li",                ["chole"],          False),
    ("spec_multi", "had paneer tikka and butter naan",               ["paneer tikka", "naan"], False),
    ("spec_multi", "me thepla ane keri no chhundo khadha",            ["thepla"],         False),
    ("spec_multi", "ate idli sambar and vada",                       ["idli"],           False),
    ("spec_multi", "had rajma chawal and papad",                     ["rajma"],          False),
    ("spec_multi", "me dhokla ane khakhra khadha",                    ["dhokla", "khakhra"], False),
    ("spec_multi", "ate gajar halwa and kaju katli",                 ["halwa"],          False),

    # ── all-specific with heavy wording variation ────────────────────
    ("spec_vary", "so today morning I finally had my aloo paratha",   ["paratha"],        False),
    ("spec_vary", "guess what, I ate two whole gulab jamun",          ["gulab jamun"],    False),
    ("spec_vary", "aaje to me full plate biryani zaap gayo",          ["biryani"],        False),
    ("spec_vary", "yaar maine aaj kaju katli kha li",                 ["kaju katli"],     False),
    ("spec_vary", "had a nice big masala dosa for brunch",            ["dosa"],           False),
    ("spec_vary", "me sachu kau to be thepla khadha",                 ["thepla"],         False),
    ("spec_vary", "ended up eating a vada pav on the way",            ["vada pav"],       False),
    ("spec_vary", "khaya toh maine bas ek samosa hi",                 ["samosa"],         False),
    ("spec_vary", "treated myself to some kaju katli today",          ["kaju katli"],     False),
    ("spec_vary", "had my regular 2 idli this morning",               ["idli"],           False),
]


def _clarification_raised(resp, flow, detected: list[str]) -> bool:
    """A clarification was raised if the pipeline asked the user to identify /
    specify a food, or left a generic item unresolved while proceeding."""
    intent = (resp.intent or "").lower()
    if intent in ("clarification_needed", "unknown"):
        return True
    try:
        if flow is not None and getattr(flow, "unresolved_items", None):
            return True
    except Exception:
        pass
    # A "couldn't identify / which food" style message also counts.
    msg = (resp.message or "").lower()
    ask_markers = ("couldn't identify", "could not identify", "which food",
                   "kayu food", "kyu food", "be specific", "specific food",
                   "what did you", "which dish", "kayi", "konsa", "kaunsa")
    if intent not in ("log_food",) and any(a in msg for a in ask_markers):
        return True
    return False


async def _run(engine: ChatbotEngine, text: str):
    await engine.flow_mgr.save_flow_state(PendingFlow())
    resp = await engine.process_message(text, auto_log=False)
    try:
        flow = await engine.flow_mgr.get_pending_flow()
    except Exception:
        flow = None

    detected: list[str] = []
    try:
        if flow is not None:
            for it in (flow.items or []):
                n = it.get("food_name") or it.get("food_name_display")
                if n:
                    detected.append(n)
    except Exception:
        pass
    clarify = _clarification_raised(resp, flow, detected)
    return detected, clarify, (resp.intent or "")


def _norm(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()


def _grade(expected_foods, expected_clarify, detected, clarify):
    if expected_clarify:
        return clarify
    # expected NO clarification → must NOT clarify AND detect the specific foods
    if clarify:
        return False
    det_join = " | ".join(_norm(d) for d in detected)
    for want in expected_foods:
        w = _norm(want)
        if w and w not in det_join:
            return False
    return True


async def main():
    user = {
        "user_id": "benchmark_specificity_user",
        "name": "Specificity Benchmark User",
        "age": 30, "gender": "male", "height_cm": 175, "weight_kg": 70,
        "activity_level": "moderate", "fitness_goal": "maintain",
    }
    engine = ChatbotEngine(async_db, user, LLMService())
    await engine.flow_mgr.save_flow_state(PendingFlow())

    print("=" * 108)
    print("  Food-Specificity & Clarification Benchmark — production config (UNCHANGED)")
    print(f"  Total questions: {len(CASES)}")
    print("=" * 108)
    print(f"{'#':>3}  {'CATEGORY':<12} {'MESSAGE':<46} {'CLARIFY?':<9} {'GOT':<9} RESULT")
    print("-" * 108)

    results = []
    for i, (cat, text, exp_foods, exp_clar) in enumerate(CASES, 1):
        try:
            detected, clarify, intent = await _run(engine, text)
        except Exception as e:
            detected, clarify, intent = [], False, f"error:{e}"
        passed = _grade(exp_foods, exp_clar, detected, clarify)
        results.append({
            "n": i, "cat": cat, "input": text, "exp_foods": exp_foods,
            "exp_clarify": exp_clar, "detected": detected, "clarify": clarify,
            "intent": intent, "passed": passed,
        })
        mark = "PASS" if passed else "FAIL"
        print(f"{i:>3}  {cat:<12} {text[:45]:<46} {str(exp_clar):<9} {str(clarify):<9} {mark}")
        await asyncio.sleep(0.02)

    await engine.flow_mgr.save_flow_state(PendingFlow())

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    acc = (passed / total * 100.0) if total else 0.0

    print("\n" + "=" * 108)
    print("  SUMMARY")
    print("=" * 108)
    print(f"  Total tests : {total}")
    print(f"  Passed      : {passed}")
    print(f"  Failed      : {failed}")
    print(f"  Accuracy    : {acc:.1f}%")

    print("\n  Accuracy by category:")
    cats: dict[str, list[int]] = {}
    for r in results:
        cats.setdefault(r["cat"], [0, 0])
        cats[r["cat"]][0] += 1
        if r["passed"]:
            cats[r["cat"]][1] += 1
    for c in sorted(cats):
        tot, pas = cats[c]
        print(f"    {c:<14} {pas}/{tot}  ({pas / tot * 100:.0f}%)")

    fails = [r for r in results if not r["passed"]]
    if fails:
        print("\n  FAILED cases (grouped by category):")
        by_cat: dict[str, list[dict]] = {}
        for r in fails:
            by_cat.setdefault(r["cat"], []).append(r)
        for c in sorted(by_cat):
            print(f"    [{c}]")
            for r in by_cat[c]:
                print(f"       #{r['n']} input={r['input']!r}")
                print(f"           expected_clarify={r['exp_clarify']}  expected_foods={r['exp_foods']}")
                print(f"           got clarify={r['clarify']}  detected={r['detected']}  intent={r['intent']!r}")
    else:
        print("\n  No failures.")

    print("=" * 108)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
