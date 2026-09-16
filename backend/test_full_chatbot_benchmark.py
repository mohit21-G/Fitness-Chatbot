"""
Full-pipeline benchmark for the ENTIRE Fitness AI chatbot (500+ NEW questions).

Scope
-----
Exercises the COMPLETE production pipeline through the exact production entry
point `ChatbotEngine.process_message(...)` — the same call the chatbot API
makes — using the production Qwen intent parser, DB, transliteration/fuzzy
search, external sources, deterministic routing, and reject gates. It does NOT
modify production code, prompts, the model, the DB, or routing.

Coverage (all questions are NEW — none repeat/paraphrase the strings in
test_llm_typo_benchmark.py or test_daily_log_routing_benchmark.py):
  • food_log        — logging a food (with/without quantity, variants, multi-food)
  • exercise_log    — logging an exercise (reps/sets/duration/distance)
  • calorie_macro   — "how many calories/protein in <food>" knowledge lookups
  • daily_log_food  — reading logged food (summary / a meal), MUST NOT log food
  • exercise_read   — reading logged exercise
  • multilingual    — gu / hi / roman-gu / hinglish / english food logging
  • typo            — spelling/phonetic variants that must still resolve to food
  • multi_food      — several foods in one sentence
  • junk            — random / non-food / non-exercise → must be REJECTED
  • ambiguous       — bare quantities / vague inputs → clarify, never invent food

Outcome model (behavioural, production-faithful)
------------------------------------------------
Each message is classified into the ROUTE the pipeline actually took:
  food_log | exercise_log | calorie | daily_log | exercise_read | reject | other
A case PASSES when the observed route matches the expected route. For daily_log
we additionally require that NO food/quantity/variant detection was triggered.

Usage:  python test_full_chatbot_benchmark.py     (needs MongoDB; uses live Qwen)
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

# Route sentinels (expected values)
FOOD   = "food_log"
EXLOG  = "exercise_log"
CAL    = "calorie"
DLOG   = "daily_log"
EXREAD = "exercise_read"
REJECT = "reject"

# ---------------------------------------------------------------------------
# CASES: (category, message, expected_route)
# All strings below are NEW (not present in the two prior benchmarks).
# ---------------------------------------------------------------------------
CASES: list[tuple[str, str, str]] = []


def add(cat, route, msgs):
    for m in msgs:
        CASES.append((cat, m, route))


# ── FOOD LOGGING — English, with quantity (NEW foods/phrasings) ─────────────
add("food_qty_en", FOOD, [
    "I ate 3 aloo paratha this morning",
    "logged 2 bowls of rajma for lunch",
    "had 4 pieces of dhokla at tea",
    "consumed one plate of chole bhature",
    "grabbed 2 vada pav on the way home",
    "finished a bowl of curd rice",
    "ate 5 gulab jamun after dinner",
    "had 2 slices of whole wheat bread",
    "drank 1 glass of sweet lassi",
    "ate half a plate of pav bhaji",
    "had 3 spoons of peanut butter",
    "polished off 2 masala dosa",
    "ate 1 katori of kheer",
    "had two boiled eggs for breakfast",
    "ate a cup of muesli with milk",
    "had 6 almonds as a snack",
    "ate 2 tablespoons of hummus",
    "finished 1 bowl of oats porridge",
    "had 3 jalebi",
    "ate a large bowl of fruit salad",
])

# ── FOOD LOGGING — English, no quantity ─────────────────────────────────────
add("food_noqty_en", FOOD, [
    "I had rajma chawal",
    "ate some aloo gobi",
    "had palak paneer",
    "munched on roasted chana",
    "ate baingan bharta",
    "had a plate of poha",
    "drank buttermilk",
    "ate methi thepla",
    "had chana masala",
    "ate a banana",
    "had grilled chicken breast",
    "ate tofu stir fry",
    "had mushroom soup",
    "ate a peanut chikki",
    "had sabudana khichdi",
])

# ── FOOD LOGGING — multilingual (gu / hi / roman-gu / hinglish) ─────────────
add("food_multiling", FOOD, [
    "aaje me batata poha khadha",
    "me sवारे upma khadho",       # devanagari mixed
    "maine subah aloo puri khayi",
    "aaj raat dal dhokli khadhi",
    "me bapore kadhi khichdi khadhi",
    "hu ae undhiyu khadhu",
    "aaj maine matar paneer khaya",
    "me thodu shrikhand khadhu",
    "raatre bhindi masala khadhi",
    "aaj nashte me besan chilla khaya",
    "mne sanje handvo bhavyo etle khadho",
    "aaj lunch me curd aur roti li",
    "me full plate biryani khadhi",
    "aaje savare methi na gota khadha",
    "maine ek katori drumstick sambar piya",
    "hu ae ghar nu shaak rotli khadhu",
    "aaj dopahar me kadhi khichdi khaya",
    "me aaje makai no chevdo khadho",
    "raat ne dahi vada khadha",
    "aaje mej par lapsi hati te khadhi",
])

# ── FOOD LOGGING — typos / phonetic (NEW spellings) ─────────────────────────
add("food_typo", FOOD, [
    "ate palek paneer",
    "had rajmah chawel",
    "chole bhatoore for lunch",
    "ate aaloo gobhi",
    "had veg biriani",
    "ate paav bhajji",
    "drank buttermlik",
    "ate mysore paak",
    "had kaju katlii",
    "ate rasgula",
    "had chiken curry",
    "ate mushrom masala",
    "had veg pulaoo",
    "ate bhindi fry sabjii",
    "had gajar halwaa",
])

# ── MULTI-FOOD (several items in one message) ───────────────────────────────
add("multi_food", FOOD, [
    "I had 2 roti, dal and a bowl of rice",
    "ate idli sambar and a filter coffee",
    "had poha, a boiled egg and orange juice",
    "lunch was rajma, rice and salad",
    "aaje me thepla, chhundo ane chaas lidha",
    "breakfast: 2 paratha, curd and pickle",
    "had chole with bhature and a lassi",
    "ate dosa, vada and coconut chutney",
    "dinner was khichdi, kadhi and papad",
    "me sanje bhel, pani puri ane masala chaas khadha",
    "had a burger, fries and a coke",
    "ate paneer tikka and butter naan",
    "morning I had oats, banana and almonds",
    "aaj maine samosa, kachori aur jalebi khayi",
    "had grilled fish, rice and steamed veggies",
])

# ── EXERCISE LOGGING (NEW exercises/phrasings) ──────────────────────────────
add("exercise_log", EXLOG, [
    "did 45 minutes of skipping",
    "ran 5 km this evening",
    "completed 20 burpees",
    "did 3 sets of 12 bench press",
    "went swimming for 30 minutes",
    "did 100 jumping jacks",
    "cycled for 40 minutes",
    "did 25 lunges on each leg",
    "held a plank for 2 minutes",
    "did 50 mountain climbers",
    "walked 8000 steps today",
    "did 15 minutes of surya namaskar",
    "aaje me 30 minute daudyo",
    "maine 40 pushups lagaye",
    "did 4 rounds of HIIT",
    "rowed for 20 minutes",
    "did 30 squats",
    "climbed stairs for 15 minutes",
    "did an hour of power yoga",
    "sprinted 10 rounds of 100m",
    "aaj 20 minute cycling kari",
    "did 3 sets of deadlifts",
    "skipped rope for 500 skips",
    "did 60 crunches",
    "went for a 3 km jog",
])

# ── CALORIE / MACRO knowledge lookups (get_calories) ────────────────────────
add("calorie_macro", CAL, [
    "how many calories in a banana",
    "calories in 100g paneer",
    "how much protein in one egg",
    "kitni calories hoti hai ek roti me",
    "how many calories does a samosa have",
    "protein content of moong dal",
    "calories in a cup of milk",
    "how much fat in 100g cheese",
    "calories in one apple",
    "how many calories in 2 idli",
    "carbs in a bowl of rice",
    "how many calories in a mango",
    "calories in a tablespoon of ghee",
    "protein in 100 grams of chicken",
    "how many calories in dosa",
])

# ── DAILY LOG reads — food (must NOT trigger food detection) ────────────────
add("daily_log_food", DLOG, [
    "what have I logged so far today",
    "show me everything I ate this morning",
    "recap my meals for today",
    "give me the rundown of today's food",
    "list out my dinner items",
    "aaj ka khana ginn ke batao",
    "what's on my plate log for lunch",
    "pull up my breakfast entries",
    "summarise my eating today",
    "aaje sudhi me su su lidhu batavo",
    "how much have I eaten so far",
    "run down my snacks list",
    "what did my lunch consist of",
    "read back my breakfast",
    "tell me my food tally for today",
])

# ── EXERCISE reads ──────────────────────────────────────────────────────────
add("exercise_read", EXREAD, [
    "what workouts did I log today",
    "show my exercise for today",
    "how much did I burn today",
    "list my workouts",
    "aaj maine kaunsi exercise ki",
    "recap my training today",
    "what exercises have I done today",
    "show today's workout log",
    "aaje me su exercise kari batavo",
    "how many calories did I burn working out",
])

# ── JUNK / non-food / non-exercise → REJECT ─────────────────────────────────
add("junk", REJECT, [
    "kjhgfd",
    "wpoeiru",
    "zzxxccvv",
    "the weather is nice today",
    "my phone battery is low",
    "can you tell me a joke",
    "what time is the movie",
    "asdkjhqwe",
    "mnbvcxz",
    "plkoijuhyg",
    "tell me about the stock market",
    "who won the cricket match",
    "qazwsxedc",
    "random gibberish text here",
    "fjdksla",
    "how do I reset my password",
    "book me a cab",
    "trrtrtrt",
    "vbnmghjk",
    "what's the capital of france",
])

# ── AMBIGUOUS / edge (vague → clarify, never invent a food) ─────────────────
add("ambiguous", REJECT, [
    "i ate something",
    "had a bit of that thing",
    "some food",
    "khadhu kaik",
    "kuch khaya tha",
    "ate a lot",
    "had my usual",
    "the regular stuff",
])

# ===========================================================================
#  WAVE 2 — additional NEW questions to exceed 500 total (all distinct)
# ===========================================================================

# ── FOOD LOGGING — more English with quantity ───────────────────────────────
add("food_qty_en", FOOD, [
    "ate 2 stuffed capsicum",
    "had 3 tablespoons of dalia",
    "consumed 250 ml of coconut water",
    "ate 4 dates after workout",
    "had a bowl of tomato soup",
    "ate 2 egg omelette",
    "had 1 scoop of whey protein",
    "ate 3 khaman pieces",
    "had two slices of pineapple",
    "ate a handful of walnuts",
    "had 1 plate of veg fried rice",
    "ate 2 spring rolls",
    "had a cup of black coffee",
    "ate 5 crackers with cheese",
    "had one bowl of moong dal khichdi",
    "ate 2 scoops of vanilla ice cream",
    "had 3 pieces of dark chocolate",
    "ate a plate of steamed momos",
    "had 1 glass of orange juice",
    "ate 6 grapes",
    "had 2 whole wheat wraps",
    "ate a bowl of mixed sprouts",
    "had 1 idiyappam with stew",
    "ate 4 pieces of paneer tikka",
    "had a bowl of ragi porridge",
])

# ── FOOD LOGGING — more no quantity ─────────────────────────────────────────
add("food_noqty_en", FOOD, [
    "had lauki sabzi",
    "ate karela fry",
    "had tinda masala",
    "ate egg bhurji",
    "had fish curry",
    "ate mutton keema",
    "had veg manchurian",
    "ate schezwan noodles",
    "had corn chaat",
    "ate sprout salad",
    "had beetroot poriyal",
    "ate cabbage thoran",
    "had drumstick sambar",
    "ate paneer bhurji",
    "had aloo methi",
    "ate stuffed brinjal",
    "had raw mango pickle",
    "ate coconut barfi",
    "had sesame ladoo",
    "ate dry fruit halwa",
])

# ── FOOD LOGGING — more multilingual ────────────────────────────────────────
add("food_multiling", FOOD, [
    "aaje me dudhi na muthiya khadha",
    "maine subah aloo ke parathe khaye",
    "aaj raat me tinda ki sabzi khayi",
    "hu ae ringan no olo khadho",
    "me bapore surti locho khadho",
    "aaj maine bhindi ki sabzi khayi",
    "raatre khichu khadhu",
    "me thodu doodhpak lidhu",
    "aaj nashte me fafda jalebi khadha",
    "maine ek katori sevaiyan khayi",
    "hu ae methi thepla ne athanu khadhu",
    "aaj dopahar me bhatura chole liye",
    "me aaje khaman dhokla khadha",
    "raat ne veg pulav khadho",
    "aaje savare gathiya khadha",
    "maine subah daliya khaya",
    "aaj me ghar nu dal bhaat khadhu",
    "me sanje bataka poha ne chaas lidha",
    "hu ae mag ni dal khadhi",
    "aaje me tuvar ni dal ne bhaat khadho",
])

# ── FOOD LOGGING — more typos ───────────────────────────────────────────────
add("food_typo", FOOD, [
    "ate mater paner",
    "had veg kofta",
    "ate dhabha dal",
    "had chiken biriyani",
    "ate cheeze sandwich",
    "had masala doosa",
    "ate idlee sambhar",
    "had gobhi paratha",
    "ate rasmalaii",
    "had kaaju curry",
    "ate methee thepla",
    "had baigan bhartha",
    "ate palek soup",
    "had veg cutlet",
    "ate sooji halwa",
    "had aamras puri",
    "ate mix veg sabjii",
    "had lemon riceh",
    "ate curd riceh",
    "had tomato ricee",
])

# ── MULTI-FOOD — more ───────────────────────────────────────────────────────
add("multi_food", FOOD, [
    "had upma, coconut chutney and coffee",
    "ate 2 chapati with bhindi and dal",
    "lunch was curd rice, papad and pickle",
    "aaje me rotli, shaak ane dahi lidha",
    "had a sandwich, an apple and green tea",
    "ate pongal, vada and sambar",
    "dinner was 3 phulka, mixed veg and salad",
    "me sanje sev usal ane chaas lidha",
    "had scrambled eggs, toast and juice",
    "ate paratha, curd, and a boiled egg",
    "breakfast was cornflakes, milk and a banana",
    "had chicken curry, rice and raita",
    "ate dhokla, khandvi and masala tea",
    "aaj maine daal, chawal aur bhindi khayi",
    "had pasta, garlic bread and a salad",
])

# ── EXERCISE LOGGING — more ─────────────────────────────────────────────────
add("exercise_log", EXLOG, [
    "did 30 minutes of zumba",
    "completed 15 pull ups",
    "ran on the treadmill for 25 minutes",
    "did 3 sets of shoulder press",
    "went hiking for 2 hours",
    "did 40 russian twists",
    "cycled 12 km today",
    "did 20 tricep dips",
    "held a wall sit for 90 seconds",
    "did 5 sets of 10 bicep curls",
    "walked briskly for 35 minutes",
    "did 12 sun salutations",
    "aaje me 1 ghanta chalyo",
    "maine 25 squats kiye",
    "did 6 rounds of tabata",
    "did 200 skips with a rope",
    "did 3 sets of leg press",
    "played badminton for 45 minutes",
    "did 50 bicycle crunches",
    "jogged 4 km in the park",
    "aaj 15 minute plank workout kiya",
    "did 4 sets of lat pulldowns",
    "did 80 high knees",
    "did 10 pistol squats each leg",
    "went for a 6 km evening walk",
])

# ── CALORIE / MACRO — more ──────────────────────────────────────────────────
add("calorie_macro", CAL, [
    "how many calories in an orange",
    "calories in a bowl of poha",
    "how much protein in tofu",
    "calories in one chapati with ghee",
    "how many calories in a plate of biryani",
    "carbs in a slice of bread",
    "how much fat in an avocado",
    "calories in a cup of green tea",
    "protein in 100g soya chunks",
    "how many calories in a laddu",
    "calories in a spoon of honey",
    "how much sugar in a can of cola",
    "calories in a boiled potato",
    "protein in a glass of milk",
    "how many calories in a bowl of dal",
    "fiber in a bowl of oats",
    "calories in one jalebi",
    "how much protein in chickpeas",
    "calories in a handful of peanuts",
    "how many calories in a plate of upma",
])

# ── DAILY LOG reads — more ──────────────────────────────────────────────────
add("daily_log_food", DLOG, [
    "walk me through what I ate today",
    "give me my meal history for today",
    "what's logged under dinner",
    "read out my lunch log",
    "aaj ka khana wala log kholo",
    "show me my food diary today",
    "what all is in my breakfast slot",
    "recap the snacks I logged",
    "aaje nu jaman batavo",
    "list the items I ate at dinner",
    "how many things did I log today",
    "show my consumed food today",
    "what do I have logged for the morning meal",
    "pull my evening snack entries",
    "give me a food breakdown for today",
])

# ── EXERCISE reads — more ───────────────────────────────────────────────────
add("exercise_read", EXREAD, [
    "read out my workout log",
    "what physical activity did I log today",
    "show my training summary",
    "how many workouts today",
    "aaj ke workouts dikhao",
    "list the exercises I logged",
    "what did I do in the gym today",
    "show my activity for today",
    "recap my exercises today",
    "how much did I work out today",
])

# ── JUNK — more ─────────────────────────────────────────────────────────────
add("junk", REJECT, [
    "yuiophjk",
    "lorem ipsum dolor",
    "my car won't start",
    "what's your favourite colour",
    "ddffgghh",
    "send an email to my boss",
    "aaaaaa bbbbb",
    "the meeting is at 5pm",
    "translate this to spanish",
    "wqertyu",
    "play some music",
    "how tall is mount everest",
    "zzzzzzz",
    "set an alarm for 6am",
    "poqwie",
    "is it going to rain tomorrow",
    "kkllmmnn",
    "what's 25 times 4",
    "hjklasdf",
    "tell me a bedtime story",
    "cvbnmasd",
    "who is the president",
    "rtyuiop",
    "open the calculator app",
    "mmnnbbvv",
])

# ── AMBIGUOUS — more ────────────────────────────────────────────────────────
add("ambiguous", REJECT, [
    "ate the leftovers",
    "had a little snack",
    "some drink",
    "khaidhu badhu",
    "thoda sa khaya",
    "had whatever was there",
    "ate my meal",
    "grabbed a bite",
    "kaik peedhu",
    "had that dish again",
    "some homemade thing",
    "ate the special",
])


def _flow_food_detection(resp, flow) -> bool:
    intent = (resp.intent or "").lower()
    if intent == "log_food":
        return True
    if "flow:awaiting_variant" in intent or "flow:awaiting_quantity" in intent or "multi_choice" in intent:
        return True
    if getattr(resp, "needs_confirmation", False) and (resp.pending_action or {}).get("type") == "log_food":
        return True
    try:
        if flow is not None:
            if flow.items or flow.pending_items:
                return True
            if getattr(flow, "food_name", None) or getattr(flow, "food_query", None):
                return True
    except Exception:
        pass
    return False


def _is_exercise_route(resp) -> bool:
    intent = (resp.intent or "").lower()
    if intent == "log_exercise":
        return True
    if "awaiting_exercise_amount" in intent:
        return True
    if getattr(resp, "needs_confirmation", False) and (resp.pending_action or {}).get("type") == "log_exercise":
        return True
    return False


async def _route_of(engine: ChatbotEngine, text: str) -> tuple[str, str]:
    """Return (route, intent) for one message run through production."""
    await engine.flow_mgr.save_flow_state(PendingFlow())
    resp = await engine.process_message(text, auto_log=False)
    intent = (resp.intent or "").lower()

    try:
        flow = await engine.flow_mgr.get_pending_flow()
    except Exception:
        flow = None

    food = _flow_food_detection(resp, flow)
    if _is_exercise_route(resp):
        return EXLOG, intent
    if food:
        return FOOD, intent
    if intent == "get_calories":
        # get_calories with a resolved food = a real knowledge lookup.
        return CAL, intent
    if intent == "query_exercise":
        return EXREAD, intent
    if intent in ("get_summary", "query_meal"):
        return DLOG, intent
    if intent in ("clarification_needed", "unknown", "cancelled"):
        return REJECT, intent
    if intent == "greeting":
        return "greeting", intent
    if intent == "get_profile":
        return "profile", intent
    return "other", intent


def _grade(expected: str, route: str) -> bool:
    return route == expected


async def main():
    user = {
        "user_id": "benchmark_full_user",
        "name": "Full Benchmark User",
        "age": 30, "gender": "male", "height_cm": 175, "weight_kg": 70,
        "activity_level": "moderate", "fitness_goal": "maintain",
    }
    engine = ChatbotEngine(async_db, user, LLMService())
    await engine.flow_mgr.save_flow_state(PendingFlow())

    print("=" * 104)
    print("  FULL Fitness-AI Chatbot Benchmark — production config (UNCHANGED)")
    print(f"  Total questions: {len(CASES)}")
    print("=" * 104)
    print(f"{'#':>4}  {'CATEGORY':<16} {'MESSAGE':<44} {'EXP':<13} {'GOT':<13} RESULT")
    print("-" * 104)

    results = []
    for i, (cat, text, expected) in enumerate(CASES, 1):
        try:
            route, intent = await _route_of(engine, text)
        except Exception as e:
            route, intent = f"error", f"error:{e}"
        passed = _grade(expected, route)
        results.append({"n": i, "cat": cat, "input": text,
                        "expected": expected, "route": route,
                        "intent": intent, "passed": passed})
        mark = "PASS" if passed else "FAIL"
        print(f"{i:>4}  {cat:<16} {text[:43]:<44} {expected:<13} {route[:13]:<13} {mark}")
        await asyncio.sleep(0.02)

    await engine.flow_mgr.save_flow_state(PendingFlow())

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    acc = (passed / total * 100.0) if total else 0.0

    print("\n" + "=" * 104)
    print("  SUMMARY")
    print("=" * 104)
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
        print(f"    {c:<18} {pas}/{tot}  ({pas / tot * 100:.0f}%)")

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
                print(f"           expected={r['expected']}  got={r['route']}  intent={r['intent']!r}")
    else:
        print("\n  No failures.")

    print("=" * 104)
    sys.exit(1 if failed else 0)


# ===========================================================================
#  WAVE 3 — additional NEW questions to comfortably exceed 500 total
# ===========================================================================

# ── FOOD LOGGING — English with quantity (more) ─────────────────────────────
add("food_qty_en", FOOD, [
    "ate 2 methi muthia",
    "had 3 dhebra with tea",
    "consumed 200 ml of chaas",
    "ate 1 plate of ragda pattice",
    "had 2 aloo tikki",
    "ate 4 pieces of khakhra",
    "had a bowl of veg dalia",
    "ate 2 sabudana vada",
    "had 1 katori of aamras",
    "ate 3 dhokla with chutney",
    "had 2 bread pakora",
    "ate a cup of sprouts bhel",
    "had 5 makhana handful",
    "ate 2 stuffed paratha",
    "had 1 glass of nimbu paani",
])

# ── FOOD LOGGING — no quantity (more) ───────────────────────────────────────
add("food_noqty_en", FOOD, [
    "ate turai sabzi",
    "had chana chaat",
    "ate rava upma",
    "had veg hakka noodles",
    "ate paneer butter masala",
    "had egg curry",
    "ate rajgira sheera",
    "had kuttu ki khichdi",
    "ate singhare ka halwa",
    "had makhana kheer",
])

# ── FOOD LOGGING — multilingual (more) ──────────────────────────────────────
add("food_multiling", FOOD, [
    "aaje me vaghareli khichdi khadhi",
    "maine subah thepla aur chai li",
    "aaj raat me paneer bhurji khayi",
    "hu ae bajra no rotlo khadho",
    "me bapore dal dhokli khadhi",
    "aaj maine gatte ki sabzi khayi",
    "raatre veg biryani khadhi",
    "me thodu basundi lidhu",
    "aaj nashte me poha jalebi khadha",
    "maine ek plate chole khaye",
])

# ── FOOD LOGGING — typos (more) ─────────────────────────────────────────────
add("food_typo", FOOD, [
    "ate veg biryanii",
    "had kadhi khichdee",
    "ate methi theplaa",
    "had aloo paranthaa",
    "ate paneer tikaa",
    "had chaat papdii",
    "ate moong daal cheela",
    "had veg frankiee",
    "ate corn palak sabjii",
    "had jeera ricee",
])

# ── MULTI-FOOD (more) ───────────────────────────────────────────────────────
add("multi_food", FOOD, [
    "had khichdi, kadhi, papad and pickle",
    "ate 2 dosa, sambar and two chutneys",
    "breakfast: paratha, curd, tea and a banana",
    "aaje me khaman, fafda ane chutney lidha",
    "had rice, rasam, poriyal and curd",
    "ate a wrap, fries, and a milkshake",
    "dinner was soup, salad and grilled paneer",
    "me sanje dabeli, vada pav ane chaas khadha",
    "had 3 puri, aloo sabzi and halwa",
    "ate noodles, manchurian and spring rolls",
])

# ── EXERCISE LOGGING (more) ─────────────────────────────────────────────────
add("exercise_log", EXLOG, [
    "did 30 box jumps",
    "ran intervals for 20 minutes",
    "did 3 sets of chest fly",
    "went for a 5 km cycle ride",
    "did 45 seconds of jump squats",
    "completed 18 chin ups",
    "did 3 sets of calf raises",
    "swam 20 laps",
    "did 60 seconds of side plank each side",
    "walked 10000 steps today",
    "did 4 sets of hammer curls",
    "aaje me 20 minute yoga karyu",
    "did 100 skips warm up",
    "maine 30 lunges kiye",
    "did a 40 minute spin class",
])

# ── CALORIE / MACRO (more) ──────────────────────────────────────────────────
add("calorie_macro", CAL, [
    "how many calories in a samosa pav",
    "protein in a bowl of rajma",
    "calories in 2 aloo paratha",
    "how much fat in a cup of paneer",
    "calories in a glass of buttermilk",
    "carbs in a plate of poha",
    "how many calories in a gulab jamun",
    "protein in 100 grams of curd",
    "calories in a bowl of sambar",
    "how many calories in a cup of tea with sugar",
])

# ── DAILY LOG reads — food (more) ───────────────────────────────────────────
add("daily_log_food", DLOG, [
    "give me my calorie intake so far",
    "what have I consumed at lunch",
    "read back today's meals",
    "aaj kitna khana log hua",
    "show my morning meal entries",
])

# ── EXERCISE reads (more) ───────────────────────────────────────────────────
add("exercise_read", EXREAD, [
    "what have I trained today",
    "show my burned calories log",
    "list today's physical activity",
    "aaj ki exercise ka hisaab batao",
    "recap the workouts I did",
])

# ── JUNK (more) ─────────────────────────────────────────────────────────────
add("junk", REJECT, [
    "wewewewe",
    "what's on tv tonight",
    "qpwoeirut",
    "remind me to call mom",
    "asasasas",
    "how's the traffic",
    "zxzxzxzx",
    "tell me the news",
    "ghghghgh",
    "what's my horoscope",
])

# ── AMBIGUOUS (more) ────────────────────────────────────────────────────────
add("ambiguous", REJECT, [
    "ate the usual breakfast",
    "had a small portion",
    "kaik lidhu hatu",
    "some sweet thing",
    "had a couple of those",
])

# ===========================================================================
#  WAVE 4 — final top-up to exceed 500 total (all distinct)
# ===========================================================================
add("food_qty_en", FOOD, [
    "ate 2 corn cheese balls",
    "had 3 mini idli",
    "ate 1 bowl of veg poha upma",
    "had 2 paneer kathi rolls",
    "ate 4 mathri",
])
add("food_multiling", FOOD, [
    "aaje me sukhi bhaji rotli khadhi",
    "maine subah sabudana khichdi khayi",
    "aaj raat me veg kolhapuri khayi",
    "hu ae makai no chevdo khadho",
    "me bapore varan bhaat khadho",
])
add("exercise_log", EXLOG, [
    "did 3 sets of overhead press",
    "cycled uphill for 15 minutes",
    "did 40 flutter kicks",
    "ran a 2 km time trial",
    "did 5 sets of face pulls",
])
add("junk", REJECT, [
    "poiu qwer",
    "what's the wifi password",
    "asdf ghjk zxcv",
    "book a hotel room",
    "lalalalala",
])


if __name__ == "__main__":
    asyncio.run(main())
