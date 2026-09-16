"""
LLM Service — Cloudflare Workers AI (Qwen3)
Parses user messages into structured intents for the chatbot engine.

API: https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/qwen/qwen3-30b-a3b-fp8
Model: Qwen3 (via Workers AI)

The LLM ONLY extracts intent + entities. It does NOT calculate calories.
All nutrition data comes from the database via existing calculators.
"""
from __future__ import annotations

import os
import json
import re
import httpx
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "")
CF_MODEL = os.getenv("CF_MODEL", "@cf/qwen/qwen3-30b-a3b-fp8")
CF_API_URL = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_MODEL}"


# ---------------------------------------------------------------------------
# Intent Types
# ---------------------------------------------------------------------------

class Intent:
    LOG_FOOD = "log_food"
    LOG_EXERCISE = "log_exercise"
    GET_SUMMARY = "get_summary"
    GET_CALORIES = "get_calories"           # ask about a food's nutrition
    GET_PROFILE = "get_profile"
    QUERY_MEAL = "query_meal"               # "aaje breakfast ma su lidhu?" — check today's logs
    QUERY_EXERCISE = "query_exercise"       # "aaje me kya exercise kari?" — check today's exercise logs
    SKIP_MEAL = "skip_meal"                 # "breakfast nathi karyu" — mark meal as skipped
    RECOMMEND_WORKOUT = "recommend_workout" # "what should I do tomorrow?" — workout suggestion
    CLARIFICATION_NEEDED = "clarification_needed"
    GREETING = "greeting"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class LLMParseResult:
    """Structured output from the LLM intent parser."""
    intent: str
    food_query: Optional[str] = None
    quantity: Optional[str] = None
    variant: Optional[str] = None
    meal_type: Optional[str] = None
    exercise_input: Optional[str] = None
    date: Optional[str] = None            # YYYY-MM-DD or "today"/"yesterday"
    clarification_question: Optional[str] = None
    raw_response: str = ""
    success: bool = True
    error: Optional[str] = None

    # ── Structured entity fields (additive; safe defaults) ──────────────
    # These enrich extraction so downstream can log without re-parsing, and so
    # the bot can ask ONLY for a specific missing detail. All optional — older
    # readers that use food_query/quantity/exercise_input keep working.
    foods: list = field(default_factory=list)   # multi-item: [{food_query, quantity, variant, meal_type}, ...]
    reps: Optional[float] = None                 # exercise reps (counted movements)
    sets: Optional[float] = None                 # exercise sets
    distance: Optional[float] = None             # exercise distance amount
    distance_unit: Optional[str] = None          # km / m / miles
    duration_min: Optional[float] = None         # exercise duration in minutes
    intensity: Optional[str] = None              # light / moderate / vigorous (as stated)
    exercise_query: Optional[str] = None         # bare exercise name (no amount/verb)
    time_of_day: Optional[str] = None            # morning / afternoon / evening / night (as stated)
    missing_detail: Optional[str] = None         # which single field is missing: quantity|variant|amount|meal_type
    exact_term: Optional[str] = None             # the user's EXACT food/exercise words (never substituted)


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a fitness chatbot intent parser. Your ONLY job is to extract structured data from user messages about food logging, exercise logging, or nutrition queries.

CRITICAL OUTPUT RULES — READ FIRST:
• Respond with ONLY a single JSON object. No markdown, no code fences, no explanation.
• EVERY log_exercise response MUST populate exercise_query, exercise_input, AND at least one of: duration_min, reps, sets, or distance. Never leave all four null when the user provided an amount.
• EVERY log_food response MUST populate the "foods" array with at least one item. Never return foods:[].
• If the user message is gibberish, random characters, or completely off-topic (weather, jokes, travel, etc.) → intent MUST be "clarification_needed". Do NOT guess a food or exercise.
• If a food is logged without a quantity → set missing_detail:"quantity". If without a meal → set missing_detail:"meal_type".
• If an exercise is logged without an amount (no reps/duration/distance) → set missing_detail:"amount".

CORE PRINCIPLES:
1. Respond ONLY with valid JSON. No extra text, no markdown.
2. You do NOT calculate calories or nutrition. Just extract what the user said.
3. Understand ANY language, script, and writing style — English, Hindi, Gujarati, Hinglish, Roman transliterations, typos, phonetic spellings. Infer meaning from context like a fluent local speaker.
4. NEVER substitute the user's food/exercise with a different one. Fix spelling/transliteration only. Put verbatim words in "exact_term"; corrected English name in "food_query"/"exercise_query".
5. Extract EVERY detail present: quantity, unit, reps, sets, distance, duration, intensity, time of day, meal context. Leave a field null ONLY if truly absent.
6. If exactly ONE required detail is missing → return the log intent + set missing_detail. Only use "clarification_needed" when the food/exercise cannot be identified at all.
7. Capture EVERY food in a separate "foods" array entry. Connectives (and/aur/ane/sathe/with) are NOT foods.

OUTPUT FORMAT (JSON only, all fields present):
{"intent":"log_food|log_exercise|get_summary|get_calories|get_profile|query_meal|query_exercise|recommend_workout|clarification_needed|skip_meal|greeting|unknown","exact_term":null,"food_query":null,"quantity":null,"variant":null,"meal_type":null,"foods":[],"exercise_query":null,"exercise_input":null,"reps":null,"sets":null,"distance":null,"distance_unit":null,"duration_min":null,"intensity":null,"time_of_day":null,"date":"today","missing_detail":null,"clarification_question":null}

RECOMMEND_WORKOUT intent:
• Use intent "recommend_workout" when the user asks what workout/exercise to do tomorrow, next, or suggests a training plan.
• Examples: "what should I do tomorrow?", "suggest a workout", "what exercise for tomorrow?", "kal kya karna chahiye?", "aavti kal exercise suchavjo", "mane workout suggest karo", "recommend a workout plan".

EXERCISE EXTRACTION RULES (most important — always fill these fields):
• exercise_query = English name of the exercise (e.g. "jogging", "squats", "cycling")
• duration_min   = numeric minutes when duration is given ("30 minutes" → 30, "1 hour" → 60)
• reps           = numeric rep count when reps/movements are given ("50 push-ups" → 50)
• sets           = numeric set count when sets are mentioned ("3 sets" → 3)
• distance       = numeric distance value; distance_unit = "km"|"m"|"miles"
• exercise_input = compact phrase combining exercise + amount, e.g. "jogging 30 minutes", "3 sets of 12 squats", "cycling 5 km"
• missing_detail = "amount" ONLY when no reps/duration/distance/sets were given at all

FOOD EXTRACTION RULES:
• foods array: list EVERY food mentioned, each as {"food_query":"...","quantity":"...","variant":null,"meal_type":"..."}
• food_query: canonical English food name (fix typos/transliterations, keep the same dish)
• quantity: exactly what the user said ("2 pieces", "1 bowl", "150g", "1 glass")
• missing_detail: "quantity" when amount absent; "meal_type" when meal context absent and both are missing → use "quantity"
• variant: cooking method only (ghee/fried/boiled/grilled/steamed/baked/oil/butter)

DATE RULES:
• "today" / "aaj" / "aaje"                          → date:"today"
• "yesterday" / "kal" / "kale" / "gatkale" / "gatkal" / "gay kale" / "gay kal" → date:"yesterday"
• Any YYYY-MM-DD string → use it verbatim

JUNK/OFF-TOPIC DETECTION (use clarification_needed):
• Random characters with no food/exercise meaning: "abcdef", "qwerty", "xyz123", "asdfgh"
• Completely off-topic: weather, jokes, travel, news, translation, alarms, passwords
• Vague with no identifiable food: "I ate something", "kuch khaya", "had stuff"
• DO NOT try to find a food in gibberish — return clarification_needed immediately

MULTILINGUAL NORMALIZATION:
• Roman Gujarati food → English: roti/rotli→roti, thepla→thepla, dal bhaat→dal+rice, bhakri→bhakri, shaak→mixed vegetable, chaas/chhas→buttermilk, dudh→milk, dahi→curd, ande→egg, murghi→chicken, bhaji→mixed vegetable, kanda→onion, bataka→potato, marcha→chilli
• Roman Gujarati exercise: chalvun/chalo→walking, daudvun→running, kasarat→exercise, taravun→swimming
• Hindi food: daal→dal, chawal→rice, sabzi→mixed vegetable, anda→egg, doodh→milk, chai→chai, poori→puri
• Hindi exercise: daudna→running, paidhal chalna→walking, taarna→swimming, kasrat→exercise

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLETE WORKED EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

── EXERCISE WITH DURATION ──
User: "did 30 minutes of jogging"
→ {"intent":"log_exercise","exact_term":"jogging","exercise_query":"jogging","exercise_input":"jogging 30 minutes","duration_min":30,"reps":null,"sets":null,"distance":null,"distance_unit":null,"date":"today","missing_detail":null}

User: "aaj 40 minute cycling ki"
→ {"intent":"log_exercise","exact_term":"cycling","exercise_query":"cycling","exercise_input":"cycling 40 minutes","duration_min":40,"reps":null,"sets":null,"distance":null,"distance_unit":null,"date":"today","missing_detail":null}

User: "did 45 minutes of yoga"
→ {"intent":"log_exercise","exact_term":"yoga","exercise_query":"yoga","exercise_input":"yoga 45 minutes","duration_min":45,"reps":null,"sets":null,"distance":null,"distance_unit":null,"date":"today","missing_detail":null}

User: "me aaje 1 ghanta chalyo"
→ {"intent":"log_exercise","exact_term":"chalyo","exercise_query":"walking","exercise_input":"walking 60 minutes","duration_min":60,"reps":null,"sets":null,"distance":null,"distance_unit":null,"date":"today","missing_detail":null}

── EXERCISE WITH REPS ──
User: "did 50 push-ups"
→ {"intent":"log_exercise","exact_term":"push-ups","exercise_query":"push-ups","exercise_input":"50 push-ups","reps":50,"sets":null,"duration_min":null,"distance":null,"distance_unit":null,"date":"today","missing_detail":null}

User: "me aaje 50 push-ups kara"
→ {"intent":"log_exercise","exact_term":"push-ups","exercise_query":"push-ups","exercise_input":"50 push-ups","reps":50,"sets":null,"duration_min":null,"distance":null,"distance_unit":null,"date":"today","missing_detail":null}

User: "completed 100 jumping jacks"
→ {"intent":"log_exercise","exact_term":"jumping jacks","exercise_query":"jumping jacks","exercise_input":"100 jumping jacks","reps":100,"sets":null,"duration_min":null,"distance":null,"distance_unit":null,"date":"today","missing_detail":null}

── EXERCISE WITH SETS × REPS ──
User: "completed 3 sets of 12 squats"
→ {"intent":"log_exercise","exact_term":"squats","exercise_query":"squats","exercise_input":"3 sets of 12 squats","sets":3,"reps":12,"duration_min":null,"distance":null,"distance_unit":null,"date":"today","missing_detail":null}

User: "aaj mene 4 sets of 10 bench press kiya"
→ {"intent":"log_exercise","exact_term":"bench press","exercise_query":"bench press","exercise_input":"4 sets of 10 bench press","sets":4,"reps":10,"duration_min":null,"distance":null,"distance_unit":null,"date":"today","missing_detail":null}

── EXERCISE WITH DISTANCE ──
User: "ran 5 km this morning"
→ {"intent":"log_exercise","exact_term":"running","exercise_query":"running","exercise_input":"running 5 km","distance":5,"distance_unit":"km","duration_min":null,"reps":null,"sets":null,"time_of_day":"morning","date":"today","missing_detail":null}

User: "cycled 10 km this evening"
→ {"intent":"log_exercise","exact_term":"cycling","exercise_query":"cycling","exercise_input":"cycling 10 km","distance":10,"distance_unit":"km","duration_min":null,"reps":null,"sets":null,"time_of_day":"evening","date":"today","missing_detail":null}

── EXERCISE WITHOUT AMOUNT (missing_detail) ──
User: "aaje me push ups mara"
→ {"intent":"log_exercise","exact_term":"push ups","exercise_query":"push-ups","exercise_input":"push-ups","reps":null,"sets":null,"duration_min":null,"distance":null,"distance_unit":null,"date":"today","missing_detail":"amount"}

User: "did some jogging today"
→ {"intent":"log_exercise","exact_term":"jogging","exercise_query":"jogging","exercise_input":"jogging","reps":null,"sets":null,"duration_min":null,"distance":null,"distance_unit":null,"date":"today","missing_detail":"amount"}

── MULTI-FOOD (CRITICAL: foods array must contain ALL items) ──
User: "I had 2 roti, 1 bowl dal and salad for lunch"
→ {"intent":"log_food","exact_term":"roti, dal, salad","food_query":"roti","quantity":"2 pieces","meal_type":"lunch","foods":[{"food_query":"roti","quantity":"2 pieces","variant":null,"meal_type":"lunch"},{"food_query":"dal","quantity":"1 bowl","variant":null,"meal_type":"lunch"},{"food_query":"salad","quantity":"1 serving","variant":null,"meal_type":"lunch"}],"date":"today","missing_detail":null}

User: "aaje me breakfast ma thepla, chundo ane chai lidha"
→ {"intent":"log_food","exact_term":"thepla, chundo, chai","food_query":"thepla","quantity":"2 pieces","meal_type":"breakfast","foods":[{"food_query":"thepla","quantity":"2 pieces","variant":null,"meal_type":"breakfast"},{"food_query":"chundo","quantity":"1 serving","variant":null,"meal_type":"breakfast"},{"food_query":"chai","quantity":"1 cup","variant":null,"meal_type":"breakfast"}],"date":"today","missing_detail":null}

User: "Breakfast was idli, sambar and coconut chutney"
→ {"intent":"log_food","exact_term":"idli, sambar, coconut chutney","food_query":"idli","quantity":"2 pieces","meal_type":"breakfast","foods":[{"food_query":"idli","quantity":"2 pieces","variant":null,"meal_type":"breakfast"},{"food_query":"sambar","quantity":"1 bowl","variant":null,"meal_type":"breakfast"},{"food_query":"coconut chutney","quantity":"1 serving","variant":null,"meal_type":"breakfast"}],"date":"today","missing_detail":null}

User: "maine subah doodh, anda aur toast khayi"
→ {"intent":"log_food","exact_term":"doodh, anda, toast","food_query":"milk","quantity":"1 glass","meal_type":"breakfast","foods":[{"food_query":"milk","quantity":"1 glass","variant":null,"meal_type":"breakfast"},{"food_query":"egg","quantity":"1 piece","variant":null,"meal_type":"breakfast"},{"food_query":"toast","quantity":"2 slices","variant":null,"meal_type":"breakfast"}],"date":"today","missing_detail":null}

User: "dinner was grilled chicken, rice and steamed vegetables"
→ {"intent":"log_food","exact_term":"chicken, rice, vegetables","food_query":"chicken","quantity":"150g","variant":"grilled","meal_type":"dinner","foods":[{"food_query":"chicken","quantity":"150g","variant":"grilled","meal_type":"dinner"},{"food_query":"rice","quantity":"1 bowl","variant":null,"meal_type":"dinner"},{"food_query":"steamed vegetables","quantity":"1 serving","variant":"steamed","meal_type":"dinner"}],"date":"today","missing_detail":null}

User: "raatre me 2 rotli ane dal khadha"
→ {"intent":"log_food","exact_term":"rotli, dal","food_query":"roti","quantity":"2 pieces","meal_type":"dinner","foods":[{"food_query":"roti","quantity":"2 pieces","variant":null,"meal_type":"dinner"},{"food_query":"dal","quantity":"1 serving","variant":null,"meal_type":"dinner"}],"date":"today","missing_detail":null}

── SINGLE FOOD ──
User: "maine breakfast me 3 idli khaayi with ghee"
→ {"intent":"log_food","exact_term":"idli","food_query":"idli","quantity":"3 pieces","variant":"ghee","meal_type":"breakfast","foods":[{"food_query":"idli","quantity":"3 pieces","variant":"ghee","meal_type":"breakfast"}],"date":"today","missing_detail":null}

User: "chocolate ice cream khadhu"
→ {"intent":"log_food","exact_term":"chocolate ice cream","food_query":"chocolate ice cream","quantity":null,"meal_type":null,"foods":[{"food_query":"chocolate ice cream","quantity":null,"variant":null,"meal_type":null}],"date":"today","missing_detail":"quantity"}

User: "ate paneer tikka for dinner"
→ {"intent":"log_food","exact_term":"paneer tikka","food_query":"paneer tikka","quantity":null,"meal_type":"dinner","foods":[{"food_query":"paneer tikka","quantity":null,"variant":null,"meal_type":"dinner"}],"date":"today","missing_detail":"quantity"}

User: "had 2 roti"
→ {"intent":"log_food","exact_term":"roti","food_query":"roti","quantity":"2 pieces","meal_type":null,"foods":[{"food_query":"roti","quantity":"2 pieces","variant":null,"meal_type":null}],"date":"today","missing_detail":"meal_type"}

── GUJARATI SINGLE FOOD ──
User: "aaje me 2 thepla khadha"
→ {"intent":"log_food","exact_term":"thepla","food_query":"thepla","quantity":"2 pieces","meal_type":null,"foods":[{"food_query":"thepla","quantity":"2 pieces","variant":null,"meal_type":null}],"date":"today","missing_detail":"meal_type"}

User: "savare bhakri ane chhas lidha"
→ {"intent":"log_food","exact_term":"bhakri, chhas","food_query":"bhakri","quantity":"1 piece","meal_type":"breakfast","foods":[{"food_query":"bhakri","quantity":"1 piece","variant":null,"meal_type":"breakfast"},{"food_query":"chhas","quantity":"1 glass","variant":null,"meal_type":"breakfast"}],"date":"today","missing_detail":null}

User: "hu sanje 1 samosa khadho"
→ {"intent":"log_food","exact_term":"samosa","food_query":"samosa","quantity":"1 piece","meal_type":"snack","foods":[{"food_query":"samosa","quantity":"1 piece","variant":null,"meal_type":"snack"}],"date":"today","missing_detail":null}

── HINDI FOOD ──
User: "maine subah 2 ande khaaye"
→ {"intent":"log_food","exact_term":"ande","food_query":"egg","quantity":"2 pieces","meal_type":"breakfast","foods":[{"food_query":"egg","quantity":"2 pieces","variant":null,"meal_type":"breakfast"}],"date":"today","missing_detail":null}

User: "shaam ko chai ke saath 2 biscuit khayi"
→ {"intent":"log_food","exact_term":"chai, biscuit","food_query":"chai","quantity":"1 cup","meal_type":"snack","foods":[{"food_query":"chai","quantity":"1 cup","variant":null,"meal_type":"snack"},{"food_query":"biscuit","quantity":"2 pieces","variant":null,"meal_type":"snack"}],"date":"today","missing_detail":null}

User: "raat ko 3 roti aur sabzi khayi"
→ {"intent":"log_food","exact_term":"roti, sabzi","food_query":"roti","quantity":"3 pieces","meal_type":"dinner","foods":[{"food_query":"roti","quantity":"3 pieces","variant":null,"meal_type":"dinner"},{"food_query":"mixed vegetable","quantity":"1 serving","variant":null,"meal_type":"dinner"}],"date":"today","missing_detail":null}

── NUTRITION QUERIES ──
User: "paneer me kitni calories hoti hai?"
→ {"intent":"get_calories","exact_term":"paneer","food_query":"paneer","quantity":"100g","date":null,"missing_detail":null}

User: "how many calories in a banana"
→ {"intent":"get_calories","exact_term":"banana","food_query":"banana","quantity":"1 piece","date":null,"missing_detail":null}

User: "how much protein in 100g paneer"
→ {"intent":"get_calories","exact_term":"paneer","food_query":"paneer","quantity":"100g","date":null,"missing_detail":null}

User: "calories in 1 roti"
→ {"intent":"get_calories","exact_term":"roti","food_query":"roti","quantity":"1 piece","date":null,"missing_detail":null}

── DAILY LOG READS ──
User: "aaj ka summary dikhao"
→ {"intent":"get_summary","date":"today","missing_detail":null}

User: "what exercise did I do today"
→ {"intent":"query_exercise","date":"today","missing_detail":null}

User: "aaj mene kya exercise ki"
→ {"intent":"query_exercise","date":"today","missing_detail":null}

── YESTERDAY DATE ──
User: "yesterday I had dal chawal for lunch"
→ {"intent":"log_food","exact_term":"dal chawal","food_query":"dal","quantity":"1 serving","meal_type":"lunch","foods":[{"food_query":"dal","quantity":"1 serving","variant":null,"meal_type":"lunch"},{"food_query":"rice","quantity":"1 serving","variant":null,"meal_type":"lunch"}],"date":"yesterday","missing_detail":null}

User: "gatkale me thepla khadhi"
→ {"intent":"log_food","exact_term":"thepla","food_query":"thepla","quantity":"1 piece","meal_type":null,"foods":[{"food_query":"thepla","quantity":"1 piece","variant":null,"meal_type":null}],"date":"yesterday","missing_detail":"meal_type"}

User: "kal subah paratha khaya"
→ {"intent":"log_food","exact_term":"paratha","food_query":"paratha","quantity":"1 piece","meal_type":"breakfast","foods":[{"food_query":"paratha","quantity":"1 piece","variant":null,"meal_type":"breakfast"}],"date":"yesterday","missing_detail":null}

── SKIP MEAL ──
User: "I skipped breakfast today"
→ {"intent":"skip_meal","meal_type":"breakfast","date":"today","missing_detail":null}

User: "aaj lunch nahi khaya"
→ {"intent":"skip_meal","meal_type":"lunch","date":"today","missing_detail":null}

── JUNK / OFF-TOPIC → clarification_needed ──
User: "abcdef"
→ {"intent":"clarification_needed","clarification_question":"I couldn't identify a food or exercise. What would you like to log?","missing_detail":null}

User: "qwerty"
→ {"intent":"clarification_needed","clarification_question":"I couldn't identify a food or exercise. What would you like to log?","missing_detail":null}

User: "what is the weather today"
→ {"intent":"clarification_needed","clarification_question":"I can only help with food/exercise logging and nutrition. What would you like to track?","missing_detail":null}

User: "book me a cab"
→ {"intent":"clarification_needed","clarification_question":"I can only help with food/exercise logging and nutrition. What would you like to track?","missing_detail":null}

User: "tell me a joke"
→ {"intent":"clarification_needed","clarification_question":"I can only help with food/exercise logging and nutrition. What would you like to track?","missing_detail":null}

User: "I ate something"
→ {"intent":"clarification_needed","clarification_question":"What food did you eat? Please specify the name.","missing_detail":null}

User: "kuch khaya"
→ {"intent":"clarification_needed","clarification_question":"Kya khaya? Please food ka naam batayein.","missing_detail":null}

── GREETINGS ──
User: "hi" / "hello" / "namaste" / "kem cho"
→ {"intent":"greeting","missing_detail":null}

── MEAL-TIME SEMANTICS ──
User: "aaje me bapor pachhi ek samosa khadha"
→ {"intent":"log_food","exact_term":"samosa","food_query":"samosa","quantity":"1 piece","meal_type":"snack","foods":[{"food_query":"samosa","quantity":"1 piece","variant":null,"meal_type":"snack"}],"time_of_day":"afternoon","date":"today","missing_detail":null}

User: "raat ne pehla thodu bhel khadhu"
→ {"intent":"log_food","exact_term":"bhel","food_query":"bhel","quantity":null,"meal_type":"snack","foods":[{"food_query":"bhel","quantity":null,"variant":null,"meal_type":"snack"}],"time_of_day":"evening","date":"today","missing_detail":"quantity"}

User: "aaje me vehli savar ma poha khadho"
→ {"intent":"log_food","exact_term":"poha","food_query":"poha","quantity":null,"meal_type":"breakfast","foods":[{"food_query":"poha","quantity":null,"variant":null,"meal_type":"breakfast"}],"time_of_day":"morning","date":"today","missing_detail":"quantity"}

MEAL-TYPE RULE: Morning/subah/savare/vehli savar → breakfast. Midday/dopahar/bapore/lunch → lunch. Afternoon/tea-time/sanje/bapor pachhi/shaam → snack. Evening/shaam → snack. Night/raat/raatre/dinner → dinner. After-any-meal modifier → snack.

ALWAYS respond with ONLY the JSON object. Any field not mentioned by the user → null (foods:[] only for non-food intents)."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class LLMService:
    """Cloudflare Workers AI Qwen3 service for intent parsing."""

    def __init__(
        self,
        account_id: Optional[str] = None,
        api_token: Optional[str] = None,
        model: Optional[str] = None,
    ):
        # Load CF credentials: try the app's settings object first (available
        # when the FastAPI server is running), fall back to reading .env
        # directly so standalone scripts (benchmarks, eval) work even when
        # MongoDB is unreachable.
        if account_id and api_token:
            self.account_id = account_id
            self.api_token  = api_token
            self.model      = model or os.getenv("CF_MODEL", "@cf/qwen/qwen3-30b-a3b-fp8")
        else:
            try:
                from database import get_settings
                settings = get_settings()
                self.account_id = account_id or settings.CF_ACCOUNT_ID
                self.api_token  = api_token  or settings.CF_API_TOKEN
                self.model      = model      or settings.CF_MODEL
            except Exception:
                # database module unavailable (e.g. Atlas unreachable) —
                # fall back to env vars loaded from .env file by dotenv.
                _env_file = os.path.join(os.path.dirname(__file__), ".env")
                if os.path.exists(_env_file):
                    with open(_env_file) as _f:
                        for _line in _f:
                            _line = _line.strip()
                            if _line and not _line.startswith("#") and "=" in _line:
                                _k, _, _v = _line.partition("=")
                                os.environ.setdefault(_k.strip(), _v.strip())
                self.account_id = account_id or os.getenv("CF_ACCOUNT_ID", "")
                self.api_token  = api_token  or os.getenv("CF_API_TOKEN",  "")
                self.model      = model      or os.getenv("CF_MODEL", "@cf/qwen/qwen3-30b-a3b-fp8")
        self.api_url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{self.account_id}/ai/run/{self.model}"
        )

    async def parse_intent(self, user_message: str, context: Optional[str] = None) -> LLMParseResult:
        """
        Send user message to Qwen3 and get structured intent.

        Args:
            user_message: The user's text (could be in Hindi/Gujarati/English)
            context: Optional conversation context (e.g. previous clarification)

        Returns:
            LLMParseResult with extracted intent and entities.
        """
        if not self.account_id or not self.api_token:
            # Fallback: use rule-based parsing if no API keys
            return self._fallback_parse(user_message)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if context:
            messages.append({"role": "user", "content": f"[Context: {context}]"})
        messages.append({"role": "user", "content": user_message})
        # Inject a partial assistant message that begins with </think> to
        # immediately end Qwen3's thinking chain and force straight JSON output.
        # This suppresses the <think>…</think> reasoning block that consumes
        # tokens and can truncate the JSON response at max_tokens.
        messages.append({"role": "assistant", "content": "</think>\n"})

        payload = {
            "messages": messages,
            # 1500 tokens is enough for the JSON even on multi-food sentences,
            # and the increased budget ensures the response is never truncated.
            "max_tokens": 1500,
            "temperature": 0.1,
        }
        # If a LoRA fine-tune has been uploaded to Cloudflare (via
        # training/upload_lora_cf.py), activate it by setting CF_FINETUNE_ID in
        # backend/.env — one env var switches from base Qwen3 to the fine-tuned
        # adapter without any other code changes.
        finetune_id = os.getenv("CF_FINETUNE_ID", "").strip()
        if finetune_id:
            payload["lora"] = finetune_id
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.api_url, json=payload, headers=headers)

            if response.status_code != 200:
                # Fallback to rule-based
                return self._fallback_parse(user_message)

            data = response.json()
            # Workers AI returns {"result": {"response": "..."}}
            raw_text = ""
            if "result" in data:
                raw_text = data["result"].get("response", "")
            elif "choices" in data:
                raw_text = data["choices"][0].get("message", {}).get("content", "")

            return self._parse_json_response(raw_text, user_message)

        except Exception as e:
            # On any error, use fallback
            return self._fallback_parse(user_message)

    def parse_intent_sync(self, user_message: str, context: Optional[str] = None) -> LLMParseResult:
        """Synchronous version for testing."""
        if not self.account_id or not self.api_token:
            return self._fallback_parse(user_message)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if context:
            messages.append({"role": "user", "content": f"[Context: {context}]"})
        messages.append({"role": "user", "content": user_message})
        # Same thinking-suppression and token budget as parse_intent.
        messages.append({"role": "assistant", "content": "</think>\n"})

        payload = {"messages": messages, "max_tokens": 1500, "temperature": 0.1}
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(self.api_url, json=payload, headers=headers)

            if response.status_code != 200:
                return self._fallback_parse(user_message)

            data = response.json()
            raw_text = ""
            if "result" in data:
                raw_text = data["result"].get("response", "")
            elif "choices" in data:
                raw_text = data["choices"][0].get("message", {}).get("content", "")

            return self._parse_json_response(raw_text, user_message)

        except Exception:
            return self._fallback_parse(user_message)

    # ------------------------------------------------------------------
    # Food-plausibility gate
    # ------------------------------------------------------------------
    def _looks_like_non_food(self, text: str) -> Optional[bool]:
        """
        Fast, offline heuristic that decides if `text` is CLEARLY not a food.

        Returns:
          True  → clearly NOT a food (gibberish / greeting / conversational).
          False → looks like it COULD be a food name (let LLM/DB decide).
          None  → uncertain (defer to the LLM check).

        This never *confirms* a food on its own — it only rejects the obvious
        non-food cases so we don't fabricate a food from random text.
        """
        t = (text or "").strip().lower()
        if not t:
            return True

        # Non-Latin script (Gujarati/Hindi/Tamil/…): the Latin-only heuristics
        # below cannot judge these, and they are very likely real food names in
        # another language. Defer to the LLM/DB (return None) — never reject.
        if re.search(r"[^\x00-\x7f]", t):
            return None

        latin_letters = re.sub(r"[^a-z]", "", t)
        # No alphabetic content at all in Latin text (e.g. "123", "!!!", "42.5").
        if not latin_letters:
            return True

        # Pure conversational / greeting / status phrases — never a food.
        conversational = {
            "hi", "hello", "hey", "yo", "hii", "hiii", "namaste", "namaskar",
            "how are you", "how r u", "how are u", "whats up", "what's up", "sup",
            "good morning", "good evening", "good night", "good afternoon",
            "today is good", "i am good", "im good", "i am fine", "im fine",
            "thanks", "thank you", "ok", "okay", "yes", "no", "bye", "good",
            "nice", "cool", "great", "test", "testing", "who are you", "what can you do",
        }
        if t in conversational:
            return True
        # A short phrase that is ALL common English stop/greeting words → not food.
        _stopwords = {
            "i", "am", "is", "are", "was", "were", "the", "a", "an", "to", "of",
            "and", "or", "you", "u", "r", "how", "what", "why", "when", "where",
            "today", "good", "bad", "nice", "fine", "ok", "okay", "hello", "hi",
            "hey", "thanks", "thank", "yes", "no", "me", "my", "it", "this", "that",
            "here", "there", "now", "then", "so", "very", "really", "just",
            # additional filler / vague quantity words (never a food by themselves)
            "some", "any", "a", "bit", "little", "lot", "lots", "couple", "few",
            "had", "ate", "eat", "eaten", "having", "have", "again", "usual",
            "regular", "special", "homemade", "leftover", "leftovers", "usually",
            "whatever", "something", "anything", "everything", "stuff", "thing",
            "things", "that", "those", "these", "kaik", "thoda", "thodu", "sa",
            "badhu", "khaidhu", "peedhu", "of", "my", "for",
            # vague eating/portion words (not a specific food)
            "grabbed", "bite", "bites", "portion", "portions", "small", "big",
            "large", "dish", "dishes", "meal", "meals", "food", "foods", "drink",
            "drinks", "sweet", "sweets", "savoury", "savory", "item", "items",
            "plate", "bowl", "glass", "cup", "serving", "servings", "khaidhu",
            "peedhu", "lidhu", "khaya", "tha", "again", "kuch", "kasu", "kayi",
            "koi", "jara", "jaraak",
        }
        tokens = re.findall(r"[a-z]+", t)
        if tokens and all(tok in _stopwords for tok in tokens):
            return True

        # Clearly NON-FOOD subject nouns — appear in off-topic / junk requests
        # ("book me a cab", "what's the capital of france", "reset my password").
        # If EVERY content token is a known non-food word (after removing stop
        # words), the message is not about food. This never rejects a real food
        # because these words are not food names.
        _non_food_terms = {
            "weather", "cab", "taxi", "capital", "france", "password", "movie",
            "cricket", "match", "stock", "stocks", "market", "email", "boss",
            "music", "song", "everest", "mount", "president", "traffic", "news",
            "horoscope", "wifi", "hotel", "room", "phone", "battery",
            "car", "joke", "jokes", "story", "bedtime", "colour", "color", "spanish",
            "translate", "reset", "book", "remind", "alarm", "calculator", "app",
            "tv", "mom", "meeting", "rain", "tomorrow", "times",
            "money", "salary", "game", "won",
            "favourite", "favorite", "world", "country", "city",
            "lorem", "ipsum", "dolor", "sit", "amet", "consectetur",
            # additional off-topic verbs / nouns common in junk requests
            "tell", "play", "send", "open", "set", "search", "find", "show",
            "call", "book", "schedule", "order", "buy", "watch", "listen",
            "language", "english", "hindi", "gujarati", "translate",
            "temperature", "humidity", "forecast",
            "password", "login", "account", "settings",
        }
        content = [tok for tok in tokens if tok not in _stopwords]
        if content and all(tok in _non_food_terms for tok in content):
            return True

        # Multi-word off-topic phrases that survive the per-token check
        _offtopic_phrases = [
            "tell me a joke", "tell me a story", "play music", "play a song",
            "book a cab", "book me a cab", "order food", "call a taxi",
            "what is the weather", "how is the weather",
            "send an email", "send email", "set an alarm", "set alarm",
            "remind me to", "translate this", "open the", "search for",
        ]
        if any(phr in t for phr in _offtopic_phrases):
            return True

        # Letters glued to digits ("xyz123", "abc42") — not a food name.
        # Exception: quantity+unit patterns are food context ("150g", "250ml",
        # "2 pieces") — never reject those, they appear in real food sentences.
        if re.search(r"^\d+(?:\.\d+)?\s*(?:g|gm|ml|kg|l)\b", t):
            # Starts with a weight/volume quantity — clearly a food context
            return None
        for raw_tok in t.split():
            if re.search(r"[a-z]", raw_tok) and re.search(r"\d", raw_tok):
                # Allow standard quantity tokens like "150g", "200ml", "2kg"
                if re.fullmatch(r"\d+(?:\.\d+)?(?:g|gm|gms|ml|kg|l|oz)\b", raw_tok, re.I):
                    continue
                return True

        # ── Per-token gibberish detection ──────────────────────────────────
        # A single content token that is gibberish makes the whole phrase a
        # non-food (a real food name never contains a keyboard-mash word). This
        # catches BOTH "dvrbbweeg" (single) and "dvrbbweeg somthing" / "asdfg
        # qwerty" (multi-word) that previously slipped through to the LLM.
        content_tokens = [tok for tok in tokens if tok not in _stopwords]
        if content_tokens and any(self._token_is_gibberish(tok) for tok in content_tokens):
            return True

        return None  # uncertain → let the LLM confirm

    @staticmethod
    def _token_is_gibberish(tok: str) -> bool:
        """
        Heuristic gibberish test for a single alphabetic token. Conservative so
        real short foods (chhas, fafda, bhindi, dal, ghee) are NEVER flagged.
        """
        if len(tok) < 3:
            return False
        # Exception: known transliterations with consonant clusters like "bhkri", "bhkhri"
        if tok in ("bhkri", "bhkhri", "bhkari"):
            return False
        # A real word/food almost always contains a true vowel (a/e/i/o/u).
        # 'y' alone does NOT count, so "vvysdv"/"bcdfg"/"dvrbbwg" are gibberish
        # while "chhas"/"fafda"/"bhindi" (which have a/e/i/o/u) are kept.
        if not re.search(r"[aeiou]", tok):
            return True
        # 4+ consecutive consonants — implausible for a real word/food
        # (e.g. "dvrbbweeg" → "dvrbb", "asdfg" → "sdfg"). Real Indian foods top
        # out at digraph/trigraph clusters (chh, ghr, str) so this won't catch them.
        if re.search(r"[bcdfghjklmnpqrstvwxz]{4,}", tok):
            return True
        return False

    async def is_food_query(self, text: str) -> bool:
        """
        True only when `text` is plausibly the name of a real food/drink.

        Used to GATE the AI-estimate tier so random / conversational / gibberish
        text is never fabricated into a food. Combines a deterministic non-food
        reject with an LLM yes/no confirmation (LLM used only when configured and
        the heuristic is uncertain).
        """
        verdict = self._looks_like_non_food(text)
        if verdict is True:
            return False

        # If the LLM is available, confirm with a strict yes/no. This catches
        # things the heuristic can't (unusual but real dishes vs. plausible-
        # looking nonsense). On any error we FALL BACK to the heuristic verdict.
        if self.account_id and self.api_token:
            prompt = (
                "Is the following text the name of a real, specific food or drink "
                "(a dish, ingredient, beverage, snack, or packaged food item)?\n"
                f"Text: \"{text.strip()}\"\n"
                "Answer with ONLY one word: YES or NO.\n"
                "Answer NO for greetings, questions, random letters, gibberish, "
                "or general conversation that is not naming a food."
            )
            try:
                payload = {
                    "messages": [
                        {"role": "system", "content": "You are a strict food-name classifier. Reply only YES or NO."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 4,
                    "temperature": 0.0,
                }
                headers = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
                async with httpx.AsyncClient(timeout=8.0) as client:
                    resp = await client.post(self.api_url, json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    raw = ""
                    if "result" in data:
                        raw = data["result"].get("response", "")
                    elif "choices" in data:
                        raw = data["choices"][0].get("message", {}).get("content", "")
                    ans = (raw or "").strip().lower()
                    if ans.startswith("yes"):
                        return True
                    if ans.startswith("no"):
                        return False
            except Exception:
                pass  # fall through to heuristic verdict

        # No LLM (or LLM failed): accept only when the heuristic did NOT reject.
        # verdict is False (looks food-like) or None (uncertain) → allow, since
        # the confident DB/external tiers already failed and the text passed the
        # non-food guard.
        return verdict is not True

    async def estimate_food_nutrition(self, food_name: str, quantity: Optional[str] = None) -> dict:
        """
        Estimate calories and macros for food items NOT present in local database.
        Uses Cloudflare LLM (Qwen3) if available, with a smart rule-based fallback.
        """
        food_clean = food_name.strip()
        qty_clean = quantity or "1 serving"

        # Attempt LLM estimation if configured
        if self.account_id and self.api_token:
            prompt = (
                f"Estimate nutrition for this food item: '{food_clean}' (Quantity: '{qty_clean}').\n"
                "Respond ONLY with valid JSON in this exact structure:\n"
                "{\n"
                '  "food_name_display": "Clean display name",\n'
                '  "serving_size_g": 150,\n'
                '  "serving_unit": "serving",\n'
                '  "calories_kcal": 220,\n'
                '  "protein_g": 6.0,\n'
                '  "carbs_g": 35.0,\n'
                '  "fat_g": 7.0,\n'
                '  "fiber_g": 2.0\n'
                "}"
            )
            try:
                payload = {
                    "messages": [
                        {"role": "system", "content": "You are a professional nutritionist. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 200,
                    "temperature": 0.1
                }
                headers = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(self.api_url, json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = ""
                    if "result" in data:
                        raw_text = data["result"].get("response", "")
                    elif "choices" in data:
                        raw_text = data["choices"][0].get("message", {}).get("content", "")

                    raw_text = raw_text.strip()
                    if raw_text.startswith("```"):
                        raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text[3:]
                        if raw_text.endswith("```"):
                            raw_text = raw_text[:-3]
                        raw_text = raw_text.strip()
                    parsed = json.loads(raw_text)
                    if "calories_kcal" in parsed:
                        return parsed
            except Exception:
                pass  # Fall through to heuristic estimation

        return self._heuristic_nutrition_estimator(food_clean, qty_clean)

    def _heuristic_nutrition_estimator(self, food_name: str, quantity: str) -> dict:
        """Fallback nutritionist estimator for foods outside database."""
        food_low = food_name.lower()
        cals, protein, carbs, fat, fiber = 200.0, 5.0, 30.0, 6.0, 2.0
        unit = "serving"
        size_g = 150.0

        if any(w in food_low for w in ["smoothie", "shake", "juice", "drink"]):
            cals, protein, carbs, fat, fiber = 180.0, 4.0, 38.0, 2.5, 3.0
            unit = "glass"
            size_g = 250.0
        elif any(w in food_low for w in ["dabeli", "vada pav", "sandwich", "burger", "taco", "wrap"]):
            cals, protein, carbs, fat, fiber = 230.0, 6.0, 35.0, 8.0, 2.5
            unit = "piece"
            size_g = 140.0
        elif any(w in food_low for w in ["khichu", "dhokla", "handvo", "khandvi", "patra"]):
            cals, protein, carbs, fat, fiber = 160.0, 4.5, 28.0, 4.0, 2.0
            unit = "plate"
            size_g = 120.0
        elif any(w in food_low for w in ["pizza", "pasta", "noodle", "noodles", "chowmein", "lasagna"]):
            cals, protein, carbs, fat, fiber = 320.0, 10.0, 45.0, 12.0, 3.0
            unit = "plate"
            size_g = 200.0
        elif any(w in food_low for w in ["salad", "sprouts", "fruit", "berries", "apple", "banana"]):
            cals, protein, carbs, fat, fiber = 110.0, 3.0, 22.0, 1.0, 4.5
            unit = "bowl"
            size_g = 150.0
        elif any(w in food_low for w in ["cake", "pastry", "ice cream", "sweet", "halwa", "kheer", "dessert"]):
            cals, protein, carbs, fat, fiber = 280.0, 3.5, 42.0, 11.0, 1.0
            unit = "serving"
            size_g = 100.0
        elif any(w in food_low for w in ["curry", "sabji", "gravy", "paneer", "chicken", "mutton", "fish"]):
            cals, protein, carbs, fat, fiber = 240.0, 12.0, 15.0, 14.0, 2.0
            unit = "bowl"
            size_g = 180.0

        # Adjust for count if multiplier present (e.g. 2 pieces)
        count_match = _re.search(r"(\d+(?:\.\d+)?)", quantity)
        if count_match:
            try:
                multiplier = float(count_match.group(1))
                if 0.1 <= multiplier <= 10.0:
                    cals *= multiplier
                    protein *= multiplier
                    carbs *= multiplier
                    fat *= multiplier
                    fiber *= multiplier
            except ValueError:
                pass

        return {
            "food_name_display": food_name.title(),
            "serving_size_g": size_g,
            "serving_unit": unit,
            "calories_kcal": round(cals, 1),
            "protein_g": round(protein, 1),
            "carbs_g": round(carbs, 1),
            "fat_g": round(fat, 1),
            "fiber_g": round(fiber, 1),
        }

    async def estimate_exercise_met(self, exercise_name: str) -> dict:
        """
        Estimate a MET value + measurement unit + category for an exercise NOT
        found locally or in the external sources (final fallback tier).
        Uses the Cloudflare LLM if configured, else a rule-based heuristic.

        Returns: {"exercise_name_display", "met", "category", "measurement_unit"}
        """
        ex_clean = (exercise_name or "").strip()
        if not ex_clean:
            return {}

        if self.account_id and self.api_token:
            prompt = (
                f"Estimate the physical activity intensity for the exercise: '{ex_clean}'.\n"
                "Respond ONLY with valid JSON in this exact structure:\n"
                "{\n"
                '  "exercise_name_display": "Clean display name",\n'
                '  "met": 6.0,\n'
                '  "category": "Cardio|Strength|Yoga|Sports|Fitness",\n'
                '  "measurement_unit": "reps|minutes"\n'
                "}\n"
                "Use the 2024 Compendium of Physical Activities MET conventions. "
                "Use 'reps' for counted movements (push-ups, squats), else 'minutes'."
            )
            try:
                payload = {
                    "messages": [
                        {"role": "system", "content": "You are an exercise physiologist. Respond only with valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 150,
                    "temperature": 0.1,
                }
                headers = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(self.api_url, json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = ""
                    if "result" in data:
                        raw_text = data["result"].get("response", "")
                    elif "choices" in data:
                        raw_text = data["choices"][0].get("message", {}).get("content", "")
                    raw_text = raw_text.strip()
                    if raw_text.startswith("```"):
                        raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text[3:]
                        if raw_text.endswith("```"):
                            raw_text = raw_text[:-3]
                        raw_text = raw_text.strip()
                    parsed = json.loads(raw_text)
                    if parsed.get("met"):
                        return parsed
            except Exception:
                pass  # fall through to heuristic

        return self._heuristic_exercise_estimator(ex_clean)

    def _resolve_exercise_noun(self, message: str) -> Optional[str]:
        """
        Data-driven check: does `message` name a known exercise/activity?

        Consults (a) the embedded 2024 Compendium activity table and (b) the
        local MongoDB `exercises` collection — NO hardcoded per-exercise list.
        Requires an activity signal (a duration/amount OR an activity verb) so
        plain food sentences are never misrouted. Returns the resolved exercise
        query string (name only) or None.
        """
        m = (message or "").lower().strip()
        if not m:
            return None

        # Activity signal: an activity verb, a number, OR an exercise metric
        # unit (reps/sets/km/kg/…). These metric units are unambiguous workout
        # signals — a food is never measured in reps/sets/km/kg.
        _activity_verbs = (
            "did", "do", "done", "play", "played", "playing", "perform", "performed",
            "kari", "karya", "karyu", "kiya", "kiye", "kita", "kiti", "marya", "maryu",
            "mari", "khelya", "khel", "ramyo", "ramya", "ramyu",  # play (gu/hi)
        )
        _exercise_metric_units = (
            "minute", "minutes", "min", "mins", "hour", "hours", "hr", "hrs",
            "rep", "reps", "round", "rounds", "set", "sets", "km", "kms",
            "kilometer", "kilometre", "lap", "laps", "step", "steps",
            "pushup", "pushups", "push-up", "push-ups",
            "squat", "squats", "crunch", "crunches", "lunge", "lunges",
            "burpee", "burpees", "situp", "situps", "pullup", "pullups",
        )
        has_metric = any(re.search(r"\b" + re.escape(u) + r"\b", m) for u in _exercise_metric_units)
        has_verb = any(re.search(r"\b" + re.escape(v) + r"\b", m) for v in _activity_verbs)
        # An exercise signal requires an exercise metric unit or an activity verb.
        # Digits alone are NOT an exercise signal (digits are standard in food portions like "1 plate", "100g").
        has_signal = has_metric or has_verb

        # Reduce to the candidate noun by stripping numbers, units and noise.
        try:
            from exercise_calculator import _clean_exercise_name, parse_exercise_input
            parsed = parse_exercise_input(m)
            candidate = (parsed.exercise_query or _clean_exercise_name(m)).strip()
        except Exception:
            candidate = m
        if not candidate or len(candidate) < 2:
            return None

        # Disallow standard food portion units and containers from being treated as candidate exercise nouns
        _FOOD_PORTION_UNITS = {
            "plate", "plates", "bowl", "bowls", "cup", "cups", "glass", "glasses",
            "slice", "slices", "piece", "pieces", "serving", "servings", "katori",
            "katoris", "vatki", "g", "gm", "gms", "gram", "grams", "kg", "ml", "spoon",
            "spoons", "tbsp", "tsp", "tablespoon", "tablespoons", "teaspoon", "teaspoons",
            "rotlo", "rotla", "roti", "rotis", "thepla", "theplas", "bhakri", "bhakhri",
            "portion", "portions", "packet", "packets",
        }
        if candidate in _FOOD_PORTION_UNITS:
            return None

        # (a) Compendium activity table (offline, authoritative for MET names).
        # A CONFIDENT exact activity name is itself a strong exercise signal, so a
        # bare workout term ("bench press", "skipping") routes correctly even
        # without a number/verb. A weak/fuzzy hit still needs an explicit signal
        # so ordinary food words are never misrouted.
        try:
            from external_exercise_api import _compendium_lookup, COMPENDIUM_MET
            comp = _compendium_lookup(candidate)
            if comp:
                canonical = comp[0]
                is_exact = candidate in COMPENDIUM_MET or candidate == canonical
                if is_exact or has_signal:
                    return candidate
        except Exception:
            pass

        # (b) Local exercises collection — exact/alias/fuzzy via ExerciseSearcher.
        try:
            from database import sync_db
            from exercise_calculator import ExerciseSearcher
            ex, conf = ExerciseSearcher(sync_db).search(candidate)
            if ex:
                # An exact DB match is a strong signal on its own; a fuzzy match
                # still requires an explicit activity signal.
                if conf >= 0.95 or (conf >= 0.7 and has_signal):
                    return candidate
        except Exception:
            pass

        return None

    def _heuristic_exercise_estimator(self, exercise_name: str) -> dict:
        """Rule-based MET estimator for exercises outside all data sources."""
        low = exercise_name.lower()
        # Counted-movement keywords → reps, else duration-based.
        rep_words = ("push", "pull", "squat", "lunge", "curl", "press", "crunch",
                     "sit-up", "situp", "raise", "dip", "row", "burpee", "namaskar")
        is_reps = any(w in low for w in rep_words)
        # Rough intensity buckets.
        if any(w in low for w in ("run", "sprint", "hiit", "jump", "skip", "box", "spin")):
            met = 9.0
            category = "Cardio"
        elif any(w in low for w in ("yoga", "stretch", "medit", "pranayam", "pilates")):
            met = 3.0
            category = "Yoga"
        elif any(w in low for w in ("cricket", "football", "tennis", "sport", "badminton", "basketball", "hockey")):
            met = 6.5
            category = "Sports"
        elif is_reps:
            met = 5.5
            category = "Strength"
        else:
            met = 5.0
            category = "Fitness"
        return {
            "exercise_name_display": exercise_name.title(),
            "met": met,
            "category": category,
            "measurement_unit": "reps" if is_reps else "minutes",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_json_response(self, raw_text: str, original_message: str) -> LLMParseResult:
        """Parse the LLM's JSON response into LLMParseResult."""
        try:
            # Strip markdown code blocks if present
            text = raw_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            data = json.loads(text)
            # Normalise date — LLM may return Gujarati/Hindi words instead of "yesterday"/"today"
            raw_date = data.get("date")
            if raw_date in ("kale", "kal", "gay kale", "gay kal"):
                raw_date = "yesterday"
            elif raw_date in ("aaje", "aaj"):
                raw_date = "today"

            def _num(v):
                """Coerce a JSON value into a float, or None."""
                if v is None or v == "":
                    return None
                try:
                    return float(v)
                except (TypeError, ValueError):
                    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
                    return float(m.group()) if m else None

            # Multi-food array (each item its own dict). Fall back to the single
            # food_query/quantity/variant/meal_type when the model didn't supply one.
            foods = data.get("foods")
            if not isinstance(foods, list):
                foods = []
            foods = [f for f in foods if isinstance(f, dict) and (f.get("food_query"))]
            if not foods and data.get("food_query"):
                foods = [{
                    "food_query": data.get("food_query"),
                    "quantity": data.get("quantity"),
                    "variant": data.get("variant"),
                    "meal_type": data.get("meal_type"),
                }]

            exercise_query = data.get("exercise_query")
            exercise_input = data.get("exercise_input")
            # If the model gave a structured exercise but no phrase, synthesize one
            # so downstream (which reads exercise_input) still works.
            if not exercise_input and exercise_query:
                parts = []
                if data.get("sets") and data.get("reps"):
                    parts.append(f"{int(_num(data.get('sets')) or 0)} sets of {int(_num(data.get('reps')) or 0)}")
                elif _num(data.get("reps")) is not None:
                    parts.append(f"{int(_num(data.get('reps')))} reps")
                elif _num(data.get("duration_min")) is not None:
                    parts.append(f"{int(_num(data.get('duration_min')))} minutes")
                elif _num(data.get("distance")) is not None:
                    parts.append(f"{_num(data.get('distance')):g} {data.get('distance_unit') or 'km'}")
                exercise_input = (f"{exercise_query} " + " ".join(parts)).strip() if parts else exercise_query

            return LLMParseResult(
                intent=data.get("intent", Intent.UNKNOWN),
                food_query=data.get("food_query"),
                quantity=data.get("quantity"),
                variant=data.get("variant"),
                meal_type=data.get("meal_type"),
                exercise_input=exercise_input,
                date=raw_date,
                clarification_question=data.get("clarification_question"),
                raw_response=raw_text,
                success=True,
                # structured entities
                foods=foods,
                reps=_num(data.get("reps")),
                sets=_num(data.get("sets")),
                distance=_num(data.get("distance")),
                distance_unit=data.get("distance_unit"),
                duration_min=_num(data.get("duration_min")),
                intensity=data.get("intensity"),
                exercise_query=exercise_query,
                time_of_day=data.get("time_of_day"),
                missing_detail=data.get("missing_detail"),
                exact_term=data.get("exact_term"),
            )
        except (json.JSONDecodeError, KeyError):
            # If LLM returns non-JSON, fallback
            return self._fallback_parse(original_message)

    def _fallback_parse(self, message: str) -> LLMParseResult:
        """
        Rule-based fallback parser when LLM is unavailable.
        Handles basic patterns in English/Hindi/Gujarati.
        """
        msg = message.lower().strip()

        # Greetings
        greetings = {"hi", "hello", "hey", "namaste", "namaskar", "kem cho", "su haal"}
        if msg in greetings or any(msg.startswith(g + " ") for g in greetings) or msg in greetings:
            return LLMParseResult(intent=Intent.GREETING, raw_response="fallback", success=True)

        # Junk / gibberish / off-topic → clarification_needed (before food path)
        # In the fallback (no live LLM), we're conservative: treat both
        # "clearly non-food" (True) AND "uncertain" (None) short single-token
        # inputs as needing clarification, since guessing is worse than asking.
        try:
            nf = self._looks_like_non_food(msg)
            if nf is True:
                return LLMParseResult(
                    intent=Intent.CLARIFICATION_NEEDED,
                    clarification_question=(
                        "I couldn't identify a food or exercise. What would you like to log?"
                    ),
                    raw_response="fallback",
                    success=True,
                )
            # Single token that is ONLY letters a-z with no known food/exercise
            # signal, and the token itself is 6+ chars entirely in one character-
            # class run (e.g. "abcdef", "qwerty", "asdfgh") → gibberish.
            tokens = msg.split()
            if len(tokens) == 1 and len(msg) >= 4:
                tok = tokens[0]
                # Keyboard-row / alphabetical sequences
                _keyboard_rows = {
                    "qwerty","qwert","werty","qwertyui","asdfgh","sdfgh","zxcvbn",
                    "abcdef","abcde","bcdef","abcd","efghi","mnopq",
                }
                if tok in _keyboard_rows:
                    return LLMParseResult(
                        intent=Intent.CLARIFICATION_NEEDED,
                        clarification_question=(
                            "I couldn't identify a food or exercise. What would you like to log?"
                        ),
                        raw_response="fallback",
                        success=True,
                    )
                # All same character repeated (aaaa, bbbbb)
                if len(set(tok)) == 1:
                    return LLMParseResult(
                        intent=Intent.CLARIFICATION_NEEDED,
                        clarification_question=(
                            "I couldn't identify a food or exercise. What would you like to log?"
                        ),
                        raw_response="fallback",
                        success=True,
                    )
        except Exception:
            pass

        # Profile keywords
        profile_kw = ["my profile", "mera profile", "profile dikhao", "my stats", "my details"]
        if any(kw in msg for kw in profile_kw):
            return LLMParseResult(intent=Intent.GET_PROFILE, raw_response="fallback", success=True)

        # Recommend-workout keywords (check before summary so "workout plan" doesn't bleed)
        _workout_rec_kw = [
            "recommend", "suggest a workout", "suggest workout", "workout suggest",
            "what should i do tomorrow", "what exercise tomorrow", "exercise tomorrow",
            "kal kya karna", "kal kya exercise", "workout plan", "training plan",
            "aavti kal exercise", "kal exercise suchav", "mane workout", "workout suchav",
            "next workout", "workout for tomorrow", "exercise for tomorrow",
        ]
        if any(kw in msg for kw in _workout_rec_kw):
            return LLMParseResult(intent=Intent.RECOMMEND_WORKOUT, raw_response="fallback", success=True)

        # Summary keywords
        summary_kw = ["summary", "aaj ka", "today summary", "dikhao", "report",
                      "kitna khaya", "show today", "kal ka summary", "yesterday summary",
                      "gatkale nu summary", "gatkal no", "kal ki report",
                      "items today", "items dikhao", "items aaj"]
        if any(kw in msg for kw in summary_kw):
            # Check if a specific meal is mentioned → query_meal not get_summary
            _meal_in_msg = None
            for _m, _words in [("breakfast", ["breakfast","nashta","savare","subah","morning"]),
                                ("lunch",     ["lunch","dopahar","bapore"]),
                                ("dinner",    ["dinner","raat","ratre"]),
                                ("snack",     ["snack","sanje","shaam","nasta"])]:
                if any(w in msg for w in _words):
                    _meal_in_msg = _m
                    break
            _sum_date = "yesterday" if any(
                w in msg for w in ["yesterday","kal","kale","gatkale","gatkal"]
            ) else "today"
            if _meal_in_msg and any(cue in msg for cue in ["items", "su khadhu", "kya khaya", "dikhao"]):
                return LLMParseResult(intent=Intent.QUERY_MEAL, meal_type=_meal_in_msg,
                                      date=_sum_date, raw_response="fallback", success=True)
            return LLMParseResult(intent=Intent.GET_SUMMARY, date=_sum_date,
                                  raw_response="fallback", success=True)

        # Exercise-read query BEFORE calorie check — "how many calories did I burn"
        # is an exercise history read, not a food calorie lookup.
        burn_kw = ["burn", "burned", "burnt", "calories burned", "how much did i burn",
                   "how many calories did i burn", "calories i burned"]
        if any(kw in msg for kw in burn_kw):
            return LLMParseResult(
                intent=Intent.QUERY_EXERCISE,
                date="yesterday" if any(w in msg for w in ["yesterday","kale","kal"]) else "today",
                raw_response="fallback", success=True,
            )

        # ── Daily-total calorie reads → get_summary (not get_calories) ──────────
        # "total calories today", "how many calories did I eat today", etc.
        # Must be checked BEFORE the nutrition-lookup block.
        _personal_total_kw = [
            "total calories today", "total calories yesterday",
            "calories today", "calories yesterday",
            "how many calories did i eat", "how much did i eat",
            "calories consumed", "calorie count today", "calorie count yesterday",
            "aaj kitni calories", "aaje ketli calorie",
            "gatkale ketli calorie", "gatkal ketli calorie",
            "gatkale no total", "gatkal no total",
            "kitna khaya aaj", "kitna khaya aaje",
        ]
        if any(kw in msg for kw in _personal_total_kw):
            _tot_date = "yesterday" if any(
                w in msg for w in ["yesterday", "kal", "kale", "gatkale", "gatkal"]
            ) else "today"
            return LLMParseResult(intent=Intent.GET_SUMMARY, date=_tot_date,
                                  raw_response="fallback", success=True)

        # Calories / macro query — route to get_calories when asking about a
        # specific food's nutrition content.
        _macro_kw = ["calorie", "calories", "kcal",
                     "protein", "carb", "carbs", "carbohydrate",
                     "fat", "fats", "fiber", "fibre",
                     "macro", "macros", "nutrition",
                     "kitni calories", "kitna protein", "kitna fat",
                     "kitne carbs", "ketli calorie", "ketlu protein"]
        if any(ck in msg for ck in _macro_kw):
            # Guard: if the message is clearly an eating statement ("had 1 cup of
            # coffee with sugar", "ate 150g almonds") rather than a nutrition
            # question, let it fall through to the food-logging path.
            _eating_verbs = ("had ", "ate ", "drank ", "eat ", "drink ",
                             "khadhu", "khadhi", "khadha", "khadho", "pidhi", "piya")
            _eating_stmt = (msg.split()[0] in ("had","ate","drank","eat","drink")
                            or any(v in msg for v in _eating_verbs))
            _question_markers = ["how much", "how many", "kitni", "kitna", "ketli",
                                  "ketlu", "calories in", "protein in", "fat in",
                                  "carbs in", "what is the", "nutrition of",
                                  "me kitni", "me kitna", "hoti hai", "hota hai"]
            _is_question = any(q in msg for q in _question_markers)
            if _eating_stmt and not _is_question:
                pass  # eating statement — fall through to food logging
            else:
                # Use targeted regex to extract (food, quantity)
                import re as _nre
                food_part = None
                qty_part  = None

                # Strip leading question words so they don't bleed into food name
                food_msg = _nre.sub(
                    r"^(?:how\s+(?:much|many)\s+(?:calories?|protein|fat|carbs?|"
                    r"carbohydrates?|fiber|fibre|macros?|nutrition|kcal)\s*"
                    r"(?:is\s+there\s+in|does\s+.+?\s+have|are\s+there\s+in|"
                    r"in|of|for|hoti?\s+hai|hota?\s+hai)?\s*)",
                    "", msg, flags=_nre.I
                ).strip()
                # Strip trailing question fragments
                food_msg = _nre.sub(
                    r"\s*(?:me\s+)?kit(?:ni|na|lu)\s*(?:calorie|calories?|protein|fat|"
                    r"carbs?|fiber|fibre|nutrition|macro)?(?:\s+hoti?\s+hai)?$",
                    "", food_msg, flags=_nre.I
                ).strip()
                food_msg = _nre.sub(r"^content\s+of\s+", "", food_msg, flags=_nre.I).strip()

                if food_msg:
                    # Leading quantity: "100g paneer", "1 tablespoon ghee"
                    qty_match = _nre.match(
                        r"^(\d+(?:\.\d+)?\s*(?:g|gm|ml|kg|pieces?|piece|tbsp|"
                        r"tablespoons?|tsp|teaspoons?|bowls?|cups?|plates?|"
                        r"glasses?|katoris?|servings?|slices?))\s+(.+)$",
                        food_msg, _nre.I)
                    if qty_match:
                        qty_part  = qty_match.group(1).strip()
                        food_part = qty_match.group(2).strip()
                    else:
                        # Trailing quantity: "biryani 1 plate", "dal 1 katori"
                        trailing = _nre.search(
                            r"\s+(\d+(?:\.\d+)?\s*(?:g|gm|ml|kg|pieces?|piece|tbsp|"
                            r"tablespoons?|tsp|teaspoons?|bowls?|cups?|plates?|"
                            r"glasses?|katoris?|servings?|slices?))\s*$",
                            food_msg, _nre.I)
                        if trailing:
                            qty_part  = trailing.group(1).strip()
                            food_part = food_msg[:trailing.start()].strip()
                        else:
                            food_part = food_msg

                if food_part:
                    return LLMParseResult(
                        intent=Intent.GET_CALORIES,
                        food_query=food_part,
                        quantity=qty_part or "100g",
                        raw_response="fallback",
                        success=True,
                    )

        # Exercise keywords (with common misspellings and variations)
        exercise_names = [
            # English
            "jogging", "joging", "jog",
            "running", "run", "runnig",
            "walking", "walk", "walkig",
            "cycling", "cycle",
            "yoga",
            "push-up", "pushup", "push up", "pushups", "push-ups",
            "squat", "squats",
            "swimming", "swim",
            "exercise", "excersise", "excerise", "excercise",
            "workout", "work out", "workut",
            "gym",
            "plank", "planks",
            "burpee", "burpees",
            "deadlift", "deadlifts",
            "zumba",
            "boxing",
            "surya namaskar", "suryanamaskar",
            "jumping jack", "jumping jacks",
            "skipping", "skip",
            "stretching", "stretch",
            "lunges", "lunge",
            "crunches", "crunch",
            "sit-up", "situp", "sit ups",
            "pull up", "pull-up", "pullup", "pull ups",
            "treadmill",
            "hiit", "h.i.i.t",   # added HIIT
            "bench press", "shoulder press", "bicep curl", "lat pulldown",
            "mountain climber", "jumping rope", "rope jumping",
            # Hindi Romanised exercise terms
            "vyayam",         # व्यायाम
            "kasrat",         # कसरत
            "kasarat",
            "daud",           # दौड़
            "dauding",
            "paidhal",        # पैदल (walking)
            "tairana",        # तैरना (swimming)
            "saikil",         # साइकिल (cycling)
            # Gujarati Romanised exercise terms
            "kasarat",        # કસરત
            "daudvun",
            "chalvun",        # walking
            "taravun",        # swimming
            "saikal",         # cycling
            # Punjabi Romanised exercise terms
            "kasrat",
            "daudna",         # ਦੌੜਨਾ
            "turdna",         # ਤੁਰਨਾ (walking)
            "bhagna",         # running
            "tahelna",        # ਟਹਿਲਣਾ (walking)
            "tairna",         # swimming
            # Post-normalisation: normalise_transcript() converts native script
            # to these Roman forms before the LLM / fallback sees the text
        ]
        time_kw = ["minute", "min", "hour", "rep", "reps", "round", "set", "km", "kilometer"]
        # Hindi/Gujarati/Punjabi past-tense verbs and noise words to strip when
        # building exercise_input. These appear after the exercise name and amount
        # in Indian-language sentences: "50 pushups marya" → "50 pushups"
        hindi_noise = [
            # Hindi past tense
            "kiya", "kiye", "kia",
            # Gujarati past tense
            "kari", "karyu", "karya", "kar",
            "marya", "maryu", "maryo", "mari",   # ← THE MISSING ONES ("marya" = did/performed)
            "kara", "kare",
            "karela", "kareli", "karelu",
            "lagavya", "lagavyu", "lagaya",
            # Punjabi past tense
            "kita", "kiti", "kite",
            # Filler prepositions / connectives
            "ki", "me", "ne", "hu",
            "mein", "main",
            "ka", "ke",
            "ne", "da", "di", "de", "nu", "te",  # Punjabi particles
            # Date / time context words (stripped since they don't help search)
            "aaj", "aaje", "kal", "kale", "subah", "savare",
            "aj",                                 # Punjabi today
            "maine", "mene",
            # English filler / pronouns / temporal (fixes "today i did 20 push-ups")
            "i", "we", "you", "my", "did", "do", "done", "doing",
            "have", "had", "today", "yesterday", "morning", "evening",
            "night", "the", "a", "an", "was", "were", "just", "some", "and",
        ]

        # --- SKIP MEAL (check BEFORE exercise so "I skipped breakfast" doesn't
        #     match the exercise keyword "skip") ---
        _skip_early_kw = ["skipped", "skip meal", "skip kiya", "skip karyu",
                          "nathi karyu", "nathi khadhu", "nathi lidhu",
                          "nahi khaya", "nathi jamya"]
        if any(kw in msg for kw in _skip_early_kw):
            _skip_meal_val = None
            if any(w in msg for w in ["breakfast", "breckfast", "nashta", "savare"]):
                _skip_meal_val = "breakfast"
            elif any(w in msg for w in ["lunch", "dopahar", "bapore"]):
                _skip_meal_val = "lunch"
            elif any(w in msg for w in ["dinner", "raat", "sanje"]):
                _skip_meal_val = "dinner"
            elif any(w in msg for w in ["snack", "nasta"]):
                _skip_meal_val = "snack"
            return LLMParseResult(
                intent=Intent.SKIP_MEAL,
                meal_type=_skip_meal_val,
                date="today",
                raw_response="fallback",
                success=True,
            )

        # --- EXERCISE QUERY: "aaje me kya exercise kari?", "workout su karyu?" ---
        # Normalise common exercise word misspellings before checking query keywords,
        # so "excercise kari", "excersise kari" etc. all match "exercise kari".
        _q_msg = re.sub(r"\bexcercise\b", "exercise", msg, flags=re.IGNORECASE)
        _q_msg = re.sub(r"\bexcersise\b", "exercise", _q_msg, flags=re.IGNORECASE)
        _q_msg = re.sub(r"\bexcerise\b",  "exercise", _q_msg, flags=re.IGNORECASE)
        _q_msg = re.sub(r"\bworkut\b",    "workout",  _q_msg, flags=re.IGNORECASE)

        # Only if NO duration/amount is present (otherwise it's a LOG intent)
        exercise_query_kw = [
            # Gujarati
            "kya exercise", "kya workout", "exercise kari",
            "workout karyu", "workout su", "exercise su",
            "kya kya exercise", "ketli exercise", "kitni exercise",
            "ky ky exercise", "ky ky workout",
            "su su exercise", "su su workout",
            "gym gayu", "yoga karyu", "yoga kari",
            "excercise ma su", "exercise ma su", "excersise ma su",
            "su karu", "su karyu", "kya karu", "kya kiya",
            # Hindi
            "kya kya exercise", "kya kya workout",
            "kaun si exercise", "kaun sa workout",
            "kitni exercise", "kitna exercise",
            "exercise ki", "workout ki",
            # English — "show my X", "what X", "list X", "my X today"
            "what exercise", "what workout", "how much exercise",
            "which exercise", "which exercises",
            "show my workout", "show my exercise", "show my training",
            "show workout", "show exercise", "show training",
            "list my workout", "list my exercise", "list workouts",
            "my workout today", "my exercise today", "my training today",
            "today workout", "today exercise", "today training",
            "workout log", "exercise log", "training log",
            "what did i do at the gym", "what physical activity",
            "ketla km", "kitne km", "how many km",
            # Punjabi
            "ki exercise", "ki workout", "exercise kiti",
            "workout kita", "gym gaya", "yoga kita",
        ]
        has_duration = any(tk in _q_msg for tk in time_kw) or any(c.isdigit() for c in _q_msg)
        if any(kw in _q_msg for kw in exercise_query_kw) and not has_duration:
            return LLMParseResult(
                intent=Intent.QUERY_EXERCISE,
                date="yesterday" if any(w in msg for w in ["kale", "kal", "yesterday"]) else "today",
                raw_response="fallback",
                success=True,
            )

        # --- EXERCISE LOGGING ---
        # Pre-normalise the message through the same typo map ExerciseSearcher uses,
        # so misspellings like "puchups", "squot", "cyclin" all match exercise_names.
        _exercise_typo_map = {
            "runing": "running",     "runnig": "running",
            "joging": "jogging",     "joggin": "jogging",
            "walkng": "walking",     "walkin": "walking",     "walkig": "walking",
            "cyclng": "cycling",     "cyclin": "cycling",     "cycle":  "cycling",
            "saikal": "cycling",     "saikil": "cycling",
            "yga": "yoga",           "yog": "yoga",           "yoag": "yoga",
            "puchups": "pushups",    "pushup": "pushups",
            "push up": "pushups",    "push ups": "pushups",   "push-ups": "pushups",
            "psuhups": "pushups",    "puhsups": "pushups",    "pusups": "pushups",
            "pushap": "pushups",     "pushaps": "pushups",
            "squot": "squats",       "squat": "squats",       "squott": "squats",
            "squuts": "squats",      "sqots": "squats",       "sqats": "squats",
            "swiming": "swimming",   "swimig": "swimming",
            "plak": "plank",         "planks": "plank",
            "crunch": "crunches",    "cruches": "crunches",
            "situp": "sit-ups",      "sit up": "sit-ups",     "situps": "sit-ups",
            "pullup": "pull-ups",    "pull up": "pull-ups",   "pullups": "pull-ups",
            "burpee": "burpees",     "burpes": "burpees",
            "lunge": "lunges",
            "workut": "workout",     "work out": "workout",
            "skiping": "skipping",
            # Indian romanised
            "kasrat": "exercise",    "kasarat": "exercise",
            "daud": "running",       "daudna": "running",
            # Past-tense conjugations → canonical exercise names
            "cycled": "cycling",     "ran": "running",
            "swam": "swimming",      "walked": "walking",
            "jogged": "jogging",     "swum": "swimming",
        }
        # Apply multi-word entries first (longest first), then single-word
        msg_normalised = msg
        for wrong, right in sorted(_exercise_typo_map.items(), key=lambda x: -len(x[0])):
            msg_normalised = re.sub(
                r"\b" + re.escape(wrong) + r"\b", right, msg_normalised, flags=re.IGNORECASE
            )

        matched_exercise = None
        for ex in exercise_names:
            if ex in msg_normalised:
                matched_exercise = ex
                break

        # ── Data-driven exercise detection (no hardcoded names) ──────────
        # If the keyword list missed it, consult the exercise dataset + the
        # Compendium activity table. This lets external activities (kabaddi,
        # cricket, elliptical, pranayama, etc.) route to LOG_EXERCISE instead of
        # being mis-parsed as food (e.g. "minutes" fuzzy-matching "Minute Maid").
        if not matched_exercise:
            resolved_ex = self._resolve_exercise_noun(msg_normalised)
            if resolved_ex:
                matched_exercise = resolved_ex

        if matched_exercise:
            # Build clean exercise input using the normalised message
            exercise_input = msg_normalised.strip()
            clean_words = []
            for word in exercise_input.lower().split():
                if word not in hindi_noise and word not in ["ka", "ki", "ke", "nu", "ni"]:
                    clean_words.append(word)
            exercise_input = " ".join(clean_words)

            # Check if it has a time/rep/distance component
            has_amount = any(tk in exercise_input for tk in time_kw) or any(c.isdigit() for c in exercise_input)
            if not has_amount:
                exercise_input = matched_exercise

            # Parse structured exercise fields from exercise_input so downstream
            # (ExerciseCalculator, A-to-Z grader) receives duration/reps/distance.
            ex_q = matched_exercise
            dur_min = None
            reps = None
            sets = None
            dist = None
            dist_unit = None
            missing = "amount"

            # ── Direct regex extraction (always runs first, no DB needed) ──
            # Sets × reps: "3 sets of 12", "3x12", "4 sets 10 reps"
            _sr = re.search(
                r"(\d+)\s*(?:sets?\s+(?:of\s+)?|x\s*)(\d+)(?:\s*reps?)?",
                exercise_input, re.I)
            if _sr:
                sets, reps, missing = float(_sr.group(1)), float(_sr.group(2)), None
            # Duration: "30 min", "45 minutes", "1 hour", "1 ghanta"
            if dur_min is None:
                _dm = re.search(
                    r"(\d+(?:\.\d+)?)\s*(?:min(?:utes?)?|minute|hrs?|hour|ghanta|ghante)",
                    exercise_input, re.I)
                if _dm:
                    v = float(_dm.group(1))
                    # convert hours/ghanta
                    if re.search(r"h(?:our|r)", _dm.group(0), re.I) or "ghanta" in _dm.group(0).lower():
                        v *= 60
                    dur_min, missing = v, None
            # Distance: "10 km", "5.5 kms", "1000 m", "3 miles"
            if dist is None:
                _dd = re.search(
                    r"(\d+(?:\.\d+)?)\s*(km|kms|kilometer|kilometres?|m\b|meters?|metres?|miles?)",
                    exercise_input, re.I)
                if _dd:
                    dist = float(_dd.group(1))
                    u = _dd.group(2).lower()
                    dist_unit = "km" if u.startswith("k") else ("m" if u in ("m","meters","metres") else "miles")
                    missing = None
            # Bare rep count (after sets/duration/distance are resolved):
            # "50 push-ups", "100 jumping jacks", "aaje 50 push-ups"
            if reps is None and dur_min is None and dist is None:
                _rc = re.search(r"(\d+)", exercise_input)
                if _rc:
                    reps, missing = float(_rc.group(1)), None

            # ── parse_exercise_input for exercise_query name normalisation ──
            try:
                from exercise_calculator import parse_exercise_input as _pex
                ep = _pex(exercise_input)
                if ep.exercise_query:
                    ex_q = ep.exercise_query
            except Exception:
                pass

            return LLMParseResult(
                intent=Intent.LOG_EXERCISE,
                exact_term=matched_exercise,
                exercise_query=ex_q,
                exercise_input=exercise_input,
                duration_min=dur_min,
                reps=reps,
                sets=sets,
                distance=dist,
                distance_unit=dist_unit,
                date="today",
                missing_detail=missing,
                raw_response="fallback",
                success=True,
            )

        # Food logging (default assumption for most messages)
        # --- MEAL QUERY: "aaje breakfast ma su lidhu?", "what did I eat for lunch?" ---
        meal_query_kw = ["su lidhu", "su khadhu", "su jamya", "su khayu",
                         "kya khaya", "what did i eat", "what i ate", "su idhu",
                         "su hattu", "show breakfast", "show lunch", "show dinner",
                         "breakfast ma su", "lunch ma su", "dinner ma su",
                         "what did i have", "what have i eaten",
                         "yesterday what", "what yesterday"]
        if any(kw in msg for kw in meal_query_kw):
            # Detect which meal they're asking about
            qmeal = None
            if any(w in msg for w in ["breakfast", "breckfast", "brecfast", "brekfast", "breakfst", "nashta", "savare", "savaar", "savar"]):
                qmeal = "breakfast"
            elif any(w in msg for w in ["lunch", "dopahar", "bhojan", "bapore"]):
                qmeal = "lunch"
            elif any(w in msg for w in ["dinner", "raat", "sanje", "ratre"]):
                qmeal = "dinner"
            elif any(w in msg for w in ["snack", "nasta"]):
                qmeal = "snack"
            # Detect date: "kale" = yesterday, "aaje" = today
            qdate = "today"
            if any(w in msg for w in ["kale", "kal", "yesterday"]):
                qdate = "yesterday"
            return LLMParseResult(
                intent=Intent.QUERY_MEAL,
                meal_type=qmeal,
                date=qdate,
                raw_response="fallback",
                success=True,
            )

        # --- SKIP MEAL: "breakfast nathi karyu", "lunch skip kiya" ---
        skip_kw = ["nathi karyu", "nathi khadhu", "nathi lidhu", "skip kiya",
                   "skip karyu", "nahi khaya", "skipped", "nathi jamya"]
        if any(kw in msg for kw in skip_kw):
            skip_meal = None
            if any(w in msg for w in ["breakfast", "breckfast", "nashta", "savare"]):
                skip_meal = "breakfast"
            elif any(w in msg for w in ["lunch", "dopahar", "bapore"]):
                skip_meal = "lunch"
            elif any(w in msg for w in ["dinner", "raat", "sanje"]):
                skip_meal = "dinner"
            elif any(w in msg for w in ["snack", "nasta"]):
                skip_meal = "snack"
            return LLMParseResult(
                intent=Intent.SKIP_MEAL,
                meal_type=skip_meal,
                date="today",
                raw_response="fallback",
                success=True,
            )

        # Look for meal type (includes common misspellings + Gujarati/Hindi variants)
        meal = None
        if any(w in msg for w in ["breakfast", "breckfast", "brekfast", "breakfst", "nashta",
                                   "subah", "subhe", "morning",
                                   "savare", "savre", "savaar", "savar", "savarne"]):
            meal = "breakfast"
        elif any(w in msg for w in ["lunch", "dopahar", "dopaher", "bhojan",
                                     "bapore", "baporme", "bapori", "bpoore", "bpore"]):
            meal = "lunch"
        elif any(w in msg for w in ["dinner", "raat", "khana", "ratre", "rate",
                                     "dinr", "dinnr"]):
            meal = "dinner"
        elif any(w in msg for w in ["snack", "nasta", "nasto", "timepass",
                                     "tea time", "teatime"]):
            meal = "snack"
        # "evening" / "sanje" / "shaam" → snack (NOT dinner)
        elif any(w in msg for w in ["evening", "sanje", "sanjeye", "shaam", "sham"]):
            meal = "snack"

        # Relative post-meal modifiers → snack regardless of meal detected above
        # "after dinner", "bapor pachhi", "after lunch", "raat ne pehla"
        _after_patterns = ["after dinner", "after lunch", "after breakfast",
                           "bapor pachhi", "raat ne pehla", "lunch ke baad",
                           "dinner ke baad", "post lunch", "post dinner"]
        if any(p in msg for p in _after_patterns):
            meal = "snack"

        # Variant detection
        variant = None
        for v in ["ghee", "fried", "boiled", "grilled", "steamed", "oil"]:
            if v in msg:
                variant = v
                break

        # If we have some food-like content, assume log_food
        if len(msg) > 2:
            quantity, food_query = _extract_quantity_and_food(message, meal, variant)
            fq = food_query or message.strip()
            # Determine missing_detail
            has_qty = bool(quantity)
            has_meal = bool(meal)
            if not has_qty and not has_meal:
                missing = "quantity"
            elif not has_qty:
                missing = "quantity"
            elif not has_meal:
                missing = "meal_type"
            else:
                missing = None
            # Always populate foods[] so downstream and graders see the list
            foods_list = [{"food_query": fq, "quantity": quantity,
                           "variant": variant, "meal_type": meal}]
            # yesterday detection
            date_val = "yesterday" if any(w in msg for w in ["yesterday","kale","kal","gatkale","gatkal"]) else "today"
            return LLMParseResult(
                intent=Intent.LOG_FOOD,
                exact_term=fq,
                food_query=fq,
                quantity=quantity,
                variant=variant,
                meal_type=meal,
                foods=foods_list,
                date=date_val,
                missing_detail=missing,
                raw_response="fallback",
                success=True,
            )

        return LLMParseResult(
            intent=Intent.UNKNOWN,
            raw_response="fallback",
            success=True,
            clarification_question="I didn't understand. Please tell me what food you ate or exercise you did.",
        )


# ---------------------------------------------------------------------------
# Quantity + food extraction helper (used by the fallback parser)
# ---------------------------------------------------------------------------

import re as _re

# Matches "150g", "200 ml", "2 pieces", "1 bowl", "1.5 cups", "3 katori", etc.
_QTY_PATTERN = _re.compile(
    r"\b(\d+(?:\.\d+)?)\s*"
    r"(g|gm|gms|gram|grams|ml|mls|"
    r"pieces?|pcs?|nos?|"
    r"bowl(?:s)?|katori(?:s)?|cup(?:s)?|plate(?:s)?|glass(?:es)?|serving(?:s)?|"
    r"slice(?:s)?|tbsp|tsp|tablespoon(?:s)?|teaspoon(?:s)?|"
    r"bowel(?:s)?)"   # "bowel" = common typo for "bowl"
    r"\b",
    _re.IGNORECASE,
)

# "a bowl of …", "a cup of …", "a glass of …" — indefinite article + container
_INDEF_QTY_PATTERN = _re.compile(
    r"\ba\s+(bowl|cup|glass|plate|katori|serving|slice|tablespoon|tbsp|teaspoon|tsp)\b",
    _re.IGNORECASE,
)
# Food-named units: "2 parathas", "3 rotis", "4 idlis" etc.
# The unit word IS the food name — so we extract the food from the match itself.
_FOOD_UNIT_PATTERN = _re.compile(
    r"\b(\d+(?:\.\d+)?)\s*"
    r"(parathas?|rotis?|rotlis?|chapattis?|chapatis?|"
    r"idlis?|dosas?|puris?|theplas?|bhakris?|bhakhris?|samosas?)\b",
    _re.IGNORECASE,
)

# Words to strip from the food name once consumed as metadata
_MEAL_WORDS = {
    "breakfast", "breckfast", "brekfast", "breakfst", "nashta", "subah", "morning",
    "savare", "savaar", "savar",
    "lunch", "dopahar", "bhojan", "afternoon", "bapore", "bpoore", "bpore",
    "dinner", "raat", "evening", "ratre", "sanje", "dinr", "dinnr",
    "snack", "snacks",
    "nasta", "nasto", "timepass",
    "tea time", "teatime",              # "2 samosa at tea time" — tea time = snack context
    "aaje", "aaj",
    # Gujarati script meal & time words
    "આજે", "સવારે", "બપોરે", "સાંજે", "રાત્રે", "નાસ્તો", "સવાર", "બપોર",
}
_VARIANT_WORDS = {"ghee", "fried", "boiled", "grilled", "steamed", "oil", "butter", "normal", "plain"}
_FILLER_WORDS = {
    "i", "had", "ate", "eaten", "have", "with", "me", "mein", "ma",
    "maine", "mene", "khaya", "khayi", "khaayi", "khadhu", "khadhi", "lidhu", "lidhi",
    "jamyu", "jamya", "jamyo", "khai", "khayi",
    # drinking verbs — Gujarati/Hindi
    "pidhu", "pidhi", "pidha", "piyu", "piyi", "piya", "piye",
    "pi", "peyu", "peyi", "peya",
    # pronouns / filler particles / noise tokens
    "n", "ne", "hu", "karyu", "kari", "karya", "kar", "gayu", "gaya",
    "aaj", "today", "aur", "and", "a", "an", "the", "of", "for", "my",
    # Hindi/Gujarati particles that never form part of a food name
    "ko", "ka", "ki", "ke", "nu", "ni", "na", "no",
    "se", "par", "pe", "tak",
    "ek", "do", "teen", "char",    # number words already parsed as quantity
    # Past-tense eating/drinking verbs (Gujarati) — never part of a food name
    "khadha", "khadhi", "khadhu", "khadho",
    "lidha", "lidhi", "lidhu",
    "jamyo", "jamya", "jamyu",
    "at",       # English: "at tea time", "at dinner"
    "after", "before", "around", # relative time words
    "tea", "time",  # so "at tea time" → stripped → no food residue
    # Gujarati script fillers & verbs
    "મેં", "ખાધી", "ખાધું", "ખાધા", "પીધી", "પીધું", "જમ્યા", "જમ્યો", "અને", "સાથે", "હું", "હતું", "છે",
}


def _extract_quantity_and_food(message: str, meal: Optional[str], variant: Optional[str]):
    """
    Pull a quantity phrase out of the raw message and return
    (quantity_string_or_None, cleaned_food_name).

    Examples:
      "150g butter chicken dinner" -> ("150g", "butter chicken")
      "2 rotis lunch"              -> ("2 rotis", "roti")
      "ate 2 parathas with ghee"   -> ("2 pieces", "paratha")
      "paneer"                     -> (None, "paneer")
    """
    text = message.strip()

    # Clean punctuation noise, preserving Indic script marks
    text_clean = _re.sub(r"[^\w\s\u0900-\u0d7f]", " ", text)
    text_clean = _re.sub(r"\s+", " ", text_clean).strip()

    # ── Step 1: extract quantity ────────────────────────────────────────────
    quantity = None
    food_from_unit = None  # food name recovered from "2 parathas", "3 rotis" etc.

    # First check food-named units ("2 parathas" → qty="2 pieces", food="paratha")
    food_unit_match = _FOOD_UNIT_PATTERN.search(text_clean)
    if food_unit_match:
        num   = food_unit_match.group(1)
        unit  = food_unit_match.group(2).lower()
        quantity = f"{num} pieces"
        # Singularise the unit word to get the canonical food name
        _SINGULARS = {
            "parathas": "paratha", "paratha": "paratha",
            "rotis": "roti", "roti": "roti",
            "rotlis": "roti", "rotli": "roti",
            "chapattis": "chapati", "chapatti": "chapati",
            "chapatis": "chapati", "chapati": "chapati",
            "idlis": "idli", "idli": "idli",
            "dosas": "dosa", "dosa": "dosa",
            "puris": "puri", "puri": "puri",
            "theplas": "thepla", "thepla": "thepla",
            "bhakris": "bhakri", "bhakri": "bhakri",
            "bhakhris": "bhakri", "bhakhri": "bhakri",
            "samosas": "samosa", "samosa": "samosa",
        }
        food_from_unit = _SINGULARS.get(unit, unit)
        # Remove the matched food-unit from text_clean so it doesn't pollute residue
        text_clean = (text_clean[:food_unit_match.start()] +
                      " " + text_clean[food_unit_match.end():])
    else:
        match = _QTY_PATTERN.search(text_clean)
        if match:
            quantity = match.group(0).strip()
            text_clean = text_clean[:match.start()] + " " + text_clean[match.end():]
        elif match_orig := _QTY_PATTERN.search(text):
            quantity = match_orig.group(0).strip()

    # Bare number with no unit ("2 idli", "morning ma 2 bhakri") → count in pieces
    if quantity is None:
        bare = _re.search(r"\b(\d+(?:\.\d+)?)\b", text_clean)
        if bare:
            quantity = f"{bare.group(1)} pieces"
            text_clean = text_clean[:bare.start()] + " " + text_clean[bare.end():]

    # "a bowl of …", "a glass of …" — indefinite article container
    if quantity is None:
        indef = _INDEF_QTY_PATTERN.search(text_clean)
        if indef:
            quantity = f"1 {indef.group(1).lower()}"
            text_clean = text_clean[:indef.start()] + " " + text_clean[indef.end():]

    # Also look for a trailing qty phrase AFTER the food name in the original
    # text, e.g. "calories in biryani 1 plate" → qty="1 plate"
    if quantity is None or quantity == "100g":
        trailing_qty = _QTY_PATTERN.search(text)
        if trailing_qty:
            quantity = trailing_qty.group(0).strip()

    # ── Step 2: remove filler / meal / connector words ─────────────────────
    tokens = [t for t in _re.split(r"\s+", text_clean) if t]
    cleaned = []
    for tok in tokens:
        low = tok.lower().strip(".,!?")
        if not low or (len(low) == 1 and low not in ("g", "m")):
            continue
        if low in _MEAL_WORDS or low in _FILLER_WORDS:
            continue
        cleaned.append(tok)

    food_query = " ".join(cleaned).strip()

    # ── Step 3: strip trailing variant words that follow a real food noun ──
    # "paratha with ghee" → "paratha"  (ghee is the variant, not the food)
    # "roti oil"          → "roti"
    # But "butter chicken" / "ghee rice" stay intact (prefix variant = part of name)
    food_words = food_query.lower().split()
    if len(food_words) >= 2:
        last = food_words[-1]
        # Only strip the LAST token if it's a standalone variant word AND the
        # remaining food name is non-empty (so "ghee" alone is not stripped).
        if last in _VARIANT_WORDS and len(food_words) >= 2:
            food_query = " ".join(food_query.split()[:-1]).strip()
        # Also strip "with <variant>" or "sathe <variant>"
        food_query = _re.sub(
            r"\s+(?:with|sathe|ke saath|ke sath)\s+(" +
            "|".join(_VARIANT_WORDS) + r")\s*$",
            "", food_query, flags=_re.IGNORECASE
        ).strip()

    # ── Step 4: apply food-name normalizations ─────────────────────────────
    # If we recovered a food name from the quantity match ("2 parathas" →
    # food_from_unit="paratha"), prefer it when:
    #   a) the residue is completely empty, OR
    #   b) the residue consists only of variant words (e.g. just "ghee"),
    #   c) food_from_unit was set AND a multi-food connective ("aur"/"ane")
    #      was present so residue is a DIFFERENT food — keep food_from_unit
    #      as the primary food (the other food is handled by multi-food logic).
    if food_from_unit:
        residue_lower = food_query.lower().strip()
        residue_only_variant = residue_lower in _VARIANT_WORDS or not residue_lower
        if residue_only_variant:
            food_query = food_from_unit
        else:
            # Multi-food: "3 roti aur sabzi" → primary = food_from_unit ("roti")
            # The residue is a different food ("sabzi"), not a modifier.
            # Keep food_from_unit as primary for the single food_query field.
            # (The foods[] array would list both, but the fallback only handles primary.)
            _connectives = {"aur", "ane", "and", "sathe", "or", "tatha", "pan", "bhi"}
            msg_lower = message.lower()
            if any(c in msg_lower for c in _connectives):
                food_query = food_from_unit

    # Plural → singular for common Indian foods
    _PLURAL_MAP = {
        "parathas": "paratha", "rotis": "roti", "rotlis": "roti",
        "chapattis": "chapati", "chapatis": "chapati",
        "idlis": "idli", "dosas": "dosa", "puris": "puri",
        "samosas": "samosa", "bhakris": "bhakri",
        "theplas": "thepla", "kachoris": "kachori",
    }
    # Typo / phonetic corrections for common items
    _FOOD_TYPO_MAP = {
        "samoosa": "samosa", "samosha": "samosa", "samosaa": "samosa",
        "thaepla": "thepla",  "theepla": "thepla",
        "methi thaepla": "methi thepla",
        "dahi wada": "dahi vada", "dahi wadas": "dahi vada",
        "idlee": "idli",  "iddli": "idli",
        "dosha": "dosa",  "dhosa": "dosa",
        "rajmah": "rajma", "rajmaah": "rajma",
        "paneeer": "paneer", "panier": "paneer",
        "biriyani": "biryani", "briyani": "biryani", "briyaani": "biryani",
        "chhole": "chole", "chhola": "chole",
    }
    # Hindi→English food name normalization
    _HI_FOOD_MAP = {
        "anda": "egg",  "ande": "egg",   "ande khaaye": "egg",
        "doodh": "milk", "dudh": "milk",
        "chawal": "rice", "chaawal": "rice",
        "sabzi": "mixed vegetable", "sabji": "mixed vegetable",
        "daal": "dal",
        "murgi": "chicken", "murghi": "chicken",
        "machli": "fish", "maach": "fish",
        "gosht": "mutton", "maas": "mutton",
        "roti": "roti",  # already canonical but keep for mapping pass
    }
    fq_lower = food_query.lower()
    # Apply plural → singular
    for wrong, right in _PLURAL_MAP.items():
        if fq_lower == wrong or fq_lower.endswith(" " + wrong):
            food_query = food_query[:-len(wrong)] + right
            fq_lower = food_query.lower()
    # Apply multi-word typo map (longest first)
    for wrong, right in sorted(_FOOD_TYPO_MAP.items(), key=lambda x: -len(x[0])):
        if wrong in fq_lower:
            food_query = _re.sub(_re.escape(wrong), right, food_query, flags=_re.IGNORECASE)
            fq_lower = food_query.lower()
    # Apply Hindi→English normalization (exact or trailing word)
    for hi, en in sorted(_HI_FOOD_MAP.items(), key=lambda x: -len(x[0])):
        if fq_lower == hi:
            food_query = en
            fq_lower = food_query.lower()
            break
        if fq_lower.endswith(" " + hi):
            food_query = food_query[:-len(hi)] + en
            fq_lower = food_query.lower()

    # Normalise common quantity typos before returning
    if quantity:
        quantity = _re.sub(r"\bbowel(?:s)?\b", "bowl", quantity, flags=_re.I)

    return quantity, food_query.strip()
