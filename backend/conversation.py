"""
Conversation Module — Memory + Multi-turn Flow Manager
Combines conversation history persistence and the clarification state machine.

Exports:
  - ConversationStore       (message history per user)
  - ConversationFlowManager (multi-turn flow state)
  - PendingFlow, FlowState  (flow dataclasses)
  - detect_language, get_response  (multilingual helpers)
  - VARIANT_WORDS, YES_WORDS, NO_WORDS, MEAL_WORDS  (word sets)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase


# ============================================================================
# CONVERSATION MEMORY
# ============================================================================

MAX_HISTORY = 10


class ConversationStore:
    """Manages conversation history per user in MongoDB."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db["conversation_messages"]

    async def add_message(self, user_id: str, role: str, content: str, intent: Optional[str] = None):
        doc = {
            "user_id": user_id,
            "role": role,
            "content": content,
            "intent": intent,
            "created_at": datetime.now(timezone.utc),
        }
        await self.collection.insert_one(doc)
        await self._prune(user_id)

    async def get_history(self, user_id: str, limit: int = MAX_HISTORY) -> list[dict]:
        cursor = self.collection.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        docs.reverse()
        return [{"role": d["role"], "content": d["content"], "intent": d.get("intent")} for d in docs]

    async def get_context_string(self, user_id: str, limit: int = 5) -> str:
        history = await self.get_history(user_id, limit=limit)
        if not history:
            return ""
        parts = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Bot"
            parts.append(f"{role}: {msg['content'][:100]}")
        return " | ".join(parts)

    async def get_last_intent(self, user_id: str) -> Optional[str]:
        doc = await self.collection.find_one(
            {"user_id": user_id, "role": "assistant"},
            sort=[("created_at", -1)]
        )
        return doc.get("intent") if doc else None

    async def get_last_exercise(self, user_id: str) -> Optional[str]:
        """Return the exercise_name from the most-recent log_exercise assistant message.

        Used by the connector-word fallback ("pn me 6 set marya" = also did 6
        more of the previous exercise) so follow-up sentences don't require the
        user to repeat the exercise name.
        """
        import re as _re
        # Walk recent assistant messages looking for a log_exercise intent
        cursor = self.collection.find(
            {"user_id": user_id, "role": "assistant"},
            sort=[("created_at", -1)]
        ).limit(10)
        async for doc in cursor:
            intent = doc.get("intent", "")
            if "log_exercise" not in intent and "exercise" not in intent:
                continue
            content = doc.get("content", "")
            # Response template: "💪 **Push-ups** — **30 reps** = …" or similar
            m = _re.search(r"\*\*(.+?)\*\*", content)
            if m:
                name = m.group(1).strip().lower()
                # Skip quantity phrases like "30 reps"
                if not _re.search(r"^\d", name):
                    return name
        return None

    async def get_last_meal(self, user_id: str) -> Optional[str]:
        """Return the meal type from the most-recent food log assistant message.

        Used to pre-populate shared_meal_type for the next food log so the bot
        doesn't ask meal type again when the user logs several foods in a row
        within the same session.  Returns None when no recent meal is found or
        when more than 30 minutes have passed (stale context).
        """
        from datetime import datetime, timezone, timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
        cursor = self.collection.find(
            {"user_id": user_id, "role": "assistant",
             "created_at": {"$gte": cutoff}},
            sort=[("created_at", -1)]
        ).limit(10)
        async for doc in cursor:
            intent = doc.get("intent", "")
            if "log_food" not in intent and "food_logged" not in intent:
                continue
            content = doc.get("content", "")
            # Response templates mention the meal name in the log line, e.g.
            # "Logged 1 roti for **Breakfast**" or "Breakfast ma logged"
            import re as _re
            for meal in ("breakfast", "lunch", "dinner", "snack", "morning",
                         "evening", "afternoon"):
                if meal in content.lower():
                    return meal
        return None

    async def clear_history(self, user_id: str):
        await self.collection.delete_many({"user_id": user_id})

    async def _prune(self, user_id: str):
        count = await self.collection.count_documents({"user_id": user_id})
        if count > MAX_HISTORY:
            excess = count - MAX_HISTORY
            oldest = self.collection.find({"user_id": user_id}).sort("created_at", 1).limit(excess)
            ids = [d["_id"] async for d in oldest]
            if ids:
                await self.collection.delete_many({"_id": {"$in": ids}})


# ============================================================================
# FLOW STATES
# ============================================================================

class FlowState:
    IDLE = "idle"
    AWAITING_VARIANT = "awaiting_variant"
    AWAITING_QUANTITY = "awaiting_quantity"
    AWAITING_MEAL_TYPE = "awaiting_meal_type"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    AWAITING_EXERCISE_CONFIRM = "awaiting_exercise_confirm"
    AWAITING_EXERCISE_AMOUNT = "awaiting_exercise_amount"  # ask "ketla pushups?"
    AWAITING_MULTI_CHOICE = "awaiting_multi_choice"   # Save Both / Save X / Save Y
    AWAITING_GENERIC_RESOLUTION = "awaiting_generic_resolution"


@dataclass
class PendingFlow:
    """State of an ongoing multi-turn flow — handles both single and multi-item sessions."""
    state: str = FlowState.IDLE
    # Current item being clarified
    food_query: Optional[str] = None
    food_id: Optional[str] = None
    food_name: Optional[str] = None
    food_name_display: Optional[str] = None
    quantity: Optional[str] = None
    variant: Optional[str] = None
    meal_type: Optional[str] = None
    date: Optional[str] = None
    # Exercise
    exercise_input: Optional[str] = None
    exercise_data: Optional[dict] = None
    # ── Source & AI estimated support ─────────────────────────────────────
    source: str = "Local Database"
    is_ai_estimated: Optional[bool] = False
    ai_nutrition: Optional[dict] = None
    # ── Multi-item sequential queue support ────────────────────────────────
    items: list = None                  # list of food item dicts being processed sequentially
    current_index: int = 0              # current index in items queue
    shared_meal_type: Optional[str] = None # shared meal slot for all items
    pending_items: list = None          # list of pending_action dicts
    remaining_items: list = None        # list of raw item strings
    unresolved_items: list = None       # raw item strings that could not be resolved

    def to_json(self) -> str:
        import datetime as _dt
        d = self.__dict__.copy()
        if d.get("items") is None:
            d["items"] = []
        if d.get("pending_items") is None:
            d["pending_items"] = []
        if d.get("remaining_items") is None:
            d["remaining_items"] = []
        if d.get("unresolved_items") is None:
            d["unresolved_items"] = []

        def _clean(obj):
            """Recursively strip datetime objects and other non-serializable types."""
            if isinstance(obj, (_dt.datetime, _dt.date)):
                return obj.isoformat()
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(i) for i in obj]
            return obj

        return json.dumps(_clean(d))

    @classmethod
    def from_json(cls, data: str) -> "PendingFlow":
        try:
            d = json.loads(data)
            # Handle fields added after old records were saved
            d.setdefault("items", [])
            d.setdefault("current_index", 0)
            d.setdefault("shared_meal_type", None)
            d.setdefault("pending_items", [])
            d.setdefault("remaining_items", [])
            d.setdefault("unresolved_items", [])
            d.setdefault("is_ai_estimated", False)
            d.setdefault("ai_nutrition", None)
            d.setdefault("source", "Local Database")
            return cls(**d)
        except (json.JSONDecodeError, TypeError):
            return cls()


# ============================================================================
# MULTILINGUAL RESPONSES
# ============================================================================

RESPONSES = {
    "ask_variant": {
        "gu": "{food_emoji} **{food}** — kevi rite (preparation) banavelu?",
        "hi": "{food_emoji} **{food}** — kaisa preparation tha?",
        "en": "{food_emoji} Which preparation variant for **{food}**?",
    },
    "ask_quantity": {
        "gu": "{food_emoji} **{food}** ketli khadhi? (e.g. 2 piece, 1 bowl, 150g)",
        "hi": "{food_emoji} **{food}** kitni khaayi? (e.g. 2 piece, 1 bowl, 150g)",
        "en": "{food_emoji} How much **{food}** did you have? (e.g. 2 pieces, 1 bowl, 150g)",
    },
    "ask_quantity_drink": {
        "gu": "{food_emoji} **{food}** ketli pidhi? (e.g. 1 cup, 1 glass, 200ml)",
        "hi": "{food_emoji} **{food}** kitni piyi? (e.g. 1 cup, 1 glass, 200ml)",
        "en": "{food_emoji} How much **{food}** did you drink? (e.g. 1 cup, 1 glass, 200ml)",
    },
    "ask_meal_type": {
        "gu": "Tame {food_emoji} **{meal_context}** 🍳 **Savare**, 🍱 **Bapore**, 🌇 **Sanje** ke 🍽️ **Raatre** jamya?",
        "hi": "Ye {food_emoji} **{meal_context}** — 🍳 **Breakfast**, 🍱 **Lunch**, 🌇 **Shaam** ya 🍽️ **Dinner** me?",
        "en": "Was this {food_emoji} **{meal_context}** for 🍳 **Breakfast**, 🍱 **Lunch**, 🌇 **Evening**, or 🍽️ **Dinner**?",
    },
    "confirm_food": {
        "gu": "Tame **{qty}** {food_emoji} **{food}**{variant_text} khadhi. Approx **{calories:.0f} kcal** (**Protein**: **{protein:.1f}g** | **Carbs**: **{carbs:.1f}g** | **Fat**: **{fat:.1f}g**).\nShu aa tamara {meal_emoji} **{meal}** daily log ma save karu?",
        "hi": "Aapne **{qty}** {food_emoji} **{food}**{variant_text} khaayi. Approx **{calories:.0f} kcal** (**Protein**: **{protein:.1f}g** | **Carbs**: **{carbs:.1f}g** | **Fat**: **{fat:.1f}g**).\nKya ye aapke {meal_emoji} **{meal}** me save karu?",
        "en": "You had **{qty}** {food_emoji} **{food}**{variant_text}. Approx **{calories:.0f} kcal** (**Protein**: **{protein:.1f}g** | **Carbs**: **{carbs:.1f}g** | **Fat**: **{fat:.1f}g**).\nShould I save this to your {meal_emoji} **{meal}** log?",
    },
    "confirm_food_drink": {
        "gu": "Tame **{qty}** {food_emoji} **{food}**{variant_text} pidhi. Approx **{calories:.0f} kcal** (**Protein**: **{protein:.1f}g** | **Carbs**: **{carbs:.1f}g** | **Fat**: **{fat:.1f}g**).\nShu aa tamara {meal_emoji} **{meal}** daily log ma save karu?",
        "hi": "Aapne **{qty}** {food_emoji} **{food}**{variant_text} piyi. Approx **{calories:.0f} kcal** (**Protein**: **{protein:.1f}g** | **Carbs**: **{carbs:.1f}g** | **Fat**: **{fat:.1f}g**).\nKya ye aapke {meal_emoji} **{meal}** me save karu?",
        "en": "You had **{qty}** {food_emoji} **{food}**{variant_text}. Approx **{calories:.0f} kcal** (**Protein**: **{protein:.1f}g** | **Carbs**: **{carbs:.1f}g** | **Fat**: **{fat:.1f}g**).\nShould I save this to your {meal_emoji} **{meal}** log?",
    },
    "confirm_food_ai": {
        "gu": "{food_emoji} **{food}** — **{qty}** = ~**{calories:.0f} kcal** (**Protein**: **{protein:.1f}g** | **Carbs**: **{carbs:.1f}g** | **Fat**: **{fat:.1f}g**).\nShu aa tamara {meal_emoji} **{meal}** log ma save karu?",
        "hi": "{food_emoji} **{food}** — **{qty}** = ~**{calories:.0f} kcal** (**Protein**: **{protein:.1f}g** | **Carbs**: **{carbs:.1f}g** | **Fat**: **{fat:.1f}g**).\nKya ise aapke {meal_emoji} **{meal}** me save karein?",
        "en": "{food_emoji} **{food}** — **{qty}** = ~**{calories:.0f} kcal** (**Protein**: **{protein:.1f}g** | **Carbs**: **{carbs:.1f}g** | **Fat**: **{fat:.1f}g**).\nShould I save this to your {meal_emoji} **{meal}** log?",
    },
    "confirm_exercise": {
        "gu": "{exercise_emoji} **{exercise}** — **{amount:.0f} {unit}** = approx **{calories:.0f} kcal burned**.\nSave karu?",
        "hi": "{exercise_emoji} **{exercise}** — **{amount:.0f} {unit}** = approx **{calories:.0f} kcal burned**.\nSave karu?",
        "en": "{exercise_emoji} **{exercise}** — **{amount:.0f} {unit}** = approx **{calories:.0f} kcal burned**.\nShould I save this?",
    },
    "ask_exercise_amount": {
        "gu": "Ketla **{unit}** {exercise_emoji} **{exercise}** kara? (e.g. 15, 30, 50)",
        "hi": "Kitne **{unit}** {exercise_emoji} **{exercise}** kiye? (e.g. 15, 30, 50)",
        "en": "How many **{unit}** of {exercise_emoji} **{exercise}** did you do? (e.g. 15, 30, 50)",
    },
    "food_logged": {
        "gu": "Done! {food_emoji} **{food}** (**{calories:.0f} kcal**) tamara {meal_emoji} **{meal}** ma save thai gai.",
        "hi": "Done! {food_emoji} **{food}** (**{calories:.0f} kcal**) aapke {meal_emoji} **{meal}** me save ho gai.",
        "en": "Done! {food_emoji} **{food}** (**{calories:.0f} kcal**) saved to your {meal_emoji} **{meal}**.",
    },
    "exercise_logged": {
        "gu": "Done! {exercise_emoji} **{exercise}** (**{calories:.0f} kcal burned**) log thai gayu.",
        "hi": "Done! {exercise_emoji} **{exercise}** (**{calories:.0f} kcal burned**) log ho gaya.",
        "en": "Done! {exercise_emoji} **{exercise}** (**{calories:.0f} kcal burned**) logged.",
    },
    "greeting": {
        "gu": "Kem cho {name}! Tamaru daily calorie target **{target:.0f} kcal** chhe.\nAaje su khadhu ke koi exercise kari? Mane jnavo!",
        "hi": "Namaste {name}! Aapka daily calorie target **{target:.0f} kcal** hai.\nAaj kya khaya ya koi exercise ki? Batayein!",
        "en": "Hi {name}! Your daily calorie target is **{target:.0f} kcal**.\nTell me what you ate or exercised today!",
    },
    "summary": {
        "gu": "Tamaru aaj nu summary ({date}):\n- Khadhu: **{consumed:.0f} kcal** ({food_count} items)\n- Burn: **{burned:.0f} kcal** ({ex_count} exercises)\n- Net: **{net:.0f} kcal**\n- Baki: **{remaining:.0f} kcal** (target: **{target:.0f}**)\n- **Protein**: **{protein:.0f}g** | **Carbs**: **{carbs:.0f}g** | **Fat**: **{fat:.0f}g**",
        "hi": "Aapka aaj ka summary ({date}):\n- Khaya: **{consumed:.0f} kcal** ({food_count} items)\n- Burn: **{burned:.0f} kcal** ({ex_count} exercises)\n- Net: **{net:.0f} kcal**\n- Baaki: **{remaining:.0f} kcal** (target: **{target:.0f}**)\n- **Protein**: **{protein:.0f}g** | **Carbs**: **{carbs:.0f}g** | **Fat**: **{fat:.0f}g**",
        "en": "Your summary for {date}:\n- Eaten: **{consumed:.0f} kcal** ({food_count} items)\n- Burned: **{burned:.0f} kcal** ({ex_count} exercises)\n- Net: **{net:.0f} kcal**\n- Remaining: **{remaining:.0f} kcal** (target: **{target:.0f}**)\n- **Protein**: **{protein:.0f}g** | **Carbs**: **{carbs:.0f}g** | **Fat**: **{fat:.0f}g**",
    },
    "safety_disclaimer": {
        "gu": "\n(Note: Aa approximate values chhe. Medical advice nathi.)",
        "hi": "\n(Note: Ye approximate values hain. Medical advice nahi hai.)",
        "en": "\n(Note: These are estimated values, not medical advice.)",
    },
    "what_food": {
        "gu": "Su khadhu? Food nu naam aane quantity jnavo.",
        "hi": "Kya khaya? Food ka naam aur quantity batayein.",
        "en": "What did you eat? Tell me the food name and quantity.",
    },
    "food_not_found": {
        "gu": "'{food}' nathi maltu. Bijo naam ke spelling try karo.",
        "hi": "'{food}' nahi mila. Dusra naam ya spelling try karein.",
        "en": "Sorry, couldn't find '{food}'. Try a different name or spelling.",
    },
    "not_food": {
        "gu": "Aa message ma mane koi food madyu nahi. Kaya food nu log karvu chhe te jnavo.",
        "hi": "Is message me mujhe koi food nahi mila. Kaunsa food log karna hai bataye.",
        "en": "I couldn't identify a food in that message. Tell me the food you'd like to log.",
    },
    "cancelled": {
        "gu": "Okay, cancel kari didhu. Biju koi help joiye?",
        "hi": "Okay, cancel kar diya. Aur koi help chahiye?",
        "en": "Okay, cancelled. Anything else I can help with?",
    },
}

# Word sets used by the flow manager
VARIANT_WORDS = {"ghee", "normal", "fried", "boiled", "grilled", "steamed", "baked",
                 "roasted", "oil", "butter", "plain", "deep fried", "raw",
                 "ghee vali", "ghee wali", "oil vali", "tel vali", "saadi", "normal vali"}

YES_WORDS = {"yes", "haa", "ha", "haan", "ok", "okay", "sure", "save karo", "save kar",
             "log karo", "log kar", "confirm", "done", "ho", "saru", "thik", "save", "save all", "save all items",
             # localised button labels
             "haa, save karo", "haan, save karo", "yes, save", "save all",
             "haa, add karo", "haan, add kijiye", "yes, add it",
             "haan, add karo", "સાચવો", "હા, સાચવો",
             # save both / save item
             "badha save karo", "dono save karo", "save both", "badhu save karo",
             }

NO_WORDS = {"no", "nahi", "na", "nako", "cancel", "nai", "nope", "rehen de", "chhodo",
            # localised button labels
            "na, cancel", "nahi, cancel", "no, cancel",
            "na, bas", "nahi, bas", "no, that's all",
            "na, skip karyu", "nahi, skip karo", "no, skip",
            "skip karyu", "skip karo",
            }

MEAL_WORDS = {
    "breakfast": {
        "breakfast", "nashta", "subah", "morning", "nash", "savaar",
        # Gujarati
        "savare", "savre", "savar", "savaar", "savarne",
        # Hindi
        "subhe", "subah",
        # button labels
        "savare (breakfast)", "subah (breakfast)",
        # common misspellings
        "breckfast", "brekfast",
    },
    "lunch": {
        "lunch", "dopahar", "bhojan", "bapore", "bpoore",
        # Gujarati
        "bapore", "baporme", "bapori",
        # Hindi
        "dopahar", "dopaher", "dpahar",
        # button labels
        "bapore (lunch)", "dopahar (lunch)",
    },
    # snack before dinner — "evening snack" can map to evening/snack
    "snack": {
        "snack", "nasta", "timepass",
        # Gujarati
        "nasto", "નાસ્તો",
        "evening snack",
    },
    "dinner": {
        "dinner", "raat", "raatre", "ratre", "rate", "supper",
        "evening", "sanje", "sanj", "sanjeye", "shaam", "sham", "saanj",
        # Gujarati
        "રાત", "રાત્રે", "સાંજ", "સાંજે",
        # Hindi
        "रात", "रात्रि", "शाम",
        # button labels
        "raat (dinner)", "raatre (dinner)", "sanje (dinner)", "sanje (evening)",
    },
}


# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

def detect_language(text: str) -> str:
    """
    Detect language from text — supports all major Indian languages.
    Returns language code used for response templates.
    For languages without dedicated templates, falls back to English.

    Supported scripts:
      Gujarati (0A80-0AFF), Devanagari/Hindi/Marathi (0900-097F),
      Bengali (0980-09FF), Tamil (0B80-0BFF), Telugu (0C00-0C7F),
      Kannada (0C80-0CFF), Malayalam (0D00-0D7F), Gurmukhi/Punjabi (0A00-0A7F),
      Odia (0B00-0B7F), Urdu/Arabic (0600-06FF)
    """
    # Script-based detection
    if any("\u0A80" <= c <= "\u0AFF" for c in text):
        return "gu"  # Gujarati script
    if any("\u0900" <= c <= "\u097F" for c in text):
        return "hi"  # Devanagari (Hindi, Marathi, Sanskrit, Nepali)
    if any("\u0980" <= c <= "\u09FF" for c in text):
        return "en"  # Bengali script → respond in English
    if any("\u0B80" <= c <= "\u0BFF" for c in text):
        return "en"  # Tamil script → respond in English
    if any("\u0C00" <= c <= "\u0C7F" for c in text):
        return "en"  # Telugu script → respond in English
    if any("\u0C80" <= c <= "\u0CFF" for c in text):
        return "en"  # Kannada script → respond in English
    if any("\u0D00" <= c <= "\u0D7F" for c in text):
        return "en"  # Malayalam script → respond in English
    if any("\u0A00" <= c <= "\u0A7F" for c in text):
        return "en"  # Gurmukhi/Punjabi script → respond in English
    if any("\u0B00" <= c <= "\u0B7F" for c in text):
        return "en"  # Odia script → respond in English
    if any("\u0600" <= c <= "\u06FF" for c in text):
        return "en"  # Urdu/Arabic script → respond in English

    # Romanized keyword detection
    lower = text.lower()
    gu_kw = [
        # greetings / questions
        "kem cho", "su khadhu", "tamara", "mane", "saru", "bapore",
        "aaje", "savare", "gay kale",
        # eating verbs
        "khadhi", "khadhu", "khadha", "khadho", "lidhu", "lidhi", "lidha",
        "jamya", "jamyu", "jamyo",
        # drinking verbs
        "pidhu", "pidhi", "pidha", "piyu", "piyi",
        # quantity words
        "ketli", "ketlu", "ketlo", "ketla",
        # other Gujarati markers
        "chhu", "chhe", "nathi", "hatu", "hati", "hatun", "rotlo", "rotla",
    ]
    hi_kw = [
        "kya khaya", "kitni", "khayi", "maine", "dikhao", "batao",
        "khaya", "kiya", "khaayi",
        # Hindi food words that won't appear in English context
        "subah", "dopahar",
    ]
    if any(kw in lower for kw in gu_kw):
        return "gu"
    if any(kw in lower for kw in hi_kw):
        return "hi"
    return "en"


def get_response(key: str, lang: str, **kwargs) -> str:
    """Get a response template in the appropriate language with auto-resolved emojis."""
    from food_emoji import get_food_emoji, get_meal_emoji, get_exercise_emoji

    # Automatically resolve emojis if not provided
    if "food" in kwargs and "food_emoji" not in kwargs:
        kwargs["food_emoji"] = get_food_emoji(str(kwargs.get("food", "")))
    if "meal" in kwargs:
        raw_meal = str(kwargs.get("meal", "") or "snack")
        if "meal_emoji" not in kwargs:
            kwargs["meal_emoji"] = get_meal_emoji(raw_meal)
        kwargs["meal"] = raw_meal.replace("_", " ").title()
    if "exercise" in kwargs and "exercise_emoji" not in kwargs:
        kwargs["exercise_emoji"] = get_exercise_emoji(str(kwargs.get("exercise", "")))
    if "meal_context" in kwargs and "food_emoji" not in kwargs:
        kwargs["food_emoji"] = get_food_emoji(str(kwargs.get("meal_context", "")))

    templates = RESPONSES.get(key, {})
    template = templates.get(lang, templates.get("en", ""))

    # Contextual Gujarati gender agreement for foods
    if key == "ask_quantity" and lang == "gu":
        food_str = str(kwargs.get("food", "")).lower()
        if any(w in food_str for w in ["rotlo", "rotla"]):
            template = "{food_emoji} **{food}** ketlo khadho? (e.g. 1 rotlo, 2 rotla, 3 rotla)"

    try:
        return template.format(**kwargs)
    except (KeyError, IndexError):
        return templates.get("en", "").format(**kwargs) if templates.get("en") else str(kwargs)


# ── Localised button labels ───────────────────────────────────────────────

_OPTS = {
    # confirmation buttons
    "yes_save": {"gu": "Haa, Save karo",    "hi": "Haan, Save karo",   "en": "Yes, Save"},
    "no_cancel": {"gu": "Na, Cancel",        "hi": "Nahi, Cancel",      "en": "No, Cancel"},
    # query-meal no-records buttons
    "yes_add":  {"gu": "Haa, add karo",      "hi": "Haan, add kijiye",  "en": "Yes, add it"},
    "no_skip":  {"gu": "Na, skip karyu",     "hi": "Nahi, skip karo",   "en": "No, skip"},
    # duplicate-meal add-more buttons
    "yes_add_more": {"gu": "Haa, add karo",  "hi": "Haan, add karo",    "en": "Yes, add it"},
    "no_bas":   {"gu": "Na, bas",            "hi": "Nahi, bas",         "en": "No, that's all"},
    # save-both for multi-food
    "save_both": {"gu": "Badha Save karo",   "hi": "Dono Save karo",    "en": "Save Both"},
    "cancel":    {"gu": "Cancel",            "hi": "Cancel",            "en": "Cancel"},
    # meal type options
    "meal_breakfast": {"gu": "🍳 Savare (Breakfast)", "hi": "🍳 Subah (Breakfast)", "en": "🍳 Breakfast"},
    "meal_lunch":     {"gu": "🍱 Bapore (Lunch)",     "hi": "🍱 Dopahar (Lunch)",   "en": "🍱 Lunch"},
    "meal_evening":   {"gu": "🌇 Sanje (Evening)",    "hi": "🌇 Shaam (Evening)",   "en": "🌇 Evening"},
    "meal_dinner":    {"gu": "🍽️ Raatre (Dinner)",    "hi": "🍽️ Raat (Dinner)",     "en": "🍽️ Dinner"},
    "meal_snack":     {"gu": "☕ Snack",               "hi": "☕ Snack",             "en": "☕ Snack"},
}


def get_option(key: str, lang: str) -> str:
    """Return a localised button label."""
    row = _OPTS.get(key, {})
    return row.get(lang, row.get("en", key))


def is_word_in_text(words_set: set, text: str) -> bool:
    """Check if any word/phrase in words_set appears as a distinct word in text."""
    text_lower = text.lower()
    words_in_text = set(re.findall(r"\b[\w\',]+\b", text_lower))
    for w in words_set:
        if " " in w:
            if w in text_lower:
                return True
        else:
            if w in words_in_text:
                return True
    return False


# ============================================================================
# FLOW MANAGER
# ============================================================================

class ConversationFlowManager:
    """Manages multi-turn conversation flows using dedicated MongoDB collection 'flow_states'."""

    def __init__(self, db, user_id: str):
        self.db = db
        self.user_id = user_id
        self.collection = db["flow_states"]

    async def get_pending_flow(self) -> PendingFlow:
        doc = await self.collection.find_one({"user_id": self.user_id}, sort=[("updated_at", -1)])
        if not doc:
            return PendingFlow()

        try:
            state_data = doc.get("state_data", "{}")
            return PendingFlow.from_json(state_data)
        except Exception:
            return PendingFlow()

    async def save_flow_state(self, flow: PendingFlow):
        # Always clear existing flow state for this user
        await self.collection.delete_many({"user_id": self.user_id})

        if flow.state == FlowState.IDLE:
            return

        # Save new flow state
        await self.collection.insert_one({
            "user_id": self.user_id,
            "state_data": flow.to_json(),
            "updated_at": datetime.now(timezone.utc),
        })

    def interpret_user_response(self, message: str, flow: PendingFlow) -> PendingFlow:
        """Update flow state based on user's response."""
        msg_lower = message.lower().strip()

        if flow.state == FlowState.AWAITING_VARIANT:
            if is_word_in_text(NO_WORDS, msg_lower) and any(w in msg_lower for w in {"cancel", "nahi", "nako", "chhodo", "rehen de"}):
                flow.state = "cancelled"
                return flow

            # Clean and set variant from user input button/text
            clean_v = msg_lower.replace(" vali", "").replace(" wali", "").strip()
            flow.variant = clean_v if clean_v else "normal"

            if flow.quantity is None or flow.quantity == "":
                flow.state = FlowState.AWAITING_QUANTITY
            elif flow.meal_type is None or flow.meal_type == "":
                flow.state = FlowState.AWAITING_MEAL_TYPE
            else:
                flow.state = FlowState.AWAITING_CONFIRMATION

        elif flow.state == FlowState.AWAITING_QUANTITY:
            flow.quantity = message.strip()
            if flow.meal_type is None or flow.meal_type == "":
                flow.state = FlowState.AWAITING_MEAL_TYPE
            else:
                flow.state = FlowState.AWAITING_CONFIRMATION

        elif flow.state == FlowState.AWAITING_MEAL_TYPE:
            for meal, words in MEAL_WORDS.items():
                if any(w in msg_lower for w in words):
                    flow.meal_type = meal
                    break
            if not flow.meal_type:
                flow.meal_type = "snack"
            flow.state = FlowState.AWAITING_CONFIRMATION

        elif flow.state == FlowState.AWAITING_CONFIRMATION:
            if is_word_in_text(YES_WORDS, msg_lower):
                flow.state = "confirmed"
            elif is_word_in_text(NO_WORDS | {"cancel", "nahi", "na", "nako", "chhodo", "rehen de", "skip", "no"}, msg_lower):
                flow.state = "cancelled"
            else:
                flow.state = "cancelled"

        elif flow.state == FlowState.AWAITING_EXERCISE_CONFIRM:
            flow.state = "confirmed" if is_word_in_text(YES_WORDS, msg_lower) else "cancelled"

        elif flow.state == FlowState.AWAITING_MULTI_CHOICE:
            # Options: "Save Both", "Save <name1>", "Save <name2>", "Cancel"
            if any(w in msg_lower for w in NO_WORDS | {"cancel"}):
                flow.state = "cancelled"
            elif "both" in msg_lower or "badha" in msg_lower or "dono" in msg_lower:
                flow.state = "multi_confirmed_all"
            else:
                # User said a specific food name — try to match
                items = flow.pending_items or []
                matched = [i for i in items
                           if i.get("food_name", "").lower() in msg_lower
                           or msg_lower in i.get("food_name", "").lower()]
                if matched:
                    flow.pending_items = matched  # keep only what they want
                    flow.state = "multi_confirmed_selected"
                elif any(w in msg_lower for w in YES_WORDS):
                    flow.state = "multi_confirmed_all"
                else:
                    flow.state = "multi_confirmed_all"  # default to save all

        return flow

    def needs_variant_clarification(self, food_name: str) -> bool:
        variant_foods = {
            "bhakri", "roti", "paratha", "dosa", "idli", "rice", "dal",
            "khichdi", "poha", "upma", "chapati", "naan", "puri",
            "egg", "paneer", "chicken", "fish", "aloo gobi",
        }
        return any(vf in food_name.lower() for vf in variant_foods)
