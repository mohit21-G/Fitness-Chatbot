"""
Chatbot Engine — MongoDB version
Orchestrates: LLM intent → Food/Exercise Search → Nutrition Calculator → Daily Log

IMPORTANT: The chatbot NEVER invents calorie data. All nutrition comes from MongoDB.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

IST = timezone(timedelta(hours=5, minutes=30))

from llm_service import LLMService, LLMParseResult, Intent, classify_recommendation_intent
from food_repository import FoodRepositorySync
from search_engine import FoodSearchEngine
from quantity_parser import parse_quantity, ParsedQuantity
from nutrition_calculator import NutritionCalculator
from exercise_calculator import ExerciseSearcher, ExerciseCalculator, parse_exercise_input, ParsedExerciseInput
from daily_log_repository import DailyLogRepository, meal_sort_key
from food_emoji import get_food_emoji, get_meal_emoji, get_exercise_emoji
from workout_recommender import build_workout_recommendation, format_recommendation
from bmr_tdee_calculator import compute_full_target
from external_food_api import ExternalFoodAPIService
from external_exercise_api import ExternalExerciseAPIService
from conversation import (
    ConversationStore, ConversationFlowManager, PendingFlow, FlowState,
    detect_language, get_response, get_option, VARIANT_WORDS, YES_WORDS, NO_WORDS,
    is_word_in_text,
)


from food_specificity import is_generic_food_query, get_generic_clarification_message
from database import sync_db


# ---------------------------------------------------------------------------
# Time / meal expression vocabulary (single source of truth)
# ---------------------------------------------------------------------------
# Used by BOTH meal-type detection (_detect_meal_type) and food-name cleaning
# (_clean_food_query) so a time/meal word can never leak into a food name.
# These are the common English/Hindi/Gujarati transliterations; the LLM is the
# primary, open-ended classifier — this is the deterministic fallback vocabulary.

MEAL_TIME_ANCHORS = {
    "breakfast": ["breakfast", "breckfast", "brekfast", "brakfast", "nashta", "nasta",
                  "subah", "subhe", "savare", "savre", "savar", "savaar", "savarne", "morning", "brunch"],
    "lunch": ["lunch", "dopahar", "dopaher", "dpahar", "bhojan", "midday", "noon", "afternoon",
              "bapor", "bapore", "baporme", "bapori", "bopor", "bopore", "bpore", "bpor"],
    "dinner": ["dinner", "raat", "ratre", "rate", "supper", "night",
               "sanje", "sanjeye", "shaam", "sham", "sanj", "saanj", "evening"],
    # Explicit snack / tea time context maps to snack.
    "snack": ["snack", "snacks", "nasto", "tea time", "teatime", "chai time",
              "rondhe", "ronde", "vehli sanje", "vaheli"],
}

# Relative modifiers meaning around/after/before another meal → the food is a snack.
MEAL_RELATIVE_MODIFIERS = {
    "after":  ["after", "post", "pachhi", "pachi", "pchi", "baad", "bad", "pade", "pachi"],
    "before": ["before", "pre", "pela", "pehla", "pehle", "pahela", "aage"],
    "around": ["around", "between", "mid", "midway", "aspaas", "vacche", "vche"],
}

# Tokens that are ALSO real food/drink names — must NEVER be stripped from a food
# query even though they appear inside a time phrase (e.g. "tea time", "chai time").
# Without this guard, "green tea" would wrongly become "green", "chai" → "".
_FOOD_AMBIGUOUS_TIME_TOKENS = {"chai", "tea", "time", "green"}

# Multi-word time phrases stripped as a whole phrase from a food query (only when
# the complete phrase is present), so their individual tokens (chai/tea/time)
# survive when used as a real food.
MEAL_TIME_PHRASES = ["tea time", "teatime", "chai time", "vehli sanje"]

# Flat set of SINGLE-WORD time/meal/modifier tokens that are never a food — these
# are safe to remove from a food-name query so a WHEN word never leaks into the
# food name. Food-ambiguous tokens are excluded.
_ALL_MEAL_TIME_TOKENS = set()
for _words in MEAL_TIME_ANCHORS.values():
    for _w in _words:
        if " " in _w:
            continue  # multi-word phrases handled separately
        _ALL_MEAL_TIME_TOKENS.add(_w)
for _words in MEAL_RELATIVE_MODIFIERS.values():
    _ALL_MEAL_TIME_TOKENS.update(_words)
_ALL_MEAL_TIME_TOKENS -= _FOOD_AMBIGUOUS_TIME_TOKENS


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

@dataclass
class ChatResponse:
    message: str
    intent: str
    action_taken: Optional[str] = None
    data: dict = field(default_factory=dict)
    needs_confirmation: bool = False
    pending_action: Optional[dict] = None
    options: list = field(default_factory=list)  # Clickable button options
    success: bool = True
    language: str = "en"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ChatbotEngine:
    """Main chatbot orchestrator (MongoDB, async)."""

    def __init__(self, db, user: dict, llm: LLMService):
        self.db = db                     # async motor database
        self.user = user                 # user document (dict)
        self.llm = llm
        self.memory = ConversationStore(db)
        self.flow_mgr = ConversationFlowManager(db, user["user_id"])
        self.log_repo = DailyLogRepository(db)
        # Sync repos for fuzzy search (CPU-bound, uses pymongo)
        self.food_repo = FoodRepositorySync(sync_db)
        self.search_engine = FoodSearchEngine(self.food_repo)
        self.external_api = ExternalFoodAPIService()
        self.exercise_searcher = ExerciseSearcher(sync_db)
        self.external_exercise_api = ExternalExerciseAPIService()
        self.lang = "en"

    async def process_message(self, message: str, context: Optional[str] = None, auto_log: bool = False) -> ChatResponse:
        new_lang = detect_language(message)
        if new_lang != "en" or not getattr(self, "lang", None):
            self.lang = new_lang

        if not context:
            context = await self.memory.get_context_string(self.user["user_id"], limit=5)

        await self.memory.add_message(self.user["user_id"], "user", message)

        # ── UI button short-circuit ───────────────────────────────────────
        # Button labels from the query-meal "no records" response must not be
        # fed into the food-search pipeline as food names.
        msg_stripped = message.strip().lower()
        _DISMISS_PHRASES = {
            # Gujarati
            "na", "naa", "nai", "nako", "na, skip karyu", "skip karyu", "na, skip",
            "na, bas", "bas", "okay, bas", "na, cancel", "cancel",
            # Hindi
            "nahi", "nahi, skip karo", "skip karo", "nahi, bas", "nahi, cancel", "rehen de", "chhodo",
            # English
            "no", "nope", "no, skip", "no, that's all", "no, cancel",
        }
        if msg_stripped in _DISMISS_PHRASES:
            await self.flow_mgr.save_flow_state(PendingFlow())
            response = ChatResponse(
                message=get_response("cancelled", self.lang),
                intent="cancelled", success=True, language=self.lang,
            )
            await self.memory.add_message(self.user["user_id"], "assistant", response.message, intent="cancelled")
            return response

        # ── Daily-Log Read Interruption Check ────────────────────────────
        # If the user asks for a Daily-Log read ("aaj nu total log", "today's total",
        # "aaje shu khadhu?"), immediately abandon any pending food/exercise flow
        # and route directly to the read handler. Never ask quantity/variant for a log read!
        _dl_read = self._classify_daily_log_read(message)
        if _dl_read is not None:
            await self.flow_mgr.save_flow_state(PendingFlow())  # reset flow to IDLE
            read_parsed = LLMParseResult(
                intent=_dl_read["intent"], date=_dl_read["date"],
                meal_type=_dl_read["meal_type"],
                raw_response="deterministic_daily_log", success=True,
            )
            if _dl_read["intent"] == Intent.GET_SUMMARY:
                res = await self._handle_get_summary(read_parsed)
            else:
                res = await self._handle_query_meal(read_parsed)
            await self.memory.add_message(self.user["user_id"], "assistant",
                                          res.message, intent=res.intent)
            return res

        # ── Deterministic pre-routing (BEFORE food/LLM detection) ─────────
        # Route unambiguous intents deterministically so they can NEVER leak
        # into the food-logging flow. Runs only when there is NO active flow
        # (so it never hijacks an in-progress clarification). Order matters:
        #   1. Nutrition lookup   ("calories in a banana")   → get_calories
        #   2. Exercise read      ("show my workout log")    → query_exercise
        #   3. Exercise log       ("did 3 sets of squats")   → log_exercise
        #   4. Daily-Log read     ("show today's summary")   → get_summary/query_meal
        # This does not change the Qwen model/config — it's a deterministic
        # fast-path in front of the existing LLM parse.
        _existing_flow = await self.flow_mgr.get_pending_flow()
        if _existing_flow.state == FlowState.IDLE:
            det = await self._deterministic_pre_route(message, auto_log)
            if det is not None:
                await self.memory.add_message(self.user["user_id"], "assistant",
                                              det.message, intent=det.intent)
                return det

        # ── Active multi-turn flow ───────────────────────────────────────
        pending_flow = await self.flow_mgr.get_pending_flow()
        # When waiting for exercise amount, never treat the reply as an intent switch
        _in_exercise_amount_flow = (pending_flow.state == FlowState.AWAITING_EXERCISE_AMOUNT)
        # When waiting for food quantity/variant/meal/choice/generic, if the user is answering
        # what the pending flow is awaiting, never treat it as an intent switch!
        _is_answering_pending_food_flow = False
        if pending_flow.state == FlowState.AWAITING_QUANTITY:
            _is_answering_pending_food_flow = self._looks_like_quantity_answer(message)
        elif pending_flow.state == FlowState.AWAITING_VARIANT:
            _is_answering_pending_food_flow = not self._is_intent_switch(message)
        elif pending_flow.state == FlowState.AWAITING_MEAL_TYPE:
            from conversation import MEAL_WORDS
            all_meal_words = {w for words in MEAL_WORDS.values() for w in words}
            _is_answering_pending_food_flow = any(w in message.lower() for w in all_meal_words) or len(message.strip().split()) <= 2
        elif pending_flow.state in (FlowState.AWAITING_MULTI_CHOICE, FlowState.AWAITING_GENERIC_RESOLUTION):
            _is_answering_pending_food_flow = not self._is_intent_switch(message)

        if pending_flow.state != FlowState.IDLE and (
            _in_exercise_amount_flow or _is_answering_pending_food_flow or not self._is_intent_switch(message)
        ):

            # Check if user sent a NEW food while we were clarifying a different one
            if self._is_new_food_while_in_flow(message, pending_flow):
                # Auto-confirm previous food item if food_id exists
                prev_food_name = pending_flow.food_name_display or pending_flow.food_name or "Item"
                prev_meal = pending_flow.meal_type or "snack"
                prev_qty = pending_flow.quantity or ("1 cup" if self._is_beverage(pending_flow.food_name or "") else "1 serving")
                prefix_msg = ""

                if pending_flow.food_id:
                    food_doc = self.food_repo.get_by_id(pending_flow.food_id)
                    if food_doc:
                        parsed_qty = parse_quantity(prev_qty)
                        nutrition = NutritionCalculator().calculate(food_doc, parsed_qty, variant=pending_flow.variant or "normal")
                        pending_action = self._build_food_pending(food_doc, nutrition, prev_qty, prev_meal,
                                                                  date.today().isoformat(), 1.0)
                        await self._force_food_log(pending_action)
                        prefix_msg = f"Logged {prev_qty} {prev_food_name} for {prev_meal} ({nutrition.calories:.0f} kcal).\n\n"

                # Reset flow and continue with new food query
                await self.flow_mgr.save_flow_state(PendingFlow())

                # Clean message of previous food name if present (e.g. "chai sathe me bhakhri khadhi" -> "bhakhri")
                cleaned_msg = message
                if pending_flow.food_name:
                    cleaned_msg = re.sub(rf"\b{re.escape(pending_flow.food_name)}\b", "", cleaned_msg, flags=re.IGNORECASE).strip()

                parse_result = await self.llm.parse_intent(cleaned_msg, context=context)
                if not parse_result.meal_type:
                    parse_result.meal_type = prev_meal

                res = await self._handle_log_food(parse_result, original_message=cleaned_msg, auto_log=auto_log)
                res.message = prefix_msg + res.message
                await self.memory.add_message(self.user["user_id"], "assistant", res.message, intent=res.intent)
                return res
            else:
                response = await self._continue_flow(message, pending_flow, auto_log)
                if response is not None:
                    display_intent = response.intent if not response.intent.startswith("flow:") else "flow_response"
                    await self.memory.add_message(self.user["user_id"], "assistant", response.message, intent=display_intent)
                    return response

        # If the user switched topics mid-flow, abandon the pending flow
        elif pending_flow.state != FlowState.IDLE:
            await self.flow_mgr.save_flow_state(PendingFlow())  # reset to IDLE

        # ── Multi-food detection ─────────────────────────────────────────
        multi_items = self._detect_multi_food(message)
        if multi_items and len(multi_items) > 1:
            response = await self._handle_multi_food(multi_items, message, auto_log=False)
            await self.memory.add_message(self.user["user_id"], "assistant", response.message, intent=response.intent)
            return response

        # ── LLM intent parsing ───────────────────────────────────────────
        parse_result = await self.llm.parse_intent(message, context=context)

        if not parse_result.success:
            response = ChatResponse(
                message="Sorry, I couldn't understand that. Could you rephrase?",
                intent=Intent.UNKNOWN, success=False, language=self.lang,
            )
            await self.memory.add_message(self.user["user_id"], "assistant", response.message, intent=response.intent)
            return response

        # ── Deterministic food salvage (before routing) ───────────────────
        # The LLM occasionally fails to parse very short / native-script food
        # words (e.g. Gujarati "ચા", Hindi "दाल") and returns intent=unknown with
        # no food_query. Before falling through to the generic reply, try a
        # DETERMINISTIC search on the raw message. We accept the salvage ONLY on
        # an EXACT match (exact_name / exact_alias / variant_strip) — never fuzzy
        # or low-confidence — so junk ("abc", "qwerty") can never be salvaged and
        # junk rejection is not loosened. This runs entirely offline (no LLM,
        # no model/config change) and complements the normalize()/transliteration
        # layer that maps native script and typos to canonical DB foods.
        intent = parse_result.intent
        _actionable = {
            Intent.LOG_FOOD, Intent.LOG_EXERCISE, Intent.GET_SUMMARY,
            Intent.GET_CALORIES, Intent.GET_PROFILE, Intent.QUERY_MEAL,
            Intent.QUERY_EXERCISE, Intent.SKIP_MEAL, Intent.RECOMMEND_WORKOUT,
            Intent.RECOMMEND_MEAL,
        }
        if intent not in _actionable and not (parse_result.food_query or "").strip():
            salvaged_q = self._clean_food_query(message, original_message=message)
            # Restrict salvage to SHORT, single-token inputs. The LLM handles
            # multi-word sentences reliably; only terse native-script / typo
            # words (e.g. "ચા", "दाल") slip through as intent=unknown. Requiring a
            # single token prevents conversational phrases like "hello how are
            # you" from being salvaged when a stray token happens to match a
            # branded food name.
            if salvaged_q and len(salvaged_q.split()) == 1:
                # Never salvage something the deterministic gate flags as
                # non-food/gibberish (keeps junk rejection intact).
                try:
                    non_food = self.llm._looks_like_non_food(salvaged_q) is True
                except Exception:
                    non_food = False
                if not non_food:
                    try:
                        sres = self.search_engine.search(salvaged_q)
                        if sres.match_type.value in ("exact_name", "exact_alias", "variant_strip"):
                            parse_result.intent = Intent.LOG_FOOD
                            parse_result.food_query = salvaged_q
                            intent = Intent.LOG_FOOD
                    except Exception:
                        pass

        # ── Misroute Guard (Deterministic) ────────────────────────────────
        # 1. Hallucinated exercise, post-workout eat statements, or pure food quantity
        #    (e.g. LLM saw "run" in "brunch", or "workout" in "ate 4 dates after workout",
        #     or standalone food portion like "1 plate", "2 pieces", "100g")
        if intent == Intent.LOG_EXERCISE:
            ex_word = (parse_result.exercise_input or "").strip().lower()
            is_hallucinated = ex_word and not re.search(r"\b" + re.escape(ex_word) + r"\b", message.lower())
            is_eat_past = self._is_explicit_past_tense(message)
            is_pure_food_quantity = not self._looks_like_exercise(message) and self._looks_like_quantity_answer(message)
            if is_hallucinated or is_eat_past or is_pure_food_quantity:
                cleaned_food = self._clean_food_query(message, original_message=message)
                if cleaned_food:
                    f_search = self.search_engine.search(cleaned_food)
                    if f_search.food and f_search.match_type.value in ("exact_name", "exact_alias", "variant_strip"):
                        intent = Intent.LOG_FOOD
                        parse_result.intent = Intent.LOG_FOOD
                        parse_result.food_query = cleaned_food
                        if not parse_result.quantity:
                            qty_p = parse_quantity(message)
                            if qty_p.amount is not None:
                                parse_result.quantity = f"{qty_p.amount} {qty_p.unit or 'piece'}"
                elif is_pure_food_quantity:
                    intent = Intent.CLARIFICATION_NEEDED
                    parse_result.intent = Intent.CLARIFICATION_NEEDED
                    parse_result.clarification_question = get_response("what_food", self.lang)

        # 2. False summary on eat statements (e.g. "yaar maine aaj kaju katli kha li" has eat verb + real food,
        #    not asking for summary / log view)
        if intent in (Intent.GET_SUMMARY, Intent.QUERY_MEAL) and self._is_explicit_past_tense(message):
            cleaned_food = self._clean_food_query(message, original_message=message)
            if cleaned_food:
                f_search = self.search_engine.search(cleaned_food)
                if f_search.food and f_search.match_type.value in ("exact_name", "exact_alias", "variant_strip", "fuzzy_name", "fuzzy_alias"):
                    intent = Intent.LOG_FOOD
                    parse_result.intent = Intent.LOG_FOOD
                    parse_result.food_query = cleaned_food

        # 3. Calorie / macro inquiry misrouted to log_food due to portion/quantity units
        #    (e.g. "carbs in a slice of bread", "protein in 100g soya chunks", "how much sugar in a can of cola")
        _cal_query_pattern = r"^(?:how\s+many\s+calories|how\s+much\s+(?:calories|protein|carbs?|fat|sugar)|calories|protein|carbs?|fat|sugar)\s+(?:in|for|of)\b"
        if intent != Intent.GET_CALORIES and re.search(_cal_query_pattern, message.lower().strip()):
            intent = Intent.GET_CALORIES
            parse_result.intent = Intent.GET_CALORIES
            match_part = re.sub(_cal_query_pattern, "", message.lower().strip(), count=1).strip()
            qty_parsed = parse_quantity(match_part)
            cleaned_food = self._clean_food_query(match_part, original_message=message)
            parse_result.food_query = cleaned_food or match_part
            if qty_parsed.amount is not None and qty_parsed.unit:
                parse_result.quantity = f"{qty_parsed.amount} {qty_parsed.unit}"

        # 4. Recommendation intent disambiguation / override
        rec_intent = classify_recommendation_intent(message)
        if rec_intent:
            intent = rec_intent
            parse_result.intent = rec_intent

        # ── Route ────────────────────────────────────────────────────────
        if intent == Intent.GREETING:
            response = self._handle_greeting()
        elif intent == Intent.RECOMMEND_WORKOUT:
            response = await self._handle_recommend_workout()
        elif intent == Intent.RECOMMEND_MEAL:
            response = await self._handle_recommend_meal(parse_result, original_message=message)
        elif intent == Intent.LOG_FOOD:
            response = await self._handle_log_food(parse_result, auto_log, original_message=message)
        elif intent == Intent.LOG_EXERCISE:
            response = await self._handle_log_exercise(parse_result, auto_log, original_message=message)
        elif intent == Intent.GET_SUMMARY:
            response = await self._handle_get_summary(parse_result)
        elif intent == Intent.GET_CALORIES:
            response = self._handle_get_calories(parse_result)
        elif intent == Intent.GET_PROFILE:
            response = self._handle_get_profile()
        elif intent == Intent.QUERY_MEAL:
            response = await self._handle_query_meal(parse_result)
        elif intent == Intent.QUERY_EXERCISE:
            response = await self._handle_query_exercise(parse_result)
        elif intent == Intent.SKIP_MEAL:
            response = await self._handle_skip_meal(parse_result)
        elif intent == Intent.CLARIFICATION_NEEDED:
            response = ChatResponse(
                message=parse_result.clarification_question or get_response("what_food", self.lang),
                intent=Intent.CLARIFICATION_NEEDED, success=True, language=self.lang,
            )
        else:
            response = ChatResponse(
                message="I can help you log food, log exercise, check calories, or see your daily summary.",
                intent=Intent.UNKNOWN, success=True, language=self.lang,
            )

        await self.memory.add_message(self.user["user_id"], "assistant", response.message, intent=response.intent)
        return response

    # ------------------------------------------------------------------
    # Multi-turn Flow
    # ------------------------------------------------------------------

    async def _continue_flow(self, message: str, flow: PendingFlow, auto_log: bool) -> ChatResponse:
        msg_lower = message.lower().strip()

        # ── Sequential Queue Active Flow Handler ───────────────────────────
        if flow.items and len(flow.items) > 0 and flow.state not in ("confirmed", "multi_confirmed_all", "multi_confirmed_selected"):
            # Check cancellation
            if any(w in msg_lower for w in {"cancel", "skip", "nako", "chhodo", "rehen de"}) or (msg_lower in {"na", "no", "nahi", "cancel"}):
                await self.flow_mgr.save_flow_state(PendingFlow())
                return ChatResponse(message=get_response("cancelled", self.lang), intent="cancelled", success=True, language=self.lang)

            # Check confirmation when in AWAITING_MULTI_CHOICE
            if flow.state == FlowState.AWAITING_MULTI_CHOICE:
                if is_word_in_text(YES_WORDS | {"save", "save all", "save both", "yes"}, msg_lower):
                    flow.state = "multi_confirmed_all"
                    return await self._execute_flow_log(flow)
                elif is_word_in_text(NO_WORDS | {"cancel", "no"}, msg_lower):
                    await self.flow_mgr.save_flow_state(PendingFlow())
                    return ChatResponse(message=get_response("cancelled", self.lang), intent="cancelled", success=True, language=self.lang)
                else:
                    # User sent a follow-up food item (e.g. "sathe 1 bowl dal pan khadhi")
                    cleaned_food_q = self._clean_food_query(message, original_message=message)
                    if cleaned_food_q:
                        new_item = await self._build_food_item_queue_object(cleaned_food_q, message, shared_meal_type=flow.shared_meal_type)
                        if new_item:
                            flow.items.append(new_item)
                            await self.flow_mgr.save_flow_state(flow)
                            return await self._process_sequential_queue(flow)

            if flow.state == FlowState.AWAITING_GENERIC_RESOLUTION:
                # Check cancellation / skip
                if any(w in msg_lower for w in {"cancel", "skip", "nako", "chhodo", "rehen de"}) or (msg_lower in {"na", "no", "nahi", "cancel", "nathi"}):
                    if getattr(flow, "unresolved_items", None) and len(flow.unresolved_items) > 0:
                        flow.unresolved_items.pop(0)
                    await self.flow_mgr.save_flow_state(flow)
                    return await self._process_sequential_queue(flow)

                # User provided specific dish name (e.g. "bhindi nu shaak" or "sev tameta")
                cleaned_food_q = self._clean_food_query(message, original_message=message)
                qty_parsed = parse_quantity(message)
                explicit_qty = None
                if qty_parsed.amount is not None and qty_parsed.amount > 0 and qty_parsed.unit != "serving":
                    explicit_qty = f"{qty_parsed.amount:g} {qty_parsed.unit}"

                new_item = await self._build_food_item_queue_object(
                    cleaned_food_q or message, message, shared_meal_type=flow.shared_meal_type
                )
                if new_item:
                    if explicit_qty and not new_item.get("quantity"):
                        new_item["quantity"] = explicit_qty
                        new_item["needs_quantity"] = False
                    if getattr(flow, "unresolved_items", None) and len(flow.unresolved_items) > 0:
                        flow.unresolved_items.pop(0)
                    flow.items.append(new_item)
                    flow.current_index = len(flow.items) - 1
                    await self.flow_mgr.save_flow_state(flow)
                    return await self._process_sequential_queue(flow)
                else:
                    return ChatResponse(
                        message=get_response("food_not_found", self.lang, food=message),
                        intent=Intent.CLARIFICATION_NEEDED,
                        options=[],
                        success=True,
                        language=self.lang,
                    )

            idx = flow.current_index
            if idx < len(flow.items):
                curr = flow.items[idx]
                if flow.state == FlowState.AWAITING_QUANTITY:
                    # Check if reply has both a quantity for curr AND a new food (e.g. "2 rotla ane bhindi nu shaak")
                    multi_parts = self._detect_multi_food(message)
                    if multi_parts and len(multi_parts) > 1:
                        if self._looks_like_quantity_answer(multi_parts[0]):
                            curr["quantity"] = self._extract_quantity_answer(multi_parts[0])
                            curr["needs_quantity"] = False
                            remaining_text = " and ".join(multi_parts[1:])
                            added = await self._enqueue_new_foods_from_reply(remaining_text, flow)
                            await self.flow_mgr.save_flow_state(flow)
                            return await self._process_sequential_queue(flow)

                    # PRIORITY GUARD: a short reply that resolves to a KNOWN food
                    # (e.g. "Dal", "Chhas", "Jalebi" typed while we asked "Roti
                    # ketli?") is a NEW food, not a quantity. Enqueue it and keep
                    # the current item's quantity still pending — the user is NOT
                    # forced to answer before adding another food.
                    if self._reply_is_known_food(message):
                        added = await self._enqueue_new_foods_from_reply(message, flow)
                        if added:
                            await self.flow_mgr.save_flow_state(flow)
                            return await self._process_sequential_queue(flow)
                        # Could not enqueue (shouldn't happen) → re-ask quantity.
                        await self.flow_mgr.save_flow_state(flow)
                        return await self._process_sequential_queue(flow)
                    # GUARD: the reply may not be a quantity at all — it might be a
                    # NEW food (e.g. "sathe jalebi pn khadhi" while asked Fafda qty).
                    # Only store it as a quantity when it actually looks like one.
                    if self._looks_like_quantity_answer(message):
                        curr["quantity"] = self._extract_quantity_answer(message)
                        curr["needs_quantity"] = False
                    else:
                        # Try to treat it as one or more NEW foods and enqueue them,
                        # leaving the current item's quantity still pending.
                        added = await self._enqueue_new_foods_from_reply(message, flow)
                        if added:
                            await self.flow_mgr.save_flow_state(flow)
                            return await self._process_sequential_queue(flow)
                        # Not a quantity and not a recognisable food → re-ask quantity.
                        await self.flow_mgr.save_flow_state(flow)
                        return await self._process_sequential_queue(flow)
                elif flow.state == FlowState.AWAITING_VARIANT:
                    v_clean = message.strip().lower().replace(" vali", "").replace(" wali", "").strip()
                    # Is the reply one of the actually-offered variant options for
                    # THIS food? If so it is a genuine variant answer (highest
                    # priority — "normal"/"ghee"/etc. win even if a same-named food
                    # exists somewhere in the DB).
                    offered = {o.lower() for o in (curr.get("food_doc") or {}).get("preparation_variants", [])}
                    offered |= {"normal", "ghee", "butter", "oil", "fried", "steamed",
                                "grilled", "boiled", "tandoori", "plain"}
                    is_offered_variant = v_clean in offered

                    # PRIORITY GUARD: a reply that resolves to a KNOWN food and is
                    # NOT one of the offered variants is a NEW food (e.g. "Dal"
                    # typed while we asked Roti's preparation), not a variant.
                    if not is_offered_variant and self._reply_is_known_food(message):
                        added = await self._enqueue_new_foods_from_reply(message, flow)
                        if added:
                            await self.flow_mgr.save_flow_state(flow)
                            return await self._process_sequential_queue(flow)

                    # Guard against a new-food sentence landing in the variant slot.
                    if is_offered_variant or self._looks_like_variant_answer(v_clean, curr):
                        curr["variant"] = v_clean if v_clean else "normal"
                        curr["needs_variant"] = False
                    else:
                        added = await self._enqueue_new_foods_from_reply(message, flow)
                        if added:
                            await self.flow_mgr.save_flow_state(flow)
                            return await self._process_sequential_queue(flow)
                        # fall through: accept whatever they typed as the variant
                        curr["variant"] = v_clean if v_clean else "normal"
                        curr["needs_variant"] = False

            if flow.state == FlowState.AWAITING_MEAL_TYPE:
                from conversation import MEAL_WORDS
                # Classify the reply by its semantic meaning first (handles
                # relative answers like "after lunch" → snack), then fall back
                # to direct meal-word matching.
                matched_meal = self._detect_meal_type(message)
                if not matched_meal:
                    for m_type, words in MEAL_WORDS.items():
                        if any(w in msg_lower for w in words):
                            matched_meal = m_type
                            break
                if matched_meal:
                    flow.shared_meal_type = matched_meal
                elif self._reply_is_known_food(message):
                    # A new food was typed instead of a meal slot — enqueue it and
                    # keep asking; do NOT swallow it or force a default meal yet.
                    added = await self._enqueue_new_foods_from_reply(message, flow)
                    if added:
                        await self.flow_mgr.save_flow_state(flow)
                        return await self._process_sequential_queue(flow)
                    flow.shared_meal_type = flow.shared_meal_type or "snack"
                else:
                    flow.shared_meal_type = flow.shared_meal_type or "snack"

            return await self._process_sequential_queue(flow)

        flow = self.flow_mgr.interpret_user_response(message, flow)

        if flow.state == "cancelled":
            await self.flow_mgr.save_flow_state(PendingFlow())
            if any(w in msg_lower for w in NO_WORDS | {"cancel", "skip", "nahi", "na", "nako", "chhodo", "rehen de", "no", "nah"}):
                return ChatResponse(message=get_response("cancelled", self.lang), intent="cancelled", success=True, language=self.lang)
            return None

        if flow.state in ("confirmed", "multi_confirmed_all", "multi_confirmed_selected"):
            return await self._execute_flow_log(flow)

        await self.flow_mgr.save_flow_state(flow)

        if flow.state == FlowState.AWAITING_MULTI_CHOICE:
            # Re-display the multi-choice options
            items = flow.pending_items or []
            total_cal = sum(i.get("calories", 0) for i in items)
            items_display = "\n".join(f"  • {i['food_name']} — {i['calories']:.0f} kcal" for i in items)
            if len(items) == 1:
                options = [get_option("yes_save", self.lang), get_option("no_cancel", self.lang)]
            else:
                options = ([get_option("save_both", self.lang)]
                           + [f"Save {i['food_name']}" for i in items]
                           + [get_option("cancel", self.lang)])
            msg = f"Saving:\n{items_display}\n\nTotal: {total_cal:.0f} kcal. Which to save?"
            return ChatResponse(message=msg, intent=f"flow:{flow.state}",
                                options=options, success=True, language=self.lang)

        if flow.state == FlowState.AWAITING_VARIANT:
            msg = get_response("ask_variant", self.lang, food=flow.food_name_display or flow.food_query)
            options = self._get_variant_options(flow.food_name or flow.food_query or "")
        elif flow.state == FlowState.AWAITING_QUANTITY:
            msg = get_response("ask_quantity", self.lang, food=flow.food_name_display or flow.food_query)
            food_doc = None
            if flow.food_id:
                food_doc = self.food_repo.get_by_id(flow.food_id)
            if not food_doc:
                food_doc = self.food_repo.get_by_exact_name(flow.food_name or flow.food_query or "")
            cat = food_doc.get("category", "") if food_doc else ""
            sg = food_doc.get("serving_size_g", 100) if food_doc else 100
            options = self._get_quantity_options(flow.food_name or flow.food_query or "", cat, sg, food_doc=food_doc)
        elif flow.state == FlowState.AWAITING_MEAL_TYPE:
            msg = get_response("ask_meal_type", self.lang, meal_context=flow.food_name_display or flow.food_query)
            options = [
                get_option("meal_breakfast", self.lang),
                get_option("meal_lunch", self.lang),
                get_option("meal_dinner", self.lang),
                get_option("meal_snack", self.lang),
            ]
        elif flow.state == FlowState.AWAITING_CONFIRMATION:
            return await self._build_confirmation_response(flow)
        elif flow.state == FlowState.AWAITING_EXERCISE_AMOUNT:
            # User is providing the amount for an exercise (scenario 2)
            return await self._handle_exercise_amount_reply(message, flow)
        else:
            return ChatResponse(message="Let's start over. What did you eat?", intent=Intent.UNKNOWN, success=True, language=self.lang)

        return ChatResponse(message=msg, intent=f"flow:{flow.state}", options=options, success=True, language=self.lang)

    async def _handle_exercise_amount_reply(self, message: str, flow: PendingFlow) -> ChatResponse:
        """
        User has replied with the exercise amount (e.g. "50") after the bot asked
        "Ketla pushups mara?". Parse the amount, calculate calories, show confirmation.
        """
        # If user replied with a full structured workout routine, route to routine handler
        from exercise_calculator import parse_workout_routine
        routine = parse_workout_routine(message)
        if routine and routine.items:
            await self.flow_mgr.save_flow_state(PendingFlow())
            return await self._handle_workout_routine(routine, message, auto_log=False)

        ex_data = flow.exercise_data or {}
        exercise_id = ex_data.get("exercise_id")
        native_unit = ex_data.get("native_unit", "reps")
        log_date_str = ex_data.get("log_date", "today")

        # Parse the amount from user's reply
        from exercise_calculator import parse_duration_minutes
        parsed_dur = parse_duration_minutes(message.strip())
        is_duration_msg = any(w in message.lower() for w in ["hour", "houre", "hr", "ghanta", "kalak", "min", "sec"])
        if parsed_dur is not None and (native_unit in ("minutes", "mins", "minute") or is_duration_msg):
            amount = parsed_dur
            parsed_unit = "minutes"
        else:
            ex_parsed = parse_exercise_input(message.strip())
            amount = ex_parsed.amount
            parsed_unit = ex_parsed.unit
            if amount is None:
                import re as _re
                m = _re.search(r"\d+(?:\.\d+)?", message)
                if m:
                    amount = float(m.group())

        if amount is None or amount <= 0:
            # Still no amount — ask again
            msg = get_response(
                "ask_exercise_amount", self.lang,
                exercise=ex_data.get("exercise_name_display", "exercise"),
                unit=native_unit,
            )
            return ChatResponse(message=msg,
                                intent=f"flow:{FlowState.AWAITING_EXERCISE_AMOUNT}",
                                success=True, language=self.lang)

        # Fetch the exercise doc from DB
        exercise = None
        if exercise_id:
            exercise = self.exercise_searcher.collection.find_one(
                {"exercise_id": exercise_id}, {"_id": 0}
            )
        if exercise is None:
            exercise_name = ex_data.get("exercise_name", "")
            exercise, _ = self.exercise_searcher.search(exercise_name)
            # Local miss → external pipeline (ExerciseDB → wger → Compendium → AI)
            if not exercise and exercise_name:
                exercise, _ = await self._resolve_exercise_external(exercise_name)

        if not exercise:
            await self.flow_mgr.save_flow_state(PendingFlow())
            return ChatResponse(
                message="Sorry, I lost track of the exercise. Please try again.",
                intent=Intent.CLARIFICATION_NEEDED, success=True, language=self.lang,
            )

        # Prefer the display name captured when the flow started — it preserves
        # the user's exact wording (never a generic substitution).
        exercise = dict(exercise)
        display_name = ex_data.get("exercise_name_display") or exercise["exercise_name_display"]
        exercise["exercise_name_display"] = display_name

        calc_result = ExerciseCalculator().calculate(
            exercise, amount=amount, unit=parsed_unit, weight_kg=self._user_weight_kg(),
        )
        log_date = self._resolve_date(log_date_str)

        pending = {
            "type": "log_exercise",
            "user_id": self.user["user_id"],
            "log_date": log_date.isoformat(),
            "exercise_id": exercise["exercise_id"],
            "exercise_name": exercise["exercise_name"],
            "exercise_name_display": display_name,
            "category": exercise["category"],
            "exercise_input": f"{int(amount)} {parsed_unit or native_unit} {exercise['exercise_name']}",
            "amount": calc_result.amount,
            "unit": calc_result.unit,
            "calories_min": calc_result.calories_min,
            "calories_avg": calc_result.calories_avg,
            "calories_max": calc_result.calories_max,
            "search_confidence": ex_data.get("search_confidence", 1.0),
        }

        # Clear the AWAITING_EXERCISE_AMOUNT flow state
        await self.flow_mgr.save_flow_state(PendingFlow())

        msg = get_response(
            "confirm_exercise", self.lang,
            exercise=exercise["exercise_name_display"],
            amount=calc_result.amount,
            unit=calc_result.unit,
            calories=calc_result.calories_avg,
        )
        msg += get_response("safety_disclaimer", self.lang)

        return ChatResponse(
            message=msg, intent=Intent.LOG_EXERCISE,
            data={"exercise": pending},
            needs_confirmation=True, pending_action=pending,
            success=True, language=self.lang,
        )

    async def _build_confirmation_response(self, flow: PendingFlow) -> ChatResponse:
        # Use food_id for lookup if available (avoids re-searching by query string)
        food = None
        confidence = 1.0
        if flow.is_ai_estimated and flow.ai_nutrition:
            est = flow.ai_nutrition
            food = {
                "food_id": flow.food_id or f"ai_est_{flow.food_name}",
                "food_name": flow.food_name or "custom food",
                "food_name_display": flow.food_name_display or flow.food_query.title() if flow.food_query else "Custom Food",
                "category": "AI Estimated",
                "serving_size_g": est.get("serving_size_g", 150),
                "serving_unit": est.get("serving_unit", "serving"),
                "calories_kcal": est.get("calories_kcal", 200.0),
                "protein_g": est.get("protein_g", 5.0),
                "carbs_g": est.get("carbs_g", 30.0),
                "fat_g": est.get("fat_g", 6.0),
                "fiber_g": est.get("fiber_g", 2.0),
                "source": flow.source or "AI Estimated",
                "is_ai_estimated": True
            }
            confidence = 0.85
        elif flow.food_id:
            food = self.food_repo.get_by_id(flow.food_id)

        if food is None and flow.food_query:
            # Fallback re-search across tiers if not in local DB by ID
            off_food = await self.external_api.search_open_food_facts(flow.food_query)
            if off_food:
                food = off_food
            else:
                usda_food = await self.external_api.search_usda(flow.food_query)
                if usda_food:
                    food = usda_food

        if food is None:
            search_result = self.search_engine.search(flow.food_query or "")
            if search_result.match_type.value in ("no_match", "low_confidence"):
                return ChatResponse(message=get_response("food_not_found", self.lang, food=flow.food_query),
                                    intent=Intent.CLARIFICATION_NEEDED, success=True, language=self.lang)
            food = search_result.food
            confidence = search_result.confidence

        quantity = flow.quantity or "1 serving"
        parsed_qty = parse_quantity(quantity)
        if parsed_qty.amount is None or parsed_qty.amount <= 0 or not parsed_qty.unit:
            parsed_qty = ParsedQuantity(amount=1.0, unit="serving", raw_input=quantity, confidence=1.0)
        nutrition = NutritionCalculator().calculate(food, parsed_qty, variant=flow.variant)

        variant_text = f" ({flow.variant})" if flow.variant and flow.variant != "normal" else ""
        if flow.is_ai_estimated:
            msg_key = "confirm_food_ai"
        elif food.get("category") in ("Beverages", "Drinks"):
            msg_key = "confirm_food_drink"
        else:
            msg_key = "confirm_food"
        msg = get_response(msg_key, self.lang,
                           qty=quantity, food=food["food_name_display"], variant_text=variant_text,
                           calories=nutrition.calories, protein=nutrition.protein_g,
                           carbs=nutrition.carbs_g, fat=nutrition.fat_g,
                           meal=flow.meal_type or "snack",
                           source=flow.source or food.get("source", "Local Database"))
        msg += get_response("safety_disclaimer", self.lang)

        flow.state = FlowState.AWAITING_CONFIRMATION
        await self.flow_mgr.save_flow_state(flow)

        pending_action = self._build_food_pending(
            food, nutrition, quantity, flow.meal_type or "snack",
            self._resolve_date(flow.date).isoformat(), confidence,
        )

        return ChatResponse(message=msg, intent=Intent.LOG_FOOD, needs_confirmation=True,
                            pending_action=pending_action, data={"nutrition": pending_action},
                            success=True, language=self.lang)

    async def _execute_flow_log(self, flow: PendingFlow) -> ChatResponse:
        # ── Exercise log ──────────────────────────────────────────────────
        if flow.exercise_data:
            data = flow.exercise_data
            await self.log_repo.create_exercise_log(**data)
            await self.flow_mgr.save_flow_state(PendingFlow())
            return ChatResponse(
                message=get_response("exercise_logged", self.lang,
                                     exercise=data.get("exercise_name_display", "Exercise"),
                                     calories=data.get("calories_avg", 0)),
                intent=Intent.LOG_EXERCISE, action_taken="exercise_logged",
                data={"calories_burned": data.get("calories_avg", 0)}, success=True, language=self.lang)

        # ── Multi-item confirmed (Save Both / Save X / Save Y / Save All) ────────────
        if flow.state in ("multi_confirmed_all", "multi_confirmed_selected") or (flow.pending_items and len(flow.pending_items) > 0):
            items = flow.pending_items or []
            meal_type = flow.shared_meal_type or flow.meal_type or "snack"
            log_date_val = self._resolve_date(flow.date)
            return await self._save_multi_items(items, meal_type, self.user["user_id"], log_date_val)

        # ── Single-item confirmed ─────────────────────────────────────────
        food = None
        confidence = 1.0
        if flow.is_ai_estimated and flow.ai_nutrition:
            est = flow.ai_nutrition
            food = {
                "food_id": flow.food_id or f"ai_est_{flow.food_name}",
                "food_name": flow.food_name or "custom food",
                "food_name_display": flow.food_name_display or flow.food_query.title() if flow.food_query else "Custom Food",
                "category": "AI Estimated",
                "serving_size_g": est.get("serving_size_g", 150),
                "serving_unit": est.get("serving_unit", "serving"),
                "calories_kcal": est.get("calories_kcal", 200.0),
                "protein_g": est.get("protein_g", 5.0),
                "carbs_g": est.get("carbs_g", 30.0),
                "fat_g": est.get("fat_g", 6.0),
                "fiber_g": est.get("fiber_g", 2.0),
                "source": flow.source or "AI Estimated",
                "is_ai_estimated": True
            }
            confidence = 0.85
        elif flow.food_id:
            food = self.food_repo.get_by_id(flow.food_id)

        if food is None and flow.food_query:
            off_food = await self.external_api.search_open_food_facts(flow.food_query)
            if off_food:
                food = off_food
            else:
                usda_food = await self.external_api.search_usda(flow.food_query)
                if usda_food:
                    food = usda_food

        if food is None:
            search_result = self.search_engine.search(flow.food_query or "")
            if search_result.match_type.value in ("no_match", "low_confidence"):
                return ChatResponse(message=get_response("food_not_found", self.lang, food=flow.food_query),
                                    intent=Intent.CLARIFICATION_NEEDED, success=True, language=self.lang)
            food = search_result.food
            confidence = search_result.confidence
        quantity = flow.quantity or "1 serving"
        parsed_qty = parse_quantity(quantity)
        if parsed_qty.amount is None or parsed_qty.amount <= 0 or not parsed_qty.unit:
            parsed_qty = ParsedQuantity(amount=1.0, unit="serving", raw_input=quantity, confidence=1.0)
        nutrition = NutritionCalculator().calculate(food, parsed_qty, variant=flow.variant)

        log_date_val = self._resolve_date(flow.date)

        await self.log_repo.create_food_log(
            user_id=self.user["user_id"], log_date=log_date_val,
            meal_type=flow.meal_type or "snack",
            food_id=food["food_id"], food_name=food["food_name"],
            food_name_display=food["food_name_display"],
            quantity_input=quantity, quantity_amount=nutrition.quantity_amount,
            quantity_unit=nutrition.quantity_unit, quantity_grams=nutrition.quantity_grams,
            variant=nutrition.variant if nutrition.variant != "normal" else None,
            calories=nutrition.calories, protein_g=nutrition.protein_g,
            carbs_g=nutrition.carbs_g, fat_g=nutrition.fat_g, fiber_g=nutrition.fiber_g,
            search_confidence=confidence,
            source=flow.source or food.get("source", "Local Database"),
        )

        await self.flow_mgr.save_flow_state(PendingFlow())

        msg = "✓ Log saved successfully!\n" + get_response("food_logged", self.lang, food=food["food_name_display"],
                                 calories=nutrition.calories, meal=flow.meal_type or "snack")
        day_log = await self._build_day_log_text(self.user["user_id"], log_date_val, self.lang)
        if day_log:
            msg += "\n\n" + day_log
        recommendation = await self._build_nutrition_recommendation(
            self.user["user_id"], log_date_val, flow.meal_type or "snack", self.lang
        )
        if recommendation:
            msg += "\n\n---\n" + recommendation
        return ChatResponse(
            message=msg,
            intent=Intent.LOG_FOOD, action_taken="food_logged",
            data={"calories": nutrition.calories, "food_name": food["food_name_display"], "show_toast": True, "toast_message": "✓ Log saved successfully"},
            success=True, language=self.lang)

    # ------------------------------------------------------------------
    # Intent Handlers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Workout Recommendation
    # ------------------------------------------------------------------

    def _looks_like_meal_recommendation(self, message: str) -> bool:
        """
        Deterministically detect a nutrition / meal recommendation request.
        Must NEVER route to workout recommendations.
        """
        return classify_recommendation_intent(message) == Intent.RECOMMEND_MEAL

    def _looks_like_workout_recommendation(self, message: str) -> bool:
        """
        Deterministically detect a workout recommendation request.
        Must NOT fire on:
        - Exercise LOG messages ("did 30 min yoga")
        - Exercise READ messages ("show my workout history")
        - Meal / nutrition recommendation requests ("post workout meal", "suggest high protein breakfast")
        """
        rec_intent = classify_recommendation_intent(message)
        if rec_intent == Intent.RECOMMEND_MEAL:
            return False
        if rec_intent == Intent.RECOMMEND_WORKOUT:
            return True
        return False

    async def _handle_recommend_meal(self, parsed: Optional[LLMParseResult] = None, original_message: str = "") -> ChatResponse:
        """
        Build and return a personalized nutrition/meal recommendation using the user's
        profile, daily calorie/macro targets, logged intake today, and meal context.
        Supports English, Hindi, and Gujarati responses.
        """
        try:
            from bmr_tdee_calculator import compute_full_target

            msg = (original_message or "").lower()
            today = datetime.now(IST).date()
            user_id = self.user.get("user_id", "")

            # Get user calorie & macro target
            target = compute_full_target(self.user)
            calorie_target = target.calorie_target
            protein_target = getattr(target, "protein_target_g", getattr(target, "protein_g", 50.0))
            goal = self.user.get("goal", "maintain")

            # Get today's consumed food totals to find remaining calories
            consumed_cals = 0.0
            consumed_protein = 0.0
            try:
                food_totals = await self.log_repo.get_daily_food_totals(user_id, today)
                consumed_cals = float(food_totals.get("total_calories", 0.0) or 0.0)
                consumed_protein = float(food_totals.get("total_protein", 0.0) or 0.0)
            except Exception:
                pass

            remaining_cals = max(0, int(calorie_target - consumed_cals))
            remaining_protein = max(0, int(protein_target - consumed_protein))

            # Detect meal context
            meal_context = "general"
            if any(w in msg for w in ["post workout", "post-workout", "after workout", "after gym", "gym pachi", "gym baad"]):
                meal_context = "post_workout"
            elif any(w in msg for w in ["pre workout", "pre-workout", "before workout", "before gym", "gym pehle"]):
                meal_context = "pre_workout"
            elif any(w in msg for w in ["breakfast", "nashta", "nasto", "savare", "subah"]):
                meal_context = "breakfast"
            elif any(w in msg for w in ["lunch", "dopahar", "bapore", "bapor"]):
                meal_context = "lunch"
            elif any(w in msg for w in ["dinner", "raat", "ratre"]):
                meal_context = "dinner"
            elif any(w in msg for w in ["snack", "snacks", "sanje", "shaam"]):
                meal_context = "snack"

            # Recommendations by context and language
            if self.lang == "gu":
                title_map = {
                    "breakfast": "સવારના નાસ્તા (Breakfast) માટે સ્વસ્થ સૂચનો:",
                    "lunch": "બપોરના ભોજન (Lunch) માટે સંતુલિત સૂચનો:",
                    "dinner": "રાત્રિ ભોજન (Dinner) માટે હલકા અને પૌષ્ટિક સૂચનો:",
                    "snack": "સાંજના નાસ્તા (Snacks) માટે પૌષ્ટિક વિકલ્પો:",
                    "post_workout": "વર્કઆઉટ પછી (Post-Workout) પ્રોટીન-યુક્ત ખોરાક:",
                    "pre_workout": "વર્કઆઉટ પહેલાં (Pre-Workout) એનર્જી આપતો ખોરાક:",
                    "general": "તમારા લક્ષ્ય મુજબ સ્વસ્થ આહાર સૂચનો:",
                }
                options_map = {
                    "breakfast": [
                        "૧. વઘારેલા પૌંઆ (શેકેલા સીંગદાણા અને અંકુરિત મગ સાથે) (~૨૨૦ kcal, ૬g protein)",
                        "૨. બેસન અથવા મગની દાળનો પુડલો (પનીરના છીણ સાથે) (~૨૪૦ kcal, ૧૪g protein)",
                        "૩. ઓટ્સ દૂધ અને બદામ/ચિયા સીડ્સ સાથે (~૨૬૦ kcal, ૯g protein)",
                        "૪. બાફેલા ઈંડા (૨ નંગ) અથવા પનીર ભુરજી + ૧ રોટલી (~૨૫૦ kcal, ૧૫g protein)",
                    ],
                    "lunch": [
                        "૧. ૨ ઘઉં/બાજરીની રોટલી + ૧ વાટકી મિક્સ દાળ/કઠોળ + ૧ વાટકી શાક + કાચું સલાડ (~૩૮૦ kcal, ૧૬g protein)",
                        "૨. દાળ, બ્રાઉન રાઇસ / ખીચડી, લીલું શાક અને ૧ ગ્લાસ છાશ (~૩૫૦ kcal, ૧૨g protein)",
                        "૩. પનીર / સોયા સબ્જી, ૧ રોટલો અને તાજી છાશ (~૪૨૦ kcal, ૨૦g protein)",
                    ],
                    "dinner": [
                        "૧. મગની દાળની હલકી ખીચડી, શેકેલો પાપડ અને તાજી છાશ (~૩૦૦ kcal, ૧૦g protein)",
                        "૨. દૂધી/પાલક સૂપ સાથે પનીર અથવા ટોફુ ટિક્કા (~૨૪૦ kcal, ૧૬g protein)",
                        "૩. ૧-૨ ફુલકા રોટલી + વઘારેલું કઠોળ (મગ/ચણા) + કાકડી/ટામેટા સલાડ (~૩૨૦ kcal, ૧૩g protein)",
                    ],
                    "snack": [
                        "૧. શેકેલા મખાના (૧ વાટકી) (~૧૧૦ kcal, ૩g protein)",
                        "૨. બાફેલા ચણા/મગની ચાટ (લીંબુ અને કાકડી સાથે) (~૧૫૦ kcal, ૭g protein)",
                        "૩. ૧ ગ્લાસ જીરા છાશ અને મુઠ્ઠીભર શેકેલા ચણા (~૧૩૦ kcal, ૬g protein)",
                    ],
                    "post_workout": [
                        "૧. વ્હે પ્રોટીન શેક (૧ સ્કૂપ) + ૧ કેળું (~૨૨૦ kcal, ૨૬g protein)",
                        "૨. ૩ બાફેલા ઈંડા (સફેદ ભાગ) + ૧ ટોસ્ટ (~૧૬૦ kcal, ૧૪g protein)",
                        "૩. પનીર ભુરજી (૧૦૦g) અથવા શેકેલા સોયા ચંક્સ (~૧૮૦ kcal, ૧૮g protein)",
                        "૪. મીઠો વગરનું દહીં/ગ્રીક યોગર્ટ ડ્રાયફ્રૂટ્સ સાથે (~૧૭૦ kcal, ૧૨g protein)",
                    ],
                    "pre_workout": [
                        "૧. ૧ કેળું અને ૫-૬ બદામ (~૧૪૦ kcal, ૩g protein)",
                        "૨. ૨ ખજૂર પીનટ બટર સાથે (~૧૫૦ kcal, ૪g protein)",
                        "૩. હલકી ઓટ્સ પોરીજ (~૧૬૦ kcal, ૫g protein)",
                    ],
                    "general": [
                        "૧. પ્રોટીન-યુક્ત આહાર: દાળ, કઠોળ, પનીર, ઈંડા અને દહીંનો ઉપયોગ વધારો.",
                        "૨. વધુ ફાઇબર અને વિટામિન્સ માટે દરરોજ પુષ્કળ સલાડ અને શાકભાજી લો.",
                        "૩. તળેલા અને ખાંડવાળા ખોરાકથી દૂર રહો.",
                    ],
                }
                body = f"🥗 **{title_map[meal_context]}**\n\n"
                body += "\n".join(options_map[meal_context])
                body += f"\n\n📊 *દૈનિક લક્ષ્ય: {calorie_target} kcal | બાકી: ~{remaining_cals} kcal | પ્રોટીન લક્ષ્ય: {protein_target}g*"

            elif self.lang == "hi":
                title_map = {
                    "breakfast": "सुबह के नाश्ते (Breakfast) के लिए पौष्टिक सुझाव:",
                    "lunch": "दोपहर के खाने (Lunch) के लिए संतुलित सुझाव:",
                    "dinner": "रात के खाने (Dinner) के लिए हल्के और हेल्दी सुझाव:",
                    "snack": "शाम के नाश्ते (Snacks) के लिए हेल्दी ऑप्शंस:",
                    "post_workout": "वर्कआउट के बाद (Post-Workout) प्रोटीन से भरपूर डाइट:",
                    "pre_workout": "वर्कआउट से पहले (Pre-Workout) एनर्जी देने वाला खाना:",
                    "general": "आपके फिटनेस गोल के अनुसार हेल्दी डाइट सुझाव:",
                }
                options_map = {
                    "breakfast": [
                        "1. पोहा (मूंगफली और अंकुरित मूंग के साथ) (~220 kcal, 6g protein)",
                        "2. बेसन या मूंग दाल का चीला पनीर स्टफिंग के साथ (~240 kcal, 14g protein)",
                        "3. ओट्स दूध, बादाम और चिया सीड्स के साथ (~260 kcal, 9g protein)",
                        "4. 2 उबले अंडे / पनीर भुर्जी + 1 मल्टीग्रेन टोस्ट (~250 kcal, 15g protein)",
                    ],
                    "lunch": [
                        "1. 2 रोटी + 1 कटोरी दाल/राजमा/छोले + 1 कटोरी हरी सब्जी + सलाद (~380 kcal, 16g protein)",
                        "2. ब्राउन राइस/दाल खिचड़ी + ताजी छाछ + सलाद (~350 kcal, 12g protein)",
                        "3. पनीर/सोया भुर्जी + 2 फुल्का रोटी + खीरा-टमाटर सलाद (~400 kcal, 20g protein)",
                    ],
                    "dinner": [
                        "1. मूंग दाल खिचड़ी, भुना हुआ पापड़ और दही/छाछ (~300 kcal, 10g protein)",
                        "2. ग्रिल्ड पनीर/टोफू और सौते की हुई सब्जियां (~240 kcal, 16g protein)",
                        "3. 1-2 रोटी + पालक दाल + ताजा सलाद (~320 kcal, 13g protein)",
                    ],
                    "snack": [
                        "1. रोस्टेड मखाना (1 कटोरी) (~110 kcal, 3g protein)",
                        "2. उबले चने की चाट (नींबू और प्याज-टमाटर के साथ) (~150 kcal, 7g protein)",
                        "3. 1 ग्लास जीरा छाछ + मुट्ठीभर भुने चने (~130 kcal, 6g protein)",
                    ],
                    "post_workout": [
                        "1. व्हे प्रोटीन शेक (1 स्कूप) + 1 केला (~220 kcal, 26g protein)",
                        "2. 3 उबले अंडे (सफेद भाग) + 1 ब्राउन ब्रेड टोस्ट (~160 kcal, 14g protein)",
                        "3. पनीर भुर्जी (100g) या सत्तू ड्रिंक (~180 kcal, 16g protein)",
                        "4. ग्रीक योगर्ट / ताजा दही ड्राईफ्रूट्स के साथ (~170 kcal, 12g protein)",
                    ],
                    "pre_workout": [
                        "1. 1 केला और 5-6 बादाम (~140 kcal, 3g protein)",
                        "2. 2 खजूर पीनट बटर के साथ (~150 kcal, 4g protein)",
                        "3. हल्की ओट्स कटोरी (~160 kcal, 5g protein)",
                    ],
                    "general": [
                        "1. प्रोटीन स्रोतों (दाल, पनीर, सोया, अंडे) को हर भोजन में शामिल करें।",
                        "2. रिफाइंड चीनी और ज्यादा तेल-मसाले से बचें।",
                        "3. दिन भर में पर्याप्त पानी और फाइबर युक्त सलाद लें।",
                    ],
                }
                body = f"🥗 **{title_map[meal_context]}**\n\n"
                body += "\n".join(options_map[meal_context])
                body += f"\n\n📊 *दैनिक लक्ष्य: {calorie_target} kcal | शेष: ~{remaining_cals} kcal | प्रोटीन लक्ष्य: {protein_target}g*"

            else:
                title_map = {
                    "breakfast": "Nutritious & High-Protein Breakfast Ideas:",
                    "lunch": "Balanced & Wholesome Lunch Suggestions:",
                    "dinner": "Light, Nourishing Dinner Recommendations:",
                    "snack": "Smart & Healthy Snack Options:",
                    "post_workout": "Post-Workout Recovery & Protein Fuel:",
                    "pre_workout": "Pre-Workout Energy Boost Options:",
                    "general": "Personalized Healthy Meal Recommendations:",
                }
                options_map = {
                    "breakfast": [
                        "1. Moong Dal or Besan Chilla with grated paneer (~240 kcal, 14g protein)",
                        "2. Rolled Oats porridge with milk, almonds & chia seeds (~260 kcal, 9g protein)",
                        "3. 2 Boiled eggs / Paneer bhurji with 1 slice whole wheat toast (~250 kcal, 15g protein)",
                        "4. Vegetable Poha with roasted peanuts & sprouted pulses (~220 kcal, 6g protein)",
                    ],
                    "lunch": [
                        "1. 2 Whole wheat phulkas + 1 bowl Dal/Rajma/Chole + green sabzi + fresh salad (~380 kcal, 16g protein)",
                        "2. Brown rice bowl with grilled paneer/tofu or chicken breast & steamed veggies (~420 kcal, 24g protein)",
                        "3. Moong dal khichdi with 1 glass probiotic buttermilk (chaas) & cucumber salad (~350 kcal, 12g protein)",
                    ],
                    "dinner": [
                        "1. Light yellow dal with sautéed spinach & 1-2 thin rotis (~310 kcal, 13g protein)",
                        "2. Grilled paneer/tofu or chicken salad with olive oil & lemon dressing (~260 kcal, 18g protein)",
                        "3. Vegetable soup with boiled chickpea / edamame bowl (~220 kcal, 11g protein)",
                    ],
                    "snack": [
                        "1. Roasted makhana (foxnuts) with a pinch of rock salt (~110 kcal, 3g protein)",
                        "2. Sprouted moong or boiled kala chana chaat (~150 kcal, 7g protein)",
                        "3. 1 glass spiced buttermilk (chaas) + handful roasted grams (~130 kcal, 6g protein)",
                    ],
                    "post_workout": [
                        "1. 1 scoop Whey protein shake + 1 medium banana (~220 kcal, 26g protein)",
                        "2. 3 Boiled egg whites + 1 slice whole grain toast (~160 kcal, 14g protein)",
                        "3. 100g Low-fat Paneer / Tofu scramble with veggies (~180 kcal, 16g protein)",
                        "4. Greek yogurt with mixed berries & pumpkin seeds (~170 kcal, 14g protein)",
                    ],
                    "pre_workout": [
                        "1. 1 Medium banana + 5-6 soaked almonds (~140 kcal, 3g protein)",
                        "2. 2 Medjool dates with 1 tsp natural peanut butter (~150 kcal, 4g protein)",
                        "3. Half cup warm oatmeal (~150 kcal, 4g protein)",
                    ],
                    "general": [
                        "1. Focus on adequate protein (paneer, eggs, lentils, soy, dairy) across all meals.",
                        "2. Keep refined sugar and deep-fried items minimal.",
                        "3. Stay well-hydrated and include generous fresh salads for micronutrients.",
                    ],
                }
                body = f"🥗 **{title_map[meal_context]}**\n\n"
                body += "\n".join(options_map[meal_context])
                body += f"\n\n📊 *Daily Target: {calorie_target} kcal | Remaining: ~{remaining_cals} kcal | Protein Target: {protein_target}g*"

            return ChatResponse(
                message=body,
                intent=Intent.RECOMMEND_MEAL,
                data={
                    "meal_context": meal_context,
                    "target_calories": calorie_target,
                    "remaining_calories": remaining_cals,
                    "target_protein_g": protein_target,
                    "goal": goal,
                },
                success=True,
                language=self.lang,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Meal recommendation failed: %s", e)
            return ChatResponse(
                message="Here are some healthy meal ideas: try high-protein options like moong dal chilla, paneer salad, oats, or dal with whole wheat roti.",
                intent=Intent.RECOMMEND_MEAL,
                success=True,
                language=self.lang,
            )

    async def _handle_recommend_workout(self) -> ChatResponse:
        """
        Build and return a workout recommendation using today's data,
        7-day exercise history, user goal/profile, and calorie/protein status.
        Returns a graceful error message if anything fails — never raises.
        """
        try:
            from bmr_tdee_calculator import compute_full_target
            from datetime import date as _date, timedelta as _td

            today     = datetime.now(IST).date()
            week_ago  = today - _td(days=7)
            user_id   = self.user["user_id"]

            # Gather all data in parallel via asyncio
            import asyncio
            food_totals_t, exercise_totals_t, exercise_logs_7d = await asyncio.gather(
                self.log_repo.get_daily_food_totals(user_id, today),
                self.log_repo.get_daily_exercise_totals(user_id, today),
                self.log_repo.get_exercise_logs_by_range(user_id, week_ago, today),
            )

            calorie_target = compute_full_target(self.user)

            rec = build_workout_recommendation(
                db                    = sync_db,
                user                  = self.user,
                exercise_logs_7d      = exercise_logs_7d,
                today_food_totals     = food_totals_t,
                today_exercise_totals = exercise_totals_t,
                calorie_target        = calorie_target,
            )

            body = format_recommendation(rec, lang=self.lang)

            return ChatResponse(
                message   = body,
                intent    = Intent.RECOMMEND_WORKOUT,
                data      = {
                    "today_burned_kcal":   rec.today_burned_cal,
                    "recommended_cats":    [b.category for b in rec.recommended_blocks],
                    "underfuelled":        rec.underfuelled,
                    "protein_warning":     rec.protein_warning,
                    "blocked_groups":      list(rec.blocked_groups.keys()),
                },
                success   = True,
                language  = self.lang,
            )

        except Exception as e:
            # Recommendation is a bonus — never surface a crash to the user
            import logging
            logging.getLogger(__name__).warning("Workout recommendation failed: %s", e)

            def _L(en, hi, gu):
                if self.lang == "gu": return gu
                if self.lang == "hi": return hi
                return en

            return ChatResponse(
                message = _L(
                    "Sorry, I couldn't generate a workout recommendation right now. "
                    "Try logging some exercise first so I can learn your training history.",
                    "Abhi workout recommendation generate nahi ho saki. "
                    "Pehle kuch exercise log karein taaki main aapki history samajh sakoon.",
                    "Abhi workout recommendation bani shaki nahi. "
                    "Pahela koi exercise log karo jyathi hu tamari history samaji shaku.",
                ),
                intent   = Intent.RECOMMEND_WORKOUT,
                success  = False,
                language = self.lang,
            )

    def _handle_greeting(self) -> ChatResponse:
        name = self.user["name"].split()[0] if self.user.get("name") else "there"
        target = compute_full_target(self.user)
        return ChatResponse(
            message=get_response("greeting", self.lang, name=name, target=target.calorie_target),
            intent=Intent.GREETING, data={"calorie_target": target.calorie_target},
            success=True, language=self.lang)

    async def _handle_log_food(self, parsed: LLMParseResult, auto_log: bool, original_message: str = "") -> ChatResponse:
        # ── Multi-food from the LLM's foods[] array ────────────────────────
        # The LLM extracts EVERY food it understood (each with its own quantity,
        # variant and meal_type). If it found more than one, log them all via the
        # sequential queue so NOTHING is dropped or merged. This complements the
        # regex splitter (which ran earlier in process_message) and covers
        # sentences the regex missed. Guarded to >1 so single-food is unchanged.
        llm_foods = [f for f in (getattr(parsed, "foods", None) or [])
                     if isinstance(f, dict) and (f.get("food_query") or "").strip()]
        if len(llm_foods) > 1:
            multi_resp = await self._handle_multi_food_from_llm(llm_foods, parsed, original_message)
            if multi_resp is not None:
                return multi_resp
            # else: fall through to single-food handling (nothing resolved)

        food_query = self._clean_food_query(parsed.food_query or "", original_message=original_message)
        # Fallback if food_query is empty or didn't match (e.g. Qwen put "2 idli" in quantity and "regular this" in food_query)
        s_fq = self.search_engine.search(food_query) if food_query else None
        if not s_fq or s_fq.match_type.value not in ("exact_name", "exact_alias", "variant_strip"):
            if parsed.quantity:
                qty_parsed = parse_quantity(parsed.quantity)
                if qty_parsed.unit:
                    unit_search = self.search_engine.search(qty_parsed.unit)
                    if unit_search.match_type.value in ("exact_name", "exact_alias", "variant_strip"):
                        food_query = qty_parsed.unit
                        s_fq = unit_search
            if (not s_fq or s_fq.match_type.value not in ("exact_name", "exact_alias", "variant_strip")) and original_message:
                orig_clean = self._clean_food_query(original_message, original_message=original_message)
                if orig_clean:
                    orig_search = self.search_engine.search(orig_clean)
                    if orig_search.match_type.value in ("exact_name", "exact_alias", "variant_strip"):
                        food_query = orig_clean

        if not food_query:
            return ChatResponse(message=get_response("what_food", self.lang),
                                intent=Intent.CLARIFICATION_NEEDED, success=True, language=self.lang)

        # ── GENERIC FOOD CLARIFICATION TIER ────────────────────────────────
        # Standalone generic category terms (e.g. "shak", "curry", "sabji", "fruit",
        # "sweet", "juice", "snack", "soup") must NEVER resolve to an arbitrary
        # concrete or branded guess (like Beef Curry or French cereal). Clarify!
        is_gen, gen_cat = is_generic_food_query(original_message or food_query)
        if is_gen:
            clarify_msg = get_generic_clarification_message(gen_cat, self.lang)
            return ChatResponse(
                message=clarify_msg,
                intent=Intent.CLARIFICATION_NEEDED,
                success=True,
                language=self.lang,
            )

        # Meal type is decided by the SEMANTIC meaning of the time expression.
        # The LLM is the authority for open-ended, language-agnostic inference
        # (it understands unknown/colloquial/dialect/misspelled/STT time words
        # from context and, per the prompt's confidence rule, leaves meal_type
        # null when it cannot understand a time expression — so we ask instead
        # of guessing). The local classifier is only a deterministic fast-path /
        # offline fallback and must NEVER override the LLM's semantic result.
        # Priority:
        #   1. LLM meal_type (semantic understanding, incl. relative expressions).
        #   2. LLM time_of_day mapped to a meal.
        #   3. Local classifier (used mainly when the LLM is unavailable).
        # When none resolve, meal_type stays None and the queue asks the user
        # (clarification) rather than guessing.
        local_meal = self._detect_meal_type(original_message) or self._detect_meal_type(food_query or "")
        meal_type = (
            parsed.meal_type
            or self._meal_from_time_of_day(getattr(parsed, "time_of_day", None))
            or local_meal
        )
        if meal_type == "snack" and local_meal == "dinner":
            meal_type = "dinner"

        # If no meal was detected from this message, reuse the meal from the most
        # recent food log in this session (within 30 min) so the bot doesn't ask
        # the same meal question repeatedly when logging several foods in a row.
        if not meal_type:
            meal_type = await self.memory.get_last_meal(self.user["user_id"])

        q_obj = await self._build_food_item_queue_object(food_query, original_message or food_query, shared_meal_type=meal_type)
        if not q_obj:
            # Nothing matched the DB or a valid external source → reject.
            # Choose the wording deterministically (no LLM): if the text is
            # clearly non-food/gibberish, say "couldn't identify a food";
            # otherwise it's a plausible name we just don't have → "not found".
            try:
                clearly_non_food = self.llm._looks_like_non_food(food_query) is True
            except Exception:
                clearly_non_food = False
            key = "not_food" if clearly_non_food else "food_not_found"
            return ChatResponse(message=get_response(key, self.lang, food=food_query),
                                intent=Intent.CLARIFICATION_NEEDED, success=True, language=self.lang)

        flow = PendingFlow(
            items=[q_obj],
            current_index=0,
            shared_meal_type=meal_type,
        )
        await self.flow_mgr.save_flow_state(flow)
        return await self._process_sequential_queue(flow)

    async def _handle_multi_food_from_llm(self, llm_foods: list, parsed: LLMParseResult,
                                          original_message: str) -> Optional[ChatResponse]:
        """
        Build a sequential-queue flow from the LLM's foods[] array so multiple
        foods (in any meal) are all logged — none dropped, none merged. Each item
        keeps its OWN quantity / variant / meal_type as extracted by the LLM.

        Returns a ChatResponse, or None if NOTHING resolved (caller then falls
        back to single-food handling).
        """
        # Sentence-level meal type used only as a fallback for items the LLM
        # didn't tag individually.
        local_meal = self._detect_meal_type(original_message)
        shared_meal = (
            parsed.meal_type
            or self._meal_from_time_of_day(getattr(parsed, "time_of_day", None))
            or local_meal
        )
        if shared_meal == "snack" and local_meal == "dinner":
            shared_meal = "dinner"

        queue_items: list[dict] = []
        failed_items: list[str] = []
        seen_ids: set = set()

        for f in llm_foods:
            name = (f.get("food_query") or "").strip()
            if not name:
                continue
            qty = (f.get("quantity") or "").strip()
            variant = (f.get("variant") or "").strip()
            item_meal = (f.get("meal_type") or "").strip() or shared_meal
            # Synthesize a raw item string so the existing resolver picks up the
            # LLM-supplied quantity + variant (e.g. "2 pieces idli ghee").
            raw_parts = [p for p in (qty, name, variant) if p]
            raw_item = " ".join(raw_parts) if raw_parts else name

            q_obj = await self._build_food_item_queue_object(name, raw_item, shared_meal_type=item_meal)
            if q_obj:
                # De-duplicate identical foods, but keep distinct items intact.
                fid = q_obj.get("food_id")
                if fid in seen_ids:
                    continue
                seen_ids.add(fid)
                queue_items.append(q_obj)
            else:
                failed_items.append(name)

        if not queue_items:
            return None  # let caller try single-food handling

        flow = PendingFlow(
            items=queue_items,
            current_index=0,
            shared_meal_type=shared_meal,
        )
        if failed_items:
            flow.unresolved_items = failed_items
        await self.flow_mgr.save_flow_state(flow)
        return await self._process_sequential_queue(flow)

    def _user_weight_kg(self) -> Optional[float]:
        """The user's ACTUAL body weight (kg) from their saved profile, used to
        scale exercise calories burned. Returns None only when the profile has
        no usable weight — in which case the calculator falls back to the
        reference weight (no scaling) rather than a fabricated default."""
        raw = self.user.get("weight_kg", self.user.get("weight"))
        try:
            w = float(raw)
            return w if w > 0 else None
        except (TypeError, ValueError):
            return None

    async def _handle_log_exercise(self, parsed: LLMParseResult, auto_log: bool, original_message: str = "") -> ChatResponse:
        # ── Structured workout routine check (multi-exercise/muscle-group) ──
        from exercise_calculator import parse_workout_routine
        routine = None
        check_text = original_message or parsed.exercise_input or ""
        if parsed.exercises or check_text:
            routine = parse_workout_routine(check_text)
        if routine and routine.items:
            return await self._handle_workout_routine(routine, check_text, auto_log=auto_log)

        exercise_input = parsed.exercise_input
        if not exercise_input:
            return ChatResponse(message="What exercise did you do? Include duration or reps.",
                                intent=Intent.CLARIFICATION_NEEDED, success=True, language=self.lang)

        ex_parsed = parse_exercise_input(exercise_input)
        if not ex_parsed.exercise_query:
            return ChatResponse(message="I couldn't understand the exercise. Try again with name and duration.",
                                intent=Intent.CLARIFICATION_NEEDED, success=True, language=self.lang)

        # ── Connector-word fallback: "pn me 6 set marya" ──────────────────
        # If the exercise_query is a known connector/filler word (pn, pan, ane)
        # and an amount was detected, this is a follow-up log for the last
        # exercise stored in conversation memory. Retrieve it.
        _CONNECTOR_NOISE = {"pn", "pan", "ane", "also", "too", "again", "more"}
        if ex_parsed.exercise_query.lower() in _CONNECTOR_NOISE and ex_parsed.amount:
            last_ex_name = await self.memory.get_last_exercise(self.user["user_id"])
            if last_ex_name:
                ex_parsed = ParsedExerciseInput(
                    exercise_query=last_ex_name,
                    amount=ex_parsed.amount,
                    unit=ex_parsed.unit,
                    raw_input=exercise_input,
                )
            else:
                return ChatResponse(
                    message=get_response("what_exercise", self.lang) if hasattr(
                        __builtins__, "get_response") else
                    "Which exercise are you referring to? I couldn't find your previous activity.",
                    intent=Intent.CLARIFICATION_NEEDED, success=True, language=self.lang,
                )

        # Disallow food portion units / containers from being logged as exercises
        _FOOD_PORTION_UNITS = {
            "plate", "plates", "bowl", "bowls", "cup", "cups", "glass", "glasses",
            "slice", "slices", "piece", "pieces", "pcs", "serving", "servings",
            "katori", "katoris", "vatki", "g", "gm", "gms", "gram", "grams", "kg",
            "ml", "spoon", "spoons", "tbsp", "tsp", "tablespoon", "tablespoons",
            "teaspoon", "teaspoons", "rotlo", "rotla", "roti", "rotis", "thepla",
            "theplas", "bhakri", "bhakhri", "portion", "portions", "packet", "packets",
        }
        if ex_parsed.exercise_query.lower() in _FOOD_PORTION_UNITS:
            return ChatResponse(
                message=get_response("what_food", self.lang),
                intent=Intent.CLARIFICATION_NEEDED,
                success=True,
                language=self.lang,
            )

        exercise, confidence = self.exercise_searcher.search(ex_parsed.exercise_query)

        # MongoDB miss → try external sources (ExerciseDB → wger → Compendium → AI)
        if not exercise:
            exercise, confidence = await self._resolve_exercise_external(ex_parsed.exercise_query)

        if not exercise:
            return ChatResponse(message=f"I couldn't find exercise '{ex_parsed.exercise_query}'. "
                                        f"Try jogging, yoga, push-ups, cycling, etc.",
                                intent=Intent.CLARIFICATION_NEEDED, success=True, language=self.lang)

        # Preserve the user's EXACT exercise wording when it's a richer phrase than
        # the resolved canonical name (e.g. "power yoga" resolving to "yoga",
        # "incline push-ups" → "push-ups"). Never substitute a generic display
        # for what the user actually said. Prefer the LLM-provided exact_term,
        # else the parsed query. Falls back to the DB display name unchanged.
        exercise = dict(exercise)
        user_term = (parsed.exact_term or ex_parsed.exercise_query or "").strip()
        preserved = self._display_phrase_from_query(
            user_term.lower(), (exercise.get("exercise_name") or "").lower()
        )
        if preserved:
            exercise["exercise_name_display"] = preserved

        # ── Scenario 2: no amount given → ask "ketla pushups mara?" ──────────
        if ex_parsed.amount is None and not auto_log:
            native_unit = exercise.get("measurement_unit", "reps")
            # Store exercise context in flow so we can complete it after the user replies
            flow = PendingFlow(
                state=FlowState.AWAITING_EXERCISE_AMOUNT,
                exercise_input=exercise_input,
                exercise_data={
                    "exercise_id":           exercise["exercise_id"],
                    "exercise_name":         exercise["exercise_name"],
                    "exercise_name_display": exercise["exercise_name_display"],
                    "category":              exercise["category"],
                    "native_unit":           native_unit,
                    "search_confidence":     confidence,
                    "log_date":              self._resolve_date(parsed.date).isoformat(),
                    "user_id":               self.user["user_id"],
                },
            )
            await self.flow_mgr.save_flow_state(flow)

            msg = get_response(
                "ask_exercise_amount", self.lang,
                exercise=exercise["exercise_name_display"],
                unit=native_unit,
            )
            return ChatResponse(
                message=msg,
                intent=f"flow:{FlowState.AWAITING_EXERCISE_AMOUNT}",
                success=True, language=self.lang,
            )

        # ── Scenario 1: amount given → calculate and show confirmation ────────
        # Burn scales with the user's ACTUAL weight (heavier → more calories).
        calc_result = ExerciseCalculator().calculate(
            exercise, amount=ex_parsed.amount, unit=ex_parsed.unit,
            weight_kg=self._user_weight_kg(),
        )
        log_date = self._resolve_date(parsed.date)

        pending = {
            "type": "log_exercise", "user_id": self.user["user_id"],
            "log_date": log_date.isoformat(),
            "exercise_id": exercise["exercise_id"], "exercise_name": exercise["exercise_name"],
            "exercise_name_display": exercise["exercise_name_display"], "category": exercise["category"],
            "exercise_input": exercise_input, "amount": calc_result.amount, "unit": calc_result.unit,
            "calories_min": calc_result.calories_min, "calories_avg": calc_result.calories_avg,
            "calories_max": calc_result.calories_max, "search_confidence": confidence,
        }

        if auto_log:
            return await self._execute_exercise_log(pending)

        msg = get_response("confirm_exercise", self.lang,
                           exercise=exercise["exercise_name_display"],
                           amount=calc_result.amount, unit=calc_result.unit,
                           calories=calc_result.calories_avg)
        msg += get_response("safety_disclaimer", self.lang)

        return ChatResponse(message=msg, intent=Intent.LOG_EXERCISE, data={"exercise": pending},
                            needs_confirmation=True, pending_action=pending, success=True, language=self.lang)

    async def _handle_workout_routine(
        self, routine, original_message: str, auto_log: bool = False
    ) -> ChatResponse:
        """
        Handle structured multi-exercise / multi-muscle-group workout routine.
        Calculates calories from reps/sets or MET without inventing duration.
        """
        from exercise_calculator import calculate_routine
        calc_res = calculate_routine(routine, weight_kg=self._user_weight_kg())

        # If neither reps nor duration were provided, ask for amount
        if routine.total_reps == 0 and routine.duration_min is None and calc_res.calories_avg <= 0:
            msg = get_response("ask_routine_amount", self.lang)
            return ChatResponse(
                message=msg,
                intent=f"flow:{FlowState.AWAITING_EXERCISE_AMOUNT}",
                success=True,
                language=self.lang,
            )

        log_date = self._resolve_date("today")
        pending = {
            "type": "log_exercise",
            "user_id": self.user["user_id"],
            "log_date": log_date.isoformat(),
            "exercise_id": "workout_routine",
            "exercise_name": "Workout Routine",
            "exercise_name_display": routine.summary_text,
            "category": "Strength",
            "exercise_input": routine.summary_text,
            "amount": routine.total_reps or (routine.duration_min if routine.duration_min is not None else 1),
            "unit": "reps" if routine.total_reps else ("minutes" if routine.duration_min is not None else "session"),
            "duration_min": routine.duration_min,
            "reps": routine.total_reps,
            "sets": routine.total_sets,
            "routine_items": [
                {
                    "muscle_group": it.muscle_group,
                    "exercise_name": it.exercise_name,
                    "exercise_count": it.exercise_count,
                    "sets": it.sets,
                    "reps": it.reps,
                    "duration_min": it.duration_min,
                }
                for it in routine.items
            ],
            "calories_min": calc_res.calories_min,
            "calories_avg": calc_res.calories_avg,
            "calories_max": calc_res.calories_max,
            "search_confidence": 1.0,
        }

        if auto_log:
            return await self._execute_exercise_log(pending)

        flow = PendingFlow(
            state=FlowState.AWAITING_CONFIRMATION,
            exercise_input=routine.summary_text,
            exercise_data=pending,
        )
        await self.flow_mgr.save_flow_state(flow)

        rep_info = f" — {routine.total_reps} reps" if routine.total_reps and "reps" not in routine.summary_text else ""
        if routine.duration_min is not None:
            rep_info += f", {int(routine.duration_min)} minutes"

        msg = get_response(
            "confirm_workout_routine", self.lang,
            routine_summary=f"{routine.summary_text}{rep_info}",
            calories=calc_res.calories_avg,
        )
        msg += get_response("safety_disclaimer", self.lang)

        return ChatResponse(
            message=msg,
            intent=Intent.LOG_EXERCISE,
            data={"exercise": pending},
            needs_confirmation=True,
            pending_action=pending,
            options=[
                get_option("yes_save", self.lang),
                get_option("no_cancel", self.lang),
            ],
            success=True,
            language=self.lang,
        )

    async def _resolve_exercise_external(self, query: str) -> tuple[Optional[dict], float]:
        """
        Resolve an exercise the local MongoDB couldn't confidently match, using
        the external pipeline:  ExerciseDB → wger → Compendium → AI estimate.

        Validated results are cached into the local `exercises` collection so
        repeated queries never re-hit the APIs. Returns (exercise_doc, confidence)
        shaped for ExerciseCalculator, or (None, 0.0). Mirrors the food 4-tier
        chain in _build_food_item_queue_object. Never exposes source details.
        """
        q = (query or "").strip()
        if not q:
            return None, 0.0

        _FOOD_PORTION_UNITS = {
            "plate", "plates", "bowl", "bowls", "cup", "cups", "glass", "glasses",
            "slice", "slices", "piece", "pieces", "pcs", "serving", "servings",
            "katori", "katoris", "vatki", "g", "gm", "gms", "gram", "grams", "kg",
            "ml", "spoon", "spoons", "tbsp", "tsp", "tablespoon", "tablespoons",
            "teaspoon", "teaspoons", "rotlo", "rotla", "roti", "rotis", "thepla",
            "theplas", "bhakri", "bhakhri", "portion", "portions", "packet", "packets",
        }
        if q.lower() in _FOOD_PORTION_UNITS:
            return None, 0.0

        weight_kg = float(self.user.get("weight_kg") or self.user.get("weight") or 70.0)

        # ── Tiers 1-3: ExerciseDB → wger → Compendium ────────────────────
        exercise_doc = None
        try:
            exercise_doc = await self.external_exercise_api.resolve(q, weight_kg=weight_kg)
        except Exception:
            exercise_doc = None

        # ── Tier 4: AI estimate ──────────────────────────────────────────
        if not exercise_doc:
            try:
                est = await self.llm.estimate_exercise_met(q)
            except Exception:
                est = None
            if est and est.get("met"):
                from external_exercise_api import _build_exercise_doc
                exercise_doc = _build_exercise_doc(
                    canonical_name=q,
                    display_name=est.get("exercise_name_display", q.title()),
                    met=float(est["met"]),
                    category=est.get("category", "Fitness"),
                    measurement_unit=est.get("measurement_unit", "minutes"),
                    source="ai_estimated",
                    weight_kg=weight_kg,
                )
                if exercise_doc:
                    exercise_doc["is_ai_estimated"] = True

        if not exercise_doc:
            return None, 0.0

        # Cache the validated external/AI exercise into MongoDB (overwrite-safe).
        try:
            stored = self.exercise_searcher.save_learned_exercise_sync(q, exercise_doc)
            if stored:
                exercise_doc = stored
        except Exception:
            pass

        # External/AI matches are treated as medium confidence.
        return exercise_doc, 0.6

    async def _handle_get_summary(self, parsed: LLMParseResult) -> ChatResponse:
        log_date = self._resolve_date(parsed.date)
        target = compute_full_target(self.user)
        food_totals = await self.log_repo.get_daily_food_totals(self.user["user_id"], log_date)
        exercise_totals = await self.log_repo.get_daily_exercise_totals(self.user["user_id"], log_date)

        consumed = food_totals["total_calories_consumed"]
        burned = exercise_totals["total_calories_burned"]
        net = consumed - burned
        remaining = target.calorie_target - net

        msg = get_response("summary", self.lang, date=log_date.isoformat(),
                           consumed=consumed, food_count=food_totals["food_entries"],
                           burned=burned, ex_count=exercise_totals["exercise_entries"],
                           net=net, remaining=remaining, target=target.calorie_target,
                           protein=food_totals["total_protein_g"],
                           carbs=food_totals["total_carbs_g"], fat=food_totals["total_fat_g"])
        msg += get_response("safety_disclaimer", self.lang)

        return ChatResponse(message=msg, intent=Intent.GET_SUMMARY, data={
            "date": log_date.isoformat(), "calorie_target": target.calorie_target,
            "consumed": consumed, "burned": burned, "net": net, "remaining": remaining,
            "protein_g": food_totals["total_protein_g"], "carbs_g": food_totals["total_carbs_g"],
            "fat_g": food_totals["total_fat_g"],
        }, success=True, language=self.lang)

    async def _handle_query_meal(self, parsed: LLMParseResult) -> ChatResponse:
        """Handle 'aaje breakfast ma su lidhu?' — show today's logged items for that meal."""
        meal_type = parsed.meal_type
        log_date = self._resolve_date(parsed.date)

        food_logs = await self.log_repo.get_food_logs_by_date(self.user["user_id"], log_date)

        # Filter by meal type if specified
        if meal_type:
            meal_logs = [l for l in food_logs if l.get("meal_type") == meal_type and l.get("food_id") != "skipped"]
        else:
            meal_logs = [l for l in food_logs if l.get("food_id") != "skipped"]

        meal_label = meal_type or "all meals"

        if not meal_logs:
            # No logs found for this meal
            IST_TZ = timezone(timedelta(hours=5, minutes=30))
            today_ist = datetime.now(IST_TZ).date()
            if log_date == today_ist:
                day_word_gu, day_word_hi, day_word_en = "aaje", "aaj", "today"
            elif log_date == today_ist - timedelta(days=1):
                day_word_gu, day_word_hi, day_word_en = "gay kale", "kal", "yesterday"
            else:
                day_word_gu = day_word_hi = day_word_en = log_date.isoformat()

            if self.lang == "gu":
                msg = f"Tame {day_word_gu} {meal_label} log nathi karyu. Shu tame {meal_label} add karva mango cho?"
            elif self.lang == "hi":
                msg = f"Aapne {day_word_hi} {meal_label} log nahi kiya hai. Kya aap {meal_label} add karna chahte hain?"
            else:
                msg = f"You haven't logged any {meal_label} {day_word_en}. Would you like to add something?"
            return ChatResponse(
                message=msg, intent=Intent.QUERY_MEAL,
                options=[get_option("yes_add", self.lang), get_option("no_skip", self.lang)],
                data={"meal_type": meal_type, "logged_count": 0, "date": log_date.isoformat()},
                success=True, language=self.lang,
            )

        # Build response with all logged items
        total_cal = sum(l.get("calories", 0) or 0 for l in meal_logs)
        total_protein = sum(l.get("protein_g", 0) or 0 for l in meal_logs)
        total_carbs = sum(l.get("carbs_g", 0) or 0 for l in meal_logs)
        total_fat = sum(l.get("fat_g", 0) or 0 for l in meal_logs)

        if meal_type:
            items_text = "\n".join(self._format_food_log_item(l) for l in meal_logs)
        else:
            by_meal: dict[str, list[str]] = {}
            for l in meal_logs:
                m = (l.get("meal_type") or "snack").strip().lower()
                by_meal.setdefault(m, []).append(self._format_food_log_item(l))
            ordered_meals = sorted(by_meal.keys(), key=meal_sort_key)
            blocks = []
            for m in ordered_meals:
                m_label = self._meal_header_label(m)
                blocks.append(f"**{m_label}**\n" + "\n".join(by_meal[m]))
            items_text = "\n\n".join(blocks)

        # Determine correct date label
        IST_TZ = timezone(timedelta(hours=5, minutes=30))
        today_ist = datetime.now(IST_TZ).date()
        if log_date == today_ist:
            date_label_gu = "aaje nu"
            date_label_hi = "aaj ka"
            date_label_en = "today"
        elif log_date == today_ist - timedelta(days=1):
            date_label_gu = "gay kal nu"
            date_label_hi = "kal ka"
            date_label_en = "yesterday"
        else:
            date_label_gu = f"{log_date.isoformat()} nu"
            date_label_hi = f"{log_date.isoformat()} ka"
            date_label_en = f"on {log_date.isoformat()}"

        display_label = self._meal_header_label(meal_type) if meal_type else "Daily Log"

        summary_footer = (
            f"**Total**: **{total_cal:.0f} kcal**\n"
            f"**Protein**: **{total_protein:.0f}g** | **Carbs**: **{total_carbs:.0f}g** | **Fat**: **{total_fat:.0f}g**"
        )

        if self.lang == "gu":
            msg = f"Tamaru {date_label_gu} **{display_label}** ({log_date.isoformat()}):\n{items_text}\n\n{summary_footer}"
        elif self.lang == "hi":
            msg = f"Aapka {date_label_hi} **{display_label}** ({log_date.isoformat()}):\n{items_text}\n\n{summary_footer}"
        else:
            msg = f"Your **{display_label}** {date_label_en} ({log_date.isoformat()}):\n{items_text}\n\n{summary_footer}"

        return ChatResponse(
            message=msg, intent=Intent.QUERY_MEAL,
            data={"meal_type": meal_type, "logged_count": len(meal_logs),
                  "total_calories": total_cal, "date": log_date.isoformat()},
            success=True, language=self.lang,
        )

    async def _handle_query_exercise(self, parsed: LLMParseResult) -> ChatResponse:
        """Handle 'aaje me kya exercise kari?' — show today's exercise logs."""
        log_date = self._resolve_date(parsed.date)

        exercise_logs = await self.log_repo.get_exercise_logs_by_date(self.user["user_id"], log_date)

        # Date label
        IST_TZ = timezone(timedelta(hours=5, minutes=30))
        today_ist = datetime.now(IST_TZ).date()
        if log_date == today_ist:
            day_word = {"gu": "aaje", "hi": "aaj", "en": "today"}.get(self.lang, "today")
        elif log_date == today_ist - timedelta(days=1):
            day_word = {"gu": "gay kale", "hi": "kal", "en": "yesterday"}.get(self.lang, "yesterday")
        else:
            day_word = log_date.isoformat()

        if not exercise_logs:
            if self.lang == "gu":
                msg = f"Tame {day_word} koi exercise log nathi karyu."
            elif self.lang == "hi":
                msg = f"Aapne {day_word} koi exercise log nahi kiya hai."
            else:
                msg = f"You haven't logged any exercise {day_word}."
            return ChatResponse(message=msg, intent=Intent.QUERY_EXERCISE,
                                data={"date": log_date.isoformat(), "exercise_count": 0},
                                success=True, language=self.lang)

        # Build response
        total_burned = sum(l.get("calories_avg", 0) for l in exercise_logs)
        items = []
        for log in exercise_logs:
            time_str = ""
            if log.get("created_at"):
                try:
                    ct = log["created_at"]
                    if isinstance(ct, datetime):
                        if ct.tzinfo is None or ct.utcoffset() == timedelta(0):
                            ct = ct.replace(tzinfo=timezone.utc).astimezone(IST_TZ)
                        time_str = ct.strftime("%I:%M %p")
                    elif isinstance(ct, str):
                        time_str = ct.split("T")[1][:5] if "T" in ct else ""
                except Exception:
                    pass
            ex_name = log.get('exercise_name_display', log.get('exercise_name', '?'))
            ex_emoji = get_exercise_emoji(ex_name, log.get("category", ""))
            amt = log.get('amount', 0)
            unit = log.get('unit', '')
            cals = log.get('calories_avg', 0)
            items.append(
                f"• {ex_emoji} **{ex_name}** — **{amt:.0f} {unit}** — **{cals:.0f} kcal burned**"
                f"{' (' + time_str + ')' if time_str else ''}"
            )

        items_text = "\n".join(items)

        if self.lang == "gu":
            msg = f"Tamaru {day_word} nu exercise ({log_date.isoformat()}):\n{items_text}\n\n**Total burned**: **{total_burned:.0f} kcal**"
        elif self.lang == "hi":
            msg = f"Aapka {day_word} ka exercise ({log_date.isoformat()}):\n{items_text}\n\n**Total burned**: **{total_burned:.0f} kcal**"
        else:
            msg = f"Your exercise {day_word} ({log_date.isoformat()}):\n{items_text}\n\n**Total burned**: **{total_burned:.0f} kcal**"

        return ChatResponse(message=msg, intent=Intent.QUERY_EXERCISE,
                            data={"date": log_date.isoformat(), "exercise_count": len(exercise_logs),
                                  "total_burned": total_burned},
                            success=True, language=self.lang)

    async def _handle_skip_meal(self, parsed: LLMParseResult) -> ChatResponse:
        """Handle 'breakfast nathi karyu' — log a skipped meal entry."""
        meal_type = parsed.meal_type or "snack"
        log_date = self._resolve_date(parsed.date)

        # Save a "skipped" entry with 0 calories
        await self.log_repo.create_food_log(
            user_id=self.user["user_id"],
            log_date=log_date,
            meal_type=meal_type,
            food_id="skipped",
            food_name="meal_skipped",
            food_name_display=f"{meal_type.capitalize()} Skipped",
            quantity_input="0",
            quantity_amount=0,
            quantity_unit="none",
            quantity_grams=0,
            variant=None,
            calories=0,
            protein_g=0,
            carbs_g=0,
            fat_g=0,
            fiber_g=0,
            search_confidence=1.0,
        )

        # Clear flow state
        await self.flow_mgr.save_flow_state(PendingFlow())

        if self.lang == "gu":
            msg = f"Okay, {meal_type} skip thayu chhe aaje. Log save thai gayu."
        elif self.lang == "hi":
            msg = f"Okay, {meal_type} skip ho gaya aaj. Log save ho gaya."
        else:
            msg = f"Okay, {meal_type} marked as skipped for today."

        return ChatResponse(
            message=msg, intent=Intent.SKIP_MEAL,
            action_taken="food_logged",
            data={"meal_type": meal_type, "skipped": True},
            success=True, language=self.lang,
        )

    def _handle_get_calories(self, parsed: LLMParseResult) -> ChatResponse:
        food_query = parsed.food_query
        if not food_query:
            return ChatResponse(message="Which food do you want to know about?",
                                intent=Intent.CLARIFICATION_NEEDED, success=True, language=self.lang)

        search_result = self.search_engine.search(food_query)
        if search_result.match_type.value in ("no_match", "low_confidence"):
            return ChatResponse(message=get_response("food_not_found", self.lang, food=food_query),
                                intent=Intent.GET_CALORIES, success=False, language=self.lang)

        food = search_result.food
        quantity = parsed.quantity or "100g"
        parsed_qty = parse_quantity(quantity)
        if parsed_qty.amount is None or parsed_qty.amount <= 0 or not parsed_qty.unit:
            parsed_qty = ParsedQuantity(amount=1.0, unit="serving", raw_input=quantity, confidence=1.0)
        nutrition = NutritionCalculator().calculate(food, parsed_qty)

        f_emoji = get_food_emoji(food["food_name_display"], food.get("category", ""))
        msg = (f"{f_emoji} **{food['food_name_display']}** (**{quantity}**):\n"
               f"- **Calories**: **{nutrition.calories:.0f} kcal**\n"
               f"- **Protein**: **{nutrition.protein_g:.1f}g**\n"
               f"- **Carbs**: **{nutrition.carbs_g:.1f}g**\n"
               f"- **Fat**: **{nutrition.fat_g:.1f}g**\n"
               f"- **Fiber**: **{nutrition.fiber_g:.1f}g**")
        msg += get_response("safety_disclaimer", self.lang)

        return ChatResponse(message=msg, intent=Intent.GET_CALORIES, data={
            "food_name": food["food_name_display"], "quantity": quantity,
            "calories": nutrition.calories, "protein_g": nutrition.protein_g,
            "carbs_g": nutrition.carbs_g, "fat_g": nutrition.fat_g,
        }, success=True, language=self.lang)

    def _handle_get_profile(self) -> ChatResponse:
        u = self.user
        target = compute_full_target(u)
        msg = (f"Your profile:\n"
               f"- {u['name']} | {u['age']}yo | {u['gender']}\n"
               f"- {u['height_cm']}cm | {u['weight_kg']}kg\n"
               f"- Goal: {u['fitness_goal']} | Activity: {u['activity_level']}\n"
               f"- BMR: {target.bmr:.0f} | TDEE: {target.tdee:.0f} | Target: {target.calorie_target:.0f} kcal/day")
        return ChatResponse(message=msg, intent=Intent.GET_PROFILE, data={
            "calorie_target": target.calorie_target, "bmr": target.bmr, "tdee": target.tdee,
        }, success=True, language=self.lang)

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    async def execute_confirmed_action(self, pending_action: dict) -> ChatResponse:
        action_type = pending_action.get("type")
        if action_type == "log_food":
            # If _force_add flag set, skip duplicate check
            if pending_action.get("_force_add"):
                return await self._force_food_log(pending_action)
            return await self._execute_food_log(pending_action)
        elif action_type == "log_exercise":
            return await self._execute_exercise_log(pending_action)
        elif action_type == "multi_food":
            # Save each item using pre-computed nutrition data — no re-search
            items = pending_action.get("items", [])
            meal_type = pending_action.get("meal_type", "snack")
            user_id = pending_action.get("user_id", self.user["user_id"])
            log_date = self._resolve_date(pending_action.get("log_date", "today"))
            return await self._save_multi_items(items, meal_type, user_id, log_date)
        return ChatResponse(message="Unknown action.", intent=Intent.UNKNOWN, success=False)

    async def _save_multi_items(self, items: list, meal_type: str, user_id: str, log_date) -> ChatResponse:
        """
        Save a list of pre-resolved food items directly to the DB.
        Each item must already contain full nutrition data (food_id, calories, protein_g, etc.)
        — no re-search is performed here.
        """
        saved_count = 0
        total_cal = 0.0

        for item in items:
            food_id = item.get("food_id")
            food_name = item.get("food_name_raw") or item.get("food_name", "")
            food_name_display = item.get("food_name_display") or item.get("food_name", "")
            quantity_input = item.get("quantity", "1 serving")
            quantity_amount = item.get("quantity_amount", 1.0)
            quantity_unit = item.get("quantity_unit", "serving")
            quantity_grams = item.get("quantity_grams", 100.0)
            variant = item.get("variant")
            calories = item.get("calories", 0.0)
            protein_g = item.get("protein_g", 0.0)
            carbs_g = item.get("carbs_g", 0.0)
            fat_g = item.get("fat_g", 0.0)
            fiber_g = item.get("fiber_g", 0.0)
            search_confidence = item.get("search_confidence", 1.0)
            item_meal = item.get("meal_type", meal_type)
            item_source = item.get("source", "Local Database")  # internal only, never shown
            item_date_raw = item.get("log_date")
            item_date = self._resolve_date(item_date_raw) if item_date_raw else log_date

            if not food_id:
                continue  # skip items that couldn't be resolved

            await self.log_repo.create_food_log(
                user_id=user_id, log_date=item_date, meal_type=item_meal,
                food_id=food_id, food_name=food_name, food_name_display=food_name_display,
                quantity_input=quantity_input, quantity_amount=quantity_amount,
                quantity_unit=quantity_unit, quantity_grams=quantity_grams,
                variant=variant, calories=calories,
                protein_g=protein_g, carbs_g=carbs_g, fat_g=fat_g, fiber_g=fiber_g,
                search_confidence=search_confidence, source=item_source,
            )
            saved_count += 1
            total_cal += calories

        await self.flow_mgr.save_flow_state(PendingFlow())

        display_meal = self._meal_header_label(meal_type)
        if self.lang == "gu":
            msg = f"✓ Log saved successfully!\nDone! {saved_count} item{'' if saved_count == 1 else 's'} tamara **{display_meal}** ma save thai gaya. **Total**: **{total_cal:.0f} kcal**"
        elif self.lang == "hi":
            msg = f"✓ Log saved successfully!\nDone! {saved_count} item{'' if saved_count == 1 else 's'} aapke **{display_meal}** me save ho gaye. **Total**: **{total_cal:.0f} kcal**"
        else:
            msg = f"✓ Log saved successfully!\nDone! {saved_count} item{'' if saved_count == 1 else 's'} saved to your **{display_meal}**. **Total**: **{total_cal:.0f} kcal**"

        # Show the full day's log (all meals so far, grouped chronologically).
        day_log = await self._build_day_log_text(user_id, log_date, self.lang)
        if day_log:
            msg += "\n\n" + day_log

        recommendation = await self._build_nutrition_recommendation(
            user_id, log_date, meal_type, self.lang
        )
        if recommendation:
            msg += "\n\n---\n" + recommendation

        return ChatResponse(
            message=msg, intent=Intent.LOG_FOOD, action_taken="food_logged",
            data={"logged_count": saved_count, "total_calories": total_cal, "show_toast": True, "toast_message": "✓ Log saved successfully"},
            success=True, language=self.lang,
        )

    async def _execute_food_log(self, data: dict) -> ChatResponse:
        log_date = self._resolve_date(data.get("log_date"))
        meal_type = data["meal_type"]

        # Check if this meal already has entries for today
        existing_logs = await self.log_repo.get_food_logs_by_date(data["user_id"], log_date)
        meal_logs = [l for l in existing_logs if l.get("meal_type") == meal_type and l.get("food_id") != "skipped"]

        if meal_logs:
            # Meal already logged — ask if they want to add more or replace
            existing_items = ", ".join(l.get("food_name_display", "?") for l in meal_logs)
            total_cal = sum(l.get("calories", 0) for l in meal_logs)

            meal_header = self._meal_header_label(meal_type)

            if self.lang == "gu":
                msg = (f"Tame aaje **{meal_header}** ma already log kari chhe: {existing_items} (**{total_cal:.0f} kcal**).\n"
                       f"Shu tame aa item pan add karva mango cho?")
            elif self.lang == "hi":
                msg = (f"Aapne aaj **{meal_header}** me already log kiya hai: {existing_items} (**{total_cal:.0f} kcal**).\n"
                       f"Kya aap ye item bhi add karna chahte hain?")
            else:
                msg = (f"You already logged **{meal_header}**: {existing_items} (**{total_cal:.0f} kcal**).\n"
                       f"Do you want to add this item too?")

            return ChatResponse(
                message=msg, intent=Intent.LOG_FOOD,
                needs_confirmation=True, pending_action={**data, "_force_add": True},
                options=[get_option("yes_add_more", self.lang), get_option("no_bas", self.lang)],
                data={"existing_items": existing_items, "existing_calories": total_cal},
                success=True, language=self.lang,
            )

        # No existing meal — log directly
        return await self._force_food_log(data, log_date)

    async def _force_food_log(self, data: dict, log_date=None) -> ChatResponse:
        """Actually save the food log (no duplicate check)."""
        if log_date is None:
            log_date = self._resolve_date(data.get("log_date"))
        await self.log_repo.create_food_log(
            user_id=data["user_id"], log_date=log_date, meal_type=data["meal_type"],
            food_id=data["food_id"], food_name=data["food_name"],
            food_name_display=data["food_name_display"], quantity_input=data["quantity_input"],
            quantity_amount=data["quantity_amount"], quantity_unit=data["quantity_unit"],
            quantity_grams=data["quantity_grams"], variant=data.get("variant"),
            calories=data["calories"], protein_g=data["protein_g"], carbs_g=data["carbs_g"],
            fat_g=data["fat_g"], fiber_g=data["fiber_g"],
            search_confidence=data.get("search_confidence"),
            source=data.get("source", "Local Database"),
        )
        await self.flow_mgr.save_flow_state(PendingFlow())
        msg = get_response("food_logged", self.lang, food=data["food_name_display"],
                           calories=data["calories"], meal=data["meal_type"])
        day_log = await self._build_day_log_text(data["user_id"], log_date, self.lang)
        if day_log:
            msg += "\n\n" + day_log
        recommendation = await self._build_nutrition_recommendation(
            data["user_id"], log_date, data["meal_type"], self.lang
        )
        if recommendation:
            msg += "\n\n---\n" + recommendation
        return ChatResponse(
            message=msg,
            intent=Intent.LOG_FOOD, action_taken="food_logged",
            data={"calories": data["calories"]}, success=True, language=self.lang)

    async def _execute_exercise_log(self, data: dict) -> ChatResponse:
        log_date = self._resolve_date(data.get("log_date"))
        await self.log_repo.create_exercise_log(
            user_id=data["user_id"], log_date=log_date,
            exercise_id=data["exercise_id"], exercise_name=data["exercise_name"],
            exercise_name_display=data["exercise_name_display"], category=data["category"],
            exercise_input=data["exercise_input"], amount=data["amount"], unit=data["unit"],
            calories_min=data["calories_min"], calories_avg=data["calories_avg"],
            calories_max=data["calories_max"], search_confidence=data.get("search_confidence"),
        )
        # Clear flow state after successful log
        await self.flow_mgr.save_flow_state(PendingFlow())
        return ChatResponse(
            message=get_response("exercise_logged", self.lang,
                                 exercise=data["exercise_name_display"], calories=data["calories_avg"]),
            intent=Intent.LOG_EXERCISE, action_taken="exercise_logged",
            data={"calories_burned": data["calories_avg"]}, success=True, language=self.lang)

    # ------------------------------------------------------------------
    # Multi-food
    # ------------------------------------------------------------------

    def _is_intent_switch(self, message: str) -> bool:
        """
        Detect if user abandoned the current clarification flow to ask something
        completely different — so we drop the pending flow and handle the new intent.

        Returns True for:
          - Summary / profile / calorie queries
          - Greetings
          - Exercise log or exercise query messages (the main bug this fixes)
        """
        # ── Daily-Log Read intent switch ────────────────────────────────
        if self._classify_daily_log_read(message) is not None:
            return True

        m = message.lower().strip()

        # ── Non-exercise intent switches ────────────────────────────────
        switch_phrases = (
            "summary", "dikhao", "report", "kitna khaya",
            "calorie", "nutrition", "total log", "today's total", "todays total",
            "aaj nu total", "aaj ka total", "aaje shu khadhu", "su khadhu",
            "my profile", "mera profile", "profile",
            "namaste", "kem cho",
            "baki chhe", "remaining", "burn kari", "burned",
        )
        if any(p in m for p in switch_phrases):
            return True
        if m in {"hi", "hello", "hey"}:
            return True

        # ── Exercise intent switch ───────────────────────────────────────
        # Any message that looks like an exercise log or exercise query must
        # break out of a pending food flow, not be fed into it as a variant
        # or quantity answer.
        if self._looks_like_exercise(m) or self._looks_like_structured_exercise(message):
            return True

        try:
            from exercise_calculator import parse_workout_routine
            if parse_workout_routine(message):
                return True
        except Exception:
            pass

        return False

    def _looks_like_exercise(self, msg_lower: str) -> bool:
        """
        Returns True if the (lowercased) message appears to be an exercise
        intent — log or query.  Used both by _is_intent_switch() and by the
        flow guard in process_message().

        Design: checks a broad keyword set so that Gujarati/Hindi/Punjabi
        Romanised + English forms all match.  Does NOT hardcode exercise DB
        names — uses a representative trigger set.
        """
        # Guard: if the message is a food quantity/portion answer without exercise signals,
        # it is NEVER an exercise!
        _exercise_verbs_and_metrics = (
            "jogging", "jog", "joged", "running", "run", "ran", "walking", "walk", "walked",
            "cycling", "cycle", "cycled", "yoga", "swimming", "swim", "swam", "exercise",
            "workout", "gym", "pushup", "pushups", "push-up", "push-ups", "squat", "squats",
            "plank", "planks", "burpee", "burpees", "crunch", "crunches", "lunge", "lunges",
            "situp", "situps", "pullup", "pullups", "reps", "rep", "sets", "set", "rounds",
            "round", "km", "kms", "lap", "laps", "steps", "step", "did", "kari", "karya",
            "karyu", "kiya", "kiye", "vyayam", "kasrat", "kasarat", "daud"
        )
        has_workout_signal = any(re.search(r"\b" + re.escape(w) + r"\b", msg_lower) for w in _exercise_verbs_and_metrics)
        if not has_workout_signal and self._looks_like_quantity_answer(msg_lower):
            return False

        # Duration / rep patterns alongside any fitness verb = exercise
        # e.g. "30 minute jogging", "15 pushups", "pandrah pushups"
        _HAS_AMOUNT = re.compile(
            r"\b\d+\s*(?:min(?:ute)?s?|rep s?|reps?|rounds?|km|k\.?m\.?|"
            r"laps?|hours?|pushups?|squats?|steps?)\b"
            r"|(?:^|\s)\d{1,3}(?:\s+\w+)+",
            re.IGNORECASE,
        )

        _EXERCISE_TRIGGERS = {
            # English
            "jogging", "jog", "joged",
            "running", "run", "ran",
            "walking", "walk", "walked",
            "cycling", "cycle", "cycled",
            "yoga",
            "pushup", "pushups", "push-up", "push-ups", "push up",
            "squat", "squats",
            "swimming", "swim", "swam",
            "exercise", "exercises", "excercise", "excersise",
            "workout", "work out",
            "gym",
            "plank", "planks",
            "burpee", "burpees",
            "deadlift",
            "zumba",
            "boxing",
            "surya namaskar", "suryanamaskar",
            "jumping jack", "jumping jacks",
            "skipping",
            "stretching", "stretch",
            "lunges", "lunge",
            "crunches", "crunch",
            "sit-up", "situp", "sit ups",
            "pull up", "pull-up", "pullup",
            # Hindi / Gujarati Romanised exercise terms
            "kari",      # "jogging kari" / "exercise kari"
            "karya",     # "pushups karya"
            "karyu",     # "exercise karyu"
            "kiya",      # "yoga kiya"
            "kiye",      # "pushups kiye"
            "vyayam",    # Hindi: exercise/workout
            "kasrat",    "kasarat",          # Gujarati/Hindi: exercise
            "daud",      "dauding",          # Hindi: running
            # Punjabi Romanised
            "exercise kiti",  "workout kita",
            "daudna",    "turdna",           # Punjabi: running / walking
        }

        # 1. Exact phrase match
        for kw in _EXERCISE_TRIGGERS:
            if re.search(r"\b" + re.escape(kw) + r"\b", msg_lower):
                return True

        # 2. Digit + known exercise noun (covers "15 pushups", "30 jogging" etc.)
        digit_exercise = re.compile(
            r"\b\d+\s+(?:pushups?|push-ups?|squats?|crunches?|lunges?|"
            r"burpees?|planks?|situps?|pull-ups?|pullups?|reps?)\b",
            re.IGNORECASE,
        )
        if digit_exercise.search(msg_lower):
            return True

        # 3. Data-driven: consult the exercise dataset + Compendium (no hardcoded
        #    names) so external activities (kabaddi, cricket, elliptical, etc.)
        #    are also recognised as exercise intents.
        try:
            if self.llm._resolve_exercise_noun(msg_lower):
                return True
        except Exception:
            pass

        return False

    def _is_new_food_while_in_flow(self, message: str, flow) -> bool:
        """
        Returns True only when the user clearly mentioned a NEW, unrelated food
        mid-clarification — so we can park the current item and start fresh.

        Critically, this must NOT fire for:
          - Variant answers   ("Ghee", "Normal", "Oil")   while AWAITING_VARIANT
          - Quantity answers  ("2 pieces", "1 bowl", "150g") while AWAITING_QUANTITY
          - Meal-type answers ("breakfast", "savare")     while AWAITING_MEAL_TYPE
        """
        state = flow.state

        # Only intercept these three states; ignore confirmation / idle / etc.
        if state not in (FlowState.AWAITING_VARIANT,
                         FlowState.AWAITING_QUANTITY,
                         FlowState.AWAITING_MEAL_TYPE):
            return False

        msg_lower = message.lower().strip()

        # ── AWAITING_VARIANT: user is answering the variant question ──────
        # Never treat a variant-state reply as a new-food trigger.
        # "Ghee" → variant answer, not a new food.
        if state == FlowState.AWAITING_VARIANT:
            # If it contains a variant word it's definitely a variant answer
            if any(v in msg_lower for v in VARIANT_WORDS):
                return False
            # If it contains a yes/no word it's answering the question
            if any(w in msg_lower for w in YES_WORDS | NO_WORDS):
                return False
            # Short single-word reply in a variant state → treat as variant answer
            if len(msg_lower.split()) <= 2:
                return False

        # ── AWAITING_QUANTITY: user is answering the quantity question ─────
        import re as _re
        _QTY_RE = _re.compile(
            r"\b\d+(?:\.\d+)?\s*"
            r"(?:g|gm|grams?|ml|pieces?|pcs?|bowls?|katoris?|cups?|"
            r"plates?|glasses?|servings?|rotis?)\b",
            _re.IGNORECASE,
        )
        if state == FlowState.AWAITING_QUANTITY:
            # Starts with a digit → it's a quantity answer
            if _re.match(r"^\s*\d", message):
                return False
            # Contains a quantity pattern → quantity answer
            if _QTY_RE.search(message):
                return False
            # Short reply (≤ 3 words) in quantity state → treat as quantity
            if len(msg_lower.split()) <= 3:
                return False

        # ── AWAITING_MEAL_TYPE: user is picking a meal slot ───────────────
        if state == FlowState.AWAITING_MEAL_TYPE:
            from conversation import MEAL_WORDS
            all_meal_words = {w for words in MEAL_WORDS.values() for w in words}
            if any(w in msg_lower for w in all_meal_words):
                return False
            # Short reply → likely a meal answer, not a new food
            if len(msg_lower.split()) <= 2:
                return False

        # ── Strip current food name & connective words before searching ────
        cleaned_search_text = msg_lower
        current = (flow.food_name or flow.food_query or "").lower()
        if current:
            cleaned_search_text = _re.sub(rf"\b{_re.escape(current)}\b", "", cleaned_search_text, flags=_re.IGNORECASE).strip()
        connectives = ["sathe me", "sathe", "saath me", "saath", "ke saath", "with", "ane", "and", "plus"]
        for conn in connectives:
            cleaned_search_text = _re.sub(rf"\b{_re.escape(conn)}\b", "", cleaned_search_text, flags=_re.IGNORECASE).strip()

        food_target = self._clean_food_query("", original_message=cleaned_search_text)

        # ── Search the cleaned text as a food and check if it's a different food ─
        if food_target:
            search_result = self.search_engine.search(food_target)
            if search_result.match_type.value in ("exact_name", "exact_alias", "fuzzy_name"):
                if search_result.confidence >= 0.6:
                    found_name = (search_result.food or {}).get("food_name", "").lower()
                    if found_name and found_name != current and found_name not in current and current not in found_name:
                        return True

        return False

    def _detect_multi_food(self, message: str) -> list[str]:
        """
        Detect multiple food items in a message.
        Handles patterns like: "bataka ni sabji, roti, dal ane 1 glass chhaas lidha"
        """
        msg = message.strip()

        # Remove leading context: "aaje me lunch ma", "maine breakfast me", "I had ... for dinner"
        # Pattern 1: "... ma/me/mein <food items> lidha/khayi"
        food_part = msg
        for pattern in [
            r"^.*?\b(?:ma|me|mein)\s+",           # "aaje me lunch ma "
            r"^.*?\b(?:for)\s+(?:breakfast|lunch|dinner|snack)\s*$",  # trailing "for dinner"
            r"^(?:i\s+)?(?:had|ate|eaten)\s+",     # "I had ", "ate "
        ]:
            stripped = re.sub(pattern, "", food_part, count=1, flags=re.IGNORECASE).strip()
            if len(stripped) > 3 and stripped != food_part:
                food_part = stripped
                break

        # Remove trailing "for breakfast/lunch/dinner"
        food_part = re.sub(r"\s+for\s+(?:breakfast|lunch|dinner|snack)\s*$", "", food_part, flags=re.IGNORECASE)

        # Remove trailing past-tense verbs
        food_part = re.sub(
            r"\s*\b(lidha|lidhi|lidhu|khadha|khadhi|khadhu|khayi|khaya|khaye|"
            r"khai|jamya|jamyu|hatta|hattu|hati|ate|eaten|had|kiya|kiye)\b\s*$",
            "", food_part, flags=re.IGNORECASE
        ).strip()

        # Split on separators: "ane", "અને", "aur", "और", "and", "or", comma, "sathe", "with", "ne", "ને"
        # Longer connectives first so "sathe sathe" / "ke saath" match before "sathe"/"saath".
        separators = r'\s+(?:sathe sathe|saath saath|ke saath|sathe me|saath me|ane|aur|and|or|tatha|sathe|saath|with|pan|pn|bhi|plus|ne|અને|और|ને)\s+|[,&+]'
        parts = re.split(separators, food_part, flags=re.IGNORECASE)
        parts = [p.strip().strip(".-") for p in parts if p.strip() and len(p.strip()) > 1]

        return parts if len(parts) > 1 else (parts if parts else [])

    def _is_explicit_past_tense(self, message: str) -> bool:
        """Detect if user explicitly stated they already ate (past tense)."""
        msg = message.lower()
        past_words = ["lidha", "lidhi", "lidhu", "khadha", "khadhi", "khadhu",
                      "khayi", "khaya", "khaye", "khai", "jamya", "jamyu",
                      "hatta", "hattu", "hati", "ate", "eaten", "had",
                      "kiya", "kiye", "ki", "kari", "karyu", "kha", "khaa", "khi", "khali"]
        return any(w in msg for w in past_words)

    async def _build_food_item_queue_object(self, food_query: str, raw_item: str, shared_meal_type: Optional[str] = None) -> Optional[dict]:
        """Resolve a raw food item query into a structured queue item dict."""
        cleaned_query = self._clean_food_query(food_query, original_message=raw_item)
        if not cleaned_query:
            return None

        # ── Standalone generic umbrella terms must NEVER resolve to arbitrary/branded foods ──
        is_gen_raw, _ = is_generic_food_query(raw_item)
        is_gen_clean, _ = is_generic_food_query(cleaned_query)
        if is_gen_raw or is_gen_clean:
            return None

        # Resolve food via 4 tiers
        food = None
        source = "Local Database"
        is_ai_estimated = False

        res = self.search_engine.search(cleaned_query)
        mt = res.match_type.value

        # ── DETERMINISTIC food-plausibility veto (NO LLM) ──────────────────
        # Only used to stop a *coincidental* alias/fuzzy/external match on text
        # that is DEFINITIVELY not food (gibberish / greeting / conversational),
        # e.g. "hello" fuzzy-matching a branded "hello ginger kombucha". It never
        # fires the LLM. Returns True only for clearly-non-food input; typos of
        # real foods ("bhkhari") are NOT vetoed here because they land on a
        # high-confidence exact/alias/fuzzy match (see below), and a definitive
        # gibberish string simply won't produce such a match.
        try:
            _nonfood = self.llm._looks_like_non_food(cleaned_query) is True
        except Exception:
            _nonfood = False

        # A food is accepted ONLY when it matches our MongoDB (name / alias /
        # variant / confident fuzzy) or a valid external source. We NEVER invent
        # or predict a food. Confident local match types are trusted directly;
        # low-confidence local matches need a strong score; external needs
        # compatibility + the non-food veto.
        weak_local_candidate = None

        if mt in ("exact_name", "exact_alias", "variant_strip"):
            # Definitive DB match (curated food, its alias, or a variant-stripped
            # name). Typos resolve here via the transliteration map + exact match.
            # Apply the deterministic non-food veto here too: junk text can
            # coincidentally hit an obscure branded ALIAS at conf 1.0 (e.g.
            # "book cab" → "spongebob ... candy book", "meal" → "cracker, meal").
            # The veto is True only for clearly-non-food text, so real foods
            # (dal/roti/paneer → None) are never affected.
            if not _nonfood:
                food = res.food
                source = "Local Database"
        elif mt in ("fuzzy_name", "fuzzy_alias"):
            # Fuzzy match = a spelling variant of a REAL food in the DB. The
            # search engine only returns these above its confidence threshold, so
            # "bhkhari"→bhakri passes while gibberish does not reach here. Reject
            # only if the query is DEFINITIVELY non-food (deterministic veto).
            if not _nonfood:
                food = res.food
                source = "Local Database"
        elif mt == "low_confidence" and res.candidates:
            top = res.candidates[0]
            conf = top.get("confidence", 0)
            cand_food = self.food_repo.get_by_id(top["food_id"])
            # Accept a low-confidence candidate ONLY when it's a strong near-match
            # (typo of a real food) AND not definitively non-food. This rejects
            # "abc" (0.57) / "qwerty" (0.67) while keeping close typos.
            if cand_food and not _nonfood and conf >= 0.78:
                food = cand_food
                source = "Local Database"

        # ── Deterministic compound-dish fallback (NO LLM, NO external) ─────
        # A multi-word dish often fails to match as a whole ("tofu stir fry",
        # "drumstick sambar", "bhindi fry sabji", "moong dal khichdi") even
        # though its HEAD noun is a real curated food. Before going external /
        # rejecting, retry deterministically by (a) stripping cooking-method /
        # portion modifiers, then (b) trying the head noun. Accept ONLY an
        # exact/alias match so junk can never be rescued (a junk head token that
        # is non-food is already vetoed by _nonfood above).
        # NOTE: this runs even when the whole-phrase _nonfood heuristic is True,
        # because a vowel-less cooking word ("fry", "dry") can falsely trip the
        # gibberish check. Safety is preserved by (1) skipping any retry token
        # that is itself clearly non-food, and (2) accepting ONLY an exact/alias
        # match — junk head tokens ("bite", "meal", "portion") are non-food and
        # are skipped by the per-token veto below.
        if food is None:
            _method_words = {
                "fry", "fried", "stir", "stir-fry", "roast", "roasted", "grilled",
                "grill", "masala", "curry", "gravy", "dry", "tadka", "tikka",
                "bhurji", "sabji", "sabzi", "shaak", "shak", "ki", "ka", "ke", "nu", "ni",
                "na", "with", "and", "of", "fresh", "homemade", "spicy",
                "subji", "subzi", "wali", "wala", "vale",
                # preparation words (English + Gujarati mapped forms)
                "boiled", "steamed", "soaked", "baked",
                # NOTE: "pani" and "water" are intentionally NOT here — they are
                # part of meaningful food phrases ("mag nu pani" = moong water,
                # "chana nu pani" = chana water) and must reach the search engine
                # intact so _resolve_prep_compound can handle them correctly.
            }
            _tokens = [w for w in re.findall(r"[a-z]+", cleaned_query.lower())]
            _content = [w for w in _tokens if w not in _method_words]
            _retry_queries = []
            # (a) modifiers stripped, full remaining phrase
            if _content and len(_content) < len(_tokens):
                _retry_queries.append(" ".join(_content))
            # (b) head noun (last content token) and (c) first content token
            if _content:
                if _content[-1] not in _retry_queries:
                    _retry_queries.append(_content[-1])
                if _content[0] not in _retry_queries:
                    _retry_queries.append(_content[0])
            for _rq in _retry_queries:
                if not _rq or len(_rq) < 3:
                    continue
                # Skip a retry token that is itself clearly non-food (keeps junk
                # like "bite"/"meal"/"portion" from being rescued).
                try:
                    if self.llm._looks_like_non_food(_rq) is True:
                        continue
                except Exception:
                    pass
                _rr = self.search_engine.search(_rq)
                if _rr.match_type.value in ("exact_name", "exact_alias", "variant_strip"):
                    food = _rr.food
                    source = "Local Database"
                    break

        def is_compat(q_str: str, f_doc: dict) -> bool:
            if not f_doc:
                return False
            q_words = set(re.findall(r"\w+", q_str.lower()))
            f_words = set(re.findall(r"\w+", f_doc.get("food_name", "").lower()))
            distinct_terms = {
                "leg", "drumstick", "breast", "wings", "thigh", "lolipop", "lollipop", "popcorn", "nuggets",
                "burger", "pizza", "sandwich", "roll", "frankie", "momos", "dumplings",
                "biryani", "pulao", "khichdi", "curry", "gravy", "soup",
                "roti", "rotlo", "paratha", "bhakri", "naan", "dosa", "idli", "khakhra", "fafda",
                "thepla", "kheer", "halwa", "raita", "salad", "puri", "kulcha", "bhatura", "chilla", "pancake",
                "shake", "juice", "tea", "coffee", "lassi", "chhas", "cake", "cookie", "biscuit",
                "burfi", "barfi", "ladoo", "mithai"
            }
            q_distinct = q_words.intersection(distinct_terms)
            if q_distinct and not q_distinct.intersection(f_words):
                return False

            # Dish-family incompatibility: a flatbread is never a dessert/soup/raita
            bread_terms = {"thepla", "roti", "rotlo", "paratha", "bhakri", "naan", "kulcha", "puri", "chilla"}
            dessert_terms = {"kheer", "halwa", "burfi", "barfi", "ladoo", "sweet", "mithai", "cake", "pudding"}
            liquid_terms = {"soup", "shorba", "raita", "juice", "shake"}

            if q_words.intersection(bread_terms) and f_words.intersection(dessert_terms.union(liquid_terms)):
                return False
            if q_words.intersection(dessert_terms) and f_words.intersection(bread_terms):
                return False

            heavy_types = {"sandwich", "burger", "pizza", "momos", "roll", "frankie", "taco", "burrito"}
            for ht in heavy_types:
                if ht in f_words and ht not in q_words:
                    return False
            return True

        # Whether the query is substantial enough to look up in external DBs.
        # Very short single tokens ("abc", "shu", "xy") are ambiguous fragments
        # that spuriously match branded products (e.g. "abc" → "ABC Mini
        # Chocolates"), so we never send them to external sources. Multi-word or
        # ≥4-char queries are allowed. Purely deterministic — no LLM.
        _q_tokens = re.findall(r"[a-z]+", cleaned_query.lower())
        _latin_placeholders = {"lorem", "ipsum", "dolor", "sit", "amet", "consectetur"}
        _external_worthy = (
            not _nonfood
            and not any(tok in _latin_placeholders for tok in _q_tokens)
            and len(cleaned_query.strip()) >= 4
            and not (len(_q_tokens) == 1 and len(_q_tokens[0]) < 4)
        )

        # External food sources (Open Food Facts, USDA) — a VALID external food
        # match is acceptable. Gated deterministically (non-food veto + minimum
        # query substance) so conversational/gibberish/short fragments can't
        # match a branded product. No LLM call.
        if food is None and _external_worthy:
            try:
                off = await self.external_api.search_open_food_facts(cleaned_query)
                if off and is_compat(cleaned_query, off):
                    food = off
                    source = "Open Food Facts"
                    try:
                        self.food_repo.save_learned_food_sync(cleaned_query, food)
                    except Exception:
                        pass   # save failure must never drop the resolved food
            except Exception:
                pass   # API failure → fall through to USDA, never null out food

        if food is None and _external_worthy:
            try:
                usda = await self.external_api.search_usda(cleaned_query)
                if usda and is_compat(cleaned_query, usda):
                    food = usda
                    source = "USDA FoodData"
                    try:
                        self.food_repo.save_learned_food_sync(cleaned_query, food)
                    except Exception:
                        pass   # save failure must never drop the resolved food
            except Exception:
                pass   # API failure → fall through, never null out food

        # NO AI-ESTIMATE / INVENT TIER.
        # Food is accepted ONLY from MongoDB (name/alias/variant/confident fuzzy)
        # or a valid external source above. If nothing matched, the item is
        # rejected — we never ask the LLM to invent or predict a food.
        if not food:
            return None

        # Extract pre-supplied quantity / variant if present in raw_item
        parsed_qty = parse_quantity(raw_item)
        if parsed_qty.confidence >= 0.8 and parsed_qty.amount is not None and parsed_qty.amount > 0 and parsed_qty.unit:
            supplied_qty_str = f"{parsed_qty.amount:g} {parsed_qty.unit}"
            needs_qty = False
        else:
            supplied_qty_str = None
            needs_qty = True

        # Check variant support dynamically from food.preparation_variants schema
        prep_variants = food.get("preparation_variants") or food.get("variant_options") or []
        needs_var = False
        supplied_variant = None

        if len(prep_variants) > 1:
            raw_lower = raw_item.lower()
            for v_opt in prep_variants:
                v_clean = v_opt.lower().strip()
                if v_clean != "normal" and v_clean in raw_lower:
                    supplied_variant = v_clean
                    break
            if not supplied_variant:
                for v_word in ["ghee", "butter", "oil", "fried", "steamed", "grilled", "boiled", "tandoori", "black", "skimmed"]:
                    if v_word in raw_lower:
                        supplied_variant = v_word
                        break
            if not supplied_variant:
                needs_var = True
        else:
            needs_var = False

        # Preserve the fuller spoken/typed food name when the user's phrase is
        # richer than the canonical display name (e.g. "mandu vada" resolved via
        # an alias to the food "Vada"). This keeps the complete understood name
        # visible to the user instead of truncating it to the canonical word,
        # while still logging against the resolved food_id/nutrition.
        display_name = food["food_name_display"]
        canonical_name = (food.get("food_name") or "").lower()
        user_phrase = self._display_phrase_from_query(cleaned_query, canonical_name)
        if user_phrase:
            display_name = user_phrase
        elif any(w in (raw_item.lower() + " " + cleaned_query.lower()) for w in ["rotlo", "rotla"]):
            raw_low = raw_item.lower()
            if any(b in raw_low for b in ["bajra", "bajara", "bajri"]):
                if "rotla" in raw_low:
                    display_name = "Bajra na Rotla"
                else:
                    display_name = "Bajra no Rotlo"
            else:
                clean_words = [w for w in cleaned_query.split() if w.lower() not in ["i", "ate", "had", "eaten", "a", "an", "the", "one", "two", "1", "2"]]
                if clean_words and any(w.lower() in ["rotlo", "rotla"] for w in clean_words):
                    display_name = " ".join(w.capitalize() for w in clean_words)

        f_id = food.get("food_id") or food.get("id") or f"food_ext_{abs(hash(str(food.get('food_name', '')))) % 100000000:08x}"
        res_food_name = food.get("food_name", cleaned_query)
        if any(w in (raw_item.lower() + " " + cleaned_query.lower()) for w in ["rotlo", "rotla"]):
            res_food_name = display_name

        return {
            "food_id": f_id,
            "food_name": res_food_name,
            "food_name_display": display_name or res_food_name.title(),
            "category": food.get("category", ""),
            "serving_size_g": food.get("serving_size_g", 100),
            "quantity": supplied_qty_str,
            "variant": supplied_variant or "normal",
            "needs_quantity": needs_qty,
            "needs_variant": needs_var,
            "source": source,
            "is_ai_estimated": is_ai_estimated,
            "food_doc": food,
            # Per-item meal type (preserved so mixed-meal multi-food logs
            # correctly; falls back to the flow's shared meal type at save time).
            "meal_type": shared_meal_type,
        }

    def _display_phrase_from_query(self, cleaned_query: str, canonical_name: str) -> Optional[str]:
        """
        If the user's cleaned query is a richer name that still contains the
        canonical food word (e.g. query "mandu vada" for canonical "vada"),
        return a Title-Cased display phrase preserving the user's full name.
        Returns None when the query is not a strict superset of the canonical
        word (so we keep the DB's display name in the normal case).
        """
        if not cleaned_query or not canonical_name:
            return None
        q_tokens = cleaned_query.lower().split()
        c_tokens = canonical_name.lower().split()
        if len(q_tokens) <= len(c_tokens):
            return None
        # Every canonical word must be present in the query for it to be the
        # "same dish, richer name" case.
        if not all(ct in q_tokens for ct in c_tokens):
            return None
        return " ".join(w.capitalize() for w in q_tokens)

    async def _process_sequential_queue(self, flow: PendingFlow) -> ChatResponse:
        """
        Process items sequentially from queue.
        Order of questions for each item:
          1. Needs Variant? (ALWAYS ASK FIRST if food has >1 preparation_variants)
          2. Needs Quantity? (ASK SECOND if missing)
          3. Needs Meal Slot? (ASK THIRD)
          4. Confirm & Save.
        """
        items = flow.items or []
        if not items:
            if getattr(flow, "unresolved_items", None):
                generic_clarifications = []
                for item_text in flow.unresolved_items:
                    is_gen, gen_cat = is_generic_food_query(item_text)
                    if is_gen:
                        cmsg = get_generic_clarification_message(gen_cat, self.lang)
                        if cmsg not in generic_clarifications:
                            generic_clarifications.append(cmsg)
                    else:
                        cmsg = get_response("food_not_found", self.lang, food=item_text)
                        if cmsg not in generic_clarifications:
                            generic_clarifications.append(cmsg)
                if generic_clarifications:
                    return ChatResponse(
                        message="\n\n".join(generic_clarifications),
                        intent=Intent.CLARIFICATION_NEEDED, success=True, language=self.lang,
                    )
            return ChatResponse(message="What did you eat today?", intent=Intent.UNKNOWN, success=True, language=self.lang)

        def _attach_unresolved_clarifications(prompt_msg: str) -> str:
            if not getattr(flow, "unresolved_items", None):
                return prompt_msg
            clarifications = []
            for u in flow.unresolved_items:
                is_gen, gen_cat = is_generic_food_query(u)
                if is_gen:
                    gen_msg = get_generic_clarification_message(gen_cat, self.lang)
                    if gen_msg and gen_msg not in clarifications and gen_msg not in prompt_msg:
                        clarifications.append(gen_msg)
                else:
                    not_found_msg = get_response("food_not_found", self.lang, food=u)
                    if not_found_msg and not_found_msg not in clarifications and not_found_msg not in prompt_msg:
                        clarifications.append(not_found_msg)
            if clarifications:
                return prompt_msg + "\n\n" + "\n\n".join(clarifications)
            return prompt_msg

        # Loop through queue items starting from current_index
        for idx in range(flow.current_index, len(items)):
            item = items[idx]

            # 1. Needs Variant? (ALWAYS ASK FIRST)
            if item.get("needs_variant") and (not item.get("variant") or item.get("variant") == "normal"):
                flow.current_index = idx
                flow.state = FlowState.AWAITING_VARIANT
                flow.food_name = item["food_name"]
                flow.food_name_display = item["food_name_display"]
                await self.flow_mgr.save_flow_state(flow)

                msg = get_response("ask_variant", self.lang, food=item["food_name_display"])
                options = self._get_variant_options(item["food_name"], item.get("category", ""), food_doc=item.get("food_doc"))
                return ChatResponse(message=msg, intent=f"flow:{flow.state}", options=options, success=True, language=self.lang)

            # 2. Needs Quantity? (ASK SECOND)
            if item.get("needs_quantity") and not item.get("quantity"):
                flow.current_index = idx
                flow.state = FlowState.AWAITING_QUANTITY
                flow.food_name = item["food_name"]
                flow.food_name_display = item["food_name_display"]
                await self.flow_mgr.save_flow_state(flow)

                is_drink = self._is_beverage(item["food_name"], item.get("category", ""))
                ask_key = "ask_quantity_drink" if is_drink else "ask_quantity"
                msg = get_response(ask_key, self.lang, food=item["food_name_display"])
                options = self._get_quantity_options(item["food_name"], item.get("category", ""), item.get("serving_size_g", 100), food_doc=item.get("food_doc"))
                return ChatResponse(message=msg, intent=f"flow:{flow.state}", options=options, success=True, language=self.lang)

        # 3. Needs Shared Meal Slot?
        if not flow.shared_meal_type:
            flow.state = FlowState.AWAITING_MEAL_TYPE
            await self.flow_mgr.save_flow_state(flow)

            items_context = ", ".join(it["food_name_display"] for it in items)
            msg = get_response("ask_meal_type", self.lang, meal_context=items_context)
            options = [
                get_option("meal_breakfast", self.lang),
                get_option("meal_lunch", self.lang),
                get_option("meal_dinner", self.lang),
                get_option("meal_snack", self.lang),
            ]
            return ChatResponse(message=msg, intent=f"flow:{flow.state}", options=options, success=True, language=self.lang)

        # 4. MANDATORY VALIDATION PASS before any calorie calculation.
        #    If any item is still missing required info, re-enter the queue to ask.
        missing = self.validate_food_items_before_calculation(items)
        if missing:
            # Point current_index at the first incomplete item and re-ask.
            flow.current_index = missing[0]["index"]
            await self.flow_mgr.save_flow_state(flow)
            # Recurse: the loop above will now stop at the first missing field.
            return await self._process_sequential_queue(flow)

        # Check if any unresolved items remain (e.g. generic "shak" from multi-food input).
        # Handle them one-by-one in the SAME meal flow BEFORE final confirmation.
        if getattr(flow, "unresolved_items", None) and len(flow.unresolved_items) > 0:
            unresolved_term = flow.unresolved_items[0]
            is_gen, gen_cat = is_generic_food_query(unresolved_term)
            if is_gen:
                clarify_msg = get_generic_clarification_message(gen_cat, self.lang)
            else:
                clarify_msg = get_response("food_not_found", self.lang, food=unresolved_term)
            flow.state = FlowState.AWAITING_GENERIC_RESOLUTION
            await self.flow_mgr.save_flow_state(flow)
            return ChatResponse(
                message=clarify_msg,
                intent=f"flow:{flow.state}",
                options=[],
                success=True,
                language=self.lang,
            )

        # 5. ALL ITEMS & MEAL SLOT FULLY RESOLVED — calculate combined totals & confirm.
        pending_action_items = []
        summary_lines = []
        total_cals = 0.0
        total_protein = 0.0
        total_carbs = 0.0
        total_fat = 0.0

        meal_type = flow.shared_meal_type or "snack"

        for it in items:
            food_doc = it["food_doc"]
            qty_str = it["quantity"] or "1 serving"
            parsed_qty = parse_quantity(qty_str)
            if parsed_qty.amount is None or parsed_qty.amount <= 0 or not parsed_qty.unit:
                parsed_qty = ParsedQuantity(amount=1.0, unit="serving", raw_input=qty_str, confidence=1.0)
            nutrition = NutritionCalculator().calculate(food_doc, parsed_qty, variant=it.get("variant", "normal"))

            total_cals += nutrition.calories
            total_protein += nutrition.protein_g
            total_carbs += nutrition.carbs_g
            total_fat += nutrition.fat_g

            v_text = f" ({nutrition.variant})" if nutrition.variant and nutrition.variant != "normal" else ""
            # NOTE: never expose source/database tags in user-facing text.
            f_emoji = get_food_emoji(it["food_name_display"], it.get("category", ""))
            summary_lines.append(
                f"• {f_emoji} **{it['food_name_display']}{v_text}** — **{qty_str}** — **{nutrition.calories:.0f} kcal**"
            )

            pending_action = self._build_food_pending(food_doc, nutrition, qty_str, meal_type,
                                                      date.today().isoformat(), 1.0)
            pending_action_items.append(pending_action)

        flow.state = FlowState.AWAITING_MULTI_CHOICE
        flow.pending_items = pending_action_items
        await self.flow_mgr.save_flow_state(flow)

        summary_text = "\n".join(summary_lines)
        meal_display = self._meal_display_name(meal_type, self.lang)
        meal_emoji = get_meal_emoji(meal_type)
        macros = f"**Protein**: **{total_protein:.0f}g** | **Carbs**: **{total_carbs:.0f}g** | **Fat**: **{total_fat:.0f}g**"

        if self.lang == "gu":
            msg = (f"**{meal_emoji} {meal_display}**\n\n{summary_text}\n\n"
                   f"**Total**: **{total_cals:.0f} calories** ({macros})\n\n"
                   f"Shu aa tamara **{meal_emoji} {meal_display}** daily log ma save karu?")
        elif self.lang == "hi":
            msg = (f"**{meal_emoji} {meal_display}**\n\n{summary_text}\n\n"
                   f"**Total**: **{total_cals:.0f} calories** ({macros})\n\n"
                   f"Kya ye aapke **{meal_emoji} {meal_display}** log me save karein?")
        else:
            msg = (f"**{meal_emoji} {meal_display}**\n\n{summary_text}\n\n"
                   f"**Total**: **{total_cals:.0f} calories** ({macros})\n\n"
                   f"Should I save this to your **{meal_emoji} {meal_display}** log?")
        opts = [get_option("yes_save", self.lang), get_option("no_cancel", self.lang)]

        msg += get_response("safety_disclaimer", self.lang)
        return ChatResponse(message=msg, intent=f"flow:{flow.state}", options=opts, success=True, language=self.lang)

    def _meal_display_name(self, meal_type: str, lang: str) -> str:
        """Return a nicely-capitalised meal header in the user's language."""
        names = {
            "breakfast": {"gu": "Breakfast", "hi": "Breakfast", "en": "Breakfast"},
            "morning":   {"gu": "Morning",   "hi": "Morning",   "en": "Morning"},
            "lunch":     {"gu": "Lunch",     "hi": "Lunch",     "en": "Lunch"},
            "afternoon": {"gu": "Afternoon", "hi": "Afternoon", "en": "Afternoon"},
            "snack":     {"gu": "Snack",     "hi": "Snack",     "en": "Snack"},
            "evening":   {"gu": "Evening",   "hi": "Evening",   "en": "Evening"},
            "dinner":    {"gu": "Dinner",    "hi": "Dinner",    "en": "Dinner"},
        }
        row = names.get(meal_type, {})
        return row.get(lang, (meal_type or "Meal").capitalize())

    def _format_food_log_item(self, log: dict) -> str:
        name = log.get("food_name_display") or log.get("food_name") or "?"
        cat = log.get("category", "")
        emoji = get_food_emoji(name, cat)
        qty = (log.get("quantity_input") or "").strip()
        cal = log.get("calories", 0) or 0
        time_str = ""
        if log.get("created_at"):
            try:
                IST_TZ = timezone(timedelta(hours=5, minutes=30))
                ct = log["created_at"]
                if isinstance(ct, datetime):
                    if ct.tzinfo is None or ct.utcoffset() == timedelta(0):
                        ct = ct.replace(tzinfo=timezone.utc).astimezone(IST_TZ)
                    time_str = ct.strftime("%I:%M %p")
                elif isinstance(ct, str):
                    time_str = ct.split("T")[1][:5] if "T" in ct else ""
            except Exception:
                pass
        time_suffix = f" ({time_str})" if time_str else ""
        if qty:
            return f"• {emoji} **{name}** — **{qty}** — **{cal:.0f} kcal**{time_suffix}"
        else:
            return f"• {emoji} **{name}** — **{cal:.0f} kcal**{time_suffix}"

    def _meal_header_label(self, meal_type: str) -> str:
        """Display label for a meal section in the day log (covers all six
        chronological meals, not just breakfast/lunch/dinner/snack)."""
        clean = (meal_type or "").strip().lower()
        labels = {
            "breakfast": "Breakfast", "morning": "Morning", "lunch": "Lunch",
            "afternoon": "Afternoon", "snack": "Snack", "evening": "Evening",
            "dinner": "Dinner",
        }
        title = labels.get(clean, (meal_type or "Meal").capitalize())
        emoji = get_meal_emoji(clean)
        return f"{emoji} {title}" if emoji else title

    async def _build_day_log_text(self, user_id: str, log_date, lang: str) -> str:
        """
        Build the full day's food log grouped by meal, in chronological order
        (Breakfast → Morning → Lunch → Afternoon → Snack → Evening → Dinner).

        Shows ONLY meals that actually have logged items, each item formatted
        with emoji, bold food name, quantity and calories. Skipped-meal placeholder
        rows are excluded. Returns an empty string when nothing is logged for the day.
        Read-only — never modifies any log or state.
        """
        try:
            logs = await self.log_repo.get_food_logs_by_date(user_id, log_date)
        except Exception:
            return ""

        # Group items by meal, preserving per-meal chronological order
        # (get_food_logs_by_date is sorted by created_at).
        by_meal: dict[str, list[str]] = {}
        total_cal = 0.0
        total_protein = 0.0
        total_carbs = 0.0
        total_fat = 0.0

        for lg in logs:
            if lg.get("food_id") == "skipped":
                continue  # skip "meal skipped" placeholder entries
            meal = (lg.get("meal_type") or "snack").strip().lower()
            name = lg.get("food_name_display") or lg.get("food_name") or ""
            if not name:
                continue

            total_cal += (lg.get("calories", 0) or 0)
            total_protein += (lg.get("protein_g", 0) or 0)
            total_carbs += (lg.get("carbs_g", 0) or 0)
            total_fat += (lg.get("fat_g", 0) or 0)

            bullet = self._format_food_log_item(lg)
            by_meal.setdefault(meal, []).append(bullet)

        if not by_meal:
            return ""

        # Chronological meal order (unknown meals sort last, never dropped).
        ordered_meals = sorted(by_meal.keys(), key=meal_sort_key)

        title = {"gu": "Aaje sudhi no log:", "hi": "Aaj tak ka log:", "en": "Today's log so far:"}.get(lang, "Today's log so far:")
        lines = [f"**{title}**"]
        for meal in ordered_meals:
            lines.append("")
            lines.append(f"**{self._meal_header_label(meal)}**")
            lines.extend(by_meal[meal])

        lines.append("")
        lines.append(f"**Total**: **{total_cal:.0f} kcal**")
        lines.append(f"**Protein**: **{total_protein:.0f}g** | **Carbs**: **{total_carbs:.0f}g** | **Fat**: **{total_fat:.0f}g**")
        return "\n".join(lines)

    async def _build_nutrition_recommendation(
        self,
        user_id: str,
        log_date,
        meal_type: str,
        lang: str,
    ) -> str:
        """
        Generate a personalized, data-driven nutrition recommendation after
        a food is logged.

        Structure:
          • "Today so far" — real totals from DB
          • "What to focus on" — macro gap analysis
          • "Best next meal options" — practical suggestions based on remaining budget

        Rules (strictly enforced):
        - Reads only real logged data + user profile. Never invents values.
        - Post-workout advice shown only when exercise_entries > 0 for today.
        - Dinner path gives a closing note; non-dinner gives forward suggestions.
        - Returns "" silently on any error — this is a bonus, not a critical path.
        """
        try:
            from bmr_tdee_calculator import compute_full_target

            target          = compute_full_target(self.user)
            food_totals     = await self.log_repo.get_daily_food_totals(user_id, log_date)
            exercise_totals = await self.log_repo.get_daily_exercise_totals(user_id, log_date)

            consumed_cal  = float(food_totals.get("total_calories_consumed") or 0)
            consumed_prot = float(food_totals.get("total_protein_g") or 0)
            consumed_carb = float(food_totals.get("total_carbs_g") or 0)
            consumed_fat  = float(food_totals.get("total_fat_g") or 0)

            burned_cal   = float(exercise_totals.get("total_calories_burned") or 0)
            did_exercise = (exercise_totals.get("exercise_entries") or 0) > 0

            cal_target  = float(target.calorie_target  or 2000)
            prot_target = float(target.protein_target_g or 100)
            carb_target = float(target.carbs_target_g   or 250)
            fat_target  = float(target.fat_target_g     or 65)

            net_cal        = consumed_cal - burned_cal
            remaining_cal  = cal_target  - net_cal
            remaining_prot = prot_target - consumed_prot
            remaining_carb = carb_target - consumed_carb
            remaining_fat  = fat_target  - consumed_fat

            if consumed_cal <= 0:
                return ""

            # ── Macro status ───────────────────────────────────────────────
            def pct(v, t):
                return (v / t * 100) if t > 0 else 100.0

            def status(v, t):
                p = pct(v, t)
                if p > 105: return "over"
                if p > 85:  return "near"
                if p > 50:  return "on_track"
                return "low"

            cal_st   = status(net_cal,        cal_target)
            prot_st  = status(consumed_prot,  prot_target)
            carb_st  = status(consumed_carb,  carb_target)
            fat_st   = status(consumed_fat,   fat_target)

            meal_lower   = (meal_type or "snack").lower()
            is_last_meal = meal_lower == "dinner"

            # ── Language helper ────────────────────────────────────────────
            def L(en, hi, gu):
                if lang == "gu": return gu
                if lang == "hi": return hi
                return en

            # ── SECTION 1 — Today so far ───────────────────────────────────
            h1 = L("📊 **Today so far**", "📊 **आज तक**", "📊 **આજ સુધી**")

            ex_line = ""
            if did_exercise:
                ex_line = "\n" + L(
                    f"• 🏋️ Exercise burned: **{burned_cal:.0f} kcal**",
                    f"• 🏋️ Exercise se burn: **{burned_cal:.0f} kcal**",
                    f"• 🏋️ Exercise thi burn: **{burned_cal:.0f} kcal**",
                )

            remaining_disp = max(0.0, remaining_cal)
            sec1 = L(
                (f"{h1}\n"
                 f"• 🍽️ Consumed: **{consumed_cal:.0f} kcal** "
                 f"| Protein: **{consumed_prot:.0f}g** | Carbs: **{consumed_carb:.0f}g** | Fat: **{consumed_fat:.0f}g**"
                 f"{ex_line}\n"
                 f"• 🎯 Remaining: **{remaining_disp:.0f} kcal** (Daily target: {cal_target:.0f} kcal)"),
                (f"{h1}\n"
                 f"• 🍽️ Kha liya: **{consumed_cal:.0f} kcal** "
                 f"| Protein: **{consumed_prot:.0f}g** | Carbs: **{consumed_carb:.0f}g** | Fat: **{consumed_fat:.0f}g**"
                 f"{ex_line}\n"
                 f"• 🎯 Baki: **{remaining_disp:.0f} kcal** (Daily target: {cal_target:.0f} kcal)"),
                (f"{h1}\n"
                 f"• 🍽️ Khadhu: **{consumed_cal:.0f} kcal** "
                 f"| Protein: **{consumed_prot:.0f}g** | Carbs: **{consumed_carb:.0f}g** | Fat: **{consumed_fat:.0f}g**"
                 f"{ex_line}\n"
                 f"• 🎯 Baki: **{remaining_disp:.0f} kcal** (Daily target: {cal_target:.0f} kcal)"),
            )

            # ── SECTION 2 — What to focus on ──────────────────────────────
            h2 = L("💡 **What to focus on**", "💡 **किस पर ध्यान दें**", "💡 **શેના પર ધ્યાન આપો**")
            pts: list[str] = []

            # Calorie budget commentary
            if cal_st == "over":
                over = net_cal - cal_target
                pts.append(L(
                    f"⚠️ **{over:.0f} kcal over target** today — keep the next meal very light.",
                    f"⚠️ Target se **{over:.0f} kcal zyada** ho gaye — agli meal bahut halki rakhein.",
                    f"⚠️ Target thi **{over:.0f} kcal vadhare** — aagal meal khub halku rakho.",
                ))
            elif cal_st == "near":
                pts.append(L(
                    f"✅ Calories almost at target (**{remaining_disp:.0f} kcal** left) — light snack only if needed.",
                    f"✅ Calories target ke kareeb hain (**{remaining_disp:.0f} kcal** baki) — zaroorat ho toh halka snack.",
                    f"✅ Calories target ni najik chhe (**{remaining_disp:.0f} kcal** baki) — jarur hoy to halku snack lo.",
                ))
            else:
                pts.append(L(
                    f"📈 **{remaining_disp:.0f} kcal** still available — room for a proper meal.",
                    f"📈 **{remaining_disp:.0f} kcal** abhi baaki hain — ek sahi meal ke liye jagah hai.",
                    f"📈 **{remaining_disp:.0f} kcal** haju baki chhe — ek saras meal mate jagah chhe.",
                ))

            # Protein gap
            if prot_st == "low" and remaining_prot > 10:
                if did_exercise:
                    pts.append(L(
                        f"💪 Protein **{remaining_prot:.0f}g short** — critical for post-workout muscle recovery.",
                        f"💪 Protein **{remaining_prot:.0f}g kam** — workout ke baad muscle recovery ke liye zaroori.",
                        f"💪 Protein **{remaining_prot:.0f}g occhu** — workout pachhi muscle recovery mate zaruri.",
                    ))
                else:
                    pts.append(L(
                        f"🥚 Protein needs a boost (**{remaining_prot:.0f}g** more to reach daily target).",
                        f"🥚 Protein aur chahiye (**{remaining_prot:.0f}g** target tak pahunchne ke liye).",
                        f"🥚 Protein vadharvanu jaruri chhe (**{remaining_prot:.0f}g** target sudhi pahochhva).",
                    ))
            elif prot_st in ("on_track", "near"):
                pts.append(L(
                    f"✅ Protein on track — **{consumed_prot:.0f}/{prot_target:.0f}g** consumed.",
                    f"✅ Protein sahi chal raha hai — **{consumed_prot:.0f}/{prot_target:.0f}g** kha liya.",
                    f"✅ Protein sari rite chhe — **{consumed_prot:.0f}/{prot_target:.0f}g** khadhu.",
                ))
            elif prot_st == "over":
                pts.append(L(
                    f"⚠️ Protein is above target (**{consumed_prot:.0f}g**). No need to prioritise it further.",
                    f"⚠️ Protein target se zyada hai (**{consumed_prot:.0f}g**). Aur zaroorat nahi.",
                    f"⚠️ Protein target thi vadhare chhe (**{consumed_prot:.0f}g**). Vadhare zarur nathi.",
                ))

            # Carbs / fat flags
            if carb_st == "over":
                pts.append(L(
                    f"🍞 Carbs are high (**{consumed_carb:.0f}g**). Choose protein/fat sources for the next meal.",
                    f"🍞 Carbs zyada hain (**{consumed_carb:.0f}g**). Agli meal me protein/fat-rich options lo.",
                    f"🍞 Carbs vadhare chhe (**{consumed_carb:.0f}g**). Aagal meal ma protein/fat choose karo.",
                ))
            if fat_st == "over":
                pts.append(L(
                    f"🧈 Fat is high (**{consumed_fat:.0f}g**). Avoid fried/oily options next.",
                    f"🧈 Fat zyada hai (**{consumed_fat:.0f}g**). Agli meal me tali/oily cheezein avoid karein.",
                    f"🧈 Fat vadhare chhe (**{consumed_fat:.0f}g**). Aagal meal ma tel/ghee avoid karo.",
                ))

            if not pts:
                pts.append(L(
                    "✅ Macros are balanced — great job so far!",
                    "✅ Macros balanced hain — bahut achha!",
                    "✅ Macros balanced chhe — saras kaam!",
                ))

            sec2 = h2 + "\n" + "\n".join(pts)

            # ── SECTION 3 — Next meal suggestions ─────────────────────────
            sec3 = ""

            # ── SECTION 3 — Next meal suggestions (data-driven from MongoDB) ─
            sec3 = ""

            if is_last_meal:
                # Dinner already logged — give a closing note; no forward suggestions.
                if prot_st == "low" and remaining_prot > 20 and remaining_disp > 50:
                    sec3 = L(
                        (f"🌙 **Dinner tip:** You still need **{remaining_prot:.0f}g protein** — "
                         f"include dal, paneer, eggs, or curd to hit your target."),
                        (f"🌙 **Dinner tip:** Abhi **{remaining_prot:.0f}g protein** chahiye — "
                         f"dal, paneer, anda, ya dahi include karein."),
                        (f"🌙 **Dinner tip:** Haju **{remaining_prot:.0f}g protein** joiye — "
                         f"dal, paneer, anda, ya dahi samavesh karo."),
                    )
                elif cal_st == "over":
                    sec3 = L(
                        "🌙 Since you're over your calorie budget, skip or minimise dinner — "
                        "a light soup or salad is fine.",
                        "🌙 Calorie budget zyada ho gaya hai, dinner skip karein ya bahut halka rakhein — "
                        "soup ya salad theek hai.",
                        "🌙 Calorie budget vadhare thai gayo chhe, dinner skip karo ya khub halku rakho — "
                        "soup ya salad thik chhe.",
                    )
            else:
                # Non-dinner meals — query the DB for real suggestions.
                h3 = L(
                    "🍱 **Best next meal options**",
                    "🍱 **अगले खाने के लिए सुझाव**",
                    "🍱 **આગળ ના ભોજન માટે સૂચન**",
                )

                if cal_st == "over" and prot_st == "low" and remaining_prot > 10:
                    # Calories over but protein lacking — show lean protein note
                    # and query DB for lean options within a tight budget.
                    effective_budget = min(200.0, remaining_cal + 150)  # small headroom
                    db_lines = self._suggest_meals_from_db(
                        remaining_cal=effective_budget,
                        remaining_prot=remaining_prot,
                        prot_status=prot_st,
                        carb_status=carb_st,
                        fat_status=fat_st,
                        did_exercise=did_exercise,
                        is_last_meal=False,
                        lang=lang,
                        max_suggestions=3,
                    )
                    prot_note = L(
                        f"Calories over budget but protein still low (**{remaining_prot:.0f}g** needed) — "
                        f"lean, low-calorie protein sources only:",
                        f"Calories zyada hain par protein kam hai (**{remaining_prot:.0f}g** chahiye) — "
                        f"sirf lean, low-calorie protein lo:",
                        f"Calories vadhare chhe pan protein occhu chhe (**{remaining_prot:.0f}g** joiye) — "
                        f"lean, low-calorie protein j lo:",
                    )
                    if db_lines:
                        sec3 = h3 + "\n" + prot_note + "\n" + "\n".join(db_lines)
                    else:
                        sec3 = h3 + "\n• " + L(
                            "No suitable foods found in the database for the current constraint. "
                            "Try egg whites, plain curd, or grilled fish — lean proteins that add "
                            "protein with minimal calories.",
                            "Database mein suitable foods nahi mile. "
                            "Egg whites, sada dahi, ya grilled fish try karein.",
                            "Database ma suitable foods na malya. "
                            "Egg whites, sado dahi, ya grilled fish try karo.",
                        )

                elif remaining_disp > 100:
                    # Normal forward-suggestion path — query DB with real budget.
                    db_lines = self._suggest_meals_from_db(
                        remaining_cal=remaining_disp,
                        remaining_prot=remaining_prot,
                        prot_status=prot_st,
                        carb_status=carb_st,
                        fat_status=fat_st,
                        did_exercise=did_exercise,
                        is_last_meal=False,
                        lang=lang,
                        max_suggestions=4,
                    )
                    if db_lines:
                        budget_note = L(
                            f"Fitting your **{remaining_disp:.0f} kcal** budget",
                            f"Aapke **{remaining_disp:.0f} kcal** budget ke liye",
                            f"Tamara **{remaining_disp:.0f} kcal** budget mate",
                        )
                        sec3 = h3 + "\n" + f"*{budget_note}:*\n" + "\n".join(db_lines)
                    else:
                        # DB returned nothing usable — honest fallback message
                        sec3 = h3 + "\n• " + L(
                            "Not enough matching food data in the database for a suggestion right now.",
                            "Abhi database mein sufficient matching food data nahi hai.",
                            "Abhi database ma sufficient matching food data nathi.",
                        )

            # ── Assemble ────────────────────────────────────────────────────
            parts = [sec1, sec2]
            if sec3:
                parts.append(sec3)
            return "\n\n".join(parts)

        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Meal suggestion helpers (data-driven from MongoDB)
    # ------------------------------------------------------------------

    def _suggest_meals_from_db(
        self,
        remaining_cal: float,
        remaining_prot: float,
        prot_status: str,
        carb_status: str,
        fat_status: str,
        did_exercise: bool,
        is_last_meal: bool,
        lang: str,
        max_suggestions: int = 4,
    ) -> list[str]:
        """
        Query MongoDB for real foods, compute actual nutrition via
        NutritionCalculator, and return 2-4 ranked meal suggestion strings.

        Rules:
        - Respects user diet_type: veg/non_veg/eggetarian/vegan.
          Uses vegetarian_status filter AND a category-keyword guard to catch
          foods that are mislabelled in the DB (e.g. fish stored as "Veg").
        - Returns only is_verified=True foods with valid calories_kcal.
        - Computes real kcal/protein/carbs/fat via NutritionCalculator (1 serving).
        - Diversifies output: one item per category in the pool, different
          categories per paired combo, no repeated base food names.
        - Returns [] when the DB has insufficient matching data.
        """
        from nutrition_calculator import NutritionCalculator
        from quantity_parser import ParsedQuantity as PQ

        calc = NutritionCalculator()

        # ── Diet-type → vegetarian_status filter + category guard ─────
        diet = str(self.user.get("diet_type") or "non_veg").lower()
        is_veg_user = diet in ("veg", "vegan")

        if is_veg_user:
            veg_filter: dict = {"vegetarian_status": "Veg"}
            # Compile a regex that will exclude foods whose *category* contains
            # non-veg keywords — guards against DB data quality issues where a
            # fish/meat item has been stored as vegetarian_status="Veg".
            _cat_exclude = re.compile(
                r"\b(?:fish|seafood|meat|poultry|chicken|mutton|lamb|beef|pork|shrimp|prawn)",
                re.I,
            )
        elif diet == "eggetarian":
            veg_filter = {"vegetarian_status": {"$in": ["Veg", "Eggetarian"]}}
            _cat_exclude = re.compile(
                r"\b(?:seafood|mutton|lamb|beef|pork|shrimp|prawn)", re.I
            )
        else:
            veg_filter = {}
            _cat_exclude = None

        # ── Calorie budget per individual item ────────────────────────
        max_item_cal = max(50.0, min(remaining_cal * 0.85, 650.0)) if remaining_cal > 0 else 350.0
        min_item_cal = 30.0

        # ── Priority categories (goal + context-sensitive) ───────────────
        goal = str(self.user.get("fitness_goal") or "maintain").lower()

        # Goal → ordered macro priority: (protein_weight, carb_weight, fat_weight)
        # Used later for scoring; here drives category list priority.
        _goal_prio: dict[str, list[str]] = {
            "lose_weight": [
                "Dal", "Vegetable", "Salad", "Protein/Vegetarian",
                "Main Dish", "Curry", "Breakfast", "Soup",
                "Protein/Fish", "Protein/Meat",
                "Bread", "Snack",
            ],
            "gain_muscle": [
                "Protein/Vegetarian", "Protein/Fish", "Protein/Meat",
                "Dal", "Main Dish", "Curry",
                "Bread", "Rice Dish", "Breakfast", "Snack",
            ],
            "gain_weight": [
                "Rice Dish", "Bread", "Main Dish", "Curry",
                "Dal", "Breakfast", "Snack",
                "Protein/Vegetarian", "Protein/Fish",
            ],
            "maintain": [
                "Breakfast", "Dal", "Main Dish", "Curry", "Bread",
                "Rice Dish", "Snack", "Vegetable", "Salad",
                "Protein/Vegetarian", "Protein/Fish", "Protein/Meat",
            ],
        }
        base_cats = _goal_prio.get(goal, _goal_prio["maintain"])

        # Context override: exercise or low protein always boosts protein cats first
        if did_exercise or prot_status == "low":
            prot_cats = (["Protein/Fish", "Protein/Meat"] if not is_veg_user
                         else ["Protein/Vegetarian"])
            base_cats = prot_cats + [c for c in base_cats if c not in prot_cats]
        elif carb_status == "over":
            low_carb_cats = ["Protein/Vegetarian", "Protein/Fish", "Protein/Meat",
                             "Vegetable", "Salad", "Dal"]
            base_cats = low_carb_cats

        # Filter by veg if needed
        if is_veg_user:
            base_cats = [c for c in base_cats
                         if c not in ("Protein/Fish", "Protein/Meat")]

        priority_cats = base_cats

        # ── Fetch candidates from MongoDB ─────────────────────────────
        def _fetch_cats(category_list: list[str], per_cat: int = 10) -> list[dict]:
            out: list[dict] = []
            seen: set[str] = set()
            for cat in category_list:
                q: dict = {
                    **veg_filter,
                    "category": {"$regex": cat, "$options": "i"},
                    "calories_kcal": {"$gte": min_item_cal, "$lte": max_item_cal},
                    "is_verified": True,
                }
                for d in self.food_repo.foods.find(q, {"_id": 0}).limit(per_cat):
                    fid = d.get("food_id")
                    cat_val = d.get("category", "")
                    if _cat_exclude and _cat_exclude.search(cat_val):
                        continue   # DB data quality guard
                    if fid not in seen:
                        seen.add(fid)
                        out.append(d)
            return out

        candidates = _fetch_cats(priority_cats, per_cat=10)

        # Broad fallback if too few
        if len(candidates) < 6:
            broad_q: dict = {
                **veg_filter,
                "calories_kcal": {"$gte": min_item_cal, "$lte": max_item_cal},
                "is_verified": True,
            }
            seen_ids = {c["food_id"] for c in candidates}
            for d in self.food_repo.foods.find(broad_q, {"_id": 0}).limit(50):
                fid = d.get("food_id")
                cat_val = d.get("category", "")
                if _cat_exclude and _cat_exclude.search(cat_val):
                    continue
                if fid not in seen_ids:
                    candidates.append(d)
                    seen_ids.add(fid)

        if not candidates:
            return []

        # ── Compute real nutrition for each candidate ─────────────────
        qty_1s = PQ(amount=1.0, unit="serving", raw_input="1 serving", confidence=1.0)

        # Goal → scoring weights (protein_w, carb_w, fat_w, cal_fit_w)
        _goal_weights = {
            "lose_weight": (0.35, 0.05, 0.10, 0.50),
            "gain_muscle": (0.45, 0.15, 0.05, 0.35),
            "gain_weight": (0.15, 0.35, 0.15, 0.35),
            "maintain":    (0.25, 0.20, 0.10, 0.45),
        }
        pw, cw, fw, calw = _goal_weights.get(goal, _goal_weights["maintain"])

        # Keywords that indicate uncommon / restaurant-only / non-practical foods.
        # Suggestions should be everyday home-cooked Indian foods.
        _impractical = re.compile(
            r"\b(cocktail|flambe|souffle|mousse|parfait|tartare|carpaccio|"
            r"bisque|creme brulee|fondue|tempura|teriyaki|sashimi|taco|burrito|"
            r"lasagna|risotto|paella|couscous|quinoa|acai|granola bar)\b",
            re.I,
        )

        scored: list[dict] = []
        for food in candidates:
            try:
                if not (food.get("calories_kcal") and food.get("serving_size_g")):
                    continue
                # Skip clearly non-practical / non-Indian foods
                fname = food.get("food_name", "")
                if _impractical.search(fname):
                    continue
                r = calc.calculate(food, qty_1s)
                if r.calories < min_item_cal:
                    continue
                # Protein density: g protein per 100 kcal
                pdens = (r.protein_g / r.calories * 100) if r.calories > 0 else 0
                # Carb density: g carbs per 100 kcal
                cdens = (r.carbs_g  / r.calories * 100) if r.calories > 0 else 0
                # Calorie fit: 1.0 when 2 servings ≈ remaining budget
                cal_fit = 1.0 - abs(r.calories * 2 - remaining_cal) / max(remaining_cal, 1)
                # Goal-weighted score
                score = (calw * max(0, cal_fit) +
                         pw   * min(pdens / 15.0, 1.0) +
                         cw   * min(cdens / 30.0, 1.0) +
                         fw   * (1.0 - min(r.fat_g / max(r.calories * 0.01, 1), 1.0)))
                scored.append({
                    "food":   food,
                    "result": r,
                    "score":  score,
                    "pdens":  pdens,
                    "cat":    (food.get("category") or "").lower(),
                })
            except Exception:
                continue

        if not scored:
            return []

        scored.sort(key=lambda x: x["score"], reverse=True)

        # ── Build a diverse pool: one item per category ───────────────
        seen_cats: set[str] = set()
        pool: list[dict] = []
        for item in scored:
            if item["cat"] not in seen_cats:
                seen_cats.add(item["cat"])
                pool.append(item)
            if len(pool) >= 12:
                break

        # ── Form 2-4 combinations ─────────────────────────────────────
        suggestions: list[dict] = []

        # Pass 1: paired combos from different categories
        combos_seen: set[str] = set()
        for i in range(min(6, len(pool))):
            for j in range(i + 1, min(8, len(pool))):
                if pool[i]["cat"] == pool[j]["cat"]:
                    continue
                key = f"{i}-{j}"
                if key in combos_seen:
                    continue
                combos_seen.add(key)
                try:
                    pr, sr = pool[i]["result"], pool[j]["result"]
                    tot_cal  = pr.calories + sr.calories
                    tot_prot = pr.protein_g + sr.protein_g
                    tot_carb = pr.carbs_g   + sr.carbs_g
                    tot_fat  = pr.fat_g     + sr.fat_g
                    # Accept if total fits within 35-115 % of remaining budget
                    if remaining_cal > 0 and not (0.35 * remaining_cal <= tot_cal <= 1.15 * remaining_cal):
                        continue
                    name = (f"{pool[i]['food']['food_name_display']} + "
                            f"{pool[j]['food']['food_name_display']}")
                    suggestions.append({"display": name, "calories": round(tot_cal),
                                        "protein": round(tot_prot, 1),
                                        "carbs":   round(tot_carb, 1),
                                        "fat":     round(tot_fat,  1)})
                except Exception:
                    continue
                if len(suggestions) >= max_suggestions - 1:
                    break
            if len(suggestions) >= max_suggestions - 1:
                break

        # Pass 2: single-item options to fill up to max_suggestions
        used_names: set[str] = {sg["display"].split("+")[0].strip() for sg in suggestions}
        for item in pool:
            if len(suggestions) >= max_suggestions:
                break
            name = item["food"]["food_name_display"]
            if name in used_names:
                continue
            r = item["result"]
            suggestions.append({"display": name, "calories": round(r.calories),
                                 "protein": round(r.protein_g, 1),
                                 "carbs":   round(r.carbs_g, 1),
                                 "fat":     round(r.fat_g,   1)})

        # ── Format output strings ─────────────────────────────────────
        from food_emoji import get_food_emoji
        lines: list[str] = []
        for sg in suggestions[:max_suggestions]:
            emoji = get_food_emoji(sg["display"].split("+")[0].strip())
            lines.append(
                f"• {emoji} **{sg['display']}** — ~**{sg['calories']} kcal** "
                f"| Protein: **{sg['protein']}g** "
                f"| Carbs: **{sg['carbs']}g** "
                f"| Fat: **{sg['fat']}g**"
            )
        return lines


    def validate_food_items_before_calculation(self, items: list) -> list:
        """
        Mandatory validation pass. Inspect ALL foods together BEFORE any
        calorie calculation. Returns a list of dicts describing what is
        calorie calculation. Returns a list of dicts describing what is
        missing per item, e.g. [{"index": 0, "field": "quantity"}].
        An empty list means every item is complete and calculation may proceed.

        Checks per item:
          - food name present + nutrition doc available
          - quantity present
          - required variant present (only when food has >1 real variants)
        """
        problems = []
        for idx, it in enumerate(items):
            # food name + nutrition data
            if not it.get("food_name") or not it.get("food_doc"):
                problems.append({"index": idx, "field": "food_name"})
                continue
            # quantity
            if it.get("needs_quantity") and not it.get("quantity"):
                problems.append({"index": idx, "field": "quantity"})
                continue
            # variant (only if the food genuinely requires a choice)
            if it.get("needs_variant") and (not it.get("variant") or it.get("variant") == "normal"):
                problems.append({"index": idx, "field": "variant"})
                continue
        return problems

    # ------------------------------------------------------------------
    # Answer validation helpers (prevent food-name corruption)
    # ------------------------------------------------------------------

    def _looks_like_quantity_answer(self, message: str) -> bool:
        """
        True when the message is plausibly a quantity answer (button or typed):
        "2", "2 pieces", "1 bowl", "150g", "देढ़", "દોઢ ભાખરી", "1.5 plates".
        False when it is clearly a new-food sentence such as
        "sathe jalebi pn khadhi" (contains a connective + a verb but no leading qty).
        """
        m = message.strip().lower()
        if not m:
            return False

        # Connective words that signal a NEW food, not a quantity.
        new_food_markers = (
            " ane ", " aur ", " and ", " sathe", " saath", " ke saath",
            " pan ", " pn ", " bhi ", " plus ", " with ", " sāthe",
            "અને", "और", "સાથે",
        )
        padded = f" {m} "
        has_new_food_marker = any(mk in padded for mk in new_food_markers)

        # Past-tense eating verbs indicate a full sentence, not a quantity.
        eat_verbs = ("khadhi", "khadha", "khadhu", "khaya", "khayi", "khaye",
                     "lidhu", "lidhi", "lidha", "khai", "jamya", "ate", "eaten", "had")
        has_eat_verb = any(re.search(rf"\b{v}\b", m) for v in eat_verbs)

        # A leading digit / number word is a strong quantity signal.
        starts_with_number = bool(re.match(r"^\s*\d", m))

        # Recognised quantity units anywhere.
        unit_re = re.compile(
            r"\b\d*\.?\d*\s*(?:g|gm|gram|grams|kg|ml|l|litre|liter|"
            r"piece|pieces|pcs|bowl|bowls|katori|katoris|cup|cups|"
            r"plate|plates|glass|glasses|serving|servings|slice|slices|"
            r"roti|rotis|tbsp|tsp|scoop|scoops)\b",
            re.IGNORECASE,
        )
        has_unit = bool(unit_re.search(m))

        # Vernacular fractional quantity words (deodh/dodh = 1.5, adhu/aadhi = 0.5, etc.)
        qty_words = ("dodh", "deodh", "dedh", "adhu", "aadhu", "aadhi", "adhi",
                     "sava", "paune", "ek", "be", "be", "do", "teen", "char",
                     "ਦੋ", "देढ़", "દોઢ", "ડોઢ")
        has_qty_word = any(w in m for w in qty_words)

        # If it clearly names a new food (marker + verb), it's NOT a quantity.
        if has_new_food_marker and has_eat_verb:
            return False

        # Positive quantity signals.
        if starts_with_number or has_unit:
            return True
        if has_qty_word and not has_eat_verb:
            return True

        # Short single/double token reply with no verb → treat as quantity
        # (covers a bare button click like "2 pieces" or "1 bowl").
        if len(m.split()) <= 2 and not has_eat_verb and not has_new_food_marker:
            return True

        return False

    def _extract_quantity_answer(self, message: str) -> str:
        """
        Normalise a typed quantity answer to a clean 'amount unit' string.
        Falls back to the raw trimmed message if parsing is inconclusive.
        """
        parsed = parse_quantity(message)
        if parsed.amount is not None and parsed.amount > 0 and parsed.unit:
            return f"{parsed.amount:g} {parsed.unit}"
        return message.strip()

    def _reply_is_known_food(self, message: str) -> bool:
        """
        True when a short flow reply actually resolves to a KNOWN food in the
        local database (exact/alias/variant/fuzzy) rather than being a genuine
        quantity/variant answer.

        This is what lets a bare one-word food like "Dal", "Chhas" or "Jalebi"
        — typed while the bot is asking "Roti ketli?" — be recognised as a NEW
        food and enqueued, instead of being stored as the current food's
        quantity.

        LOCAL-ONLY on purpose: we must not fire external API/AI calls just to
        decide whether a reply is a quantity. A real quantity answer ("2",
        "1 bowl", "150 g", "2 roti") is excluded up front so it stays a quantity.
        """
        m = message.strip().lower()
        if not m:
            return False

        # A leading number is a quantity answer ("2", "2 pieces", "2 roti"),
        # never a new-food-only reply.
        if re.match(r"^\s*\d", m):
            return False

        # Pure quantity/unit tokens ("bowl", "plate", "glass", "katori"…) are
        # quantity answers, not foods.
        pure_qty_units = {
            "bowl", "bowls", "katori", "katoris", "cup", "cups", "plate", "plates",
            "glass", "glasses", "piece", "pieces", "pcs", "serving", "servings",
            "slice", "slices", "scoop", "scoops", "tbsp", "tsp", "g", "gm", "gram",
            "grams", "kg", "ml", "l",
        }
        if all(tok in pure_qty_units for tok in m.split()):
            return False

        cleaned = self._clean_food_query(message, original_message=message)
        if not cleaned:
            return False

        try:
            res = self.search_engine.search(cleaned)
        except Exception:
            return False

        mt = res.match_type.value
        if mt in ("exact_name", "exact_alias", "variant_strip"):
            return True
        if mt in ("fuzzy_name", "fuzzy_alias") and res.confidence >= 0.80:
            return True
        return False

    def _looks_like_variant_answer(self, v_clean: str, curr: dict) -> bool:
        """
        True when the message is a plausible variant answer for the current food.
        Rejects new-food sentences (connective + eat verb).
        """
        m = f" {v_clean} "
        new_food_markers = (" ane ", " aur ", " and ", " sathe", " pan ", " pn ",
                            " bhi ", " with ", "અને", "और", "સાથે")
        eat_verbs = ("khadhi", "khadha", "khaya", "lidhu", "lidhi", "ate", "had")
        if any(mk in m for mk in new_food_markers) and any(re.search(rf"\b{v}\b", v_clean) for v in eat_verbs):
            return False
        # A short reply or one that matches an offered variant option is valid.
        opts = (curr.get("food_doc") or {}).get("preparation_variants") or []
        opts_lower = {o.lower() for o in opts}
        if v_clean in opts_lower:
            return True
        return len(v_clean.split()) <= 2

    async def _enqueue_new_foods_from_reply(self, message: str, flow: PendingFlow) -> bool:
        """
        The user typed a new food (or foods) instead of answering the pending
        question. Detect them, build queue objects, and append to flow.items so
        each food keeps its own independent state. The currently-pending item is
        left untouched (its needs_quantity/needs_variant stays True).

        Returns True if at least one new food was enqueued.
        """
        # Try multi-food split first, else treat the whole message as one food.
        parts = self._detect_multi_food(message)
        if not parts:
            single = self._clean_food_query(message, original_message=message)
            parts = [single] if single else []

        existing_ids = {it.get("food_id") for it in (flow.items or [])}
        added = False
        for part in parts:
            cleaned = self._clean_food_query(part, original_message=part)
            if not cleaned:
                continue
            q_obj = await self._build_food_item_queue_object(
                cleaned, part, shared_meal_type=flow.shared_meal_type
            )
            if q_obj and q_obj.get("food_id") not in existing_ids:
                flow.items.append(q_obj)
                existing_ids.add(q_obj.get("food_id"))
                added = True

        if added and getattr(flow, "unresolved_items", None):
            remaining = []
            for u in flow.unresolved_items:
                is_gen, _ = is_generic_food_query(u)
                if not is_gen:
                    remaining.append(u)
            flow.unresolved_items = remaining

        return added

    async def _handle_multi_food(self, items: list[str], original_msg: str, auto_log: bool) -> ChatResponse:
        """
        Handle multiple food items sequentially via queue.
        Resolves each food and builds a questionnaire queue.
        Any item that cannot be resolved is reported back to the user
        instead of being silently dropped.
        """
        meal = self._detect_meal_type(original_msg)
        queue_items = []
        failed_items = []

        for item_text in items:
            q_obj = await self._build_food_item_queue_object(item_text, item_text, shared_meal_type=meal)
            if q_obj:
                queue_items.append(q_obj)
            else:
                # Track the human-readable name of the item that failed to resolve
                clean_name = self._clean_food_query(item_text, original_message=item_text)
                failed_items.append(clean_name or item_text.strip())

        if not queue_items:
            generic_clarifications = []
            for item_text in failed_items:
                is_gen, gen_cat = is_generic_food_query(item_text)
                if is_gen:
                    cmsg = get_generic_clarification_message(gen_cat, self.lang)
                    if cmsg not in generic_clarifications:
                        generic_clarifications.append(cmsg)
            if generic_clarifications:
                return ChatResponse(
                    message="\n\n".join(generic_clarifications),
                    intent=Intent.CLARIFICATION_NEEDED, success=True, language=self.lang,
                )
            names = ", ".join(failed_items) if failed_items else "those foods"
            return ChatResponse(
                message=f"Sorry, I couldn't find: {names}. Could you try different names or spellings?",
                intent=Intent.CLARIFICATION_NEEDED, success=False, language=self.lang,
            )

        flow = PendingFlow(
            items=queue_items,
            current_index=0,
            shared_meal_type=meal,
        )
        # Remember any items we couldn't resolve so the queue processor can
        # mention them in the first response.
        if failed_items:
            flow.unresolved_items = failed_items
        await self.flow_mgr.save_flow_state(flow)
        return await self._process_sequential_queue(flow)

    def _clean_food_item(self, item: str) -> str:
        """Clean a single food item text — remove filler/meal words, keep food name + quantity."""
        words = item.lower().strip().split()
        filler = {"me", "ma", "mein", "ne", "ka", "ki", "ke", "nu", "ni", "na",
                  "ek", "aaje", "aaj", "kale", "kal"}
        cleaned = [w for w in words if w not in filler]
        return " ".join(cleaned).strip()

    # ------------------------------------------------------------------
    # Deterministic Daily-Log (read/view) routing
    # ------------------------------------------------------------------
    def _resolve_read_date(self, msg: str) -> str:
        """Deterministically resolve TODAY vs YESTERDAY for a Daily-Log READ.

        NEVER trusts an LLM-generated absolute date. Looks only for explicit
        yesterday/today markers in the raw message (en/hi/gu/roman-gu). Defaults
        to 'today' when no day marker is present.
        """
        m = " " + msg.lower().strip() + " "
        yesterday_markers = (
            "yesterday", "yday", "kal ", " kal", "kal ka", "kal ki", "kal ke",
            "gay kale", "gay kal", "gate kale", "kale ", " kale", "gai kal",
            "beeta kal", "pichhle din",
        )
        # "kale"/"kal" are yesterday in gu/hi. Guard: only treat as yesterday
        # when it appears as a standalone token (already padded with spaces).
        for mk in yesterday_markers:
            if mk in m:
                return "yesterday"
        return "today"

    def _extract_read_meal(self, msg: str) -> Optional[str]:
        """Deterministically extract the meal a READ query refers to.

        Unlike `_detect_meal_type` (which collapses evening→snack for LOGGING),
        this keeps breakfast/lunch/snack/evening/dinner DISTINCT because a user
        reading their log asks for a specific slot. Covers en/hi/gu/roman-gu/
        hinglish. Returns None when no meal slot is named (whole-day read).
        """
        m = " " + msg.lower().strip() + " "

        def has_any(words) -> bool:
            return any((" " + w + " ") in m for w in words)

        # Order matters: check the more specific 'evening' and 'snack' slots
        # before the broad daytime anchors so "evening"/"sanje" is not swallowed.
        breakfast = ["breakfast", "brekfast", "breckfast", "nashta", "nashte",
                     "subah", "subhe", "savar", "savare", "savre", "savaar",
                     "morning"]
        lunch = ["lunch", "dopahar", "dopaher", "bhojan", "bapor", "bapore",
                 "bapori", "baporme", "midday", "noon"]
        # dinner: explicit night/dinner words
        dinner = ["dinner", "supper", "raat", "raatre", "ratre", "raatri"]
        # evening: distinct slot (kept separate from generic snack)
        evening = ["evening", "sanje", "sanj", "saanj", "sanjeye", "shaam", "sham"]
        # snack: explicit snack words. Gujarati "nasta"/"nasto" = snack; the
        # Hindi "nashta"/"nashte" (breakfast) is handled above. Aligns with
        # conversation.MEAL_WORDS (nasta/nasto → snack).
        snack = ["snack", "snacks", "nasta", "nasto", "timepass"]

        if has_any(evening):
            return "evening"
        if has_any(snack):
            return "snack"
        if has_any(dinner):
            return "dinner"
        if has_any(lunch):
            return "lunch"
        if has_any(breakfast):
            return "breakfast"
        return None

    def _looks_like_nutrition_lookup(self, msg: str) -> Optional[str]:
        """Deterministically detect a nutrition-knowledge lookup and return the
        food term, e.g. "how many calories in a banana" → "banana".

        A nutrition lookup asks about the macros/calories OF A FOOD (generic
        knowledge), NOT the user's own daily total. It must route to
        get_calories, never to Daily-Log.

        Signal = a macro/energy word + a preposition pointing at a food
        (in / of / me / mein) AND no personal/day context (today, yesterday,
        i ate, i logged, so far, khaya). Returns the food string, or None.
        """
        m = " " + msg.lower().strip() + " "

        # A macro/energy keyword must be present.
        macro_kw = ("calorie", "calories", "kcal", "protein", "carb", "carbs",
                    "carbohydrate", "fat", "fats", "fiber", "fibre", "sugar",
                    "macros", "nutrition",
                    # fitness abbreviations used in gym context
                    " pn ",   # "pn" = protein in Gujarati gym slang
                    " prot ", " cal ",
                    )
        if not any(k in m for k in macro_kw):
            return None

        # Personal / daily-total context → this is a Daily-Log total, not a
        # generic food lookup. Bail so the Daily-Log router handles it.
        personal_ctx = (
            " today ", " yesterday ", " so far ", " i eat ", " i ate ",
            " i had ", " i have ", " i logged ", " i consumed ", " did i ",
            " my ", " aaj ", " aaje ", " kal ", " gay kale ", " khaya ",
            " khadhu ", " khadhi ", " lidhu ", " eaten ", " burn ", " burned ",
            " total ", " remaining ", " baki ",
        )
        if any(c in m for c in personal_ctx):
            return None

        # A preposition / question structure that introduces the food object.
        import re as _re
        _macro = r"(?:calories?|kcal|protein|carbs?|carbohydrates?|fat|fats|fiber|fibre|sugar|macros)"
        prep_patterns = [
            rf"\b{_macro}\b.*?\bin\b\s+(.*)",
            rf"\b{_macro}\b.*?\bof\b\s+(.*)",
            rf"\bcontent\s+of\b\s+(.*)",
            rf"\bdoes\s+(.*?)\s+have\b",                    # "how many calories does a samosa have"
            rf"\bhow\s+many\s+{_macro}\b.*?\b(?:does|in|for)\b\s+(.*)",
            r"(.*?)\s+(?:me|mein|ma)\s+(?:kitni|kitna|ketli|ketlu)\b",   # "roti me kitni calories"
            rf"\b(?:kitni|kitna|ketli|ketlu)\b\s+{_macro}\b.*?\b(?:hoti|hota|hoy|che|hai)\b\s*(?:hai)?\s*(.*)",
        ]
        for pat in prep_patterns:
            mt = _re.search(pat, m.strip())
            if mt:
                food = (mt.group(1) or "").strip()
                # Strip leading quantity phrases: "100g", "100 grams", "a slice
                # of", "a can of", "a bowl of", "one cup of", articles, etc.
                food = _re.sub(r"^\s*\d+\s*(?:g|gm|gms|gram|grams|kg|ml|l|oz)\b", " ", food)
                food = _re.sub(r"\b(?:a|an|one|two|three|ek|do|teen)\s+"
                               r"(?:slice|slices|piece|pieces|bowl|bowls|cup|cups|"
                               r"glass|glasses|plate|plates|can|cans|katori|katoris|"
                               r"spoon|spoons|tablespoon|handful|scoop)\s+of\b", " ", food)
                food = _re.sub(r"\b(a|an|the|one|ek|of|have|has|kitni|kitna|ketli|"
                               r"ketlu|hoti|hota|hai|hoy|che|me|mein|does|it|there|"
                               r"is|are|per|each|grams?|gm|gms|g|kg|ml|oz)\b", " ", food)
                food = _re.sub(r"[^\w\s/-]", " ", food)
                food = _re.sub(r"\s+", " ", food).strip()
                if food and len(food) >= 2:
                    return food
        return None

    def _classify_exercise_read(self, msg: str) -> Optional[str]:
        """Deterministically detect an EXERCISE-history READ (query_exercise).

        e.g. "show my workout log", "how much did I burn today",
             "aaj ki exercise batao". Returns a date string ('today'/'yesterday')
        when it is an exercise read, else None.

        Requires BOTH a read/question cue AND an exercise-domain word, and must
        NOT contain a food word (so food reads aren't hijacked).
        """
        m = " " + msg.lower().strip() + " "

        exercise_domain = (
            "workout", "workouts", "work out", "exercise", "exercises",
            "excercise", "training", "trained", "train", "gym", "cardio",
            "physical activity", "activity", "vyayam", "kasrat", "kasarat",
        )
        # "burn/burned calories" is an exercise read too.
        burn_cues = ("burn", "burned", "burnt", "calories i burn",
                     "calories did i burn", "burn working out")

        has_domain = any(w in m for w in exercise_domain)
        has_burn = any(b in m for b in burn_cues)
        if not (has_domain or has_burn):
            return None

        read_cues = (
            "show", "view", "display", "dikhao", "dikha do", "batao", "batavo",
            "list", "recap", "log", "logs", "summary", "history", "hisaab",
            "how much", "how many", "what did i", "what have i", "what physical",
            "kaunsi", "kya", "su exercise", "kari batavo", "did i do",
            "did i log", "have i done", "have i logged",
        )
        if not any(c in m for c in read_cues):
            return None

        # Guard: if a concrete amount is present it's a LOG, not a read
        # ("did 20 pushups"), so let exercise-log routing handle it.
        import re as _re
        if _re.search(r"\b\d+\s*(?:min|minute|minutes|rep|reps|set|sets|km|steps|laps|rounds)\b", m):
            return None

        return self._resolve_read_date(msg)

    def _classify_daily_log_read(self, msg: str):
        """Deterministically decide whether a message is a Daily-Log READ.

        A READ is a request to VIEW/SHOW already-logged data (a day's summary /
        total calories, or the foods in a specific meal). It must bypass food
        detection entirely.

        Returns a dict {intent, date, meal_type} when the message is a READ, or
        None when it is not (so normal LLM/food routing continues).

        Deterministic and language-agnostic across en/hi/gu/roman-gu/hinglish.
        Designed to NEVER catch a food-LOGGING statement:
          • Logging statements name a concrete food and/or a quantity
            ("2 roti", "poha khaya"); they lack a view/question cue.
          • Reads carry a view/question cue ("show", "dikhao", "batao",
            "kitni calories", "su khadhu", "what did I eat", "summary", "report")
            and no concrete food+quantity.
        """
        m_raw = " " + msg.lower().strip() + " "
        m = " " + re.sub(r"[^\w\s]", " ", msg).lower() + " "

        # If the message contains a digit it is almost certainly a logging
        # statement ("2 roti", "1 bowl dal") — never a read. Bail out.
        # Exception: pure nutrition lookups may carry a quantity ("100g paneer")
        # and are handled by the nutrition-lookup router, not here.
        if any(ch.isdigit() for ch in msg) and self._looks_like_nutrition_lookup(msg) is None:
            return None

        # A generic nutrition-knowledge lookup ("calories in a banana") is NOT a
        # Daily-Log read — it must route to get_calories. Bail so the caller's
        # nutrition-lookup router handles it.
        if self._looks_like_nutrition_lookup(msg) is not None:
            return None

        # An exercise-history read ("show my workout log") is NOT a food read.
        if self._classify_exercise_read(msg) is not None:
            return None

        # ── View / show / report cues (whole-day or meal) ──────────────────
        show_cues = (
            "show", "view", "display", "dikhao", "dikha do", "dikhaao",
            "batao", "bata do", "batavo", "batao na", "dekhao", "dekha",
            "report", "summary", "daily log", "day log", "full day log",
            "pura log", "aakho log", "aakha din", "poora log", "total log",
            "aaj nu log", "aaj ka log", "today log", "todays log", "today s log",
            "log batavo", "log batao", "log dikhao", "log show",
            # read-back / recap phrasings
            "recap", "read back", "read out", "rundown", "run down",
            "walk me through", "pull up", "pull my", "list out", "list the",
            "list my", "give me my", "give me a", "tell me my", "summarise",
            "summarize", "meal history", "food diary", "food breakdown",
            "food tally", "food log", "meals for today", "logged under",
            "consist of", "have i logged", "did i log", "things did i log",
            "my plate log", "logged for", "entries", "ginn ke batao",
            "khana wala log", "jaman batavo", "khana log",
        )
        # ── Total-calorie READ cues (route to summary, NOT get_calories) ───
        total_cal_cues = (
            "total calorie", "total calories", "calorie count", "calories count",
            "how many calories", "how much calorie", "how many calorie",
            "calories today", "calories yesterday", "calorie total",
            "kitni calorie", "kitni calories", "kitna calorie", "ketli calorie",
            "ketli calories", "total kitna", "total kitni", "total ketli",
            "total ketlu", "kul calorie", "kul kitni", "net calorie",
            "calories eaten", "calorie eaten", "in total today", "in total",
            "how much did i eat", "how much i ate", "how much have i eaten",
            "total log", "today's total", "todays total", "today s total", "today total",
            "aaj nu total", "aaj ka total", "aaje no total", "aaje nu total",
            "aaj total", "aaje total", "din ka total", "day total", "total aaj", "total aaje",
        )
        # ── Interrogative "what did I eat/have" READ cues ──────────────────
        # Kept flexible: "what ... did i eat/have" may have a meal word between
        # (e.g. "what snacks did I have"), so we match the split forms too.
        what_ate_cues = (
            "what did i eat", "what did i have", "what all did i eat",
            "what have i eaten", "what did i consume", "what i ate",
            "what did i", "what all did i", "what have i", "did i eat",
            "did i have", "did i consume", "what all is", "what is in",
            "what's in", "whats in", "what all in", "in my breakfast",
            "in my lunch", "in my dinner", "in my snack",
            "kya khaya", "kya khaaya", "kya liya", "kya tha", "kitna khaya",
            "su khadhu", "su khadu", "su khadh", "su lidhu", "su jamyu",
            "su su khadhu", "su su khadu", "shu khadhu", "shu lidhu",
            "aaje shu khadhu", "aaje su khadhu", "aaj kya khaya", "aaje su jamyu", "aaje shu jamyu",
        )
        # ── Meal-read noun cues: "<meal> foods/items" is a READ (list a meal) ─
        meal_read_noun_cues = ("foods", "items", "food items", "khana", "khana dikhao")

        has_show = any(c in m or c in m_raw for c in show_cues)
        has_total = any(c in m or c in m_raw for c in total_cal_cues)
        has_what = any(c in m or c in m_raw for c in what_ate_cues)
        # A named meal + a "foods/items" noun (or a show/what cue) is a meal read.
        _meal_named = self._extract_read_meal(msg) is not None
        has_meal_noun = _meal_named and any(c in m or c in m_raw for c in meal_read_noun_cues)

        if not (has_show or has_total or has_what or has_meal_noun):
            return None

        read_date = self._resolve_read_date(msg)
        meal = self._extract_read_meal(msg)

        # A whole-day total / summary / full-day view → get_summary.
        # A meal-scoped read (or an unscoped "what did I eat" list) → query_meal.
        if has_total and meal is None:
            intent = Intent.GET_SUMMARY
        elif meal is not None:
            intent = Intent.QUERY_MEAL
        elif has_what or has_meal_noun:
            # "what all did I eat today" (no meal) → list the day's items.
            intent = Intent.QUERY_MEAL
        else:
            # "show today's summary / report / full day log" → summary.
            intent = Intent.GET_SUMMARY

        return {"intent": intent, "date": read_date, "meal_type": meal}

    def _looks_like_structured_exercise(self, msg: str) -> bool:
        """Detect STRUCTURED exercise phrasing even when the exercise NAME is not
        in the trigger set / dataset (e.g. "did 3 sets of overhead press",
        "did 6 rounds of tabata", "completed 18 chin ups").

        Matches an exercise ACTION verb and/or a sets/reps/rounds structure.
        Deliberately conservative: it requires an exercise-domain structure so a
        food sentence ("had 3 bowls of dal") cannot match — food uses portion
        units (bowl/plate/cup/katori/glass/piece), never sets/reps/rounds.
        """
        import re as _re
        m = " " + msg.lower().strip() + " "

        # sets / reps / rounds structure → exercise (foods never use these).
        if _re.search(r"\b\d+\s*(?:sets?|reps?|rounds?|rep\b)\b", m):
            return True
        if _re.search(r"\bsets?\s+of\b", m) or _re.search(r"\brounds?\s+of\b", m):
            return True

        # Exercise action verbs + a distance/duration/count (not a food portion).
        action_verbs = (
            "did", "completed", "performed", "held", "ran", "run", "jogged",
            "walked", "swam", "swim", "cycled", "rowed", "sprinted", "climbed",
            "skipped", "lifted", "stretched", "played", "ghanta", "chalyo",
        )
        has_verb = any((" " + v + " ") in m for v in action_verbs)
        # amount that is NOT a food portion unit
        has_amount = _re.search(
            r"\b\d+\s*(?:min|mins|minute|minutes|hour|hours|km|k\.?m\.?|"
            r"steps?|laps?|m\b|meters?|metres?|seconds?|secs?|ghanta|ghante|"
            r"skips?|kicks?)\b", m)
        if has_verb and has_amount:
            return True

        # "did/completed/performed <N> <activity words>" with NO food-portion
        # unit → exercise (e.g. "did 200 skips", "did 80 high knees",
        # "completed 18 chin ups", "did 30 box jumps"). Foods with a count use a
        # portion unit (piece/plate/bowl/cup/…), which we explicitly exclude.
        _did_verbs = ("did", "completed", "performed", "finished", "knocked out")
        _food_portion = (
            r"\b\d+\s*(?:piece|pieces|plate|plates|bowl|bowls|cup|cups|glass|"
            r"glasses|katori|katoris|slice|slices|spoon|spoons|tablespoon|"
            r"tablespoons|scoop|scoops|packet|packets|serving|servings)\b")
        if any((" " + v + " ") in m for v in _did_verbs):
            if _re.search(r"\b\d+\b", m) and not _re.search(_food_portion, m):
                return True
        return False

    def _is_offtopic_request(self, msg: str) -> bool:
        """Deterministically detect an OFF-TOPIC assistant request / question that
        is neither food, exercise, nor a log read — so it is rejected instead of
        being coerced into the food flow by a coincidental branded-alias match.

        Conservative: fires only on clear command/question structures paired
        with a non-food subject, and bails if any eating/food/exercise/read cue
        is present (those are handled by the normal routers).
        """
        m = " " + msg.lower().strip() + " "
        import re as _re

        # If it mentions eating/food/drinking or exercise or a log-read, it is
        # NOT off-topic — let the proper router handle it.
        food_ex_cues = (
            " ate ", " eat ", " eaten ", " khaya ", " khadha ", " khadhi ",
            " khadhu ", " khadho ", " khayi ", " pidhi ", " piya ", " drank ",
            " workout", " exercise", " calorie", " calories", " protein",
            " carbs", " summary", " log ", " logged ", " breakfast", " lunch",
            " dinner", " snack", " meal ", " meals ",
        )
        # NOTE: " meal " above would also catch "ate my meal"; but the eating
        # verb handling elsewhere rejects that as ambiguous. Off-topic gate only
        # needs the clear non-food requests.
        if any(c in m for c in food_ex_cues):
            return False

        # Off-topic command verbs / assistant-request phrasing.
        offtopic_patterns = [
            r"\b(?:book|reserve)\s+(?:me\s+)?(?:a\s+)?(?:cab|taxi|hotel|room|table|flight|ticket)",
            r"\breset\s+(?:my\s+)?password",
            r"\bsend\s+(?:an?\s+)?(?:email|message|text|mail)",
            r"\b(?:remind|reminder)\b",
            r"\bset\s+(?:an?\s+)?alarm",
            r"\bopen\s+the\b",
            r"\bplay\s+(?:some\s+)?(?:music|song|songs|video)",
            r"\btranslate\b",
            r"\b(?:tell|give)\s+me\s+(?:a\s+)?(?:joke|story|news|the\s+news|bedtime)",
            r"\btell\s+me\s+about\b",
            r"\bwhat'?s?\s+(?:the\s+)?(?:capital|weather|wifi|time|score|news|traffic)\b",
            r"\bwhat\s+time\b",
            r"\bhow\s+(?:tall|far|big|old|much\s+does)\b",
            r"\bwho\s+(?:won|is|are)\b",
            r"\bwhat'?s?\s+(?:on\s+tv|my\s+horoscope|your\s+favou?rite)\b",
            r"\bhow'?s?\s+the\s+(?:traffic|weather)\b",
            r"\bis\s+it\s+going\s+to\s+rain\b",
            r"\bstock\s+market\b",
            r"\b(?:phone|car|battery|laptop)\b.*\b(?:low|won'?t|broken|dead|start)\b",
            r"\bcapital\s+of\b",
            r"\bmount\s+everest\b",
            r"\bwhat'?s?\s+\d+\s*(?:times|x|\*|plus|minus)\b",
            r"\bcricket\s+match\b",
            r"\bwhat\s+can\s+you\s+do\b",
        ]
        for pat in offtopic_patterns:
            if _re.search(pat, m):
                return True
        return False

    async def _deterministic_pre_route(self, message: str, auto_log: bool):
        """Unified deterministic router run before the LLM/food path.

        Returns a ChatResponse when it confidently classifies the message into a
        non-food-logging intent, else None (fall through to normal routing).
        Precedence: off-topic reject → nutrition → exercise read → exercise log
        → daily-log.
        """
        # 0. Off-topic / non-food command or question → REJECT deterministically.
        #    These are conversational/assistant requests ("book me a cab",
        #    "how do I reset my password", "tell me the news") that the LLM
        #    sometimes mislabels as log_food, then a stray word coincidentally
        #    matches a branded alias. Detect them by their command/question
        #    structure on the RAW message (before food cleaning strips the cue).
        if self._is_offtopic_request(message):
            return ChatResponse(
                message=get_response("not_food", self.lang, food=message.strip()),
                intent=Intent.CLARIFICATION_NEEDED, success=True, language=self.lang,
            )

        # 1. Nutrition-knowledge lookup → get_calories (NOT Daily-Log).
        nut_food = self._looks_like_nutrition_lookup(message)
        if nut_food:
            # Only accept if the food actually resolves — otherwise fall through
            # so we don't answer nutrition for a non-food.
            try:
                sres = self.search_engine.search(nut_food)
                if sres.match_type.value in ("exact_name", "exact_alias",
                                             "variant_strip", "fuzzy_name",
                                             "fuzzy_alias"):
                    parsed = LLMParseResult(
                        intent=Intent.GET_CALORIES, food_query=nut_food,
                        raw_response="deterministic_nutrition", success=True,
                    )
                    return self._handle_get_calories(parsed)
            except Exception:
                pass

        # 2. Exercise-history read → query_exercise.
        ex_read_date = self._classify_exercise_read(message)
        if ex_read_date is not None:
            parsed = LLMParseResult(
                intent=Intent.QUERY_EXERCISE, date=ex_read_date,
                raw_response="deterministic_exercise_read", success=True,
            )
            return await self._handle_query_exercise(parsed)

        # 2b. Nutrition / meal recommendation request → recommend_meal (checked before workout).
        if self._looks_like_meal_recommendation(message):
            parsed = LLMParseResult(
                intent=Intent.RECOMMEND_MEAL,
                raw_response="deterministic_meal_rec", success=True,
            )
            return await self._handle_recommend_meal(parsed, original_message=message)

        # 2c. Workout recommendation request → recommend_workout.
        if self._looks_like_workout_recommendation(message):
            return await self._handle_recommend_workout()

        # 3a. Daily-Log food read → get_summary / query_meal. Checked BEFORE
        #     exercise-log so read phrasings that contain an exercise trigger
        #     word ("pull up my breakfast entries", "run down my snacks list")
        #     are treated as reads, not as exercise logs.
        _early_read = self._classify_daily_log_read(message)
        if _early_read is not None:
            read_parsed = LLMParseResult(
                intent=_early_read["intent"], date=_early_read["date"],
                meal_type=_early_read["meal_type"],
                raw_response="deterministic_daily_log", success=True,
            )
            if _early_read["intent"] == Intent.GET_SUMMARY:
                return await self._handle_get_summary(read_parsed)
            return await self._handle_query_meal(read_parsed)

        # 3. Exercise LOG → log_exercise (deterministic, before food).
        #    Requires a clear exercise signal AND an amount/verb so we don't
        #    grab bare exercise nouns that might be a food read.
        #    Guard: if the message uses an EATING verb (and has no sets/reps/
        #    rounds structure) it's food, even if it mentions a workout context
        #    ("ate 4 dates after workout").
        import re as _re2
        _mlow = " " + message.lower().strip() + " "
        _eating_verb = any(v in _mlow for v in (
            " ate ", " eaten ", " khaya ", " khadha ", " khadhi ", " khadhu ",
            " khadho ", " khayi ", " pidhi ", " pidhu ", " piya ",
        ))
        _has_set_rep = bool(_re2.search(r"\b\d+\s*(?:sets?|reps?|rounds?)\b", _mlow)
                            or _re2.search(r"\b(?:sets?|rounds?)\s+of\b", _mlow))
        if (not (_eating_verb and not _has_set_rep)) and (
                self._looks_like_exercise(message.lower().strip())
                or self._looks_like_structured_exercise(message)):
            parsed = LLMParseResult(
                intent=Intent.LOG_EXERCISE, exercise_input=message,
                raw_response="deterministic_exercise_log", success=True,
            )
            resp = await self._handle_log_exercise(parsed, auto_log, original_message=message)
            # If exercise resolution failed (clarification), fall through to the
            # LLM path so a mis-detected exercise can still be tried as food.
            if resp is not None and resp.intent != Intent.CLARIFICATION_NEEDED:
                return resp

        # 4. Daily-Log food read → get_summary / query_meal.
        read = self._classify_daily_log_read(message)
        if read is not None:
            read_parsed = LLMParseResult(
                intent=read["intent"], date=read["date"],
                meal_type=read["meal_type"], raw_response="deterministic_daily_log",
                success=True,
            )
            if read["intent"] == Intent.GET_SUMMARY:
                return await self._handle_get_summary(read_parsed)
            return await self._handle_query_meal(read_parsed)

        return None

    def _detect_meal_type(self, msg: str) -> Optional[str]:
        """
        Classify meal type from the SEMANTIC meaning of the time expression in
        the message — not just a raw keyword.

        This is the rule-based fallback (used when the LLM is unavailable); the
        LLM path handles open-ended multilingual understanding. Here we still
        reason about the whole phrase so relative expressions are correct:

          • A meal/food eaten AROUND / AFTER / BEFORE another meal is a SNACK,
            not that meal. e.g. "bapor pachhi" (after lunch) → snack,
            "before dinner" → snack, "after dinner" → snack (late snack).
          • Morning → breakfast, midday main meal → lunch,
            afternoon/tea-time/between-meals → snack, evening → snack,
            night/dinner-time → dinner.

        Returns the meal type or None if no time/meal context is present.
        Covers common English/Hindi/Gujarati transliterations; the LLM covers
        anything beyond these.
        """
        m = " " + msg.lower().strip() + " "

        breakfast_anchors = MEAL_TIME_ANCHORS["breakfast"]
        lunch_anchors = MEAL_TIME_ANCHORS["lunch"]
        dinner_anchors = MEAL_TIME_ANCHORS["dinner"]
        snack_anchors = MEAL_TIME_ANCHORS["snack"]

        def has_any(words):
            return any((" " + w + " ") in m or (" " + w) in m or (w + " ") in m
                       for w in words)

        # ── Relative MODIFIERS — "around/after/before" another meal → snack ──
        # An item eaten relative to a main meal is a snack.
        has_relative = (has_any(MEAL_RELATIVE_MODIFIERS["after"])
                        or has_any(MEAL_RELATIVE_MODIFIERS["before"])
                        or has_any(MEAL_RELATIVE_MODIFIERS["around"]))

        # If the phrase relates the food to a MAIN meal or a daytime anchor
        # (lunch/breakfast/afternoon/morning/dinner) via a modifier, it's a snack.
        relates_to_meal = (has_any(lunch_anchors) or has_any(breakfast_anchors)
                           or has_any(dinner_anchors))
        if has_relative and relates_to_meal:
            return "snack"

        # ── Explicit snack context (tea-time, between meals, evening, "snack") ─
        if has_any(snack_anchors):
            return "snack"

        # ── Base anchors (no modifier) ─────────────────────────────────────
        if has_any(breakfast_anchors):
            return "breakfast"
        if has_any(lunch_anchors):
            return "lunch"
        if has_any(dinner_anchors):
            return "dinner"
        return None

    def _meal_from_time_of_day(self, time_of_day: Optional[str]) -> Optional[str]:
        """
        Map the LLM-extracted time_of_day to a meal type semantically:
          morning → breakfast, afternoon → snack (between-meal item),
          evening → snack, night → dinner.
        Returns None when time_of_day is absent/unknown.
        """
        if not time_of_day:
            return None
        t = str(time_of_day).strip().lower()
        return {
            "morning": "breakfast",
            "afternoon": "snack",
            "evening": "dinner",
            "night": "dinner",
        }.get(t)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------


    def _get_variant_options(self, food_name: str, category: str = "", food_doc: Optional[dict] = None) -> list[str]:
        """
        Get food variant options dynamically from MongoDB food_doc schema.
        No hardcoded food-specific lists in Python code!
        """
        if food_doc and food_doc.get("preparation_variants"):
            return food_doc["preparation_variants"]

        if food_doc and food_doc.get("variant_options"):
            return food_doc["variant_options"]

        if food_doc and food_doc.get("preparation_variant") and food_doc.get("preparation_variant") != "normal":
            return [food_doc["preparation_variant"].title(), "Normal"]

        return ["Normal"]

    def _is_beverage(self, food_name: str, category: str = "") -> bool:
        """Detect if food is a drink or liquid beverage (chai, coffee, milk, chaas, juice, etc.)."""
        name = (food_name or "").lower().strip()
        cat = (category or "").lower().strip()
        beverages = {
            "chai", "tea", "green tea", "black tea", "herbal tea", "lemon tea",
            "coffee", "kofi", "latte", "cappuccino", "espresso", "milk", "doodh",
            "dudh", "chaas", "buttermilk", "lassi", "juice", "water", "soda",
            "cold drink", "cola", "pepsi", "coke", "shake", "smoothie", "soup"
        }
        if any(b in name for b in beverages) or "beverage" in cat or "drink" in cat or "tea" in cat:
            return True
        return False

    def _get_quantity_options(self, food_name: str, category: str = "", serving_size_g: float = 100, food_doc: Optional[dict] = None) -> list[str]:
        """
        Get food-specific quantity options dynamically from MongoDB food_doc schema.
        No hardcoded food-specific lists in Python code!
        """
        if food_doc and food_doc.get("quantity_options"):
            opts = food_doc["quantity_options"]
            fname = (food_doc.get("food_name") or food_name or "").lower()
            if any(w in fname for w in ["rotlo", "rotla"]):
                if any(bad in opts for bad in ["1 serving", "1 plate", "100g", "200g"]):
                    from import_to_mongodb import generate_food_schema_fields
                    sch = generate_food_schema_fields(fname, category or food_doc.get("category", ""), "", "rotlo")
                    if sch.get("quantity_options"):
                        return sch["quantity_options"]
            return opts

        fname = (food_name or (food_doc.get("food_name") if food_doc else "") or "").lower()
        if any(w in fname for w in ["rotlo", "rotla"]):
            from import_to_mongodb import generate_food_schema_fields
            sch = generate_food_schema_fields(fname, category, "", "rotlo")
            if sch.get("quantity_options"):
                return sch["quantity_options"]

        return ["1 serving", "100g", "200g", "0.5 serving"]

    def _build_food_pending(self, food: dict, nutrition, quantity: str, meal_type: str,
                             log_date_str: str, confidence: float) -> dict:
        return {
            "type": "log_food", "user_id": self.user["user_id"], "log_date": log_date_str,
            "meal_type": meal_type, "food_id": food["food_id"], "food_name": food["food_name"],
            "food_name_display": food["food_name_display"], "quantity_input": quantity,
            "quantity_amount": nutrition.quantity_amount, "quantity_unit": nutrition.quantity_unit,
            "quantity_grams": nutrition.quantity_grams,
            "variant": nutrition.variant if nutrition.variant != "normal" else None,
            "calories": nutrition.calories, "protein_g": nutrition.protein_g,
            "carbs_g": nutrition.carbs_g, "fat_g": nutrition.fat_g, "fiber_g": nutrition.fiber_g,
            "search_confidence": confidence,
            # internal only — used for DB source column, never shown to the user
            "source": food.get("source", "Local Database"),
        }

    def _resolve_date(self, date_str: Optional[str]) -> date:
        """Resolve date string to a date in IST timezone.
        Handles: 'today', 'yesterday', Gujarati 'kale'/'gay kale', Hindi 'kal',
        and ISO date strings (YYYY-MM-DD).
        """
        IST = timezone(timedelta(hours=5, minutes=30))
        today_ist = datetime.now(IST).date()

        if not date_str or date_str in ("today", "aaje", "aaj"):
            return today_ist
        if date_str in ("yesterday", "kale", "kal", "gay kale", "gay kal", "kal ki"):
            return today_ist - timedelta(days=1)
        try:
            return date.fromisoformat(date_str)
        except (ValueError, TypeError):
            return today_ist

    def _clean_food_query(self, query: str, original_message: str = "") -> str:
        """Clean food query string — remove filler/verb words, leading quantity +
        unit tokens, and Gujarati/Hindi genitive connectors. Preserves numbered
        dish names like Cheezy-7. The quantity itself is extracted separately from
        the raw item, so it is safe to strip units here for a clean food-search term."""
        raw = (original_message or query).lower().strip()

        # Check for known numbered dishes in raw message
        for brand_item in ["cheezy 7 pizza", "cheezy-7 pizza", "cheezy 7", "cheezy-7", "7 cheese pizza", "7up", "5 star"]:
            if brand_item in raw:
                return "cheezy-7 pizza" if "cheezy" in brand_item else brand_item

        text = (query or raw).lower().strip()

        # Normalise stray typo punctuation (e.g. "n=me" → "n me") and sentence endings (?, !, ., ,) so filler
        # tokens separate cleanly. Keep '-' and '/' — they appear in real food
        # names (cheezy-7, maki/roll).
        text = re.sub(r"[=]+", " ", text)
        text = re.sub(r"[?!.,]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Remove multi-word time PHRASES as a whole (e.g. "tea time") so their
        # individual tokens (tea/chai/time) survive when used as a real food.
        for _phrase in MEAL_TIME_PHRASES:
            text = re.sub(r"\b" + re.escape(_phrase) + r"\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()

        # Strip standalone quantity+unit tokens so they don't pollute the food
        # search (e.g. "1 bowl dal" → "dal", "150g paneer" → "paneer").
        # A quantity is a number optionally followed by a known unit.
        _UNIT = (r"g|gm|gram|grams|kg|ml|l|litre|liter|"
                 r"piece|pieces|pcs|bowl|bowls|katori|katoris|cup|cups|"
                 r"plate|plates|glass|glasses|serving|servings|slice|slices|"
                 r"tbsp|tsp|scoop|scoops|nos?")
        # number + unit  (e.g. "1 bowl", "150 g", "2 pieces")
        text = re.sub(rf"\b\d+\.?\d*\s*(?:{_UNIT})\b", " ", text, flags=re.IGNORECASE)
        # bare leading number (e.g. "2 bhakri" → "bhakri") — only strip standalone digits
        text = re.sub(r"\b\d+\.?\d*\b", " ", text)
        # any leftover standalone unit word
        text = re.sub(rf"\b(?:{_UNIT})\b", " ", text, flags=re.IGNORECASE)

        words = text.split()
        verbs_and_fillers = {
            # eating / drinking verbs
            "khadha", "khadhi", "khadhu", "khadho", "khaya", "khayi", "khaye",
            "khadhni", "khadhne", "khadhna", "khadhu", "khavu", "khava", "khai",
            "kha", "khaa", "khi",
            "lidha", "lidhi", "lidhu", "khao", "khiya", "jamya", "jamyu",
            "li", "liya", "liye", "liyu", "leli", "lidho",  # Hindi/Guj "took/had"
            "pidhi", "pidhu", "pidha", "pidho", "pido", "peeyo", "peena", "peeni",
            "peeno", "piyo", "piya", "piyi", "pi", "piya", "piyo",  # Hindi "pi/piya" = drank
            "drank", "drink", "drinking", "eat", "eating", "having", "took", "take",
            # question quantity words ("how much / how many")
            "ketlo", "ketla", "ketli", "ketlu", "kitna", "kitni", "kitne",
            # time / pronoun / article fillers (English + Romanised)
            "aaje", "aaj", "aje", "aj", "ajj", "me", "main", "ne", "ek",
            "ate", "eaten", "had", "n",
            "maine", "mene", "mein", "mne", "hu", "hoon",
            "kale", "kal", "gay kale", "gay kal",   # yesterday (date handled separately)
            "i", "we", "you", "my", "a", "an", "the", "of", "just", "some",
            "today", "yesterday", "was", "were", "am", "is", "have", "has",
            # meal-context words (they indicate WHEN, not WHAT — safe to drop from food query)
            "morning", "subah", "subhe", "savare", "savar",
            "afternoon", "dopahar", "bapore",
            "evening", "night", "raat", "sanje", "shaam",
            "breakfast", "lunch", "dinner", "snack", "nashta", "nasto",
            "ma", "mai",
            # connective / conjunction words that may survive a split
            "sathe", "saath", "pan", "pn", "bhi", "ane", "aur", "and", "with", "plus",
            # Gujarati/Hindi genitive connectors ("nariyal ni chattni" → "nariyal chattni")
            # These are safe to drop: they only join words and are never a food name alone.
            "ni", "nu", "no", "na",           # Gujarati genitive
            "ki", "ka", "ke",                  # Hindi genitive
            # Colloquial fillers and prepositions
            "ended", "up", "on", "way", "treated", "myself", "to", "whole", "guess", "what",
            "full", "zaap", "gayo", "sachu", "kau", "nice", "big", "toh", "so", "finally",
            "for", "brunch", "regular", "this", "that", "these", "those", "yaar",
            "workout", "gym", "training",
        }
        # Also strip every recognised time/meal expression word (bapor, bopore,
        # rondhe, pachhi, pela, evening, …) so a WHEN phrase never leaks into the
        # food name (e.g. "bapor pachhi samosa" → "samosa", not "bapor pachhi samosa").
        # This shares one vocabulary with _detect_meal_type so they stay in sync.
        drop_words = verbs_and_fillers | _ALL_MEAL_TIME_TOKENS
        cleaned = [w for w in words if w not in drop_words]
        cleaned_str = " ".join(cleaned).strip()
        return cleaned_str if cleaned_str else (query or "").strip()
