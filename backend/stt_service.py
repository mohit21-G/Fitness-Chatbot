"""
Speech-to-Text Service — Groq Whisper Large V3
Supports Gujarati, Hindi, English and all Indian languages.

Pipeline:
  1. Send audio to Groq Whisper with Indian vocabulary prompt (no forced language)
  2. Receive raw transcript + Whisper's detected language
  3. Run second-stage language validator to correct wrong detections
     - Gujarati mistaken as English/Korean/Japanese
     - Hindi mistaken as Urdu
  4. Run normaliser: Gujarati/Hindi food terms → canonical English search terms
  5. Return: raw_text, normalised text, corrected language, confidence

Key design:
  - Never force a language code to Whisper (breaks multilingual detection)
  - Vocabulary prompt biases Whisper toward correct Indian food spellings
  - Second-stage correction ensures we never serve Urdu/Korean output for Indian speech
"""
from __future__ import annotations

import re
import httpx
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
WHISPER_MODEL = "whisper-large-v3"

# Languages Whisper should NOT produce for Indian speech
# If Whisper returns one of these for speech that looks Indian, we correct it.
WRONG_LANG_FOR_INDIA = {"ur", "ko", "ja", "zh", "ar", "fa", "tr", "ms", "tl"}


# ---------------------------------------------------------------------------
# Vocabulary prompt for Whisper
# ---------------------------------------------------------------------------

INDIAN_FOOD_PROMPT = (
    "Fitness and food tracking in Indian languages. "
    "Common foods: "
    "bhakri, bhakhri, rotli, roti, thepla, dhokla, fafda, jalebi, "
    "gathiya, khakhra, patra, undhiyu, shrikhand, basundi, "
    "kadhi, dal, daal, chaas, chhaas, lassi, "
    "pani puri, panipuri, sev puri, dahi puri, vada pav, "
    "poha, upma, idli, dosa, uttapam, sambar, rasam, "
    "biryani, pulao, khichdi, rajma, chole, dal makhani, "
    "sabji, shaak, bhaji, palak, methi, dudhi, ringna, bataka, "
    "ghee, tel, butter, makhan, dahi, yogurt, paneer, "
    "rice, chawal, paratha, naan, chapati. "
    "Quantities: katori, bowl, glass, plate, piece, grams. "
    "Exercise: yoga, jogging, running, walking, cycling, gym, workout, squats. "
    "Meals: nashta, savare, bapore, sanje. "
    "Actions: khadhu, lidhu, khayi, khaya, ate, had."
)


# ---------------------------------------------------------------------------
# Second-stage language detector / corrector
# ---------------------------------------------------------------------------

# Unicode script ranges for Indian languages
_GU_SCRIPT = ("\u0A80", "\u0AFF")   # Gujarati
_HI_SCRIPT  = ("\u0900", "\u097F")   # Devanagari (Hindi, Marathi)
_BN_SCRIPT  = ("\u0980", "\u09FF")   # Bengali
_TA_SCRIPT  = ("\u0B80", "\u0BFF")   # Tamil
_TE_SCRIPT  = ("\u0C00", "\u0C7F")   # Telugu
_KN_SCRIPT  = ("\u0C80", "\u0CFF")   # Kannada
_ML_SCRIPT  = ("\u0D00", "\u0D7F")   # Malayalam
_PA_SCRIPT  = ("\u0A00", "\u0A7F")   # Gurmukhi (Punjabi)

# Romanised Indian keyword sets (used when text is in Roman script)
_GU_ROMAN_KW = {
    "bhakri", "bhakhri", "rotli", "thepla", "dhokla", "gathiya", "fafda",
    "undhiyu", "shrikhand", "basundi", "chhaas", "chaas", "khadhu", "lidhu",
    "lidha", "khadhi", "khadha", "tamara", "tame", "aaje", "savare", "bapore",
    "sanje", "su", "kem", "cho", "ketlu", "ketli", "khadhu", "lidhu",
    "shaak", "vati", "dudhi", "ringna", "bataka", "tindola",
}
_HI_ROMAN_KW = {
    "khayi", "khaya", "khaye", "khaayi", "maine", "mene", "aaj", "kal", "subah",
    "dopahar", "raat", "nashta", "kya", "kitna", "kitni", "dikhao", "batao",
    "chawal", "makhan", "daal", "rajmah", "chhole", "poha", "paratha",
    # South Indian food names commonly used in Hindi sentences
    "idli", "dosa", "sambar", "upma", "uttapam",
    # Common Hindi fitness verbs
    "kiya", "kiye", "karenge", "khaayenge",
}


def _contains_script(text: str, lo: str, hi: str) -> bool:
    return any(lo <= c <= hi for c in text)


def _count_roman_kw(text: str, kwset: set) -> int:
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return len(words & kwset)


def detect_indian_language(raw_text: str, whisper_lang: Optional[str]) -> tuple[str, float]:
    """
    Second-stage language detection / correction.

    Returns (corrected_lang_code, confidence_0_to_1).

    Logic:
      1. Check for native Indian scripts in the raw text (high confidence).
      2. If Roman script, count Gujarati vs Hindi keyword hits.
      3. If Whisper returned a non-Indian language for clearly Indian text,
         correct it.
      4. Trust Whisper for English.
    """
    text = raw_text.strip()
    if not text:
        return (whisper_lang or "en"), 0.5

    # ── 1. Native script detection (definitive) ────────────────────────
    if _contains_script(text, *_GU_SCRIPT):
        return "gu", 0.99
    if _contains_script(text, *_HI_SCRIPT):
        return "hi", 0.99
    if _contains_script(text, *_BN_SCRIPT):
        return "bn", 0.99
    if _contains_script(text, *_TA_SCRIPT):
        return "ta", 0.99
    if _contains_script(text, *_TE_SCRIPT):
        return "te", 0.99
    if _contains_script(text, *_KN_SCRIPT):
        return "kn", 0.99
    if _contains_script(text, *_ML_SCRIPT):
        return "ml", 0.99
    if _contains_script(text, *_PA_SCRIPT):
        return "pa", 0.99

    # ── 2. Roman-script Indian keyword counting ────────────────────────
    gu_hits = _count_roman_kw(text, _GU_ROMAN_KW)
    hi_hits = _count_roman_kw(text, _HI_ROMAN_KW)

    # Get whisper lang for downstream checks
    wl = (whisper_lang or "").lower()

    if gu_hits >= 2:
        confidence = min(0.5 + gu_hits * 0.1, 0.95)
        return "gu", confidence
    if hi_hits >= 2:
        confidence = min(0.5 + hi_hits * 0.1, 0.95)
        return "hi", confidence
    # Single keyword hit — only override if Whisper itself was wrong
    if gu_hits == 1 and wl in WRONG_LANG_FOR_INDIA:
        return "gu", 0.55
    if hi_hits == 1 and wl in WRONG_LANG_FOR_INDIA:
        return "hi", 0.55

    # ── 3. Correct known wrong Whisper detections ──────────────────────
    # (wl already assigned above)

    # Whisper returned a clearly wrong language for Indian speech
    # (e.g. Gujarati detected as Korean, Urdu, Japanese)
    if wl in WRONG_LANG_FOR_INDIA:
        # The text is Roman — could be Romanised Hindi/Gujarati
        # Default to Hindi (most common Indian language after English)
        return "hi", 0.6

    # Urdu is sometimes returned for Hindi — correct if text has no Arabic chars
    if wl == "ur":
        has_arabic = any("\u0600" <= c <= "\u06FF" for c in text)
        if not has_arabic:
            return "hi", 0.8  # Likely Hindi misidentified as Urdu

    # ── 4. Trust Whisper for English and known Indian codes ────────────
    trusted_indian = {"hi", "gu", "mr", "bn", "ta", "te", "kn", "ml", "pa",
                      "or", "as", "ne", "sa"}
    if wl in trusted_indian:
        return wl, 0.85
    if wl == "en":
        return "en", 0.95

    # Fallback: return Whisper's value with low confidence
    return (wl or "en"), 0.5


# ---------------------------------------------------------------------------
# Normaliser — Whisper output → canonical food search terms
# ---------------------------------------------------------------------------

NORMALISE_MAP = {
    # Gujarati → search-friendly
    "bhakhri": "bhakri",
    "rotli": "roti",
    "batata": "aloo",
    "bataka": "aloo",
    "ringna": "brinjal",
    "dudhi": "bottle gourd",
    "chaas": "buttermilk",
    "chhaas": "buttermilk",
    "dahi": "curd",
    "shaak": "sabji",
    "sabzi": "sabji",
    "panipuri": "pani puri",
    "gathia": "gathiya",
    "khakra": "khakhra",
    "dhokala": "dhokla",
    "thepala": "thepla",
    # Hindi variants
    "chawal": "rice",
    "makhan": "butter",
    "daal": "dal",
    "rajmah": "rajma",
    "chhole": "chole",
    # Whisper spelling variants
    "idlee": "idli",
    "dossai": "dosa",
    "dosai": "dosa",
    "biriyani": "biryani",
    "briyani": "biryani",
    # Devanagari common output when Whisper picks Hindi
    "भाखरी": "bhakri",
    "रोटली": "roti",
    "दाल": "dal",
    "चावल": "rice",
    "इडली": "idli",
    "डोसा": "dosa",
    "पोहा": "poha",
    "खिचड़ी": "khichdi",
    # Gujarati script output
    "ભાખરી": "bhakri",
    "રોટલી": "roti",
    "ખીચડી": "khichdi",
    "ઇડલી": "idli",
    "ભાત": "rice",
}

# ---------------------------------------------------------------------------
# Indian number words → digits
# Used by normalise_transcript() so any downstream parser sees "15" not "pandrah".
# Covers: Hindi (Romanised), Gujarati (Romanised), common Whisper/Sarvam output
# ---------------------------------------------------------------------------

_INDIAN_NUMBERS: dict[str, int] = {
    # ── Shared (same spelling in both) ───────────────────────────────────────
    # NOTE: "do" removed — collides with English "do" (→ "2" breaks "what exercise did i do")
    # NOTE: "be" removed — could collide with English "be"
    "ek":      1,   "char":    4,
    "paanch":  5,   "panch":   5,   "saat":    7,   "aath":    8,
    "das":    10,   "bees":   20,   "tees":   30,   "pachaas": 50,
    "pachpan":55,   "saath":  60,   "sattar": 70,   "assi":   80,   "sau": 100,
    # ── Hindi (Romanised) ────────────────────────────────────────────────────
    "teen":    3,   "nau":     9,
    "gyarah": 11,   "barah":  12,   "terah":  13,   "chaudah":14,
    "pandrah":15,   "pandraha":15,  "pandara":15,
    "solah":  16,   "satrah": 17,   "atharah":18,   "unnis":  19,
    "ikkees": 21,   "baaees": 22,   "teis":   23,   "chaubees":24,
    "pachees":25,   "pachis": 25,
    "untees": 29,   "chaalees":40,  "pachaas":50,
    "pachhattar":75,"nabbe":  90,   "pachaanbe":95,
    # ── Gujarati (Romanised) ─────────────────────────────────────────────────
    # NOTE: "be" (2) and "bar" (12) removed — collide with common English words
    "tran":    3,   "nav":     9,
    "agiyar": 11,   "ter":    13,   "chaud":  14,
    "pandar": 15,   "sol":    16,   "ognis":  19,   "vis":    20,
    "ekvis":  21,   "tevis":  23,   "chovis": 24,   "pachvis":25,
    "trish":  30,   "chaalis":40,   "pachas": 50,   "so":    100,
}

# Devanagari/Gujarati → Roman transliteration for full phrases
# Applied to words that survive the map above
_SCRIPT_TO_ROMAN = {
    # ── Gujarati script ──────────────────────────────────────────────────────
    "ભાત": "rice",      "ચા": "chai",      "ખીચડી": "khichdi",
    "ઘી": "ghee",       "દૂધ": "milk",     "દહીં": "curd",
    "નાસ્તો": "nashta", "ભોજન": "food",    "કસરત": "exercise",
    "ચાલવું": "walking","દોડવું": "running",
    # Gujarati exercise / fitness words
    "પુશ-અપ્સ": "pushups",   "પુશ-અપ": "pushup",
    "પુશ અપ": "pushup",      "સ્ક્વૉટ્સ": "squats",
    "સ્ક્વૉટ": "squat",      "ક્રન્ચ": "crunch",
    "ક્રન્ચ": "crunches",    "સ્ટ્રેચ": "stretching",
    "સ્ટ્રેચિંગ": "stretching",
    "સ્વિમિંગ": "swimming",  "સ્વિમ": "swim",
    "સાઇકલ": "cycling",      "સાઇકલ ચલાવ": "cycling",
    "ઝૂમ્બા": "zumba",       "જિમ": "gym",
    "યોગ": "yoga",           "આસન": "yoga",
    "સૂર્ય નમસ્કાર": "surya namaskar",
    "ટ્રેડમિલ": "treadmill",
    "બૉક્સિંગ": "boxing",
    "ઉઠક-બેઠક": "squats",
    # ── Devanagari script (Hindi/Marathi) ────────────────────────────────────
    "भाखरी": "bhakri",  "रोटली": "roti",   "इडली": "idli",
    "भात": "rice",      "चाय": "chai",     "खिचड़ी": "khichdi",
    "घी": "ghee",       "दूध": "milk",     "दही": "curd",
    "नाश्ता": "nashta", "भोजन": "food",    "कसरत": "exercise",
    "चलना": "walking",  "दौड़ना": "running","योग": "yoga",
    "आज": "today",      "कल": "yesterday",
    # Hindi food words
    "दाल": "dal",       "चावल": "rice",    "रोटी": "roti",
    "पनीर": "paneer",   "सब्जी": "sabji",  "पोहा": "poha",
    "उपमा": "upma",     "इडली": "idli",    "डोसा": "dosa",
    "बिरयानी": "biryani","राजमा": "rajma", "छोले": "chole",
    "परांठा": "paratha","मक्खन": "butter", "लस्सी": "lassi",
    # Hindi exercise / fitness words
    "पुश-अप्स": "pushups",  "पुश-अप": "pushup",
    "पुश अप्स": "pushups",  "पुश अप": "pushup",
    "स्क्वॉट्स": "squats",  "स्क्वॉट": "squat",
    "क्रंच": "crunches",    "क्रंचेज": "crunches",
    "दौड़": "running",      "दौड़ना": "running",
    "चाल": "walking",       "चलना": "walking",
    "तैरना": "swimming",    "तैराकी": "swimming",
    "साइकिल": "cycling",    "साइकिल चलाना": "cycling",
    "जिम": "gym",            "जिम जाना": "gym",
    "ज़ुम्बा": "zumba",      "ज़ुम्बा": "zumba",
    "बॉक्सिंग": "boxing",
    "सूर्य नमस्कार": "surya namaskar",
    "ट्रेडमिल": "treadmill",
    "उठक-बैठक": "squats",
    "व्यायाम": "exercise",  "एक्सरसाइज": "exercise",
    "वर्कआउट": "workout",
    # ── Punjabi (Gurmukhi script) ────────────────────────────────────────────
    "ਕਸਰਤ": "exercise",   "ਕਸਰਤਾਂ": "exercises",
    "ਦੌੜਨਾ": "running",   "ਤੁਰਨਾ": "walking",
    "ਤੈਰਾਕੀ": "swimming", "ਸਾਈਕਲਿੰਗ": "cycling",
    "ਯੋਗਾ": "yoga",       "ਜਿੰਮ": "gym",
    "ਪੁਸ਼-ਅੱਪ": "pushup", "ਪੁਸ਼-ਅੱਪਸ": "pushups",
    "ਪੁਸ਼ ਅੱਪ": "pushup", "ਪੁਸ਼ ਅੱਪਸ": "pushups",
    "ਸਕੁਆਟ": "squat",     "ਸਕੁਆਟਸ": "squats",
    "ਕਰੰਚ": "crunch",     "ਕਰੰਚੇਜ਼": "crunches",
    "ਜ਼ੁੰਬਾ": "zumba",    "ਬਾਕਸਿੰਗ": "boxing",
    "ਵਰਕਆਊਟ": "workout",  "ਦੌੜ": "running",
    "ਭਾਗਣਾ": "running",   "ਟਹਿਲਣਾ": "walking",
    # Punjabi number words (Gurmukhi digits also map via _INDIAN_NUMBERS)
    "ਇੱਕ": "1",  "ਦੋ": "2",  "ਤਿੰਨ": "3",  "ਚਾਰ": "4",
    "ਪੰਜ": "5",  "ਛੇ": "6",  "ਸੱਤ": "7",  "ਅੱਠ": "8",
    "ਨੌਂ": "9",  "ਦਸ": "10", "ਪੰਦਰਾਂ": "15","ਵੀਹ": "20",
    "ਤੀਹ": "30", "ਚਾਲੀ": "40","ਪੰਜਾਹ": "50",
    # Punjabi food
    "ਰੋਟੀ": "roti",   "ਦਾਲ": "dal",   "ਚਾਵਲ": "rice",
    "ਪਨੀਰ": "paneer", "ਸਾਗ": "saag",  "ਲੱਸੀ": "lassi",
    "ਦਹੀ": "curd",    "ਘਿਓ": "ghee",  "ਦੁੱਧ": "milk",
    # ── Bengali script (basic exercise) ─────────────────────────────────────
    "ব্যায়াম": "exercise", "দৌড়ানো": "running",
    "হাঁটা": "walking",    "সাঁতার": "swimming",
    "যোগব্যায়াম": "yoga",
}


def normalise_transcript(text: str) -> str:
    """
    Normalise Whisper/Sarvam output → canonical terms for food search and exercise parsing.
    1. Replace native-script words with Roman equivalents.
    2. Convert Indian number words (ek/do/teen/… pandrah/bees/…) to digits.
    3. Apply Roman-to-canonical food/exercise map.
    """
    result = text

    # Step 1: Replace native-script words with Roman.
    # Sort longest-first so "ચાλvun" (walking) replaces before "ચા" (chai).
    for script_word, roman in sorted(_SCRIPT_TO_ROMAN.items(), key=lambda x: -len(x[0])):
        result = result.replace(script_word, roman)

    # Step 2: Convert Indian number words → digits (word-boundary, case-insensitive)
    # This ensures "pandrah pushups" → "15 pushups" before the exercise parser sees it.
    for word, digit in _INDIAN_NUMBERS.items():
        result = re.sub(
            r"\b" + re.escape(word) + r"\b",
            str(digit),
            result,
            flags=re.IGNORECASE,
        )

    # Step 3: Word-boundary Roman replacements (food/exercise canonical names)
    for wrong, right in NORMALISE_MAP.items():
        if any(ord(c) > 127 for c in wrong):
            # Already handled above; skip
            continue
        result = re.sub(
            r"\b" + re.escape(wrong) + r"\b",
            right,
            result,
            flags=re.IGNORECASE,
        )

    return result.strip()


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class STTResult:
    text: str                      # normalised — used for chatbot processing
    raw_text: str                  # verbatim output — shown to user
    language: str                  # corrected language code
    confidence: float              # language detection confidence 0-1
    duration_seconds: Optional[float]
    success: bool
    error: Optional[str] = None
    provider: str = "whisper"      # "whisper" | "sarvam"
    translit_text: str = ""        # Roman transliteration (Sarvam only)


# ---------------------------------------------------------------------------
# STT Service
# ---------------------------------------------------------------------------

class STTService:
    """
    Unified Speech-to-Text service.

    Provider selection (via `provider` param or SARVAM_API_KEY presence):

      "auto"    — use Sarvam if SARVAM_API_KEY is set, otherwise Whisper.
      "sarvam"  — always use Sarvam Saaras v3; raise if key missing.
      "whisper" — always use Groq Whisper Large V3.

    Whisper is always available as a fallback when Sarvam is unreachable.
    """

    def __init__(
        self,
        api_key:  Optional[str] = None,   # Groq key (Whisper)
        provider: str = "auto",            # "auto" | "sarvam" | "whisper"
    ):
        from database import get_settings
        settings = get_settings()

        self.groq_api_key = (api_key or settings.GROQ_API_KEY or "").strip()
        self.sarvam_api_key = (settings.SARVAM_API_KEY or "").strip()
        self.provider = provider.lower().strip()

        if self.provider == "whisper" and not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not set. Set it in backend/.env")
        if self.provider == "sarvam" and not self.sarvam_api_key:
            raise ValueError(
                "SARVAM_API_KEY not set. "
                "Get a free key at https://console.sarvam.ai"
            )
        if self.provider == "auto" and not self.groq_api_key and not self.sarvam_api_key:
            raise ValueError(
                "No STT API key found. Set GROQ_API_KEY or SARVAM_API_KEY in backend/.env"
            )

    def _effective_provider(self) -> str:
        """Resolve 'auto' to the actual provider to use."""
        if self.provider == "auto":
            return "sarvam" if self.sarvam_api_key else "whisper"
        return self.provider

    async def transcribe(
        self,
        audio_data: bytes,
        filename:   str = "audio.webm",
        language:   Optional[str] = None,
    ) -> STTResult:
        """
        Transcribe audio. Provider is resolved at runtime.

        language: 2-letter code (en/hi/gu/…) or None for auto-detect.
        Falls back to Whisper automatically if Sarvam fails.
        """
        if not audio_data:
            return STTResult(
                text="", raw_text="", language="en", confidence=0.0,
                duration_seconds=None, success=False,
                error="No audio data provided", provider=self._effective_provider(),
            )

        effective = self._effective_provider()

        if effective == "sarvam":
            result = await self._transcribe_sarvam(audio_data, filename, language)
            # Auto-fallback to Whisper if Sarvam hard-fails
            if not result.success and self.groq_api_key:
                fallback = await self._transcribe_whisper(audio_data, filename, language)
                fallback.error = (
                    f"[Sarvam failed: {result.error}] "
                    f"[Whisper fallback] {fallback.error or ''}"
                ).strip()
                return fallback
            return result

        # Whisper
        return await self._transcribe_whisper(audio_data, filename, language)

    # ─────────────────────────────────────────────────────────────────────────
    # Sarvam backend
    # ─────────────────────────────────────────────────────────────────────────

    async def _transcribe_sarvam(
        self,
        audio_data: bytes,
        filename:   str,
        language:   Optional[str],
    ) -> STTResult:
        """Call Sarvam Saaras v3 and convert to STTResult."""
        try:
            from sarvam_stt import SarvamSTTService
            svc = SarvamSTTService(api_key=self.sarvam_api_key)
        except ValueError as exc:
            return STTResult(
                text="", raw_text="", language="en", confidence=0.0,
                duration_seconds=None, success=False,
                error=str(exc), provider="sarvam",
            )

        r = await svc.transcribe(
            audio_data, filename=filename, language=language, dual_call=True
        )

        if not r.success:
            return STTResult(
                text="", raw_text="", language="en", confidence=0.0,
                duration_seconds=None, success=False,
                error=r.error, provider="sarvam",
            )

        if not r.raw_text:
            return STTResult(
                text="", raw_text="", language=r.language, confidence=0.0,
                duration_seconds=None, success=False,
                error="No speech detected", provider="sarvam",
            )

        return STTResult(
            text=r.normalised_text or r.translit_text or r.raw_text,
            raw_text=r.raw_text,
            language=r.language,
            confidence=r.confidence,
            duration_seconds=None,   # Sarvam REST doesn't return duration
            success=True,
            provider="sarvam",
            translit_text=r.translit_text,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Whisper backend  (original logic preserved intact)
    # ─────────────────────────────────────────────────────────────────────────

    async def _transcribe_whisper(
        self,
        audio_data: bytes,
        filename:   str,
        language:   Optional[str],
    ) -> STTResult:
        """Groq Whisper Large V3 with Indian language correction."""
        if not self.groq_api_key:
            return STTResult(
                text="", raw_text="", language="en", confidence=0.0,
                duration_seconds=None, success=False,
                error="GROQ_API_KEY not set", provider="whisper",
            )

        ct_map = {
            "webm": "audio/webm", "ogg": "audio/ogg",
            "mp4": "audio/mp4",   "mp3": "audio/mpeg",
            "wav": "audio/wav",   "m4a": "audio/mp4",
        }
        ext   = filename.rsplit(".", 1)[-1].lower() if "." in filename else "webm"
        files = {"file": (filename, audio_data, ct_map.get(ext, "audio/webm"))}

        data: dict = {
            "model": WHISPER_MODEL,
            "response_format": "verbose_json",
            "prompt": INDIAN_FOOD_PROMPT,
        }
        if language and language in {
            "en", "hi", "gu", "mr", "bn", "ta", "te", "kn", "ml", "pa"
        }:
            data["language"] = language

        headers = {"Authorization": f"Bearer {self.groq_api_key}"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    GROQ_STT_URL, headers=headers, files=files, data=data
                )

            if response.status_code != 200:
                return STTResult(
                    text="", raw_text="", language="en", confidence=0.0,
                    duration_seconds=None, success=False,
                    error=f"Groq {response.status_code}: {response.text[:300]}",
                    provider="whisper",
                )

            result      = response.json()
            raw_text    = result.get("text", "").strip()
            whisper_lang= result.get("language") or language
            duration    = result.get("duration")

            if not raw_text:
                return STTResult(
                    text="", raw_text="", language=whisper_lang or "en",
                    confidence=0.0, duration_seconds=duration,
                    success=False, error="No speech detected", provider="whisper",
                )

            corrected_lang, lang_conf = detect_indian_language(raw_text, whisper_lang)
            normalised = normalise_transcript(raw_text)

            return STTResult(
                text=normalised, raw_text=raw_text,
                language=corrected_lang, confidence=lang_conf,
                duration_seconds=duration, success=True, provider="whisper",
            )

        except httpx.TimeoutException:
            return STTResult(
                text="", raw_text="", language="en", confidence=0.0,
                duration_seconds=None, success=False,
                error="Whisper timeout (30s)", provider="whisper",
            )
        except Exception as exc:
            return STTResult(
                text="", raw_text="", language="en", confidence=0.0,
                duration_seconds=None, success=False,
                error=f"Whisper error: {str(exc)[:200]}", provider="whisper",
            )
