"""
Sarvam AI — Saaras v3 Speech-to-Text Client
============================================
Cloud-only. No local model. API key required (console.sarvam.ai).

Endpoint : POST https://api.sarvam.ai/speech-to-text
Model    : saaras:v3  (default, recommended)
Auth     : api-subscription-key header

Strategy for fitness chatbot
─────────────────────────────
We call the API **twice** per voice request:

  Call 1 — mode="transcribe"
    Transcript in the *original* language/script.
    Used as the human-readable display text.
    language_code returned tells us what language was spoken.

  Call 2 — mode="translit"   (only if language is Indic, not English)
    Roman-script transliteration of the same audio.
    Much easier for the downstream food-search pipeline which expects
    ASCII food names (bhakri, dal, idli …).
    We then run the existing normalise_transcript() pass on top.

If only one API call is allowed (to save quota) we use mode="translit"
and skip Call 1 — controlled by the `dual_call` parameter.

Languages supported (BCP-47 codes):
  gu-IN  hi-IN  mr-IN  bn-IN  ta-IN  te-IN  pa-IN  kn-IN
  ml-IN  od-IN  as-IN  ur-IN  en-IN  (+ unknown for auto-detect)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import httpx

# ── API constants ────────────────────────────────────────────────────────────

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL   = "saaras:v3"
SARVAM_TIMEOUT = 30.0   # seconds

# BCP-47 codes Sarvam understands
SARVAM_LANG_CODES = {
    "gu": "gu-IN",  "hi": "hi-IN",  "mr": "mr-IN",
    "bn": "bn-IN",  "ta": "ta-IN",  "te": "te-IN",
    "pa": "pa-IN",  "kn": "kn-IN",  "ml": "ml-IN",
    "od": "od-IN",  "as": "as-IN",  "ur": "ur-IN",
    "en": "en-IN",
}

# Inverse: Sarvam BCP-47 → our 2-letter code
_BCP47_TO_CODE = {v: k for k, v in SARVAM_LANG_CODES.items()}

# Content-type map for audio uploads
_CT_MAP = {
    "webm": "audio/webm", "ogg":  "audio/ogg",
    "mp4":  "audio/mp4",  "m4a":  "audio/mp4",
    "mp3":  "audio/mpeg", "wav":  "audio/wav",
    "flac": "audio/flac", "aac":  "audio/aac",
    "opus": "audio/ogg",
}


# ── Result ───────────────────────────────────────────────────────────────────

@dataclass
class SarvamSTTResult:
    """Single-call result from Saaras v3."""
    transcript:    str            # text returned by Sarvam
    language_code: str            # our 2-letter code (gu/hi/en/…)
    bcp47_code:    str            # raw BCP-47 from Sarvam (gu-IN/…)
    confidence:    float          # language_probability from Sarvam (0–1)
    mode:          str            # which mode was used
    request_id:    Optional[str]  # Sarvam request_id for debugging
    success:       bool
    error:         Optional[str]  = None


@dataclass
class SarvamFullResult:
    """
    Combined result after calling Sarvam (and optionally normalising).

    Attributes
    ──────────
    raw_text        Native-script transcript (mode=transcribe).
    translit_text   Roman-script transliteration (mode=translit).
                    Empty string if only one call was made.
    normalised_text translit_text after run through normalise_transcript().
                    Falls back to raw_text normalised if translit unavailable.
    language        2-letter code detected by Sarvam.
    bcp47           Full BCP-47 code.
    confidence      Language confidence 0–1.
    provider        Always "sarvam".
    success         False on hard error.
    error           Error message if success=False.
    """
    raw_text:        str
    translit_text:   str
    normalised_text: str
    language:        str
    bcp47:           str
    confidence:      float
    provider:        str = "sarvam"
    success:         bool = True
    error:           Optional[str] = None


# ── Client ───────────────────────────────────────────────────────────────────

class SarvamSTTService:
    """
    Sarvam Saaras v3 cloud STT.
    Raises ValueError at construction if key is missing.
    """

    def __init__(self, api_key: Optional[str] = None):
        from database import get_settings
        settings = get_settings()
        self.api_key = (api_key or settings.SARVAM_API_KEY or "").strip()
        if not self.api_key:
            raise ValueError(
                "SARVAM_API_KEY is not set. "
                "Get a free key at https://console.sarvam.ai and add it to backend/.env"
            )

    # ── low-level call ───────────────────────────────────────────────────────

    async def _call(
        self,
        audio_data: bytes,
        filename:   str,
        mode:       str,
        language_code: Optional[str] = None,  # BCP-47 or None → auto
    ) -> SarvamSTTResult:
        """Make a single Sarvam API call."""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "webm"
        content_type = _CT_MAP.get(ext, "audio/webm")

        form: dict = {
            "model": (None, SARVAM_MODEL),
            "mode":  (None, mode),
        }
        if language_code and language_code != "unknown":
            form["language_code"] = (None, language_code)
        # Always include file last
        form["file"] = (filename, audio_data, content_type)

        headers = {"api-subscription-key": self.api_key}

        try:
            async with httpx.AsyncClient(timeout=SARVAM_TIMEOUT) as client:
                resp = await client.post(
                    SARVAM_STT_URL,
                    headers=headers,
                    files=form,
                )

            if resp.status_code != 200:
                return SarvamSTTResult(
                    transcript="", language_code="en", bcp47_code="en-IN",
                    confidence=0.0, mode=mode, request_id=None,
                    success=False,
                    error=f"Sarvam HTTP {resp.status_code}: {resp.text[:300]}",
                )

            body = resp.json()
            transcript  = (body.get("transcript") or "").strip()
            bcp47       = body.get("language_code") or "unknown"
            lang2       = _BCP47_TO_CODE.get(bcp47, bcp47.split("-")[0].lower())
            prob        = float(body.get("language_probability") or 0.85)
            request_id  = body.get("request_id")

            return SarvamSTTResult(
                transcript=transcript, language_code=lang2, bcp47_code=bcp47,
                confidence=prob, mode=mode, request_id=request_id,
                success=True,
            )

        except httpx.TimeoutException:
            return SarvamSTTResult(
                transcript="", language_code="en", bcp47_code="en-IN",
                confidence=0.0, mode=mode, request_id=None,
                success=False, error=f"Sarvam timeout after {SARVAM_TIMEOUT}s",
            )
        except Exception as exc:
            return SarvamSTTResult(
                transcript="", language_code="en", bcp47_code="en-IN",
                confidence=0.0, mode=mode, request_id=None,
                success=False, error=f"Sarvam error: {str(exc)[:200]}",
            )

    # ── public API ───────────────────────────────────────────────────────────

    async def transcribe(
        self,
        audio_data:    bytes,
        filename:      str  = "audio.webm",
        language:      Optional[str] = None,   # our 2-letter code or None
        dual_call:     bool = True,             # True = transcribe + translit
    ) -> SarvamFullResult:
        """
        Full transcription pipeline.

        dual_call=True (default):
          Call 1 — mode=transcribe   → native-script display text
          Call 2 — mode=translit     → Roman text for food search
          (skips call 2 for English since it's already Roman)

        dual_call=False:
          Single call with mode=translit (saves API quota).
        """
        if not audio_data:
            return SarvamFullResult(
                raw_text="", translit_text="", normalised_text="",
                language="en", bcp47="en-IN", confidence=0.0,
                success=False, error="No audio data provided",
            )

        # Resolve BCP-47 for the requested language
        bcp47_hint: Optional[str] = None
        if language:
            bcp47_hint = SARVAM_LANG_CODES.get(language)  # may be None → auto

        from stt_service import normalise_transcript  # reuse existing normaliser

        if not dual_call:
            # ── Single translit call ──────────────────────────────────────
            r = await self._call(audio_data, filename, "translit", bcp47_hint)
            if not r.success:
                return SarvamFullResult(
                    raw_text="", translit_text="", normalised_text="",
                    language="en", bcp47="en-IN", confidence=0.0,
                    success=False, error=r.error,
                )
            normalised = normalise_transcript(r.transcript)
            return SarvamFullResult(
                raw_text=r.transcript,
                translit_text=r.transcript,
                normalised_text=normalised,
                language=r.language_code,
                bcp47=r.bcp47_code,
                confidence=r.confidence,
            )

        # ── Dual call (transcribe + translit) ─────────────────────────────
        # Call 1: native-script transcription
        r_native = await self._call(audio_data, filename, "transcribe", bcp47_hint)
        if not r_native.success:
            return SarvamFullResult(
                raw_text="", translit_text="", normalised_text="",
                language="en", bcp47="en-IN", confidence=0.0,
                success=False, error=r_native.error,
            )

        detected_lang = r_native.language_code
        detected_bcp47 = r_native.bcp47_code

        # Call 2: translit (Roman) — skip for English, it's already Roman
        translit_text = ""
        if detected_lang == "en":
            translit_text = r_native.transcript   # already Roman
            normalised    = normalise_transcript(r_native.transcript)
        else:
            # Use the detected language for the translit call (more accurate)
            bcp47_for_translit = detected_bcp47
            r_translit = await self._call(
                audio_data, filename, "translit", bcp47_for_translit
            )
            if r_translit.success and r_translit.transcript:
                translit_text = r_translit.transcript
                normalised    = normalise_transcript(translit_text)
            else:
                # Translit failed — fall back to normalising the native text
                translit_text = ""
                normalised    = normalise_transcript(r_native.transcript)

        return SarvamFullResult(
            raw_text=r_native.transcript,
            translit_text=translit_text,
            normalised_text=normalised,
            language=detected_lang,
            bcp47=detected_bcp47,
            confidence=r_native.confidence,
        )

    async def transcribe_with_mode(
        self,
        audio_data: bytes,
        filename:   str,
        mode:       str,              # transcribe | translit | translate | codemix
        language:   Optional[str] = None,
    ) -> SarvamSTTResult:
        """Low-level single call — useful for testing individual modes."""
        bcp47 = SARVAM_LANG_CODES.get(language or "", None)
        return await self._call(audio_data, filename, mode, bcp47)
