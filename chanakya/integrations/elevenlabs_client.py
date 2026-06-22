"""
elevenlabs_client.py — ElevenLabs TTS integration.

Provides:
  synthesise(text, voice_id) -> bytes
    - Uses model_id="eleven_monolingual_v1"
    - Caches results by sha256(text + voice_id) with 24-hour TTL
    - Raises ElevenLabsSynthesisError on API failure
    - Checks credit balance after each synthesis; alerts user if < 1000 chars remain
"""

import hashlib
import logging
import time

import httpx

logger = logging.getLogger(__name__)


class ElevenLabsSynthesisError(Exception):
    """Raised when ElevenLabs TTS synthesis fails."""


# ---------------------------------------------------------------------------
# Module-level audio cache: {cache_key: (audio_bytes, timestamp)}
# ---------------------------------------------------------------------------

_audio_cache: dict[str, tuple[bytes, float]] = {}
_CACHE_TTL_SECONDS = 86400  # 24 hours

# Circuit breaker — disabled after a payment/auth failure until this timestamp
_disabled_until: float = 0.0
_DISABLE_DURATION_SECONDS = 600  # 10 minutes


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_key(text: str, voice_id: str) -> str:
    """Compute a SHA-256 cache key for the given text and voice_id."""
    return hashlib.sha256(f"{text}{voice_id}".encode()).hexdigest()


def _get_cached(text: str, voice_id: str) -> bytes | None:
    """Return cached audio bytes if present and not expired, else None."""
    key = _cache_key(text, voice_id)
    if key in _audio_cache:
        audio_bytes, ts = _audio_cache[key]
        if time.time() - ts < _CACHE_TTL_SECONDS:
            return audio_bytes
        del _audio_cache[key]
    return None


def _set_cache(text: str, voice_id: str, audio_bytes: bytes) -> None:
    """Store audio bytes in the cache with the current timestamp."""
    key = _cache_key(text, voice_id)
    _audio_cache[key] = (audio_bytes, time.time())


# ---------------------------------------------------------------------------
# ElevenLabs client
# ---------------------------------------------------------------------------


class ElevenLabsClient:
    """Thin wrapper around the ElevenLabs TTS REST API."""

    BASE_URL = "https://api.elevenlabs.io/v1"
    MODEL_ID = "eleven_turbo_v2_5"

    def synthesise(self, text: str, voice_id: str) -> bytes:
        """Synthesise text to audio bytes using the ElevenLabs TTS API.

        Checks the in-memory cache first (SHA-256 key, 24-hour TTL).

        Args:
            text: The text to synthesise.
            voice_id: The ElevenLabs voice ID to use.

        Returns:
            Raw audio bytes (MP3).

        Raises:
            ElevenLabsSynthesisError: On any API or network failure.
        """
        from chanakya.config import ELEVENLABS_API_KEY

        # Check cache first
        cached = _get_cached(text, voice_id)
        if cached is not None:
            return cached

        url = f"{self.BASE_URL}/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": self.MODEL_ID,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }

        # Circuit breaker — skip immediately if recently disabled by a 401/402
        if time.time() < _disabled_until:
            raise ElevenLabsSynthesisError(
                "ElevenLabs disabled (payment issue) — circuit breaker active"
            )

        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
            response.raise_for_status()
            audio_bytes = response.content
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 402, 403):
                # Payment / auth failure — disable for 10 minutes to stop spam
                global _disabled_until
                _disabled_until = time.time() + _DISABLE_DURATION_SECONDS
                logger.warning(
                    "ElevenLabs %d (payment/auth) — disabling for %ds to prevent retry spam",
                    status, _DISABLE_DURATION_SECONDS,
                )
            raise ElevenLabsSynthesisError(
                f"ElevenLabs API error {status}: {exc.response.text[:200]}"
            ) from exc
        except Exception as exc:
            raise ElevenLabsSynthesisError(
                f"ElevenLabs request failed: {exc}"
            ) from exc

        _set_cache(text, voice_id, audio_bytes)

        # Credit balance check removed — API key doesn't have /v1/user permission
        # Monitor usage via ElevenLabs dashboard instead

        return audio_bytes
