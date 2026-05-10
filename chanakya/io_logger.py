"""
io_logger.py — Centralised I/O logging for every channel.

Logs every inbound message/call and every outbound response so nothing
is ever invisible.  All helpers are fire-and-forget: they never raise.

Usage
-----
    from chanakya.io_logger import log_input, log_output, log_llm, log_api_call

Channels: TELEGRAM | CALL | SCHEDULER | API
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("chanakya.io")

# ── helpers ──────────────────────────────────────────────────────────────────

def _uid(user: dict | None) -> str:
    if not user:
        return "?"
    return str(user.get("telegram_id") or user.get("_id") or "?")


def _clip(text: Any, n: int = 300) -> str:
    s = str(text) if text is not None else ""
    return s[:n] + ("…" if len(s) > n else "")


# ── public API ────────────────────────────────────────────────────────────────

def log_input(
    channel: str,
    user_id: str | None,
    content: str,
    *,
    extra: dict | None = None,
) -> None:
    """Log any inbound message/event."""
    try:
        parts = [f"[IN] [{channel}]"]
        if user_id:
            parts.append(f"user={user_id}")
        parts.append(_clip(content))
        if extra:
            parts.append(str(extra))
        logger.info(" | ".join(parts))
    except Exception:  # noqa: BLE001
        pass


def log_output(
    channel: str,
    user_id: str | None,
    content: str,
    *,
    verdict: str | None = None,
    extra: dict | None = None,
) -> None:
    """Log any outbound response/message."""
    try:
        parts = [f"[OUT] [{channel}]"]
        if user_id:
            parts.append(f"user={user_id}")
        if verdict:
            parts.append(f"verdict={verdict}")
        parts.append(_clip(content))
        if extra:
            parts.append(str(extra))
        logger.info(" | ".join(parts))
    except Exception:  # noqa: BLE001
        pass


def log_llm(
    user_id: str | None,
    model: str,
    interaction_type: str,
    prompt_preview: str,
    response_preview: str,
    *,
    latency_ms: float | None = None,
    verdict: str | None = None,
    tokens: dict | None = None,
) -> None:
    """Log an LLM request/response pair."""
    try:
        parts = [f"[LLM] user={user_id or '?'}", f"model={model}", f"type={interaction_type}"]
        if latency_ms is not None:
            parts.append(f"latency={latency_ms:.0f}ms")
        if verdict:
            parts.append(f"verdict={verdict}")
        if tokens:
            parts.append(f"tokens={tokens}")
        logger.info(" | ".join(parts))
        logger.debug("[LLM PROMPT] %s", _clip(prompt_preview, 500))
        logger.debug("[LLM RESPONSE] %s", _clip(response_preview, 500))
    except Exception:  # noqa: BLE001
        pass


def log_api_call(
    service: str,
    method: str,
    endpoint: str,
    *,
    user_id: str | None = None,
    request_preview: str | None = None,
    response_preview: str | None = None,
    status_code: int | None = None,
    latency_ms: float | None = None,
    error: str | None = None,
) -> None:
    """Log an outbound API call (Twilio, ElevenLabs, OpenAI, etc.)."""
    try:
        parts = [f"[API] [{service}] {method} {endpoint}"]
        if user_id:
            parts.append(f"user={user_id}")
        if status_code is not None:
            parts.append(f"status={status_code}")
        if latency_ms is not None:
            parts.append(f"latency={latency_ms:.0f}ms")
        if error:
            parts.append(f"error={_clip(error, 200)}")
        logger.info(" | ".join(parts))
        if request_preview:
            logger.debug("[API REQ] %s", _clip(request_preview, 400))
        if response_preview:
            logger.debug("[API RESP] %s", _clip(response_preview, 400))
    except Exception:  # noqa: BLE001
        pass


class Timer:
    """Simple wall-clock timer for measuring latency."""

    def __init__(self) -> None:
        self._start = time.monotonic()

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000
