"""
llm_provider.py — Reusable LLM provider functions.

Three independent callers (OpenRouter, Gemini native, OpenAI) + a fallback chain:
  1. Gemini 2.5 Flash free via OpenRouter
  2. Gemini 2.5 Flash via native Gemini API
  3. gpt-4o-mini via OpenAI

Any caller can also be used standalone with explicit model/api_key params.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Status codes that mean "rate limited / quota" — trigger next provider
_RATE_LIMIT_STATUSES = {429, 402, 503, 529}


# ---------------------------------------------------------------------------
# Per-provider callers
# ---------------------------------------------------------------------------

async def call_openrouter(
    messages: List[Dict[str, Any]],
    model: str,
    api_key: str,
    temperature: float = 0.4,
    max_tokens: Optional[int] = None,
    timeout: float = 30.0,
) -> str:
    """Call any model via OpenRouter (OpenAI-compatible endpoint)."""
    payload: dict = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens:
        payload["max_tokens"] = max_tokens

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://chanakya.ai",
                "X-Title": "Chanakya Dharma Engine",
            },
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def call_gemini(
    messages: List[Dict[str, Any]],
    model: str,
    api_key: str,
    temperature: float = 0.4,
    max_tokens: Optional[int] = None,
    timeout: float = 30.0,
) -> str:
    """Call Gemini via the native Google AI API.

    `messages` must be in OpenAI role/content format — this function converts them.
    System messages are merged into the first user turn as a preamble.
    """
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    chat_messages = [m for m in messages if m["role"] != "system"]

    # Convert to Gemini contents format
    contents = []
    for m in chat_messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    # Prepend system prompt into first user turn
    if system_parts and contents:
        preamble = "\n\n".join(system_parts) + "\n\n"
        contents[0]["parts"][0]["text"] = preamble + contents[0]["parts"][0]["text"]
    elif system_parts:
        contents = [{"role": "user", "parts": [{"text": "\n\n".join(system_parts)}]}]

    generation_config: dict = {"temperature": temperature}
    if max_tokens:
        generation_config["maxOutputTokens"] = max_tokens

    payload = {"contents": contents, "generationConfig": generation_config}

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": api_key},
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


async def call_openai(
    messages: List[Dict[str, Any]],
    model: str,
    api_key: str,
    temperature: float = 0.4,
    max_tokens: Optional[int] = None,
    timeout: float = 30.0,
) -> str:
    """Call any model via OpenAI API."""
    payload: dict = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens:
        payload["max_completion_tokens"] = max_tokens

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Fallback chain: OpenRouter → Gemini native → OpenAI gpt-4o-mini
# ---------------------------------------------------------------------------

async def call_with_fallback(
    messages: List[Dict[str, Any]],
    temperature: float = 0.4,
    max_tokens: Optional[int] = None,
    timeout: float = 30.0,
) -> str:
    """Try free providers in order; fall back to gpt-4o-mini on rate limit or error.

    Order:
      1. google/gemini-2.5-flash:free via OpenRouter
      2. gemini-2.5-flash via Gemini native API
      3. gpt-4o-mini via OpenAI
    """
    from chanakya.config import OPENROUTER_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY

    providers = []

    # Primary: native Gemini API (free tier via Google AI Studio key)
    if GEMINI_API_KEY:
        providers.append({
            "name": "Gemini native/gemini-2.5-flash",
            "fn": call_gemini,
            "kwargs": {
                "model": "gemini-2.5-flash",
                "api_key": GEMINI_API_KEY,
            },
        })

    # Always add OpenAI as final fallback
    providers.append({
        "name": "OpenAI/gpt-4o-mini",
        "fn": call_openai,
        "kwargs": {
            "model": "gpt-4o-mini",
            "api_key": OPENAI_API_KEY,
        },
    })

    last_error: Exception | None = None
    for provider in providers:
        try:
            result = await provider["fn"](
                messages,
                **provider["kwargs"],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            logger.debug("call_with_fallback: success via %s", provider["name"])
            return result
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in _RATE_LIMIT_STATUSES:
                logger.warning(
                    "%s rate-limited (%d) — trying next provider",
                    provider["name"], status,
                )
            else:
                logger.warning(
                    "%s error %d — trying next provider: %s",
                    provider["name"], status, exc.response.text[:150],
                )
            last_error = exc
        except Exception as exc:
            logger.warning("%s failed — trying next provider: %s", provider["name"], exc)
            last_error = exc

    logger.error("All LLM providers exhausted. Last error: %s", last_error)
    return f"Error: All LLM providers unavailable ({last_error})"


# ---------------------------------------------------------------------------
# Legacy compat: call_llm (explicit model, used by council_tools etc.)
# ---------------------------------------------------------------------------

async def call_llm(
    messages: List[Dict[str, Any]],
    model_name: str,
    temperature: float = 0.4,
    max_tokens: Optional[int] = None,
    timeout: float = 90.0,
) -> str:
    """Call a specific model. Prefix 'openrouter/' to route via OpenRouter."""
    from chanakya.config import OPENROUTER_API_KEY, OPENAI_API_KEY

    if model_name.startswith("openrouter/"):
        api_model = model_name[len("openrouter/"):]
        try:
            return await call_openrouter(messages, api_model, OPENROUTER_API_KEY, temperature, max_tokens, timeout)
        except Exception as exc:
            logger.error("call_llm openrouter/%s failed: %s", api_model, exc)
            return f"Error: {exc}"
    else:
        try:
            return await call_openai(messages, model_name, OPENAI_API_KEY, temperature, max_tokens, timeout)
        except Exception as exc:
            logger.error("call_llm openai/%s failed: %s", model_name, exc)
            return f"Error: {exc}"
