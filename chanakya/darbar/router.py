"""
router.py — The Mantri (Router/Classifier).

Classifies user intent and routes to the appropriate specialist agent.
Uses fast-path regex for common patterns, falls back to nano LLM for ambiguous inputs.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta

import httpx

from chanakya.darbar.state import DarbarState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fast-path regex patterns (no LLM call needed, <1ms)
# ---------------------------------------------------------------------------

_FAST_PATHS: list[tuple[re.Pattern, str, str, int]] = [
    # (regex, specialist, intent, context_tier_needed)
    (re.compile(r"\b(ledger|penalty|debt|money|spend|budget|financial|expense|income|salary|invest)\b", re.I),
     "kautilya", "finance", 2),
    (re.compile(r"\b(sleep|energy|mood|water|meditation|ritual|health|tired|exhausted|gym|workout|weight|calories|fasting)\b", re.I),
     "charaka", "health", 2),
    (re.compile(r"\b(code|deploy|architecture|bug|system design|leetcode|algorithm|refactor|git|PR|pull request)\b", re.I),
     "vishvakarma", "tech", 1),
]

# Intent patterns that stay with Chanakya but help classify
_INTENT_PATTERNS: list[tuple[re.Pattern, str, int]] = [
    (re.compile(r"\b(schedule|reschedule|cancel|meeting|event|tomorrow|today at|set.*time)\b", re.I),
     "schedule", 2),
    (re.compile(r"\b(mindset|quote|principle|wisdom|goal|remember this|add this)\b", re.I),
     "mindset", 1),
    (re.compile(r"\b(streak|commitment|accountability|punishment|war mode)\b", re.I),
     "accountability", 2),
]

# ---------------------------------------------------------------------------
# LLM Router Schema
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM_PROMPT = """You classify messages for a personal accountability coach system.
Given a user message, classify:
- intent: what the user wants (casual, schedule, finance, health, tech, mindset, accountability, checkpoint_response)
- specialist: who handles it (chanakya=general/mentor, kautilya=finance, charaka=health, vishvakarma=tech)
- context_tier_needed: how much background context is needed (1=minimal, 2=daily, 3=weekly patterns, 4=deep memory)
- urgency: how important (low=casual chat, medium=normal request, high=deadline/commitment, critical=emergency)

Return ONLY a JSON object:
{"intent": "...", "specialist": "...", "context_tier_needed": N, "urgency": "...", "reasoning": "one sentence"}"""


# ---------------------------------------------------------------------------
# Main routing function
# ---------------------------------------------------------------------------


async def route(state: DarbarState, user: dict) -> DarbarState:
    """Classify intent and determine which specialist handles this message.

    Priority order:
    1. Bypass conditions (checkpoint responses, scheduled interactions)
    2. Fast-path regex matching
    3. LLM-based classification (nano model)
    4. Default to chanakya (guru) with tier 1
    """
    text = state.scrubbed_input or state.raw_input

    # 1. Bypass: checkpoint responses get full context
    if _is_checkpoint_response(user):
        state.specialist = "chanakya"
        state.intent = "checkpoint_response"
        state.context_tier_needed = 2
        state.urgency = "high"
        state.routed_via = "bypass"
        state.router_reasoning = "User responding to recent checkpoint"
        return state

    # 2. Bypass: scheduled interactions (EOD, MORNING_TODO, etc.) already have type set
    if state.interaction_type not in ("MENTOR_TALK", "CHECKPOINT"):
        state.specialist = "chanakya"
        state.intent = state.interaction_type.lower()
        state.context_tier_needed = 3 if state.interaction_type in ("EOD", "WEEKLY_REVIEW") else 2
        state.urgency = "high"
        state.routed_via = "bypass"
        state.router_reasoning = f"Scheduled interaction: {state.interaction_type}"
        return state

    # 3. Fast-path: specialist routing via regex
    for pattern, specialist, intent, tier in _FAST_PATHS:
        if pattern.search(text):
            state.specialist = specialist
            state.intent = intent
            state.context_tier_needed = tier
            state.urgency = "medium"
            state.routed_via = "fast_path"
            state.router_reasoning = f"Keyword match: {intent}"
            return state

    # 4. Fast-path: intent classification for chanakya
    for pattern, intent, tier in _INTENT_PATTERNS:
        if pattern.search(text):
            state.specialist = "chanakya"
            state.intent = intent
            state.context_tier_needed = tier
            state.urgency = "medium"
            state.routed_via = "fast_path"
            state.router_reasoning = f"Intent match: {intent}"
            return state

    # 5. Short casual messages — skip LLM routing
    if len(text.split()) <= 4 and not any(c in text for c in "?!"):
        state.specialist = "chanakya"
        state.intent = "casual"
        state.context_tier_needed = 1
        state.urgency = "low"
        state.routed_via = "fast_path"
        state.router_reasoning = "Short casual message"
        return state

    # 6. LLM routing (nano model)
    llm_result = await _llm_route(text, user)
    if llm_result:
        state.specialist = llm_result.get("specialist", "chanakya")
        state.intent = llm_result.get("intent", "casual")
        state.context_tier_needed = llm_result.get("context_tier_needed", 2)
        state.urgency = llm_result.get("urgency", "medium")
        state.router_reasoning = llm_result.get("reasoning", "")
        state.routed_via = "llm"
        return state

    # 7. Default fallback
    state.specialist = "chanakya"
    state.intent = "casual"
    state.context_tier_needed = 2
    state.urgency = "medium"
    state.routed_via = "fast_path"
    state.router_reasoning = "Default fallback"
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_checkpoint_response(user: dict) -> bool:
    """Check if the user is responding to a recent scheduled checkpoint."""
    from chanakya.db.mongo import interaction_logs

    last_scheduled = interaction_logs.find_one(
        {
            "user_id": user["_id"],
            "trigger_type": "SCHEDULED",
            "user_response": None,
            "timestamp": {"$gte": datetime.utcnow() - timedelta(minutes=30)},
        },
        sort=[("timestamp", -1)],
    )
    return last_scheduled is not None


async def _llm_route(text: str, user: dict) -> dict | None:
    """Use nano model for intent classification when fast-path doesn't match."""
    user_context = (
        f"User: {user.get('name', 'User')}, "
        f"mode: {user.get('current_mode', 'NORMAL')}, "
        f"time: {datetime.utcnow().strftime('%H:%M')}, "
        f"activity: {user.get('current_activity', 'FREE_TIME')}"
    )

    try:
        from chanakya.agent.llm_provider import call_with_fallback
        content = (await call_with_fallback(
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context: {user_context}\nMessage: {text}"},
            ],
            temperature=0.1,
            max_tokens=150,
            timeout=5.0,
        )).strip()

            # Parse JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    data = json.loads(content[start:end])
                else:
                    return None

            # Validate
            valid_specialists = {"chanakya", "kautilya", "charaka", "vishvakarma"}
            specialist = data.get("specialist", "chanakya")
            if specialist not in valid_specialists:
                specialist = "chanakya"

            tier = data.get("context_tier_needed", 2)
            if not isinstance(tier, int) or tier < 1 or tier > 4:
                tier = 2

            valid_urgency = {"low", "medium", "high", "critical"}
            urgency = data.get("urgency", "medium")
            if urgency not in valid_urgency:
                urgency = "medium"

            return {
                "specialist": specialist,
                "intent": data.get("intent", "casual"),
                "context_tier_needed": tier,
                "urgency": urgency,
                "reasoning": data.get("reasoning", ""),
            }

    except Exception as exc:
        logger.warning("LLM routing failed (falling back to default): %s", exc)
        return None
