"""
orchestrator.py — The Darbar pipeline orchestrator.

Full flow:
  1. Privacy scrub input
  2. Route (Mantri) — fast-path or LLM classification
  3. Inject goal nudges (if pending)
  4. Invoke specialist with scoped tools
  5. Synthesize voice (if specialist != chanakya)
  6. Dharma Gate quality check — logs issues, never blocks
  7. Privacy unscrub response
  8. Return LLMDecision (same interface as legacy agent)
"""

from __future__ import annotations

import logging
import time

from chanakya.agent.privacy_scrubber import scrub_context, unscrub_response
from chanakya.darbar.router import route
from chanakya.darbar.specialists import invoke_specialist
from chanakya.darbar.state import DarbarState
from chanakya.models.llm_decision import ActionItem, LLMDecision

logger = logging.getLogger(__name__)


async def process(
    user: dict,
    raw_input: str,
    interaction_type: str,
    media_url: str | None = None,
) -> LLMDecision | None:
    """Execute the full Darbar multi-agent pipeline.

    Returns an LLMDecision — same interface as the legacy ChanakyaAgent.invoke().
    """
    start_time = time.time()

    # 1. Initialize state
    state = DarbarState(
        user_id=str(user.get("_id", "")),
        raw_input=raw_input,
        media_url=media_url,
        interaction_type=interaction_type,
    )

    # 2. Privacy scrub
    state.scrubbed_input = scrub_context(raw_input, user["_id"])

    # 3. Route (Mantri)
    state = await route(state, user)
    logger.info(
        "Darbar routed [%s]: specialist=%s, intent=%s, tier=%d, via=%s — %s",
        user.get("name", "?"), state.specialist, state.intent,
        state.context_tier_needed, state.routed_via, state.router_reasoning,
    )

    # 3.5 Inject pending goal nudges into input (if any)
    from chanakya.darbar.goal_sentinel import deliver_pending_nudges
    nudges = await deliver_pending_nudges(user["_id"])
    if nudges:
        nudge_block = "\n".join(f"• {n}" for n in nudges[:3])
        state.scrubbed_input += (
            f"\n\n[GOAL REMINDER from Sentinel: {nudge_block}]"
        )

    # 4. Invoke specialist
    state = await invoke_specialist(state, user)

    # 5. Synthesize (if non-chanakya specialist)
    if state.specialist != "chanakya" and state.specialist_response:
        state = await _synthesize(state, user)

    # 6. Dharma Gate (quality check for medium+ urgency — logs only, never blocks)
    if state.urgency in ("medium", "high", "critical") and state.specialist != "chanakya":
        state = await _dharma_gate(state, user)

    # 7. Determine final response
    final_text = state.final_response or state.specialist_response
    if not final_text:
        final_text = "Something went wrong. Chanakya's mind is clouded. Try again."

    # 8. Privacy unscrub
    final_text = unscrub_response(final_text, user["_id"])

    # 9. Build LLMDecision
    state.latency_ms = (time.time() - start_time) * 1000

    actions = []
    for a in state.actions:
        if isinstance(a, dict):
            actions.append(ActionItem(type=a.get("type", ""), params=a.get("params", {})))

    decision = LLMDecision(
        verdict=state.verdict,
        response_text=final_text,
        tone=state.tone,
        reasoning=state.reasoning,
        model_used=state.model_used,
        actions=actions,
    )

    logger.info(
        "Darbar complete [%s]: specialist=%s, latency=%.0fms, verdict=%s",
        user.get("name", "?"), state.specialist, state.latency_ms, state.verdict,
    )

    # Background: kick off learning cycle if due (never blocks response)
    import asyncio
    from chanakya.darbar.learning_extractor import should_run_learning, run_learning_cycle
    if should_run_learning(user["_id"]):
        asyncio.ensure_future(run_learning_cycle(user["_id"]))

    return decision


# ---------------------------------------------------------------------------
# Synthesizer — wrap specialist output in Chanakya's guru voice
# ---------------------------------------------------------------------------


async def _synthesize(state: DarbarState, user: dict) -> DarbarState:
    """Rewrite specialist output in Chanakya's voice (nano model)."""
    name = user.get("name", "the user")
    prompt = (
        f"You are Chanakya — the greatest guru. A specialist ({state.specialist}) has analyzed "
        f"a query from {name} and produced this report:\n\n"
        f"{state.specialist_response}\n\n"
        "Rewrite this in YOUR voice — the guru's voice. Keep ALL the substance and data. "
        "Change only the style: sharp, direct, punchy. Use a reference from the Gita or "
        "Mahabharata if it lands naturally. Max 3-4 sentences unless data requires more."
    )

    try:
        from chanakya.agent.llm_provider import call_with_fallback
        synthesized = (await call_with_fallback(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=400,
            timeout=8.0,
        )).strip()
        if synthesized:
            state.final_response = synthesized
            return state
    except Exception as exc:
        logger.warning("Synthesis failed (using raw specialist response): %s", exc)

    state.final_response = state.specialist_response
    return state


# ---------------------------------------------------------------------------
# Dharma Gate — quality check before sending
# ---------------------------------------------------------------------------


async def _dharma_gate(state: DarbarState, user: dict) -> DarbarState:
    """Quick quality check: is the response too soft, off-topic, or contradictory?"""
    response_to_check = state.final_response or state.specialist_response
    if len(response_to_check) < 50:
        state.gate_passed = True
        return state

    prompt = (
        "You are a quality gate for an accountability guru AI. Check this response:\n\n"
        f"User said: {state.scrubbed_input[:200]}\n"
        f"Response: {response_to_check[:500]}\n\n"
        "Check:\n"
        "1. Is it too soft/agreeable? (The guru never coddles)\n"
        "2. Is it off-topic from what was asked?\n"
        "3. Does it say 'It's okay' or similar weak language?\n\n"
        'Return JSON: {"pass": true} or {"pass": false, "issue": "brief description"}'
    )

    try:
        from chanakya.agent.llm_provider import call_with_fallback
        content = (await call_with_fallback(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100,
            timeout=5.0,
        )).strip()

            import json
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    result = json.loads(content[start:end])
                else:
                    state.gate_passed = True
                    return state

            if result.get("pass", True):
                state.gate_passed = True
            else:
                state.gate_passed = False
                state.gate_issue = result.get("issue", "")
                logger.info("Dharma Gate flagged: %s", state.gate_issue)
                # Don't block — just log. Future: add revision attempt.

    except Exception as exc:
        logger.warning("Dharma Gate check failed (passing through): %s", exc)
        state.gate_passed = True

    return state
