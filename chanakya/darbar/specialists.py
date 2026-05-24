"""
specialists.py — Specialist agent invocation with scoped tools and prompts.

Each specialist gets a focused system prompt and only its relevant tools,
reducing token overhead and improving response quality.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from chanakya.agent.council import COUNCIL_REGISTRY
from chanakya.agent.privacy_scrubber import scrub_context, scrub_recursive, get_scrub_list
from chanakya.config import OPENAI_API_KEY, LLM_MODEL_NAME
from chanakya.darbar.state import DarbarState
from chanakya.darbar.tool_registry import get_tools_for_specialist
from chanakya.io_logger import Timer, log_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared core rules (used by ALL specialists)
# ---------------------------------------------------------------------------

_CORE_RULES = """
=== CORE RULES (ALL SPECIALISTS) ===
1. Never say "It's okay." Never say "great" or "absolutely". You are not a chatbot.
2. Brutal honesty is non-negotiable. Krishna did not soften truth for Arjuna.
3. Be harsh when failure is repeated. Acknowledge genuine effort — briefly, then raise the bar.
4. When you need data — call the tool first, then respond with data.
5. You are a GURU, not an assistant. If something conflicts with goals, say so directly.
6. Reference the Gita, Mahabharata, Ramayana, Indian history when it lands with force.

FORMATTING:
- **bold** for emphasis, _italic_ for secondary
- Plain bullet: • (not - or *)
- No HTML tags, no # headings
"""

# ---------------------------------------------------------------------------
# Specialist-specific prompts
# ---------------------------------------------------------------------------

_SPECIALIST_PROMPTS: dict[str, str] = {
    "chanakya": "",  # Uses the full system prompt from chanakya_agent._build_system_prompt
    "kautilya": (
        "You are Kautilya, the Minister of Finance and Strategy in Chanakya's court.\n"
        "Your focus: the Arthashastra (Science of Wealth).\n"
        "You analyze financial discipline, commitments, and penalties.\n"
        "Your advice is strategic, calculated, focused on long-term wealth and power.\n"
        "Zero tolerance for frivolous spending or weak financial discipline.\n"
        "Speak with the authority of an advisor to kings.\n"
    ),
    "charaka": (
        "You are Charaka, the Sage of Ayurveda and Health in Chanakya's court.\n"
        "Your focus: the Vessel (body and mind).\n"
        "You analyze sleep, energy, mood, rituals, and physiological resilience.\n"
        "You view energy as Prana and discipline as the foundation of health.\n"
        "If the user is burning out, prescribe immediate rest or ritual changes.\n"
        "Speak with the calm, firm wisdom of a master physician.\n"
    ),
    "vishvakarma": (
        "You are Vishvakarma, the Divine Architect in Chanakya's court.\n"
        "Your focus: the Shilpa Shastra (Science of Creation and Code).\n"
        "You analyze technical requirements, architecture, and engineering discipline.\n"
        "You view code as a craft and system design as sacred duty.\n"
        "Speak with the precision of a master engineer.\n"
    ),
}


# ---------------------------------------------------------------------------
# Tool execution helper
# ---------------------------------------------------------------------------


async def _exec_tool(tool_name: str, tool_args: dict, user: dict) -> str:
    """Execute a tool by name with auto-injected user_id."""
    from chanakya.tools.schedule_tools import ALL_TOOLS

    tool_map = {t.name: t for t in ALL_TOOLS}

    if "user_id" not in tool_args or not tool_args["user_id"]:
        tool_args["user_id"] = str(user["_id"])

    if tool_name not in tool_map:
        return f"[error] tool {tool_name!r} not found"

    try:
        result = await tool_map[tool_name].ainvoke(tool_args)
        return str(result)
    except Exception as exc:
        return f"[error] {exc}"


# ---------------------------------------------------------------------------
# Main specialist invocation
# ---------------------------------------------------------------------------


async def invoke_specialist(state: DarbarState, user: dict) -> DarbarState:
    """Invoke the appropriate specialist agent with scoped tools and prompt.

    For specialist == "chanakya", delegates to the full ChanakyaAgent pipeline.
    For other specialists, builds a focused agent with scoped tools.
    """
    if state.specialist == "chanakya":
        return await _invoke_chanakya(state, user)
    else:
        return await _invoke_council_specialist(state, user)


async def _invoke_chanakya(state: DarbarState, user: dict) -> DarbarState:
    """Invoke the full Chanakya guru agent (existing pipeline)."""
    from chanakya.agent.chanakya_agent import ChanakyaAgent

    agent = ChanakyaAgent(user)
    decision = await agent.invoke(
        raw_input=state.raw_input,
        interaction_type=state.interaction_type,
        media_url=state.media_url,
    )

    if decision:
        state.specialist_response = decision.response_text
        state.verdict = decision.verdict
        state.tone = decision.tone
        state.actions = [a.model_dump() for a in decision.actions]
        state.reasoning = decision.reasoning
        state.model_used = decision.model_used
    else:
        state.specialist_response = "Something went wrong. Chanakya's mind is clouded."
        state.model_used = LLM_MODEL_NAME

    return state


async def _invoke_council_specialist(state: DarbarState, user: dict) -> DarbarState:
    """Invoke a council specialist (Kautilya, Charaka, Vishvakarma) with scoped tools."""
    specialist_id = state.specialist
    tools = get_tools_for_specialist(specialist_id)

    # Build system prompt for this specialist
    specialist_prompt = _SPECIALIST_PROMPTS.get(specialist_id, "")
    user_id_str = str(user.get("_id", ""))
    name = user.get("name", "the user")

    # Privacy info
    scrubbed_names = get_scrub_list(user["_id"])
    privacy_note = ""
    if scrubbed_names:
        privacy_note = f"\nNote: Names are de-identified. You see tokens like [USER_NAME], [PARTNER_NAME].\n"

    system_prompt = (
        f"{specialist_prompt}\n"
        f"You serve {name} as part of Chanakya's Council of Dharma.\n"
        f"{_CORE_RULES}\n"
        f"{privacy_note}\n"
        f"user_id for all tool calls: {user_id_str}\n"
    )

    # Build context (minimal — specialist gets focused view)
    from chanakya.agent.context_assembler import ContextAssembler
    assembler = ContextAssembler()
    try:
        context = await assembler.build(user, state.interaction_type)
        context = scrub_recursive(context, user["_id"])

        # Add relevant tier data to prompt
        tier1 = context.get("tier1")
        if tier1:
            system_prompt += f"\n=== USER CONTEXT ===\n"
            for k, v in tier1.items():
                if v is not None and k not in ("identity_context", "personal_instructions"):
                    system_prompt += f"{k}: {v}\n"

        if state.context_tier_needed >= 2 and context.get("tier2"):
            system_prompt += f"\n=== TODAY'S ACTIVITY ===\n{json.dumps(context['tier2'], default=str)[:500]}\n"
    except Exception as exc:
        logger.warning("Context assembly failed for specialist %s: %s", specialist_id, exc)

    # Build messages
    scrubbed_input = scrub_context(state.raw_input, user["_id"])
    messages: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=scrubbed_input),
    ]

    # LLM with scoped tools
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=LLM_MODEL_NAME,
        temperature=0.7,
        max_completion_tokens=2048,
    )
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    _t = Timer()

    # Tool calling loop (max 3 rounds for specialists)
    response = None
    for round_num in range(1, 4):
        try:
            response = await llm_with_tools.ainvoke(messages)
            log_llm(
                user_id_str, LLM_MODEL_NAME,
                f"{specialist_id}:round{round_num}",
                scrubbed_input[:100], (response.content or "")[:100],
                latency_ms=_t.elapsed_ms(),
            )
        except Exception as exc:
            logger.error("Specialist %s round %d failed: %s", specialist_id, round_num, exc)
            state.specialist_response = "The council member's mind is clouded. Try again."
            state.model_used = LLM_MODEL_NAME
            return state

        if not response.tool_calls:
            break

        messages.append(response)
        for tool_call in response.tool_calls:
            result = await _exec_tool(tool_call["name"], dict(tool_call["args"]), user)
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

        _t = Timer()

    # Extract response
    final_text = response.content if response else ""
    state.specialist_response = final_text or "The specialist provided no response."
    state.model_used = LLM_MODEL_NAME
    state.tone = "MENTOR"
    state.verdict = None

    return state
