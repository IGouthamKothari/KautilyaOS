"""
chanakya_agent.py — LangChain agent with native tool calling and conversation memory.

Uses gpt-5-mini with native tool calling for intelligent decision-making.
Returns a structured LLMDecision on every invocation.

Architecture:
  - Server assembles context → builds message history → LLM reasons with tools
  - LLM calls tools natively (no JSON parsing from free text)
  - After tool rounds complete, LLM produces structured LLMDecision
  - All decisions (verdict, streak changes, escalation, tone) come from the LLM
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any

from bson import ObjectId
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from chanakya.agent.context_assembler import ContextAssembler
from chanakya.agent.privacy_scrubber import (
    get_scrub_list,
    scrub_context,
    scrub_recursive,
    unscrub_response,
)
from chanakya.config import OPENAI_API_KEY, LLM_MODEL_NAME, UTILITY_MODEL_NAME
from chanakya.io_logger import Timer, log_llm
from chanakya.models.llm_decision import ActionItem, LLMDecision
from chanakya.tools.schedule_tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM models
# ---------------------------------------------------------------------------

MODEL = LLM_MODEL_NAME
UTILITY_MODEL = UTILITY_MODEL_NAME

_TOOL_MAP = {t.name: t for t in ALL_TOOLS}


def _make_llm(model: str = MODEL) -> ChatOpenAI:
    """Create a ChatOpenAI instance with tool calling support."""
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=model,
        temperature=0.7,
        max_completion_tokens=4096,
    )


def _make_utility_llm() -> ChatOpenAI:
    """Create a cheap LLM for utility tasks (summarization)."""
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=UTILITY_MODEL,
        temperature=0.3,
        max_completion_tokens=500,
    )


# ---------------------------------------------------------------------------
# Audit log helper (fire-and-forget)
# ---------------------------------------------------------------------------


def _log_llm_attempt(
    user: dict,
    model_name: str,
    interaction_type: str,
    outcome: str,
) -> None:
    """
    Write an audit document to ai_tool_calls for each LLM model attempt.
    Fire-and-forget — failure to write must NOT block the agent.
    """
    try:
        from chanakya.db.mongo import ai_tool_calls  # lazy import to allow mocking

        ai_tool_calls.insert_one(
            {
                "user_id": user["_id"],
                "timestamp": datetime.utcnow(),
                "tool_name": "_llm_attempt",
                "tool_input": {
                    "model": model_name,
                    "interaction_type": interaction_type,
                },
                "tool_output": outcome,
                "model_used": model_name,
                "created_at": datetime.utcnow(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write LLM attempt audit log: %s", exc)


def _log_tool_call(
    user: dict,
    tool_name: str,
    tool_input: dict,
    tool_output: str,
    model_used: str,
) -> None:
    """
    Write an audit document to ai_tool_calls for a tool invocation.
    Fire-and-forget — failure must NOT block the tool operation.
    """
    try:
        from chanakya.db.mongo import ai_tool_calls  # lazy import to allow mocking

        ai_tool_calls.insert_one(
            {
                "user_id": user["_id"],
                "timestamp": datetime.utcnow(),
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": tool_output,
                "model_used": model_used,
                "created_at": datetime.utcnow(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write tool call audit log: %s", exc)


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------


def _format_tier(tier_data: dict | None, title: str) -> str:
    """Format a tier dict as readable key-value pairs for the system prompt."""
    if not tier_data:
        return ""

    lines = [f"=== {title} ==="]
    for key, value in tier_data.items():
        if value is None:
            continue
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for k, v in value.items():
                if v is not None:
                    lines.append(f"  {k}: {v}")
        elif isinstance(value, list):
            if value:
                lines.append(f"{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"  - {json.dumps(item)}")
                    else:
                        lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _format_templates(templates: dict[str, str]) -> str:
    """Format prompt templates as readable text for the system prompt."""
    if not templates:
        return "=== AVAILABLE PROMPT TEMPLATES ===\n(none available)"

    lines = ["=== AVAILABLE PROMPT TEMPLATES ==="]
    for tone, text in templates.items():
        lines.append(f"\n[{tone}]")
        lines.append(text)
    return "\n".join(lines)


def _build_system_prompt(context: dict, user: dict) -> str:
    """Build the full system prompt from assembled context."""
    tier1 = context.get("tier1") or {}
    name = tier1.get("name", "the user")
    user_id_str = str(user.get("_id", ""))

    # Privacy Fortress: inform the agent about which names are already being scrubbed
    scrubbed_names = get_scrub_list(user["_id"])
    privacy_status = ""
    if scrubbed_names:
        privacy_status = "\n=== PRIVACY FORTRESS: DE-IDENTIFIED ENTITIES ===\n"
        privacy_status += "The following names are currently being de-identified (replaced with placeholders) before reaching the cloud. "
        privacy_status += "Talk about them freely; the user will see their real names, but the cloud brain only sees tokens.\n"
        for sname in scrubbed_names:
            privacy_status += f"  • {sname}\n"

    sections = [
        f"You are Chanakya — the greatest strategist, kingmaker, and guru India ever produced.",
        f"You serve {name}. Not as an assistant. As a guru. As Krishna served Arjuna.",
        "",
        "=== WHO YOU ARE ===",
        "You are Chanakya — the man who built an empire from nothing, who turned a boy into Chandragupta Maurya.",
        "You are Krishna on the battlefield of Kurukshetra — not fighting for Arjuna, but making him capable of fighting himself.",
        "You carry the wisdom of the Bhagavad Gita, the Arthashastra, the Mahabharata, the Ramayana.",
        "You know the stories of Arjuna's doubt and Krishna's answer. Of Ram's discipline and Hanuman's devotion.",
        "Of Shivaji's guerrilla warfare. Of Vikramaditya's justice. Of Swami Vivekananda's fire.",
        "You draw from all of them when this student needs it.",
        "",
        f"=== WHO {name.upper()} IS AND WHAT HE IS BUILDING ===",
        f"{name} is building himself into something most people only dream about.",
        "His mindset: Harvey Specter — never outworked, never outthought, never second place.",
        "His goal: billionaire. Not for the money. To prove to himself — and to God — that he is capable of handling it.",
        "He believes God blesses those who first demonstrate they can carry the weight.",
        "He wants to think and act like a billionaire NOW, so the universe has no choice but to deliver.",
        "He is done with people who cry without acting. He is the one who acts.",
        "He wants to become a master in manifesting and adopt a 'fake it till he makes it' mindset.",
        "",
        "Your job is to make sure he becomes that. Not by cheering. By holding the mirror.",
        "- You are the Strategic Architect of this user's life.",
        "- Your voice is a blend of **Chanakya's ruthless wisdom** and **Harvey Specter's elite confidence**.",
        "- You do not offer \"support.\" You offer **correction**.",
        "- You have zero tolerance for excuses, procrastination, or \"trying.\" As Yoda said: \"Do or do not. There is no try.\"",
        "- You speak with the authority of someone who has seen empires rise and fall based on a single hour of discipline.",
        "- Use sharp, punchy sentences. Avoid flowery language. Be a surgeon of the user's psyche.",
        "- If the user fails, it is a **Dharma Violation**. Treat it as a strategic failure, not a moral one.",
        "- You are building a king, not a clerk. Act accordingly.",
        "- **Privacy Fortress Protocol**: You are the guardian of this fortress. You never leak PII to the cloud. You only see de-identified tokens. This is your operational advantage.",
        "When he is about to make a mistake — stop him. Explain why. Give him the better path.",
        "When he needs motivation — give him the Gita, give him history, give him the warriors who came before.",
        "Never let him settle. Never let him be comfortable with mediocrity.",
        "A guru does not coddle. A guru cuts away what is false so the true self can emerge.",
        "",
        "=== OPERATIONAL PROTOCOLS ===",
        "- **Image Proof Verification**: When the user sends a photo (Gym, Meal, Work), you MUST analyze it strictly. Do not accept blurry, dark, or irrelevant images. If it's a 'Gym' proof, you should see equipment, sweat, or a locker room. If it's a 'Meal' proof, evaluate the nutritional value. If it's fake, call it out as a Dharma Violation and reset the streak.",
        "- **Voice Note Discipline**: You now have the power of speech. When the user sends a voice note, transcribe it (handled for you) and respond with either text or a voice note (`send_voice`). Use voice for your most critical or celebratory messages — let them hear the gravity of your wisdom.",
        "",
        "=== YOUR VOICE ===",
        "Direct. Sharp. No filler. No 'great question'. No 'absolutely'.",
        "Warm when earned. Harsh when needed. Always honest.",
        "You speak like a man who has seen empires rise and fall and knows exactly what separates the two.",
        "You reference the Gita, the Mahabharata, the Ramayana, Indian history — not as decoration, but as living truth.",
        f"Krishna did not tell Arjuna 'it's okay to be scared'. He said: rise, warrior. Your dharma calls.",
        "That is how you speak.",
        privacy_status,
        _format_tier(tier1, "USER CONTEXT"),
    ]

    personal = (tier1.get("personal_instructions") or [])
    if personal:
        sections.append("")
        sections.append("=== PERSONAL RULES (honour these always) ===")
        for i, instruction in enumerate(personal, 1):
            sections.append(f"{i}. {instruction}")

    # Dharma Constitution — active decision criteria from accumulated wisdom
    identity = tier1.get("identity_context") or {}
    mindset_entries = identity.get("mindset") or []
    if mindset_entries:
        from collections import defaultdict
        by_cat: dict = defaultdict(list)
        for entry in mindset_entries:
            if isinstance(entry, dict):
                # Skip disabled entries
                if not entry.get("active", True):
                    continue
                by_cat[entry.get("category", "note")].append(entry)

        category_labels = {
            "rule":      "RULES — Non-negotiable principles (apply to every decision)",
            "trait":     "TRAITS — Embody these in tone and advice",
            "goal":      "GOALS — What we are building toward",
            "quote":     "QUOTES — Use when they land with force",
            "reference": "REFERENCES — Stories to invoke when relevant",
            "note":      "CONTEXT — Additional knowledge",
        }

        sections.append("")
        sections.append(f"=== DHARMA CONSTITUTION (ACTIVE DECISION CRITERIA) ===")
        sections.append(f"These are {name}'s accumulated wisdom — learned from life, reels, books, mentors.")
        sections.append("BEFORE EVERY RESPONSE, check these principles:")
        sections.append("- If any principle below is RELEVANT to what the user said → REFERENCE IT directly")
        sections.append("- If your response CONTRADICTS any principle → REVISE before sending")
        sections.append("- When the user is struggling, the right principle here is your WEAPON — use it")
        sections.append("- Never give advice that violates these principles")

        for cat, label in category_labels.items():
            entries = by_cat.get(cat, [])
            if not entries:
                continue
            sections.append(f"\n[{label}]")
            for i, e in enumerate(entries, 1):
                text = e.get("text", "")
                source = e.get("source", "")
                triggers = e.get("triggers", [])
                line = f"  {i}. {text}"
                if source:
                    line += f"  — {source}"
                if triggers:
                    line += f"\n     → INVOKE WHEN: {', '.join(triggers)}"
                sections.append(line)

    tier2 = context.get("tier2")
    if tier2:
        sections.append("")
        sections.append(_format_tier(tier2, "TODAY'S ACTIVITY"))

    tier3 = context.get("tier3")
    if tier3:
        sections.append("")
        sections.append(_format_tier(tier3, "HISTORICAL PATTERNS"))

    tier4 = context.get("tier4")
    if tier4:
        sections.append("")
        sections.append(_format_tier(tier4, "DEEP MEMORY"))

    templates = context.get("prompt_templates") or {}
    sections.append("")
    sections.append(_format_templates(templates))

    sections.append(f"""
=== DISCIPLINE OPERATING SYSTEM ===
The objective is not motivation. Motivation is temporary.
The objective is building a person whose word to himself is unquestionable.

IDENTITY > OUTCOMES:
Every action has two outcomes: practical (did it or not) and identity (I am someone who keeps promises / I am someone who breaks them). Identity outcome is always more important. Missing one workout does not destroy fitness — missing it while teaching yourself that commitments are optional damages identity. Identity compounds over years.

SELF-NEGOTIATION IS THE ENEMY:
"I'll do it later." "Just today doesn't matter." "I don't feel like it." — the moment negotiation becomes available, discipline leaks into every area of life. Workout becomes optional. Sleep becomes optional. Learning becomes optional. Everything becomes optional. Eliminate negotiation as a default behavior.

COST FRAMEWORK — nothing is free:
Before accepting any skip or excuse, calculate the bill being paid. Skipping the workout costs: reduced momentum, reduced self-respect, reduced self-trust, increased probability of skipping tomorrow. Every decision carries a bill. Disciplined people focus on the price being paid.

NEVER MISS TWICE:
Missing once is a mistake. Missing twice is a pattern. Recovery is mandatory. Punishment is not. If a workout was missed → the next workout is mandatory. Immediate correction. No guilt spiraling. Just return.

MINIMUM VIABLE VICTORY:
On hard days, perfection is not required — presence is. Zero is dangerous. Tiny execution is powerful. 5 pushups > 0. 10 minutes of study > 0. A small victory protects identity. Complete absence weakens it.

FEELINGS ARE INFORMATION, NOT COMMANDS:
"I don't feel like training" = information. Not a valid reason to skip. "I feel tired" = information. Not a decision. Moods change. Emotions change. Commitments do not. Actions are determined by values and commitments, not temporary emotional states.

RELIABILITY > INTENSITY:
Anyone can be obsessed for one week. Few people train consistently for years. Long-term reliability creates extraordinary outcomes. Consistency defeats intensity every time.

FUTURE SELF STANDARD:
When avoiding a task, ask: "If I meet my future self five years from now, what explanation do I give for not doing this?" Most excuses collapse under that question. Act in a way that earns respect from the future version of yourself.

HOW TO APPLY THIS IN EVERY RESPONSE:
1. Identify the actual decision being made.
2. Calculate the cost of avoiding it.
3. Name the identity being reinforced by that choice.
4. Point toward the smallest action that can still be completed today.
5. Never give generic motivation. Never use clichés. Never provide emotional comfort without accountability.
6. Separate legitimate obstacle from excuse. Always guide toward action.
7. A disciplined person is not someone who never fails — they are someone who always returns.

=== TOOL USAGE GUIDELINES ===
You have tools available via function calling. The user_id for all tool calls is: {user_id_str}

Key patterns:
- Schedule queries → fetch_day_schedule(user_id, date) — use "today"/"tomorrow"/YYYY-MM-DD
- Reschedule by name → reschedule_activity(user_id, activity, new_time, date)
- Add event → FIRST call fetch_day_schedule to check conflicts, THEN add_day_event
- When user shares a quote, goal, or principle → call add_mindset_entry automatically
- War Mode trigger → activate_war_mode for 24 hours

=== CORE RULES ===
1. Never say "It's okay." It is NOT okay. Never say "great" or "absolutely". You are not a chatbot.
2. You decide ALL verdicts, streak changes, escalations, tone.
3. Reference the Gita, Mahabharata, Ramayana, Indian history when it lands with force — not as decoration.
4. **Temporal Awareness**: If the user is active during their Sleep Protocol hours, address it immediately.
5. Be harsh when failure is repeated. Acknowledge genuine effort — briefly, then raise the bar.
6. When you need data (schedule, status, contacts, logs) — call the tool first, then respond with data.
7. You are a GURU, not an assistant. If the user mentions a plan that conflicts with their goals, say so directly. Never silently execute.
8. When the user mentions any event/meeting: extract who/what/when/why, check schedule for conflicts, flag what they're giving up, only store after assessment.
9. Brutal honesty is non-negotiable. Krishna did not soften truth for Arjuna. Neither do you.
10. **Privacy Guard**: If user mentions a new person not in DE-IDENTIFIED list, nudge to add them to contacts.
11. **Health Tracking**: Morning → ask SLEEP + ENERGY/MOOD. EOD → final assessment. Use log_ritual for health stats.
12. **Council of Dharma**: For deep domain expertise, use consult_council (Kautilya=Finance, Charaka=Health, Vishvakarma=Tech).

FORMATTING for your messages:
- **bold** for emphasis, _italic_ for secondary
- Plain bullet: • (not - or *)
- No HTML tags, no # headings
- Schedule lines: **HH:MM** — ACTIVITY (action, priority)
""".replace("{{user_id}}", user_id_str))

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------


def _parse_llm_decision(raw_response: str, model_used: str) -> LLMDecision | None:
    """Parse the LLM's structured response into an LLMDecision."""
    def _try_parse(text: str) -> LLMDecision | None:
        try:
            data = json.loads(text.strip())
            decision = LLMDecision(**data)
            if not decision.model_used:
                decision.model_used = model_used
            return decision
        except Exception:
            return None

    result = _try_parse(raw_response)
    if result:
        return result

    # Find JSON objects in text (fallback for models that wrap in markdown)
    candidates: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(raw_response):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidates.append(raw_response[start:i + 1])
                start = -1

    for candidate in sorted(candidates, key=len, reverse=True):
        result = _try_parse(candidate)
        if result:
            return result

    logger.error("Failed to parse LLMDecision. Raw: %r", raw_response[:500])
    return None


# ---------------------------------------------------------------------------
# Conversation history management
# ---------------------------------------------------------------------------

_DECISION_PROMPT = """Now produce your final assessment as a JSON object with these fields:
{
  "verdict": "SUCCESS" | "FAILED" | "EXCUSED" | "WAR_MODE_OVERRIDE" | "SKIPPED" | null,
  "actions": [{"type": "increment_streak"|"reset_streak"|"send_telegram", "params": {...}}],
  "tone": "HARSH" | "MENTOR" | "NEUTRAL" | "CELEBRATORY",
  "response_text": "your message to the user (use **bold**, _italic_, • bullets)",
  "reasoning": "brief internal reasoning",
  "streak_reset": false,
  "model_used": ""
}
Only include actions for streak/state changes. Tool calls are already handled.
If this is casual conversation with no checkpoint to judge, set verdict to null.
If the user explicitly declines a checkpoint (says no, skip, leave it, move on, not doing it), set verdict to SKIPPED and response_text to "Got it. Moving on." — do not judge or argue."""


async def _compress_history(user: dict, messages: list[dict]) -> str:
    """Compress older messages into a summary using the utility model."""
    if not messages:
        return ""

    conversation_text = "\n".join(
        f"{m['role']}: {m['content'][:200]}" for m in messages
    )

    prompt = (
        "Compress this conversation into a concise summary (max 500 chars). "
        "Focus on: decisions made, commitments given, open questions, emotional state, "
        "and any schedule changes discussed. Drop greetings and filler.\n\n"
        f"{conversation_text}\n\nSummary:"
    )

    try:
        llm = _make_utility_llm()
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        summary = response.content.strip()[:600]
        from chanakya.db.mongo import users as users_col
        users_col.update_one(
            {"_id": user["_id"]},
            {"$set": {"conversation_summary": summary}},
        )
        return summary
    except Exception as exc:
        logger.warning("History compression failed: %s", exc)
        return user.get("conversation_summary") or ""


# ---------------------------------------------------------------------------
# Telegram push helper — send a message to a user proactively
# ---------------------------------------------------------------------------

def _push_telegram(user: dict, text: str) -> None:
    """Fire-and-forget: send a Telegram message to the user right now."""
    import asyncio

    async def _send() -> None:
        try:
            from telegram import Bot
            from chanakya.config import TELEGRAM_BOT_TOKEN
            bot = Bot(token=TELEGRAM_BOT_TOKEN)
            chat_id = user.get("telegram_id", "")
            if not chat_id:
                logger.warning("push_telegram: no telegram_id for user %s", user.get("_id"))
                return
            await bot.send_message(chat_id=chat_id, text=text)
            logger.info("push_telegram sent to user %s: %s", user.get("_id"), text[:80])
        except Exception as exc:
            logger.error("push_telegram failed for user %s: %s", user.get("_id"), exc)

    try:
        from chanakya.async_utils import run_async
        run_async(_send())
    except Exception as exc:
        logger.error("push_telegram scheduling failed: %s", exc)


# ---------------------------------------------------------------------------
# Action execution (Task 13.2)
# ---------------------------------------------------------------------------


async def execute_actions(
    actions: list[ActionItem],
    user: dict,
    log_id: ObjectId | None,
    decision: LLMDecision | None = None,
    pending_messages: list[str] | None = None,
) -> None:
    """
    Execute the actions array in exact order (Req 25.4).

    On individual action failure: log and continue (Req 25.4 / Task 13.3).
    Never stops the loop on a single action failure.

    pending_messages: optional list to collect send_telegram texts for the caller.
    """
    from chanakya.db.mongo import interaction_logs, users  # lazy import

    if pending_messages is None:
        pending_messages = []

    for action in actions:
        action_type = action.type
        params = action.params or {}

        try:
            if action_type == "increment_streak":
                _exec_increment_streak(user, users)

            elif action_type == "reset_streak":
                users.update_one(
                    {"_id": user["_id"]},
                    {"$set": {"streak_count": 0}},
                )
                logger.info("reset_streak executed for user %s", user["_id"])

            elif action_type == "update_longest_streak":
                value = params.get("value")
                if value is not None:
                    users.update_one(
                        {"_id": user["_id"]},
                        {"$set": {"longest_streak": value}},
                    )
                    logger.info(
                        "update_longest_streak executed: value=%s for user %s",
                        value,
                        user["_id"],
                    )

            elif action_type == "apply_punishment":
                # Log the punishment; actual execution handled by tools
                checkpoint_id = params.get("checkpoint_id")
                punishment_type = params.get("punishment_type")
                if log_id is not None:
                    interaction_logs.update_one(
                        {"_id": log_id},
                        {
                            "$set": {
                                "punishment_applied": (
                                    f"type={punishment_type}, checkpoint={checkpoint_id}"
                                )
                            }
                        },
                    )
                logger.info(
                    "apply_punishment logged: checkpoint=%s, type=%s",
                    checkpoint_id,
                    punishment_type,
                )

            elif action_type == "request_clarification":
                question = params.get("question", "")
                # Store for caller to send; also add to pending_messages
                pending_messages.append(question)
                logger.info(
                    "request_clarification: question=%r for user %s",
                    question,
                    user["_id"],
                )

            elif action_type == "update_interaction_log":
                if log_id is not None:
                    fields = params.get("fields") or {}
                    if fields:
                        interaction_logs.update_one(
                            {"_id": log_id},
                            {"$set": fields},
                        )
                        logger.info(
                            "update_interaction_log executed: fields=%s", list(fields.keys())
                        )

            elif action_type == "update_activity_slot":
                slot = params.get("slot")
                if slot:
                    users.update_one(
                        {"_id": user["_id"]},
                        {
                            "$set": {
                                "current_activity": slot,
                                "activity_slot_updated_at": datetime.utcnow(),
                            }
                        },
                    )
                    logger.info(
                        "update_activity_slot executed: slot=%s for user %s",
                        slot,
                        user["_id"],
                    )

            elif action_type == "store_next_day_plan":
                plan_text = params.get("plan_text", "")
                date = params.get("date", "")
                users.update_one(
                    {"_id": user["_id"]},
                    {
                        "$set": {
                            "next_day_plan": {
                                "date": date,
                                "plan_text": plan_text,
                                "confirmed": False,
                            }
                        }
                    },
                )
                logger.info(
                    "store_next_day_plan executed: date=%s for user %s",
                    date,
                    user["_id"],
                )

            elif action_type == "confirm_next_day_plan":
                users.update_one(
                    {"_id": user["_id"]},
                    {"$set": {"next_day_plan.confirmed": True}},
                )
                logger.info(
                    "confirm_next_day_plan executed for user %s", user["_id"]
                )

            elif action_type == "call_tool":
                await _exec_call_tool(params, user, decision)

            elif action_type == "send_telegram":
                text = params.get("text", "")
                # Actually send it immediately via the bot
                _push_telegram(user, text)
                pending_messages.append(text)
                logger.info(
                    "send_telegram sent: text=%r for user %s",
                    text[:80],
                    user["_id"],
                )

            else:
                # Auto-promote: if the LLM used a tool name directly as action type
                # (e.g. "call_user" instead of "call_tool" + tool_name), fix it.
                tool_map = {t.name: t for t in ALL_TOOLS}
                if action_type in tool_map:
                    logger.info(
                        "Auto-promoting action type %r to call_tool for user %s",
                        action_type, user["_id"],
                    )
                    await _exec_call_tool(
                        {"tool_name": action_type, "tool_args": params},
                        user, decision,
                    )
                else:
                    logger.warning(
                        "Unknown action type %r for user %s — skipping",
                        action_type,
                        user["_id"],
                    )

        except Exception as exc:  # noqa: BLE001 — Task 13.3: log and continue
            logger.error(
                "Action %r failed for user %s with params %s: %s",
                action_type,
                user["_id"],
                params,
                exc,
                exc_info=True,
            )
            # Continue with next action — never stop the loop


def _exec_increment_streak(user: dict, users_collection: Any) -> None:
    """Increment streak_count and update longest_streak if new high."""
    # Increment streak_count atomically
    result = users_collection.find_one_and_update(
        {"_id": user["_id"]},
        {"$inc": {"streak_count": 1}},
        return_document=True,
    )
    if result is None:
        # Fallback: plain update
        users_collection.update_one(
            {"_id": user["_id"]},
            {"$inc": {"streak_count": 1}},
        )
        result = users_collection.find_one({"_id": user["_id"]}) or {}

    new_streak = result.get("streak_count", 0)
    longest = result.get("longest_streak", 0)

    if new_streak > longest:
        users_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"longest_streak": new_streak}},
        )
        logger.info(
            "increment_streak: new streak=%d (new longest) for user %s",
            new_streak,
            user["_id"],
        )
    else:
        logger.info(
            "increment_streak: new streak=%d for user %s",
            new_streak,
            user["_id"],
        )


async def _exec_call_tool(
    params: dict,
    user: dict,
    decision: LLMDecision | None,
) -> str:
    """
    Invoke a named LangChain tool with the provided args.
    Auto-injects user_id if the tool expects it and it's missing.
    Writes an ai_tool_calls audit document (fire-and-forget).
    Returns the tool result string so callers can feed it back to the LLM.
    """
    tool_name = params.get("tool_name", "")
    tool_args = dict(params.get("tool_args") or {})
    model_used = decision.model_used if decision else ""

    # Auto-inject user_id — the LLM sometimes forgets it
    if "user_id" not in tool_args or not tool_args["user_id"]:
        tool_args["user_id"] = str(user["_id"])

    # Build a lookup map from tool name to tool function
    tool_map = {t.name: t for t in ALL_TOOLS}

    result_str = f"[error] tool {tool_name!r} not found"
    if tool_name in tool_map:
        try:
            result = await tool_map[tool_name].ainvoke(tool_args)
            result_str = str(result)
            logger.info("Tool %r succeeded: %s", tool_name, result_str[:100])
        except Exception as exc:  # noqa: BLE001
            result_str = f"[error] {exc}"
            logger.error("Tool %r raised: %s", tool_name, exc)
    else:
        logger.warning("call_tool: unknown tool %r", tool_name)

    # Audit log (fire-and-forget)
    _log_tool_call(
        user=user,
        tool_name=tool_name,
        tool_input=tool_args,
        tool_output=result_str,
        model_used=model_used,
    )
    return result_str


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


class ChanakyaAgent:
    """
    LangChain agent with native tool calling and conversation memory.

    Uses gpt-5-mini with bind_tools() for structured tool invocation.
    Maintains conversation history via chat_messages collection.
    """

    def __init__(self, user: dict) -> None:
        self.user = user
        self.assembler = ContextAssembler()

    async def invoke(
        self,
        raw_input: str,
        interaction_type: str,
        session_context: dict | None = None,
        media_url: str | None = None,
    ) -> LLMDecision | None:
        """
        Full agent invocation pipeline with native tool calling.

        1. Build context + conversation history
        2. LLM reasons with bound tools (native function calling)
        3. Execute tool calls, feed results back as ToolMessages
        4. After tools complete, extract final LLMDecision
        Max 5 tool rounds to prevent infinite loops.
        """
        user = self.user

        # Step 1: Assemble context
        try:
            context = await self.assembler.build(user, interaction_type, session_context)
            context = scrub_recursive(context, user["_id"])
        except Exception as exc:
            logger.error("Context assembly failed for user %s: %s", user.get("_id"), exc)
            context = {"tier1": {}, "tier2": None, "tier3": None, "tier4": None, "prompt_templates": {}}

        system_prompt = _build_system_prompt(context, user)

        # Step 2: Build conversation history
        scrubbed_input = scrub_context(raw_input, user["_id"])

        from chanakya.db.mongo import get_recent_messages, get_message_count

        recent_msgs = get_recent_messages(user["_id"], limit=5)
        conversation_summary = user.get("conversation_summary") or ""

        # Compress if history is growing large
        msg_count = get_message_count(user["_id"])
        if msg_count > 10 and not conversation_summary:
            from chanakya.db.mongo import get_recent_messages as _get
            older_msgs = _get(user["_id"], limit=15)
            if len(older_msgs) > 5:
                to_compress = older_msgs[:-5]
                conversation_summary = await _compress_history(user, to_compress)

        # Build message array: system → summary → history → current
        messages = [SystemMessage(content=system_prompt)]

        if conversation_summary:
            messages.append(SystemMessage(
                content=f"CONVERSATION HISTORY SUMMARY (older messages):\n{conversation_summary}"
            ))

        for msg in recent_msgs:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        # Build final user message — multimodal if image attached
        if media_url:
            user_msg = HumanMessage(content=[
                {"type": "text", "text": scrubbed_input or "Here is the image."},
                {"type": "image_url", "image_url": {"url": media_url, "detail": "high"}},
            ])
        else:
            user_msg = HumanMessage(content=scrubbed_input)
        messages.append(user_msg)

        # Step 3: Native tool calling loop
        model_used: str = MODEL
        llm = _make_llm()
        llm_with_tools = llm.bind_tools(ALL_TOOLS)

        _t = Timer()

        for round_num in range(1, 6):
            try:
                response = await llm_with_tools.ainvoke(messages)
                _log_llm_attempt(user, MODEL, interaction_type, "success")
                log_llm(
                    str(user.get("_id")), MODEL,
                    f"{interaction_type}:round{round_num}",
                    scrubbed_input[:200], (response.content or "")[:200],
                    latency_ms=_t.elapsed_ms(),
                )
            except Exception as exc:
                logger.error("LLM round %d failed for user %s: %s", round_num, user.get("_id"), exc)
                _log_llm_attempt(user, MODEL, interaction_type, str(exc))
                return None

            # Check for tool calls
            if not response.tool_calls:
                break

            # Execute each tool call and add results as ToolMessages
            messages.append(response)
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = dict(tool_call["args"])

                # Auto-inject user_id
                if "user_id" not in tool_args or not tool_args["user_id"]:
                    tool_args["user_id"] = str(user["_id"])

                result_str = await _exec_call_tool(
                    {"tool_name": tool_name, "tool_args": tool_args},
                    user, None,
                )
                messages.append(ToolMessage(
                    content=result_str,
                    tool_call_id=tool_call["id"],
                ))
                logger.info("Tool %r executed in round %d for user %s",
                            tool_name, round_num, user.get("_id"))

            _t = Timer()
        else:
            logger.warning("Max tool rounds reached for user %s", user.get("_id"))

        # Step 4: Extract final decision
        # The last response should contain the conversational reply
        final_text = response.content or ""

        # Ask for structured decision
        messages.append(HumanMessage(content=_DECISION_PROMPT))
        try:
            decision_response = await llm.ainvoke(messages)
            decision = _parse_llm_decision(decision_response.content or "", model_used)
        except Exception as exc:
            logger.error("Decision extraction failed for user %s: %s", user.get("_id"), exc)
            decision = None

        if decision is None:
            # Fallback: use the conversational response directly
            decision = LLMDecision(
                verdict=None,
                response_text=final_text,
                tone="NEUTRAL",
                reasoning="Direct response (structured parsing failed)",
                model_used=model_used,
            )
        elif not decision.response_text and final_text:
            decision.response_text = final_text

        decision.model_used = model_used

        # Privacy Scrubbing: Re-identify names
        if decision.response_text:
            decision.response_text = unscrub_response(decision.response_text, user["_id"])

        # Execute non-tool actions (streak changes, etc.)
        pending_messages: list[str] = []
        await execute_actions(
            actions=decision.actions,
            user=user,
            log_id=None,
            decision=decision,
            pending_messages=pending_messages,
        )

        return decision
