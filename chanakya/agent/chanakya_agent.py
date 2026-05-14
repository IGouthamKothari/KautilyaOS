"""
chanakya_agent.py — LangChain agent wrapper.

Uses gpt-5.4-nano-2026-03-17 directly via the OpenAI API.
Returns a structured LLMDecision on every invocation.

Architecture contract (Req 25 — LLM-as-Brain):
  - Server assembles context → sends to LLM → LLM returns LLMDecision
  - Server NEVER makes decisions based on metric thresholds
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

from chanakya.agent.context_assembler import ContextAssembler
from chanakya.agent.privacy_scrubber import (
    get_scrub_list,
    scrub_context,
    scrub_recursive,
    unscrub_response,
)
from chanakya.config import OPENAI_API_KEY, LLM_MODEL_NAME
from chanakya.io_logger import Timer, log_llm
from chanakya.models.llm_decision import ActionItem, LLMDecision
from chanakya.tools.schedule_tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM model — loaded from config
# ---------------------------------------------------------------------------

MODEL = LLM_MODEL_NAME


def _make_llm(model: str = MODEL) -> ChatOpenAI:
    """Create a ChatOpenAI instance pointing at OpenAI."""
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=model,
        temperature=0.7,
        max_completion_tokens=2048,
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

    # Typed mindset/identity entries — injected by category
    identity = tier1.get("identity_context") or {}
    mindset_entries = identity.get("mindset") or []
    if mindset_entries:
        # Group by category
        from collections import defaultdict
        by_cat: dict = defaultdict(list)
        for entry in mindset_entries:
            if isinstance(entry, dict):
                by_cat[entry.get("category", "note")].append(entry)

        category_labels = {
            "quote":     "QUOTES TO EMBODY",
            "goal":      "LIFE GOALS",
            "trait":     "CHARACTER TRAITS TO BUILD",
            "rule":      "PERSONAL RULES",
            "reference": "REFERENCES — USE THESE WHEN RELEVANT",
            "note":      "ADDITIONAL CONTEXT",
        }
        sections.append("")
        sections.append("=== GOUTHAM'S MINDSET & IDENTITY (live in this, always) ===")
        for cat, label in category_labels.items():
            entries = by_cat.get(cat, [])
            if not entries:
                continue
            sections.append(f"\n[{label}]")
            for e in entries:
                text = e.get("text", "")
                source = e.get("source", "")
                line = f"  • {text}"
                if source:
                    line += f"  — {source}"
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

    tool_list = "\n".join(f"  - {t.name}: {(t.description or '').splitlines()[0]}" for t in ALL_TOOLS)

    sections.append(f"""
=== HOW TO CALL TOOLS ===
When you need data or want to take an action, include a call_tool action in your response.
The server will execute it, then call you again with the result so you can respond properly.

TOOL CALL FORMAT — always use this exact structure:
  {{"type": "call_tool", "params": {{"tool_name": "<name>", "tool_args": {{...}}}}}}

The user_id for all tool calls is: {user_id_str}

AVAILABLE TOOLS:
{tool_list}

TOOL USAGE RULES:
- Schedule queries → fetch_day_schedule(user_id, date) — use "today"/"tomorrow"/YYYY-MM-DD
- Past date log → get_day_log(user_id, date)
- Add one-off event/reminder → add_day_event(user_id, date, time_str, activity, display_name, description, note)
  • activity: CAPS_UNDERSCORE label e.g. "MEETING_KARTIK"
  • display_name: clean readable name e.g. "Meeting with Kartik" — always derive from user's words
  • description: 1-2 sentences on what this is and why it matters — always generate from context
  • BEFORE adding: call fetch_day_schedule to check for conflicts at that time
  • If it conflicts with LeetCode, gym, deep work, or sleep — flag it. Don't silently accept it.
- Reschedule by name → reschedule_activity(user_id, activity, new_time, date) — PREFERRED for time changes
  • Use this when user says "move gym to 7:30", "shift wake up to 7am", "gym later at 7:30"
  • No event_id needed — finds by activity name automatically
  • date defaults to "today", supports "tomorrow" or YYYY-MM-DD
- Modify this date only → update_day_event(event_id, field, value, scope="this_date") — use when you have the id from fetch_day_schedule
- Modify all weekdays → update_day_event(..., scope="all_weekdays") OR update_schedule_activity
- Save contact → save_contact(user_id, name, phone, relationship)
- Call user → call_user(user_id)
- Proxy call → place_proxy_call(user_id, contact_name, topic)
- Status → get_user_status(user_id)
- Mindset → add_mindset_note / add_mindset_entry / get_mindset_notes / remove_mindset_note / clear_mindset_notes
  • add_mindset_entry(user_id, category, text, source) — use for quotes, goals, traits, rules, references
  • category: "quote" | "goal" | "trait" | "rule" | "reference" | "note"
  • When user shares a quote, goal, or principle → detect it and call add_mindset_entry automatically
  • source: who said it (e.g. "Harvey Specter", "Bhagavad Gita 2.47")
- War mode on → activate_war_mode(user_id, duration_hours)
- War mode off → deactivate_war_mode(user_id)
- Phone → set_user_phone(user_id, phone)
- Push message now → send_telegram_message(user_id, message) — send proactively at any time
- Schedule message → schedule_message(user_id, message, send_at) — send at HH:MM today

=== RESPONSE FORMAT ===
Always respond with a single valid JSON object — no markdown, no code fences, just raw JSON:
{{
  "verdict": "SUCCESS" | "FAILED" | "EXCUSED" | "WAR_MODE_OVERRIDE" | null,
  "actions": [
    {{"type": "call_tool", "params": {{"tool_name": "...", "tool_args": {{...}}}}}}
  ],
  "tone": "HARSH" | "MENTOR" | "NEUTRAL" | "CELEBRATORY",
  "response_text": "message to send to user",
  "reasoning": "your internal reasoning",
  "streak_reset": false,
  "model_used": "{MODEL}"
}}

Other valid action types (besides call_tool):
  increment_streak, reset_streak, update_longest_streak, apply_punishment,
  request_clarification, update_interaction_log, update_activity_slot,
  store_next_day_plan, confirm_next_day_plan, send_telegram

FORMATTING for response_text:
- **bold** for emphasis, _italic_ for secondary
- Plain bullet: • (not - or *)
- No HTML tags, no # headings
- Schedule lines: **HH:MM** — ACTIVITY (action, priority)

=== CORE RULES ===
1. Never say "It's okay." It is NOT okay. Never say "great" or "absolutely". You are not a chatbot.
2. You decide ALL verdicts, streak changes, escalations, tone. Server executes.
3. Reference the Gita, Mahabharata, Ramayana, and Indian warrior history when it lands with force — not as decoration.
4. **Temporal Awareness**: You are aware of the Current Time. It is your primary instrument of discipline. If the user is awake and active at an hour that contradicts their Sleep Protocol (e.g., 1:00 AM), your response_text MUST address this immediately. Ask for the reason, cite the lack of discipline, and pivot back to the dharma only after the warning is delivered.
5. Be harsh when failure is repeated. Acknowledge genuine effort — briefly, then raise the bar.
5. NEVER tell the user to use a slash command. Handle everything via tools.
6. When you need data to answer (schedule, status, contacts, logs) — call the tool first.
   The server will give you the result and ask you to respond again.
7. If the user says "call me" → call_user. If "call mom about X" → place_proxy_call.
8. War Mode trigger word → activate_war_mode for 24 hours.
9. You are a GURU, not an assistant. A guru guides, cautions, and stops the student when wrong.
   - If the user mentions a plan that conflicts with their goals, say so directly.
   - If they're about to waste time, call it out before storing anything.
   - If a meeting or event cuts into LeetCode, gym, or deep work — flag the conflict explicitly.
   - Never just silently execute. Always respond with your honest assessment first.
10. When the user mentions any event, meeting, or plan in natural language:
    - Extract: who, what, when, why
    - Generate a clean display_name (e.g. "Meeting with Kartik") and description (what it is, why it matters)
    - Check the schedule for that date/time first using fetch_day_schedule
    - If there's a conflict, tell the user what they're giving up — then ask if they still want to proceed
    - Only call add_day_event after this assessment
11. Brutal honesty is non-negotiable. If the user is making a mistake, say it plainly.
    Krishna did not soften the truth for Arjuna on the battlefield. Neither do you.
    "It is better to perform one's own dharma imperfectly than to perform another's perfectly." — Gita 3.35
    They all had one thing in common: they did not stop.
13. **Privacy Guard (The Fortress)**: If the user mentions a new person, organization, or sensitive entity not listed in the 'DE-IDENTIFIED ENTITIES' section:
    - Acknowledge the person/entity warmly in your response.
    - Add a polite but firm nudge: "I've noticed you mentioned [Name]. Shall I add them to your private contacts? This ensures they are de-identified in my cloud brain, keeping your 'Privacy Fortress' airtight."
    - Only call `save_contact` after the user confirms or explicitly asks you to remember them.
14. **Ritual Discipline (Health Tracking)**: You are responsible for the user's vessel (the body/mind).
    - **Morning Todo**: You MUST ask for the previous night's SLEEP (hours) and current ENERGY/MOOD (1-10).
    - **EOD Report**: Ask for a final assessment of MOOD and ENERGY.
    - **Proactive Adjustment**: Check `last_rituals` in the context. If you see low energy or bad sleep for 2+ days, adjust the plan (e.g. "You are running on empty. Skip the late-night LeetCode; prioritize 8 hours of sleep. Dharma requires a sharp blade.")
    - Use `log_ritual` whenever they share health stats.
15. **The Council of Dharma (Delegation)**: You are the Prime Minister, but you are not alone. You have a Cabinet of Experts:
    - **Kautilya (Finance)**: For strategy, wealth, and ledger analysis.
    - **Charaka (Health)**: For bio-hacking, sleep, and ritual trends.
    - **Vishvakarma (Technology)**: For codebase, architecture, and engineering.
    - If a query requires deep domain expertise, use `consult_council`. Provide a clear 'Briefing' to the expert.
    - Synthesize their 'Council Report' into your final response. Never parrot them blindly; you are the final authority.
    - You must include the user's ID when calling the council.
""".replace("{user_id}", user_id_str))

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# LLM response parsing (Task 13.1)
# ---------------------------------------------------------------------------


def _parse_llm_decision(raw_response: str, model_used: str) -> LLMDecision | None:
    """
    Parse the LLM's raw text response into an LLMDecision.

    Strategy:
      1. Try to parse the full response as JSON.
      2. Find all top-level JSON objects in the response, try each from largest to smallest.
         This handles the case where the LLM outputs a stray tool-call object before the
         main LLMDecision object (a known gpt-5.4-nano quirk).
      3. If parsing fails: log raw response, return None.
    """
    def _try_parse(text: str) -> LLMDecision | None:
        try:
            data = json.loads(text.strip())
            decision = LLMDecision(**data)
            if not decision.model_used:
                decision.model_used = model_used
            return decision
        except Exception:
            return None

    # Step 1: try full response as JSON
    result = _try_parse(raw_response)
    if result:
        return result

    # Step 2: find all top-level JSON objects, try largest first
    # Walk the string tracking brace depth to extract each complete {...} block
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

    # Sort by length descending — the LLMDecision object is always the largest
    for candidate in sorted(candidates, key=len, reverse=True):
        result = _try_parse(candidate)
        if result:
            return result

    # Step 3: parsing failed
    logger.error(
        "Failed to parse LLMDecision from raw response. Raw: %r",
        raw_response[:500],
    )
    return None


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
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_send())
        else:
            loop.run_until_complete(_send())
    except Exception as exc:
        logger.error("push_telegram scheduling failed: %s", exc)


# ---------------------------------------------------------------------------
# Action execution (Task 13.2)
# ---------------------------------------------------------------------------


def execute_actions(
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
                _exec_call_tool(params, user, decision)

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

            elif action_type == "send_voice":
                text = params.get("text", "")
                if text:
                    async def _send_v():
                        try:
                            from chanakya.integrations.elevenlabs_client import ElevenLabsClient
                            from chanakya.config import ELEVENLABS_DEFAULT_VOICE_ID, TELEGRAM_BOT_TOKEN
                            from telegram import Bot
                            import io

                            voice_id = user.get("elevenlabs_voice_id") or ELEVENLABS_DEFAULT_VOICE_ID
                            client = ElevenLabsClient()
                            audio_bytes = client.synthesise(text, voice_id)
                            
                            bot = Bot(token=TELEGRAM_BOT_TOKEN)
                            chat_id = user.get("telegram_id")
                            if chat_id:
                                await bot.send_voice(chat_id=chat_id, voice=io.BytesIO(audio_bytes))
                                logger.info("send_voice executed for user %s: %s", user["_id"], text[:50])
                        except Exception as e:
                            logger.error("Failed to send voice note to %s: %s", user["_id"], e)
                    
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.ensure_future(_send_v())
                        else:
                            asyncio.run(_send_v())
                    except Exception as e:
                        logger.error("Failed to schedule send_voice for %s: %s", user["_id"], e)
                
                pending_messages.append(f"[Voice Note: {text}]")

            else:
                # Auto-promote: if the LLM used a tool name directly as action type
                # (e.g. "call_user" instead of "call_tool" + tool_name), fix it.
                tool_map = {t.name: t for t in ALL_TOOLS}
                if action_type in tool_map:
                    logger.info(
                        "Auto-promoting action type %r to call_tool for user %s",
                        action_type, user["_id"],
                    )
                    _exec_call_tool(
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


def _exec_call_tool(
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
            result = tool_map[tool_name].invoke(tool_args)
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
    LangChain agent wrapper for Chanakya.

    Instantiated fresh per interaction (stateless).
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
        Full agent invocation pipeline with multi-turn tool loop.

        Round 1: LLM sees user message → may request tool calls
        Round 2+: Server runs tools, feeds results back → LLM responds with data
        Max 3 tool rounds to prevent infinite loops.
        Final round: execute non-tool actions, return decision.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        user = self.user

        # Tools that return data the LLM needs to formulate a response
        DATA_TOOLS = {
            "fetch_schedule", "fetch_day_schedule", "get_day_log",
            "list_contacts", "get_user_status", "get_mindset_notes",
        }
        # Tools that take action and should confirm to the LLM
        ACTION_TOOLS = {
            "update_schedule_activity", "add_day_event", "update_day_event",
            "delete_day_event", "save_contact", "delete_contact",
            "set_user_phone", "place_proxy_call", "call_user",
            "add_mindset_note", "remove_mindset_note", "clear_mindset_notes",
            "set_morning_todo_time", "reload_prompt_templates",
            "activate_war_mode", "deactivate_war_mode",
            "modify_wakeup_time", "add_daily_checkpoint",
            "update_morning_todo_time", "escalate_punishment",
            "send_telegram_message", "schedule_message",
            "add_mindset_entry", "cancel_scheduled_message",
            "reschedule_activity",
        }
        ALL_FEEDBACK_TOOLS = DATA_TOOLS | ACTION_TOOLS

        # Step 1: Assemble context
        try:
            context = self.assembler.build(user, interaction_type, session_context)
            # Privacy Scrubbing: De-identify names/PII before they hit the cloud
            context = scrub_recursive(context, user["_id"])
        except Exception as exc:
            logger.error("Context assembly failed for user %s: %s", user.get("_id"), exc)
            context = {"tier1": {}, "tier2": None, "tier3": None, "tier4": None, "prompt_templates": {}}

        system_prompt = _build_system_prompt(context, user)
        # Privacy Scrubbing: De-identify user's raw input
        scrubbed_input = scrub_context(raw_input, user["_id"])
        human_message = scrubbed_input
        if media_url:
            human_message = f"{raw_input}\n[Media URL: {media_url}]"

        model_used: str = MODEL
        decision: LLMDecision | None = None
        tool_map = {t.name: t for t in ALL_TOOLS}
        tool_map_names = set(tool_map.keys())

        # Multi-turn tool loop — max 3 rounds
        current_human = human_message
        for round_num in range(1, 4):
            # LLM call
            raw_response: str | None = None
            try:
                llm = _make_llm()
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=current_human),
                ]
                _t = Timer()
                response = llm.invoke(messages)
                raw_response = response.content
                _log_llm_attempt(user, MODEL, interaction_type, "success")
                log_llm(
                    str(user.get("_id")), MODEL,
                    f"{interaction_type}:round{round_num}",
                    current_human, raw_response,
                    latency_ms=_t.elapsed_ms(),
                )
                logger.info("LLM round %d succeeded for user %s", round_num, user.get("_id"))
            except Exception as exc:
                logger.error("LLM round %d failed for user %s: %s", round_num, user.get("_id"), exc)
                _log_llm_attempt(user, MODEL, interaction_type, str(exc))
                break

            if raw_response is None:
                break

            decision = _parse_llm_decision(raw_response, model_used)
            if decision is None:
                break

            # Collect tool calls from this round
            tool_calls_this_round: list[tuple[str, dict]] = []  # (tool_name, tool_args)
            non_tool_actions: list[ActionItem] = []

            for action in decision.actions:
                atype = action.type
                aparams = action.params or {}

                if atype == "call_tool":
                    tname = aparams.get("tool_name", "")
                    targs = aparams.get("tool_args") or {}
                    tool_calls_this_round.append((tname, targs))
                elif atype in tool_map_names:
                    # LLM used tool name directly as action type — auto-promote
                    tool_calls_this_round.append((atype, aparams))
                else:
                    non_tool_actions.append(action)

            if not tool_calls_this_round:
                # No tools needed — final answer
                decision.actions = non_tool_actions
                break

            if round_num == 3:
                # Last round — execute tools but don't loop again
                for tname, targs in tool_calls_this_round:
                    _exec_call_tool({"tool_name": tname, "tool_args": targs}, user, decision)
                decision.actions = non_tool_actions
                break

            # Execute tools and collect results for next round
            tool_results: list[str] = []
            for tname, targs in tool_calls_this_round:
                result_str = _exec_call_tool({"tool_name": tname, "tool_args": targs}, user, decision)
                if tname in ALL_FEEDBACK_TOOLS:
                    tool_results.append(f"[{tname}]\n{result_str}")
                    logger.info("Tool %r result collected for round %d", tname, round_num + 1)

            if not tool_results:
                # All tools were fire-and-forget, no feedback needed
                decision.actions = non_tool_actions
                break

            # Build next round's human message with tool results
            current_human = (
                f"Original request: {raw_input}\n\n"
                f"Tool results from round {round_num}:\n"
                + "\n\n".join(tool_results)
                + "\n\nNow respond to the user using this data. "
                "Return a valid LLMDecision JSON."
            )
            decision.actions = non_tool_actions  # carry forward in case loop ends

        if decision is None:
            return None

        # Privacy Scrubbing: Re-identify names in the final response back to the user
        if decision.response_text:
            decision.response_text = unscrub_response(decision.response_text, user["_id"])

        # Execute remaining non-tool actions
        pending_messages: list[str] = []
        execute_actions(
            actions=decision.actions,
            user=user,
            log_id=None,
            decision=decision,
            pending_messages=pending_messages,
        )

        return decision
