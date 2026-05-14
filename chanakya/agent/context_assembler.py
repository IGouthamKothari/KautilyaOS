"""
context_assembler.py — Tiered context assembly for LLM calls.

Builds a structured Python dict from MongoDB data across up to four tiers:
  Tier 1 — Core Identity (always included, includes rolling conversation context)
  Tier 2 — Daily Context (non-trivial interactions)
  Tier 3 — Historical Pattern Context (CHECK_IN, EOD, ESCALATION, MENTOR_TALK)
  Tier 4 — Deep Memory (EOD, WEEKLY_REVIEW, explicit agent request)

Never passes raw MongoDB documents, ObjectIds, or embedding vectors to the LLM.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import pytz

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared display helper — CAPS_UNDERSCORE → "Title Case"
# ---------------------------------------------------------------------------

def humanize(text: str) -> str:
    """Convert CAPS_UNDERSCORE identifiers to readable Title Case.

    Examples:
        EOD_REPORT        → "End of Day Report"
        WAKE_UP           → "Wake Up"
        LEETCODE_HOUR     → "Leetcode Hour"
        OFFICE_WORK       → "Office Work"
        FREE_TIME         → "Free Time"
        MORNING_TODO      → "Morning Todo"
        SLEEP_PROTOCOL    → "Sleep Protocol"
        SYSTEM_DESIGN_OR_READING → "System Design or Reading"
    """
    _SPECIAL: dict[str, str] = {
        "EOD_REPORT":               "End of Day Report",
        "EOD":                      "End of Day",
        "MORNING_TODO":             "Morning Todo",
        "MORNING_TODO_DELIVERY":    "Morning Todo Delivery",
        "WAKE_UP":                  "Wake Up",
        "POST_GYM_ROUTINE":         "Post Gym Routine",
        "PERSONAL_TIME":            "Personal Time",
        "OFFICE_WORK":              "Office Work",
        "MIDDAY_CHECKPOINT":        "Midday Check",
        "LUNCH_BREAK":              "Lunch Break",
        "LEETCODE_HOUR":            "Leetcode Hour",
        "LEETCODE_DOUBLE":          "Leetcode Double Session",
        "LEETCODE_OPTIONAL":        "Leetcode (Optional)",
        "BOUNDARY_CHECK":           "Boundary Check",
        "SYSTEM_DESIGN_OR_READING": "System Design or Reading",
        "SLEEP_PROTOCOL":           "Sleep Protocol",
        "SKILL_DEVELOPMENT":        "Skill Development",
        "EVENING_CHECKPOINT":       "Evening Check",
        "NEXT_WEEK_PREP":           "Next Week Prep",
        "WEEKLY_REVIEW":            "Weekly Review",
        "LIFE_ADMIN":               "Life Admin",
        "REST_OR_HOBBY":            "Rest or Hobby",
        "FREE_TIME":                "Free Time",
        "CHECK_IN":                 "Check In",
        "MENTOR_TALK":              "Mentor Talk",
        "ESCALATION":               "Escalation",
        "CHECKPOINT":               "Checkpoint",
        "COMMAND_RESPONSE":         "Command Response",
        "WAR_MODE":                 "War Mode",
        "NORMAL":                   "Normal",
        "AWAY":                     "Away",
        "INJURED":                  "Injured",
        "IMAGE_DEMAND":             "Photo Check",
        "TELEGRAM_TEXT":            "Message",
        "TELEGRAM_VOICE":           "Voice Message",
        "CALL":                     "Call",
    }
    if text in _SPECIAL:
        return _SPECIAL[text]
    # Generic fallback: split on underscore, title-case each word
    return " ".join(w.capitalize() for w in text.split("_"))


# ---------------------------------------------------------------------------
# Tier inclusion rules
# ---------------------------------------------------------------------------

# interaction_types that include each tier
_TIER2_TYPES = {
    "CHECKPOINT",
    "MORNING_TODO",
    "CHECK_IN",
    "ESCALATION",
    "MENTOR_TALK",
    "EOD",
    "WEEKLY_REVIEW",
}

_TIER3_TYPES = {
    "CHECK_IN",
    "ESCALATION",
    "MENTOR_TALK",
    "EOD",
    "WEEKLY_REVIEW",
}

_TIER4_TYPES = {
    "EOD",
    "WEEKLY_REVIEW",
}

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class NoTemplateFoundError(Exception):
    def __init__(self, activity_slot: str, interaction_type: str):
        super().__init__(
            f"No prompt template found for activity_slot={activity_slot!r}, "
            f"interaction_type={interaction_type!r} (including FREE_TIME and GENERIC fallbacks)"
        )


# ---------------------------------------------------------------------------
# Module-level template cache
# ---------------------------------------------------------------------------

_template_cache: dict[tuple, dict] = {}


def clear_template_cache() -> None:
    """Clear the in-memory prompt template cache. Called by /reloadtemplates."""
    global _template_cache
    _template_cache.clear()
    logger.info("Prompt template cache cleared.")


# ---------------------------------------------------------------------------
# Rolling conversation context — cross-channel awareness
# ---------------------------------------------------------------------------

_MAX_CONTEXT_CHARS = 400  # keep summaries tight — this goes into every LLM call


async def update_conversation_context(
    user: dict,
    role: str,          # "user" or "assistant"
    content: str,
    channel: str = "text",  # "text" | "call"
) -> None:
    """Compress the latest exchange into the rolling conversation_context summary.

    Called after every text reply and after every call turn so that the next
    interaction — regardless of channel — starts with full awareness of what
    was just discussed.

    The summary is stored on users.conversation_context (plain string, ≤ 400 chars).
    On any failure the existing summary is preserved unchanged.

    Args:
        user:    The user document dict (must have _id).
        role:    Who spoke — "user" or "assistant".
        content: What was said / written.
        channel: "text" or "call" — included in the summary for channel awareness.
    """
    import httpx
    from chanakya.config import OPENAI_API_KEY
    from chanakya.db.mongo import users as users_col

    from chanakya.agent.privacy_scrubber import scrub_context, unscrub_response
    
    existing_summary = user.get("conversation_context") or ""
    # Privacy Scrubbing: De-identify data before summarization in the cloud
    existing_summary_scrubbed = scrub_context(existing_summary, user["_id"])
    content_scrubbed = scrub_context(content, user["_id"])

    channel_label = "📞 call" if channel == "call" else "💬 text"
    new_turn = f"[{channel_label}] {role}: {content_scrubbed[:300]}"

    prompt = (
        "You maintain a rolling conversation summary for an AI accountability coach.\n"
        "Keep the summary under 400 words. Focus on: what was discussed, "
        "decisions made, open questions, and current emotional/motivational state.\n"
        "Drop old details when space runs out — keep only what's relevant NOW.\n\n"
        f"EXISTING SUMMARY:\n{existing_summary_scrubbed or '(none yet)'}\n\n"
        f"NEW TURN:\n{new_turn}\n\n"
        "Return ONLY the updated summary. No explanation. No quotes."
    )

    try:
        from chanakya.config import OPENAI_API_KEY, LLM_MODEL_NAME
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": 120,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            new_summary = resp.json()["choices"][0]["message"]["content"].strip()
            
            # Privacy: Re-identify names before storing in our private DB
            new_summary = unscrub_response(new_summary, user["_id"])
            
            # Hard-cap to avoid drift
            new_summary = new_summary[:_MAX_CONTEXT_CHARS]

        users_col.update_one(
            {"_id": user["_id"]},
            {"$set": {"conversation_context": new_summary}},
        )
        # Update in-place so the caller's dict reflects the new value immediately
        user["conversation_context"] = new_summary
        logger.debug(
            "conversation_context updated for user %s: %r", user["_id"], new_summary[:80]
        )
    except Exception as exc:
        logger.warning(
            "Failed to update conversation_context for user %s: %s — keeping existing.",
            user["_id"],
            exc,
        )


# ---------------------------------------------------------------------------
# Timezone helper
# ---------------------------------------------------------------------------


def _get_timezone(tz_str: str) -> pytz.BaseTzInfo:
    """Return a pytz timezone, falling back to Asia/Kolkata on invalid input."""
    try:
        return pytz.timezone(tz_str)
    except pytz.UnknownTimeZoneError:
        logger.warning(
            "Unknown timezone %r; falling back to Asia/Kolkata.", tz_str
        )
        return pytz.timezone("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Failure count helpers
# ---------------------------------------------------------------------------


def _compute_weekly_failures(user: dict) -> int:
    """
    Count FAILED verdicts in interaction_logs since Monday 00:00 local time.

    Uses the user's IANA timezone to determine the week boundary.
    Returns a plain int — no ObjectIds, no datetimes.
    """
    from chanakya.db.mongo import interaction_logs  # lazy import to allow mocking

    tz = _get_timezone(user.get("timezone", "Asia/Kolkata"))
    now_local = datetime.now(tz)
    days_since_monday = now_local.weekday()  # 0 = Monday
    week_start_local = now_local.replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - timedelta(days=days_since_monday)
    week_start_utc = week_start_local.astimezone(pytz.utc).replace(tzinfo=None)

    count = interaction_logs.count_documents(
        {
            "user_id": user["_id"],
            "ai_evaluation.verdict": "FAILED",
            "timestamp": {"$gte": week_start_utc},
        }
    )
    return int(count)


def _compute_monthly_failures(user: dict) -> int:
    """
    Count FAILED verdicts in interaction_logs since the 1st of the current month
    at 00:00 local time.

    Returns a plain int.
    """
    from chanakya.db.mongo import interaction_logs  # lazy import to allow mocking

    tz = _get_timezone(user.get("timezone", "Asia/Kolkata"))
    now_local = datetime.now(tz)
    month_start_local = now_local.replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    month_start_utc = month_start_local.astimezone(pytz.utc).replace(tzinfo=None)

    count = interaction_logs.count_documents(
        {
            "user_id": user["_id"],
            "ai_evaluation.verdict": "FAILED",
            "timestamp": {"$gte": month_start_utc},
        }
    )
    return int(count)


# ---------------------------------------------------------------------------
# Personal instructions + mindset identity helper
# ---------------------------------------------------------------------------


def _get_identity_context_for_prompt(user: dict) -> dict | None:
    """Fetch both flat instructions and typed mindset entries from MongoDB.

    Returns a dict with 'instructions' and 'mindset' keys, or None if empty.
    """
    try:
        from chanakya.db.mongo import get_all_identity_context
        data = get_all_identity_context(user["_id"])
        has_content = data.get("instructions") or data.get("mindset")
        return data if has_content else None
    except Exception as exc:
        logger.warning("Failed to fetch identity context: %s", exc)
        return None


def _get_personal_instructions_for_context(user: dict) -> list[str] | None:
    """Fetch personal instructions from MongoDB. Returns None if empty."""
    try:
        from chanakya.db.mongo import get_personal_instructions
        items = get_personal_instructions(user["_id"])
        return items if items else None
    except Exception as exc:
        logger.warning("Failed to fetch personal instructions: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Task 7 — Tier 1: Core Identity
# ---------------------------------------------------------------------------


def _build_tier1(user: dict) -> dict:
    """
    Build Tier 1 — Core Identity context.

    Returns a dict of JSON-serialisable primitives only.
    No ObjectIds, no datetime objects, no embedding vectors.
    """
    tz = _get_timezone(user.get("timezone", "Asia/Kolkata"))
    now_local = datetime.now(tz)

    today_date = now_local.strftime("%Y-%m-%d")
    day_of_week = now_local.strftime("%A")  # e.g. "Monday"

    # Relationship summary — safe extraction from potentially missing/non-dict field
    rel_config = user.get("relationship_config")
    if isinstance(rel_config, dict):
        partner_name = rel_config.get("partner_name")
        drain_level = rel_config.get("partner_drain_level")
    else:
        partner_name = None
        drain_level = None

    return {
        "name": user["name"],
        "streak_count": user.get("streak_count", 0),
        "longest_streak": user.get("longest_streak", 0),
        "failure_count_this_week": _compute_weekly_failures(user),
        "failure_count_this_month": _compute_monthly_failures(user),
        "current_mode": user.get("current_mode", "NORMAL"),
        "current_activity_slot": user.get("current_activity", "FREE_TIME"),
        "relationship_summary": {
            "partner_name": partner_name,
            "drain_level": drain_level,
        },
        "today_date": today_date,
        "day_of_week": day_of_week,
        "current_time": now_local.strftime("%I:%M %p"), # High-precision local time
        "timezone": user.get("timezone", "Asia/Kolkata"),
        # Rolling cross-channel conversation summary — updated after every text/call turn.
        # Gives the LLM awareness of what was discussed recently regardless of channel.
        "conversation_context": user.get("conversation_context") or None,
        # Personal instructions — editable at runtime via natural language.
        # Shapes how Chanakya interprets and responds to this user.
        "personal_instructions": _get_personal_instructions_for_context(user),
        # Typed mindset/identity entries — quotes, goals, traits, rules, references.
        # Injected into every prompt so Chanakya always knows who this person is building.
        "identity_context": _get_identity_context_for_prompt(user),
        # Health & Ritual logs (Sleep, Mood, Energy, etc.)
        "last_rituals": user.get("last_ritual") or {},
    }


# ---------------------------------------------------------------------------
# Task 8 — Tier 2: Daily Context
# ---------------------------------------------------------------------------


def _build_tier2(user: dict) -> dict:
    """
    Build Tier 2 — Daily Context.

    Returns a dict of JSON-serialisable primitives only.
    """
    from chanakya.db.mongo import checkpoints, interaction_logs  # lazy import

    tz = _get_timezone(user.get("timezone", "Asia/Kolkata"))
    now_local = datetime.now(tz)
    today_start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_utc = today_start_local.astimezone(pytz.utc).replace(tzinfo=None)

    # 8.1 — todays_checkpoints
    todays_checkpoints_list = []
    today_logs = list(
        interaction_logs.find(
            {
                "user_id": user["_id"],
                "timestamp": {"$gte": today_start_utc},
            }
        )
    )

    for log in today_logs:
        checkpoint_id = log.get("checkpoint_id")
        checkpoint_name: str | None = None
        checkpoint_time: str | None = None

        if checkpoint_id is not None:
            cp_doc = checkpoints.find_one({"_id": checkpoint_id})
            if cp_doc:
                checkpoint_name = cp_doc.get("prompt_template") or str(checkpoint_id)
                checkpoint_time = cp_doc.get("time")

        # Fall back to string representation if no checkpoint found
        if checkpoint_name is None:
            checkpoint_name = str(checkpoint_id) if checkpoint_id else "unknown"

        ai_eval = log.get("ai_evaluation") or {}
        verdict = ai_eval.get("verdict") if isinstance(ai_eval, dict) else None

        ts = log.get("timestamp")
        triggered_at = (
            ts.isoformat() + "Z"
            if isinstance(ts, datetime)
            else str(ts) if ts else None
        )

        todays_checkpoints_list.append(
            {
                "name": checkpoint_name,
                "time": checkpoint_time,
                "verdict": verdict,
                "triggered_at": triggered_at,
            }
        )

    # 8.2 — morning_todo
    next_day_plan = user.get("next_day_plan") or {}
    if isinstance(next_day_plan, dict):
        morning_todo = next_day_plan.get("plan_text")
    else:
        morning_todo = None

    # 8.2 — last_eod_report: most recent SCHEDULED TELEGRAM log with message_sent
    last_eod_log = interaction_logs.find_one(
        {
            "user_id": user["_id"],
            "trigger_type": "SCHEDULED",
            "channel": "TELEGRAM",
            "message_sent": {"$exists": True, "$ne": ""},
        },
        sort=[("timestamp", -1)],
    )
    last_eod_report = last_eod_log["message_sent"] if last_eod_log else None

    # 8.2 — active_escalations: checkpoints where failure_punishment.type != WARN
    active_escalations = []
    escalated_cps = list(
        checkpoints.find(
            {
                "user_id": user["_id"],
                "active": True,
                "failure_punishment.type": {"$nin": ["WARN", None]},
            }
        )
    )
    for cp in escalated_cps:
        fp = cp.get("failure_punishment") or {}
        punishment_type = fp.get("type") if isinstance(fp, dict) else None
        cp_name = cp.get("prompt_template") or str(cp.get("_id", "unknown"))
        active_escalations.append(
            {
                "checkpoint_name": cp_name,
                "current_punishment_level": punishment_type,
            }
        )

    # 8.3 — mood_energy
    mood_energy = user.get("mood_energy")

    return {
        "todays_checkpoints": todays_checkpoints_list,
        "morning_todo": morning_todo,
        "last_eod_report": last_eod_report,
        "active_escalations": active_escalations,
        "mood_energy": mood_energy,
    }


# ---------------------------------------------------------------------------
# Task 9 — Tier 3: Historical Pattern Context
# ---------------------------------------------------------------------------


def _build_tier3(user: dict, activity_slot: str) -> dict:
    """
    Build Tier 3 — Historical Pattern Context.

    Returns a dict of JSON-serialisable primitives only.
    """
    from chanakya.db.mongo import checkpoints, interaction_logs, user_state_snapshots  # lazy import

    tz = _get_timezone(user.get("timezone", "Asia/Kolkata"))
    now_local = datetime.now(tz)

    # 9.1 — weekly_summary: last 7 days, one sentence per checkpoint per day
    seven_days_ago_local = now_local.replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - timedelta(days=6)
    seven_days_ago_utc = seven_days_ago_local.astimezone(pytz.utc).replace(tzinfo=None)

    recent_logs = list(
        interaction_logs.find(
            {
                "user_id": user["_id"],
                "timestamp": {"$gte": seven_days_ago_utc},
            },
            sort=[("timestamp", 1)],
        )
    )

    # Group logs by local date
    from collections import defaultdict

    logs_by_date: dict[str, list[dict]] = defaultdict(list)
    for log in recent_logs:
        ts = log.get("timestamp")
        if isinstance(ts, datetime):
            # Convert UTC timestamp to local date
            ts_utc = ts.replace(tzinfo=pytz.utc) if ts.tzinfo is None else ts
            ts_local = ts_utc.astimezone(tz)
            date_str = ts_local.strftime("%Y-%m-%d")
        else:
            date_str = "unknown"
        logs_by_date[date_str].append(log)

    weekly_summary = []
    # Iterate over the last 7 days in order
    for i in range(6, -1, -1):
        day_local = now_local.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=i)
        date_str = day_local.strftime("%Y-%m-%d")
        day_name = day_local.strftime("%A")

        day_logs = logs_by_date.get(date_str, [])
        if not day_logs:
            continue

        # Build checkpoint summaries for this day
        cp_summaries = []
        for log in day_logs:
            checkpoint_id = log.get("checkpoint_id")
            cp_name: str | None = None
            if checkpoint_id is not None:
                cp_doc = checkpoints.find_one({"_id": checkpoint_id})
                if cp_doc:
                    cp_name = cp_doc.get("prompt_template") or str(checkpoint_id)
            if cp_name is None:
                cp_name = str(checkpoint_id) if checkpoint_id else "Checkpoint"

            ai_eval = log.get("ai_evaluation") or {}
            verdict = (
                ai_eval.get("verdict") if isinstance(ai_eval, dict) else None
            ) or "UNKNOWN"
            cp_summaries.append(f"{cp_name} {verdict}")

        if cp_summaries:
            weekly_summary.append(f"{day_name}: {', '.join(cp_summaries)}")

    # 9.2 — recent_snapshots: last 3 user_state_snapshots, text only
    snapshots = list(
        user_state_snapshots.find(
            {"user_id": user["_id"]},
            sort=[("date", -1)],
            limit=3,
        )
    )
    recent_snapshots = [s["summary"] for s in snapshots if "summary" in s]

    # 9.3 — recurring_failure_patterns
    patterns = user.get("recurring_failure_patterns") or []
    recurring_failure_patterns = []
    for p in patterns:
        if isinstance(p, dict):
            recurring_failure_patterns.append(
                {
                    "description": p.get("pattern_description", ""),
                    "times_observed": p.get("times_observed", 0),
                }
            )

    result: dict[str, Any] = {
        "weekly_summary": weekly_summary,
        "recent_snapshots": recent_snapshots,
        "recurring_failure_patterns": recurring_failure_patterns,
    }

    # 9.4 — leetcode_context (only when activity_slot == "LEETCODE")
    if activity_slot == "LEETCODE":
        leetcode_session = user.get("leetcode_session") or {}
        result["leetcode_context"] = {
            "problem_name": leetcode_session.get("problem_name"),
            "time_spent_minutes": leetcode_session.get("time_spent_minutes"),
            "approach_discussed": leetcode_session.get("approach_discussed"),
        }

    # 9.5 — office_context (only when activity_slot == "OFFICE_WORK")
    if activity_slot == "OFFICE_WORK":
        office_session = user.get("office_session") or {}
        result["office_context"] = {
            "current_task": office_session.get("current_task"),
            "deadline_pressure": office_session.get("deadline_pressure"),
        }

    # 9.6 — gym_context (only when activity_slot == "GYM")
    if activity_slot == "GYM":
        from chanakya.db.mongo import interaction_logs as il  # already imported above

        tz2 = _get_timezone(user.get("timezone", "Asia/Kolkata"))
        now_local2 = datetime.now(tz2)
        today_start_local2 = now_local2.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        today_start_utc2 = today_start_local2.astimezone(pytz.utc).replace(tzinfo=None)

        # Look for a gym checkpoint that fired today
        gym_checkpoint_passed = False
        today_gym_logs = list(
            interaction_logs.find(
                {
                    "user_id": user["_id"],
                    "timestamp": {"$gte": today_start_utc2},
                }
            )
        )
        for log in today_gym_logs:
            cp_id = log.get("checkpoint_id")
            if cp_id is not None:
                cp_doc = checkpoints.find_one({"_id": cp_id})
                if cp_doc and cp_doc.get("action_type") in ("CALL", "TELEGRAM_TEXT", "TELEGRAM_VOICE"):
                    ai_eval = log.get("ai_evaluation") or {}
                    verdict = ai_eval.get("verdict") if isinstance(ai_eval, dict) else None
                    if verdict == "SUCCESS":
                        gym_checkpoint_passed = True
                        break

        gym_session = user.get("gym_session") or {}
        result["gym_context"] = {
            "workout_type": gym_session.get("workout_type"),
            "gym_checkpoint_passed": gym_checkpoint_passed,
        }

    return result


# ---------------------------------------------------------------------------
# Task 10 — Tier 4: Deep Memory
# ---------------------------------------------------------------------------


async def _build_tier4(user: dict, today_context_text: str) -> dict:
    """
    Build Tier 4 — Deep Memory.

    Attempts Atlas Vector Search; falls back to recent snapshots on failure.
    Returns a dict of JSON-serialisable primitives only.
    """
    from chanakya.db.mongo import user_state_snapshots  # lazy import

    # 10.1 — Vector similarity search with fallback
    query_vector = []
    try:
        import httpx
        from chanakya.config import OPENAI_API_KEY
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": today_context_text,
                    "model": "text-embedding-3-small"
                }
            )
            if resp.status_code == 200:
                query_vector = resp.json()["data"][0]["embedding"]
    except Exception as exc:
        logger.warning("Failed to generate embedding for vector search: %s", exc)

    if not query_vector:
        logger.warning("No query vector available; falling back to recent snapshots.")
        similar_snapshots = list(
            user_state_snapshots.find(
                {"user_id": user["_id"]},
                sort=[("date", -1)],
                limit=3,
            )
        )
    else:
        try:
            similar_snapshots = list(
                user_state_snapshots.aggregate(
                    [
                        {
                            "$vectorSearch": {
                                "index": "user_state_snapshots_embeddings_vector",
                                "path": "embeddings",
                                "queryVector": query_vector,
                                "numCandidates": 20,
                                "limit": 3,
                                "filter": {"user_id": user["_id"]},
                            }
                        }
                    ]
                )
            )
        except Exception:
            logger.warning("Vector search unavailable; falling back to recent snapshots.")
            similar_snapshots = list(
                user_state_snapshots.find(
                    {"user_id": user["_id"]},
                    sort=[("date", -1)],
                    limit=3,
                )
            )

    similar_past_days = [s["summary"] for s in similar_snapshots if "summary" in s]

    # 10.2 — long_term_pattern_notes
    long_term_pattern_notes = user.get("long_term_pattern_notes") or []
    # Ensure it's a list of plain strings
    if isinstance(long_term_pattern_notes, list):
        long_term_pattern_notes = [
            str(note) for note in long_term_pattern_notes if note is not None
        ]
    else:
        long_term_pattern_notes = []

    return {
        "similar_past_days": similar_past_days,
        "long_term_pattern_notes": long_term_pattern_notes,
    }


# ---------------------------------------------------------------------------
# Task 11 — Prompt template selection and variable substitution
# ---------------------------------------------------------------------------


def get_prompt_templates(activity_slot: str, interaction_type: str) -> dict[str, str]:
    """
    Return a dict of {tone: template_text} for all matching templates.

    Fallback chain:
      1. Exact (activity_slot, interaction_type)
      2. ("FREE_TIME", interaction_type) — logs WARNING
      3. ("GENERIC", interaction_type) — logs WARNING
      4. Raises NoTemplateFoundError

    Results are cached in _template_cache keyed by (activity_slot, interaction_type).
    """
    from chanakya.db.mongo import prompt_templates  # lazy import

    cache_key = (activity_slot, interaction_type)
    if cache_key in _template_cache:
        return _template_cache[cache_key]

    def _query_templates(slot: str) -> dict[str, str]:
        docs = list(
            prompt_templates.find(
                {"activity_slot": slot, "interaction_type": interaction_type}
            )
        )
        return {doc["tone"]: doc["template_text"] for doc in docs if "tone" in doc and "template_text" in doc}

    # Step 1: exact match
    result = _query_templates(activity_slot)

    if not result and activity_slot != "FREE_TIME":
        # Step 2: FREE_TIME fallback
        logger.warning(
            "No prompt template for activity_slot=%r, interaction_type=%r; "
            "falling back to FREE_TIME.",
            activity_slot,
            interaction_type,
        )
        result = _query_templates("FREE_TIME")

    if not result and activity_slot != "GENERIC":
        # Step 3: GENERIC fallback
        logger.warning(
            "No prompt template for activity_slot=FREE_TIME, interaction_type=%r; "
            "falling back to GENERIC.",
            interaction_type,
        )
        result = _query_templates("GENERIC")

    if not result:
        raise NoTemplateFoundError(activity_slot, interaction_type)

    _template_cache[cache_key] = result
    return result


def render_template(template_text: str, context: dict) -> str:
    """
    Substitute all {variable} placeholders using values from the flattened context dict.

    For unresolvable placeholders: log WARNING and replace with empty string.
    Never raises.
    """
    flat = _flatten_context(context)

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key in flat and flat[key] is not None:
            return str(flat[key])
        logger.warning("Unresolvable template placeholder: {%s}", key)
        return ""

    return re.sub(r"\{(\w+)\}", _replace, template_text)


def _flatten_context(context: dict, prefix: str = "") -> dict[str, Any]:
    """
    Recursively flatten a nested dict into a single-level dict.

    Nested keys are joined with underscores: {"tier1": {"name": "X"}} → {"name": "X", "tier1_name": "X"}
    Top-level keys from nested dicts are also promoted to the root level.
    """
    flat: dict[str, Any] = {}
    for key, value in context.items():
        full_key = f"{prefix}{key}" if prefix else key
        flat[full_key] = value
        if isinstance(value, dict):
            nested = _flatten_context(value, prefix=f"{full_key}_")
            flat.update(nested)
            # Also promote nested keys without prefix for convenience
            for nested_key, nested_val in value.items():
                if nested_key not in flat:
                    flat[nested_key] = nested_val
    return flat


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


class ContextAssembler:
    """
    Assembles the tiered context dict for an LLM call.

    Usage:
        assembler = ContextAssembler()
        context = assembler.build(user, interaction_type, session_context)
    """

    async def build(
        self,
        user: dict,
        interaction_type: str,
        session_context: dict | None = None,
    ) -> dict:
        """
        Build and return the full tiered context dict for an LLM call.

        session_context: optional dict with ephemeral session data
        (e.g., leetcode_session, office_session, gym_session, mood_energy)
        that is not persisted to MongoDB but is available for this interaction.

        Returns:
            {
                "tier1": {...},           # always present
                "tier2": {...} | None,    # present for non-COMMAND_RESPONSE
                "tier3": {...} | None,    # present for CHECK_IN, EOD, ESCALATION, MENTOR_TALK
                "tier4": {...} | None,    # present for EOD, WEEKLY_REVIEW
                "prompt_templates": {     # always present — all tones for this slot+type
                    "HARSH": "...",
                    "MENTOR": "...",
                    ...
                },
            }
        """
        # Merge session_context into user dict (ephemeral, not persisted)
        effective_user = dict(user)
        if session_context:
            effective_user.update(session_context)

        # Tier 1 — always included
        tier1 = _build_tier1(effective_user)
        activity_slot = tier1["current_activity_slot"]

        # Tier 2 — all non-COMMAND_RESPONSE interactions
        tier2 = None
        if interaction_type in _TIER2_TYPES:
            tier2 = _build_tier2(effective_user)

        # Tier 3 — CHECK_IN, ESCALATION, MENTOR_TALK, EOD, WEEKLY_REVIEW
        tier3 = None
        if interaction_type in _TIER3_TYPES:
            tier3 = _build_tier3(effective_user, activity_slot)

        # Tier 4 — EOD, WEEKLY_REVIEW
        tier4 = None
        if interaction_type in _TIER4_TYPES:
            # Build a brief text summary of today's context for vector search
            today_context_text = f"{tier1.get('today_date', '')} {activity_slot}"
            tier4 = await _build_tier4(effective_user, today_context_text)

        # Prompt templates — always fetched
        try:
            templates = get_prompt_templates(activity_slot, interaction_type)
        except NoTemplateFoundError:
            logger.warning(
                "No prompt templates found for slot=%r, type=%r; using empty dict.",
                activity_slot,
                interaction_type,
            )
            templates = {}

        return {
            "tier1": tier1,
            "tier2": tier2,
            "tier3": tier3,
            "tier4": tier4,
            "prompt_templates": templates,
        }
