"""
goal_sentinel.py — Proactive goal monitoring.

Runs every 6 hours to check if stated goals are being neglected.
Queues proactive nudges via Telegram when goals go unattended.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from chanakya.db.mongo import db, interaction_logs, users

logger = logging.getLogger(__name__)

goal_nudges = db["goal_nudges"]

_SENTINEL_PROMPT = """You are a goal-monitoring system for a personal accountability coach.

Given:
- A user's stated goals/mindset entries
- Their recent activity (last 48 hours of interactions)

Identify goals that are being NEGLECTED — mentioned as important but with zero recent action.
Only flag genuinely neglected goals, not ones that are on schedule.

Return ONLY valid JSON:
{"neglected": [{"goal": "short description", "reason": "why it's neglected", "nudge": "one sharp sentence to remind them"}]}

If nothing is neglected, return: {"neglected": []}"""


async def check_goals(user_id) -> list[dict]:
    """Check for neglected goals and return nudge suggestions."""
    user = users.find_one({"_id": user_id})
    if not user:
        return []

    # Gather goals from mindset entries
    from chanakya.db.mongo import get_all_identity_context
    identity = get_all_identity_context(user_id)
    mindset_list = identity.get("mindset", []) if identity else []

    # mindset is a list of dicts like [{"category": "goal", "text": "..."}, ...]
    goals = [e["text"] for e in mindset_list if isinstance(e, dict) and e.get("category") == "goal"]
    rules = [e["text"] for e in mindset_list if isinstance(e, dict) and e.get("category") == "rule"]
    if not goals and not rules:
        return []

    goals_text = "\n".join(f"- [GOAL] {g}" for g in goals)
    rules_text = "\n".join(f"- [RULE] {r}" for r in rules[:5])

    # Recent activity (last 48h)
    cutoff = datetime.utcnow() - timedelta(hours=48)
    recent = list(
        interaction_logs.find(
            {"user_id": user_id, "timestamp": {"$gte": cutoff}},
            sort=[("timestamp", -1)],
            limit=15,
        )
    )

    activity_lines = []
    for log in recent:
        response = (log.get("user_response") or "")[:100]
        sent = (log.get("message_sent") or "")[:100]
        if response or sent:
            activity_lines.append(f"- {response or sent}")

    activity_text = "\n".join(activity_lines) if activity_lines else "(no recent activity)"

    input_text = (
        f"GOALS:\n{goals_text}\n\nRULES:\n{rules_text}\n\n"
        f"RECENT ACTIVITY (48h):\n{activity_text}"
    )

    try:
        from chanakya.agent.llm_provider import call_with_fallback
        content = (await call_with_fallback(
            messages=[
                {"role": "system", "content": _SENTINEL_PROMPT},
                {"role": "user", "content": input_text},
            ],
            temperature=0.2,
            max_tokens=300,
            timeout=10.0,
        )).strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
            else:
                return []

        return data.get("neglected", [])

    except Exception as exc:
        logger.warning("Goal sentinel check failed for user %s: %s", user_id, exc)
        return []


async def run_goal_sentinel(user_id) -> None:
    """Full sentinel cycle: check goals → queue nudges."""
    neglected = await check_goals(user_id)
    if not neglected:
        return

    now = datetime.utcnow()

    for item in neglected:
        # Avoid duplicate nudges within 12 hours
        existing = goal_nudges.find_one({
            "user_id": user_id,
            "goal": item.get("goal", ""),
            "created_at": {"$gte": now - timedelta(hours=12)},
        })
        if existing:
            continue

        goal_nudges.insert_one({
            "user_id": user_id,
            "goal": item.get("goal", ""),
            "reason": item.get("reason", ""),
            "nudge_text": item.get("nudge", ""),
            "delivered": False,
            "created_at": now,
        })

    logger.info(
        "Goal sentinel found %d neglected goals for user %s",
        len(neglected), user_id,
    )


async def deliver_pending_nudges(user_id) -> list[str]:
    """Get undelivered nudge texts and mark them delivered. Called during interactions."""
    pending = list(
        goal_nudges.find({"user_id": user_id, "delivered": False})
    )

    nudge_texts = []
    for nudge in pending:
        nudge_texts.append(nudge["nudge_text"])
        goal_nudges.update_one({"_id": nudge["_id"]}, {"$set": {"delivered": True}})

    return nudge_texts
