"""
learning_extractor.py — Self-learning loop for Chanakya.

Runs periodically (every 5 interactions or every 2 hours) to extract behavioral
patterns from recent interactions. Stores insights in a learning_log collection
and updates user.learned_patterns (max 20, FIFO) for injection into Tier 1 context.

Never blocks user-facing responses — runs as a background APScheduler job.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from chanakya.db.mongo import db, interaction_logs, users

logger = logging.getLogger(__name__)

learning_log = db["learning_log"]

_MAX_PATTERNS = 20

_EXTRACTION_PROMPT = """You analyze interaction logs between an AI accountability coach and a user.
Extract behavioral patterns — what works, what doesn't, and recurring behaviors.

From these recent interactions, extract:
1. EFFECTIVE: What approaches made the user respond well? (max 3)
2. INEFFECTIVE: What approaches got pushback or no engagement? (max 2)
3. PATTERNS: Recurring user behaviors worth tracking (max 3)

Each item should be one concise sentence.

Return ONLY valid JSON:
{"effective": ["..."], "ineffective": ["..."], "patterns": ["..."]}"""


async def extract_learnings(user_id) -> dict | None:
    """Analyze recent interactions and extract behavioral patterns."""
    recent = list(
        interaction_logs.find(
            {"user_id": user_id, "user_response": {"$ne": None}},
            sort=[("timestamp", -1)],
            limit=10,
        )
    )

    if len(recent) < 3:
        return None

    summaries = []
    for log in recent:
        verdict = ""
        ai_eval = log.get("ai_evaluation") or {}
        if isinstance(ai_eval, dict):
            verdict = ai_eval.get("verdict") or ""

        summaries.append(
            f"- Coach: {(log.get('message_sent') or '')[:150]}\n"
            f"  User: {(log.get('user_response') or '')[:150]}\n"
            f"  Verdict: {verdict}"
        )

    interaction_text = "\n".join(summaries)

    try:
        from chanakya.agent.llm_provider import call_with_fallback
        content = (await call_with_fallback(
            messages=[
                {"role": "system", "content": _EXTRACTION_PROMPT},
                {"role": "user", "content": interaction_text},
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
                return None

        return data

    except Exception as exc:
        logger.warning("Learning extraction failed for user %s: %s", user_id, exc)
        return None


async def run_learning_cycle(user_id) -> None:
    """Full learning cycle: extract → store → update user patterns."""
    data = await extract_learnings(user_id)
    if not data:
        return

    now = datetime.utcnow()

    learning_log.insert_one({
        "user_id": user_id,
        "timestamp": now,
        "effective": data.get("effective", []),
        "ineffective": data.get("ineffective", []),
        "patterns": data.get("patterns", []),
    })

    new_patterns = []
    for item in data.get("effective", []):
        new_patterns.append({"type": "effective", "text": item, "observed_at": now})
    for item in data.get("ineffective", []):
        new_patterns.append({"type": "ineffective", "text": item, "observed_at": now})
    for item in data.get("patterns", []):
        new_patterns.append({"type": "behavior", "text": item, "observed_at": now})

    user = users.find_one({"_id": user_id})
    if not user:
        return

    existing = user.get("learned_patterns") or []
    combined = existing + new_patterns
    # FIFO: keep only the most recent _MAX_PATTERNS
    if len(combined) > _MAX_PATTERNS:
        combined = combined[-_MAX_PATTERNS:]

    users.update_one(
        {"_id": user_id},
        {"$set": {"learned_patterns": combined, "last_learning_run": now}},
    )

    logger.info(
        "Learning cycle complete for user %s: +%d patterns (total %d)",
        user_id, len(new_patterns), len(combined),
    )


def should_run_learning(user_id) -> bool:
    """Check if enough interactions have passed since last learning run."""
    user = users.find_one({"_id": user_id})
    if not user:
        return False

    last_run = user.get("last_learning_run")
    if not last_run:
        return True

    # Run if 2+ hours since last run
    if datetime.utcnow() - last_run > timedelta(hours=2):
        return True

    # Run if 5+ new interactions since last run
    new_count = interaction_logs.count_documents({
        "user_id": user_id,
        "user_response": {"$ne": None},
        "timestamp": {"$gt": last_run},
    })
    return new_count >= 5
