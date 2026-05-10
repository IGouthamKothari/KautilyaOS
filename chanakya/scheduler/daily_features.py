"""
daily_features.py — EOD report, morning todo, random check-ins, and daily snapshots.

Implements:
  - Task 26: End-of-Day report (fire_eod_checkpoint, handle_eod_reply)
  - Task 27: Morning todo delivery (should_send_morning_todo, fire_morning_todo)
  - Task 28: Random mentor check-ins (compute_daily_checkin_times,
             get_or_create_checkin_schedule, should_fire_checkin, fire_checkin)
  - Task 29: Daily state snapshots and vector memory
             (generate_daily_snapshot, _compute_and_store_embedding)
  - Task 30: Weekly review trigger (fire_weekly_review) — Sunday EOD
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta

import pytz

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task 28.2 — Module-level check-in schedule state (in-memory + DB persistence)
# ---------------------------------------------------------------------------

# In-memory fast-path: {user_id_str: [HH:MM, ...]}
_daily_checkin_schedule: dict[str, list[str]] = {}
# {user_id_str: "YYYY-MM-DD"}
_checkin_schedule_date: dict[str, str] = {}

_CHECKIN_SCHEDULE_COLLECTION = "checkin_schedules"


def _persist_checkin_schedule(user_id, local_date: str, times: list[str]) -> None:
    """Persist the daily check-in schedule to DB so restarts don't re-fire."""
    try:
        from chanakya.db.mongo import db
        db[_CHECKIN_SCHEDULE_COLLECTION].update_one(
            {"user_id": user_id, "date": local_date},
            {"$set": {"times": times, "updated_at": datetime.utcnow()}},
            upsert=True,
        )
    except Exception as exc:
        logger.warning("Failed to persist check-in schedule: %s", exc)


def _load_checkin_schedule(user_id, local_date: str) -> list[str] | None:
    """Load persisted check-in schedule from DB. Returns None if not found."""
    try:
        from chanakya.db.mongo import db
        doc = db[_CHECKIN_SCHEDULE_COLLECTION].find_one(
            {"user_id": user_id, "date": local_date}
        )
        return doc["times"] if doc else None
    except Exception as exc:
        logger.warning("Failed to load check-in schedule: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Task 26.1 — Fire EOD checkpoint
# ---------------------------------------------------------------------------


async def fire_eod_checkpoint(user: dict) -> None:
    """
    Fire the EOD report for a user.
    Called by the checkpoint runner when the EOD checkpoint fires.
    """
    from chanakya.agent.chanakya_agent import ChanakyaAgent, execute_actions
    from chanakya.scheduler.checkpoint_runner import _execute_telegram_text

    agent = ChanakyaAgent(user)
    decision = await agent.invoke(
        raw_input="Generate end-of-day report and next-day plan.",
        interaction_type="EOD",
    )

    if decision is None:
        logger.warning("EOD agent returned None for user %s", user["_id"])
        return

    # Send the EOD report text
    _execute_telegram_text(user, decision.response_text)

    # Execute actions (store_next_day_plan will be in the actions array)
    execute_actions(decision.actions, user, log_id=None, decision=decision)

    logger.info("EOD report sent to user %s", user["_id"])


# ---------------------------------------------------------------------------
# Task 26.4 — Handle EOD reply
# ---------------------------------------------------------------------------


async def handle_eod_reply(user: dict, user_input: str) -> str:
    """
    Handle user reply to an EOD message.
    Routes to agent with interaction_type=EOD.
    Returns the agent's response_text.
    """
    from chanakya.agent.chanakya_agent import ChanakyaAgent, execute_actions

    agent = ChanakyaAgent(user)
    decision = await agent.invoke(
        raw_input=user_input,
        interaction_type="EOD",
    )

    if decision is None:
        return "Something went wrong. Please try again."

    execute_actions(decision.actions, user, log_id=None, decision=decision)
    return decision.response_text


# ---------------------------------------------------------------------------
# Task 27.1 — Should send morning todo
# ---------------------------------------------------------------------------


def should_send_morning_todo(user: dict) -> bool:
    """Return True if morning todo should be sent for this user."""
    if not user.get("morning_todo_time"):
        logger.warning(
            "morning_todo_time not set for user %s; skipping morning todo.",
            user["_id"],
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Task 27.2 / 27.3 — Fire morning todo
# ---------------------------------------------------------------------------


async def fire_morning_todo(user: dict) -> None:
    """
    Fire the morning todo delivery for a user.
    Called by the checkpoint runner when the morning_todo_time checkpoint fires.
    """
    from chanakya.agent.chanakya_agent import ChanakyaAgent
    from chanakya.db.mongo import users as users_col
    from chanakya.scheduler.checkpoint_runner import _execute_telegram_text

    if not should_send_morning_todo(user):
        return

    # Check for confirmed next_day_plan from previous night
    next_day_plan = user.get("next_day_plan") or {}
    has_confirmed_plan = (
        isinstance(next_day_plan, dict)
        and next_day_plan.get("confirmed") is True
        and next_day_plan.get("plan_text")
    )

    # 27.3: If no confirmed plan, increment fallback count and include notice
    if not has_confirmed_plan:
        users_col.update_one(
            {"_id": user["_id"]},
            {"$inc": {"morning_todo_fallback_count": 1}},
        )
        logger.info(
            "Morning todo fallback used for user %s (count incremented)",
            user["_id"],
        )

    session_context: dict = {}
    if has_confirmed_plan:
        session_context["confirmed_plan"] = next_day_plan.get("plan_text")

    agent = ChanakyaAgent(user)
    decision = await agent.invoke(
        raw_input="Generate morning todo list for today.",
        interaction_type="MORNING_TODO",
        session_context=session_context,
    )

    if decision is None:
        logger.warning("Morning todo agent returned None for user %s", user["_id"])
        return

    _execute_telegram_text(user, decision.response_text)


# ---------------------------------------------------------------------------
# Task 28.1 — Compute daily check-in schedule
# ---------------------------------------------------------------------------


def compute_daily_checkin_times(user: dict) -> list[str]:
    """
    Compute N random check-in times for today within the user's active-hours window.

    Rules:
    - N is between checkin_min_per_day and checkin_max_per_day
    - No two check-ins closer than 90 minutes apart
    - All times within checkin_window_start and checkin_window_end (user's local timezone)

    Returns list of HH:MM strings in user's local timezone.
    """
    tz_str = user.get("timezone", "Asia/Kolkata")
    try:
        tz = pytz.timezone(tz_str)
    except pytz.UnknownTimeZoneError:
        tz = pytz.timezone("Asia/Kolkata")

    min_count = user.get("checkin_min_per_day", 2)
    max_count = user.get("checkin_max_per_day", 4)
    n = random.randint(min_count, max_count)

    window_start = user.get("checkin_window_start", "09:00")
    window_end = user.get("checkin_window_end", "21:00")

    # Convert window to minutes since midnight
    start_h, start_m = map(int, window_start.split(":"))
    end_h, end_m = map(int, window_end.split(":"))
    start_minutes = start_h * 60 + start_m
    end_minutes = end_h * 60 + end_m

    if end_minutes <= start_minutes:
        logger.warning(
            "Invalid check-in window for user %s; using defaults.", user["_id"]
        )
        start_minutes = 9 * 60
        end_minutes = 21 * 60

    # Generate N times with minimum 90-minute gaps
    MIN_GAP = 90  # minutes
    times: list[int] = []
    attempts = 0
    max_attempts = 100

    while len(times) < n and attempts < max_attempts:
        attempts += 1
        candidate = random.randint(start_minutes, end_minutes - 1)

        # Check 90-minute gap from all existing times
        if all(abs(candidate - t) >= MIN_GAP for t in times):
            times.append(candidate)

    # Convert back to HH:MM strings
    result = []
    for t in sorted(times):
        h = t // 60
        m = t % 60
        result.append(f"{h:02d}:{m:02d}")

    return result


# ---------------------------------------------------------------------------
# Task 28.2 — Get or create check-in schedule
# ---------------------------------------------------------------------------


def get_or_create_checkin_schedule(user: dict, local_date: str) -> list[str]:
    """Get today's check-in schedule, creating and persisting it if needed.

    Load order: in-memory cache → DB → compute fresh.
    Resets if the date has changed (new day).
    """
    user_id_str = str(user["_id"])

    # In-memory hit for today
    if (
        user_id_str in _daily_checkin_schedule
        and _checkin_schedule_date.get(user_id_str) == local_date
    ):
        return _daily_checkin_schedule[user_id_str]

    # Try DB (survives restarts)
    persisted = _load_checkin_schedule(user["_id"], local_date)
    if persisted:
        _daily_checkin_schedule[user_id_str] = persisted
        _checkin_schedule_date[user_id_str] = local_date
        logger.info(
            "Check-in schedule loaded from DB for user %s on %s: %s",
            user["_id"], local_date, persisted,
        )
        return persisted

    # Compute fresh and persist
    times = compute_daily_checkin_times(user)
    _daily_checkin_schedule[user_id_str] = times
    _checkin_schedule_date[user_id_str] = local_date
    _persist_checkin_schedule(user["_id"], local_date, times)
    logger.info(
        "Check-in schedule computed for user %s on %s: %s",
        user["_id"], local_date, times,
    )
    return times


# ---------------------------------------------------------------------------
# Task 28.3 — Should fire check-in
# ---------------------------------------------------------------------------


def should_fire_checkin(user: dict) -> bool:
    """Return True if a check-in should be initiated for this user."""
    current_mode = user.get("current_mode", "NORMAL")
    activity_slot = user.get("current_activity", "FREE_TIME")

    if current_mode in ("WAR_MODE", "AWAY"):
        return False
    if activity_slot == "SLEEP":
        return False
    return True


# ---------------------------------------------------------------------------
# Task 28.4 / 28.5 — Fire check-in (two-way voice call)
# ---------------------------------------------------------------------------


async def fire_checkin(user: dict) -> None:
    """
    Fire a random mentor check-in as a two-way voice call.

    If the user has no phone number or WEBHOOK_URL is not set, falls back
    to a Telegram text message so the check-in is never silently dropped.
    """
    from chanakya.agent.chanakya_agent import ChanakyaAgent, execute_actions
    from chanakya.config import WEBHOOK_URL
    from chanakya.db.mongo import interaction_logs
    from chanakya.scheduler.checkpoint_runner import _execute_telegram_text

    if not should_fire_checkin(user):
        return

    phone = user.get("phone", "")
    activity_slot = user.get("current_activity", "FREE_TIME")

    # ------------------------------------------------------------------
    # Step 1: Ask the agent to generate a conversational opening
    # ------------------------------------------------------------------
    agent = ChanakyaAgent(user)
    decision = await agent.invoke(
        raw_input=(
            f"Initiate a brief two-way mentor check-in call. "
            f"Current activity: {activity_slot}. "
            "Generate a short opening (2-3 sentences) that starts a real conversation — "
            "ask one sharp question about what the user is doing or thinking right now."
        ),
        interaction_type="CHECK_IN",
    )

    if decision is None:
        logger.warning("Check-in agent returned None for user %s", user["_id"])
        return

    opening_text = decision.response_text or (
        f"Chanakya here. You're in your {activity_slot.replace('_', ' ').lower()} block. "
        "What are you actually working on right now?"
    )

    execute_actions(decision.actions, user, log_id=None, decision=decision)

    # ------------------------------------------------------------------
    # Step 2: Log the check-in interaction
    # ------------------------------------------------------------------
    now = datetime.utcnow()
    log_doc = {
        "user_id": user["_id"],
        "checkpoint_id": None,
        "timestamp": now,
        "trigger_type": "REACTIVE",
        "channel": "CALL" if (phone and WEBHOOK_URL) else "TELEGRAM",
        "message_sent": opening_text,
        "checkin_topic": activity_slot,
        "ai_evaluation": {"verdict": None, "confidence": None, "reasoning": None},
        "created_at": now,
    }
    try:
        log_result = interaction_logs.insert_one(log_doc)
        session_id = str(log_result.inserted_id)
    except Exception as exc:
        logger.error(
            "Failed to insert check-in log for user %s: %s", user["_id"], exc
        )
        _execute_telegram_text(user, opening_text)
        return

    # ------------------------------------------------------------------
    # Step 3: Place two-way call if possible, else fall back to Telegram
    # ------------------------------------------------------------------
    if not phone or not WEBHOOK_URL:
        logger.info(
            "No phone/WEBHOOK_URL for user %s — sending check-in as Telegram text.",
            user["_id"],
        )
        _execute_telegram_text(user, opening_text)
        return

    from chanakya.integrations.twilio_client import TwilioClient, TwilioError
    from chanakya.integrations.twilio_webhooks import create_voice_session, synthesize_call_opening

    create_voice_session(
        session_id=session_id,
        user_id=str(user["_id"]),
        context=opening_text,
        conversation_context=user.get("conversation_context") or "",
        audio_bytes=synthesize_call_opening(opening_text),
    )

    twiml_url = f"{WEBHOOK_URL.rstrip('/')}/twilio/voice/{session_id}"
    try:
        twilio = TwilioClient()
        call_sid = twilio.make_call(to=phone, twiml_url=twiml_url)
        interaction_logs.update_one(
            {"_id": log_result.inserted_id},
            {"$set": {"twilio_call_sid": call_sid}},
        )
        logger.info(
            "Random check-in call placed for user %s, session=%s, sid=%s",
            user["_id"],
            session_id,
            call_sid,
        )
    except TwilioError as exc:
        logger.warning(
            "Check-in call failed for user %s: %s — falling back to Telegram text.",
            user["_id"],
            exc,
        )
        _execute_telegram_text(user, opening_text)


# ---------------------------------------------------------------------------
# Task 29.1 — Generate daily snapshot
# ---------------------------------------------------------------------------


async def generate_daily_snapshot(user: dict) -> None:
    """
    Generate a daily state snapshot for a user.
    Called once per day (e.g., at EOD time or midnight).
    """
    from collections import Counter

    from chanakya.db.mongo import interaction_logs, user_state_snapshots

    tz_str = user.get("timezone", "Asia/Kolkata")
    try:
        tz = pytz.timezone(tz_str)
    except pytz.UnknownTimeZoneError:
        tz = pytz.timezone("Asia/Kolkata")

    now_local = datetime.now(tz)
    today_date = now_local.strftime("%Y-%m-%d")

    # Check if snapshot already exists for today
    existing = user_state_snapshots.find_one(
        {
            "user_id": user["_id"],
            "date": today_date,
        }
    )
    if existing:
        logger.debug(
            "Snapshot already exists for user %s on %s", user["_id"], today_date
        )
        return

    # Build summary from today's interaction_logs
    today_start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_utc = today_start_local.astimezone(pytz.utc).replace(tzinfo=None)

    today_logs = list(
        interaction_logs.find(
            {
                "user_id": user["_id"],
                "timestamp": {"$gte": today_start_utc},
            }
        )
    )

    # Build rich natural-language summary
    from collections import Counter
    from chanakya.db.mongo import checkpoints as cp_col

    summary_parts = [
        f"Date: {today_date}. User: {user.get('name', 'Unknown')}."
        f" Day: {now_local.strftime('%A')}."
    ]

    if today_logs:
        verdicts = []
        activities_seen = []
        user_responses = []

        for log in today_logs:
            ai_eval = log.get("ai_evaluation") or {}
            verdict = ai_eval.get("verdict") if isinstance(ai_eval, dict) else None
            if verdict:
                verdicts.append(verdict)

            # Activity from checkpoint
            cp_id = log.get("checkpoint_id")
            if cp_id:
                cp_doc = cp_col.find_one({"_id": cp_id})
                if cp_doc:
                    name = cp_doc.get("display_name") or cp_doc.get("activity", "")
                    if name:
                        activities_seen.append(f"{cp_doc.get('time', '?')} {name} → {verdict or '?'}")

            # User response snippets (mood signals)
            resp = (log.get("user_response") or "").strip()
            if resp and len(resp) > 5:
                user_responses.append(resp[:80])

        if verdicts:
            counts = Counter(verdicts)
            summary_parts.append(
                f"Verdicts: {', '.join(f'{v}={c}' for v, c in counts.items())}."
            )

        if activities_seen:
            summary_parts.append(f"Activity log: {'; '.join(activities_seen[:8])}.")

        if user_responses:
            summary_parts.append(f"User said: {' | '.join(user_responses[:3])}.")
    else:
        summary_parts.append("No interactions recorded today.")

    streak = user.get("streak_count", 0)
    mode = user.get("current_mode", "NORMAL")
    activity_slot = user.get("current_activity", "FREE_TIME")
    summary_parts.append(
        f"Streak: {streak} days. Mode: {mode}. Last activity: {activity_slot}."
    )

    summary = " ".join(summary_parts)

    # Insert snapshot (without embeddings initially)
    snapshot_doc = {
        "user_id": user["_id"],
        "date": today_date,
        "summary": summary,
        "embeddings": None,
        "created_at": datetime.utcnow(),
    }

    try:
        result = user_state_snapshots.insert_one(snapshot_doc)
        logger.info(
            "Daily snapshot created for user %s on %s", user["_id"], today_date
        )
    except Exception as exc:
        logger.error(
            "Failed to insert snapshot for user %s: %s", user["_id"], exc
        )
        return

    # 29.2: Compute embedding
    try:
        await _compute_and_store_embedding(result.inserted_id, summary)
    except Exception as exc:
        logger.warning(
            "Embedding computation raised unexpectedly for snapshot %s: %s",
            result.inserted_id,
            exc,
        )


# ---------------------------------------------------------------------------
# Task 29.2 / 29.3 — Compute and store embedding
# ---------------------------------------------------------------------------


async def _compute_and_store_embedding(snapshot_id, summary: str) -> None:
    """Compute embedding via OpenAI /v1/embeddings and store it.

    Uses OPENAI_API_KEY directly — not OpenRouter (which is optional).
    On failure: log and continue without blocking.
    """
    import httpx
    from chanakya.config import OPENAI_API_KEY
    from chanakya.db.mongo import user_state_snapshots

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "text-embedding-3-small",
                    "input": summary,
                },
            )
            response.raise_for_status()
            data = response.json()
            embedding = data["data"][0]["embedding"]

        user_state_snapshots.update_one(
            {"_id": snapshot_id},
            {"$set": {"embeddings": embedding}},
        )
        logger.info("Embedding stored for snapshot %s (%d dims)", snapshot_id, len(embedding))
    except Exception as exc:
        logger.warning(
            "Failed to compute embedding for snapshot %s: %s. "
            "Snapshot saved without embeddings.",
            snapshot_id,
            exc,
        )


# ---------------------------------------------------------------------------
# Task 30 — Weekly review trigger (Sunday EOD)
# ---------------------------------------------------------------------------


async def fire_weekly_review(user: dict) -> None:
    """Fire the weekly review for a user. Called on Sunday at EOD time.

    Uses WEEKLY_REVIEW interaction type so Tier 3 historical context is included.
    """
    from chanakya.agent.chanakya_agent import ChanakyaAgent, execute_actions
    from chanakya.scheduler.checkpoint_runner import _execute_telegram_text

    agent = ChanakyaAgent(user)
    decision = await agent.invoke(
        raw_input=(
            "Generate the weekly review. Summarise this week's performance: "
            "streaks, failures, patterns, what improved, what didn't. "
            "Then set intentions and adjustments for next week. Be direct."
        ),
        interaction_type="WEEKLY_REVIEW",
    )

    if decision is None:
        logger.warning("Weekly review agent returned None for user %s", user["_id"])
        return

    _execute_telegram_text(user, decision.response_text)
    execute_actions(decision.actions, user, log_id=None, decision=decision)
    logger.info("Weekly review sent to user %s", user["_id"])


async def calculate_daily_scores(user: dict, local_date: str) -> None:
    """Evaluate today's discipline and update the Warrior Streak.

    Criteria for Streak continuation:
    1. No FAILED verdicts in interaction_logs for today.
    2. SLEEP ritual logged.
    3. MOOD or ENERGY ritual logged.
    4. Any active commitment completed with SUCCESS (or no active commitment).
    """
    from chanakya.db.mongo import interaction_logs, rituals, users as users_col

    # Check if already processed for today
    if user.get("last_streak_update_date") == local_date:
        return

    tz_str = user.get("timezone", "Asia/Kolkata")
    try:
        tz = pytz.timezone(tz_str)
    except pytz.UnknownTimeZoneError:
        tz = pytz.timezone("Asia/Kolkata")

    now_local = datetime.now(tz)
    today_start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_utc = today_start_local.astimezone(pytz.utc).replace(tzinfo=None)

    # 1. Check for FAILED verdicts
    failed_logs = list(interaction_logs.find({
        "user_id": user["_id"],
        "timestamp": {"$gte": today_start_utc},
        "ai_evaluation.verdict": "FAILED"
    }))

    # 2. Check rituals
    has_sleep = rituals.find_one({
        "user_id": user["_id"],
        "category": "SLEEP",
        "timestamp": {"$gte": today_start_utc}
    })

    has_mood_energy = rituals.find_one({
        "user_id": user["_id"],
        "category": {"$in": ["MOOD", "ENERGY"]},
        "timestamp": {"$gte": today_start_utc}
    })

    # 3. Check commitments
    commitment = user.get("current_commitment")
    commitment_failed = False
    if commitment and commitment.get("status") == "ACTIVE":
        # If it's EOD and it's still ACTIVE, it's a failure (user didn't report)
        commitment_failed = True
    elif commitment and commitment.get("status") == "FAILED":
        commitment_failed = True

    # 4. Final Verdict
    success = (len(failed_logs) == 0) and (has_sleep is not None) and (has_mood_energy is not None) and (not commitment_failed)

    if success:
        users_col.update_one(
            {"_id": user["_id"]},
            {
                "$inc": {"warrior_streak": 1},
                "$set": {"last_streak_update_date": local_date}
            }
        )
        logger.info("Warrior Streak incremented for user %s on %s", user["_id"], local_date)
    else:
        # Penalise or reset
        reasons = []
        if failed_logs: reasons.append("failed checkpoints")
        if not has_sleep: reasons.append("missing Sleep log")
        if not has_mood_energy: reasons.append("missing Mood/Energy log")
        if commitment_failed: reasons.append("unmet commitment")

        users_col.update_one(
            {"_id": user["_id"]},
            {
                "$set": {"warrior_streak": 0, "last_streak_update_date": local_date}
            }
        )
        logger.info("Warrior Streak RESET for user %s on %s. Reason: %s", user["_id"], local_date, ", ".join(reasons))

        # Apply financial penalty
        penalty_amount = 1000
        currency = user.get("currency", "INR")
        users_col.update_one(
            {"_id": user["_id"]},
            {
                "$inc": {"accountability_ledger.balance": penalty_amount},
                "$push": {"accountability_ledger.history": {
                    "at": datetime.utcnow(),
                    "amount": penalty_amount,
                    "reason": f"Streak reset: {', '.join(reasons)}"
                }}
            }
        )

        # Notify user of reset and penalty
        from chanakya.scheduler.checkpoint_runner import _execute_telegram_text
        _execute_telegram_text(
            user, 
            f"⚠️ <b>Dharma Violation</b>\n"
            f"Your Warrior Streak has been RESET to zero.\n"
            f"Reason: {', '.join(reasons)}.\n"
            f"💰 <b>Penalty:</b> {penalty_amount} {currency} added to your debt ledger.\n"
            f"Tomorrow is a new battle. Do not fail again."
        )
