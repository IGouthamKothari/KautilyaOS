"""
checkpoint_runner.py — APScheduler-based checkpoint runner.

Executes every 60 seconds. For each active user:
  - Expires WAR_MODE if war_mode_expires has passed
  - Converts UTC to user's local timezone
  - Queries due checkpoints and fires them
  - Handles deduplication (23-hour window)
  - Applies mode and activity-slot filters
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from datetime import datetime, timedelta

import pytz
from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task 22.1 — APScheduler setup
# ---------------------------------------------------------------------------

scheduler = BackgroundScheduler()


def start_runner() -> None:
    """Start the checkpoint runner. Called once at app startup."""
    scheduler.add_job(run_once, "interval", seconds=60, id="checkpoint_runner")
    scheduler.start()
    logger.info("Checkpoint runner started.")


def stop_runner() -> None:
    """Stop the checkpoint runner. Called at app shutdown."""
    scheduler.shutdown(wait=False)
    logger.info("Checkpoint runner stopped.")


# ---------------------------------------------------------------------------
# Task 22.2 — MongoDB exponential backoff wrapper
# ---------------------------------------------------------------------------


def _with_backoff(fn, *args, **kwargs):
    """
    Execute fn(*args, **kwargs) with exponential backoff on exception.

    Delays: 1s, 2s, 4s, 8s, 16s, 32s (capped).
    Logs each retry with the failure reason.
    """
    delay = 1
    max_delay = 32
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.warning(
                "DB operation attempt %d failed: %s. Retrying in %ds.",
                attempt,
                exc,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, max_delay)


# ---------------------------------------------------------------------------
# Task 22.3 — WAR_MODE expiry check
# ---------------------------------------------------------------------------


def _expire_war_mode_if_needed(user: dict) -> dict:
    """
    If user is in WAR_MODE and war_mode_expires < utcnow(), flip to NORMAL.
    Returns the updated user dict.
    """
    from chanakya.db.mongo import users

    if user.get("current_mode") == "WAR_MODE":
        expires = user.get("war_mode_expires")
        if expires and expires < datetime.utcnow():
            users.update_one(
                {"_id": user["_id"]},
                {"$set": {"current_mode": "NORMAL", "war_mode_expires": None}},
            )
            user = dict(user)
            user["current_mode"] = "NORMAL"
            user["war_mode_expires"] = None
            logger.info("WAR_MODE expired for user %s", user["_id"])
    return user


# ---------------------------------------------------------------------------
# Task 23.1 — UTC to local time conversion
# ---------------------------------------------------------------------------


def _get_local_hhmm(user: dict) -> str:
    """Convert current UTC time to user's local HH:MM."""
    tz_str = user.get("timezone", "Asia/Kolkata")
    try:
        tz = pytz.timezone(tz_str)
    except pytz.UnknownTimeZoneError:
        logger.warning(
            "Invalid timezone %r for user %s; using Asia/Kolkata",
            tz_str,
            user.get("_id"),
        )
        tz = pytz.timezone("Asia/Kolkata")

    now_local = datetime.now(tz)
    return now_local.strftime("%H:%M")


# ---------------------------------------------------------------------------
# Task 23.2 — Query due checkpoints
# ---------------------------------------------------------------------------


def _get_due_checkpoints(user: dict, local_hhmm: str) -> list[dict]:
    """Return all items due right now: base checkpoints for today's weekday
    PLUS any date-specific daily_events for today's date.

    Date-specific events at the same time as a base checkpoint take precedence
    (the base one is suppressed if override_checkpoint_id is set).
    Deduplication: skip if last_triggered within past 23 hours.
    """
    from chanakya.db.mongo import checkpoints, daily_events

    tz_str = user.get("timezone", "Asia/Kolkata")
    try:
        tz = pytz.timezone(tz_str)
    except pytz.UnknownTimeZoneError:
        tz = pytz.timezone("Asia/Kolkata")

    now_local = datetime.now(tz)
    today_str = now_local.strftime("%Y-%m-%d")          # "2026-04-25"
    today_dow = now_local.strftime("%A").lower()         # "friday"
    cutoff = datetime.utcnow() - timedelta(hours=23)

    # 1. Base checkpoints for today's weekday
    base = list(
        checkpoints.find(
            {
                "user_id": user["_id"],
                "time": local_hhmm,
                "active": True,
                "$or": [
                    {"days": {"$exists": False}},
                    {"days": today_dow},
                ],
                "$and": [{
                    "$or": [
                        {"last_triggered": {"$lt": cutoff}},
                        {"last_triggered": {"$exists": False}},
                        {"last_triggered": None},
                    ]
                }],
            }
        )
    )

    # 2. Date-specific events for today
    date_events = list(
        daily_events.find(
            {
                "user_id": user["_id"],
                "date": today_str,
                "time": local_hhmm,
                "active": True,
                "fired": {"$ne": True},
            }
        )
    )

    # 3. Collect base checkpoint IDs that are overridden by a date event
    overridden_ids = {
        e["override_checkpoint_id"]
        for e in date_events
        if e.get("override_checkpoint_id")
    }

    # 4. Filter out overridden base checkpoints
    filtered_base = [cp for cp in base if cp["_id"] not in overridden_ids]

    # 5. Normalise date_events to look like checkpoint dicts for _fire_checkpoint
    normalised_events = []
    for e in date_events:
        normalised_events.append({
            "_id": e["_id"],
            "user_id": e["user_id"],
            "time": e["time"],
            "activity": e.get("activity", ""),
            "action_type": e.get("action_type", "TELEGRAM_TEXT"),
            "priority": e.get("priority", "MEDIUM"),
            "prompt_template": e.get("note") or e.get("activity", ""),
            "active": True,
            "_is_daily_event": True,   # flag so runner marks fired=True
        })

    return filtered_base + normalised_events


# ---------------------------------------------------------------------------
# Task 24 — Checkpoint deduplication and mode filtering
# ---------------------------------------------------------------------------


def _should_skip_checkpoint(user: dict, cp: dict) -> bool:
    """
    Return True if this checkpoint should be skipped based on mode/slot filters.
    Deduplication (23h) is handled in the query itself.
    """
    current_mode = user.get("current_mode", "NORMAL")
    priority = cp.get("priority", "MEDIUM")
    action_type = cp.get("action_type", "TELEGRAM_TEXT")
    activity_slot = user.get("current_activity", "FREE_TIME")

    # 24.2: WAR_MODE blocks MEDIUM and LOW
    if current_mode == "WAR_MODE" and priority in ("MEDIUM", "LOW"):
        return True

    # 24.3: AWAY blocks non-CRITICAL
    if current_mode == "AWAY" and priority != "CRITICAL":
        return True

    # 24.4: SLEEP blocks CHECK_IN
    if activity_slot == "SLEEP" and action_type == "CHECK_IN":
        return True

    return False


# ---------------------------------------------------------------------------
# Task 25 — Checkpoint firing helpers
# ---------------------------------------------------------------------------


def _execute_telegram_text(user: dict, text: str) -> None:
    """Send a Telegram text message to the user (fire-and-forget via asyncio)."""

    async def _send() -> None:
        try:
            from telegram import Bot

            from chanakya.config import TELEGRAM_BOT_TOKEN

            bot = Bot(token=TELEGRAM_BOT_TOKEN)
            await bot.send_message(chat_id=user["telegram_id"], text=text)
            logger.info("Telegram text sent to user %s", user.get("_id"))
        except Exception as exc:
            logger.error(
                "Failed to send Telegram text to user %s: %s", user.get("_id"), exc
            )

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_send())
        else:
            loop.run_until_complete(_send())
    except Exception as exc:
        logger.error(
            "Failed to schedule Telegram send for user %s: %s", user.get("_id"), exc
        )


def _execute_telegram_voice(user: dict, cp: dict, rendered_prompt: str) -> None:
    """Synthesise via ElevenLabs and send voice message to Telegram.

    Falls back to plain text if ElevenLabs synthesis fails (Req 14.3).
    """
    from chanakya.integrations.elevenlabs_client import (
        ElevenLabsClient,
        ElevenLabsSynthesisError,
    )

    voice_id = user.get("elevenlabs_voice_id", "")

    if not voice_id:
        _execute_telegram_text(user, rendered_prompt)
        return

    try:
        el_client = ElevenLabsClient()
        audio_bytes = el_client.synthesise(rendered_prompt, voice_id)
    except ElevenLabsSynthesisError as exc:
        logger.warning("ElevenLabs synthesis failed: %s. Sending plain text.", exc)
        _execute_telegram_text(user, rendered_prompt)
        return

    async def _send_voice() -> None:
        try:
            from telegram import Bot

            from chanakya.config import TELEGRAM_BOT_TOKEN

            bot = Bot(token=TELEGRAM_BOT_TOKEN)
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "chanakya_voice.mp3"
            await bot.send_voice(chat_id=user["telegram_id"], voice=audio_file)
            logger.info("Voice message sent to user %s", user.get("_id"))
        except Exception as exc:
            logger.error(
                "Failed to send voice to user %s: %s", user.get("_id"), exc
            )
            _execute_telegram_text(user, rendered_prompt)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_send_voice())
        else:
            loop.run_until_complete(_send_voice())
    except Exception as exc:
        logger.error(
            "Failed to schedule voice send for user %s: %s", user.get("_id"), exc
        )


def _execute_call_action(
    user: dict, cp: dict, rendered_prompt: str, log_id
) -> str | None:
    """Execute CALL action as a two-way voice conversation.

    Creates a voice session so Twilio uses <Gather> for back-and-forth dialogue.
    Falls back to Telegram text if Twilio or WEBHOOK_URL is unavailable.
    """
    from chanakya.config import WEBHOOK_URL
    from chanakya.integrations.twilio_client import TwilioClient, TwilioError
    from chanakya.integrations.twilio_webhooks import (
        create_voice_session,
        log_twilio_fallback,
        synthesize_call_opening,
    )

    if not WEBHOOK_URL:
        logger.warning(
            "WEBHOOK_URL not set — cannot place call for user %s. Falling back to Telegram.",
            user.get("_id"),
        )
        _execute_telegram_text(user, rendered_prompt)
        return None

    phone = user.get("phone", "")
    if not phone:
        logger.warning(
            "No phone number for user %s — falling back to Telegram text.",
            user.get("_id"),
        )
        _execute_telegram_text(user, rendered_prompt)
        return None

    session_id = str(log_id) if log_id else None
    if session_id:
        create_voice_session(
            session_id=session_id,
            user_id=str(user["_id"]),
            context=rendered_prompt,
            conversation_context=user.get("conversation_context") or "",
            audio_bytes=synthesize_call_opening(rendered_prompt),
        )

    twiml_url = f"{WEBHOOK_URL.rstrip('/')}/twilio/voice/{session_id}"

    try:
        twilio_client = TwilioClient()
        call_sid = twilio_client.make_call(to=phone, twiml_url=twiml_url)
        logger.info("Call initiated for user %s, sid=%s", user.get("_id"), call_sid)
        return call_sid
    except TwilioError as exc:
        logger.warning(
            "Twilio call failed for user %s: %s. Falling back to Telegram text.",
            user.get("_id"),
            exc,
        )
        log_twilio_fallback(user.get("_id"), str(cp["_id"]), str(exc))
        _execute_telegram_text(user, rendered_prompt)
        return None


def _fire_checkpoint(user: dict, cp: dict) -> None:
    """
    Fire a single checkpoint:
    - Render the prompt
    - Execute the action (CALL, TELEGRAM_TEXT, TELEGRAM_VOICE, IMAGE_DEMAND)
    - Update last_triggered
    - Insert interaction_log
    """
    from chanakya.agent.context_assembler import (
        NoTemplateFoundError,
        get_prompt_templates,
        render_template,
    )
    from chanakya.db.mongo import checkpoints as cp_col
    from chanakya.db.mongo import interaction_logs

    action_type = cp.get("action_type", "TELEGRAM_TEXT")

    # Render prompt — try template collection first, fall back to raw prompt_template
    prompt_text = cp.get("prompt_template", "")
    # Use display_name for human-facing text if available
    display_name = cp.get("display_name") or cp.get("activity", "").replace("_", " ").title()
    try:
        activity_slot = user.get("current_activity", "FREE_TIME")
        templates = get_prompt_templates(activity_slot, "CHECKPOINT")
        template_text = (
            templates.get("NEUTRAL") or templates.get("HARSH") or prompt_text
        )
        context = {
            "name": user.get("name", ""),
            "streak": user.get("streak_count", 0),
            "current_mode": user.get("current_mode", "NORMAL"),
            "activity": display_name,
        }
        rendered_prompt = render_template(template_text, context)
    except (NoTemplateFoundError, Exception):
        rendered_prompt = prompt_text  # Fall back to raw prompt_template

    # Insert interaction_log FIRST so we have log_id for the Twilio TwiML URL
    now = datetime.utcnow()
    log_doc = {
        "user_id": user["_id"],
        "checkpoint_id": cp["_id"],
        "timestamp": now,
        "trigger_type": "SCHEDULED",
        "channel": action_type,
        "message_sent": rendered_prompt,
        "ai_evaluation": {
            "verdict": None,
            "confidence": None,
            "reasoning": None,
        },
        "created_at": now,
    }

    log_id = None
    try:
        log_result = interaction_logs.insert_one(log_doc)
        log_id = log_result.inserted_id
    except Exception as exc:
        logger.error(
            "Failed to insert interaction_log for checkpoint %s: %s", cp["_id"], exc
        )

    # Execute action
    call_sid = None

    if action_type == "CALL":
        call_sid = _execute_call_action(user, cp, rendered_prompt, log_id)

    elif action_type == "TELEGRAM_TEXT":
        _execute_telegram_text(user, rendered_prompt)

    elif action_type == "TELEGRAM_VOICE":
        _execute_telegram_voice(user, cp, rendered_prompt)

    elif action_type == "IMAGE_DEMAND":
        # Send text requesting a photo
        _execute_telegram_text(user, rendered_prompt)

    else:
        logger.warning(
            "Unknown action_type %r for checkpoint %s", action_type, cp["_id"]
        )

    # Update last_triggered (base checkpoints) or fired=True (daily events)
    try:
        if cp.get("_is_daily_event"):
            from chanakya.db.mongo import daily_events as de_col
            de_col.update_one({"_id": cp["_id"]}, {"$set": {"fired": True, "fired_at": now}})
        else:
            cp_col.update_one({"_id": cp["_id"]}, {"$set": {"last_triggered": now}})
    except Exception as exc:
        logger.error("Failed to update trigger state for %s: %s", cp["_id"], exc)

    # Update log with call_sid if applicable
    if call_sid and log_id:
        try:
            interaction_logs.update_one(
                {"_id": log_id},
                {"$set": {"twilio_call_sid": call_sid}},
            )
        except Exception as exc:
            logger.error(
                "Failed to update call_sid in log %s: %s", log_id, exc
            )


# ---------------------------------------------------------------------------
# Task 25 — Per-user processing
# ---------------------------------------------------------------------------


def _process_user(user: dict) -> None:
    """Process all due checkpoints for a single user."""
    # 22.3: Expire WAR_MODE if needed
    user = _expire_war_mode_if_needed(user)

    # 23.1: Get local HH:MM
    local_hhmm = _get_local_hhmm(user)

    tz = pytz.timezone(user.get("timezone", "Asia/Kolkata"))
    now_local = datetime.now(tz)
    local_date = now_local.strftime("%Y-%m-%d")
    day_of_week = now_local.strftime("%A").lower()  # "sunday", "monday", etc.

    # 23.2: Query due checkpoints
    due_checkpoints = _get_due_checkpoints(user, local_hhmm)

    for cp in due_checkpoints:
        # 24: Mode/slot filtering
        if _should_skip_checkpoint(user, cp):
            logger.debug(
                "Skipping checkpoint %s for user %s (mode/slot filter)",
                cp["_id"],
                user["_id"],
            )
            continue

        # 25: Fire the checkpoint
        _fire_checkpoint(user, cp)

    # Task 28: Check and fire scheduled check-ins for this minute
    import asyncio

    from chanakya.scheduler.daily_features import (
        get_or_create_checkin_schedule,
        fire_checkin,
        should_fire_checkin,
        fire_weekly_review,
    )

    checkin_times = get_or_create_checkin_schedule(user, local_date)

    if local_hhmm in checkin_times and should_fire_checkin(user):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(fire_checkin(user))
            else:
                loop.run_until_complete(fire_checkin(user))
        except Exception as exc:
            logger.error(
                "Failed to fire check-in for user %s: %s", user["_id"], exc
            )

    # Task 30: Fire weekly review on Sunday at EOD time
    eod_time = user.get("eod_time", "21:00")
    if day_of_week == "sunday" and local_hhmm == eod_time:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(fire_weekly_review(user))
            else:
                loop.run_until_complete(fire_weekly_review(user))
            logger.info("Weekly review triggered for user %s", user["_id"])
        except Exception as exc:
            logger.error("Failed to fire weekly review for user %s: %s", user["_id"], exc)

    # Task 32: Daily Streak Update at EOD time
    if local_hhmm == eod_time:
        try:
            from chanakya.scheduler.daily_features import calculate_daily_scores
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(calculate_daily_scores(user, today_str))
            else:
                loop.run_until_complete(calculate_daily_scores(user, today_str))
        except Exception as exc:
            logger.error("Failed to calculate daily streak for user %s: %s", user["_id"], exc)


# ---------------------------------------------------------------------------
# Task 25 — Main run_once entry point
# ---------------------------------------------------------------------------


def run_once() -> None:
    """
    Single execution of the checkpoint runner.
    Called every 60 seconds by APScheduler.
    """
    from chanakya.db.mongo import users as users_col

    logger.debug("Checkpoint runner tick.")

    try:
        active_users = list(_with_backoff(users_col.find, {"active": True}))
    except Exception as exc:
        logger.error("Failed to fetch active users: %s", exc)
        return

    for user in active_users:
        try:
            _process_user(user)
        except Exception as exc:
            logger.error(
                "Error processing user %s: %s",
                user.get("_id"),
                exc,
                exc_info=True,
            )

    # Check ElevenLabs credit alert
    from chanakya.integrations.elevenlabs_client import get_and_clear_low_credit_alert

    if get_and_clear_low_credit_alert():
        logger.warning(
            "ElevenLabs credit low — alert should be sent to user."
        )
        # TODO: send Telegram alert to admin/user (requires bot reference)
