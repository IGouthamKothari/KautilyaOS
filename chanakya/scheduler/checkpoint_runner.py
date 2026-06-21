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
    """Start the scheduler and load initial jobs."""
    if not scheduler.running:
        scheduler.start()
    refresh_all_schedules()
    recover_missed_checkpoints()
    # Add a safety periodic refresh every hour in case of manual DB edits
    scheduler.add_job(refresh_all_schedules, "interval", hours=1, id="periodic_refresh")
    logger.info("Precision Checkpoint Runner started with event-driven triggers.")


def stop_runner() -> None:
    """Stop the checkpoint runner."""
    scheduler.shutdown(wait=False)
    logger.info("Checkpoint runner stopped.")


def sync_checkpoint(user: dict, cp: dict) -> None:
    """Surgically add/update a single recurring checkpoint in the scheduler."""
    try:
        _schedule_checkpoint(user, cp)
    except Exception as exc:
        logger.warning("sync_checkpoint failed for %s: %s", cp.get("_id"), exc)


def unsync_checkpoint(cp_id) -> None:
    """Remove a single checkpoint/event job from the scheduler immediately."""
    job_id = f"cp_{cp_id}"
    try:
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)
            logger.debug("Unsynced scheduler job %s", job_id)
    except Exception as exc:
        logger.warning("unsync_checkpoint failed for %s: %s", cp_id, exc)


def sync_event(user: dict, event: dict) -> None:
    """Schedule a one-time daily event using DateTrigger (today only, future time only)."""
    from apscheduler.triggers.date import DateTrigger
    import pytz

    time_str = event.get("time", "")
    event_date = event.get("date", "")
    tz = pytz.timezone(user.get("timezone", "Asia/Kolkata"))
    today_str = datetime.now(tz).strftime("%Y-%m-%d")

    if event_date != today_str or event.get("fired"):
        return  # Future date — periodic refresh handles it on the right day

    try:
        hour, minute = map(int, time_str.split(":"))
    except (ValueError, AttributeError):
        return

    now_local = datetime.now(tz)
    fire_time = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if fire_time <= now_local:
        return  # Already passed

    ov_cp = {
        "_id": event["_id"],
        "time": time_str,
        "action_type": event.get("action_type", "TELEGRAM_TEXT"),
        "priority": event.get("priority", "MEDIUM"),
        "prompt_template": event.get("note") or event.get("activity", ""),
        "display_name": event.get("display_name") or event.get("activity", ""),
        "is_daily_event": True,
        "expects_response": event.get("expects_response", False),
    }

    def fire_wrapper():
        from chanakya.db.mongo import users as users_col
        fresh_user = users_col.find_one({"_id": user["_id"]})
        if not fresh_user or not fresh_user.get("active"):
            return
        if _should_skip_checkpoint(fresh_user, ov_cp):
            return
        _fire_checkpoint(fresh_user, ov_cp)

    job_id = f"cp_{event['_id']}"
    try:
        scheduler.add_job(
            fire_wrapper,
            DateTrigger(run_date=fire_time, timezone=tz),
            id=job_id,
            replace_existing=True,
        )
        logger.info("Synced one-time event %s at %s today", event["_id"], fire_time.strftime("%H:%M"))
    except Exception as exc:
        logger.warning("sync_event failed for %s: %s", event.get("_id"), exc)


def refresh_all_schedules() -> None:
    """Clear all scheduled checkpoint jobs and rebuild from MongoDB."""
    from chanakya.db.mongo import checkpoints, users as users_col
    
    # 1. Clear existing checkpoint jobs (prefix "cp_")
    for job in scheduler.get_jobs():
        if job.id.startswith("cp_"):
            scheduler.remove_job(job.id)
    
    # 2. Load all active checkpoints for all active users
    try:
        from datetime import date
        active_users = list(users_col.find({"active": True}))
        for user in active_users:
            # A. Base checkpoints
            user_cps = list(checkpoints.find({"user_id": user["_id"], "active": True}))
            
            # B. Daily overrides for today
            tz = pytz.timezone(user.get("timezone", "Asia/Kolkata"))
            today_str = datetime.now(tz).strftime("%Y-%m-%d")
            from chanakya.db.mongo import daily_events
            overrides = list(daily_events.find({"user_id": user["_id"], "date": today_str, "active": True, "fired": {"$ne": True}}))
            
            # Track overridden base checkpoints to avoid double-firing
            overridden_ids = {ov.get("override_checkpoint_id") for ov in overrides if ov.get("override_checkpoint_id")}
            
            # Schedule base checkpoints (unless overridden)
            for cp in user_cps:
                if cp["_id"] not in overridden_ids:
                    _schedule_checkpoint(user, cp)
            
            # Schedule one-time overrides
            for ov in overrides:
                # Normalise override to look like a checkpoint for the scheduler
                ov_cp = {
                    "_id": ov["_id"],
                    "time": ov["time"],
                    "action_type": ov.get("action_type", "TELEGRAM_TEXT"),
                    "priority": ov.get("priority", "HIGH"),
                    "prompt_template": ov.get("note") or ov.get("activity", "Daily Event"),
                    "display_name": ov.get("activity", "One-time Task"),
                    "is_daily_event": True
                }
                _schedule_checkpoint(user, ov_cp)
                
        logger.info("Synchronized %d users' schedules (including daily overrides) into high-precision triggers.", len(active_users))
    except Exception as e:
        logger.error("Failed to refresh schedules from DB: %s", e)


def recover_missed_checkpoints() -> None:
    """Fire any checkpoints that were missed while the server was down.

    For each active user, checks if any recurring checkpoint's last_triggered
    is older than expected (missed its window). Fires them once with a
    max staleness of 60 minutes — anything older is considered abandoned.
    """
    from chanakya.db.mongo import checkpoints, users as users_col

    MAX_STALENESS = timedelta(minutes=60)

    try:
        active_users = list(users_col.find({"active": True}))
        recovered = 0

        for user in active_users:
            tz = pytz.timezone(user.get("timezone", "Asia/Kolkata"))
            now_local = datetime.now(tz)
            now_utc = datetime.utcnow()

            user_cps = list(checkpoints.find({"user_id": user["_id"], "active": True}))

            for cp in user_cps:
                last_triggered = cp.get("last_triggered")
                if not last_triggered:
                    continue

                time_str = cp.get("time", "")
                try:
                    hour, minute = map(int, time_str.split(":"))
                except (ValueError, AttributeError):
                    continue

                # Check if this checkpoint should have fired today
                expected_today = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)

                # Day-of-week filter
                if cp.get("days"):
                    day_names = [d.lower() for d in cp["days"]]
                    current_day = now_local.strftime("%A").lower()
                    if current_day not in day_names:
                        continue

                # Only recover if: expected time has passed, it wasn't triggered today,
                # and it's within the staleness window
                if expected_today > now_local:
                    continue  # Not due yet today

                time_since_expected = now_local - expected_today
                if time_since_expected > MAX_STALENESS:
                    continue  # Too old, skip

                # Check if it was already triggered today (within 23h dedup window)
                if last_triggered and (now_utc - last_triggered) < timedelta(hours=23):
                    # Already fired within the dedup window
                    last_triggered_local = last_triggered.replace(tzinfo=pytz.utc).astimezone(tz)
                    if last_triggered_local.date() == now_local.date():
                        continue

                # This checkpoint was missed — fire it now
                if _should_skip_checkpoint(user, cp):
                    continue

                logger.info(
                    "Recovering missed checkpoint %s (expected %s, last_triggered %s)",
                    cp["_id"], expected_today.strftime("%H:%M"), last_triggered
                )
                try:
                    _fire_checkpoint(user, cp)
                    recovered += 1
                except Exception as exc:
                    logger.error("Failed to recover checkpoint %s: %s", cp["_id"], exc)

        if recovered:
            logger.info("Recovered %d missed checkpoints on startup.", recovered)
    except Exception as e:
        logger.error("recover_missed_checkpoints failed: %s", e)


def _schedule_checkpoint(user: dict, cp: dict) -> None:
    """Schedule a single checkpoint using a CronTrigger."""
    from apscheduler.triggers.cron import CronTrigger
    
    time_str = cp["time"] # "HH:MM"
    try:
        hour, minute = map(int, time_str.split(":"))
    except ValueError:
        logger.warning("Invalid time format for checkpoint %s: %s", cp["_id"], time_str)
        return

    # Handle day-of-week filtering if present
    day_of_week = "*"
    if cp.get("days"):
        # Map full names ("monday") to APScheduler 3-letter abbreviations or numbers
        day_map = {
            "monday": "mon", "tuesday": "tue", "wednesday": "wed",
            "thursday": "thu", "friday": "fri", "saturday": "sat", "sunday": "sun"
        }
        days = [day_map.get(d.lower(), d[:3].lower()) for d in cp["days"]]
        day_of_week = ",".join(days)

    job_id = f"cp_{cp['_id']}"
    
    # We wrap the firing logic to include mode/slot filtering at runtime
    def fire_wrapper():
        # Fetch fresh user doc for mode/slot check
        from chanakya.db.mongo import users
        fresh_user = users.find_one({"_id": user["_id"]})
        if not fresh_user or not fresh_user.get("active"):
            return
            
        if _should_skip_checkpoint(fresh_user, cp):
            logger.info("Skipping triggered checkpoint %s (mode/slot filter active)", cp["_id"])
            return
            
        _fire_checkpoint(fresh_user, cp)

    scheduler.add_job(
        fire_wrapper,
        CronTrigger(hour=hour, minute=minute, day_of_week=day_of_week, timezone=user.get("timezone", "Asia/Kolkata")),
        id=job_id,
        replace_existing=True,
        misfire_grace_time=300 # Allow 5m grace for server restarts
    )


# ---------------------------------------------------------------------------
# firing logic (Unchanged, but ensure it's accessible)
# ---------------------------------------------------------------------------

def _expire_war_mode_if_needed(user: dict) -> dict:
    from chanakya.db.mongo import users
    if user.get("current_mode") == "WAR_MODE":
        expires = user.get("war_mode_expires")
        if expires and expires < datetime.utcnow():
            users.update_one({"_id": user["_id"]}, {"$set": {"current_mode": "NORMAL", "war_mode_expires": None}})
            user = dict(user)
            user["current_mode"] = "NORMAL"
            user["war_mode_expires"] = None
            return user
    return user

def _should_skip_checkpoint(user: dict, cp: dict) -> bool:
    current_mode = user.get("current_mode", "NORMAL")
    priority = cp.get("priority", "MEDIUM")
    action_type = cp.get("action_type", "TELEGRAM_TEXT")
    activity_slot = user.get("current_activity", "FREE_TIME")
    if current_mode == "WAR_MODE" and priority in ("MEDIUM", "LOW"): return True
    if current_mode == "AWAY" and priority != "CRITICAL": return True
    if activity_slot == "SLEEP" and action_type == "CHECK_IN": return True
    return False

def _fire_checkpoint(user: dict, cp: dict) -> None:
    from chanakya.agent.context_assembler import get_prompt_templates, render_template
    from chanakya.db.mongo import checkpoints as cp_col, interaction_logs
    
    action_type = cp.get("action_type", "TELEGRAM_TEXT")
    prompt_text = cp.get("prompt_template", "")
    display_name = cp.get("display_name") or cp.get("activity", "").replace("_", " ").title()
    
    try:
        activity_slot = user.get("current_activity", "FREE_TIME")
        templates = get_prompt_templates(activity_slot, "CHECKPOINT")
        template_text = templates.get("NEUTRAL") or templates.get("HARSH") or prompt_text
        rendered_prompt = render_template(template_text, {
            "name": user.get("name", ""),
            "streak": user.get("streak_count", 0),
            "current_mode": user.get("current_mode", "NORMAL"),
            "activity": display_name,
        })
    except:
        rendered_prompt = prompt_text

    now = datetime.utcnow()
    log_id = interaction_logs.insert_one({
        "user_id": user["_id"], "checkpoint_id": cp["_id"], "timestamp": now,
        "trigger_type": "SCHEDULED", "channel": action_type, "message_sent": rendered_prompt,
        "ai_evaluation": {"verdict": None}, "created_at": now
    }).inserted_id

    call_sid = None
    if action_type == "CALL": call_sid = _execute_call_action(user, cp, rendered_prompt, log_id)
    elif action_type == "TELEGRAM_TEXT": _execute_telegram_text(user, rendered_prompt)
    elif action_type == "TELEGRAM_VOICE": _execute_telegram_voice(user, cp, rendered_prompt)
    elif action_type == "IMAGE_DEMAND": _execute_telegram_text(user, rendered_prompt)
    else:
        logger.warning("Unknown action_type %r for checkpoint %s", action_type, cp["_id"])

    # Schedule engagement nudge only for checkpoints that expect a response
    expects_response = cp.get("expects_response", True) and not cp.get("is_daily_event", False)
    if expects_response:
        from chanakya.scheduler.task_runner import schedule_engagement_nudge
        interval = cp.get("persistent_nudge_interval_minutes", 5) if cp.get("persistent_nudge") else 5
        schedule_engagement_nudge(log_id, interval)

    if cp.get("is_daily_event"):
        from chanakya.db.mongo import daily_events as de_col
        de_col.update_one({"_id": cp["_id"]}, {"$set": {"fired": True, "fired_at": now}})
    else:
        cp_col.update_one({"_id": cp["_id"]}, {"$set": {"last_triggered": now}})
    if call_sid and log_id:
        interaction_logs.update_one({"_id": log_id}, {"$set": {"twilio_call_sid": call_sid}})

def _run_async(coro):
    """Run a coroutine from a sync background thread (thread-safe)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        asyncio.ensure_future(coro)
    else:
        try:
            main_loop = asyncio._get_running_loop()
            if main_loop:
                asyncio.run_coroutine_threadsafe(coro, main_loop)
            else:
                asyncio.run(coro)
        except Exception:
            asyncio.run(coro)

def _execute_telegram_text(user: dict, text: str) -> None:
    async def _send():
        from telegram import Bot
        from chanakya.config import TELEGRAM_BOT_TOKEN
        await Bot(token=TELEGRAM_BOT_TOKEN).send_message(chat_id=user["telegram_id"], text=text)
    _run_async(_send())

def _execute_telegram_voice(user: dict, cp: dict, text: str) -> None:
    from chanakya.integrations.elevenlabs_client import ElevenLabsClient
    voice_id = user.get("elevenlabs_voice_id")
    if not voice_id: _execute_telegram_text(user, text); return
    try:
        audio = ElevenLabsClient().synthesise(text, voice_id)
        async def _send():
            from telegram import Bot
            from chanakya.config import TELEGRAM_BOT_TOKEN
            await Bot(token=TELEGRAM_BOT_TOKEN).send_voice(chat_id=user["telegram_id"], voice=io.BytesIO(audio))
        _run_async(_send())
    except: _execute_telegram_text(user, text)

def _execute_call_action(user: dict, cp: dict, text: str, log_id) -> str | None:
    from chanakya.config import WEBHOOK_URL
    from chanakya.integrations.twilio_client import TwilioClient
    from chanakya.integrations.twilio_webhooks import create_voice_session, synthesize_call_opening
    if not WEBHOOK_URL or not user.get("phone"): _execute_telegram_text(user, text); return None
    phone = user["phone"] if user["phone"].startswith("+") else "+91" + user["phone"]
    if log_id: create_voice_session(str(log_id), str(user["_id"]), text, user.get("conversation_context", ""), synthesize_call_opening(text))
    try: return TwilioClient().make_call(to=phone, twiml_url=f"{WEBHOOK_URL.rstrip('/')}/twilio/voice/{log_id}")
    except: _execute_telegram_text(user, text); return None
