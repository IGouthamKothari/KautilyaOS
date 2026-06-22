"""
background_jobs.py — Periodic background jobs for the Darbar system.

Registers APScheduler jobs for:
  - Learning extractor (every 2 hours per active user)
  - Goal sentinel (every 6 hours per active user)

These never block user-facing responses. They run only when DARBAR_ENABLED=true.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler

from chanakya.db.mongo import users

logger = logging.getLogger(__name__)

_scheduler = BackgroundScheduler()


def start_darbar_jobs() -> None:
    """Start the Darbar background scheduler."""
    from chanakya.config import DARBAR_ENABLED

    if not DARBAR_ENABLED:
        logger.info("Darbar disabled — background jobs not started.")
        return

    _scheduler.add_job(
        _run_learning_for_all,
        "interval",
        hours=2,
        id="darbar_learning",
        replace_existing=True,
    )
    _scheduler.add_job(
        _run_goal_sentinel_for_all,
        "interval",
        hours=6,
        id="darbar_goal_sentinel",
        replace_existing=True,
    )
    _scheduler.add_job(
        _run_gmail_triage_for_all,
        "interval",
        minutes=15,
        id="gmail_triage",
        replace_existing=True,
    )
    _scheduler.add_job(
        _run_war_mode_expiry_check,
        "interval",
        minutes=5,
        id="war_mode_expiry",
        replace_existing=True,
    )
    _scheduler.add_job(
        _run_daily_calendar_agenda,
        "cron",
        hour=7,
        minute=30,
        id="daily_calendar_agenda",
        replace_existing=True,
    )

    if not _scheduler.running:
        _scheduler.start()

    logger.info("Darbar background jobs started (learning: 2h, sentinel: 6h, gmail: 15m, war_mode: 5m, agenda: 07:30).")


def stop_darbar_jobs() -> None:
    """Stop the Darbar background scheduler."""
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Darbar background jobs stopped.")


def _run_learning_for_all() -> None:
    """Run learning extraction for all active users who need it."""
    from chanakya.darbar.learning_extractor import should_run_learning, run_learning_cycle

    active_users = list(users.find({"active": True}))
    for user in active_users:
        uid = user["_id"]
        if should_run_learning(uid):
            try:
                _run_async(run_learning_cycle(uid))
            except Exception as exc:
                logger.warning("Learning cycle failed for user %s: %s", uid, exc)


def _run_goal_sentinel_for_all() -> None:
    """Run goal sentinel for all active users."""
    from chanakya.darbar.goal_sentinel import run_goal_sentinel

    active_users = list(users.find({"active": True}))
    for user in active_users:
        uid = user["_id"]
        try:
            _run_async(run_goal_sentinel(uid))
        except Exception as exc:
            logger.warning("Goal sentinel failed for user %s: %s", uid, exc)


def _run_gmail_triage_for_all() -> None:
    """Poll Gmail for high-priority unread emails and create checkpoints."""
    from chanakya.integrations.google_auth import is_connected
    from chanakya.integrations.google_gmail import triage_for_checkpoints
    from chanakya.db.mongo import checkpoints, interaction_logs
    from datetime import datetime, timedelta

    active_users = list(users.find({"active": True}))
    for user in active_users:
        uid = user["_id"]
        if not is_connected(uid):
            continue
        try:
            emails = triage_for_checkpoints(uid, max_results=10)
            for email in emails:
                msg_id = email["message_id"]
                # Skip if already created a checkpoint for this email
                existing = interaction_logs.find_one({
                    "user_id": uid,
                    "trigger_type": "GMAIL_TRIAGE",
                    "payload.gmail_message_id": msg_id,
                })
                if existing:
                    continue

                subject = email["subject"][:120]
                sender = email["from"]
                snippet = email["snippet"][:200]
                prompt = (
                    f"Urgent email needs your attention:\n"
                    f"From: {sender}\nSubject: {subject}\n{snippet}"
                )

                # Log as a scheduled interaction — Chanakya will nudge
                interaction_logs.insert_one({
                    "user_id": uid,
                    "timestamp": datetime.utcnow(),
                    "trigger_type": "GMAIL_TRIAGE",
                    "channel": "SYSTEM",
                    "message_sent": prompt,
                    "user_response": None,
                    "ai_evaluation": {"verdict": None, "confidence": None, "reasoning": None},
                    "payload": {"gmail_message_id": msg_id, "thread_id": email.get("thread_id")},
                    "created_at": datetime.utcnow(),
                })

                # Send Telegram notification
                try:
                    _run_async(_send_gmail_nudge(user, subject, sender, snippet))
                except Exception as exc:
                    logger.warning("Gmail nudge send failed for user %s: %s", uid, exc)

                # Mark as read so we don't nudge again on the next poll
                try:
                    from chanakya.integrations.google_gmail import mark_read
                    mark_read(uid, msg_id)
                except Exception as exc:
                    logger.warning("Gmail mark-read failed for msg %s: %s", msg_id, exc)

            logger.debug("Gmail triage done for user %s: %d important emails", uid, len(emails))
        except Exception as exc:
            logger.warning("Gmail triage failed for user %s: %s", uid, exc)


async def _send_gmail_nudge(user: dict, subject: str, sender: str, snippet: str) -> None:
    """Send a Telegram message alerting the user about an important email."""
    from chanakya.bot.telegram_bot import generic_process_message
    msg = (
        f"📧 Important email needs attention:\n"
        f"From: {sender}\n"
        f"Subject: {subject}\n"
        f"---\n{snippet[:150]}\n\n"
        f"Reply here to have Chanakya help you respond or take action."
    )
    await generic_process_message(user, msg, channel="GMAIL_TRIAGE")


def _run_war_mode_expiry_check() -> None:
    """Auto-expire WAR_MODE after its timer runs out."""
    from datetime import timezone
    active_users = list(users.find({"active": True, "current_mode": "WAR_MODE"}))
    for user in active_users:
        expires = user.get("war_mode_expires")
        if not expires:
            continue
        now = datetime.utcnow()
        # Handle both tz-aware and naive datetimes
        if hasattr(expires, "tzinfo") and expires.tzinfo is not None:
            now = datetime.now(timezone.utc)
        if now >= expires:
            users.update_one(
                {"_id": user["_id"]},
                {"$set": {"current_mode": "NORMAL", "war_mode_expires": None, "updated_at": datetime.utcnow()}}
            )
            logger.info("WAR_MODE expired for user %s — reverted to NORMAL", user["_id"])
            try:
                _run_async(_send_war_mode_expired_nudge(user))
            except Exception as exc:
                logger.warning("WAR_MODE expiry nudge failed for user %s: %s", user["_id"], exc)


async def _send_war_mode_expired_nudge(user: dict) -> None:
    from chanakya.bot.telegram_bot import generic_process_message
    await generic_process_message(
        user,
        "⚔️ WAR MODE has ended. You are back to NORMAL mode. Rest, recover, and reload.",
        channel="SYSTEM",
    )


def _run_daily_calendar_agenda() -> None:
    """Push today's Google Calendar agenda to users via Telegram each morning."""
    active_users = list(users.find({"active": True}))
    for user in active_users:
        uid = user["_id"]
        try:
            from chanakya.integrations.google_auth import is_connected
            if not is_connected(uid):
                continue
            from chanakya.integrations.google_calendar import list_events
            events = list_events(uid, days_ahead=1)
            if not events:
                msg = "📅 No events on your calendar today. A clear day — use it wisely."
            else:
                lines = ["📅 Today's agenda:"]
                for e in events:
                    time_part = e.get("start", "")
                    if "T" in time_part:
                        time_part = time_part.split("T")[1][:5]
                    loc = f" @ {e['location']}" if e.get("location") else ""
                    lines.append(f"  • {time_part} — {e['title']}{loc}")
                msg = "\n".join(lines)
            _run_async(_send_agenda_nudge(user, msg))
        except Exception as exc:
            logger.warning("Daily agenda failed for user %s: %s", uid, exc)


async def _send_agenda_nudge(user: dict, msg: str) -> None:
    from chanakya.bot.telegram_bot import generic_process_message
    await generic_process_message(user, msg, channel="SYSTEM")


def _run_async(coro):
    """Run an async coroutine from a sync scheduler thread."""
    from chanakya.async_utils import run_async
    run_async(coro)
