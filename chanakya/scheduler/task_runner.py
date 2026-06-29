"""
task_runner.py — APScheduler-based Task Manager.

Processes background tasks (COMMITMENT_CHECK) and nudge scheduling.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from bson import ObjectId

from chanakya.db.mongo import agent_tasks, users, interaction_logs

logger = logging.getLogger(__name__)

task_scheduler = BackgroundScheduler()

def start_task_runner() -> None:
    """Start the task scheduler."""
    if not task_scheduler.running:
        task_scheduler.start()
    # On startup, recover any PENDING tasks and schedule them
    recover_pending_tasks()
    logger.info("Precision Task Runner started.")


def stop_task_runner() -> None:
    """Stop the task runner."""
    task_scheduler.shutdown(wait=False)
    logger.info("Task runner stopped.")


def schedule_agent_task(task_id: ObjectId, run_at: datetime = None) -> None:
    """Schedule a specific agent task for execution."""
    if run_at is None:
        run_at = datetime.now()
    elif run_at.tzinfo is None:
        # Convert naive UTC datetime to local time for APScheduler
        import pytz
        run_at = run_at.replace(tzinfo=pytz.utc).astimezone(tz=None).replace(tzinfo=None)

    task_scheduler.add_job(
        _execute_task_by_id,
        "date",
        run_date=run_at,
        args=[task_id],
        id=f"task_{task_id}",
        replace_existing=True,
        misfire_grace_time=300
    )
    logger.info("Task %s scheduled for %s", task_id, run_at)


def schedule_engagement_nudge(log_id: ObjectId, delay_minutes: int) -> None:
    """Schedule a follow-up nudge for a specific interaction."""
    run_at = datetime.utcnow() + timedelta(minutes=delay_minutes)
    task_scheduler.add_job(
        _fire_nudge,
        "date",
        run_date=run_at,
        args=[log_id],
        id=f"nudge_{log_id}",
        replace_existing=True
    )
    logger.info("Nudge for interaction %s scheduled for +%dm", log_id, delay_minutes)


def cancel_nudge(log_id: ObjectId) -> None:
    """Cancel a pending nudge if the user has responded."""
    job_id = f"nudge_{log_id}"
    if task_scheduler.get_job(job_id):
        task_scheduler.remove_job(job_id)
        logger.info("Nudge %s cancelled due to user response.", log_id)


def recover_pending_tasks() -> None:
    """Find PENDING tasks in DB and schedule them (useful after restart).
    Also mark stale RUNNING tasks as FAILED — these were mid-execution when
    the server crashed and their Twilio callbacks will never arrive.
    """
    pending = list(agent_tasks.find({"status": "PENDING"}))
    for t in pending:
        schedule_agent_task(t["_id"])

    stale_running = list(agent_tasks.find({
        "status": "RUNNING",
        "last_attempted_at": {"$lt": datetime.utcnow() - timedelta(minutes=30)}
    }))
    for t in stale_running:
        agent_tasks.update_one(
            {"_id": t["_id"]},
            {"$set": {"status": "FAILED", "error_message": "Stale RUNNING task recovered after restart"}}
        )
    if stale_running:
        logger.info("Recovered %d stale RUNNING tasks as FAILED.", len(stale_running))


def _execute_task_by_id(task_id: ObjectId) -> None:
    """Fetch and execute a task by ID."""
    task = agent_tasks.find_one({"_id": task_id})
    if not task or task.get("status") != "PENDING":
        return

    logger.info("Executing task %s (%s)", task_id, task["task_type"])
    if task["task_type"] == "COMMITMENT_CHECK":
        _execute_commitment_check_task(task)
    else:
        agent_tasks.update_one({"_id": task_id}, {"$set": {"status": "COMPLETED", "error_message": "Unknown type"}})



def _fire_nudge(log_id: ObjectId) -> None:
    """Fire the engagement nudge for a specific log."""
    from chanakya.db.mongo import interaction_logs, checkpoints as cp_col

    log = interaction_logs.find_one({"_id": log_id})
    if not log or log.get("user_response"):
        return  # Already handled or missing
    verdict = log.get("ai_evaluation", {}).get("verdict")
    if verdict and verdict != "ABANDONED":
        return  # User was judged (including SKIPPED) — stop nudging

    uid = log["user_id"]
    user_doc = users.find_one({"_id": uid})
    if not user_doc or not user_doc.get("active"):
        return

    cp = None
    if log.get("checkpoint_id"):
        cp = cp_col.find_one({"_id": log["checkpoint_id"]})

    nudge_window = cp.get("nudge_window_minutes", 20) if cp else 20
    is_persistent = cp.get("persistent_nudge", False) if cp else False
    checkpoint_name = (cp.get("display_name") or cp.get("activity", "a checkpoint")).replace("_", " ").title() if cp else "a checkpoint"
    original_msg = log.get("message_sent", "")

    # Hard limit: stop nudging after nudge_window_minutes from the original checkpoint.
    log_timestamp = log.get("timestamp")
    if log_timestamp and (datetime.utcnow() - log_timestamp) > timedelta(minutes=nudge_window):
        logger.info("Nudge expired (>%dmin) for log %s — marking abandoned.", nudge_window, log_id)
        interaction_logs.update_one(
            {"_id": log_id},
            {"$set": {"ai_evaluation.verdict": "ABANDONED", "ai_evaluation.reasoning": f"No response within {nudge_window} minutes"}}
        )
        return

    nudge_count = log.get("nudge_count", 0) + 1

    # Hard cap: max 3 nudges total
    if nudge_count > 3:
        logger.info("Nudge cap reached (3) for log %s — stopping.", log_id)
        interaction_logs.update_one(
            {"_id": log_id},
            {"$set": {"ai_evaluation.verdict": "ABANDONED", "ai_evaluation.reasoning": "Nudge cap reached, no response"}}
        )
        return

    interaction_logs.update_one({"_id": log_id}, {"$inc": {"nudge_count": 1}})

    if nudge_count == 2:
        snippet = original_msg[:120].strip()
        async def _send_nudge2():
            from telegram import Bot
            from chanakya.config import TELEGRAM_BOT_TOKEN
            msg = (
                f"⚔️ <b>Final warning</b>\n"
                f"You haven't replied to <b>{checkpoint_name}</b>.\n"
                f"{snippet}\n\n"
                f"Reply now or this is marked abandoned."
            )
            await Bot(token=TELEGRAM_BOT_TOKEN).send_message(
                chat_id=user_doc["telegram_id"], text=msg, parse_mode="HTML"
            )
        _run_async(_send_nudge2())

        # Final nudge after 10 min, then done
        schedule_engagement_nudge(log_id, 10)

    elif nudge_count == 3:
        # Final nudge: text only, no more calls
        async def _send_final():
            from telegram import Bot
            from chanakya.config import TELEGRAM_BOT_TOKEN
            msg = f"Last reminder for <b>{checkpoint_name}</b>. No further follow-ups. Marking as abandoned if no reply."
            await Bot(token=TELEGRAM_BOT_TOKEN).send_message(
                chat_id=user_doc["telegram_id"], text=msg, parse_mode="HTML"
            )
        _run_async(_send_final())

    else:
        # First nudge: Telegram text warning
        async def _send_first():
            from telegram import Bot
            from chanakya.config import TELEGRAM_BOT_TOKEN
            msg = "⚔️ <b>Dharma Monitor</b>\nDelay is the silent killer of empires. Respond now."
            await Bot(token=TELEGRAM_BOT_TOKEN).send_message(
                chat_id=user_doc["telegram_id"], text=msg, parse_mode="HTML"
            )
        _run_async(_send_first())

        # Schedule escalation in 5 mins
        schedule_engagement_nudge(log_id, 5)


def _run_async(coro):
    """Run a coroutine from a sync background thread (thread-safe)."""
    from chanakya.async_utils import run_async
    run_async(coro)



def _execute_commitment_check_task(task: dict) -> None:
    """Follow up on a user commitment."""
    try:
        user_id = task["user_id"]
        payload = task.get("payload", {})
        task_name = payload.get("task_name", "your task")

        user_doc = users.find_one({"_id": user_id})
        if not user_doc or not user_doc.get("telegram_id"):
            return

        message = (
            f"⚔️ <b>Commitment Follow-up</b>\n"
            f"Your time for <b>{task_name}</b> has ended. Did you fulfill your dharma?\n\n"
            "Respond now with your verdict."
        )

        from telegram import Bot
        from chanakya.config import TELEGRAM_BOT_TOKEN
        
        async def _send():
            bot = Bot(token=TELEGRAM_BOT_TOKEN)
            await bot.send_message(chat_id=user_doc["telegram_id"], text=message, parse_mode="HTML")
        _run_async(_send())

        agent_tasks.update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "COMPLETED", "last_attempted_at": datetime.utcnow()}}
        )
        logger.info(f"COMMITMENT_CHECK task {task['_id']} completed for user {user_id}")

    except Exception as exc:
        logger.error(f"Failed to execute COMMITMENT_CHECK task {task['_id']}: {exc}")
        _mark_task_failed(task, str(exc))


def _mark_task_failed(task: dict, error_message: str):
    retries = task.get("retries_attempted", 0) + 1
    max_retries = task.get("max_retries", 3)
    
    status = "FAILED"
    if retries >= max_retries:
        status = "COMPLETED"
        try:
            from telegram import Bot
            from chanakya.config import TELEGRAM_BOT_TOKEN
            user_doc = users.find_one({"_id": task["user_id"]})
            if user_doc and user_doc.get("telegram_id"):
                async def _send():
                    bot = Bot(token=TELEGRAM_BOT_TOKEN)
                    await bot.send_message(
                        chat_id=user_doc["telegram_id"],
                        text=f"⚠️ <b>Task Failed Permanently</b>\nTask: {task.get('task_type')}\nError: {error_message[:200]}",
                        parse_mode="HTML"
                    )
                _run_async(_send())
        except: pass
            
    agent_tasks.update_one(
        {"_id": task["_id"]},
        {
            "$set": {
                "status": status,
                "error_message": error_message,
                "retries_attempted": retries,
                "last_attempted_at": datetime.utcnow()
            }
        }
    )

