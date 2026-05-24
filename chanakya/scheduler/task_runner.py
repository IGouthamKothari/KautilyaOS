"""
task_runner.py — APScheduler-based Task Manager.

Processes background tasks (e.g., PROXY_CALL) assigned by the Chanakya Agent.
Ensures tasks are completed, handles retries on failures, and recovers from crashes.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from bson import ObjectId

from chanakya.db.mongo import agent_tasks, users, get_contact_by_name, interaction_logs, proxy_call_logs
from chanakya.config import WEBHOOK_URL

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
        run_at = datetime.utcnow()
    
    task_scheduler.add_job(
        _execute_task_by_id,
        "date",
        run_date=run_at,
        args=[task_id],
        id=f"task_{task_id}",
        replace_existing=True
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
    """Find PENDING tasks in DB and schedule them (useful after restart)."""
    pending = list(agent_tasks.find({"status": "PENDING"}))
    for t in pending:
        schedule_agent_task(t["_id"])


def _execute_task_by_id(task_id: ObjectId) -> None:
    """Fetch and execute a task by ID."""
    task = agent_tasks.find_one({"_id": task_id})
    if not task or task.get("status") != "PENDING":
        return
        
    logger.info("Executing task %s (%s)", task_id, task["task_type"])
    if task["task_type"] == "PROXY_CALL":
        _execute_proxy_call_task(task)
    elif task["task_type"] == "CALL_USER":
        _execute_call_user_task(task)
    elif task["task_type"] == "COMMITMENT_CHECK":
        _execute_commitment_check_task(task)
    else:
        agent_tasks.update_one({"_id": task_id}, {"$set": {"status": "COMPLETED", "error_message": "Unknown type"}})


def _fire_nudge(log_id: ObjectId) -> None:
    """Fire the engagement nudge for a specific log."""
    from chanakya.db.mongo import interaction_logs, checkpoints as cp_col

    log = interaction_logs.find_one({"_id": log_id})
    if not log or log.get("user_response") or log.get("ai_evaluation", {}).get("verdict"):
        return  # Already handled or missing

    uid = log["user_id"]
    user_doc = users.find_one({"_id": uid})
    if not user_doc or not user_doc.get("active"):
        return

    # Hard limit: stop nudging after 2 hours from the original checkpoint
    log_timestamp = log.get("timestamp")
    if log_timestamp and (datetime.utcnow() - log_timestamp) > timedelta(hours=2):
        logger.info("Nudge expired (>2h) for log %s — stopping.", log_id)
        return

    cp = None
    if log.get("checkpoint_id"):
        cp = cp_col.find_one({"_id": log["checkpoint_id"]})

    is_persistent = cp.get("persistent_nudge", False) if cp else False
    nudge_count = log.get("nudge_count", 0) + 1

    # Hard cap: max 5 nudges total regardless of persistence
    if nudge_count > 5:
        logger.info("Nudge cap reached (5) for log %s — stopping.", log_id)
        return

    interaction_logs.update_one({"_id": log_id}, {"$inc": {"nudge_count": 1}})

    if nudge_count >= 2:
        # 2nd+ nudge: escalate to CALL (once, not repeatedly)
        text = "Dharma Violation. You have ignored my guidance. I am calling to correct your path."
        if is_persistent and cp:
            text = f"Persistent Nudge: {cp.get('display_name')}. Respond now."

        task_id = agent_tasks.insert_one({
            "user_id": uid, "task_type": "CALL_USER", "status": "PENDING",
            "payload": {"opening_text": text, "log_id": str(log_id)},
            "created_at": datetime.utcnow()
        }).inserted_id
        schedule_agent_task(task_id)

        # Only schedule one more follow-up (not infinite loop)
        if nudge_count < 5:
            interval = cp.get("persistent_nudge_interval_minutes", 30) if is_persistent else 30
            schedule_engagement_nudge(log_id, interval)
    else:
        # First nudge: Telegram text warning
        async def _send():
            from telegram import Bot
            from chanakya.config import TELEGRAM_BOT_TOKEN
            msg = "⚔️ <b>Dharma Monitor</b>\nDelay is the silent killer of empires. Respond now."
            await Bot(token=TELEGRAM_BOT_TOKEN).send_message(chat_id=user_doc["telegram_id"], text=msg, parse_mode="HTML")
        _run_async(_send())

        # Schedule escalation (call) in 15 mins if still no response
        schedule_engagement_nudge(log_id, 15)


def _run_async(coro):
    """Run a coroutine from a sync background thread (thread-safe).

    APScheduler runs jobs in daemon threads that have no event loop.
    We need to schedule the coroutine on the main thread's running loop.
    """
    import asyncio

    try:
        # Are we already inside a running loop on this thread?
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is not None:
        # Same thread as the event loop — fire-and-forget
        asyncio.ensure_future(coro)
        return

    # Background thread: find the main event loop via the global policy
    # and schedule the coroutine on it using the thread-safe API.
    try:
        import threading
        main_loop = None
        for thread in threading.enumerate():
            loop_attr = getattr(thread, "_asyncio_loop", None)
            if loop_attr is not None and loop_attr.is_running():
                main_loop = loop_attr
                break

        if main_loop is not None:
            asyncio.run_coroutine_threadsafe(coro, main_loop)
        else:
            # No running loop found — spin up a fresh one (last resort)
            asyncio.run(coro)
    except Exception as exc:
        logger.error("_run_async failed to schedule coroutine: %s", exc)
        try:
            asyncio.run(coro)
        except Exception:
            pass


def _execute_proxy_call_task(task: dict) -> None:
    """Execute a PROXY_CALL task."""
    try:
        user_id = task["user_id"]
        payload = task.get("payload", {})
        contact_name = payload.get("contact_name", "")
        topic = payload.get("topic", "")

        user_doc = users.find_one({"_id": user_id})
        if not user_doc:
            raise Exception("User not found.")

        from chanakya.db.mongo import get_contact_by_name
        contact = get_contact_by_name(user_id, contact_name)
        if not contact:
            raise Exception(f"No contact named '{contact_name}' found.")

        phone = contact.get("phone", "")
        if not phone:
            raise Exception(f"Contact '{contact_name}' has no phone number saved.")

        if not WEBHOOK_URL:
            raise Exception("WEBHOOK_URL not configured.")

        owner_name = user_doc.get("name", "Goutham")
        owner_telegram_id = user_doc.get("telegram_id", "")

        opening_text = (
            f"Hello, this is Chanakya, an AI assistant calling on behalf of {owner_name}. "
            f"I'm calling to {topic}. Is this a good time to talk?"
        )

        now = datetime.utcnow()
        log_doc = {
            "user_id": user_id,
            "timestamp": now,
            "trigger_type": "MANUAL",
            "channel": "PROXY_CALL",
            "message_sent": opening_text,
            "user_response": None,
            "proxy": True,
            "proxy_contact_name": contact["name"],
            "proxy_topic": topic,
            "ai_evaluation": {"verdict": None, "confidence": None, "reasoning": None},
            "created_at": now,
        }

        log_result = interaction_logs.insert_one(log_doc)
        session_id = str(log_result.inserted_id)

        from chanakya.integrations.twilio_webhooks import create_voice_session, synthesize_call_opening
        
        # Save task ID in the voice session so webhooks can mark it completed
        create_voice_session(
            session_id=session_id,
            user_id=str(user_id),
            context=opening_text,
            proxy=True,
            proxy_contact_name=contact["name"],
            proxy_topic=topic,
            owner_telegram_id=owner_telegram_id,
            owner_name=owner_name,
            audio_bytes=synthesize_call_opening(opening_text),
            task_id=str(task["_id"]),
        )

        twiml_url = f"{WEBHOOK_URL.rstrip('/')}/twilio/voice/{session_id}"

        from chanakya.integrations.twilio_client import TwilioClient
        twilio = TwilioClient()
        call_sid = twilio.make_call(to=phone, twiml_url=twiml_url)
        
        interaction_logs.update_one(
            {"_id": log_result.inserted_id},
            {"$set": {"twilio_call_sid": call_sid}},
        )

        # Task is now running, waiting for webhook callback to complete it
        agent_tasks.update_one(
            {"_id": task["_id"]},
            {
                "$set": {
                    "status": "RUNNING",
                    "result": {"session_id": session_id, "call_sid": call_sid},
                    "last_attempted_at": datetime.utcnow()
                }
            }
        )
        logger.info(f"PROXY_CALL task {task['_id']} initiated. call_sid={call_sid}")

    except Exception as exc:
        logger.error(f"Failed to execute PROXY_CALL task {task['_id']}: {exc}")
        _mark_task_failed(task, str(exc))


def _execute_call_user_task(task: dict) -> None:
    """Execute a CALL_USER task."""
    try:
        user_id = task["user_id"]
        payload = task.get("payload", {})
        opening_text = payload.get("opening_text", "Chanakya here. What do you need to discuss?")

        user_doc = users.find_one({"_id": user_id})
        if not user_doc:
            raise Exception("User not found.")

        phone = user_doc.get("phone", "")
        if not phone:
            raise Exception("No phone number on file.")

        if not WEBHOOK_URL:
            raise Exception("WEBHOOK_URL not configured.")

        now = datetime.utcnow()
        log_doc = {
            "user_id": user_id,
            "timestamp": now,
            "trigger_type": "MANUAL",
            "channel": "CALL",
            "message_sent": opening_text,
            "user_response": None,
            "ai_evaluation": {"verdict": None, "confidence": None, "reasoning": None},
            "created_at": now,
        }

        log_result = interaction_logs.insert_one(log_doc)
        session_id = str(log_result.inserted_id)

        from chanakya.integrations.twilio_webhooks import create_voice_session, synthesize_call_opening
        
        # Save task ID in the voice session so webhooks can mark it completed
        create_voice_session(
            session_id=session_id,
            user_id=str(user_id),
            context=opening_text,
            proxy=False,
            audio_bytes=synthesize_call_opening(opening_text),
            task_id=str(task["_id"]),
        )

        twiml_url = f"{WEBHOOK_URL.rstrip('/')}/twilio/voice/{session_id}"

        from chanakya.integrations.twilio_client import TwilioClient
        twilio = TwilioClient()
        call_sid = twilio.make_call(to=phone, twiml_url=twiml_url)
        
        interaction_logs.update_one(
            {"_id": log_result.inserted_id},
            {"$set": {"twilio_call_sid": call_sid}},
        )

        # Task is now running, waiting for webhook callback to complete it
        agent_tasks.update_one(
            {"_id": task["_id"]},
            {
                "$set": {
                    "status": "RUNNING",
                    "result": {"session_id": session_id, "call_sid": call_sid},
                    "last_attempted_at": datetime.utcnow()
                }
            }
        )
        logger.info(f"CALL_USER task {task['_id']} initiated. call_sid={call_sid}")

    except Exception as exc:
        logger.error(f"Failed to execute CALL_USER task {task['_id']}: {exc}")
        _mark_task_failed(task, str(exc))


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

