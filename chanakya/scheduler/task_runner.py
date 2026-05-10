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
    """Start the task runner. Called once at app startup."""
    task_scheduler.add_job(run_tasks, "interval", seconds=15, id="agent_task_runner")
    task_scheduler.add_job(monitor_engagement, "interval", minutes=5, id="engagement_monitor")
    task_scheduler.start()
    logger.info("Task runner and engagement monitor started.")

def stop_task_runner() -> None:
    """Stop the task runner. Called at app shutdown."""
    task_scheduler.shutdown(wait=False)
    logger.info("Task runner stopped.")

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
        # Mark as failed to allow retry
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
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        
        asyncio.run(bot.send_message(
            chat_id=user_doc["telegram_id"],
            text=message,
            parse_mode="HTML"
        ))

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
        status = "COMPLETED" # Giving up permanently
        # Notify user it failed permanently
        try:
            from telegram import Bot
            from chanakya.config import TELEGRAM_BOT_TOKEN
            user_doc = users.find_one({"_id": task["user_id"]})
            if user_doc and user_doc.get("telegram_id"):
                bot = Bot(token=TELEGRAM_BOT_TOKEN)
                task_name = task.get("task_type")
                asyncio.run(bot.send_message(
                    chat_id=user_doc["telegram_id"],
                    text=f"⚠️ <b>Task Failed Permanently</b>\nTask: {task_name}\nError: {error_message[:200]}",
                    parse_mode="HTML"
                ))
        except Exception as e:
            logger.error(f"Could not send failure notification for task {task['_id']}: {e}")
            
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

def monitor_engagement() -> None:
    """Detect avoidance behavior (stale checkpoints) and trigger nudges."""
    now = datetime.utcnow()
    # Find logs where user hasn't responded for > 15 mins
    # Only nudge once (nudge_sent != True)
    stale_threshold = now - timedelta(minutes=15)
    stale_logs = list(interaction_logs.find({
        "user_response": None,
        "timestamp": {"$lt": stale_threshold},
        "trigger_type": "CHECKPOINT",
        "$or": [
            {"nudge_sent": {"$ne": True}},
            {"call_escalated": {"$ne": True}}
        ]
    }))

    for log in stale_logs:
        uid = log["user_id"]
        user_doc = users.find_one({"_id": uid})
        if not user_doc or not user_doc.get("telegram_id"):
            continue

        # If already nudged once, escalate to CALL
        if log.get("nudge_sent"):
            # Check if 30 mins have passed since original checkpoint
            escalation_threshold = now - timedelta(minutes=30)
            if log["timestamp"] < escalation_threshold and not log.get("call_escalated"):
                # Schedule a CALL_USER task
                opening_text = (
                    "Dharma requires presence. You have ignored my messages for 30 minutes. "
                    "I am calling to ensure your focus has not drifted. Respond now."
                )
                agent_tasks.insert_one({
                    "user_id": uid,
                    "task_type": "CALL_USER",
                    "status": "PENDING",
                    "payload": {"opening_text": opening_text},
                    "created_at": now,
                    "last_attempted_at": None,
                    "retries_attempted": 0
                })
                interaction_logs.update_one({"_id": log["_id"]}, {"$set": {"call_escalated": True}})
                logger.info("Engagement escalated to CALL for user %s for log %s", uid, log["_id"])
            continue

        # Send Telegram nudge (First warning)
        try:
            from telegram import Bot
            from chanakya.config import TELEGRAM_BOT_TOKEN
            bot = Bot(token=TELEGRAM_BOT_TOKEN)
            
            msg = (
                "⚔️ <b>Dharma Monitor</b>\n"
                "I am waiting for your response to the last checkpoint. "
                "Delay is the silent killer of empires. Respond now."
            )
            # Use a helper or run in new loop to avoid blocking
            asyncio.run(bot.send_message(
                chat_id=user_doc["telegram_id"],
                text=msg,
                parse_mode="HTML"
            ))
            
            interaction_logs.update_one({"_id": log["_id"]}, {"$set": {"nudge_sent": True}})
            logger.info("Engagement nudge sent to user %s for log %s", uid, log["_id"])
        except Exception as e:
            logger.error("Failed to send engagement nudge: %s", e)

def run_tasks() -> None:
    now = datetime.utcnow()
    
    # Recover crashed RUNNING tasks (older than 10 mins)
    cutoff_running = now - timedelta(minutes=10)
    crashed_tasks = agent_tasks.find({"status": "RUNNING", "last_attempted_at": {"$lt": cutoff_running}})
    for ct in crashed_tasks:
        logger.warning(f"Task {ct['_id']} seems to have crashed (stuck in RUNNING). Marking as FAILED for retry.")
        _mark_task_failed(ct, "Task timed out or app crashed while running.")

    # Process PENDING tasks and FAILED tasks eligible for retry (wait 2 mins before retry)
    cutoff_failed = now - timedelta(minutes=2)
    
    tasks_to_run = list(agent_tasks.find({
        "$or": [
            {"status": "PENDING"},
            {"status": "FAILED", "retries_attempted": {"$lt": 3}, "last_attempted_at": {"$lt": cutoff_failed}},
        ]
    }).sort("created_at", 1))

    for task in tasks_to_run:
        logger.info(f"Task Manager picked up task {task['_id']} ({task['task_type']})")
        if task["task_type"] == "PROXY_CALL":
            _execute_proxy_call_task(task)
        elif task["task_type"] == "CALL_USER":
            _execute_call_user_task(task)
        elif task["task_type"] == "COMMITMENT_CHECK":
            _execute_commitment_check_task(task)
        else:
            logger.warning(f"Unknown task type: {task['task_type']}")
            agent_tasks.update_one({"_id": task["_id"]}, {"$set": {"status": "COMPLETED", "error_message": "Unknown task type."}})

