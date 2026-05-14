"""
test_endpoints.py — Internal API for system verification and mocking.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from bson import ObjectId
from datetime import datetime

from chanakya.config import WEBHOOK_URL
from chanakya.db.mongo import users, interaction_logs, agent_tasks
from chanakya.scheduler.task_runner import schedule_agent_task, recover_pending_tasks
from chanakya.bot.telegram_bot import generic_process_message

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/test", tags=["Testing"])

class TelegramMockRequest(BaseModel):
    message: str

class CallRequest(BaseModel):
    text: str
    user_id: Optional[str] = None

@router.post("/telegram")
async def mock_telegram_message(req: TelegramMockRequest):
    """Mock an incoming Telegram message from the primary user."""
    user = users.find_one({"active": True})
    if not user:
        raise HTTPException(status_code=404, detail="No active user found.")
    
    response = await generic_process_message(user, req.message, channel="MOCK_TELEGRAM")
    return {"status": "success", "chanakya_reply": response}

@router.post("/call")
async def trigger_test_call(req: CallRequest):
    """Directly inject and schedule a CALL_USER task."""
    uid_str = req.user_id
    if uid_str:
        user = users.find_one({"_id": ObjectId(uid_str)})
    else:
        user = users.find_one({"active": True})
        
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
        
    task_id = agent_tasks.insert_one({
        "user_id": user["_id"],
        "task_type": "CALL_USER",
        "status": "PENDING",
        "payload": {"opening_text": req.text},
        "created_at": datetime.utcnow()
    }).inserted_id
    
    # Manually trigger the scheduler to pick it up immediately
    schedule_agent_task(task_id)
    
    return {"status": "success", "task_id": str(task_id), "message": "Call task scheduled."}

@router.post("/recover")
async def force_recover_tasks():
    """Manually trigger the recovery of PENDING tasks."""
    recover_pending_tasks()
    return {"status": "success", "message": "Recovery sequence initiated."}

@router.post("/reset_webhook")
async def reset_telegram_webhook():
    """Force re-register the Telegram webhook using the current WEBHOOK_URL."""
    from chanakya.main import _telegram_app
    if not _telegram_app:
        raise HTTPException(status_code=500, detail="Telegram application not initialized.")
    
    if not WEBHOOK_URL:
         raise HTTPException(status_code=400, detail="WEBHOOK_URL not configured.")
         
    webhook_url = WEBHOOK_URL.rstrip("/") + "/"
    res = await _telegram_app.bot.set_webhook(url=webhook_url, drop_pending_updates=False)
    return {"status": "success", "url": webhook_url, "telegram_response": res}

@router.get("/status")
async def get_test_status():
    """Extended status for debugging."""
    from chanakya.scheduler.checkpoint_runner import scheduler as cp_scheduler
    from chanakya.scheduler.task_runner import task_scheduler
    
    cp_jobs = []
    for job in cp_scheduler.get_jobs():
        cp_jobs.append({
            "id": job.id,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            "trigger": str(job.trigger)
        })
        
    task_jobs = []
    for job in task_scheduler.get_jobs():
        task_jobs.append({
            "id": job.id,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            "trigger": str(job.trigger)
        })
        
    recent_tasks = list(agent_tasks.find().sort("created_at", -1).limit(5))
    for t in recent_tasks:
        t["_id"] = str(t["_id"])
        t["user_id"] = str(t["user_id"])
        if "created_at" in t: t["created_at"] = t["created_at"].isoformat()
        if "last_attempted_at" in t: t["last_attempted_at"] = t["last_attempted_at"].isoformat()

    return {
        "checkpoint_scheduler": {
            "running": cp_scheduler.running,
            "jobs": cp_jobs
        },
        "task_scheduler": {
            "running": task_scheduler.running,
            "jobs": task_jobs
        },
        "recent_tasks": recent_tasks
    }
