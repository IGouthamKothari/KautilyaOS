"""
dashboard_api.py — REST API for the Dharma Dashboard web UI.

Covers every operation available in the system:
  Schedule   — CRUD on checkpoints, reload schedule
  Contacts   — list / add / delete
  Rules      — personal instructions CRUD
  Actions    — call self, call contact, send message
  Logs       — recent interaction_log entries
  Profile    — read/update current_mode, streaks
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _active_user():
    from chanakya.db.mongo import users
    user = users.find_one({"active": True})
    if not user:
        raise HTTPException(status_code=404, detail="No active user found")
    return user


def _serialize(doc: dict) -> dict:
    """Convert ObjectId and datetime fields to JSON-serializable strings."""
    out = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            out[k] = str(v)
        elif isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, dict):
            out[k] = _serialize(v)
        elif isinstance(v, list):
            out[k] = [_serialize(i) if isinstance(i, dict) else (str(i) if isinstance(i, ObjectId) else i) for i in v]
        else:
            out[k] = v
    return out


# ===========================================================================
# SCHEDULE
# ===========================================================================

_ALL_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WEEKDAYS  = ["monday", "tuesday", "wednesday", "thursday", "friday"]
_WEEKENDS  = ["saturday", "sunday"]


class CheckpointUpdate(BaseModel):
    time: Optional[str] = None
    display_name: Optional[str] = None
    action_type: Optional[str] = None
    priority: Optional[str] = None
    active: Optional[bool] = None
    nudge_window_minutes: Optional[int] = None
    prompt_template: Optional[str] = None
    description: Optional[str] = None
    days: Optional[list] = None  # list of day strings e.g. ["monday","wednesday"]


class CheckpointCreate(BaseModel):
    time: str
    display_name: str
    activity: Optional[str] = None
    action_type: str = "TELEGRAM_TEXT"
    priority: str = "MEDIUM"
    days: list  # required — at least one day
    nudge_window_minutes: int = 45
    description: str = ""
    prompt_template: str = ""


@router.get("/schedule")
async def get_schedule():
    from chanakya.db.mongo import checkpoints
    user = _active_user()
    docs = list(checkpoints.find({"user_id": user["_id"]}).sort("time", 1))
    return [_serialize(d) for d in docs]


@router.post("/schedule")
async def create_checkpoint(body: CheckpointCreate):
    """Create a new checkpoint directly in MongoDB."""
    from chanakya.db.mongo import checkpoints
    from chanakya.scheduler.checkpoint_runner import refresh_all_schedules
    user = _active_user()

    valid_days = [d for d in body.days if d in _ALL_DAYS]
    if not valid_days:
        raise HTTPException(status_code=400, detail="Provide at least one valid day")

    activity = body.activity or body.display_name.lower().replace(" ", "_")
    now = datetime.utcnow()
    doc = {
        "user_id": user["_id"],
        "time": body.time,
        "activity": activity,
        "display_name": body.display_name,
        "description": body.description,
        "action_type": body.action_type,
        "priority": body.priority,
        "days": valid_days,
        "prompt_template": body.prompt_template or activity,
        "nudge_window_minutes": body.nudge_window_minutes,
        "persistent_nudge": False,
        "persistent_nudge_interval_minutes": 5,
        "active": True,
        "failure_punishment": {"type": "WARN"},
        "last_triggered": None,
        "source": "dashboard",
        "created_at": now,
        "updated_at": now,
    }
    result = checkpoints.insert_one(doc)
    refresh_all_schedules()
    doc["_id"] = result.inserted_id
    return _serialize(doc)


@router.patch("/schedule/{checkpoint_id}")
async def update_checkpoint(checkpoint_id: str, body: CheckpointUpdate):
    from chanakya.db.mongo import checkpoints
    from chanakya.scheduler.checkpoint_runner import refresh_all_schedules
    user = _active_user()
    try:
        cid = ObjectId(checkpoint_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid checkpoint_id")

    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Validate days if provided
    if "days" in updates:
        updates["days"] = [d for d in updates["days"] if d in _ALL_DAYS]
        if not updates["days"]:
            raise HTTPException(status_code=400, detail="Provide at least one valid day")

    updates["updated_at"] = datetime.utcnow()
    result = checkpoints.update_one(
        {"_id": cid, "user_id": user["_id"]}, {"$set": updates}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    refresh_all_schedules()
    doc = checkpoints.find_one({"_id": cid})
    return _serialize(doc)


@router.delete("/schedule/{checkpoint_id}")
async def delete_checkpoint(checkpoint_id: str):
    from chanakya.db.mongo import checkpoints
    from chanakya.scheduler.checkpoint_runner import refresh_all_schedules
    user = _active_user()
    try:
        cid = ObjectId(checkpoint_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid checkpoint_id")

    result = checkpoints.delete_one({"_id": cid, "user_id": user["_id"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    refresh_all_schedules()
    return {"deleted": checkpoint_id}


# ===========================================================================
# CONTACTS
# ===========================================================================

class ContactInput(BaseModel):
    name: str
    phone: str
    relationship: str = ""


@router.get("/contacts")
async def list_contacts():
    """List all contacts for the active user."""
    from chanakya.db.mongo import get_contacts
    user = _active_user()
    docs = get_contacts(user["_id"])
    return [_serialize(d) for d in docs]


@router.post("/contacts")
async def add_contact(body: ContactInput):
    """Add or update a contact."""
    from chanakya.db.mongo import add_contact
    user = _active_user()
    doc = add_contact(user["_id"], body.name, body.phone, body.relationship)
    return _serialize(doc)


@router.delete("/contacts/{name}")
async def delete_contact(name: str):
    """Delete a contact by name."""
    from chanakya.db.mongo import remove_contact
    user = _active_user()
    ok = remove_contact(user["_id"], name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"No contact named '{name}'")
    return {"deleted": name}


# ===========================================================================
# PERSONAL RULES (personal_instructions)
# ===========================================================================

class RuleInput(BaseModel):
    text: str


@router.get("/rules")
async def list_rules():
    """List all personal rules / instructions."""
    from chanakya.db.mongo import get_personal_instructions
    user = _active_user()
    items = get_personal_instructions(user["_id"])
    return [{"index": i, "text": t} for i, t in enumerate(items)]


@router.post("/rules")
async def add_rule(body: RuleInput):
    """Append a personal rule."""
    from chanakya.db.mongo import add_personal_instruction
    user = _active_user()
    count = add_personal_instruction(user["_id"], body.text)
    return {"index": count - 1, "text": body.text, "total": count}


@router.delete("/rules/{index}")
async def delete_rule(index: int):
    """Delete a rule by 0-based index."""
    from chanakya.db.mongo import remove_personal_instruction
    user = _active_user()
    ok = remove_personal_instruction(user["_id"], index)
    if not ok:
        raise HTTPException(status_code=404, detail=f"No rule at index {index}")
    return {"deleted_index": index}


# ===========================================================================
# ACTIONS — call self, call contact
# ===========================================================================

class CallSelfInput(BaseModel):
    opening_text: str = "Chanakya here. Dharma check."


class CallContactInput(BaseModel):
    contact_name: str
    topic: str


@router.post("/actions/call-self")
async def call_self(body: CallSelfInput):
    """Trigger a CALL_USER task — calls the active user immediately."""
    from chanakya.db.mongo import agent_tasks
    from chanakya.scheduler.task_runner import schedule_agent_task
    user = _active_user()
    task_id = agent_tasks.insert_one({
        "user_id": user["_id"],
        "task_type": "CALL_USER",
        "status": "PENDING",
        "payload": {"opening_text": body.opening_text},
        "created_at": datetime.utcnow(),
    }).inserted_id
    schedule_agent_task(task_id)
    return {"status": "scheduled", "task_id": str(task_id)}


@router.post("/actions/call-contact")
async def call_contact(body: CallContactInput):
    """Trigger a PROXY_CALL task — calls a saved contact on behalf of the user."""
    from chanakya.db.mongo import agent_tasks, get_contact_by_name
    from chanakya.scheduler.task_runner import schedule_agent_task
    user = _active_user()
    contact = get_contact_by_name(user["_id"], body.contact_name)
    if not contact:
        raise HTTPException(status_code=404, detail=f"No contact named '{body.contact_name}'")
    task_id = agent_tasks.insert_one({
        "user_id": user["_id"],
        "task_type": "PROXY_CALL",
        "status": "PENDING",
        "payload": {"contact_name": body.contact_name, "topic": body.topic},
        "created_at": datetime.utcnow(),
    }).inserted_id
    schedule_agent_task(task_id)
    return {"status": "scheduled", "task_id": str(task_id), "contact": body.contact_name}


# ===========================================================================
# LOGS — recent interaction_logs
# ===========================================================================

@router.get("/logs")
async def get_logs(limit: int = 30, trigger_type: str = ""):
    """Return recent interaction logs, newest first."""
    from chanakya.db.mongo import interaction_logs
    user = _active_user()
    query: dict = {"user_id": user["_id"]}
    if trigger_type:
        query["trigger_type"] = trigger_type
    docs = list(interaction_logs.find(query).sort("timestamp", -1).limit(limit))
    return [_serialize(d) for d in docs]


# ===========================================================================
# PROFILE — read/update user doc
# ===========================================================================

class ProfileUpdate(BaseModel):
    current_mode: Optional[str] = None
    eod_time: Optional[str] = None
    morning_todo_time: Optional[str] = None
    timezone: Optional[str] = None


@router.get("/profile")
async def get_profile():
    """Return the active user profile."""
    user = _active_user()
    return _serialize(user)


@router.patch("/profile")
async def update_profile(body: ProfileUpdate):
    """Update mutable profile fields."""
    from chanakya.db.mongo import users
    user = _active_user()
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    updates["updated_at"] = datetime.utcnow()
    users.update_one({"_id": user["_id"]}, {"$set": updates})
    updated = users.find_one({"_id": user["_id"]})
    return _serialize(updated)


@router.post("/profile/war-mode")
async def toggle_war_mode(activate: bool = True):
    """Activate or deactivate WAR MODE."""
    from chanakya.db.mongo import users
    user = _active_user()
    if activate:
        expires = datetime.utcnow() + timedelta(hours=24)
        users.update_one(
            {"_id": user["_id"]},
            {"$set": {"current_mode": "WAR_MODE", "war_mode_expires": expires, "updated_at": datetime.utcnow()}}
        )
        return {"current_mode": "WAR_MODE", "expires": expires.isoformat()}
    else:
        users.update_one(
            {"_id": user["_id"]},
            {"$set": {"current_mode": "NORMAL", "war_mode_expires": None, "updated_at": datetime.utcnow()}}
        )
        return {"current_mode": "NORMAL"}
