"""
schedule_tools.py — LangChain tool definitions.

Exposes tool-decorated functions to the AgentExecutor:
  - escalate_punishment
  - modify_wakeup_time
  - activate_war_mode
  - add_daily_checkpoint
  - send_emergency_alert
  - update_morning_todo_time
  - fetch_schedule
  - update_schedule_activity
  - save_contact
  - list_contacts
  - delete_contact
  - place_proxy_call
  - set_user_phone

Each tool validates inputs, writes an ai_tool_calls audit document,
and returns a plain confirmation string or descriptive error string.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta

from bson import ObjectId
from langchain_core.tools import tool

from chanakya.db.mongo import checkpoints, users
from chanakya.tools.accountability_tools import ALL_ACCOUNTABILITY_TOOLS
from chanakya.tools.ritual_tools import ALL_RITUAL_TOOLS
from chanakya.tools.council_tools import ALL_COUNCIL_TOOLS
from chanakya.tools.goal_tools import ALL_GOAL_TOOLS
from chanakya.tools.google_tools import ALL_GOOGLE_TOOLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Escalation order constant
# ---------------------------------------------------------------------------

ESCALATION_ORDER = [
    "WARN",
    "TELEGRAM_ALERT",
    "CALL_EMERGENCY_CONTACT",
    "SMS_EMERGENCY_CONTACT",
]


# ---------------------------------------------------------------------------
# Internal audit helper
# ---------------------------------------------------------------------------


def _write_audit(user_id, tool_name: str, tool_input: dict, tool_output: str) -> None:
    """Fire-and-forget audit log write to ai_tool_calls."""
    try:
        from chanakya.db.mongo import ai_tool_calls

        ai_tool_calls.insert_one(
            {
                "user_id": user_id,
                "timestamp": datetime.utcnow(),
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": tool_output,
                "created_at": datetime.utcnow(),
            }
        )
    except Exception as exc:
        logger.warning("Failed to write audit log for %s: %s", tool_name, exc)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def escalate_punishment(user_id: str, checkpoint_id: str, reason: str = "") -> str:
    """Increase punishment severity for a repeatedly failed checkpoint.

    Escalation order: WARN → TELEGRAM_ALERT → CALL_EMERGENCY_CONTACT → SMS_EMERGENCY_CONTACT.
    Call this when the user has failed the same checkpoint multiple times and the current
    punishment level is no longer sufficient.
    """
    # 14.1 — Validate checkpoint_id exists
    try:
        oid = ObjectId(checkpoint_id)
    except Exception:
        return f"Error: {checkpoint_id!r} is not a valid checkpoint ID."

    cp = checkpoints.find_one({"_id": oid})
    if cp is None:
        return f"Error: checkpoint {checkpoint_id!r} not found."

    # 14.2 — Advance punishment severity
    current_type = cp.get("failure_punishment", {}).get("type", "WARN")
    try:
        current_idx = ESCALATION_ORDER.index(current_type)
    except ValueError:
        current_idx = 0

    next_idx = min(current_idx + 1, len(ESCALATION_ORDER) - 1)
    next_type = ESCALATION_ORDER[next_idx]

    update_fields: dict = {"failure_punishment.type": next_type}

    # 14.3 — When advancing to CALL_EMERGENCY_CONTACT, set flag for next failure
    if next_type == "CALL_EMERGENCY_CONTACT":
        update_fields["failure_punishment.emergency_alert_on_next_failure"] = True

    checkpoints.update_one({"_id": oid}, {"$set": update_fields})

    # 14.4 — Audit + return
    result = (
        f"Punishment escalated for checkpoint {checkpoint_id}: "
        f"{current_type} → {next_type}. Reason: {reason}"
    )
    _write_audit(
        cp.get("user_id"),
        "escalate_punishment",
        {"checkpoint_id": checkpoint_id, "reason": reason},
        result,
    )
    return result


@tool
def schedule_one_time_activity(user_id: str, activity: str, time: str, date: str = None, action_type: str = "TELEGRAM_TEXT", override_base: bool = True) -> str:
    """Schedule a one-time event for a specific date (Today if date not provided).
    
    date format: YYYY-MM-DD. time format: HH:MM.
    If override_base=True, it will suppress the normal base checkpoint at that same time.
    Use this for: "I have a flight tomorrow, call me at 4am tomorrow", "Skip gym today", etc.
    """
    try:
        uid = ObjectId(user_id)
        if not re.match(r"^\d{2}:\d{2}$", time):
            return "Error: time must be HH:MM."
        
        if not date:
            from pytz import timezone
            tz = timezone("Asia/Kolkata")
            import datetime as dt
            date = dt.datetime.now(tz).strftime("%Y-%m-%d")
            
        from chanakya.db.mongo import daily_events
        
        # If overriding, find the base checkpoint ID
        override_id = None
        if override_base:
            base_cp = checkpoints.find_one({"user_id": uid, "time": time, "active": True})
            if base_cp:
                override_id = base_cp["_id"]
        
        event_doc = {
            "user_id": uid,
            "activity": activity,
            "time": time,
            "date": date,
            "action_type": action_type,
            "override_checkpoint_id": override_id,
            "active": True,
            "fired": False,
            "created_at": datetime.utcnow()
        }
        result_insert = daily_events.insert_one(event_doc)
        event_doc["_id"] = result_insert.inserted_id

        try:
            from chanakya.scheduler.checkpoint_runner import sync_event
            user_doc = users.find_one({"_id": uid})
            if user_doc:
                sync_event(user_doc, event_doc)
        except Exception:
            pass
            
        res = f"One-time activity '{activity}' scheduled for {date} at {time}."
        _write_audit(uid, "schedule_one_time_activity", {"activity": activity, "time": time, "date": date}, res)
        return res
    except Exception as e:
        return f"Error: {e}"


@tool
def modify_wakeup_time(user_id: str, new_time: str, reason: str = "") -> str:
    """Change the user's wake-up checkpoint time. new_time must be HH:MM in 24-hour format.

    Targets the checkpoint with action_type=CALL and the earliest scheduled time.
    Use when the user consistently fails to wake up at the current time.
    """
    # 15.1 — Validate new_time format
    if not re.match(r"^\d{2}:\d{2}$", new_time):
        return (
            f"Error: invalid time format {new_time!r}. Use HH:MM (e.g. 06:30)."
        )

    # 15.2 — Find earliest CALL checkpoint for this user
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: {user_id!r} is not a valid user ID."

    cp = checkpoints.find_one(
        {"user_id": uid, "action_type": "CALL", "active": True},
        sort=[("time", 1)],
    )
    if cp is None:
        return f"Error: no active CALL checkpoint found for user {user_id}."

    # 15.3 — Update + audit + return
    checkpoints.update_one({"_id": cp["_id"]}, {"$set": {"time": new_time}})

    try:
        from chanakya.scheduler.checkpoint_runner import sync_checkpoint
        user_doc = users.find_one({"_id": uid})
        cp["time"] = new_time
        if user_doc:
            sync_checkpoint(user_doc, cp)
    except Exception:
        pass

    # 15.4 — Audit + return
    result = f"Wake-up time changed to {new_time}. Reason: {reason}"
    _write_audit(
        uid,
        "modify_wakeup_time",
        {"user_id": user_id, "new_time": new_time, "reason": reason},
        result,
    )
    return result


@tool
def activate_war_mode(user_id: str, duration_hours: int) -> str:
    """Activate WAR_MODE for the user. Pauses all MEDIUM and LOW priority checkpoints.

    duration_hours must be between 1 and 72.
    Call this when the user sends 'War Mode' or explicitly requests focused work time.
    """
    # 16.1 — Validate duration
    if not (1 <= duration_hours <= 72):
        return (
            f"Error: duration_hours must be between 1 and 72. Got {duration_hours}."
        )

    # 16.2 — Set WAR_MODE
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: {user_id!r} is not a valid user ID."

    expires = datetime.utcnow() + timedelta(hours=duration_hours)
    users.update_one(
        {"_id": uid},
        {"$set": {"current_mode": "WAR_MODE", "war_mode_expires": expires}},
    )

    # 16.3 — Audit + return
    result = (
        f"WAR_MODE activated for {duration_hours} hours. "
        f"Critical alerts only. Expires at {expires.strftime('%Y-%m-%d %H:%M')} UTC."
    )
    _write_audit(
        uid,
        "activate_war_mode",
        {"user_id": user_id, "duration_hours": duration_hours},
        result,
    )
    return result


@tool
def add_daily_checkpoint(
    user_id: str, time_str: str, prompt: str, action_type: str = "TELEGRAM_TEXT"
) -> str:
    """Add a new checkpoint to the user's schedule. time_str must be HH:MM in 24-hour format.

    Use when you detect a new failure pattern that needs monitoring.
    """
    # 17.1 — Validate inputs
    if not re.match(r"^\d{2}:\d{2}$", time_str):
        return f"Error: invalid time format {time_str!r}. Use HH:MM."

    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: {user_id!r} is not a valid user ID."

    user_doc = users.find_one({"_id": uid})
    if user_doc is None:
        return f"Error: user {user_id!r} not found."

    # 17.3 — Insert + audit + return
    new_cp = {
        "user_id": uid,
        "time": time_str,
        "action_type": action_type,
        "prompt_template": prompt,
        "priority": "MEDIUM",
        "requires_response": True,
        "active": True,
        "created_at": datetime.utcnow(),
    }
    result_insert = checkpoints.insert_one(new_cp)
    new_cp["_id"] = result_insert.inserted_id

    try:
        from chanakya.scheduler.checkpoint_runner import sync_checkpoint
        sync_checkpoint(user_doc, new_cp)
    except Exception:
        pass

    result = f"New checkpoint added at {time_str}: {prompt[:50]}..."
    _write_audit(
        uid,
        "add_daily_checkpoint",
        {
            "user_id": user_id,
            "time_str": time_str,
            "prompt": prompt,
            "action_type": action_type,
            "_id": str(result_insert.inserted_id),
        },
        result,
    )
    return result


@tool
def send_emergency_alert(user_id: str, message: str) -> str:
    """Send an SMS to the user's emergency contact. Use only for serious failures.

    Call this when the user is unreachable or has repeatedly failed critical checkpoints.
    """
    # 18.1 — Validate emergency contact
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: {user_id!r} is not a valid user ID."

    user_doc = users.find_one({"_id": uid})
    if user_doc is None:
        return f"Error: user {user_id!r} not found."

    ec = user_doc.get("emergency_contact") or {}
    phone = ec.get("phone")
    if not phone:
        return f"Error: user {user_id!r} has no emergency_contact.phone configured."

    # 18.2 — Send SMS via TwilioClient
    from chanakya.integrations.twilio_client import TwilioClient, TwilioError

    client = TwilioClient()
    body = f"ALERT: {user_doc['name']} is unreachable. {message}"
    try:
        client.send_sms(to=phone, body=body)
    except TwilioError as exc:
        error_result = f"Error: failed to send SMS to emergency contact. {exc}"
        _write_audit(
            uid,
            "send_emergency_alert",
            {"user_id": user_id, "message": message, "phone": phone},
            error_result,
        )
        return error_result

    # 18.3 — Audit + return
    result = (
        f"Emergency alert sent to {ec.get('name', 'emergency contact')} ({phone})."
    )
    _write_audit(
        uid,
        "send_emergency_alert",
        {"user_id": user_id, "message": message, "phone": phone},
        result,
    )
    return result


@tool
def update_morning_todo_time(user_id: str, new_time: str) -> str:
    """Update the user's morning todo delivery time. new_time must be HH:MM in 24-hour format.

    Use when the user requests a change to their morning todo delivery time.
    """
    # 19.1 — Validate inputs
    if not re.match(r"^\d{2}:\d{2}$", new_time):
        return f"Error: invalid time format {new_time!r}. Use HH:MM."

    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: {user_id!r} is not a valid user ID."

    user_doc = users.find_one({"_id": uid})
    if user_doc is None:
        return f"Error: user {user_id!r} not found."

    # 19.2 — Update user + reschedule checkpoint
    users.update_one({"_id": uid}, {"$set": {"morning_todo_time": new_time}})

    # Reschedule the morning todo checkpoint
    checkpoints.update_many(
        {"user_id": uid, "action_type": "TELEGRAM_TEXT", "priority": "LOW"},
        {"$set": {"time": new_time}},
    )

    try:
        from chanakya.scheduler.checkpoint_runner import sync_checkpoint
        updated_cps = list(checkpoints.find(
            {"user_id": uid, "action_type": "TELEGRAM_TEXT", "priority": "LOW", "active": True}
        ))
        for cp in updated_cps:
            sync_checkpoint(user_doc, cp)
    except Exception:
        pass

    # 19.3 — Audit + return
    result = (
        f"Morning todo time updated to {new_time}. "
        f"Daily plan will be sent at this time."
    )
    _write_audit(
        uid,
        "update_morning_todo_time",
        {"user_id": user_id, "new_time": new_time},
        result,
    )
    return result


@tool
def fetch_schedule(user_id: str, day: str = "today") -> str:
    """Fetch the user's checkpoint schedule from the database for a specific day.

    day can be: "today", "tomorrow", or a weekday name like "monday".
    Returns only the checkpoints that apply to that day.
    Use this before discussing or modifying the schedule so you have accurate data.
    When presenting this to the user, format it cleanly — one line per checkpoint.
    """
    import pytz

    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: {user_id!r} is not a valid user ID."

    user_doc = users.find_one({"_id": uid})
    if user_doc is None:
        return f"Error: user {user_id!r} not found."

    # Resolve the target day name
    tz_str = user_doc.get("timezone", "Asia/Kolkata")
    try:
        tz = pytz.timezone(tz_str)
    except Exception:
        tz = pytz.timezone("Asia/Kolkata")

    now_local = datetime.now(tz)
    weekday_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    day_lower = day.strip().lower()
    if day_lower in ("today", ""):
        target_day = now_local.strftime("%A").lower()
    elif day_lower == "tomorrow":
        from datetime import timedelta
        target_day = (now_local + timedelta(days=1)).strftime("%A").lower()
    elif day_lower in weekday_names:
        target_day = day_lower
    else:
        target_day = now_local.strftime("%A").lower()

    # Fetch checkpoints that apply to target_day (or legacy docs with no days field)
    all_cps = list(
        checkpoints.find({"user_id": uid, "active": True}, sort=[("time", 1)])
    )

    day_cps = [
        cp for cp in all_cps
        if not cp.get("days") or target_day in cp.get("days", [])
    ]

    if not day_cps:
        return (
            f"No active checkpoints found for {target_day}. "
            "Run /confirmschedule in Telegram to load the base schedule."
        )

    lines = [f"Schedule for {user_doc.get('name', 'user')} on {target_day.capitalize()} — {len(day_cps)} checkpoints:"]
    for cp in day_cps:
        activity = cp.get("display_name") or (cp.get("activity") or cp.get("prompt_template", ""))[:30]
        action = cp.get("action_type", "?").lower().replace("_", " ")
        priority = cp.get("priority", "?").lower()
        cp_id = str(cp["_id"])
        lines.append(f"{cp['time']} | {activity} | {action} | {priority} | id={cp_id}")

    result = "\n".join(lines)
    _write_audit(uid, "fetch_schedule", {"user_id": user_id, "day": target_day}, result[:500])
    return result


@tool
def update_schedule_activity(
    user_id: str,
    checkpoint_id: str,
    field: str,
    value: str,
    reason: str = "",
) -> str:
    """Modify a single field on an existing checkpoint.

    field must be one of: time, prompt_template, action_type, priority, active.
    value is always a string — booleans use "true"/"false", times use HH:MM.
    reason is optional — describe why you're making the change.
    Use fetch_schedule first to get the checkpoint_id you want to modify.
    """
    valid_fields = {"time", "prompt_template", "action_type", "priority", "active"}
    if field not in valid_fields:
        return f"Error: field must be one of {sorted(valid_fields)}. Got {field!r}."

    try:
        uid = ObjectId(user_id)
        cp_oid = ObjectId(checkpoint_id)
    except Exception as exc:
        return f"Error: invalid ID — {exc}"

    cp = checkpoints.find_one({"_id": cp_oid, "user_id": uid})
    if cp is None:
        return f"Error: checkpoint {checkpoint_id!r} not found for this user."

    # Coerce value to the right type
    coerced: object = value
    if field == "time":
        if not re.match(r"^\d{2}:\d{2}$", value):
            return f"Error: time must be HH:MM. Got {value!r}."
    elif field == "active":
        if value.lower() not in ("true", "false"):
            return "Error: active must be 'true' or 'false'."
        coerced = value.lower() == "true"
    elif field == "priority":
        if value not in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            return "Error: priority must be CRITICAL, HIGH, MEDIUM, or LOW."
    elif field == "action_type":
        valid_actions = {"CALL", "TELEGRAM_TEXT", "TELEGRAM_VOICE", "IMAGE_DEMAND"}
        if value not in valid_actions:
            return f"Error: action_type must be one of {sorted(valid_actions)}."

    old_value = cp.get(field)
    checkpoints.update_one({"_id": cp_oid}, {"$set": {field: coerced}})

    try:
        from chanakya.scheduler.checkpoint_runner import sync_checkpoint, unsync_checkpoint
        user_doc = users.find_one({"_id": uid})
        if user_doc:
            cp[field] = coerced
            if coerced is False and field == "active":
                unsync_checkpoint(cp_oid)
            else:
                sync_checkpoint(user_doc, cp)
    except Exception:
        pass

    result = (
        f"Checkpoint {checkpoint_id} updated: {field} changed from "
        f"{old_value!r} → {coerced!r}. Reason: {reason}"
    )
    _write_audit(
        uid,
        "update_schedule_activity",
        {
            "user_id": user_id,
            "checkpoint_id": checkpoint_id,
            "field": field,
            "value": value,
            "reason": reason,
        },
        result,
    )
    return result


@tool
def save_contact(user_id: str, name: str, phone: str, relationship: str = "") -> str:
    """Save or update a contact for the user (mom, bro, boss, etc.).

    phone must be in E.164 format e.g. +919876543210.
    relationship is optional (e.g. mother, brother, manager).
    Use this when the user says things like:
      "save mom's number +919876543210"
      "add contact: Bro +919123456789 brother"
      "my mom's number is +91XXXXXXXXXX"
    """
    if not re.match(r"^\+\d{7,15}$", phone.strip()):
        return f"Error: phone must be E.164 format (e.g. +919876543210). Got {phone!r}."

    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    from chanakya.db.mongo import add_contact
    add_contact(uid, name.strip(), phone.strip(), relationship.strip())

    result = f"Contact saved: {name} ({phone})" + (f" — {relationship}" if relationship else "")
    _write_audit(uid, "save_contact", {"name": name, "phone": phone, "relationship": relationship}, result)
    return result


@tool
def list_contacts(user_id: str) -> str:
    """List all saved contacts for the user.

    Use this when the user asks:
      "who are my contacts"
      "show my contacts"
      "what numbers do you have"
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    from chanakya.db.mongo import get_contacts
    contact_list = get_contacts(uid)

    if not contact_list:
        return "No contacts saved yet. Ask the user to share a name and phone number."

    lines = [f"Contacts ({len(contact_list)} total):"]
    for c in contact_list:
        rel = f" [{c.get('relationship', '')}]" if c.get("relationship") else ""
        lines.append(f"  {c['name']}{rel} — {c['phone']}")
    return "\n".join(lines)


@tool
def delete_contact(user_id: str, name: str) -> str:
    """Delete a contact by name.

    Use when the user says:
      "remove mom from contacts"
      "delete contact Bro"
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    from chanakya.db.mongo import remove_contact
    removed = remove_contact(uid, name.strip())

    result = f"Contact '{name}' removed." if removed else f"No contact named '{name}' found."
    _write_audit(uid, "delete_contact", {"name": name}, result)
    return result


@tool
def set_user_phone(user_id: str, phone: str) -> str:
    """Set the user's own phone number for receiving Twilio calls.

    phone must be E.164 format e.g. +919876543210.
    Use when the user says:
      "my number is +91XXXXXXXXXX"
      "set my phone to +91XXXXXXXXXX"
      "call me on +91XXXXXXXXXX"
    """
    if not re.match(r"^\+\d{7,15}$", phone.strip()):
        return f"Error: phone must be E.164 format (e.g. +919876543210). Got {phone!r}."

    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    users.update_one({"_id": uid}, {"$set": {"phone": phone.strip()}})
    result = f"Your phone number set to {phone}. Chanakya will call you at this number."
    _write_audit(uid, "set_user_phone", {"phone": phone}, result)
    return result


@tool
def place_proxy_call(user_id: str, contact_name: str, topic: str) -> str:
    """Place a call to one of the user's contacts on their behalf.

    Chanakya will call the contact, introduce itself as calling on behalf of the user,
    discuss the topic, and send a summary back to the user via Telegram.

    Use when the user says things like:
      "call mom and ask what's for dinner"
      "call bro about the weekend plan"
      "ask my boss about the meeting time"
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    from chanakya.db.mongo import get_contact_by_name, users as users_col, interaction_logs, proxy_call_logs
    from chanakya.config import WEBHOOK_URL

    user_doc = users_col.find_one({"_id": uid})
    if not user_doc:
        return f"Error: user not found."

    contact = get_contact_by_name(uid, contact_name)
    if not contact:
        return (
            f"No contact named '{contact_name}' found. "
            f"Ask the user to share {contact_name}'s phone number first."
        )

    phone = contact.get("phone", "")
    if not phone:
        return f"Contact '{contact_name}' has no phone number saved."

    if not WEBHOOK_URL:
        return "Error: WEBHOOK_URL not configured — cannot place call."

    owner_name = user_doc.get("name", "Goutham")
    owner_telegram_id = user_doc.get("telegram_id", "")

    # Insert task for Task Manager to pick up
    from chanakya.db.mongo import agent_tasks
    task_doc = {
        "user_id": uid,
        "task_type": "PROXY_CALL",
        "payload": {"contact_name": contact_name, "topic": topic},
        "status": "PENDING",
        "retries_attempted": 0,
        "max_retries": 3,
        "created_at": datetime.utcnow()
    }
    
    try:
        agent_tasks.insert_one(task_doc)
        result = (
            f"Proxy call to {contact['name']} ({phone}) has been assigned to the Task Manager. "
            f"Topic: {topic}. You'll receive a summary when the task completes. "
            f"If it fails, it will be automatically retried."
        )
        _write_audit(uid, "place_proxy_call", {"contact": contact_name, "topic": topic}, result)
        return result
    except Exception as exc:
        return f"Failed to assign task: {exc}"


@tool
def get_user_status(user_id: str) -> str:
    """Return the user's current streak, failures this week, and mode.

    Use when the user asks:
      "what's my streak"
      "how am I doing"
      "show my status"
      "how many failures this week"
    """
    import pytz
    from datetime import timedelta

    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    user_doc = users.find_one({"_id": uid})
    if not user_doc:
        return "Error: user not found."

    from chanakya.db.mongo import interaction_logs as il
    tz_str = user_doc.get("timezone", "Asia/Kolkata")
    try:
        tz = pytz.timezone(tz_str)
    except Exception:
        tz = pytz.timezone("Asia/Kolkata")

    now_local = datetime.now(tz)
    days_since_monday = now_local.weekday()
    week_start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
    week_start_utc = week_start_local.astimezone(pytz.utc).replace(tzinfo=None)

    failures = il.count_documents({
        "user_id": uid,
        "ai_evaluation.verdict": "FAILED",
        "timestamp": {"$gte": week_start_utc},
    })

    streak = user_doc.get("streak_count", 0)
    longest = user_doc.get("longest_streak", 0)
    mode = user_doc.get("current_mode", "NORMAL")

    return (
        f"Streak: {streak} days (best: {longest})\n"
        f"Failures this week: {failures}\n"
        f"Mode: {mode}"
    )


@tool
def deactivate_war_mode(user_id: str) -> str:
    """Deactivate WAR_MODE and return to NORMAL schedule.

    Use when the user says:
      "peace mode"
      "deactivate war mode"
      "back to normal"
      "end war mode"
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    users.update_one({"_id": uid}, {"$set": {"current_mode": "NORMAL", "war_mode_expires": None}})
    result = "WAR_MODE deactivated. Normal schedule resuming."
    _write_audit(uid, "deactivate_war_mode", {}, result)
    return result


@tool
def add_mindset_note(user_id: str, note: str) -> str:
    """Add a personal mindset note or instruction (simple flat note).

    Use when the user says:
      "add mindset note: discipline over comfort"
      "remember: I don't negotiate with comfort"
      "add instruction: be harsh with me about gym"

    For richer typed entries (quote, goal, trait, rule, reference),
    use add_mindset_entry instead.
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    from chanakya.db.mongo import add_personal_instruction
    count = add_personal_instruction(uid, note.strip())
    result = f"Mindset note added ({count} total): \"{note.strip()}\""
    _write_audit(uid, "add_mindset_note", {"note": note}, result)
    return result


@tool
def add_mindset_entry(user_id: str, category: str, text: str, source: str = "") -> str:
    """Add a typed mindset/identity entry that shapes how Chanakya sees and guides the user.

    category must be one of:
      quote     — a quote to embody or be reminded of
      goal      — a life goal (e.g. "become a billionaire", "master system design by 2027")
      trait     — a character trait to build (e.g. "fearlessness", "discipline")
      rule      — a personal rule (e.g. "never negotiate with comfort")
      reference — a story/person Chanakya should use (e.g. "when I doubt, remind me of Arjuna")
      note      — anything else

    source is optional — who said it or where it's from
    (e.g. "Harvey Specter", "Bhagavad Gita 2.47", "Chanakya Niti").

    Use when the user says:
      "add quote: I don't get lucky, I make my own luck — Harvey Specter"
      "add goal: become a billionaire before 35"
      "add trait: think like a warrior, act like a king"
      "add rule: never miss gym twice in a row"
      "add reference: when I'm lazy, remind me of Hanuman crossing the ocean"
      "save this quote from the Gita: ..."
      "I want to remember this: ..."
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    from chanakya.db.mongo import add_mindset_entry as _add
    count = _add(uid, category=category, text=text.strip(), source=source.strip())
    source_str = f" (source: {source})" if source else ""
    result = f"Mindset {category} saved ({count} total): \"{text.strip()}\"{source_str}"
    _write_audit(uid, "add_mindset_entry", {"category": category, "text": text, "source": source}, result)
    return result


@tool
def get_mindset_notes(user_id: str) -> str:
    """Return all the user's mindset notes, goals, quotes, rules, traits, and references.

    Use when the user asks:
      "show my mindset"
      "what are my goals"
      "list my quotes"
      "show my principles"
      "what have I saved"
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    from chanakya.db.mongo import get_all_identity_context
    data = get_all_identity_context(uid)
    lines = []

    instructions = data.get("instructions", [])
    if instructions:
        lines.append("Personal Rules / Instructions:")
        for i, item in enumerate(instructions, 1):
            lines.append(f"  {i}. {item}")

    mindset = data.get("mindset", [])
    if mindset:
        from collections import defaultdict
        by_cat: dict = defaultdict(list)
        for e in mindset:
            if isinstance(e, dict):
                by_cat[e.get("category", "note")].append(e)
        cat_labels = {
            "quote": "Quotes", "goal": "Life Goals", "trait": "Traits to Build",
            "rule": "Personal Rules", "reference": "References for Chanakya", "note": "Notes",
        }
        offset = len(instructions)
        for cat, label in cat_labels.items():
            entries = by_cat.get(cat, [])
            if not entries:
                continue
            lines.append(f"\n{label}:")
            for e in entries:
                offset += 1
                text = e.get("text", "")
                source = e.get("source", "")
                line = f"  {offset}. {text}"
                if source:
                    line += f"  — {source}"
                lines.append(line)

    if not lines:
        return "Nothing saved yet. Tell me a quote, goal, rule, or trait and I'll store it."
    return "\n".join(lines)


@tool
def remove_mindset_note(user_id: str, index: int) -> str:
    """Remove a mindset entry by its 1-based number (from get_mindset_notes list).

    Use when the user says:
      "remove mindset note 2"
      "delete goal 3"
      "remove quote 1"
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    from chanakya.db.mongo import (
        remove_personal_instruction, get_personal_instructions,
        remove_mindset_entry, get_mindset_entries,
    )
    instructions = get_personal_instructions(uid)
    n_instructions = len(instructions)
    if index <= n_instructions:
        removed = remove_personal_instruction(uid, index - 1)
    else:
        removed = remove_mindset_entry(uid, index - n_instructions - 1)

    result = f"Entry #{index} removed." if removed else f"No entry at position {index}."
    _write_audit(uid, "remove_mindset_note", {"index": index}, result)
    return result


@tool
def clear_mindset_notes(user_id: str) -> str:
    """Remove ALL mindset notes, goals, quotes, rules, traits, and references.

    Use when the user says:
      "clear everything"
      "reset my mindset"
      "wipe all my notes"
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    from chanakya.db.mongo import clear_personal_instructions, clear_mindset_entries
    clear_personal_instructions(uid)
    clear_mindset_entries(uid)
    result = "All mindset entries cleared."
    _write_audit(uid, "clear_mindset_notes", {}, result)
    return result


@tool
def set_morning_todo_time(user_id: str, time_str: str) -> str:
    """Set the time Chanakya sends the morning todo plan.

    time_str must be HH:MM in 24-hour format.
    Use when the user says:
      "set my morning todo to 8:30"
      "send daily plan at 09:00"
      "change todo time to 7:45"
    """
    if not re.match(r"^\d{2}:\d{2}$", time_str.strip()):
        return f"Error: time must be HH:MM (e.g. 08:30). Got {time_str!r}."

    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    users.update_one({"_id": uid}, {"$set": {"morning_todo_time": time_str.strip()}})
    result = f"Morning todo time set to {time_str}. Daily plan will be sent then."
    _write_audit(uid, "set_morning_todo_time", {"time": time_str}, result)
    return result


@tool
def reload_prompt_templates(user_id: str) -> str:
    """Clear the prompt template cache so templates reload from DB on next interaction.

    Use when the user says:
      "reload templates"
      "refresh prompts"
      "clear template cache"
    """
    from chanakya.agent.context_assembler import clear_template_cache
    clear_template_cache()
    result = "Prompt template cache cleared. Templates will reload on next interaction."
    _write_audit(ObjectId(user_id) if user_id else None, "reload_prompt_templates", {}, result)
    return result


@tool
def call_user(user_id: str) -> str:
    """Place an on-demand voice call to the user (Chanakya calls them).

    Use when the user says:
      "call me"
      "give me a call"
      "I want to talk"
    The call will use the phone number stored in the user's profile.
    Returns a status string — the actual call is placed asynchronously.
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    user_doc = users.find_one({"_id": uid})
    if not user_doc:
        return "Error: user not found."

    phone = user_doc.get("phone", "")
    if not phone:
        return (
            "No phone number on file. "
            "Tell me your number first: \"my number is +91XXXXXXXXXX\""
        )

    from chanakya.config import WEBHOOK_URL
    if not WEBHOOK_URL:
        return "Error: WEBHOOK_URL not configured — cannot place call."

    # Build a context-aware opening
    from chanakya.agent.context_assembler import ContextAssembler
    try:
        assembler = ContextAssembler()
        # ContextAssembler.build is async — run it safely from this sync context
        try:
            _loop = asyncio.get_running_loop()
        except RuntimeError:
            _loop = None

        if _loop is not None and _loop.is_running():
            # We're inside an async context — can't use asyncio.run(), use a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, assembler.build(user_doc, "MENTOR_TALK", None))
                ctx = future.result(timeout=5)
        else:
            ctx = asyncio.run(assembler.build(user_doc, "MENTOR_TALK", None))

        tier1 = ctx.get("tier1") or {}
        opening_text = (
            f"Chanakya here. Your streak is {tier1.get('streak_count', 0)} days. "
            f"Mode: {tier1.get('current_mode', 'NORMAL')}. What do you need to discuss?"
        )
    except Exception:
        opening_text = (
            f"Chanakya here. Streak: {user_doc.get('streak_count', 0)} days. "
            "What do you need to discuss?"
        )

    from datetime import datetime as _dt
    from chanakya.db.mongo import agent_tasks
    task_doc = {
        "user_id": uid,
        "task_type": "CALL_USER",
        "payload": {"opening_text": opening_text},
        "status": "PENDING",
        "retries_attempted": 0,
        "max_retries": 3,
        "created_at": _dt.utcnow()
    }

    try:
        agent_tasks.insert_one(task_doc)
        result = (
            "Chanakya is initiating your call. You'll receive it on your registered number shortly. "
            "The task has been assigned to the Task Manager for reliable execution."
        )
        _write_audit(uid, "call_user", {"phone": phone}, result)
        return result
    except Exception as exc:
        return f"Failed to initiate call task: {exc}"


@tool
def fetch_day_schedule(user_id: str, date: str = "today") -> str:
    """Fetch the full schedule for a specific date — base weekday checkpoints merged
    with any date-specific events or reminders added for that date.

    date can be: "today", "tomorrow", "yesterday", or "YYYY-MM-DD".
    Use this when the user asks:
      "what's my schedule for tomorrow"
      "show me April 25th"
      "what do I have on Friday"
      "show schedule for 2026-05-01"
    """
    import pytz
    from chanakya.db.mongo import daily_events as de_col

    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    user_doc = users.find_one({"_id": uid})
    if not user_doc:
        return "Error: user not found."

    tz_str = user_doc.get("timezone", "Asia/Kolkata")
    try:
        tz = pytz.timezone(tz_str)
    except Exception:
        tz = pytz.timezone("Asia/Kolkata")

    now_local = datetime.now(tz)
    date_lower = date.strip().lower()

    if date_lower in ("today", ""):
        target_dt = now_local
    elif date_lower == "tomorrow":
        target_dt = now_local + timedelta(days=1)
    elif date_lower == "yesterday":
        target_dt = now_local - timedelta(days=1)
    else:
        try:
            from datetime import datetime as _dt
            target_dt = tz.localize(_dt.strptime(date.strip(), "%Y-%m-%d"))
        except ValueError:
            return f"Error: date must be YYYY-MM-DD, 'today', 'tomorrow', or 'yesterday'. Got {date!r}."

    target_date_str = target_dt.strftime("%Y-%m-%d")
    target_dow = target_dt.strftime("%A").lower()

    # Base checkpoints for that weekday
    all_base = list(checkpoints.find({"user_id": uid, "active": True}, sort=[("time", 1)]))
    base_for_day = [
        cp for cp in all_base
        if not cp.get("days") or target_dow in cp.get("days", [])
    ]

    # Date-specific events
    date_evts = list(de_col.find({"user_id": uid, "date": target_date_str, "active": True}, sort=[("time", 1)]))

    # Overridden base checkpoint IDs
    overridden = {e.get("override_checkpoint_id") for e in date_evts if e.get("override_checkpoint_id")}
    base_for_day = [cp for cp in base_for_day if cp["_id"] not in overridden]

    lines = [f"📅 Schedule for {target_date_str} ({target_dow.capitalize()})"]

    if not base_for_day and not date_evts:
        lines.append("No scheduled items.")
        return "\n".join(lines)

    # Merge and sort by time
    items = []
    for cp in base_for_day:
        items.append({
            "time": cp["time"],
            "activity": cp.get("display_name") or cp.get("activity") or cp.get("prompt_template", "")[:40],
            "action": cp.get("action_type", "").lower().replace("_", " "),
            "priority": cp.get("priority", "").lower(),
            "id": str(cp["_id"]),
            "source": "base",
            "note": cp.get("description", ""),
        })
    for e in date_evts:
        items.append({
            "time": e["time"],
            "activity": e.get("display_name") or e.get("activity", ""),
            "action": e.get("action_type", "TELEGRAM_TEXT").lower().replace("_", " "),
            "priority": e.get("priority", "medium").lower(),
            "id": str(e["_id"]),
            "source": "custom",
            "note": e.get("note", ""),
        })

    items.sort(key=lambda x: x["time"])

    for item in items:
        tag = "📌" if item["source"] == "custom" else "•"
        note_str = f" — {item['note']}" if item["note"] else ""
        # Put id first so LLM can't miss it when making tool calls
        lines.append(
            f"{tag} [{item['id']}] {item['time']} | {item['activity']} | {item['action']} | {item['priority']}{note_str}"
        )

    result = "\n".join(lines)
    _write_audit(uid, "fetch_day_schedule", {"date": target_date_str}, result[:500])
    return result


@tool
def add_day_event(
    user_id: str,
    date: str,
    time_str: str,
    activity: str,
    note: str = "",
    display_name: str = "",
    description: str = "",
    action_type: str = "TELEGRAM_TEXT",
    priority: str = "MEDIUM",
    override_checkpoint_id: str = "",
) -> str:
    """Add a date-specific event, reminder, or schedule override.

    date must be YYYY-MM-DD, "today", or "tomorrow".
    time_str must be HH:MM.
    activity is a short CAPS_UNDERSCORE label (e.g. "MEETING_KARTIK", "DENTIST").
    display_name is a clean human-readable name (e.g. "Meeting with Kartik").
    description is 1-2 sentences about what this event is and why it matters.
    note is an optional short reminder text shown at trigger time.
    override_checkpoint_id: if set, suppresses the base checkpoint with that ID on this date.

    Use when the user mentions any event, meeting, appointment, or plan:
      "meeting Kartik tomorrow 5pm"
      "dentist on April 25 at 10am"
      "team standup tomorrow 9:30"
      "remind me to call landlord on May 10 at 11am"
    Always extract display_name and description from the user's natural language.
    """
    import pytz
    from chanakya.db.mongo import daily_events as de_col

    if not re.match(r"^\d{2}:\d{2}$", time_str.strip()):
        return f"Error: time must be HH:MM. Got {time_str!r}."

    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    user_doc = users.find_one({"_id": uid})
    if not user_doc:
        return "Error: user not found."

    tz_str = user_doc.get("timezone", "Asia/Kolkata")
    try:
        tz = pytz.timezone(tz_str)
    except Exception:
        tz = pytz.timezone("Asia/Kolkata")

    now_local = datetime.now(tz)
    date_lower = date.strip().lower()
    if date_lower == "today":
        target_date_str = now_local.strftime("%Y-%m-%d")
    elif date_lower == "tomorrow":
        target_date_str = (now_local + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        try:
            from datetime import datetime as _dt
            _dt.strptime(date.strip(), "%Y-%m-%d")
            target_date_str = date.strip()
        except ValueError:
            return f"Error: date must be YYYY-MM-DD, 'today', or 'tomorrow'. Got {date!r}."

    override_oid = None
    if override_checkpoint_id:
        try:
            override_oid = ObjectId(override_checkpoint_id)
        except Exception:
            pass

    # Fall back: derive display_name from activity if not provided
    clean_display = display_name.strip() or activity.strip().replace("_", " ").title()
    clean_description = description.strip() or note.strip()

    now_utc = datetime.utcnow()
    doc = {
        "user_id": uid,
        "date": target_date_str,
        "time": time_str.strip(),
        "activity": activity.strip().upper(),
        "display_name": clean_display,
        "description": clean_description,
        "note": note.strip(),
        "action_type": action_type.upper(),
        "priority": priority.upper(),
        "source": "user",
        "active": True,
        "fired": False,
        "override_checkpoint_id": override_oid,
        "created_at": now_utc,
        "updated_at": now_utc,
    }
    result_insert = de_col.insert_one(doc)
    doc["_id"] = result_insert.inserted_id

    try:
        from chanakya.scheduler.checkpoint_runner import sync_event
        sync_event(user_doc, doc)
    except Exception:
        pass

    result = f"Event added on {target_date_str} at {time_str}: {clean_display}" + (f" — {clean_description}" if clean_description else "")
    _write_audit(uid, "add_day_event", {"date": target_date_str, "time": time_str, "activity": activity, "display_name": clean_display}, result)
    return result + f" | id={result_insert.inserted_id}"


@tool
def update_day_event(
    user_id: str,
    event_id: str,
    field: str,
    value: str,
    scope: str = "this_date",
) -> str:
    """Modify a field on a date-specific event OR on the base checkpoint for all matching weekdays.

    field: time | activity | note | action_type | priority | active
    scope:
      "this_date" — only this specific date's event (default)
      "all_weekdays" — update the base checkpoint for every matching weekday

    Use when the user says:
      "change tomorrow's gym to 7:30" → scope=this_date
      "move all Monday gym to 7:30" → scope=all_weekdays
      "cancel dentist on April 25" → field=active, value=false, scope=this_date
    """
    from chanakya.db.mongo import daily_events as de_col

    valid_fields = {"time", "activity", "note", "action_type", "priority", "active"}
    if field not in valid_fields:
        return f"Error: field must be one of {sorted(valid_fields)}."

    try:
        uid = ObjectId(user_id)
        evt_oid = ObjectId(event_id)
    except Exception as exc:
        return f"Error: invalid ID — {exc}"

    if scope == "all_weekdays":
        # Find the base checkpoint via override_checkpoint_id or by activity match
        evt = de_col.find_one({"_id": evt_oid, "user_id": uid})
        if not evt:
            return f"Error: event {event_id!r} not found."
        cp_id = evt.get("override_checkpoint_id")
        if not cp_id:
            # Try to find by activity + time in base checkpoints
            cp = checkpoints.find_one({"user_id": uid, "activity": evt.get("activity"), "active": True})
            cp_id = cp["_id"] if cp else None
        if not cp_id:
            return "Error: could not find matching base checkpoint for all_weekdays scope."
        coerced: object = value
        if field == "active":
            coerced = value.lower() == "true"
        checkpoints.update_one({"_id": cp_id}, {"$set": {field: coerced}})
        result = f"Base checkpoint updated for all weekdays: {field}={value!r}."
        try:
            from chanakya.scheduler.checkpoint_runner import sync_checkpoint, unsync_checkpoint
            updated_cp = checkpoints.find_one({"_id": cp_id})
            user_doc = users.find_one({"_id": uid})
            if updated_cp and user_doc:
                if field == "active" and coerced is False:
                    unsync_checkpoint(cp_id)
                else:
                    sync_checkpoint(user_doc, updated_cp)
        except Exception:
            pass
    else:
        evt = de_col.find_one({"_id": evt_oid, "user_id": uid})
        if not evt:
            return f"Error: event {event_id!r} not found."
        coerced = value
        if field == "active":
            coerced = value.lower() == "true"
        de_col.update_one({"_id": evt_oid}, {"$set": {field: coerced, "updated_at": datetime.utcnow()}})
        result = f"Event {event_id} updated: {field}={value!r} (this date only)."
        try:
            from chanakya.scheduler.checkpoint_runner import sync_event, unsync_checkpoint
            from chanakya.db.mongo import daily_events as _de
            if field == "active" and coerced is False:
                unsync_checkpoint(evt_oid)
            else:
                updated_evt = _de.find_one({"_id": evt_oid})
                user_doc = users.find_one({"_id": uid})
                if updated_evt and user_doc:
                    sync_event(user_doc, updated_evt)
        except Exception:
            pass

    _write_audit(uid, "update_day_event", {"event_id": event_id, "field": field, "value": value, "scope": scope}, result)
    return result


@tool
def delete_day_event(user_id: str, event_id: str) -> str:
    """Delete a date-specific event or reminder.

    Use when the user says:
      "remove the dentist reminder on April 25"
      "delete that event"
      "cancel the 10am meeting tomorrow"
    """
    from chanakya.db.mongo import daily_events as de_col

    try:
        uid = ObjectId(user_id)
        evt_oid = ObjectId(event_id)
    except Exception as exc:
        return f"Error: invalid ID — {exc}"

    result_del = de_col.delete_one({"_id": evt_oid, "user_id": uid})
    if result_del.deleted_count:
        result = f"Event {event_id} deleted."
        try:
            from chanakya.scheduler.checkpoint_runner import unsync_checkpoint
            unsync_checkpoint(evt_oid)
        except Exception:
            pass
    else:
        result = f"No event found with id={event_id}."
    _write_audit(uid, "delete_day_event", {"event_id": event_id}, result)
    return result


@tool
def get_day_log(user_id: str, date: str = "today") -> str:
    """Return a log of everything that happened on a specific date —
    what was scheduled, what fired, what the user responded, and verdicts.

    date can be: "today", "yesterday", or "YYYY-MM-DD".
    Use when the user asks:
      "what did I do on April 19th"
      "show me yesterday's log"
      "what happened on 2026-04-15"
      "how was my day on Monday"
    """
    import pytz
    from chanakya.db.mongo import interaction_logs as il_col, daily_events as de_col

    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    user_doc = users.find_one({"_id": uid})
    if not user_doc:
        return "Error: user not found."

    tz_str = user_doc.get("timezone", "Asia/Kolkata")
    try:
        tz = pytz.timezone(tz_str)
    except Exception:
        tz = pytz.timezone("Asia/Kolkata")

    now_local = datetime.now(tz)
    date_lower = date.strip().lower()
    if date_lower in ("today", ""):
        target_dt = now_local
    elif date_lower == "yesterday":
        target_dt = now_local - timedelta(days=1)
    else:
        try:
            from datetime import datetime as _dt
            target_dt = tz.localize(_dt.strptime(date.strip(), "%Y-%m-%d"))
        except ValueError:
            return f"Error: date must be YYYY-MM-DD, 'today', or 'yesterday'. Got {date!r}."

    target_date_str = target_dt.strftime("%Y-%m-%d")
    day_start_utc = target_dt.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(pytz.utc).replace(tzinfo=None)
    day_end_utc = target_dt.replace(hour=23, minute=59, second=59, microsecond=999999).astimezone(pytz.utc).replace(tzinfo=None)

    # Interaction logs for this date
    logs = list(il_col.find(
        {"user_id": uid, "timestamp": {"$gte": day_start_utc, "$lte": day_end_utc}},
        sort=[("timestamp", 1)]
    ))

    if not logs:
        return f"No activity logged for {target_date_str}."

    lines = [f"📋 Activity log for {target_date_str}:"]
    for log in logs:
        ts_local = pytz.utc.localize(log["timestamp"]).astimezone(tz)
        time_str = ts_local.strftime("%H:%M")
        channel = log.get("channel", "?")
        verdict = log.get("ai_evaluation", {}).get("verdict") or "—"
        msg = (log.get("message_sent") or "")[:60]
        resp = (log.get("user_response") or "")[:60]
        lines.append(f"  {time_str} [{channel}] verdict={verdict}")
        if msg:
            lines.append(f"    → sent: {msg}")
        if resp:
            lines.append(f"    ← user: {resp}")

    result = "\n".join(lines)
    _write_audit(uid, "get_day_log", {"date": target_date_str}, result[:300])
    return result


@tool
def send_telegram_message(user_id: str, message: str) -> str:
    """Send a Telegram message to the user immediately — proactive push, not a reply.

    Use this when Chanakya wants to send something unprompted, like:
      - A reminder mid-call: "I'll send you the updated schedule now"
      - A follow-up after a call ends
      - An alert or nudge at any time
      - Sending a summary, plan, or note to the user

    The message is sent to the user's Telegram chat right now.
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    user_doc = users.find_one({"_id": uid})
    if not user_doc:
        return "Error: user not found."

    chat_id = user_doc.get("telegram_id", "")
    if not chat_id:
        return "Error: user has no telegram_id stored."

    import asyncio
    import re as _re
    import html as _html

    def _md_to_html(text: str) -> str:
        text = _re.sub(r"<br\s*/?>", "\n", text, flags=_re.IGNORECASE)
        text = _re.sub(r"<[^>]+>", "", text)
        text = _html.unescape(text)
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = _re.sub(r"^#{1,3}\s+(.+)$", r"<b>\1</b>", text, flags=_re.MULTILINE)
        text = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = _re.sub(r"__(.+?)__", r"<b>\1</b>", text)
        text = _re.sub(r"\*([^*\n]+?)\*", r"<i>\1</i>", text)
        text = _re.sub(r"_([^_\n]+?)_", r"<i>\1</i>", text)
        text = _re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
        text = _re.sub(r"^[\-\*]\s+", "• ", text, flags=_re.MULTILINE)
        return text.strip()

    formatted = _md_to_html(message)

    async def _send():
        from telegram import Bot
        from telegram.error import BadRequest
        from chanakya.config import TELEGRAM_BOT_TOKEN
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        try:
            await bot.send_message(chat_id=chat_id, text=formatted, parse_mode="HTML")
        except BadRequest as exc:
            if "parse" in str(exc).lower() or "entities" in str(exc).lower():
                await bot.send_message(chat_id=chat_id, text=message)
            else:
                raise

    try:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None and running_loop.is_running():
            # We're inside the async agent loop on this thread — fire-and-forget.
            asyncio.ensure_future(_send())
        else:
            # Background thread (APScheduler, Twilio webhook thread, etc.)
            # Find the main event loop and schedule the coroutine on it.
            import threading
            main_loop = None
            for thread in threading.enumerate():
                if hasattr(thread, "_target") and thread._target is not None:
                    pass
                loop_attr = getattr(thread, "_asyncio_loop", None)
                if loop_attr is not None and loop_attr.is_running():
                    main_loop = loop_attr
                    break

            if main_loop is None:
                # Last resort: try the global event loop policy
                try:
                    main_loop = asyncio.get_event_loop_policy().get_event_loop()
                except RuntimeError:
                    main_loop = None

            if main_loop is not None and main_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(_send(), main_loop)
                future.result(timeout=10)  # wait up to 10s
            else:
                # No running loop anywhere — create a fresh one just for this call
                asyncio.run(_send())

        result = f"Message sent to {user_doc.get('name', 'user')} on Telegram."
        _write_audit(uid, "send_telegram_message", {"message": message[:100]}, result)
        return result
    except Exception as exc:
        return f"Error sending Telegram message: {exc}"


@tool
def schedule_message(user_id: str, message: str, send_at: str, date: str = "today") -> str:
    """Schedule a Telegram message to be sent to the user at a specific time.

    send_at must be HH:MM in 24-hour format.
    date can be "today" (default), "tomorrow", or "YYYY-MM-DD".
    The message will be sent by the checkpoint runner at that time.

    Use when the user says:
      "remind me at 9pm to do leetcode"
      "send me a message tomorrow at 8:30"
      "ping me on May 5th at 10pm with the plan"
    """
    import re as _re
    if not _re.match(r"^\d{2}:\d{2}$", send_at.strip()):
        return f"Error: send_at must be HH:MM. Got {send_at!r}."

    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    user_doc = users.find_one({"_id": uid})
    if not user_doc:
        return "Error: user not found."

    import pytz
    from chanakya.db.mongo import daily_events as de_col

    tz_str = user_doc.get("timezone", "Asia/Kolkata")
    try:
        tz = pytz.timezone(tz_str)
    except Exception:
        tz = pytz.timezone("Asia/Kolkata")

    now_local = datetime.now(tz)
    date_lower = date.strip().lower()
    if date_lower in ("today", ""):
        target_date_str = now_local.strftime("%Y-%m-%d")
    elif date_lower == "tomorrow":
        target_date_str = (now_local + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        try:
            from datetime import datetime as _dt
            _dt.strptime(date.strip(), "%Y-%m-%d")
            target_date_str = date.strip()
        except ValueError:
            return f"Error: date must be 'today', 'tomorrow', or YYYY-MM-DD. Got {date!r}."

    now_utc = datetime.utcnow()
    doc = {
        "user_id": uid,
        "date": target_date_str,
        "time": send_at.strip(),
        "activity": "SCHEDULED_MESSAGE",
        "note": message,
        "action_type": "TELEGRAM_TEXT",
        "priority": "MEDIUM",
        "source": "agent",
        "active": True,
        "fired": False,
        "created_at": now_utc,
        "updated_at": now_utc,
    }
    result_insert = de_col.insert_one(doc)
    result = f"Message scheduled for {target_date_str} at {send_at}: \"{message[:60]}\""
    _write_audit(uid, "schedule_message", {"date": target_date_str, "send_at": send_at, "message": message[:100]}, result)
    return result + f" | event_id={result_insert.inserted_id}"


@tool
def reschedule_activity(
    user_id: str,
    activity: str,
    new_time: str,
    date: str = "today",
) -> str:
    """Move an activity to a new time by name — no event_id needed.

    activity is the display name or activity key (e.g. "Gym Session", "GYM", "Wake Up", "WAKE_UP").
    new_time must be HH:MM.
    date can be "today", "tomorrow", or YYYY-MM-DD.

    Use this when the user says:
      "move gym to 7:30"
      "shift wake up to 7am tomorrow"
      "reschedule leetcode to 6pm today"
      "gym later at 7:30"
    This is the preferred tool for time changes — use it instead of update_day_event or update_schedule_activity
    when you don't have an event_id.
    """
    import pytz
    from chanakya.db.mongo import daily_events as de_col

    if not re.match(r"^\d{2}:\d{2}$", new_time.strip()):
        return f"Error: new_time must be HH:MM. Got {new_time!r}."

    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    user_doc = users.find_one({"_id": uid})
    if not user_doc:
        return "Error: user not found."

    tz_str = user_doc.get("timezone", "Asia/Kolkata")
    try:
        tz = pytz.timezone(tz_str)
    except Exception:
        tz = pytz.timezone("Asia/Kolkata")

    now_local = datetime.now(tz)
    date_lower = date.strip().lower()
    if date_lower in ("today", ""):
        target_date_str = now_local.strftime("%Y-%m-%d")
    elif date_lower == "tomorrow":
        target_date_str = (now_local + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        try:
            from datetime import datetime as _dt
            _dt.strptime(date.strip(), "%Y-%m-%d")
            target_date_str = date.strip()
        except ValueError:
            return f"Error: date must be 'today', 'tomorrow', or YYYY-MM-DD. Got {date!r}."

    activity_clean = activity.strip().upper().replace(" ", "_")
    activity_lower = activity.strip().lower()

    # 1. Check date-specific events first
    date_evts = list(de_col.find({"user_id": uid, "date": target_date_str, "active": True}))
    for evt in date_evts:
        evt_activity = (evt.get("display_name") or evt.get("activity") or "").strip()
        if (evt_activity.upper().replace(" ", "_") == activity_clean or
                evt_activity.lower() == activity_lower):
            old_time = evt.get("time")
            de_col.update_one({"_id": evt["_id"]}, {"$set": {"time": new_time.strip(), "updated_at": datetime.utcnow()}})
            result = f"Moved '{activity}' from {old_time} to {new_time} on {target_date_str}."
            try:
                from chanakya.scheduler.checkpoint_runner import unsync_checkpoint, sync_event
                unsync_checkpoint(evt["_id"])
                updated_evt = de_col.find_one({"_id": evt["_id"]})
                if updated_evt:
                    sync_event(user_doc, updated_evt)
            except Exception:
                pass
            _write_audit(uid, "reschedule_activity", {"activity": activity, "new_time": new_time, "date": target_date_str}, result)
            return result

    # 2. Check base checkpoints
    all_cps = list(checkpoints.find({"user_id": uid, "active": True}))
    for cp in all_cps:
        cp_activity = (cp.get("display_name") or cp.get("activity") or "").strip()
        if (cp_activity.upper().replace(" ", "_") == activity_clean or
                cp_activity.lower() == activity_lower):
            old_time = cp.get("time")
            # Remove any existing override for this checkpoint on this date first
            de_col.delete_many({
                "user_id": uid,
                "date": target_date_str,
                "override_checkpoint_id": cp["_id"],
            })
            # Create new override at the new time
            now_utc = datetime.utcnow()
            doc = {
                "user_id": uid,
                "date": target_date_str,
                "time": new_time.strip(),
                "activity": cp.get("activity", activity_clean),
                "display_name": cp.get("display_name", activity),
                "description": cp.get("description", ""),
                "action_type": cp.get("action_type", "TELEGRAM_TEXT"),
                "priority": cp.get("priority", "MEDIUM"),
                "override_checkpoint_id": cp["_id"],
                "source": "reschedule",
                "active": True,
                "fired": False,
                "created_at": now_utc,
                "updated_at": now_utc,
            }
            result_insert = de_col.insert_one(doc)
            doc["_id"] = result_insert.inserted_id
            result = f"Moved '{activity}' from {old_time} to {new_time} on {target_date_str}."
            try:
                from chanakya.scheduler.checkpoint_runner import unsync_checkpoint, sync_event
                # Remove any stale job for the overridden base checkpoint (for today)
                unsync_checkpoint(cp["_id"])
                sync_event(user_doc, doc)
            except Exception:
                pass
            _write_audit(uid, "reschedule_activity", {"activity": activity, "new_time": new_time, "date": target_date_str}, result)
            return result + f" | event_id={result_insert.inserted_id}"

    return f"No activity matching '{activity}' found in schedule for {target_date_str}. Use fetch_day_schedule to see exact names."


@tool
def cancel_scheduled_message(user_id: str, event_id: str) -> str:
    """Cancel a previously scheduled message by its event_id.

    Use when the user says:
      "cancel that reminder"
      "remove the scheduled message"
      "don't send that message at 9pm"
    The event_id is returned when schedule_message is called.
    """
    from chanakya.db.mongo import daily_events as de_col

    try:
        uid = ObjectId(user_id)
        evt_oid = ObjectId(event_id)
    except Exception as exc:
        return f"Error: invalid ID — {exc}"

    result_del = de_col.delete_one({
        "_id": evt_oid,
        "user_id": uid,
        "activity": "SCHEDULED_MESSAGE",
    })
    if result_del.deleted_count:
        result = f"Scheduled message {event_id} cancelled."
    else:
        result = f"No scheduled message found with id={event_id}."
    _write_audit(uid, "cancel_scheduled_message", {"event_id": event_id}, result)
    return result


# ---------------------------------------------------------------------------
# Convenience list for binding to AgentExecutor
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    escalate_punishment,
    modify_wakeup_time,
    activate_war_mode,
    deactivate_war_mode,
    add_daily_checkpoint,
    send_emergency_alert,
    update_morning_todo_time,
    fetch_schedule,
    update_schedule_activity,
    save_contact,
    list_contacts,
    delete_contact,
    place_proxy_call,
    set_user_phone,
    get_user_status,
    add_mindset_note,
    add_mindset_entry,
    get_mindset_notes,
    remove_mindset_note,
    clear_mindset_notes,
    set_morning_todo_time,
    reload_prompt_templates,
    call_user,
    fetch_day_schedule,
    add_day_event,
    update_day_event,
    delete_day_event,
    get_day_log,
    send_telegram_message,
    schedule_message,
    cancel_scheduled_message,
    reschedule_activity,
] + ALL_ACCOUNTABILITY_TOOLS + ALL_RITUAL_TOOLS + ALL_COUNCIL_TOOLS + ALL_GOAL_TOOLS + ALL_GOOGLE_TOOLS
