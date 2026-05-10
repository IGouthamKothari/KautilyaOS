from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Any
from bson import ObjectId
from langchain_core.tools import tool
from chanakya.db.mongo import users, interaction_logs, agent_tasks

logger = logging.getLogger(__name__)

def _write_audit(user_id: ObjectId, tool_name: str, tool_input: dict, tool_output: str) -> None:
    """Fire-and-forget audit log."""
    try:
        from chanakya.db.mongo import ai_tool_calls
        ai_tool_calls.insert_one({
            "user_id": user_id,
            "timestamp": datetime.utcnow(),
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": tool_output,
            "created_at": datetime.utcnow(),
        })
    except Exception as exc:
        logger.warning(f"Audit log failed for {tool_name}: {exc}")

@tool
def set_commitment(user_id: str, task_name: str, duration_minutes: int, penalty_description: str = "") -> str:
    """Set a hard commitment for a specific task and duration.
    
    Chanakya will track this and trigger a follow-up call/message when the duration ends.
    If failed, the penalty will be recorded in the user's ledger.
    
    Use when the user says:
      "I commit to 1 hour of gym"
      "I will finish this report in 45 mins"
      "Set a commitment for my LeetCode session"
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    user_doc = users.find_one({"_id": uid})
    if not user_doc:
        return "Error: user not found."

    # 1. Store the commitment on the user document for current state
    now = datetime.utcnow()
    deadline = now + timedelta(minutes=duration_minutes)
    
    commitment = {
        "task_name": task_name,
        "started_at": now,
        "deadline": deadline,
        "duration_minutes": duration_minutes,
        "penalty": penalty_description,
        "status": "ACTIVE"
    }

    users.update_one(
        {"_id": uid},
        {"$set": {"current_commitment": commitment}}
    )

    # 2. Schedule a follow-up task in the Task Manager
    task_doc = {
        "user_id": uid,
        "task_type": "COMMITMENT_CHECK",
        "payload": {
            "task_name": task_name,
            "deadline": deadline,
        },
        "status": "PENDING",
        "run_at": deadline, # Future execution
        "retries_attempted": 0,
        "max_retries": 1,
        "created_at": now
    }
    
    agent_tasks.insert_one(task_doc)

    result = (
        f"Commitment locked: **{task_name}** for {duration_minutes} minutes. "
        f"I will check in at {deadline.strftime('%H:%M UTC')}. "
        f"Dharma demands completion."
    )
    _write_audit(uid, "set_commitment", {"task": task_name, "duration": duration_minutes}, result)
    return result

@tool
def update_warrior_streak(user_id: str, change: int, reason: str) -> str:
    """Manually update the user's Warrior Streak (consecutive days of discipline).
    
    Positive change rewards discipline. Negative change (e.g. -100) resets or penalises for broken dharma.
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    if change >= 0:
        users.update_one({"_id": uid}, {"$inc": {"warrior_streak": change}})
        result = f"Warrior Streak increased by {change}. Reason: {reason}"
    else:
        # Reset or heavy penalty
        users.update_one({"_id": uid}, {"$set": {"warrior_streak": 0}})
        result = f"Warrior Streak RESET to zero. Reason: {reason}. Rise again, warrior."

    _write_audit(uid, "update_warrior_streak", {"change": change, "reason": reason}, result)
    return result

@tool
def get_accountability_ledger(user_id: str) -> str:
    """Return the user's current accountability metrics (Warrior Streak, Commitment Ledger)."""
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    user_doc = users.find_one({"_id": uid})
    if not user_doc:
        return "Error: user not found."

    streak = user_doc.get("warrior_streak", 0)
    commitment = user_doc.get("current_commitment")
    
    ledger_text = f"**Warrior Streak**: {streak} days\n"
    if commitment and commitment.get("status") == "ACTIVE":
        ledger_text += f"**Active Commitment**: {commitment['task_name']} (Deadline: {commitment['deadline'].strftime('%H:%M UTC')})"
    else:
        ledger_text += "**Active Commitment**: None"

    return ledger_text

@tool
def complete_commitment(user_id: str, success: bool, note: str = "") -> str:
    """Mark the current active commitment as completed or failed.
    
    If success=True: Increases Warrior Streak.
    If success=False: Resets Warrior Streak and logs the failure.
    
    Use when the user confirms if they finished their committed task.
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    user_doc = users.find_one({"_id": uid})
    if not user_doc or "current_commitment" not in user_doc:
        return "No active commitment found to complete."

    commitment = user_doc["current_commitment"]
    commitment["status"] = "SUCCESS" if success else "FAILED"
    commitment["completed_at"] = datetime.utcnow()
    commitment["completion_note"] = note

    if success:
        users.update_one(
            {"_id": uid},
            {
                "$set": {"current_commitment": commitment},
                "$inc": {"warrior_streak": 1}
            }
        )
        result = f"Dharma fulfilled. Commitment '{commitment['task_name']}' succeeded. Warrior Streak increased."
    else:
        users.update_one(
            {"_id": uid},
            {
                "$set": {"current_commitment": commitment, "warrior_streak": 0}
            }
        )
        result = f"Dharma broken. Commitment '{commitment['task_name']}' failed. Warrior Streak RESET to zero."

    _write_audit(uid, "complete_commitment", {"success": success, "note": note}, result)
    return result

@tool
def get_financial_ledger(user_id: str) -> str:
    """Return the current balance and recent penalty history from the financial ledger."""
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    user_doc = users.find_one({"_id": uid})
    if not user_doc:
        return "User not found."

    ledger = user_doc.get("accountability_ledger", {"balance": 0, "history": []})
    currency = user_doc.get("currency", "INR")
    
    balance = ledger.get("balance", 0)
    history = ledger.get("history", [])[-5:] # Last 5
    
    lines = [f"--- Financial Accountability Ledger ---"]
    lines.append(f"Current Debt/Penalty Balance: {balance} {currency}")
    
    if history:
        lines.append("\nRecent Penalties:")
        for entry in history:
            ts = entry.get("at", datetime.utcnow()).strftime("%Y-%m-%d %H:%M")
            lines.append(f"• {ts}: {entry['amount']} {currency} ({entry['reason']})")
    else:
        lines.append("\nNo penalties recorded yet. Dharma is strong.")
        
    return "\n".join(lines)

@tool
def apply_financial_penalty(user_id: str, amount: int, reason: str) -> str:
    """Apply a financial penalty to the user's virtual ledger for breaking dharma.
    
    Use this when the user fails a commitment or has their streak reset.
    Default amounts:
    - Missed Checkpoint: 100 INR
    - Failed Commitment: 500 INR
    - Streak Reset: 1000 INR
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    user_doc = users.find_one({"_id": uid})
    if not user_doc:
        return "User not found."

    currency = user_doc.get("currency", "INR")
    now = datetime.utcnow()
    
    entry = {
        "at": now,
        "amount": amount,
        "reason": reason
    }
    
    users.update_one(
        {"_id": uid},
        {
            "$inc": {"accountability_ledger.balance": amount},
            "$push": {"accountability_ledger.history": entry}
        }
    )
    
    result = f"Penalty Applied: {amount} {currency} added to ledger. Reason: {reason}."
    _write_audit(uid, "apply_financial_penalty", {"amount": amount, "reason": reason}, result)
    return result

ALL_ACCOUNTABILITY_TOOLS = [
    set_commitment, 
    update_warrior_streak, 
    get_accountability_ledger, 
    complete_commitment,
    get_financial_ledger,
    apply_financial_penalty
]
