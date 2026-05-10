from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Any
from bson import ObjectId
from langchain_core.tools import tool
from chanakya.db.mongo import users, rituals

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
def log_ritual(user_id: str, category: str, value: float | int | str, note: str = "") -> str:
    """Log a health or daily ritual (Sleep, Mood, Energy, Water, etc.).
    
    Categories:
    - SLEEP (hours)
    - MOOD (1-10)
    - ENERGY (1-10)
    - WATER (liters)
    - MEDITATION (minutes)
    
    Use when the user shares health data or during Morning/EOD check-ins.
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    now = datetime.utcnow()
    ritual_doc = {
        "user_id": uid,
        "category": category.upper(),
        "value": value,
        "note": note,
        "timestamp": now,
        "created_at": now
    }

    try:
        rituals.insert_one(ritual_doc)
        
        # Update user profile's last recorded values for quick access
        users.update_one(
            {"_id": uid},
            {"$set": {f"last_ritual.{category.lower()}": {"value": value, "at": now}}}
        )
        
        result = f"Ritual recorded: {category.upper()} = {value}. Note: {note if note else 'None'}"
        _write_audit(uid, "log_ritual", {"category": category, "value": value}, result)
        return result
    except Exception as exc:
        return f"Failed to log ritual: {exc}"

@tool
def get_ritual_summary(user_id: str, days: int = 7) -> str:
    """Return a summary of the user's rituals over the last X days."""
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    cutoff = datetime.utcnow() - timedelta(days=days)
    recent_rituals = list(rituals.find({
        "user_id": uid,
        "timestamp": {"$gte": cutoff}
    }).sort("timestamp", -1))

    if not recent_rituals:
        return f"No ritual logs found in the last {days} days."

    # Group by category
    summary = {}
    for r in recent_rituals:
        cat = r["category"]
        if cat not in summary:
            summary[cat] = []
        summary[cat].append(r)

    lines = [f"--- Ritual Summary (Last {days} days) ---"]
    for cat, logs in summary.items():
        # Calculate average if numeric
        numeric_values = [l["value"] for l in logs if isinstance(l["value"], (int, float))]
        if numeric_values:
            avg = sum(numeric_values) / len(numeric_values)
            lines.append(f"• {cat}: Avg {avg:.1f} (Based on {len(logs)} entries)")
        else:
            lines.append(f"• {cat}: {len(logs)} entries")
            
    return "\n".join(lines)

ALL_RITUAL_TOOLS = [log_ritual, get_ritual_summary]
