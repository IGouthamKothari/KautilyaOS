"""
load_schedule.py — Load goutham_base.json into MongoDB checkpoints collection.

Run this script AFTER reviewing the schedule preview via /schedule in Telegram.
Or call write_schedule_to_db() directly from the bot's /confirmschedule handler.

Usage (manual):
    python chanakya/scripts/load_schedule.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

# Map schedule checkpoint_type → DB action_type
_CHECKPOINT_TYPE_MAP = {
    "CALL":          "CALL",
    "TELEGRAM_TEXT": "TELEGRAM_TEXT",
    "TELEGRAM_VOICE":"TELEGRAM_VOICE",
    "IMAGE_DEMAND":  "IMAGE_DEMAND",
}

# Map schedule priority → DB priority
_PRIORITY_MAP = {
    "CRITICAL": "CRITICAL",
    "HIGH":     "HIGH",
    "MEDIUM":   "MEDIUM",
    "LOW":      "LOW",
}


def _load_base_schedule() -> dict:
    """Load goutham_base.json from config/schedules/."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "config", "schedules", "goutham_base.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_day(schedule_data: dict, day: str) -> list[dict]:
    """Resolve a day's schedule, following COPY_MONDAY references."""
    day_data = schedule_data["base_schedule"].get(day, [])
    if day_data == "COPY_MONDAY":
        day_data = schedule_data["base_schedule"]["monday"]
    return day_data if isinstance(day_data, list) else []


def format_schedule_preview(schedule_data: dict) -> str:
    """Return a human-readable HTML preview of the schedule for Telegram."""
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    lines = [
        f"📅 <b>Schedule Preview</b> (v{schedule_data.get('version', '?')})",
        f"🕐 Timezone: {schedule_data.get('timezone', 'Asia/Kolkata')}\n",
    ]

    for day in days:
        raw = schedule_data["base_schedule"].get(day)
        if raw == "COPY_MONDAY":
            lines.append(f"<b>{day.capitalize()}</b>: same as Monday")
            continue

        activities = _resolve_day(schedule_data, day)
        if not activities:
            continue

        lines.append(f"\n<b>{day.capitalize()}</b>")
        for act in activities:
            cp_type = act.get("checkpoint_type")
            icon = {
                "CALL": "📞",
                "TELEGRAM_TEXT": "💬",
                "TELEGRAM_VOICE": "🔊",
                "IMAGE_DEMAND": "📷",
            }.get(cp_type or "", "  ")
            label = act.get("display_name") or act["activity"].replace("_", " ").title()
            cp_label = f" [{cp_type}]" if cp_type else ""
            lines.append(
                f"  {icon} <code>{act['time']}</code> — {label}{cp_label} "
                f"({act.get('priority','?')}, {act.get('duration_min','?')}min)"
            )
            desc = act.get("description") or act.get("notes") or ""
            if desc:
                lines.append(f"       <i>{desc}</i>")

    goals = schedule_data.get("weekly_goals", {})
    lines.append("\n<b>Weekly Goals</b>")
    for k, v in goals.items():
        lines.append(f"  • {k.replace('_', ' ')}: {v}")

    lines.append(
        "\n\nSend /confirmschedule to write this to the database.\n"
        "Edit <code>config/schedules/goutham_base.json</code> first if anything looks wrong."
    )
    return "\n".join(lines)


def write_schedule_to_db(user_id, dry_run: bool = False) -> tuple[int, int]:
    """Write the base schedule checkpoints to MongoDB for the given user.

    Skips activities with no checkpoint_type (they are informational only).
    Upserts by (user_id, time, activity, days) so re-running is safe.
    Each checkpoint stores the list of days it applies to.

    Args:
        user_id: MongoDB ObjectId of the user.
        dry_run: If True, returns counts without writing anything.

    Returns:
        (inserted, updated) counts.
    """
    from chanakya.db.mongo import checkpoints

    schedule_data = _load_base_schedule()
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    inserted = 0
    updated = 0
    now = datetime.utcnow()

    # Build a map: (time, activity) → set of days
    slot_days: dict[tuple, set] = {}
    slot_meta: dict[tuple, dict] = {}

    for day in days:
        activities = _resolve_day(schedule_data, day)
        for act in activities:
            cp_type = act.get("checkpoint_type")
            if not cp_type:
                continue
            key = (act["time"], act["activity"])
            slot_days.setdefault(key, set()).add(day)
            slot_meta[key] = act  # last write wins (same across COPY_MONDAY days)

    for key, day_set in slot_days.items():
        act = slot_meta[key]
        cp_type = act.get("checkpoint_type")
        action_type = _CHECKPOINT_TYPE_MAP.get(cp_type, "TELEGRAM_TEXT")
        priority = _PRIORITY_MAP.get(act.get("priority", "MEDIUM"), "MEDIUM")
        sorted_days = sorted(day_set, key=lambda d: days.index(d))

        doc = {
            "user_id": user_id,
            "time": act["time"],
            "activity": act["activity"],
            "display_name": act.get("display_name") or act["activity"].replace("_", " ").title(),
            "description": act.get("description") or act.get("notes") or "",
            "action_type": action_type,
            "priority": priority,
            "days": sorted_days,
            "prompt_template": act.get("notes", act["activity"]),
            "active": True,
            "failure_punishment": {"type": "WARN"},
            "last_triggered": None,
            "created_at": now,
            "updated_at": now,
            "source": "goutham_base.json",
            "source_version": schedule_data.get("version", "1.0"),
        }

        if dry_run:
            inserted += 1
            continue

        set_fields = {k: v for k, v in doc.items() if k != "created_at"}
        set_fields["updated_at"] = now

        result = checkpoints.update_one(
            {"user_id": user_id, "time": act["time"], "activity": act["activity"]},
            {
                "$set": set_fields,
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )
        if result.upserted_id:
            inserted += 1
        else:
            updated += 1

    logger.info(
        "Schedule written to DB for user %s: %d inserted, %d updated (dry_run=%s)",
        user_id,
        inserted,
        updated,
        dry_run,
    )
    return inserted, updated


if __name__ == "__main__":
    # Manual run — find the first active user and write schedule
    from chanakya.db.mongo import users

    user = users.find_one({"active": True})
    if not user:
        print("No active user found.")
        sys.exit(1)

    schedule_data = _load_base_schedule()
    print(format_schedule_preview(schedule_data))
    print("\n--- Writing to DB ---")
    ins, upd = write_schedule_to_db(user["_id"])
    print(f"Done: {ins} inserted, {upd} updated.")
