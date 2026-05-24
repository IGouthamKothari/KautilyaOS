"""
goal_tools.py — GOAP-inspired goal tracking tools.

Allows Chanakya and specialists to create, track, and update goals
with milestones and progress monitoring.
"""

from __future__ import annotations

import logging
from datetime import datetime

from bson import ObjectId
from langchain_core.tools import tool

from chanakya.db.mongo import users, goals

logger = logging.getLogger(__name__)


def _write_audit(user_id: ObjectId, tool_name: str, tool_input: dict, tool_output: str) -> None:
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
def set_goal(
    user_id: str,
    title: str,
    description: str = "",
    category: str = "general",
    target_date: str = "",
    milestones: str = "",
) -> str:
    """Create a new goal with optional milestones and target date.

    Use when the user declares a goal:
      "I want to solve 100 leetcode problems by July"
      "My goal is to save 5 lakhs this year"
      "I want to run a half marathon by December"

    Args:
        user_id: The user's ID
        title: Short goal title (e.g. "100 LeetCode problems")
        description: Why this goal matters
        category: general|fitness|finance|career|learning|health
        target_date: Optional deadline (YYYY-MM-DD format)
        milestones: Comma-separated milestone titles (e.g. "25 problems,50 problems,75 problems")
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    user_doc = users.find_one({"_id": uid})
    if not user_doc:
        return "Error: user not found."

    milestone_list = []
    if milestones:
        for m in milestones.split(","):
            m = m.strip()
            if m:
                milestone_list.append({"title": m, "done": False})

    from chanakya.db.mongo import create_goal
    goal_id = create_goal(
        user_id=uid,
        title=title,
        description=description,
        category=category,
        target_date=target_date or None,
        milestones=milestone_list,
    )

    _write_audit(uid, "set_goal", {"title": title, "category": category}, goal_id)

    result = f"Goal created: \"{title}\""
    if milestone_list:
        result += f" with {len(milestone_list)} milestones"
    if target_date:
        result += f" (deadline: {target_date})"
    return result


@tool
def update_goal(
    user_id: str,
    goal_id: str,
    progress: int = -1,
    note: str = "",
    milestone_index: int = -1,
) -> str:
    """Update progress on a goal, add a note, or mark a milestone complete.

    Use when:
      "I finished 10 more leetcode problems" → update progress
      "Mark the first milestone done" → milestone_index=0
      "Add a note to my fitness goal" → add note

    Args:
        user_id: The user's ID
        goal_id: The goal's ID (from list_goals)
        progress: New progress percentage (0-100). Pass -1 to skip.
        note: Optional note to add to the goal's history
        milestone_index: Index of milestone to mark complete (0-based). Pass -1 to skip.
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    from chanakya.db.mongo import update_goal_progress, get_goal_by_id

    goal = get_goal_by_id(uid, goal_id)
    if not goal:
        return f"Error: goal {goal_id!r} not found."

    success = update_goal_progress(
        user_id=uid,
        goal_id=goal_id,
        progress=progress if progress >= 0 else None,
        note=note or None,
        milestone_index=milestone_index if milestone_index >= 0 else None,
    )

    if not success:
        return "Error: failed to update goal."

    _write_audit(uid, "update_goal", {"goal_id": goal_id, "progress": progress}, "updated")

    parts = [f"Goal \"{goal['title']}\" updated."]
    if progress >= 0:
        parts.append(f"Progress: {progress}%")
        if progress >= 100:
            parts.append("GOAL COMPLETED.")
    if note:
        parts.append(f"Note added.")
    if milestone_index >= 0 and milestone_index < len(goal.get("milestones", [])):
        ms = goal["milestones"][milestone_index]
        parts.append(f"Milestone \"{ms['title']}\" marked done.")

    return " ".join(parts)


@tool
def list_goals(user_id: str, status: str = "active") -> str:
    """List all goals for the user, optionally filtered by status.

    Use when:
      "What are my goals?"
      "Show my active goals"
      "List completed goals"

    Args:
        user_id: The user's ID
        status: Filter by status: active|completed|abandoned|all
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    from chanakya.db.mongo import get_goals

    filter_status = status if status != "all" else None
    user_goals = get_goals(uid, status=filter_status)

    if not user_goals:
        return f"No {status} goals found."

    lines = []
    for g in user_goals:
        milestones_done = sum(1 for m in g.get("milestones", []) if m.get("done"))
        milestones_total = len(g.get("milestones", []))
        line = f"• [{g['_id']}] {g['title']} — {g['progress']}%"
        if g.get("target_date"):
            line += f" (by {g['target_date']})"
        if milestones_total:
            line += f" [{milestones_done}/{milestones_total} milestones]"
        line += f" ({g['status']})"
        lines.append(line)

    return "\n".join(lines)


@tool
def abandon_goal_tool(user_id: str, goal_id: str, reason: str = "") -> str:
    """Abandon a goal that is no longer relevant.

    Use when the user explicitly gives up on a goal or it's no longer applicable.
    The guru should challenge this decision before executing.

    Args:
        user_id: The user's ID
        goal_id: The goal's ID
        reason: Why the goal is being abandoned
    """
    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: invalid user_id {user_id!r}."

    from chanakya.db.mongo import abandon_goal, get_goal_by_id

    goal = get_goal_by_id(uid, goal_id)
    if not goal:
        return f"Error: goal {goal_id!r} not found."

    success = abandon_goal(uid, goal_id, reason)
    if not success:
        return "Error: failed to abandon goal."

    _write_audit(uid, "abandon_goal", {"goal_id": goal_id, "reason": reason}, "abandoned")
    return f"Goal \"{goal['title']}\" marked as abandoned. Reason: {reason or 'none given'}"


ALL_GOAL_TOOLS = [set_goal, update_goal, list_goals, abandon_goal_tool]
