"""
goals_api.py — REST API for goal management.

CRUD endpoints for GOAP-inspired goal tracking with milestones.
"""

from __future__ import annotations

from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chanakya.db.mongo import (
    abandon_goal,
    create_goal,
    get_goal_by_id,
    get_goals,
    goals,
    update_goal_progress,
    users,
)

router = APIRouter(prefix="/api/goals", tags=["goals"])


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class MilestoneInput(BaseModel):
    title: str


class GoalInput(BaseModel):
    user_id: str
    title: str
    description: str = ""
    category: str = "general"
    target_date: Optional[str] = None
    milestones: list[MilestoneInput] = Field(default_factory=list)


class GoalUpdateInput(BaseModel):
    progress: Optional[int] = None
    note: Optional[str] = None
    milestone_index: Optional[int] = None


class GoalAbandonInput(BaseModel):
    reason: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _resolve_user(user_id: str) -> ObjectId:
    """Resolve user_id string to ObjectId, raise 404 if not found."""
    try:
        uid = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user_id format")
    user = users.find_one({"_id": uid})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return uid


@router.post("/")
async def create_goal_endpoint(body: GoalInput) -> dict:
    """Create a new goal with optional milestones."""
    uid = _resolve_user(body.user_id)

    milestone_list = [{"title": m.title, "done": False} for m in body.milestones]

    goal_id = create_goal(
        user_id=uid,
        title=body.title,
        description=body.description,
        category=body.category,
        target_date=body.target_date,
        milestones=milestone_list,
    )

    return {"id": goal_id, "title": body.title, "status": "active"}


@router.get("/")
async def list_goals_endpoint(user_id: str, status: str = "active") -> list[dict]:
    """List goals for a user, filtered by status."""
    uid = _resolve_user(user_id)
    filter_status = status if status != "all" else None
    return get_goals(uid, status=filter_status)


@router.get("/{goal_id}")
async def get_goal_endpoint(goal_id: str, user_id: str) -> dict:
    """Get a specific goal by ID."""
    uid = _resolve_user(user_id)
    goal = get_goal_by_id(uid, goal_id)
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    return goal


@router.patch("/{goal_id}")
async def update_goal_endpoint(goal_id: str, user_id: str, body: GoalUpdateInput) -> dict:
    """Update progress, add notes, or mark milestones on a goal."""
    uid = _resolve_user(user_id)

    goal = get_goal_by_id(uid, goal_id)
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")

    success = update_goal_progress(
        user_id=uid,
        goal_id=goal_id,
        progress=body.progress,
        note=body.note,
        milestone_index=body.milestone_index,
    )

    if not success:
        raise HTTPException(status_code=400, detail="Failed to update goal")

    return get_goal_by_id(uid, goal_id)


@router.delete("/{goal_id}")
async def abandon_goal_endpoint(goal_id: str, user_id: str, body: GoalAbandonInput = GoalAbandonInput()) -> dict:
    """Abandon a goal."""
    uid = _resolve_user(user_id)

    goal = get_goal_by_id(uid, goal_id)
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")

    success = abandon_goal(uid, goal_id, body.reason)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to abandon goal")

    return {"id": goal_id, "status": "abandoned", "reason": body.reason}
