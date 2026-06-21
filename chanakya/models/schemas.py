"""
schemas.py — Pydantic models for all MongoDB collections.

Each model mirrors the corresponding MongoDB collection schema defined in design.md.
ObjectId fields are stored as bson.ObjectId; serialisation is handled via
model_config = ConfigDict(arbitrary_types_allowed=True).
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from bson import ObjectId
from pydantic import BaseModel, Field
from pydantic import ConfigDict

# ---------------------------------------------------------------------------
# Shared / nested models
# ---------------------------------------------------------------------------

ACTIVITY_SLOT = Literal[
    "OFFICE_WORK",
    "LEETCODE",
    "GYM",
    "COMMUTE",
    "MEAL",
    "FREE_TIME",
    "SLEEP",
    "STUDY",
    "GENERIC",
]

INTERACTION_TYPE = Literal[
    "CHECKPOINT",
    "CHECK_IN",
    "EOD",
    "ESCALATION",
    "MENTOR_TALK",
    "COMMAND_RESPONSE",
    "MORNING_TODO",
]

TONE = Literal["HARSH", "MENTOR", "NEUTRAL", "CELEBRATORY"]


class EmergencyContact(BaseModel):
    """Nested model for users.emergency_contact."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    phone: str  # E.164 format
    relationship: str


class RelationshipConfig(BaseModel):
    """Nested model for users.relationship_config."""

    partner_name: Optional[str] = None
    partner_drain_level: Optional[Literal["high", "medium", "low"]] = None
    boundary_required: Optional[bool] = None


class NextDayPlan(BaseModel):
    """Nested model for users.next_day_plan."""

    date: Optional[str] = None          # "YYYY-MM-DD"
    plan_text: Optional[str] = None
    confirmed: bool = False


class RecurringFailurePattern(BaseModel):
    """Nested model for users.recurring_failure_patterns entries."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pattern_description: str
    checkpoint_ids: List[ObjectId] = Field(default_factory=list)
    detected_at: datetime
    times_observed: int


# ---------------------------------------------------------------------------
# UserProfile
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    """Mirrors the `users` MongoDB collection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: Optional[ObjectId] = Field(default=None, alias="_id")
    telegram_id: str
    name: str
    phone: str
    elevenlabs_voice_id: str
    leetcode_username: Optional[str] = None
    emergency_contact: Optional[EmergencyContact] = None
    relationship_config: Optional[RelationshipConfig] = None
    current_mode: str = "NORMAL"  # "NORMAL" | "WAR_MODE" | "INJURED" | "AWAY"
    war_mode_expires: Optional[datetime] = None
    active: bool = True
    timezone: str = "Asia/Kolkata"
    eod_time: str = "21:00"
    morning_todo_time: Optional[str] = None
    morning_todo_fallback_count: int = 0
    checkin_window_start: str = "09:00"
    checkin_window_end: str = "21:00"
    checkin_min_per_day: int = 2
    checkin_max_per_day: int = 4
    current_activity: str = "FREE_TIME"
    activity_slot_updated_at: Optional[datetime] = None
    streak_count: int = 0
    longest_streak: int = 0
    failure_count_this_week: int = 0
    next_day_plan: Optional[NextDayPlan] = None
    recurring_failure_patterns: List[RecurringFailurePattern] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

class ResponseValidation(BaseModel):
    """Nested model for checkpoints.response_validation."""

    type: str  # "TEXT" | "IMAGE" | "VOICE" | "LEETCODE_SUBMISSION"
    expected_within_minutes: int


class FailurePunishment(BaseModel):
    """Nested model for checkpoints.failure_punishment."""

    type: str  # "WARN" | "TELEGRAM_ALERT" | "CALL_EMERGENCY_CONTACT" | "SMS_EMERGENCY_CONTACT"
    target: Optional[str] = None
    message: Optional[str] = None


class Checkpoint(BaseModel):
    """Mirrors the `checkpoints` MongoDB collection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: Optional[ObjectId] = Field(default=None, alias="_id")
    schedule_id: Optional[ObjectId] = None
    user_id: ObjectId
    time: str                           # "HH:MM" in user's local timezone
    action_type: str                    # "CALL" | "TELEGRAM_TEXT" | "TELEGRAM_VOICE" | "IMAGE_DEMAND" | "COMMAND"
    priority: str                       # "HIGH" | "MEDIUM" | "LOW" | "CRITICAL"
    prompt_template: Optional[str] = None
    requires_response: bool = False
    response_validation: Optional[ResponseValidation] = None
    success_action: Optional[str] = None
    failure_punishment: Optional[FailurePunishment] = None
    persistent_nudge: bool = False
    persistent_nudge_interval_minutes: int = 5
    nudge_window_minutes: int = 45
    active: bool = True
    last_triggered: Optional[datetime] = None
    created_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# InteractionLog
# ---------------------------------------------------------------------------

class AIEvaluation(BaseModel):
    """Nested model for interaction_logs.ai_evaluation."""

    verdict: Optional[str] = None       # "SUCCESS" | "FAILED" | "EXCUSED" | "WAR_MODE_OVERRIDE" | "SKIPPED" | "ABANDONED"
    confidence: Optional[float] = None  # 0.0–1.0
    reasoning: Optional[str] = None


class InteractionLog(BaseModel):
    """Mirrors the `interaction_logs` MongoDB collection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: Optional[ObjectId] = Field(default=None, alias="_id")
    user_id: ObjectId
    checkpoint_id: Optional[ObjectId] = None
    timestamp: datetime
    trigger_type: str                   # "SCHEDULED" | "MANUAL" | "REACTIVE"
    channel: str                        # "TELEGRAM" | "TWILIO_CALL" | "WHATSAPP"
    message_sent: str
    user_response: Optional[str] = None
    media_url: Optional[str] = None
    twilio_call_sid: Optional[str] = None
    ai_evaluation: Optional[AIEvaluation] = None
    punishment_applied: Optional[str] = None
    checkin_topic: Optional[str] = None
    created_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# PromptTemplate
# ---------------------------------------------------------------------------

class PromptTemplate(BaseModel):
    """Mirrors the `prompt_templates` MongoDB collection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: Optional[ObjectId] = Field(default=None, alias="_id")
    activity_slot: ACTIVITY_SLOT
    interaction_type: INTERACTION_TYPE
    tone: TONE
    template_text: str
    version: int = 1
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# AgentTask (Task Manager)
# ---------------------------------------------------------------------------

class AgentTask(BaseModel):
    """Mirrors the `agent_tasks` MongoDB collection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: Optional[ObjectId] = Field(default=None, alias="_id")
    user_id: ObjectId
    task_type: str                      # e.g., "PROXY_CALL"
    payload: dict                       # Specific args for the task
    status: str = "PENDING"             # "PENDING", "RUNNING", "COMPLETED", "FAILED"
    result: Optional[dict] = None       # Info about outcome
    error_message: Optional[str] = None
    retries_attempted: int = 0
    max_retries: int = 3
    last_attempted_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# UserStateSnapshot
# ---------------------------------------------------------------------------

class UserStateSnapshot(BaseModel):
    """Mirrors the `user_state_snapshots` MongoDB collection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: Optional[ObjectId] = Field(default=None, alias="_id")
    user_id: ObjectId
    date: str                           # "YYYY-MM-DD" in user's local timezone
    summary: str
    embeddings: Optional[List[float]] = None
    created_at: Optional[datetime] = None
