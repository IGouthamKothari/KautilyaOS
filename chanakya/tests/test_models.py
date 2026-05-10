"""
test_models.py — Unit tests for Pydantic data models.

Covers:
  - LLMDecision and ActionItem (llm_decision.py)
  - All nested and top-level models in schemas.py
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from bson import ObjectId
from pydantic import ValidationError

from chanakya.models.llm_decision import ActionItem, LLMDecision
from chanakya.models.schemas import (
    AIEvaluation,
    Checkpoint,
    EmergencyContact,
    FailurePunishment,
    InteractionLog,
    NextDayPlan,
    PromptTemplate,
    RecurringFailurePattern,
    RelationshipConfig,
    ResponseValidation,
    UserProfile,
    UserStateSnapshot,
)


# ---------------------------------------------------------------------------
# ActionItem tests
# ---------------------------------------------------------------------------


class TestActionItem:
    def test_valid_action_type(self):
        item = ActionItem(type="increment_streak")
        assert item.type == "increment_streak"

    def test_params_defaults_to_empty_dict(self):
        item = ActionItem(type="reset_streak")
        assert item.params == {}

    def test_params_can_be_set(self):
        item = ActionItem(type="update_longest_streak", params={"value": 10})
        assert item.params == {"value": 10}

    def test_invalid_action_type_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ActionItem(type="fly_to_moon")

    def test_all_valid_action_types_accepted(self):
        valid_types = [
            "increment_streak",
            "reset_streak",
            "update_longest_streak",
            "apply_punishment",
            "request_clarification",
            "update_interaction_log",
            "update_activity_slot",
            "store_next_day_plan",
            "confirm_next_day_plan",
            "call_tool",
            "send_telegram",
            "send_voice",
        ]
        for t in valid_types:
            item = ActionItem(type=t)
            assert item.type == t


# ---------------------------------------------------------------------------
# LLMDecision tests
# ---------------------------------------------------------------------------


class TestLLMDecision:
    def test_parse_valid_json(self):
        data = {
            "verdict": "SUCCESS",
            "actions": [{"type": "increment_streak", "params": {}}],
            "tone": "CELEBRATORY",
            "response_text": "Great job!",
            "reasoning": "User completed the task.",
            "streak_reset": False,
            "model_used": "anthropic/claude-3.5-sonnet",
        }
        decision = LLMDecision.model_validate(data)
        assert decision.verdict == "SUCCESS"
        assert len(decision.actions) == 1
        assert decision.actions[0].type == "increment_streak"
        assert decision.tone == "CELEBRATORY"
        assert decision.response_text == "Great job!"
        assert decision.model_used == "anthropic/claude-3.5-sonnet"

    def test_actions_defaults_to_empty_list(self):
        decision = LLMDecision()
        assert decision.actions == []

    def test_verdict_defaults_to_none(self):
        decision = LLMDecision()
        assert decision.verdict is None

    def test_tone_defaults_to_neutral(self):
        decision = LLMDecision()
        assert decision.tone == "NEUTRAL"

    def test_streak_reset_defaults_to_false(self):
        decision = LLMDecision()
        assert decision.streak_reset is False

    def test_unknown_action_type_raises_validation_error(self):
        data = {
            "actions": [{"type": "do_something_unknown", "params": {}}],
        }
        with pytest.raises(ValidationError):
            LLMDecision.model_validate(data)

    def test_multiple_actions_parsed_in_order(self):
        data = {
            "actions": [
                {"type": "reset_streak"},
                {"type": "send_telegram", "params": {"text": "You failed."}},
                {"type": "apply_punishment", "params": {"checkpoint_id": "abc", "punishment_type": "WARN"}},
            ]
        }
        decision = LLMDecision.model_validate(data)
        assert [a.type for a in decision.actions] == [
            "reset_streak",
            "send_telegram",
            "apply_punishment",
        ]

    def test_verdict_none_is_valid(self):
        decision = LLMDecision(verdict=None)
        assert decision.verdict is None

    def test_all_fields_set(self):
        decision = LLMDecision(
            verdict="FAILED",
            actions=[ActionItem(type="apply_punishment", params={"checkpoint_id": "x", "punishment_type": "WARN"})],
            tone="HARSH",
            response_text="You failed.",
            reasoning="No response received.",
            streak_reset=True,
            model_used="openai/gpt-4o",
        )
        assert decision.verdict == "FAILED"
        assert decision.streak_reset is True
        assert decision.model_used == "openai/gpt-4o"


# ---------------------------------------------------------------------------
# EmergencyContact tests
# ---------------------------------------------------------------------------


class TestEmergencyContact:
    def test_valid_emergency_contact(self):
        ec = EmergencyContact(name="Mom", phone="+911234567890", relationship="mother")
        assert ec.name == "Mom"
        assert ec.phone == "+911234567890"
        assert ec.relationship == "mother"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            EmergencyContact(name="Mom", phone="+911234567890")  # missing relationship


# ---------------------------------------------------------------------------
# RelationshipConfig tests
# ---------------------------------------------------------------------------


class TestRelationshipConfig:
    def test_all_optional_fields(self):
        rc = RelationshipConfig()
        assert rc.partner_name is None
        assert rc.partner_drain_level is None
        assert rc.boundary_required is None

    def test_valid_drain_level(self):
        rc = RelationshipConfig(partner_name="Alice", partner_drain_level="high", boundary_required=True)
        assert rc.partner_drain_level == "high"

    def test_invalid_drain_level_raises(self):
        with pytest.raises(ValidationError):
            RelationshipConfig(partner_drain_level="extreme")


# ---------------------------------------------------------------------------
# NextDayPlan tests
# ---------------------------------------------------------------------------


class TestNextDayPlan:
    def test_defaults(self):
        plan = NextDayPlan()
        assert plan.date is None
        assert plan.plan_text is None
        assert plan.confirmed is False

    def test_with_values(self):
        plan = NextDayPlan(date="2024-01-15", plan_text="Study 4 hours", confirmed=True)
        assert plan.confirmed is True


# ---------------------------------------------------------------------------
# RecurringFailurePattern tests
# ---------------------------------------------------------------------------


class TestRecurringFailurePattern:
    def test_valid_pattern(self):
        oid = ObjectId()
        pattern = RecurringFailurePattern(
            pattern_description="Misses gym on Mondays",
            checkpoint_ids=[oid],
            detected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            times_observed=3,
        )
        assert pattern.times_observed == 3
        assert pattern.checkpoint_ids[0] == oid

    def test_checkpoint_ids_defaults_to_empty(self):
        pattern = RecurringFailurePattern(
            pattern_description="Test",
            detected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            times_observed=1,
        )
        assert pattern.checkpoint_ids == []


# ---------------------------------------------------------------------------
# UserProfile tests
# ---------------------------------------------------------------------------


class TestUserProfile:
    def _minimal(self) -> dict:
        return {
            "telegram_id": "123456789",
            "name": "Arjun",
            "phone": "+911234567890",
            "elevenlabs_voice_id": "voice_abc",
        }

    def test_minimal_required_fields(self):
        user = UserProfile(**self._minimal())
        assert user.telegram_id == "123456789"
        assert user.streak_count == 0
        assert user.longest_streak == 0
        assert user.current_mode == "NORMAL"
        assert user.timezone == "Asia/Kolkata"
        assert user.active is True

    def test_defaults_applied(self):
        user = UserProfile(**self._minimal())
        assert user.failure_count_this_week == 0
        assert user.morning_todo_time is None
        assert user.morning_todo_fallback_count == 0
        assert user.checkin_window_start == "09:00"
        assert user.checkin_window_end == "21:00"
        assert user.checkin_min_per_day == 2
        assert user.checkin_max_per_day == 4
        assert user.current_activity == "FREE_TIME"
        assert user.activity_slot_updated_at is None
        assert user.next_day_plan is None
        assert user.recurring_failure_patterns == []

    def test_with_nested_emergency_contact(self):
        data = {
            **self._minimal(),
            "emergency_contact": {
                "name": "Dad",
                "phone": "+910987654321",
                "relationship": "father",
            },
        }
        user = UserProfile(**data)
        assert user.emergency_contact is not None
        assert user.emergency_contact.name == "Dad"

    def test_with_nested_relationship_config(self):
        data = {
            **self._minimal(),
            "relationship_config": {
                "partner_name": "Alex",
                "partner_drain_level": "medium",
                "boundary_required": False,
            },
        }
        user = UserProfile(**data)
        assert user.relationship_config is not None
        assert user.relationship_config.partner_name == "Alex"

    def test_with_next_day_plan(self):
        data = {
            **self._minimal(),
            "next_day_plan": {"date": "2024-01-16", "plan_text": "LeetCode 2h", "confirmed": True},
        }
        user = UserProfile(**data)
        assert user.next_day_plan is not None
        assert user.next_day_plan.confirmed is True

    def test_objectid_field(self):
        oid = ObjectId()
        data = {**self._minimal(), "_id": oid}
        user = UserProfile.model_validate(data)
        assert user.id == oid


# ---------------------------------------------------------------------------
# ResponseValidation and FailurePunishment tests
# ---------------------------------------------------------------------------


class TestResponseValidation:
    def test_valid(self):
        rv = ResponseValidation(type="TEXT", expected_within_minutes=30)
        assert rv.type == "TEXT"
        assert rv.expected_within_minutes == 30


class TestFailurePunishment:
    def test_valid(self):
        fp = FailurePunishment(type="WARN", target="user", message="You failed!")
        assert fp.type == "WARN"

    def test_optional_fields(self):
        fp = FailurePunishment(type="TELEGRAM_ALERT")
        assert fp.target is None
        assert fp.message is None


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_minimal_required_fields(self):
        oid = ObjectId()
        cp = Checkpoint(
            user_id=oid,
            time="07:00",
            action_type="TELEGRAM_TEXT",
            priority="HIGH",
        )
        assert cp.time == "07:00"
        assert cp.active is True
        assert cp.last_triggered is None

    def test_with_nested_models(self):
        oid = ObjectId()
        cp = Checkpoint(
            user_id=oid,
            time="08:00",
            action_type="CALL",
            priority="CRITICAL",
            response_validation={"type": "VOICE", "expected_within_minutes": 5},
            failure_punishment={"type": "CALL_EMERGENCY_CONTACT", "target": "emergency", "message": "Failed!"},
        )
        assert cp.response_validation is not None
        assert cp.response_validation.type == "VOICE"
        assert cp.failure_punishment is not None
        assert cp.failure_punishment.type == "CALL_EMERGENCY_CONTACT"


# ---------------------------------------------------------------------------
# AIEvaluation tests
# ---------------------------------------------------------------------------


class TestAIEvaluation:
    def test_all_optional(self):
        ev = AIEvaluation()
        assert ev.verdict is None
        assert ev.confidence is None
        assert ev.reasoning is None

    def test_with_values(self):
        ev = AIEvaluation(verdict="SUCCESS", confidence=0.95, reasoning="Task completed.")
        assert ev.verdict == "SUCCESS"
        assert ev.confidence == 0.95


# ---------------------------------------------------------------------------
# InteractionLog tests
# ---------------------------------------------------------------------------


class TestInteractionLog:
    def test_minimal_required_fields(self):
        oid = ObjectId()
        log = InteractionLog(
            user_id=oid,
            timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
            trigger_type="SCHEDULED",
            channel="TELEGRAM",
            message_sent="Did you complete your workout?",
        )
        assert log.trigger_type == "SCHEDULED"
        assert log.user_response is None
        assert log.ai_evaluation is None

    def test_with_ai_evaluation(self):
        oid = ObjectId()
        log = InteractionLog(
            user_id=oid,
            timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
            trigger_type="MANUAL",
            channel="TELEGRAM",
            message_sent="Check in",
            ai_evaluation={"verdict": "FAILED", "confidence": 0.8, "reasoning": "No response."},
        )
        assert log.ai_evaluation is not None
        assert log.ai_evaluation.verdict == "FAILED"


# ---------------------------------------------------------------------------
# PromptTemplate tests
# ---------------------------------------------------------------------------


class TestPromptTemplate:
    def test_valid_prompt_template(self):
        pt = PromptTemplate(
            activity_slot="GYM",
            interaction_type="CHECKPOINT",
            tone="HARSH",
            template_text="Did you go to the gym today, {name}?",
        )
        assert pt.activity_slot == "GYM"
        assert pt.tone == "HARSH"
        assert pt.version == 1

    def test_invalid_activity_slot_raises(self):
        with pytest.raises(ValidationError):
            PromptTemplate(
                activity_slot="INVALID_SLOT",
                interaction_type="CHECKPOINT",
                tone="NEUTRAL",
                template_text="Hello",
            )

    def test_invalid_interaction_type_raises(self):
        with pytest.raises(ValidationError):
            PromptTemplate(
                activity_slot="GYM",
                interaction_type="UNKNOWN_TYPE",
                tone="NEUTRAL",
                template_text="Hello",
            )

    def test_invalid_tone_raises(self):
        with pytest.raises(ValidationError):
            PromptTemplate(
                activity_slot="GYM",
                interaction_type="CHECKPOINT",
                tone="ANGRY",
                template_text="Hello",
            )

    def test_all_valid_activity_slots(self):
        valid_slots = [
            "OFFICE_WORK", "LEETCODE", "GYM", "COMMUTE", "MEAL",
            "FREE_TIME", "SLEEP", "STUDY", "GENERIC",
        ]
        for slot in valid_slots:
            pt = PromptTemplate(
                activity_slot=slot,
                interaction_type="CHECK_IN",
                tone="NEUTRAL",
                template_text="Hello {name}",
            )
            assert pt.activity_slot == slot

    def test_all_valid_interaction_types(self):
        valid_types = [
            "CHECKPOINT", "CHECK_IN", "EOD", "ESCALATION",
            "MENTOR_TALK", "COMMAND_RESPONSE", "MORNING_TODO",
        ]
        for itype in valid_types:
            pt = PromptTemplate(
                activity_slot="FREE_TIME",
                interaction_type=itype,
                tone="MENTOR",
                template_text="Hello",
            )
            assert pt.interaction_type == itype

    def test_all_valid_tones(self):
        for tone in ["HARSH", "MENTOR", "NEUTRAL", "CELEBRATORY"]:
            pt = PromptTemplate(
                activity_slot="FREE_TIME",
                interaction_type="CHECK_IN",
                tone=tone,
                template_text="Hello",
            )
            assert pt.tone == tone


# ---------------------------------------------------------------------------
# UserStateSnapshot tests
# ---------------------------------------------------------------------------


class TestUserStateSnapshot:
    def test_minimal_required_fields(self):
        oid = ObjectId()
        snap = UserStateSnapshot(
            user_id=oid,
            date="2024-01-15",
            summary="Good day overall. Completed gym and LeetCode.",
        )
        assert snap.date == "2024-01-15"
        assert snap.embeddings is None

    def test_with_embeddings(self):
        oid = ObjectId()
        snap = UserStateSnapshot(
            user_id=oid,
            date="2024-01-15",
            summary="Productive day.",
            embeddings=[0.1, 0.2, 0.3, 0.4],
        )
        assert snap.embeddings is not None
        assert len(snap.embeddings) == 4
