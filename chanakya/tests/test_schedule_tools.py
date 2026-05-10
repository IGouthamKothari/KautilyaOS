"""
test_schedule_tools.py — Unit and property-based tests for schedule_tools.py.

Uses mongomock to avoid real MongoDB connections.
Mocks TwilioClient for send_emergency_alert tests.

**Validates: Requirements 5, 6, 7, 8, 9, 21**
"""

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import mongomock
import pytest
from bson import ObjectId
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Set up mongomock collections that will be injected into schedule_tools
# ---------------------------------------------------------------------------

_mock_client = mongomock.MongoClient()
_mock_db = _mock_client["chanakya"]

_mock_checkpoints = _mock_db["checkpoints"]
_mock_users = _mock_db["users"]
_mock_ai_tool_calls = _mock_db["ai_tool_calls"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user(**kwargs) -> ObjectId:
    """Insert a minimal user document and return its _id."""
    doc = {
        "name": "Test User",
        "phone": "+911234567890",
        "current_mode": "NORMAL",
        **kwargs,
    }
    result = _mock_users.insert_one(doc)
    return result.inserted_id


def _make_checkpoint(user_id: ObjectId, **kwargs) -> ObjectId:
    """Insert a minimal checkpoint document and return its _id."""
    doc = {
        "user_id": user_id,
        "time": "06:00",
        "action_type": "TELEGRAM_TEXT",
        "priority": "HIGH",
        "active": True,
        "failure_punishment": {"type": "WARN"},
        **kwargs,
    }
    result = _mock_checkpoints.insert_one(doc)
    return result.inserted_id


def _repatch():
    """Re-point schedule_tools module-level collections to the mock DB."""
    # Patch the __globals__ of the tool functions directly — this is the dict
    # the function body reads 'checkpoints' and 'users' from at call time.
    # This works even when sys.modules["chanakya.tools.schedule_tools"] has been
    # deleted by other test files.
    for fn in [escalate_punishment, modify_wakeup_time, activate_war_mode,
               add_daily_checkpoint, send_emergency_alert, update_morning_todo_time]:
        # LangChain @tool wraps the function; get the underlying func
        underlying = getattr(fn, "func", fn)
        if hasattr(underlying, "__globals__"):
            underlying.__globals__["checkpoints"] = _mock_checkpoints
            underlying.__globals__["users"] = _mock_users
    live_mongo = sys.modules.get("chanakya.db.mongo")
    if live_mongo is not None:
        live_mongo.ai_tool_calls = _mock_ai_tool_calls


@pytest.fixture(autouse=True)
def _setup_each_test():
    """Clear collections and re-patch before every test."""
    _mock_users.delete_many({})
    _mock_checkpoints.delete_many({})
    _mock_ai_tool_calls.delete_many({})
    _repatch()
    yield


# ---------------------------------------------------------------------------
# Import tools (they reference chanakya.db.mongo at call time via _write_audit)
# ---------------------------------------------------------------------------

from chanakya.tools.schedule_tools import (  # noqa: E402
    ESCALATION_ORDER,
    activate_war_mode,
    add_daily_checkpoint,
    escalate_punishment,
    modify_wakeup_time,
    send_emergency_alert,
    update_morning_todo_time,
)


# ---------------------------------------------------------------------------
# escalate_punishment tests
# ---------------------------------------------------------------------------


class TestEscalatePunishment:
    def test_warn_advances_to_telegram_alert(self):
        """Test 1: Valid checkpoint at WARN → advances to TELEGRAM_ALERT."""
        uid = _make_user()
        cp_id = _make_checkpoint(uid, failure_punishment={"type": "WARN"})

        result = escalate_punishment.invoke(
            {"user_id": str(uid), "checkpoint_id": str(cp_id), "reason": "repeated failure"}
        )

        assert "WARN" in result
        assert "TELEGRAM_ALERT" in result
        assert "repeated failure" in result

        updated = _mock_checkpoints.find_one({"_id": cp_id})
        assert updated["failure_punishment"]["type"] == "TELEGRAM_ALERT"

    def test_telegram_alert_advances_to_call_emergency_contact(self):
        """Test 2: TELEGRAM_ALERT → CALL_EMERGENCY_CONTACT."""
        uid = _make_user()
        cp_id = _make_checkpoint(uid, failure_punishment={"type": "TELEGRAM_ALERT"})

        result = escalate_punishment.invoke(
            {"user_id": str(uid), "checkpoint_id": str(cp_id), "reason": "still failing"}
        )

        assert "CALL_EMERGENCY_CONTACT" in result
        updated = _mock_checkpoints.find_one({"_id": cp_id})
        assert updated["failure_punishment"]["type"] == "CALL_EMERGENCY_CONTACT"
        assert updated["failure_punishment"].get("emergency_alert_on_next_failure") is True

    def test_sms_emergency_contact_stays_at_max(self):
        """Test 3: Already at SMS_EMERGENCY_CONTACT → stays there."""
        uid = _make_user()
        cp_id = _make_checkpoint(uid, failure_punishment={"type": "SMS_EMERGENCY_CONTACT"})

        result = escalate_punishment.invoke(
            {"user_id": str(uid), "checkpoint_id": str(cp_id), "reason": "max level"}
        )

        assert "SMS_EMERGENCY_CONTACT" in result
        updated = _mock_checkpoints.find_one({"_id": cp_id})
        assert updated["failure_punishment"]["type"] == "SMS_EMERGENCY_CONTACT"

    def test_invalid_checkpoint_id_returns_error(self):
        """Test 4: Invalid checkpoint_id → returns error string, no DB write."""
        uid = _make_user()
        fake_id = str(ObjectId())

        result = escalate_punishment.invoke(
            {"user_id": str(uid), "checkpoint_id": fake_id, "reason": "test"}
        )

        assert result.startswith("Error:")
        assert _mock_ai_tool_calls.count_documents({}) == 0

    def test_audit_document_written_on_success(self):
        """Test 5: Audit document is written on successful escalation."""
        uid = _make_user()
        cp_id = _make_checkpoint(uid, failure_punishment={"type": "WARN"})

        escalate_punishment.invoke(
            {"user_id": str(uid), "checkpoint_id": str(cp_id), "reason": "audit test"}
        )

        audit_docs = list(_mock_ai_tool_calls.find({"tool_name": "escalate_punishment"}))
        assert len(audit_docs) == 1
        assert audit_docs[0]["tool_input"]["checkpoint_id"] == str(cp_id)


# ---------------------------------------------------------------------------
# modify_wakeup_time tests
# ---------------------------------------------------------------------------


class TestModifyWakeupTime:
    def test_valid_time_updates_call_checkpoint(self):
        """Test 6: Valid time + existing CALL checkpoint → updates time, returns confirmation."""
        uid = _make_user()
        cp_id = _make_checkpoint(uid, action_type="CALL", time="05:00")

        result = modify_wakeup_time.invoke(
            {"user_id": str(uid), "new_time": "06:30", "reason": "too early"}
        )

        assert "06:30" in result
        assert "too early" in result
        updated = _mock_checkpoints.find_one({"_id": cp_id})
        assert updated["time"] == "06:30"

    def test_invalid_time_format_returns_error(self):
        """Test 7: Invalid time format → returns error string, no DB write."""
        uid = _make_user()
        _make_checkpoint(uid, action_type="CALL", time="05:00")

        result = modify_wakeup_time.invoke(
            {"user_id": str(uid), "new_time": "6:30", "reason": "bad format"}
        )

        assert result.startswith("Error:")
        assert "HH:MM" in result
        cp = _mock_checkpoints.find_one({"action_type": "CALL"})
        assert cp["time"] == "05:00"

    def test_no_call_checkpoint_returns_error(self):
        """Test 8: No CALL checkpoint found → returns error string."""
        uid = _make_user()
        _make_checkpoint(uid, action_type="TELEGRAM_TEXT")

        result = modify_wakeup_time.invoke(
            {"user_id": str(uid), "new_time": "07:00", "reason": "no call cp"}
        )

        assert result.startswith("Error:")
        assert "CALL" in result


# ---------------------------------------------------------------------------
# activate_war_mode tests
# ---------------------------------------------------------------------------


class TestActivateWarMode:
    def test_valid_duration_sets_war_mode(self):
        """Test 9: Valid duration (24h) → sets WAR_MODE + war_mode_expires."""
        uid = _make_user()

        result = activate_war_mode.invoke(
            {"user_id": str(uid), "duration_hours": 24}
        )

        assert "WAR_MODE" in result
        assert "24" in result
        updated = _mock_users.find_one({"_id": uid})
        assert updated["current_mode"] == "WAR_MODE"
        assert updated["war_mode_expires"] is not None

    def test_duration_less_than_1_returns_error(self):
        """Test 10: Duration < 1 → returns error string, no DB write."""
        uid = _make_user()

        result = activate_war_mode.invoke(
            {"user_id": str(uid), "duration_hours": 0}
        )

        assert result.startswith("Error:")
        updated = _mock_users.find_one({"_id": uid})
        assert updated.get("current_mode") != "WAR_MODE"

    def test_duration_greater_than_72_returns_error(self):
        """Test 11: Duration > 72 → returns error string, no DB write."""
        uid = _make_user()

        result = activate_war_mode.invoke(
            {"user_id": str(uid), "duration_hours": 73}
        )

        assert result.startswith("Error:")
        updated = _mock_users.find_one({"_id": uid})
        assert updated.get("current_mode") != "WAR_MODE"

    def test_audit_document_written_on_success(self):
        """Test 12: Audit document written on success."""
        uid = _make_user()

        activate_war_mode.invoke({"user_id": str(uid), "duration_hours": 8})

        audit_docs = list(_mock_ai_tool_calls.find({"tool_name": "activate_war_mode"}))
        assert len(audit_docs) == 1
        assert audit_docs[0]["tool_input"]["duration_hours"] == 8


# ---------------------------------------------------------------------------
# add_daily_checkpoint tests
# ---------------------------------------------------------------------------


class TestAddDailyCheckpoint:
    def test_valid_inputs_inserts_checkpoint(self):
        """Test 13: Valid inputs → inserts checkpoint, returns confirmation."""
        uid = _make_user()

        result = add_daily_checkpoint.invoke(
            {
                "user_id": str(uid),
                "time_str": "14:00",
                "prompt": "Did you complete your LeetCode problem today?",
            }
        )

        assert "14:00" in result
        assert "Did you complete" in result
        cp = _mock_checkpoints.find_one({"user_id": uid, "time": "14:00"})
        assert cp is not None
        assert cp["active"] is True
        assert cp["action_type"] == "TELEGRAM_TEXT"

    def test_invalid_time_format_returns_error(self):
        """Test 14: Invalid time format → returns error string, no DB insert."""
        uid = _make_user()

        result = add_daily_checkpoint.invoke(
            {"user_id": str(uid), "time_str": "2pm", "prompt": "test prompt"}
        )

        assert result.startswith("Error:")
        assert _mock_checkpoints.count_documents({}) == 0

    def test_unknown_user_id_returns_error(self):
        """Test 15: Unknown user_id → returns error string, no DB insert."""
        fake_uid = str(ObjectId())

        result = add_daily_checkpoint.invoke(
            {"user_id": fake_uid, "time_str": "10:00", "prompt": "test prompt"}
        )

        assert result.startswith("Error:")
        assert _mock_checkpoints.count_documents({}) == 0


# ---------------------------------------------------------------------------
# send_emergency_alert tests
# ---------------------------------------------------------------------------


class TestSendEmergencyAlert:
    def test_user_with_emergency_contact_sends_sms(self):
        """Test 16: User with emergency contact → sends SMS, returns confirmation."""
        uid = _make_user(
            emergency_contact={"name": "Mom", "phone": "+911111111111", "relationship": "mother"}
        )

        mock_client = MagicMock()
        mock_client.send_sms.return_value = "SM123"

        with patch(
            "chanakya.integrations.twilio_client.TwilioClient", return_value=mock_client
        ):
            result = send_emergency_alert.invoke(
                {"user_id": str(uid), "message": "User has not responded for 2 hours."}
            )

        assert "Mom" in result
        assert "+911111111111" in result
        mock_client.send_sms.assert_called_once()
        call_kwargs = mock_client.send_sms.call_args
        assert call_kwargs.kwargs["to"] == "+911111111111"
        assert "Test User" in call_kwargs.kwargs["body"]

    def test_user_without_emergency_contact_returns_error(self):
        """Test 17: User without emergency contact → returns error string, no SMS."""
        uid = _make_user()

        mock_client = MagicMock()

        with patch(
            "chanakya.integrations.twilio_client.TwilioClient", return_value=mock_client
        ):
            result = send_emergency_alert.invoke(
                {"user_id": str(uid), "message": "test"}
            )

        assert result.startswith("Error:")
        assert "emergency_contact" in result
        mock_client.send_sms.assert_not_called()

    def test_twilio_failure_returns_error_no_crash(self):
        """Test 18: Twilio failure → returns error string, no crash."""
        from chanakya.integrations.twilio_client import TwilioError

        uid = _make_user(
            emergency_contact={"name": "Dad", "phone": "+912222222222"}
        )

        mock_client = MagicMock()
        mock_client.send_sms.side_effect = TwilioError("network error")

        with patch(
            "chanakya.integrations.twilio_client.TwilioClient", return_value=mock_client
        ):
            result = send_emergency_alert.invoke(
                {"user_id": str(uid), "message": "test failure"}
            )

        assert result.startswith("Error:")
        assert "SMS" in result or "failed" in result.lower()


# ---------------------------------------------------------------------------
# update_morning_todo_time tests
# ---------------------------------------------------------------------------


class TestUpdateMorningTodoTime:
    def test_valid_time_updates_user_and_checkpoints(self):
        """Test 19: Valid time + existing user → updates morning_todo_time, returns confirmation."""
        uid = _make_user()
        _make_checkpoint(uid, action_type="TELEGRAM_TEXT", priority="LOW", time="07:00")

        result = update_morning_todo_time.invoke(
            {"user_id": str(uid), "new_time": "08:30"}
        )

        assert "08:30" in result
        updated_user = _mock_users.find_one({"_id": uid})
        assert updated_user["morning_todo_time"] == "08:30"
        updated_cp = _mock_checkpoints.find_one(
            {"user_id": uid, "action_type": "TELEGRAM_TEXT", "priority": "LOW"}
        )
        assert updated_cp["time"] == "08:30"

    def test_invalid_time_format_returns_error(self):
        """Test 20: Invalid time format → returns error string, no DB write."""
        uid = _make_user()

        result = update_morning_todo_time.invoke(
            {"user_id": str(uid), "new_time": "8:30am"}
        )

        assert result.startswith("Error:")
        updated_user = _mock_users.find_one({"_id": uid})
        assert "morning_todo_time" not in updated_user

    def test_unknown_user_id_returns_error(self):
        """Test 21: Unknown user_id → returns error string."""
        fake_uid = str(ObjectId())

        result = update_morning_todo_time.invoke(
            {"user_id": fake_uid, "new_time": "09:00"}
        )

        assert result.startswith("Error:")


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    initial_level=st.sampled_from(ESCALATION_ORDER),
)
def test_p6_escalate_punishment_always_advances_in_order(initial_level):
    """P6: escalate_punishment always advances in order, never skips, never goes backwards.

    **Validates: Requirements 5**
    """
    _mock_users.delete_many({})
    _mock_checkpoints.delete_many({})
    _mock_ai_tool_calls.delete_many({})
    _repatch()

    uid = _make_user()
    cp_id = _make_checkpoint(uid, failure_punishment={"type": initial_level})

    escalate_punishment.invoke(
        {"user_id": str(uid), "checkpoint_id": str(cp_id), "reason": "property test"}
    )

    updated = _mock_checkpoints.find_one({"_id": cp_id})
    result_type = updated["failure_punishment"]["type"]

    current_idx = ESCALATION_ORDER.index(initial_level)
    expected_idx = min(current_idx + 1, len(ESCALATION_ORDER) - 1)
    expected_type = ESCALATION_ORDER[expected_idx]

    assert result_type == expected_type, (
        f"Expected {expected_type} after {initial_level}, got {result_type}"
    )
    result_idx = ESCALATION_ORDER.index(result_type)
    assert result_idx >= current_idx
    assert result_idx - current_idx <= 1


@settings(max_examples=100)
@given(
    duration=st.integers(min_value=-100, max_value=200),
)
def test_p7_activate_war_mode_rejects_invalid_duration(duration):
    """P7 (partial): activate_war_mode always rejects duration < 1 or > 72.

    **Validates: Requirements 7**
    """
    _mock_users.delete_many({})
    _mock_checkpoints.delete_many({})
    _mock_ai_tool_calls.delete_many({})
    _repatch()

    uid = _make_user()

    result = activate_war_mode.invoke(
        {"user_id": str(uid), "duration_hours": duration}
    )

    if duration < 1 or duration > 72:
        assert result.startswith("Error:"), (
            f"Expected error for duration={duration}, got: {result!r}"
        )
        updated = _mock_users.find_one({"_id": uid})
        assert updated.get("current_mode") != "WAR_MODE"
    else:
        assert "WAR_MODE" in result
        updated = _mock_users.find_one({"_id": uid})
        assert updated["current_mode"] == "WAR_MODE"
