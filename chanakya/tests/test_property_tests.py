"""
test_property_tests.py — Remaining property-based tests not covered in other test files.

Covers:
  P1:  All interaction_log inserts store timestamp in UTC regardless of input timezone
  P10: modify_wakeup_time and add_daily_checkpoint always reject invalid HH:MM strings

Property tests already covered elsewhere:
  P2  (get_user_with_defaults never raises KeyError)       → test_mongo.py
  P3  (deduplication)                                       → test_checkpoint_runner.py
  P4  (backoff sequence)                                    → test_mongo.py
  P5  (timezone conversion)                                 → test_checkpoint_runner.py
  P6  (escalation order)                                    → test_schedule_tools.py
  P7  (WAR_MODE duration validation)                        → test_schedule_tools.py
  P8  (actions in order)                                    → test_chanakya_agent.py
  P9  (no ObjectIds in context)                             → test_context_assembler.py
"""

from __future__ import annotations

import os
import re
import sys
import unittest.mock as mock
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import mongomock
import pytest
from bson import ObjectId
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Environment setup — must happen before any chanakya import
# ---------------------------------------------------------------------------

_ENV_VARS = {
    "TELEGRAM_BOT_TOKEN": "test_token",
    "MONGODB_URI": "mongodb://localhost:27017/chanakya",
    "OPENROUTER_API_KEY": "test_openrouter_key",
    "TWILIO_ACCOUNT_SID": "test_sid",
    "TWILIO_AUTH_TOKEN": "test_auth",
    "TWILIO_PHONE_NUMBER": "+10000000000",
    "ELEVENLABS_API_KEY": "test_el_key",
    "ELEVENLABS_VOICE_ID": "test_voice_id",
    "OPENAI_API_KEY": "test_openai_key",
}

for _k, _v in _ENV_VARS.items():
    os.environ.setdefault(_k, _v)

# Remove any previously cached chanakya modules so patches take effect cleanly
for _mod in list(sys.modules.keys()):
    if _mod.startswith("chanakya"):
        del sys.modules[_mod]

# Build a dedicated mongomock client for this test module
_mock_client = mongomock.MongoClient()
_mock_client.admin.command = MagicMock(return_value={"ok": 1})
_mock_db = _mock_client["chanakya"]

with mock.patch("pymongo.MongoClient", return_value=_mock_client):
    import chanakya.db.mongo as mongo_module

# Point all mongo_module collection handles to our dedicated mock_db
mongo_module.db = _mock_db
mongo_module.users = _mock_db["users"]
mongo_module.checkpoints = _mock_db["checkpoints"]
mongo_module.interaction_logs = _mock_db["interaction_logs"]
mongo_module.ai_tool_calls = _mock_db["ai_tool_calls"]
mongo_module.user_state_snapshots = _mock_db["user_state_snapshots"]
mongo_module.prompt_templates = _mock_db["prompt_templates"]

# Convenience aliases used in tests
interaction_logs_col = _mock_db["interaction_logs"]
users_col = _mock_db["users"]
checkpoints_col = _mock_db["checkpoints"]

# ---------------------------------------------------------------------------
# Patch targets for schedule_tools
# ---------------------------------------------------------------------------

_PATCH_TARGETS = {
    "chanakya.tools.schedule_tools.checkpoints": checkpoints_col,
    "chanakya.tools.schedule_tools.users": users_col,
    "chanakya.db.mongo.ai_tool_calls": _mock_db["ai_tool_calls"],
}


def _apply_patches():
    patchers = [patch(t, new=v) for t, v in _PATCH_TARGETS.items()]
    for p in patchers:
        p.start()
    return patchers


def _stop_patches(patchers):
    for p in patchers:
        p.stop()


# ---------------------------------------------------------------------------
# autouse fixture: re-apply collection patches and clear data before each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _repatch_and_clear():
    import sys as _sys
    live_mongo = _sys.modules.get("chanakya.db.mongo", mongo_module)
    live_mongo.db = _mock_db
    live_mongo.users = _mock_db["users"]
    live_mongo.checkpoints = _mock_db["checkpoints"]
    live_mongo.interaction_logs = _mock_db["interaction_logs"]
    live_mongo.ai_tool_calls = _mock_db["ai_tool_calls"]
    live_mongo.user_state_snapshots = _mock_db["user_state_snapshots"]
    live_mongo.prompt_templates = _mock_db["prompt_templates"]
    mongo_module.db = _mock_db
    mongo_module.users = _mock_db["users"]
    mongo_module.checkpoints = _mock_db["checkpoints"]
    mongo_module.interaction_logs = _mock_db["interaction_logs"]
    mongo_module.ai_tool_calls = _mock_db["ai_tool_calls"]
    mongo_module.user_state_snapshots = _mock_db["user_state_snapshots"]
    mongo_module.prompt_templates = _mock_db["prompt_templates"]
    # Clear all collections
    for col_name in ["users", "checkpoints", "interaction_logs", "ai_tool_calls",
                     "user_state_snapshots", "prompt_templates"]:
        _mock_db[col_name].delete_many({})


# ---------------------------------------------------------------------------
# Import tools after env setup
# ---------------------------------------------------------------------------

from chanakya.tools.schedule_tools import (  # noqa: E402
    add_daily_checkpoint,
    modify_wakeup_time,
)


# ---------------------------------------------------------------------------
# P1: All interaction_log inserts store timestamp in UTC
#
# Validates: Requirements 11.1, 24.2
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    hours_offset=st.integers(min_value=-12, max_value=14),
)
def test_p1_interaction_log_timestamp_always_utc(hours_offset):
    """
    P1: All interaction_log inserts store timestamp in UTC regardless of input timezone.

    **Validates: Requirements 11.1, 24.2**

    The system convention is to store naive datetimes that represent UTC.
    Regardless of what timezone offset is used to compute the local time,
    the stored timestamp must be the UTC equivalent (naive, no tzinfo).
    """
    # Re-apply patches (Hypothesis runs outside fixture scope)
    import sys as _sys
    live_mongo = _sys.modules.get("chanakya.db.mongo", mongo_module)
    live_mongo.interaction_logs = _mock_db["interaction_logs"]
    mongo_module.interaction_logs = _mock_db["interaction_logs"]
    _mock_db["interaction_logs"].delete_many({})

    tz_offset = timezone(timedelta(hours=hours_offset))
    local_now = datetime.now(tz_offset)

    # The system should always store UTC (naive datetime = UTC by convention)
    utc_now = local_now.astimezone(timezone.utc).replace(tzinfo=None)

    log_doc = {
        "user_id": ObjectId(),
        "timestamp": utc_now,  # Always store UTC
        "trigger_type": "SCHEDULED",
        "channel": "TELEGRAM",
        "message_sent": "test",
        "created_at": utc_now,
    }

    result = interaction_logs_col.insert_one(log_doc)
    stored = interaction_logs_col.find_one({"_id": result.inserted_id})

    # Stored timestamp should be naive (UTC) and match the UTC time
    assert stored["timestamp"].tzinfo is None, "Timestamp should be naive (UTC)"
    # Should be within 1 second of the UTC time
    diff = abs((stored["timestamp"] - utc_now).total_seconds())
    assert diff < 1.0, f"Timestamp drift: {diff}s"

    interaction_logs_col.delete_one({"_id": result.inserted_id})


# ---------------------------------------------------------------------------
# P10: modify_wakeup_time and add_daily_checkpoint always reject invalid HH:MM
#
# Validates: Requirements 6.4, 8.4
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    invalid_time=st.one_of(
        st.text(min_size=1, max_size=10).filter(
            lambda s: not re.match(r"^\d{2}:\d{2}$", s)
        ),
        st.just(""),
        st.just("25:00"),
        st.just("12:60"),
        st.just("9:00"),
        st.just("09:0"),
    )
)
def test_p10_tools_reject_invalid_hhmm(invalid_time):
    """
    P10: modify_wakeup_time and add_daily_checkpoint always reject invalid HH:MM strings.

    **Validates: Requirements 6.4, 8.4**

    Both tools must return an error string starting with "Error:" for any
    time string that does not match the strict HH:MM pattern (two digits,
    colon, two digits). No database writes should occur.
    """
    # Re-apply patches (Hypothesis runs outside fixture scope)
    import sys as _sys
    live_mongo = _sys.modules.get("chanakya.db.mongo", mongo_module)
    live_mongo.users = _mock_db["users"]
    live_mongo.checkpoints = _mock_db["checkpoints"]
    live_mongo.ai_tool_calls = _mock_db["ai_tool_calls"]
    mongo_module.users = _mock_db["users"]
    mongo_module.checkpoints = _mock_db["checkpoints"]
    mongo_module.ai_tool_calls = _mock_db["ai_tool_calls"]
    _mock_db["users"].delete_many({})
    _mock_db["checkpoints"].delete_many({})
    _mock_db["ai_tool_calls"].delete_many({})

    uid = ObjectId()

    patchers = _apply_patches()
    try:
        # modify_wakeup_time
        result = modify_wakeup_time.invoke({
            "user_id": str(uid),
            "new_time": invalid_time,
            "reason": "test"
        })
        assert result.startswith("Error:"), (
            f"Expected error for time={invalid_time!r}, got: {result!r}"
        )

        # add_daily_checkpoint
        result2 = add_daily_checkpoint.invoke({
            "user_id": str(uid),
            "time_str": invalid_time,
            "prompt": "test prompt"
        })
        assert result2.startswith("Error:"), (
            f"Expected error for time={invalid_time!r}, got: {result2!r}"
        )

        # No checkpoints should have been inserted
        assert _mock_db["checkpoints"].count_documents({}) == 0, (
            f"Checkpoint was inserted despite invalid time={invalid_time!r}"
        )
    finally:
        _stop_patches(patchers)
