"""
test_checkpoint_runner.py — Unit and property-based tests for the checkpoint runner.

**Validates: Requirements 2.1, 2.2, 2.3, 7.2, 7.3, 7.4, 7.7, 13.4, 14.3**
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

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

for _mod in list(sys.modules.keys()):
    if _mod.startswith("chanakya"):
        del sys.modules[_mod]

import mongomock
import pytest
from bson import ObjectId

_mock_mongo_client = mongomock.MongoClient()
_mock_mongo_client.admin.command = MagicMock(return_value={"ok": 1})

with patch("pymongo.MongoClient", return_value=_mock_mongo_client):
    import chanakya.db.mongo as mongo_module

_mock_db = _mock_mongo_client["chanakya"]
mongo_module.db = _mock_db
mongo_module.users = _mock_db["users"]
mongo_module.schedules = _mock_db["schedules"]
mongo_module.checkpoints = _mock_db["checkpoints"]
mongo_module.interaction_logs = _mock_db["interaction_logs"]
mongo_module.ai_tool_calls = _mock_db["ai_tool_calls"]
mongo_module.user_state_snapshots = _mock_db["user_state_snapshots"]
mongo_module.prompt_templates = _mock_db["prompt_templates"]

import chanakya.scheduler.checkpoint_runner as runner_module
from chanakya.scheduler.checkpoint_runner import (
    _expire_war_mode_if_needed,
    _should_skip_checkpoint,
)


def _make_user(**overrides) -> dict:
    base = {
        "_id": ObjectId(),
        "telegram_id": "123456",
        "name": "Test User",
        "phone": "+919999999999",
        "elevenlabs_voice_id": "voice-abc",
        "active": True,
        "current_mode": "NORMAL",
        "war_mode_expires": None,
        "timezone": "Asia/Kolkata",
        "current_activity": "FREE_TIME",
        "streak_count": 0,
    }
    base.update(overrides)
    return base


def _make_checkpoint(user_id, **overrides) -> dict:
    base = {
        "_id": ObjectId(),
        "user_id": user_id,
        "time": "07:00",
        "action_type": "TELEGRAM_TEXT",
        "priority": "HIGH",
        "prompt_template": "Wake up!",
        "active": True,
        "last_triggered": None,
    }
    base.update(overrides)
    return base


def _clear_collections():
    _mock_db["users"].delete_many({})
    _mock_db["checkpoints"].delete_many({})
    _mock_db["interaction_logs"].delete_many({})


@pytest.fixture(autouse=True)
def _repatch_mongo():
    """Re-apply mongo patches before every test to guard against cross-module contamination."""
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
    _clear_collections()


@pytest.mark.skip(reason="Tested old polling design — replaced by CronTrigger-based scheduler")
class TestRunOnce(unittest.TestCase):
    def setUp(self):
        _clear_collections()

    def test_run_once_processes_all_active_users(self):
        """Test 1: run_once processes all active users."""
        user1 = _make_user()
        user2 = _make_user()
        mongo_module.users.insert_many([user1, user2])

        processed = []

        def fake_process(u):
            processed.append(u["_id"])

        with patch.object(runner_module, "_process_user", side_effect=fake_process):
            with patch.object(runner_module, "_with_backoff", side_effect=lambda fn, *a, **kw: fn(*a, **kw)):
                run_once()

        self.assertEqual(len(processed), 2)
        self.assertIn(user1["_id"], processed)
        self.assertIn(user2["_id"], processed)

    def test_run_once_skips_inactive_users(self):
        """Test 2: run_once skips inactive users (active=False)."""
        active_user = _make_user(active=True)
        inactive_user = _make_user(active=False)
        mongo_module.users.insert_many([active_user, inactive_user])

        processed = []

        def fake_process(u):
            processed.append(u["_id"])

        with patch.object(runner_module, "_process_user", side_effect=fake_process):
            with patch.object(runner_module, "_with_backoff", side_effect=lambda fn, *a, **kw: fn(*a, **kw)):
                run_once()

        self.assertEqual(len(processed), 1)
        self.assertIn(active_user["_id"], processed)
        self.assertNotIn(inactive_user["_id"], processed)


class TestWarModeExpiry(unittest.TestCase):
    def setUp(self):
        _clear_collections()

    def test_war_mode_expires_when_past_expiry(self):
        """Test 3: WAR_MODE expires when war_mode_expires < utcnow()."""
        user = _make_user(
            current_mode="WAR_MODE",
            war_mode_expires=datetime.utcnow() - timedelta(hours=1),
        )
        mongo_module.users.insert_one(user)

        updated = _expire_war_mode_if_needed(user)

        self.assertEqual(updated["current_mode"], "NORMAL")
        self.assertIsNone(updated["war_mode_expires"])

        db_user = mongo_module.users.find_one({"_id": user["_id"]})
        self.assertEqual(db_user["current_mode"], "NORMAL")
        self.assertIsNone(db_user["war_mode_expires"])

    def test_war_mode_does_not_expire_when_future_expiry(self):
        """Test 4: WAR_MODE does NOT expire when war_mode_expires > utcnow()."""
        future_expiry = datetime.utcnow() + timedelta(hours=2)
        user = _make_user(
            current_mode="WAR_MODE",
            war_mode_expires=future_expiry,
        )
        mongo_module.users.insert_one(user)

        updated = _expire_war_mode_if_needed(user)

        self.assertEqual(updated["current_mode"], "WAR_MODE")
        db_user = mongo_module.users.find_one({"_id": user["_id"]})
        self.assertEqual(db_user["current_mode"], "WAR_MODE")


@pytest.mark.skip(reason="Tested old _get_due_checkpoints polling design — replaced by CronTrigger")
class TestCheckpointDeduplication(unittest.TestCase):
    def setUp(self):
        _clear_collections()

    def _insert_and_query(self, last_triggered, local_hhmm="07:00"):
        user = _make_user()
        cp = _make_checkpoint(user["_id"], time=local_hhmm, last_triggered=last_triggered)
        mongo_module.users.insert_one(user)
        mongo_module.checkpoints.insert_one(cp)
        return user, cp, _get_due_checkpoints(user, local_hhmm)

    def test_checkpoint_within_23h_is_skipped(self):
        """Test 5: Checkpoint with last_triggered within 23h → skipped."""
        recent = datetime.utcnow() - timedelta(hours=10)
        user, cp, due = self._insert_and_query(recent)
        self.assertEqual(len(due), 0)

    def test_checkpoint_beyond_23h_fires(self):
        """Test 6: Checkpoint with last_triggered > 23h ago → fires."""
        old = datetime.utcnow() - timedelta(hours=25)
        user, cp, due = self._insert_and_query(old)
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0]["_id"], cp["_id"])

    def test_checkpoint_never_triggered_fires(self):
        """Checkpoint with no last_triggered → fires."""
        user, cp, due = self._insert_and_query(None)
        self.assertEqual(len(due), 1)


class TestModeFiltering(unittest.TestCase):
    def test_war_mode_blocks_medium_priority(self):
        """Test 7: WAR_MODE blocks MEDIUM priority checkpoint."""
        user = _make_user(current_mode="WAR_MODE")
        cp = _make_checkpoint(user["_id"], priority="MEDIUM")
        self.assertTrue(_should_skip_checkpoint(user, cp))

    def test_war_mode_blocks_low_priority(self):
        user = _make_user(current_mode="WAR_MODE")
        cp = _make_checkpoint(user["_id"], priority="LOW")
        self.assertTrue(_should_skip_checkpoint(user, cp))

    def test_war_mode_does_not_block_critical(self):
        """Test 8: WAR_MODE does NOT block CRITICAL priority checkpoint."""
        user = _make_user(current_mode="WAR_MODE")
        cp = _make_checkpoint(user["_id"], priority="CRITICAL")
        self.assertFalse(_should_skip_checkpoint(user, cp))

    def test_war_mode_does_not_block_high(self):
        user = _make_user(current_mode="WAR_MODE")
        cp = _make_checkpoint(user["_id"], priority="HIGH")
        self.assertFalse(_should_skip_checkpoint(user, cp))

    def test_away_mode_blocks_non_critical(self):
        """Test 9: AWAY mode blocks non-CRITICAL checkpoint."""
        user = _make_user(current_mode="AWAY")
        cp = _make_checkpoint(user["_id"], priority="HIGH")
        self.assertTrue(_should_skip_checkpoint(user, cp))

    def test_away_mode_allows_critical(self):
        user = _make_user(current_mode="AWAY")
        cp = _make_checkpoint(user["_id"], priority="CRITICAL")
        self.assertFalse(_should_skip_checkpoint(user, cp))

    def test_sleep_blocks_check_in(self):
        """Test 10: SLEEP activity_slot blocks CHECK_IN action_type."""
        user = _make_user(current_activity="SLEEP")
        cp = _make_checkpoint(user["_id"], action_type="CHECK_IN")
        self.assertTrue(_should_skip_checkpoint(user, cp))

    def test_sleep_does_not_block_telegram_text(self):
        user = _make_user(current_activity="SLEEP")
        cp = _make_checkpoint(user["_id"], action_type="TELEGRAM_TEXT")
        self.assertFalse(_should_skip_checkpoint(user, cp))

    def test_normal_mode_does_not_block_any_priority(self):
        user = _make_user(current_mode="NORMAL")
        for priority in ("HIGH", "MEDIUM", "LOW", "CRITICAL"):
            cp = _make_checkpoint(user["_id"], priority=priority)
            self.assertFalse(_should_skip_checkpoint(user, cp))


class TestFireCheckpoint(unittest.TestCase):
    def setUp(self):
        _clear_collections()

    def _run_fire(self, user, cp):
        with patch.object(runner_module, "_execute_telegram_text", MagicMock()):
            with patch.object(runner_module, "_execute_call_action", return_value=None):
                with patch.object(runner_module, "_execute_telegram_voice", MagicMock()):
                    with patch(
                        "chanakya.agent.context_assembler.get_prompt_templates",
                        side_effect=Exception("no templates"),
                    ):
                        runner_module._fire_checkpoint(user, cp)

    def test_fire_checkpoint_inserts_interaction_log(self):
        """Test 11: _fire_checkpoint inserts interaction_log with correct fields."""
        user = _make_user()
        cp = _make_checkpoint(user["_id"])
        mongo_module.users.insert_one(user)
        mongo_module.checkpoints.insert_one(cp)

        self._run_fire(user, cp)

        logs = list(mongo_module.interaction_logs.find({"user_id": user["_id"]}))
        self.assertEqual(len(logs), 1)
        log = logs[0]
        self.assertEqual(log["user_id"], user["_id"])
        self.assertEqual(log["checkpoint_id"], cp["_id"])
        self.assertEqual(log["trigger_type"], "SCHEDULED")
        self.assertIn("timestamp", log)
        self.assertIn("message_sent", log)

    def test_fire_checkpoint_updates_last_triggered(self):
        """Test 12: _fire_checkpoint updates last_triggered on checkpoint."""
        user = _make_user()
        cp = _make_checkpoint(user["_id"])
        mongo_module.users.insert_one(user)
        mongo_module.checkpoints.insert_one(cp)

        before = datetime.utcnow() - timedelta(seconds=1)
        self._run_fire(user, cp)
        after = datetime.utcnow() + timedelta(seconds=1)

        updated_cp = mongo_module.checkpoints.find_one({"_id": cp["_id"]})
        self.assertIsNotNone(updated_cp["last_triggered"])
        self.assertGreaterEqual(updated_cp["last_triggered"], before)
        self.assertLessEqual(updated_cp["last_triggered"], after)


class TestCallActionFallback(unittest.TestCase):
    def setUp(self):
        _clear_collections()

    def test_call_falls_back_to_telegram_on_twilio_failure(self):
        """Test 13: CALL action falls back to TELEGRAM_TEXT when Twilio fails."""
        from chanakya.integrations.twilio_client import TwilioError

        user = _make_user()
        cp = _make_checkpoint(user["_id"], action_type="CALL")

        telegram_calls = []

        def fake_telegram(u, text):
            telegram_calls.append(text)

        mock_twilio = MagicMock()
        mock_twilio.make_call.side_effect = TwilioError("Twilio down")

        with patch("chanakya.integrations.twilio_client.TwilioClient", return_value=mock_twilio):
            with patch("chanakya.integrations.twilio_webhooks.synthesize_call_opening", return_value=None):
                with patch("chanakya.integrations.twilio_webhooks.create_voice_session", return_value=None):
                    with patch.object(runner_module, "_execute_telegram_text", side_effect=fake_telegram):
                        result = runner_module._execute_call_action(
                            user, cp, "Wake up!", ObjectId()
                        )

        self.assertIsNone(result)
        self.assertEqual(len(telegram_calls), 1)
        self.assertEqual(telegram_calls[0], "Wake up!")


class TestTelegramVoiceFallback(unittest.TestCase):
    def setUp(self):
        _clear_collections()

    def test_voice_falls_back_to_text_on_elevenlabs_failure(self):
        """Test 14: TELEGRAM_VOICE falls back to plain text when ElevenLabs fails."""
        from chanakya.integrations.elevenlabs_client import ElevenLabsSynthesisError

        user = _make_user()
        cp = _make_checkpoint(user["_id"], action_type="TELEGRAM_VOICE")

        telegram_calls = []

        def fake_telegram(u, text):
            telegram_calls.append(text)

        mock_el = MagicMock()
        mock_el.synthesise.side_effect = ElevenLabsSynthesisError("API down")

        with patch("chanakya.integrations.elevenlabs_client.ElevenLabsClient", return_value=mock_el):
            with patch.object(runner_module, "_execute_telegram_text", side_effect=fake_telegram):
                runner_module._execute_telegram_voice(user, cp, "Hello!")

        self.assertEqual(len(telegram_calls), 1)
        self.assertEqual(telegram_calls[0], "Hello!")


@pytest.mark.skip(reason="Tested old _get_local_hhmm helper — no longer needed with CronTrigger timezone param")
class TestTimezoneConversion(unittest.TestCase):
    def test_invalid_timezone_falls_back_to_kolkata(self):
        """Test 15: Invalid timezone falls back to Asia/Kolkata."""
        user = _make_user(timezone="Invalid/Timezone")
        result = _get_local_hhmm(user)
        self.assertRegex(result, r"^\d{2}:\d{2}$")

    def test_valid_timezone_returns_hhmm(self):
        user = _make_user(timezone="America/New_York")
        result = _get_local_hhmm(user)
        self.assertRegex(result, r"^\d{2}:\d{2}$")


@pytest.mark.skip(reason="Tested old _get_due_checkpoints polling design — replaced by CronTrigger")
class TestPropertyDeduplication(unittest.TestCase):
    """P3: Any last_triggered within 23h always skips; beyond 23h always fires."""

    def setUp(self):
        _clear_collections()

    def test_p3_deduplication_property(self):
        from hypothesis import given, settings
        from hypothesis import strategies as st

        @settings(max_examples=100)
        @given(hours_ago=st.floats(min_value=0.01, max_value=22.99))
        def within_23h_always_skips(hours_ago):
            _clear_collections()
            user = _make_user()
            last_triggered = datetime.utcnow() - timedelta(hours=hours_ago)
            cp = _make_checkpoint(user["_id"], time="07:00", last_triggered=last_triggered)
            mongo_module.users.insert_one(user)
            mongo_module.checkpoints.insert_one(cp)
            due = _get_due_checkpoints(user, "07:00")
            assert len(due) == 0, f"Expected 0 for {hours_ago:.2f}h ago, got {len(due)}"

        @settings(max_examples=100)
        @given(hours_ago=st.floats(min_value=23.01, max_value=200.0))
        def beyond_23h_always_fires(hours_ago):
            _clear_collections()
            user = _make_user()
            last_triggered = datetime.utcnow() - timedelta(hours=hours_ago)
            cp = _make_checkpoint(user["_id"], time="07:00", last_triggered=last_triggered)
            mongo_module.users.insert_one(user)
            mongo_module.checkpoints.insert_one(cp)
            due = _get_due_checkpoints(user, "07:00")
            assert len(due) == 1, f"Expected 1 for {hours_ago:.2f}h ago, got {len(due)}"

        within_23h_always_skips()
        beyond_23h_always_fires()


@pytest.mark.skip(reason="Tested old _get_local_hhmm helper — no longer needed with CronTrigger timezone param")
class TestPropertyTimezoneConversion(unittest.TestCase):
    """P5: Any valid IANA timezone always produces a valid HH:MM string."""

    def test_p5_timezone_conversion_property(self):
        import re
        from hypothesis import given, settings
        from hypothesis import strategies as st

        valid_timezones = [
            "Asia/Kolkata", "America/New_York", "America/Los_Angeles",
            "Europe/London", "Europe/Paris", "Asia/Tokyo", "Australia/Sydney",
            "Pacific/Auckland", "America/Chicago", "America/Denver",
            "Asia/Dubai", "Asia/Singapore", "Africa/Cairo", "America/Sao_Paulo",
            "Asia/Shanghai", "Europe/Berlin", "Europe/Moscow", "America/Toronto",
            "Asia/Seoul", "Pacific/Honolulu",
        ]

        @settings(max_examples=100)
        @given(tz_str=st.sampled_from(valid_timezones))
        def valid_tz_produces_hhmm(tz_str):
            user = _make_user(timezone=tz_str)
            result = _get_local_hhmm(user)
            assert re.match(r"^\d{2}:\d{2}$", result), f"Bad format for {tz_str!r}: {result!r}"
            hour, minute = result.split(":")
            assert 0 <= int(hour) <= 23
            assert 0 <= int(minute) <= 59

        valid_tz_produces_hhmm()


if __name__ == "__main__":
    unittest.main()
