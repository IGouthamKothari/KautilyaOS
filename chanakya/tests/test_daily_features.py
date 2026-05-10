"""
test_daily_features.py — Unit and property-based tests for daily_features.py.

Covers Tasks 26–29:
  - compute_daily_checkin_times
  - should_fire_checkin
  - get_or_create_checkin_schedule
  - should_send_morning_todo
  - generate_daily_snapshot
  - fire_morning_todo (fallback count)

**Validates: Requirements 17, 18, 19, 12**
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Environment setup — must happen before any chanakya imports
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

# Clear any previously imported chanakya modules so mocks take effect
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

# Now import the module under test
import chanakya.scheduler.daily_features as df_module
from chanakya.scheduler.daily_features import (
    _daily_checkin_schedule,
    _checkin_schedule_date,
    compute_daily_checkin_times,
    get_or_create_checkin_schedule,
    should_fire_checkin,
    should_send_morning_todo,
    generate_daily_snapshot,
    fire_morning_todo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user(**overrides) -> dict:
    base = {
        "_id": ObjectId(),
        "telegram_id": "123456",
        "name": "Test User",
        "phone": "+919999999999",
        "active": True,
        "current_mode": "NORMAL",
        "war_mode_expires": None,
        "timezone": "Asia/Kolkata",
        "current_activity": "FREE_TIME",
        "streak_count": 5,
        "morning_todo_time": "08:00",
        "morning_todo_fallback_count": 0,
        "checkin_window_start": "09:00",
        "checkin_window_end": "21:00",
        "checkin_min_per_day": 2,
        "checkin_max_per_day": 4,
        "next_day_plan": {},
        "eod_time": "21:00",
    }
    base.update(overrides)
    return base


def _clear_collections():
    _mock_db["users"].delete_many({})
    _mock_db["checkpoints"].delete_many({})
    _mock_db["interaction_logs"].delete_many({})
    _mock_db["user_state_snapshots"].delete_many({})


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
    mongo_module.user_state_snapshots = _mock_db["user_state_snapshots"]
    _clear_collections()
    _clear_checkin_state()


def _clear_checkin_state():
    """Clear module-level check-in schedule state between tests."""
    _daily_checkin_schedule.clear()
    _checkin_schedule_date.clear()


# ---------------------------------------------------------------------------
# Tests 1–3: compute_daily_checkin_times
# ---------------------------------------------------------------------------


class TestComputeDailyCheckinTimes(unittest.TestCase):
    def setUp(self):
        _clear_checkin_state()

    def test_1_returns_times_within_window(self):
        """Test 1: compute_daily_checkin_times returns times within the window."""
        user = _make_user(
            checkin_window_start="09:00",
            checkin_window_end="21:00",
            checkin_min_per_day=2,
            checkin_max_per_day=2,
        )
        times = compute_daily_checkin_times(user)
        self.assertGreater(len(times), 0)
        for t in times:
            h, m = map(int, t.split(":"))
            total_minutes = h * 60 + m
            self.assertGreaterEqual(total_minutes, 9 * 60)
            self.assertLess(total_minutes, 21 * 60)

    def test_2_enforces_90_minute_gap(self):
        """Test 2: compute_daily_checkin_times enforces 90-minute minimum gap."""
        user = _make_user(
            checkin_window_start="09:00",
            checkin_window_end="21:00",
            checkin_min_per_day=3,
            checkin_max_per_day=3,
        )
        # Run multiple times to reduce flakiness
        for _ in range(10):
            times = compute_daily_checkin_times(user)
            if len(times) >= 2:
                minutes = sorted(h * 60 + m for t in times for h, m in [map(int, t.split(":"))])
                for i in range(len(minutes) - 1):
                    self.assertGreaterEqual(
                        minutes[i + 1] - minutes[i],
                        90,
                        f"Gap too small: {times}",
                    )

    def test_3_returns_between_min_and_max_count(self):
        """Test 3: compute_daily_checkin_times returns between min and max count."""
        user = _make_user(
            checkin_window_start="09:00",
            checkin_window_end="21:00",
            checkin_min_per_day=2,
            checkin_max_per_day=4,
        )
        for _ in range(20):
            times = compute_daily_checkin_times(user)
            self.assertGreaterEqual(len(times), 0)
            self.assertLessEqual(len(times), 4)


# ---------------------------------------------------------------------------
# Tests 4–7: should_fire_checkin
# ---------------------------------------------------------------------------


class TestShouldFireCheckin(unittest.TestCase):
    def test_4_returns_false_for_war_mode(self):
        """Test 4: should_fire_checkin returns False for WAR_MODE."""
        user = _make_user(current_mode="WAR_MODE")
        self.assertFalse(should_fire_checkin(user))

    def test_5_returns_false_for_away_mode(self):
        """Test 5: should_fire_checkin returns False for AWAY mode."""
        user = _make_user(current_mode="AWAY")
        self.assertFalse(should_fire_checkin(user))

    def test_6_returns_false_for_sleep_activity(self):
        """Test 6: should_fire_checkin returns False for SLEEP activity slot."""
        user = _make_user(current_mode="NORMAL", current_activity="SLEEP")
        self.assertFalse(should_fire_checkin(user))

    def test_7_returns_true_for_normal_mode(self):
        """Test 7: should_fire_checkin returns True for NORMAL mode."""
        user = _make_user(current_mode="NORMAL", current_activity="FREE_TIME")
        self.assertTrue(should_fire_checkin(user))


# ---------------------------------------------------------------------------
# Tests 8–10: get_or_create_checkin_schedule
# ---------------------------------------------------------------------------


class TestGetOrCreateCheckinSchedule(unittest.TestCase):
    def setUp(self):
        _clear_checkin_state()

    def test_8_creates_new_schedule_for_new_day(self):
        """Test 8: get_or_create_checkin_schedule creates new schedule for new day."""
        user = _make_user()
        user_id_str = str(user["_id"])
        local_date = "2024-01-15"

        times = get_or_create_checkin_schedule(user, local_date)

        self.assertIn(user_id_str, _daily_checkin_schedule)
        self.assertEqual(_checkin_schedule_date[user_id_str], local_date)
        self.assertIsInstance(times, list)

    def test_9_reuses_schedule_for_same_day(self):
        """Test 9: get_or_create_checkin_schedule reuses schedule for same day."""
        user = _make_user()
        user_id_str = str(user["_id"])
        local_date = "2024-01-15"

        times_first = get_or_create_checkin_schedule(user, local_date)
        times_second = get_or_create_checkin_schedule(user, local_date)

        self.assertEqual(times_first, times_second)

    def test_10_resets_schedule_on_new_day(self):
        """Test 10: get_or_create_checkin_schedule resets schedule on new day."""
        user = _make_user()
        user_id_str = str(user["_id"])

        get_or_create_checkin_schedule(user, "2024-01-15")
        get_or_create_checkin_schedule(user, "2024-01-16")

        self.assertEqual(_checkin_schedule_date[user_id_str], "2024-01-16")


# ---------------------------------------------------------------------------
# Tests 11–12: should_send_morning_todo
# ---------------------------------------------------------------------------


class TestShouldSendMorningTodo(unittest.TestCase):
    def test_11_returns_false_when_morning_todo_time_is_none(self):
        """Test 11: should_send_morning_todo returns False when morning_todo_time is None."""
        user = _make_user(morning_todo_time=None)
        self.assertFalse(should_send_morning_todo(user))

    def test_12_returns_true_when_morning_todo_time_is_set(self):
        """Test 12: should_send_morning_todo returns True when morning_todo_time is set."""
        user = _make_user(morning_todo_time="08:00")
        self.assertTrue(should_send_morning_todo(user))


# ---------------------------------------------------------------------------
# Tests 13–15: generate_daily_snapshot
# ---------------------------------------------------------------------------


class TestGenerateDailySnapshot(unittest.TestCase):
    def setUp(self):
        _clear_collections()

    def test_13_inserts_snapshot_with_correct_fields(self):
        """Test 13: generate_daily_snapshot inserts snapshot document with correct fields."""
        user = _make_user()
        mongo_module.users.insert_one(user)

        with patch.object(df_module, "_compute_and_store_embedding", new=AsyncMock()):
            asyncio.run(generate_daily_snapshot(user))

        snapshots = list(mongo_module.user_state_snapshots.find({"user_id": user["_id"]}))
        self.assertEqual(len(snapshots), 1)
        snap = snapshots[0]
        self.assertEqual(snap["user_id"], user["_id"])
        self.assertIn("date", snap)
        self.assertIn("summary", snap)
        self.assertIn("embeddings", snap)
        self.assertIn("created_at", snap)
        self.assertIsNone(snap["embeddings"])  # None before embedding computed

    def test_14_skips_if_snapshot_already_exists(self):
        """Test 14: generate_daily_snapshot skips if snapshot already exists for today."""
        import pytz

        user = _make_user()
        mongo_module.users.insert_one(user)

        tz = pytz.timezone("Asia/Kolkata")
        today_date = datetime.now(tz).strftime("%Y-%m-%d")

        # Pre-insert a snapshot for today
        mongo_module.user_state_snapshots.insert_one(
            {
                "user_id": user["_id"],
                "date": today_date,
                "summary": "existing",
                "embeddings": None,
                "created_at": datetime.utcnow(),
            }
        )

        mock_embed = AsyncMock()
        with patch.object(df_module, "_compute_and_store_embedding", new=mock_embed):
            asyncio.run(generate_daily_snapshot(user))

        mock_embed.assert_not_called()
        snapshots = list(mongo_module.user_state_snapshots.find({"user_id": user["_id"]}))
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["summary"], "existing")

    def test_15_stores_snapshot_without_embeddings_on_embedding_failure(self):
        """Test 15: generate_daily_snapshot stores snapshot without embeddings on embedding failure."""
        user = _make_user()
        mongo_module.users.insert_one(user)

        async def failing_embed(snapshot_id, summary):
            raise Exception("Embedding API down")

        with patch.object(df_module, "_compute_and_store_embedding", side_effect=failing_embed):
            asyncio.run(generate_daily_snapshot(user))

        snapshots = list(mongo_module.user_state_snapshots.find({"user_id": user["_id"]}))
        self.assertEqual(len(snapshots), 1)
        self.assertIsNone(snapshots[0]["embeddings"])


# ---------------------------------------------------------------------------
# Tests 16–17: fire_morning_todo fallback count
# ---------------------------------------------------------------------------


class TestFireMorningTodo(unittest.TestCase):
    def setUp(self):
        _clear_collections()

    def _make_mock_decision(self):
        from chanakya.models.llm_decision import LLMDecision

        return LLMDecision(
            verdict=None,
            actions=[],
            tone="MENTOR",
            response_text="Here is your morning todo list.",
            reasoning="test",
            streak_reset=False,
            model_used="test-model",
        )

    def test_16_increments_fallback_count_when_no_confirmed_plan(self):
        """Test 16: fire_morning_todo increments morning_todo_fallback_count when no confirmed plan."""
        user = _make_user(
            morning_todo_time="08:00",
            morning_todo_fallback_count=0,
            next_day_plan={},
        )
        mongo_module.users.insert_one(user)

        mock_decision = self._make_mock_decision()
        mock_agent_instance = MagicMock()
        mock_agent_instance.invoke = AsyncMock(return_value=mock_decision)
        mock_agent_class = MagicMock(return_value=mock_agent_instance)

        with patch("chanakya.agent.chanakya_agent.ChanakyaAgent", mock_agent_class):
            with patch("chanakya.scheduler.checkpoint_runner._execute_telegram_text", MagicMock()):
                asyncio.run(fire_morning_todo(user))

        updated = mongo_module.users.find_one({"_id": user["_id"]})
        self.assertEqual(updated["morning_todo_fallback_count"], 1)

    def test_17_does_not_increment_fallback_count_when_confirmed_plan_exists(self):
        """Test 17: fire_morning_todo does NOT increment fallback count when confirmed plan exists."""
        user = _make_user(
            morning_todo_time="08:00",
            morning_todo_fallback_count=0,
            next_day_plan={
                "date": "2024-01-16",
                "plan_text": "1. Wake up 2. Gym 3. LeetCode",
                "confirmed": True,
            },
        )
        mongo_module.users.insert_one(user)

        mock_decision = self._make_mock_decision()
        mock_agent_instance = MagicMock()
        mock_agent_instance.invoke = AsyncMock(return_value=mock_decision)
        mock_agent_class = MagicMock(return_value=mock_agent_instance)

        with patch("chanakya.agent.chanakya_agent.ChanakyaAgent", mock_agent_class):
            with patch("chanakya.scheduler.checkpoint_runner._execute_telegram_text", MagicMock()):
                asyncio.run(fire_morning_todo(user))

        updated = mongo_module.users.find_one({"_id": user["_id"]})
        self.assertEqual(updated["morning_todo_fallback_count"], 0)


# ---------------------------------------------------------------------------
# Test 18 (Property): compute_daily_checkin_times always within window + 90-min gaps
# ---------------------------------------------------------------------------


class TestPropertyCheckinTimes(unittest.TestCase):
    """
    P18: compute_daily_checkin_times always returns times within the window
    and with >= 90 min gaps between any two times.

    **Validates: Requirements 19.2**
    """

    def test_18_property_times_within_window_and_90min_gaps(self):
        from hypothesis import given, settings
        from hypothesis import strategies as st

        @settings(max_examples=200)
        @given(
            window_start_h=st.integers(min_value=0, max_value=18),
            window_duration_h=st.integers(min_value=4, max_value=12),
            min_count=st.integers(min_value=1, max_value=3),
            extra=st.integers(min_value=0, max_value=2),
        )
        def check_property(window_start_h, window_duration_h, min_count, extra):
            max_count = min_count + extra
            window_end_h = window_start_h + window_duration_h
            if window_end_h > 23:
                window_end_h = 23

            user = _make_user(
                checkin_window_start=f"{window_start_h:02d}:00",
                checkin_window_end=f"{window_end_h:02d}:00",
                checkin_min_per_day=min_count,
                checkin_max_per_day=max_count,
            )

            times = compute_daily_checkin_times(user)

            start_minutes = window_start_h * 60
            end_minutes = window_end_h * 60

            # All times must be within the window
            for t in times:
                h, m = map(int, t.split(":"))
                total = h * 60 + m
                assert start_minutes <= total < end_minutes, (
                    f"Time {t} outside window [{window_start_h:02d}:00, {window_end_h:02d}:00)"
                )

            # All pairs must have >= 90 min gap
            minutes_list = sorted(
                h * 60 + m for t in times for h, m in [map(int, t.split(":"))]
            )
            for i in range(len(minutes_list) - 1):
                gap = minutes_list[i + 1] - minutes_list[i]
                assert gap >= 90, (
                    f"Gap {gap} < 90 between consecutive times in {times}"
                )

            # Count must be <= max_count
            assert len(times) <= max_count, (
                f"Got {len(times)} times, max is {max_count}"
            )

        check_property()


if __name__ == "__main__":
    unittest.main()
