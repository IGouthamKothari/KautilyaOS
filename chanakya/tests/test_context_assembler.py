"""
test_context_assembler.py - Tests for chanakya/agent/context_assembler.py.

Uses mongomock for all DB calls. Covers:
  1.  Tier 1 always included for all interaction types
  2.  Tier 2 included for CHECKPOINT, not for COMMAND_RESPONSE
  3.  Tier 3 included for CHECK_IN, not for CHECKPOINT
  4.  Tier 4 included for EOD, not for CHECK_IN
  5.  todays_checkpoints returns clean dicts with no ObjectIds
  6.  weekly_summary returns list of strings, no raw documents
  7.  recent_snapshots returns text only, no embeddings
  8.  get_prompt_templates returns all tones for exact match
  9.  get_prompt_templates falls back to FREE_TIME when no exact match
  10. get_prompt_templates raises NoTemplateFoundError when no fallback exists
  11. render_template substitutes all known variables
  12. render_template strips unknown variables silently (no raise)
  13. clear_template_cache empties the cache
  14. P9: Context dict never contains ObjectId instances or embedding vectors
  15. P15: No ObjectIds in any tier output for any combination of interaction_type
"""

from __future__ import annotations

import asyncio as _asyncio
import os
import sys
import unittest.mock as mock
from datetime import datetime, timedelta

import mongomock
import pytest
from bson import ObjectId
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Environment setup - must happen before any chanakya import
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
_mock_client.admin.command = mock.MagicMock(return_value={"ok": 1})
_mock_db = _mock_client["chanakya"]

with mock.patch("pymongo.MongoClient", return_value=_mock_client):
    import chanakya.db.mongo as mongo_module
    import chanakya.agent.context_assembler as ca_module

# Point all mongo_module collection handles to our dedicated mock_db
mongo_module.db = _mock_db
mongo_module.users = _mock_db["users"]
mongo_module.checkpoints = _mock_db["checkpoints"]
mongo_module.interaction_logs = _mock_db["interaction_logs"]
mongo_module.user_state_snapshots = _mock_db["user_state_snapshots"]
mongo_module.prompt_templates = _mock_db["prompt_templates"]


# ---------------------------------------------------------------------------
# autouse fixture: re-apply collection patches before every test
# This ensures other test modules cannot steal our mongo_module references.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _repatch_mongo_collections():
    """
    Re-point mongo_module collection handles to our mock_db before each test.

    IMPORTANT: We must update sys.modules['chanakya.db.mongo'] directly because
    other test files (test_telegram_bot.py) delete and re-import chanakya modules
    during collection, creating a new module object in sys.modules. The lazy imports
    inside context_assembler.py use sys.modules, not our local mongo_module reference.
    """
    import sys as _sys
    live_mongo = _sys.modules.get("chanakya.db.mongo", mongo_module)
    live_mongo.db = _mock_db
    live_mongo.users = _mock_db["users"]
    live_mongo.checkpoints = _mock_db["checkpoints"]
    live_mongo.interaction_logs = _mock_db["interaction_logs"]
    live_mongo.user_state_snapshots = _mock_db["user_state_snapshots"]
    live_mongo.prompt_templates = _mock_db["prompt_templates"]
    # Also update our local reference
    mongo_module.db = _mock_db
    mongo_module.users = _mock_db["users"]
    mongo_module.checkpoints = _mock_db["checkpoints"]
    mongo_module.interaction_logs = _mock_db["interaction_logs"]
    mongo_module.user_state_snapshots = _mock_db["user_state_snapshots"]
    mongo_module.prompt_templates = _mock_db["prompt_templates"]
    # Clear all collections and template cache
    _mock_db["users"].delete_many({})
    _mock_db["checkpoints"].delete_many({})
    _mock_db["interaction_logs"].delete_many({})
    _mock_db["user_state_snapshots"].delete_many({})
    _mock_db["prompt_templates"].delete_many({})
    ca_module.clear_template_cache()


# ---------------------------------------------------------------------------
# Async helper
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run a coroutine from synchronous test code."""
    return _asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user(**kwargs) -> dict:
    """Return a minimal user dict suitable for context assembly."""
    user_id = ObjectId()
    base = {
        "_id": user_id,
        "telegram_id": "tg_test",
        "name": "Test User",
        "timezone": "Asia/Kolkata",
        "streak_count": 3,
        "longest_streak": 10,
        "current_mode": "NORMAL",
        "current_activity": "FREE_TIME",
        "recurring_failure_patterns": [],
    }
    base.update(kwargs)
    return base


def _insert_template(activity_slot: str, interaction_type: str, tone: str, text: str):
    _mock_db["prompt_templates"].insert_one(
        {
            "activity_slot": activity_slot,
            "interaction_type": interaction_type,
            "tone": tone,
            "template_text": text,
        }
    )


def _contains_objectid(obj, _seen=None) -> bool:
    """Recursively check if any value in obj is an ObjectId."""
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return False
    _seen.add(obj_id)
    if isinstance(obj, ObjectId):
        return True
    if isinstance(obj, dict):
        return any(_contains_objectid(v, _seen) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_objectid(item, _seen) for item in obj)
    return False


def _contains_embedding(obj, _seen=None) -> bool:
    """
    Recursively check if any value looks like an embedding vector
    (a list of floats with length > 10).
    """
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return False
    _seen.add(obj_id)
    if isinstance(obj, list) and len(obj) > 10 and all(isinstance(x, float) for x in obj):
        return True
    if isinstance(obj, dict):
        return any(_contains_embedding(v, _seen) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_embedding(item, _seen) for item in obj)
    return False


# ---------------------------------------------------------------------------
# Test 1: Tier 1 always included for all interaction types
# ---------------------------------------------------------------------------


class TestTier1AlwaysIncluded:
    def test_tier1_present_for_command_response(self):
        user = _make_user()
        assembler = ca_module.ContextAssembler()
        ctx = _run_async(assembler.build(user, "COMMAND_RESPONSE"))
        assert ctx["tier1"] is not None
        assert "name" in ctx["tier1"]
        assert ctx["tier1"]["name"] == "Test User"

    def test_tier1_present_for_checkpoint(self):
        user = _make_user()
        assembler = ca_module.ContextAssembler()
        ctx = _run_async(assembler.build(user, "CHECKPOINT"))
        assert ctx["tier1"] is not None

    def test_tier1_present_for_eod(self):
        user = _make_user()
        assembler = ca_module.ContextAssembler()
        ctx = _run_async(assembler.build(user, "EOD"))
        assert ctx["tier1"] is not None

    def test_tier1_contains_required_fields(self):
        user = _make_user()
        assembler = ca_module.ContextAssembler()
        ctx = _run_async(assembler.build(user, "COMMAND_RESPONSE"))
        t1 = ctx["tier1"]
        for field in [
            "name", "streak_count", "longest_streak",
            "failure_count_this_week", "failure_count_this_month",
            "current_mode", "current_activity_slot",
            "relationship_summary", "today_date", "day_of_week", "timezone",
        ]:
            assert field in t1, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# Test 2: Tier 2 included for CHECKPOINT, not for COMMAND_RESPONSE
# ---------------------------------------------------------------------------


class TestTier2Inclusion:
    def test_tier2_present_for_checkpoint(self):
        user = _make_user()
        assembler = ca_module.ContextAssembler()
        ctx = _run_async(assembler.build(user, "CHECKPOINT"))
        assert ctx["tier2"] is not None

    def test_tier2_absent_for_command_response(self):
        user = _make_user()
        assembler = ca_module.ContextAssembler()
        ctx = _run_async(assembler.build(user, "COMMAND_RESPONSE"))
        assert ctx["tier2"] is None

    def test_tier2_present_for_morning_todo(self):
        user = _make_user()
        assembler = ca_module.ContextAssembler()
        ctx = _run_async(assembler.build(user, "MORNING_TODO"))
        assert ctx["tier2"] is not None

    def test_tier2_present_for_check_in(self):
        user = _make_user()
        assembler = ca_module.ContextAssembler()
        ctx = _run_async(assembler.build(user, "CHECK_IN"))
        assert ctx["tier2"] is not None


# ---------------------------------------------------------------------------
# Test 3: Tier 3 included for CHECK_IN, not for CHECKPOINT
# ---------------------------------------------------------------------------


class TestTier3Inclusion:
    def test_tier3_present_for_check_in(self):
        user = _make_user()
        assembler = ca_module.ContextAssembler()
        ctx = _run_async(assembler.build(user, "CHECK_IN"))
        assert ctx["tier3"] is not None

    def test_tier3_absent_for_checkpoint(self):
        user = _make_user()
        assembler = ca_module.ContextAssembler()
        ctx = _run_async(assembler.build(user, "CHECKPOINT"))
        assert ctx["tier3"] is None

    def test_tier3_present_for_eod(self):
        user = _make_user()
        assembler = ca_module.ContextAssembler()
        ctx = _run_async(assembler.build(user, "EOD"))
        assert ctx["tier3"] is not None

    def test_tier3_present_for_escalation(self):
        user = _make_user()
        assembler = ca_module.ContextAssembler()
        ctx = _run_async(assembler.build(user, "ESCALATION"))
        assert ctx["tier3"] is not None


# ---------------------------------------------------------------------------
# Test 4: Tier 4 included for EOD, not for CHECK_IN
# ---------------------------------------------------------------------------


class TestTier4Inclusion:
    def test_tier4_present_for_eod(self):
        user = _make_user()
        assembler = ca_module.ContextAssembler()
        ctx = _run_async(assembler.build(user, "EOD"))
        assert ctx["tier4"] is not None

    def test_tier4_absent_for_check_in(self):
        user = _make_user()
        assembler = ca_module.ContextAssembler()
        ctx = _run_async(assembler.build(user, "CHECK_IN"))
        assert ctx["tier4"] is None

    def test_tier4_present_for_weekly_review(self):
        user = _make_user()
        assembler = ca_module.ContextAssembler()
        ctx = _run_async(assembler.build(user, "WEEKLY_REVIEW"))
        assert ctx["tier4"] is not None

    def test_tier4_absent_for_checkpoint(self):
        user = _make_user()
        assembler = ca_module.ContextAssembler()
        ctx = _run_async(assembler.build(user, "CHECKPOINT"))
        assert ctx["tier4"] is None


# ---------------------------------------------------------------------------
# Test 5: todays_checkpoints returns clean dicts with no ObjectIds
# ---------------------------------------------------------------------------


class TestTodaysCheckpoints:
    def test_todays_checkpoints_no_objectids(self):
        user = _make_user()
        cp_id = ObjectId()
        _mock_db["checkpoints"].insert_one(
            {
                "_id": cp_id,
                "user_id": user["_id"],
                "time": "06:00",
                "action_type": "TELEGRAM_TEXT",
                "priority": "HIGH",
                "prompt_template": "Wake Up",
                "active": True,
            }
        )
        now = datetime.utcnow()
        _mock_db["interaction_logs"].insert_one(
            {
                "user_id": user["_id"],
                "checkpoint_id": cp_id,
                "timestamp": now,
                "trigger_type": "SCHEDULED",
                "channel": "TELEGRAM",
                "message_sent": "Wake up!",
                "ai_evaluation": {"verdict": "SUCCESS", "confidence": 0.9, "reasoning": "ok"},
            }
        )

        tier2 = ca_module._build_tier2(user)
        checkpoints_list = tier2["todays_checkpoints"]
        assert isinstance(checkpoints_list, list)
        assert len(checkpoints_list) >= 1

        for entry in checkpoints_list:
            assert not _contains_objectid(entry), f"ObjectId found in checkpoint entry: {entry}"
            assert "name" in entry
            assert "verdict" in entry
            assert "triggered_at" in entry

    def test_todays_checkpoints_empty_when_no_logs(self):
        user = _make_user()
        tier2 = ca_module._build_tier2(user)
        assert tier2["todays_checkpoints"] == []


# ---------------------------------------------------------------------------
# Test 6: weekly_summary returns list of strings, no raw documents
# ---------------------------------------------------------------------------


class TestWeeklySummary:
    def test_weekly_summary_is_list_of_strings(self):
        user = _make_user()
        cp_id = ObjectId()
        _mock_db["checkpoints"].insert_one(
            {
                "_id": cp_id,
                "user_id": user["_id"],
                "time": "06:00",
                "action_type": "TELEGRAM_TEXT",
                "priority": "HIGH",
                "prompt_template": "Wake Up",
                "active": True,
            }
        )
        two_days_ago = datetime.utcnow() - timedelta(days=2)
        _mock_db["interaction_logs"].insert_one(
            {
                "user_id": user["_id"],
                "checkpoint_id": cp_id,
                "timestamp": two_days_ago,
                "trigger_type": "SCHEDULED",
                "channel": "TELEGRAM",
                "message_sent": "Wake up!",
                "ai_evaluation": {"verdict": "FAILED", "confidence": 0.8, "reasoning": "no"},
            }
        )

        tier3 = ca_module._build_tier3(user, "FREE_TIME")
        summary = tier3["weekly_summary"]
        assert isinstance(summary, list)
        for item in summary:
            assert isinstance(item, str), f"Non-string in weekly_summary: {type(item)}"
            assert not _contains_objectid(item)

    def test_weekly_summary_empty_when_no_logs(self):
        user = _make_user()
        tier3 = ca_module._build_tier3(user, "FREE_TIME")
        assert tier3["weekly_summary"] == []

    def test_weekly_summary_format(self):
        user = _make_user()
        cp_id = ObjectId()
        _mock_db["checkpoints"].insert_one(
            {
                "_id": cp_id,
                "user_id": user["_id"],
                "time": "06:00",
                "action_type": "TELEGRAM_TEXT",
                "priority": "HIGH",
                "prompt_template": "Gym",
                "active": True,
            }
        )
        yesterday = datetime.utcnow() - timedelta(days=1)
        _mock_db["interaction_logs"].insert_one(
            {
                "user_id": user["_id"],
                "checkpoint_id": cp_id,
                "timestamp": yesterday,
                "trigger_type": "SCHEDULED",
                "channel": "TELEGRAM",
                "message_sent": "Did you gym?",
                "ai_evaluation": {"verdict": "SUCCESS"},
            }
        )

        tier3 = ca_module._build_tier3(user, "FREE_TIME")
        summary = tier3["weekly_summary"]
        assert any("SUCCESS" in s for s in summary)


# ---------------------------------------------------------------------------
# Test 7: recent_snapshots returns text only, no embeddings
# ---------------------------------------------------------------------------


class TestRecentSnapshots:
    def test_recent_snapshots_text_only(self):
        user = _make_user()
        for i in range(3):
            _mock_db["user_state_snapshots"].insert_one(
                {
                    "user_id": user["_id"],
                    "date": f"2024-01-0{i+1}",
                    "summary": f"Day {i+1} summary text",
                    "embeddings": [0.1] * 512,
                }
            )

        tier3 = ca_module._build_tier3(user, "FREE_TIME")
        snapshots = tier3["recent_snapshots"]
        assert isinstance(snapshots, list)
        assert len(snapshots) <= 3
        for snap in snapshots:
            assert isinstance(snap, str), f"Non-string snapshot: {type(snap)}"
            assert not _contains_embedding([snap])

    def test_recent_snapshots_empty_when_none(self):
        user = _make_user()
        tier3 = ca_module._build_tier3(user, "FREE_TIME")
        assert tier3["recent_snapshots"] == []


# ---------------------------------------------------------------------------
# Test 8: get_prompt_templates returns all tones for exact match
# ---------------------------------------------------------------------------


class TestGetPromptTemplatesExactMatch:
    def test_returns_all_tones_for_exact_match(self):
        _insert_template("FREE_TIME", "CHECKPOINT", "HARSH", "Harsh template {name}")
        _insert_template("FREE_TIME", "CHECKPOINT", "MENTOR", "Mentor template {name}")
        _insert_template("FREE_TIME", "CHECKPOINT", "NEUTRAL", "Neutral template {name}")

        result = ca_module.get_prompt_templates("FREE_TIME", "CHECKPOINT")
        assert "HARSH" in result
        assert "MENTOR" in result
        assert "NEUTRAL" in result
        assert result["HARSH"] == "Harsh template {name}"
        assert result["MENTOR"] == "Mentor template {name}"

    def test_result_is_cached(self):
        _insert_template("LEETCODE", "CHECK_IN", "HARSH", "LeetCode harsh {streak}")
        result1 = ca_module.get_prompt_templates("LEETCODE", "CHECK_IN")
        result2 = ca_module.get_prompt_templates("LEETCODE", "CHECK_IN")
        assert result1 is result2


# ---------------------------------------------------------------------------
# Test 9: get_prompt_templates falls back to FREE_TIME when no exact match
# ---------------------------------------------------------------------------


class TestGetPromptTemplatesFallback:
    def test_falls_back_to_free_time(self):
        _insert_template("FREE_TIME", "CHECKPOINT", "HARSH", "Free time harsh")

        result = ca_module.get_prompt_templates("OFFICE_WORK", "CHECKPOINT")
        assert "HARSH" in result
        assert result["HARSH"] == "Free time harsh"

    def test_falls_back_to_generic(self):
        _insert_template("GENERIC", "CHECKPOINT", "NEUTRAL", "Generic neutral")

        result = ca_module.get_prompt_templates("OFFICE_WORK", "CHECKPOINT")
        assert "NEUTRAL" in result
        assert result["NEUTRAL"] == "Generic neutral"


# ---------------------------------------------------------------------------
# Test 10: get_prompt_templates raises NoTemplateFoundError when no fallback
# ---------------------------------------------------------------------------


class TestGetPromptTemplatesNoTemplate:
    def test_raises_no_template_found_error(self):
        with pytest.raises(ca_module.NoTemplateFoundError):
            ca_module.get_prompt_templates("OFFICE_WORK", "CHECKPOINT")

    def test_error_message_contains_slot_and_type(self):
        try:
            ca_module.get_prompt_templates("GYM", "EOD")
        except ca_module.NoTemplateFoundError as e:
            assert "GYM" in str(e)
            assert "EOD" in str(e)


# ---------------------------------------------------------------------------
# Test 11: render_template substitutes all known variables
# ---------------------------------------------------------------------------


class TestRenderTemplateSubstitution:
    def test_substitutes_known_variables(self):
        template = "Hello {name}, your streak is {streak_count}."
        context = {"tier1": {"name": "Alice", "streak_count": 5}}
        result = ca_module.render_template(template, context)
        assert "Alice" in result
        assert "5" in result
        assert "{name}" not in result
        assert "{streak_count}" not in result

    def test_substitutes_nested_variables(self):
        template = "Mode: {current_mode}"
        context = {"tier1": {"current_mode": "WAR_MODE"}}
        result = ca_module.render_template(template, context)
        assert "WAR_MODE" in result

    def test_substitutes_top_level_variables(self):
        template = "Date: {today_date}"
        context = {"today_date": "2024-01-15"}
        result = ca_module.render_template(template, context)
        assert "2024-01-15" in result


# ---------------------------------------------------------------------------
# Test 12: render_template strips unknown variables silently (no raise)
# ---------------------------------------------------------------------------


class TestRenderTemplateUnknownVariables:
    def test_strips_unknown_variables_silently(self):
        template = "Hello {name}, your {unknown_var} is ready."
        context = {"tier1": {"name": "Bob"}}
        result = ca_module.render_template(template, context)
        assert "Bob" in result
        assert "{unknown_var}" not in result
        assert "{name}" not in result

    def test_no_exception_on_all_unknown(self):
        template = "{totally_unknown} {also_unknown}"
        context = {}
        result = ca_module.render_template(template, context)
        assert result.strip() == ""

    def test_partial_substitution(self):
        template = "{name} has {missing_field} failures."
        context = {"name": "Charlie"}
        result = ca_module.render_template(template, context)
        assert "Charlie" in result
        assert "{missing_field}" not in result


# ---------------------------------------------------------------------------
# Test 13: clear_template_cache empties the cache
# ---------------------------------------------------------------------------


class TestClearTemplateCache:
    def test_clear_cache_empties_it(self):
        _insert_template("FREE_TIME", "CHECKPOINT", "HARSH", "Template text")
        ca_module.get_prompt_templates("FREE_TIME", "CHECKPOINT")
        assert len(ca_module._template_cache) > 0

        ca_module.clear_template_cache()
        assert len(ca_module._template_cache) == 0

    def test_after_clear_fresh_query_works(self):
        _insert_template("FREE_TIME", "EOD", "MENTOR", "EOD mentor template")
        ca_module.get_prompt_templates("FREE_TIME", "EOD")
        ca_module.clear_template_cache()

        result = ca_module.get_prompt_templates("FREE_TIME", "EOD")
        assert "MENTOR" in result


# ---------------------------------------------------------------------------
# Test 14 (P9): Context dict never contains ObjectId instances or embedding vectors
# ---------------------------------------------------------------------------


class TestNoObjectIdsOrEmbeddings:
    def test_tier1_no_objectids(self):
        user = _make_user()
        tier1 = ca_module._build_tier1(user)
        assert not _contains_objectid(tier1), "ObjectId found in tier1"

    def test_tier2_no_objectids(self):
        user = _make_user()
        cp_id = ObjectId()
        _mock_db["checkpoints"].insert_one(
            {
                "_id": cp_id,
                "user_id": user["_id"],
                "time": "06:00",
                "action_type": "TELEGRAM_TEXT",
                "priority": "HIGH",
                "prompt_template": "Wake Up",
                "active": True,
            }
        )
        _mock_db["interaction_logs"].insert_one(
            {
                "user_id": user["_id"],
                "checkpoint_id": cp_id,
                "timestamp": datetime.utcnow(),
                "trigger_type": "SCHEDULED",
                "channel": "TELEGRAM",
                "message_sent": "Wake up!",
                "ai_evaluation": {"verdict": "SUCCESS"},
            }
        )
        tier2 = ca_module._build_tier2(user)
        assert not _contains_objectid(tier2), "ObjectId found in tier2"

    def test_tier3_no_objectids_or_embeddings(self):
        user = _make_user()
        _mock_db["user_state_snapshots"].insert_one(
            {
                "user_id": user["_id"],
                "date": "2024-01-01",
                "summary": "Good day",
                "embeddings": [0.1] * 512,
            }
        )
        tier3 = ca_module._build_tier3(user, "FREE_TIME")
        assert not _contains_objectid(tier3), "ObjectId found in tier3"
        assert not _contains_embedding(tier3), "Embedding vector found in tier3"

    def test_tier4_no_objectids_or_embeddings(self):
        user = _make_user()
        _mock_db["user_state_snapshots"].insert_one(
            {
                "user_id": user["_id"],
                "date": "2024-01-01",
                "summary": "Deep memory day",
                "embeddings": [0.2] * 512,
            }
        )
        tier4 = ca_module._build_tier4(user, "context text")
        assert not _contains_objectid(tier4), "ObjectId found in tier4"
        assert not _contains_embedding(tier4), "Embedding vector found in tier4"

    def test_full_context_no_objectids(self):
        user = _make_user()
        assembler = ca_module.ContextAssembler()
        ctx = _run_async(assembler.build(user, "EOD"))
        assert not _contains_objectid(ctx), "ObjectId found in full context"
        assert not _contains_embedding(ctx), "Embedding vector found in full context"


# ---------------------------------------------------------------------------
# Test 15 (P15): Property test - no ObjectIds in any tier for any interaction_type
#
# Validates: Requirements 20.1
# ---------------------------------------------------------------------------

INTERACTION_TYPES = [
    "COMMAND_RESPONSE",
    "CHECKPOINT",
    "MORNING_TODO",
    "CHECK_IN",
    "ESCALATION",
    "MENTOR_TALK",
    "EOD",
    "WEEKLY_REVIEW",
]


@settings(max_examples=50, deadline=None)
@given(
    interaction_type=st.sampled_from(INTERACTION_TYPES),
    streak_count=st.integers(min_value=0, max_value=100),
    current_mode=st.sampled_from(["NORMAL", "WAR_MODE", "INJURED", "AWAY"]),
    activity_slot=st.sampled_from(
        ["FREE_TIME", "LEETCODE", "OFFICE_WORK", "GYM", "SLEEP", "STUDY", "COMMUTE", "MEAL"]
    ),
)
def test_p15_no_objectids_in_any_tier(
    interaction_type: str,
    streak_count: int,
    current_mode: str,
    activity_slot: str,
):
    """
    **Validates: Requirements 20.1**

    For any combination of interaction_type, the assembled context must never
    contain ObjectId instances or embedding vectors in any tier.
    """
    # Re-apply patches (Hypothesis runs outside fixture scope)
    import sys as _sys
    live_mongo = _sys.modules.get("chanakya.db.mongo", mongo_module)
    live_mongo.db = _mock_db
    live_mongo.users = _mock_db["users"]
    live_mongo.checkpoints = _mock_db["checkpoints"]
    live_mongo.interaction_logs = _mock_db["interaction_logs"]
    live_mongo.user_state_snapshots = _mock_db["user_state_snapshots"]
    live_mongo.prompt_templates = _mock_db["prompt_templates"]
    mongo_module.db = _mock_db
    mongo_module.users = _mock_db["users"]
    mongo_module.checkpoints = _mock_db["checkpoints"]
    mongo_module.interaction_logs = _mock_db["interaction_logs"]
    mongo_module.user_state_snapshots = _mock_db["user_state_snapshots"]
    mongo_module.prompt_templates = _mock_db["prompt_templates"]

    # Clear collections and cache
    _mock_db["users"].delete_many({})
    _mock_db["checkpoints"].delete_many({})
    _mock_db["interaction_logs"].delete_many({})
    _mock_db["user_state_snapshots"].delete_many({})
    _mock_db["prompt_templates"].delete_many({})
    ca_module.clear_template_cache()

    user = _make_user(
        streak_count=streak_count,
        current_mode=current_mode,
        current_activity=activity_slot,
    )

    _mock_db["user_state_snapshots"].insert_one(
        {
            "user_id": user["_id"],
            "date": "2024-01-01",
            "summary": "Test snapshot summary",
            "embeddings": [0.1] * 128,
        }
    )

    assembler = ca_module.ContextAssembler()
    ctx = _run_async(assembler.build(user, interaction_type))

    assert not _contains_objectid(ctx), (
        f"ObjectId found in context for interaction_type={interaction_type!r}, "
        f"activity_slot={activity_slot!r}"
    )
    assert not _contains_embedding(ctx), (
        f"Embedding vector found in context for interaction_type={interaction_type!r}"
    )
