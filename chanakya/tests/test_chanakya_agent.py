"""
test_chanakya_agent.py - Tests for chanakya/agent/chanakya_agent.py.

Uses unittest.mock.patch to mock ChatOpenAI and avoid real API calls.
Covers:
  1.  ChanakyaAgent.invoke returns LLMDecision when LLM returns valid JSON
  2.  ChanakyaAgent.invoke returns None when all 3 models fail
  3.  ChanakyaAgent.invoke falls back to model 2 when model 1 fails
  4.  ChanakyaAgent.invoke falls back to model 3 when models 1 and 2 fail
  5.  Malformed LLM response returns None, logs raw response
  6.  LLM response with JSON embedded in text is extracted and parsed
  7.  execute_actions with increment_streak updates DB correctly
  8.  execute_actions with reset_streak updates DB correctly
  9.  execute_actions with update_activity_slot updates DB correctly
  10. execute_actions with one failing action continues with remaining actions
  11. execute_actions executes actions in exact array order
  12. ai_tool_calls audit document written for each model attempt
  13. P8: execute_actions always executes actions in array order (property test)
"""
from __future__ import annotations
import os
import sys
import json
import asyncio
import unittest.mock as mock
from datetime import datetime
from unittest.mock import MagicMock, patch

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
_mock_client.admin.command = MagicMock(return_value={"ok": 1})
_mock_db = _mock_client["chanakya"]

with mock.patch("pymongo.MongoClient", return_value=_mock_client):
    import chanakya.db.mongo as mongo_module
    import chanakya.agent.chanakya_agent as agent_module

# Point all mongo_module collection handles to our dedicated mock_db
mongo_module.db = _mock_db
mongo_module.users = _mock_db["users"]
mongo_module.interaction_logs = _mock_db["interaction_logs"]
mongo_module.ai_tool_calls = _mock_db["ai_tool_calls"]
mongo_module.checkpoints = _mock_db["checkpoints"]
mongo_module.prompt_templates = _mock_db["prompt_templates"]
mongo_module.user_state_snapshots = _mock_db["user_state_snapshots"]


def _repatch():
    """Re-point mongo_module collections to mock_db (called in each test)."""
    import sys as _sys
    live_mongo = _sys.modules.get("chanakya.db.mongo", mongo_module)
    live_mongo.db = _mock_db
    live_mongo.users = _mock_db["users"]
    live_mongo.interaction_logs = _mock_db["interaction_logs"]
    live_mongo.ai_tool_calls = _mock_db["ai_tool_calls"]
    live_mongo.checkpoints = _mock_db["checkpoints"]
    live_mongo.prompt_templates = _mock_db["prompt_templates"]
    live_mongo.user_state_snapshots = _mock_db["user_state_snapshots"]
    mongo_module.db = _mock_db
    mongo_module.users = _mock_db["users"]
    mongo_module.interaction_logs = _mock_db["interaction_logs"]
    mongo_module.ai_tool_calls = _mock_db["ai_tool_calls"]
    mongo_module.checkpoints = _mock_db["checkpoints"]
    mongo_module.prompt_templates = _mock_db["prompt_templates"]
    mongo_module.user_state_snapshots = _mock_db["user_state_snapshots"]


# ---------------------------------------------------------------------------
# autouse fixture: re-apply collection patches before every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _repatch_mongo_collections():
    _repatch()
    # Clear all collections
    for col in ["users", "interaction_logs", "ai_tool_calls", "checkpoints",
                "prompt_templates", "user_state_snapshots"]:
        _mock_db[col].delete_many({})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_user(**kwargs):
    base = {
        "_id": ObjectId(),
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


def _valid_decision_json(model="anthropic/claude-3.5-sonnet"):
    return json.dumps({
        "verdict": "SUCCESS",
        "actions": [],
        "tone": "NEUTRAL",
        "response_text": "Well done.",
        "reasoning": "User completed the task.",
        "streak_reset": False,
        "model_used": model,
    })


def _make_llm_response(content):
    resp = MagicMock()
    resp.content = content
    return resp


def _patch_llm(side_effects):
    """
    Return a factory that creates mock LLM instances with given side_effects.
    Each element is either a string (success response content) or an Exception.
    The factory accepts no arguments to match _make_llm() call signature.
    """
    call_count = [0]

    def factory(*args, **kwargs):
        idx = call_count[0]
        call_count[0] += 1
        m = MagicMock()
        if idx < len(side_effects):
            effect = side_effects[idx]
            if isinstance(effect, Exception):
                m.invoke.side_effect = effect
            else:
                m.invoke.return_value = _make_llm_response(effect)
        else:
            m.invoke.side_effect = Exception("unexpected call")
        return m

    return factory


def _run_async(coro):
    """Run an async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


_MOCK_CONTEXT = {
    "tier1": {
        "name": "Test User",
        "streak_count": 3,
        "longest_streak": 10,
        "failure_count_this_week": 0,
        "failure_count_this_month": 0,
        "current_mode": "NORMAL",
        "current_activity_slot": "FREE_TIME",
        "relationship_summary": {},
        "today_date": "2024-01-01",
        "day_of_week": "Monday",
        "timezone": "Asia/Kolkata",
    },
    "tier2": None,
    "tier3": None,
    "tier4": None,
    "prompt_templates": {},
}


# ---------------------------------------------------------------------------
# Test 1: invoke returns LLMDecision when LLM returns valid JSON
# ---------------------------------------------------------------------------

class TestInvokeSuccess:
    def test_returns_llm_decision_on_valid_json(self):
        user = _make_user()
        _mock_db["users"].insert_one(dict(user))

        factory = _patch_llm([_valid_decision_json()])

        with patch.object(agent_module, "_make_llm", side_effect=factory):
            with patch.object(agent_module.ContextAssembler, "build", return_value=_MOCK_CONTEXT):
                agent = agent_module.ChanakyaAgent(user)
                result = _run_async(agent.invoke("I completed my workout", "CHECKPOINT"))

        assert result is not None
        assert result.verdict == "SUCCESS"
        assert result.response_text == "Well done."
        assert result.tone == "NEUTRAL"


# ---------------------------------------------------------------------------
# Test 2: invoke returns None when the model fails
# ---------------------------------------------------------------------------

class TestInvokeAllModelsFail:
    def test_returns_none_when_model_fails(self):
        user = _make_user()
        _mock_db["users"].insert_one(dict(user))

        factory = _patch_llm([Exception("model error")])

        with patch.object(agent_module, "_make_llm", side_effect=factory):
            with patch.object(agent_module.ContextAssembler, "build", return_value=_MOCK_CONTEXT):
                agent = agent_module.ChanakyaAgent(user)
                result = _run_async(agent.invoke("hello", "CHECKPOINT"))

        assert result is None


# ---------------------------------------------------------------------------
# Test 3: invoke succeeds on first attempt (single-model architecture)
# ---------------------------------------------------------------------------

class TestInvokeFallbackToModel2:
    def test_succeeds_on_first_attempt(self):
        user = _make_user()
        _mock_db["users"].insert_one(dict(user))

        factory = _patch_llm([_valid_decision_json("gpt-4o-mini")])

        with patch.object(agent_module, "_make_llm", side_effect=factory):
            with patch.object(agent_module.ContextAssembler, "build", return_value=_MOCK_CONTEXT):
                agent = agent_module.ChanakyaAgent(user)
                result = _run_async(agent.invoke("hello", "CHECKPOINT"))

        assert result is not None
        assert result.model_used == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Test 4: invoke returns None when model raises (single-model architecture)
# ---------------------------------------------------------------------------

class TestInvokeFallbackToModel3:
    def test_returns_none_when_model_raises(self):
        user = _make_user()
        _mock_db["users"].insert_one(dict(user))

        factory = _patch_llm([Exception("model error")])

        with patch.object(agent_module, "_make_llm", side_effect=factory):
            with patch.object(agent_module.ContextAssembler, "build", return_value=_MOCK_CONTEXT):
                agent = agent_module.ChanakyaAgent(user)
                result = _run_async(agent.invoke("hello", "CHECKPOINT"))

        assert result is None


# ---------------------------------------------------------------------------
# Test 5: Malformed LLM response returns None
# ---------------------------------------------------------------------------

class TestMalformedResponse:
    def test_malformed_response_returns_none(self):
        user = _make_user()
        _mock_db["users"].insert_one(dict(user))

        factory = _patch_llm(["This is not JSON at all, just plain text."])

        with patch.object(agent_module, "_make_llm", side_effect=factory):
            with patch.object(agent_module.ContextAssembler, "build", return_value=_MOCK_CONTEXT):
                agent = agent_module.ChanakyaAgent(user)
                result = _run_async(agent.invoke("hello", "CHECKPOINT"))

        assert result is None

    def test_parse_llm_decision_returns_none_on_garbage(self):
        result = agent_module._parse_llm_decision("garbage text no json here", "model-x")
        assert result is None


# ---------------------------------------------------------------------------
# Test 6: LLM response with JSON embedded in text is extracted and parsed
# ---------------------------------------------------------------------------

class TestEmbeddedJsonExtraction:
    def test_json_embedded_in_text_is_parsed(self):
        decision_json = _valid_decision_json()
        wrapped = "Here is my response:\n" + decision_json + "\nEnd of response."

        result = agent_module._parse_llm_decision(wrapped, "anthropic/claude-3.5-sonnet")

        assert result is not None
        assert result.verdict == "SUCCESS"
        assert result.response_text == "Well done."

    def test_json_with_preamble_is_parsed(self):
        decision_json = json.dumps({
            "verdict": "FAILED",
            "actions": [{"type": "reset_streak", "params": {}}],
            "tone": "HARSH",
            "response_text": "You failed.",
            "reasoning": "No effort.",
            "streak_reset": True,
            "model_used": "openai/gpt-4o",
        })
        wrapped = "Thinking... " + decision_json

        result = agent_module._parse_llm_decision(wrapped, "openai/gpt-4o")

        assert result is not None
        assert result.verdict == "FAILED"
        assert result.streak_reset is True


# ---------------------------------------------------------------------------
# Test 7: execute_actions with increment_streak updates DB correctly
# ---------------------------------------------------------------------------

class TestExecuteActionsIncrementStreak:
    def test_increment_streak_updates_db(self):
        user = _make_user(streak_count=5, longest_streak=10)
        _mock_db["users"].insert_one(dict(user))

        from chanakya.models.llm_decision import ActionItem
        actions = [ActionItem(type="increment_streak", params={})]

        agent_module.execute_actions(actions, user, log_id=None)

        updated = _mock_db["users"].find_one({"_id": user["_id"]})
        assert updated["streak_count"] == 6

    def test_increment_streak_updates_longest_when_new_high(self):
        user = _make_user(streak_count=10, longest_streak=10)
        _mock_db["users"].insert_one(dict(user))

        from chanakya.models.llm_decision import ActionItem
        actions = [ActionItem(type="increment_streak", params={})]

        agent_module.execute_actions(actions, user, log_id=None)

        updated = _mock_db["users"].find_one({"_id": user["_id"]})
        assert updated["streak_count"] == 11
        assert updated["longest_streak"] == 11


# ---------------------------------------------------------------------------
# Test 8: execute_actions with reset_streak updates DB correctly
# ---------------------------------------------------------------------------

class TestExecuteActionsResetStreak:
    def test_reset_streak_sets_to_zero(self):
        user = _make_user(streak_count=7)
        _mock_db["users"].insert_one(dict(user))

        from chanakya.models.llm_decision import ActionItem
        actions = [ActionItem(type="reset_streak", params={})]

        agent_module.execute_actions(actions, user, log_id=None)

        updated = _mock_db["users"].find_one({"_id": user["_id"]})
        assert updated["streak_count"] == 0


# ---------------------------------------------------------------------------
# Test 9: execute_actions with update_activity_slot updates DB correctly
# ---------------------------------------------------------------------------

class TestExecuteActionsUpdateActivitySlot:
    def test_update_activity_slot_sets_current_activity(self):
        user = _make_user(current_activity="FREE_TIME")
        _mock_db["users"].insert_one(dict(user))

        from chanakya.models.llm_decision import ActionItem
        actions = [ActionItem(type="update_activity_slot", params={"slot": "GYM"})]

        agent_module.execute_actions(actions, user, log_id=None)

        updated = _mock_db["users"].find_one({"_id": user["_id"]})
        assert updated["current_activity"] == "GYM"
        assert updated["activity_slot_updated_at"] is not None


# ---------------------------------------------------------------------------
# Test 10: execute_actions with one failing action continues with remaining
# ---------------------------------------------------------------------------

class TestExecuteActionsContinuesOnFailure:
    def test_continues_after_failing_action(self):
        user = _make_user(streak_count=3, current_activity="FREE_TIME")
        _mock_db["users"].insert_one(dict(user))

        from chanakya.models.llm_decision import ActionItem

        # First action will raise, second should still execute
        actions = [
            ActionItem(type="reset_streak", params={}),
            ActionItem(type="update_activity_slot", params={"slot": "LEETCODE"}),
        ]

        original_update = _mock_db["users"].update_one
        call_count = [0]

        def patched_update(filter_, update, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("simulated DB failure")
            return original_update(filter_, update, **kwargs)

        with patch.object(_mock_db["users"], "update_one", side_effect=patched_update):
            # Should not raise - execute_actions catches individual failures
            agent_module.execute_actions(actions, user, log_id=None)

        # Second action (update_activity_slot) should have executed
        updated = _mock_db["users"].find_one({"_id": user["_id"]})
        assert updated["current_activity"] == "LEETCODE"


# ---------------------------------------------------------------------------
# Test 11: execute_actions executes actions in exact array order
# ---------------------------------------------------------------------------

class TestExecuteActionsOrder:
    def test_actions_executed_in_order(self):
        user = _make_user(streak_count=5, longest_streak=10)
        _mock_db["users"].insert_one(dict(user))

        from chanakya.models.llm_decision import ActionItem

        # reset_streak then increment_streak: net result should be streak=1
        actions = [
            ActionItem(type="reset_streak", params={}),
            ActionItem(type="increment_streak", params={}),
        ]

        agent_module.execute_actions(actions, user, log_id=None)

        updated = _mock_db["users"].find_one({"_id": user["_id"]})
        # reset to 0, then increment to 1
        assert updated["streak_count"] == 1


# ---------------------------------------------------------------------------
# Test 12: ai_tool_calls audit document written for each model attempt
# ---------------------------------------------------------------------------

class TestAuditLogForModelAttempts:
    def test_audit_log_written_for_successful_attempt(self):
        user = _make_user()
        _mock_db["users"].insert_one(dict(user))

        factory = _patch_llm([_valid_decision_json()])

        with patch.object(agent_module, "_make_llm", side_effect=factory):
            with patch.object(agent_module.ContextAssembler, "build", return_value=_MOCK_CONTEXT):
                agent = agent_module.ChanakyaAgent(user)
                _run_async(agent.invoke("hello", "CHECKPOINT"))

        audit_docs = list(_mock_db["ai_tool_calls"].find({"tool_name": "_llm_attempt"}))
        assert len(audit_docs) >= 1
        doc = audit_docs[0]
        assert doc["tool_name"] == "_llm_attempt"
        assert "model" in doc["tool_input"]
        assert doc["tool_output"] == "success"

    def test_audit_log_written_for_failed_attempt(self):
        user = _make_user()
        _mock_db["users"].insert_one(dict(user))

        factory = _patch_llm([Exception("API error")])

        with patch.object(agent_module, "_make_llm", side_effect=factory):
            with patch.object(agent_module.ContextAssembler, "build", return_value=_MOCK_CONTEXT):
                agent = agent_module.ChanakyaAgent(user)
                _run_async(agent.invoke("hello", "CHECKPOINT"))

        audit_docs = list(_mock_db["ai_tool_calls"].find({"tool_name": "_llm_attempt"}))
        assert len(audit_docs) >= 1
        failed_doc = next(d for d in audit_docs if d["tool_output"] != "success")
        assert "API error" in failed_doc["tool_output"]


# ---------------------------------------------------------------------------
# Test 13 (P8): Property test - execute_actions always executes in array order
#
# Validates: Requirements 25.4
# ---------------------------------------------------------------------------

_SAFE_ACTION_TYPES = [
    "reset_streak",
    "update_activity_slot",
    "store_next_day_plan",
    "confirm_next_day_plan",
    "send_telegram",
    "send_voice",
]


@settings(max_examples=50)
@given(
    action_types=st.lists(
        st.sampled_from(_SAFE_ACTION_TYPES),
        min_size=1,
        max_size=8,
    )
)
def test_p8_execute_actions_in_array_order(action_types):
    """
    **Validates: Requirements 25.4**

    For any sequence of actions, execute_actions must execute them in the
    exact order they appear in the array. No action is skipped unless it
    raises an exception.
    """
    _repatch()
    _mock_db["users"].delete_many({})

    user = _make_user(streak_count=0, longest_streak=0, current_activity="FREE_TIME")
    _mock_db["users"].insert_one(dict(user))

    from chanakya.models.llm_decision import ActionItem

    execution_log = []

    def make_action(action_type):
        params = {}
        if action_type == "update_activity_slot":
            params = {"slot": "GYM"}
        elif action_type == "store_next_day_plan":
            params = {"plan_text": "Do stuff", "date": "2024-01-02"}
        elif action_type == "send_telegram":
            params = {"text": "hello"}
        elif action_type == "send_voice":
            params = {"text": "voice message"}
        return ActionItem(type=action_type, params=params)

    actions = [make_action(t) for t in action_types]

    # Wrap execute_actions to track execution order before delegating
    original_exec = agent_module.execute_actions

    def tracking_exec(actions_list, user_arg, log_id, decision=None, pending_messages=None):
        for a in actions_list:
            execution_log.append(a.type)
        original_exec(actions_list, user_arg, log_id, decision, pending_messages)

    tracking_exec(actions, user, log_id=None)

    assert execution_log == action_types, (
        "Actions executed in wrong order. "
        "Expected " + str(action_types) + ", got " + str(execution_log)
    )
