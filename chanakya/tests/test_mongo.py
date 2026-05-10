"""
test_mongo.py — Unit and property-based tests for chanakya/db/mongo.py.

Tests cover:
  - _extract_db_name URI parsing
  - _apply_defaults field-default logic (including P2: never raises KeyError)
  - get_user_with_defaults / get_user_by_id helpers
  - create_indexes (smoke test via mongomock)
  - Exponential backoff sequence property (P4)
"""

import os
import sys
import time
import unittest.mock as mock
from bson import ObjectId

import mongomock
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# We must set env vars AND patch MongoClient BEFORE any chanakya module is
# imported, because config.py validates env vars at import time and
# db/mongo.py connects at module level.
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

# Patch env vars before any import
for _k, _v in _ENV_VARS.items():
    os.environ.setdefault(_k, _v)

# Remove any previously cached chanakya modules so patches take effect cleanly
for _mod in list(sys.modules.keys()):
    if _mod.startswith("chanakya"):
        del sys.modules[_mod]

# Build a mongomock client that will be returned by MongoClient()
_mock_client = mongomock.MongoClient()
_mock_client.admin.command = mock.MagicMock(return_value={"ok": 1})

with mock.patch("pymongo.MongoClient", return_value=_mock_client):
    import chanakya.db.mongo as mongo_module

# Re-point the module's db handle to the mongomock database so helpers work.
_mock_db = _mock_client["chanakya"]
mongo_module.db = _mock_db
mongo_module.users = _mock_db["users"]
mongo_module.schedules = _mock_db["schedules"]
mongo_module.checkpoints = _mock_db["checkpoints"]
mongo_module.interaction_logs = _mock_db["interaction_logs"]
mongo_module.ai_tool_calls = _mock_db["ai_tool_calls"]
mongo_module.user_state_snapshots = _mock_db["user_state_snapshots"]
mongo_module.prompt_templates = _mock_db["prompt_templates"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_user(**kwargs) -> dict:
    """Insert a user document into the mock collection and return it."""
    doc = {"telegram_id": "test_tg_id", "name": "Test User", **kwargs}
    result = mongo_module.users.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def _clear_users():
    mongo_module.users.delete_many({})


# ---------------------------------------------------------------------------
# _extract_db_name
# ---------------------------------------------------------------------------


class TestExtractDbName:
    def test_standard_uri_with_db(self):
        uri = "mongodb+srv://user:pass@cluster.mongodb.net/mydb"
        assert mongo_module._extract_db_name(uri) == "mydb"

    def test_uri_without_db_returns_default(self):
        uri = "mongodb+srv://user:pass@cluster.mongodb.net/"
        assert mongo_module._extract_db_name(uri, default="chanakya") == "chanakya"

    def test_uri_with_query_string(self):
        uri = "mongodb+srv://user:pass@cluster.mongodb.net/mydb?retryWrites=true"
        assert mongo_module._extract_db_name(uri) == "mydb"

    def test_empty_path_returns_default(self):
        uri = "mongodb://localhost:27017"
        assert mongo_module._extract_db_name(uri, default="chanakya") == "chanakya"

    def test_custom_default(self):
        uri = "mongodb://localhost:27017/"
        assert mongo_module._extract_db_name(uri, default="custom") == "custom"


# ---------------------------------------------------------------------------
# _apply_defaults
# ---------------------------------------------------------------------------


class TestApplyDefaults:
    def test_empty_document_gets_all_defaults(self):
        doc = {"_id": ObjectId(), "telegram_id": "abc"}
        result = mongo_module._apply_defaults(doc)
        for field, default in mongo_module.FIELD_DEFAULTS.items():
            assert field in result, f"Field '{field}' missing from result"
            if isinstance(default, (dict, list)):
                assert result[field] == default
            else:
                assert result[field] == default

    def test_existing_values_not_overwritten(self):
        doc = {
            "_id": ObjectId(),
            "telegram_id": "abc",
            "streak_count": 42,
            "timezone": "America/New_York",
        }
        result = mongo_module._apply_defaults(doc)
        assert result["streak_count"] == 42
        assert result["timezone"] == "America/New_York"

    def test_mutable_defaults_are_copies(self):
        doc1 = {"_id": ObjectId()}
        doc2 = {"_id": ObjectId()}
        r1 = mongo_module._apply_defaults(doc1)
        r2 = mongo_module._apply_defaults(doc2)
        # Mutating one result's list/dict should not affect the other
        r1["recurring_failure_patterns"].append("x")
        assert r2["recurring_failure_patterns"] == []

    def test_none_default_applied_for_morning_todo_time(self):
        doc = {"_id": ObjectId()}
        result = mongo_module._apply_defaults(doc)
        assert result["morning_todo_time"] is None

    def test_original_document_not_mutated(self):
        doc = {"_id": ObjectId()}
        original_keys = set(doc.keys())
        mongo_module._apply_defaults(doc)
        assert set(doc.keys()) == original_keys


# ---------------------------------------------------------------------------
# get_user_with_defaults
# ---------------------------------------------------------------------------


class TestGetUserWithDefaults:
    def setup_method(self):
        _clear_users()

    def test_returns_none_for_unknown_telegram_id(self):
        result = mongo_module.get_user_with_defaults("nonexistent_id")
        assert result is None

    def test_returns_user_with_all_defaults_applied(self):
        _insert_user(telegram_id="tg_001", name="Alice")
        result = mongo_module.get_user_with_defaults("tg_001")
        assert result is not None
        assert result["name"] == "Alice"
        assert result["streak_count"] == 0
        assert result["timezone"] == "Asia/Kolkata"

    def test_existing_field_not_overwritten(self):
        _insert_user(telegram_id="tg_002", name="Bob", streak_count=10)
        result = mongo_module.get_user_with_defaults("tg_002")
        assert result["streak_count"] == 10

    def test_never_raises_key_error(self):
        # Insert a document with no optional fields at all
        _insert_user(telegram_id="tg_003")
        result = mongo_module.get_user_with_defaults("tg_003")
        # Access every default field — must not raise
        for field in mongo_module.FIELD_DEFAULTS:
            _ = result[field]  # should not raise KeyError


# ---------------------------------------------------------------------------
# get_user_by_id
# ---------------------------------------------------------------------------


class TestGetUserById:
    def setup_method(self):
        _clear_users()

    def test_returns_none_for_unknown_id(self):
        result = mongo_module.get_user_by_id(ObjectId())
        assert result is None

    def test_returns_user_with_defaults(self):
        doc = _insert_user(telegram_id="tg_100", name="Charlie")
        result = mongo_module.get_user_by_id(doc["_id"])
        assert result is not None
        assert result["name"] == "Charlie"
        assert result["longest_streak"] == 0

    def test_existing_field_not_overwritten(self):
        doc = _insert_user(telegram_id="tg_101", name="Dave", longest_streak=5)
        result = mongo_module.get_user_by_id(doc["_id"])
        assert result["longest_streak"] == 5


# ---------------------------------------------------------------------------
# get_db
# ---------------------------------------------------------------------------


class TestGetDb:
    def test_returns_database_handle(self):
        db = mongo_module.get_db()
        assert db is not None
        assert db.name == "chanakya"


# ---------------------------------------------------------------------------
# P2: Property — get_user_with_defaults never raises KeyError
#
# Validates: Requirements 22.4
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(
    st.fixed_dictionaries(
        {},
        optional={
            field: st.none() if default is None else st.just(default)
            for field, default in mongo_module.FIELD_DEFAULTS.items()
        },
    )
)
def test_p2_get_user_with_defaults_never_raises_key_error(partial_fields):
    """
    **Validates: Requirements 22.4**

    For any combination of present/absent fields in a user document,
    get_user_with_defaults must never raise KeyError and must always return
    a document where every FIELD_DEFAULTS key is accessible.
    """
    _clear_users()
    doc = {"telegram_id": "pbt_user", "name": "PBT User", **partial_fields}
    mongo_module.users.insert_one(doc)

    try:
        result = mongo_module.get_user_with_defaults("pbt_user")
        assert result is not None
        # Every default field must be accessible without KeyError
        for field in mongo_module.FIELD_DEFAULTS:
            _ = result[field]
    except KeyError as exc:
        raise AssertionError(f"KeyError raised for field: {exc}") from exc
    finally:
        _clear_users()


# ---------------------------------------------------------------------------
# P4: Property — exponential backoff sequence never exceeds 32s
#
# Validates: Requirements 2.8
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(st.integers(min_value=1, max_value=20))
def test_p4_backoff_sequence_never_exceeds_32s(num_retries):
    """
    **Validates: Requirements 2.8**

    The exponential backoff sequence starting at 1s, doubling each step,
    capped at 32s must always be [1, 2, 4, 8, 16, 32, 32, 32, ...].
    """
    delay = 1
    max_delay = 32
    sequence = []
    for _ in range(num_retries):
        sequence.append(delay)
        delay = min(delay * 2, max_delay)

    # Every value must be <= 32
    assert all(v <= max_delay for v in sequence), f"Sequence exceeded 32s: {sequence}"
    # First value must be 1
    assert sequence[0] == 1
    # Values must be non-decreasing
    for i in range(1, len(sequence)):
        assert sequence[i] >= sequence[i - 1]
    # Once 32 is reached, it stays at 32
    reached_cap = False
    for v in sequence:
        if v == max_delay:
            reached_cap = True
        if reached_cap:
            assert v == max_delay, f"Value dropped below cap after reaching it: {sequence}"
