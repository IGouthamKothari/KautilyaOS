"""
test_telegram_bot.py — Async unit tests for bot/telegram_bot.py.

Uses pytest-asyncio for async test execution, mongomock for MongoDB, and
unittest.mock for Telegram bot methods.

Test coverage:
  1.  /start with unregistered user → unregistered message, no DB write
  2.  /start with registered user → welcome message with command list
  3.  /status with registered user → streak/failure/mode formatted reply
  4.  /peace → sets current_mode=NORMAL in DB, confirmation reply
  5.  /settodotime 08:30 → valid time, updates DB, confirmation reply
  6.  /settodotime badtime → invalid format, error reply, no DB write
  7.  handle_text with registered user → inserts interaction_log, replies with agent response_text
  8.  handle_photo with no caption → caption treated as empty string, log inserted with media_url
  9.  Any handler with unregistered user → unregistered message, no further processing
  10. Agent returns None → generic error reply, no crash
"""

from __future__ import annotations

import os
import sys
import unittest.mock as mock
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import mongomock
import pytest
from bson import ObjectId

# ---------------------------------------------------------------------------
# Set env vars and patch MongoClient BEFORE any chanakya module is imported.
# This mirrors the pattern used in test_mongo.py.
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

# Remove any previously cached chanakya modules so patches take effect cleanly.
for _mod in list(sys.modules.keys()):
    if _mod.startswith("chanakya"):
        del sys.modules[_mod]

# Build a mongomock client that will be returned by MongoClient().
_mock_mongo_client = mongomock.MongoClient()
_mock_mongo_client.admin.command = mock.MagicMock(return_value={"ok": 1})

with mock.patch("pymongo.MongoClient", return_value=_mock_mongo_client):
    import chanakya.db.mongo as mongo_module
    import chanakya.bot.telegram_bot as bot_module

# Re-point all module-level collection handles to the mongomock database.
_mock_db = _mock_mongo_client["chanakya"]
mongo_module.db = _mock_db
mongo_module.users = _mock_db["users"]
mongo_module.interaction_logs = _mock_db["interaction_logs"]
bot_module.users = _mock_db["users"]
bot_module.interaction_logs = _mock_db["interaction_logs"]

# ---------------------------------------------------------------------------
# Helpers to build fake Telegram Update / Context objects
# ---------------------------------------------------------------------------


def _make_user_obj(telegram_id: str = "123456") -> MagicMock:
    user = MagicMock()
    user.id = int(telegram_id)
    return user


def _make_message(
    text: str | None = None,
    photo=None,
    caption: str | None = None,
) -> MagicMock:
    msg = MagicMock()
    msg.text = text
    msg.caption = caption
    msg.photo = photo
    msg.reply_text = AsyncMock()
    return msg


def _make_update(
    text: str | None = None,
    photo=None,
    caption: str | None = None,
    telegram_id: str = "123456",
) -> MagicMock:
    update = MagicMock()
    update.effective_user = _make_user_obj(telegram_id)
    msg = _make_message(text=text, photo=photo, caption=caption)
    update.effective_message = msg
    update.message = msg
    return update


def _make_context(bot: MagicMock | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.error = None
    ctx.bot = bot or MagicMock()
    return ctx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_collections():
    """Clear all collections before each test to ensure isolation."""
    _mock_db["users"].delete_many({})
    _mock_db["interaction_logs"].delete_many({})
    yield
    _mock_db["users"].delete_many({})
    _mock_db["interaction_logs"].delete_many({})


@pytest.fixture()
def registered_user() -> dict:
    """Insert and return a minimal registered user document."""
    doc = {
        "_id": ObjectId(),
        "telegram_id": "123456",
        "name": "Arjun",
        "phone": "+919999999999",
        "elevenlabs_voice_id": "voice_abc",
        "current_mode": "NORMAL",
        "streak_count": 5,
        "longest_streak": 10,
        "failure_count_this_week": 2,
        "timezone": "Asia/Kolkata",
        "active": True,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    _mock_db["users"].insert_one(doc)
    return doc


# ---------------------------------------------------------------------------
# Helper: patch get_user_with_defaults to use the mock DB
# ---------------------------------------------------------------------------


def _patch_get_user():
    """
    Return a context manager that patches get_user_with_defaults to query the
    mongomock users collection.
    """
    def _get_user(telegram_id: str):
        doc = _mock_db["users"].find_one({"telegram_id": telegram_id})
        if doc is None:
            return None
        # Apply minimal defaults so the handler doesn't KeyError.
        doc.setdefault("streak_count", 0)
        doc.setdefault("longest_streak", 0)
        doc.setdefault("current_mode", "NORMAL")
        doc.setdefault("timezone", "Asia/Kolkata")
        doc.setdefault("failure_count_this_week", 0)
        return doc

    return patch("chanakya.bot.telegram_bot.get_user_with_defaults", side_effect=_get_user)


# ---------------------------------------------------------------------------
# Test 1 — /start with unregistered user
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_unregistered_user():
    """Unregistered user gets auto-registered and receives the welcome message."""
    update = _make_update(text="/start", telegram_id="999999")
    ctx = _make_context()

    call_count = [0]
    minimal_user = {
        "_id": ObjectId(),
        "telegram_id": "999999",
        "name": "New User",
        "streak_count": 0,
        "longest_streak": 0,
        "current_mode": "NORMAL",
        "timezone": "Asia/Kolkata",
    }

    def _get_user_side_effect(telegram_id):
        call_count[0] += 1
        if call_count[0] == 1:
            return None  # first call: not found → triggers auto-register
        return minimal_user  # second call: newly registered user

    mock_users = MagicMock()
    mock_users.insert_one = MagicMock(return_value=MagicMock(inserted_id=minimal_user["_id"]))

    with patch("chanakya.bot.telegram_bot.get_user_with_defaults", side_effect=_get_user_side_effect):
        with patch.object(bot_module, "users", mock_users):
            await bot_module.cmd_start(update, ctx)

    update.effective_message.reply_text.assert_called_once()
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "Chanakya is watching" in reply_text or "/status" in reply_text
    mock_users.insert_one.assert_called_once()


# ---------------------------------------------------------------------------
# Test 2 — /start with registered user
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_registered_user(registered_user):
    """Registered user gets the welcome message with the full command list."""
    update = _make_update(text="/start")
    ctx = _make_context()

    with _patch_get_user():
        await bot_module.cmd_start(update, ctx)

    update.effective_message.reply_text.assert_called_once()
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "⚔️ Chanakya is watching" in reply_text
    assert "/status" in reply_text
    assert "/war" in reply_text
    assert "/peace" in reply_text
    assert "/shield" in reply_text
    assert "/settodotime" in reply_text
    assert "/reloadtemplates" in reply_text


# ---------------------------------------------------------------------------
# Test 3 — /status with registered user
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_registered_user(registered_user):
    """Status reply contains streak, failure count, and mode."""
    update = _make_update(text="/status")
    ctx = _make_context()

    with _patch_get_user():
        await bot_module.cmd_status(update, ctx)

    update.effective_message.reply_text.assert_called_once()
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "Streak:" in reply_text
    assert "Failures this week:" in reply_text
    assert "Mode:" in reply_text
    # Streak values from the registered_user fixture.
    assert "5" in reply_text   # streak_count
    assert "10" in reply_text  # longest_streak
    assert "NORMAL" in reply_text


# ---------------------------------------------------------------------------
# Test 4 — /peace sets current_mode=NORMAL in DB
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_peace_updates_db(registered_user):
    """
    /peace updates current_mode to NORMAL in the DB and sends a confirmation.
    """
    # Put user in WAR_MODE first.
    _mock_db["users"].update_one(
        {"_id": registered_user["_id"]},
        {"$set": {"current_mode": "WAR_MODE", "war_mode_expires": datetime.utcnow()}},
    )

    update = _make_update(text="/peace")
    ctx = _make_context()

    with _patch_get_user():
        await bot_module.cmd_peace(update, ctx)

    # DB should now have NORMAL mode.
    updated = _mock_db["users"].find_one({"_id": registered_user["_id"]})
    assert updated["current_mode"] == "NORMAL"
    assert updated["war_mode_expires"] is None

    update.effective_message.reply_text.assert_called_once()
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "deactivated" in reply_text.lower() or "War Mode" in reply_text


# ---------------------------------------------------------------------------
# Test 5 — /settodotime 08:30 valid
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_settodotime_valid(registered_user):
    """/settodotime 08:30 updates morning_todo_time in DB and confirms."""
    update = _make_update(text="/settodotime 08:30")
    ctx = _make_context()

    with _patch_get_user():
        await bot_module.cmd_settodotime(update, ctx)

    updated = _mock_db["users"].find_one({"_id": registered_user["_id"]})
    assert updated["morning_todo_time"] == "08:30"

    update.effective_message.reply_text.assert_called_once()
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "08:30" in reply_text
    assert "✅" in reply_text


# ---------------------------------------------------------------------------
# Test 6 — /settodotime badtime invalid
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_settodotime_invalid(registered_user):
    """/settodotime with invalid format replies with error; no DB write."""
    update = _make_update(text="/settodotime badtime")
    ctx = _make_context()

    with _patch_get_user():
        await bot_module.cmd_settodotime(update, ctx)

    # morning_todo_time should NOT have been written.
    updated = _mock_db["users"].find_one({"_id": registered_user["_id"]})
    assert updated.get("morning_todo_time") is None

    update.effective_message.reply_text.assert_called_once()
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "Invalid format" in reply_text or "invalid" in reply_text.lower()


# ---------------------------------------------------------------------------
# Test 7 — handle_text inserts interaction_log and replies with agent text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_text_inserts_log_and_replies(registered_user):
    """
    handle_text inserts an interaction_log document and replies with the
    agent's response_text.
    """
    update = _make_update(text="I finished my workout")
    ctx = _make_context()

    stub_decision = MagicMock()
    stub_decision.response_text = "Great job on the workout!"
    stub_decision.verdict = "SUCCESS"
    stub_decision.reasoning = "User confirmed gym session."

    with (
        _patch_get_user(),
        patch("chanakya.bot.telegram_bot.invoke_agent", new=AsyncMock(return_value=stub_decision)),
    ):
        await bot_module.handle_text(update, ctx)

    # One interaction_log should have been inserted.
    assert _mock_db["interaction_logs"].count_documents({}) == 1
    log = _mock_db["interaction_logs"].find_one({})
    assert log["user_response"] == "I finished my workout"
    assert log["trigger_type"] == "MANUAL"
    assert log["channel"] == "TELEGRAM"

    # Reply should contain the agent's response_text.
    update.effective_message.reply_text.assert_called_once()
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert reply_text == "Great job on the workout!"


# ---------------------------------------------------------------------------
# Test 8 — handle_photo with no caption
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_photo_no_caption(registered_user):
    """
    handle_photo with no caption treats caption as empty string and inserts
    an interaction_log with media_url set.
    """
    # Build a fake photo object.
    fake_photo = MagicMock()
    fake_photo.file_id = "file_abc123"

    fake_file = MagicMock()
    fake_file.file_path = "https://api.telegram.org/file/bot.../photo.jpg"
    fake_photo.get_file = AsyncMock(return_value=fake_file)

    bot = MagicMock()
    bot.get_file = AsyncMock(return_value=fake_file)

    update = _make_update(photo=[fake_photo], caption=None)
    ctx = _make_context(bot=bot)

    stub_decision = MagicMock()
    stub_decision.response_text = "Photo received."
    stub_decision.verdict = None
    stub_decision.reasoning = "stub"

    with (
        _patch_get_user(),
        patch("chanakya.bot.telegram_bot.invoke_agent", new=AsyncMock(return_value=stub_decision)),
    ):
        await bot_module.handle_photo(update, ctx)

    assert _mock_db["interaction_logs"].count_documents({}) == 1
    log = _mock_db["interaction_logs"].find_one({})
    assert log["media_url"] == "https://api.telegram.org/file/bot.../photo.jpg"
    # Caption is empty → user_response should be "[PHOTO] " (empty caption)
    assert log["user_response"] == "[PHOTO] "

    update.effective_message.reply_text.assert_called_once()


# ---------------------------------------------------------------------------
# Test 9 — Unregistered user in any handler → unregistered message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unregistered_user_in_status():
    """
    /status with an unregistered user replies with the unregistered message
    and does not query interaction_logs.
    """
    update = _make_update(text="/status", telegram_id="999999")
    ctx = _make_context()

    with _patch_get_user():
        await bot_module.cmd_status(update, ctx)

    update.effective_message.reply_text.assert_called_once()
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "not registered" in reply_text.lower()
    assert _mock_db["interaction_logs"].count_documents({}) == 0


@pytest.mark.asyncio
async def test_unregistered_user_in_handle_text():
    """
    handle_text with an unregistered user replies with the unregistered
    message and does not insert an interaction_log.
    """
    update = _make_update(text="hello", telegram_id="999999")
    ctx = _make_context()

    with _patch_get_user():
        await bot_module.handle_text(update, ctx)

    update.effective_message.reply_text.assert_called_once()
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "not registered" in reply_text.lower()
    assert _mock_db["interaction_logs"].count_documents({}) == 0


# ---------------------------------------------------------------------------
# Test 10 — Agent returns None → generic error reply, no crash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_text_agent_returns_none(registered_user):
    """
    When invoke_agent returns None, handle_text replies with a generic error
    message and does not crash.
    """
    update = _make_update(text="some message")
    ctx = _make_context()

    with (
        _patch_get_user(),
        patch("chanakya.bot.telegram_bot.invoke_agent", new=AsyncMock(return_value=None)),
    ):
        await bot_module.handle_text(update, ctx)

    update.effective_message.reply_text.assert_called_once()
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "something went wrong" in reply_text.lower() or "back shortly" in reply_text.lower()


@pytest.mark.asyncio
async def test_handle_photo_agent_returns_none(registered_user):
    """
    When invoke_agent returns None for a photo, handle_photo replies with a
    generic error message and does not crash.
    """
    fake_photo = MagicMock()
    fake_photo.file_id = "file_xyz"

    fake_file = MagicMock()
    fake_file.file_path = "https://api.telegram.org/file/bot.../photo2.jpg"
    fake_photo.get_file = AsyncMock(return_value=fake_file)

    bot = MagicMock()
    bot.get_file = AsyncMock(return_value=fake_file)

    update = _make_update(photo=[fake_photo], caption="test caption")
    ctx = _make_context(bot=bot)

    with (
        _patch_get_user(),
        patch("chanakya.bot.telegram_bot.invoke_agent", new=AsyncMock(return_value=None)),
    ):
        await bot_module.handle_photo(update, ctx)

    update.effective_message.reply_text.assert_called_once()
    reply_text = update.effective_message.reply_text.call_args[0][0]
    assert "something went wrong" in reply_text.lower() or "back shortly" in reply_text.lower()
