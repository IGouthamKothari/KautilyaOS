"""
telegram_bot.py — Telegram bot handler.

Single entry point: every message (text, photo, voice, audio) goes to the LLM agent.
The agent decides what to do — no slash commands needed except /start.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

import httpx
import asyncio as _asyncio
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from chanakya.config import OPENAI_API_KEY, TELEGRAM_BOT_TOKEN
from chanakya.db.mongo import (
    get_user_with_defaults, interaction_logs, users,
    store_chat_message,
)
from chanakya.io_logger import log_input, log_output
from chanakya.models.llm_decision import LLMDecision

logger = logging.getLogger(__name__)

_UNREGISTERED_MSG = "You are not registered. Contact the administrator."

# ---------------------------------------------------------------------------
# Per-user processing lock — prevents parallel agent invocations for same user
# ---------------------------------------------------------------------------
_user_locks: dict[str, _asyncio.Lock] = {}


# ---------------------------------------------------------------------------
# Markdown → Telegram HTML converter
# ---------------------------------------------------------------------------

def _md_to_html(text: str) -> str:
    import html as _html
    import re as _re
    text = _re.sub(r"<br\s*/?>", "\n", text, flags=_re.IGNORECASE)
    text = _re.sub(r"<[^>]+>", "", text)
    text = _html.unescape(text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = _re.sub(r"^#{1,3}\s+(.+)$", r"<b>\1</b>", text, flags=_re.MULTILINE)
    text = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = _re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = _re.sub(r"\*([^*\n]+?)\*", r"<i>\1</i>", text)
    text = _re.sub(r"_([^_\n]+?)_", r"<i>\1</i>", text)
    text = _re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    text = _re.sub(r"^[\-\*]\s+", "• ", text, flags=_re.MULTILINE)
    text = _re.sub(
        r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b",
        lambda m: m.group(1).replace("_", " ").title(),
        text,
    )
    return text.strip()


async def _safe_reply(message, text: str) -> None:
    from telegram.error import BadRequest
    try:
        await message.reply_text(_md_to_html(text), parse_mode="HTML")
    except BadRequest as exc:
        if "parse" in str(exc).lower() or "entities" in str(exc).lower():
            import html as _html
            plain = re.sub(r"<[^>]+>", "", text)
            plain = _html.unescape(plain)
            await message.reply_text(plain)
        else:
            raise


# ---------------------------------------------------------------------------
# Interaction type detection
# ---------------------------------------------------------------------------


def _detect_interaction_type(user: dict) -> str:
    """Detect the appropriate interaction type based on context.

    If the user is responding to a recent scheduled checkpoint (within 30 min)
    that has NOT yet received a verdict, treat as CHECKPOINT so full tier2/tier3
    context is loaded.
    Otherwise, treat as MENTOR_TALK for natural conversation (tier1 only — faster).
    """
    from datetime import timedelta
    last_scheduled = interaction_logs.find_one(
        {
            "user_id": user["_id"],
            "trigger_type": "SCHEDULED",
            "timestamp": {"$gte": datetime.utcnow() - timedelta(minutes=30)},
            # Only treat as CHECKPOINT if no verdict has been issued yet
            "ai_evaluation.verdict": None,
        },
        sort=[("timestamp", -1)],
    )
    if last_scheduled:
        return "CHECKPOINT"
    return "MENTOR_TALK"


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------

async def invoke_agent(
    user: dict,
    raw_input: str,
    interaction_type: str,
    media_url: str | None = None,
) -> LLMDecision | None:
    from chanakya.config import DARBAR_ENABLED

    if DARBAR_ENABLED:
        from chanakya.darbar.orchestrator import process
        return await process(user, raw_input, interaction_type, media_url=media_url)

    from chanakya.agent.chanakya_agent import ChanakyaAgent
    agent = ChanakyaAgent(user)
    return await agent.invoke(raw_input, interaction_type, media_url=media_url)


# ---------------------------------------------------------------------------
# Generic Message Processor (Multi-Channel)
# ---------------------------------------------------------------------------

async def generic_process_message(
    user: dict, 
    user_input: str, 
    channel: str = "TELEGRAM", 
    media_url: str | None = None
) -> str:
    """Core logic shared by Telegram and Web UI."""
    telegram_id = user.get("telegram_id", "unknown")
    log_input(channel, telegram_id, user_input)

    # Insert pending interaction_log
    now = datetime.utcnow()
    log_doc = {
        "user_id": user["_id"],
        "timestamp": now,
        "trigger_type": "MANUAL",
        "channel": channel,
        "message_sent": "",
        "user_response": user_input,
        "ai_evaluation": {"verdict": None, "confidence": None, "reasoning": None},
        "created_at": now,
    }
    if media_url:
        log_doc["media_url"] = media_url

    log_id = None
    try:
        log_id = interaction_logs.insert_one(log_doc).inserted_id
    except Exception:
        logger.exception("Failed to insert interaction_log for user %s", telegram_id)

    # Detect interaction type from context
    interaction_type = _detect_interaction_type(user)

    # Invoke agent
    try:
        llm_decision = await invoke_agent(
            user=user,
            raw_input=user_input,
            interaction_type=interaction_type,
            media_url=media_url,
        )
    except Exception:
        logger.exception("Agent raised for user %s", telegram_id)
        llm_decision = None

    if llm_decision is None:
        return "Something went wrong. Chanakya's mind is clouded. Try again."

    # Update log
    if log_id is not None:
        try:
            # Task 1.1: Cancel pending nudges for the user's last unanswered checkpoint
            from chanakya.scheduler.task_runner import cancel_nudge
            last_pending = interaction_logs.find_one(
                {"user_id": user["_id"], "trigger_type": "SCHEDULED", "user_response": None},
                sort=[("timestamp", -1)]
            )
            if last_pending:
                cancel_nudge(last_pending["_id"])
                # Mark it as 'answered' by linking to this manual response
                interaction_logs.update_one({"_id": last_pending["_id"]}, {"$set": {"user_response": user_input}})

            interaction_logs.update_one(
                {"_id": log_id},
                {"$set": {
                    "message_sent": llm_decision.response_text,
                    "ai_evaluation": {
                        "verdict": llm_decision.verdict,
                        "confidence": None,
                        "reasoning": llm_decision.reasoning,
                    },
                }},
            )
        except Exception:
            logger.exception("Failed to update interaction_log %s", log_id)

    logger.info("📤 [%s] (%s) verdict=%s: %s", telegram_id, channel, llm_decision.verdict, llm_decision.response_text[:100])
    log_output(channel, telegram_id, llm_decision.response_text, verdict=llm_decision.verdict)

    # Store messages in chat history for future context
    try:
        store_chat_message(user["_id"], "user", user_input, channel.lower())
        store_chat_message(user["_id"], "assistant", llm_decision.response_text, channel.lower())
    except Exception:
        logger.debug("Failed to store chat messages for user %s", telegram_id)

    return llm_decision.response_text


# ---------------------------------------------------------------------------
# Telegram Handlers
# ---------------------------------------------------------------------------

async def _process_message(
    update: Update,
    user_input: str,
    telegram_id: str,
    media_url: str | None = None,
) -> None:
    """Telegram-specific wrapper with locking."""
    if telegram_id not in _user_locks:
        _user_locks[telegram_id] = _asyncio.Lock()
        
    async with _user_locks[telegram_id]:
        user = get_user_with_defaults(telegram_id)
        if user is None:
            await update.effective_message.reply_text(_UNREGISTERED_MSG)
            return

        response_text = await generic_process_message(user, user_input, channel="TELEGRAM", media_url=media_url)
        await _safe_reply(update.effective_message, response_text)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_id = str(update.effective_user.id)
    tg_user = update.effective_user
    logger.info("⚡ /start from %s (%s)", tg_user.full_name, telegram_id)

    user = get_user_with_defaults(telegram_id)
    if user is None:
        new_user = {
            "telegram_id": telegram_id,
            "telegram_username": tg_user.username or "",
            "name": tg_user.full_name or tg_user.username or "User",
            "phone": "",
            "elevenlabs_voice_id": "",
            "leetcode_username": "",
            "active": True,
            "current_mode": "NORMAL",
            "timezone": "Asia/Kolkata",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        users.insert_one(new_user)
        logger.info("✅ Auto-registered: %s (%s)", tg_user.full_name, telegram_id)
        user = get_user_with_defaults(telegram_id)
    else:
        if tg_user.username and user.get("telegram_username") != str(tg_user.username):
            users.update_one({"_id": user["_id"]}, {"$set": {"telegram_username": str(tg_user.username)}})

    await update.effective_message.reply_text(
        f"⚔️ Chanakya is watching, {user['name']}.\n\n"
        "Commands:\n"
        "/status — Streak, failures, and mode\n"
        "/war — Activate War Mode\n"
        "/peace — Deactivate War Mode\n"
        "/shield — Toggle AWAY mode\n"
        "/settodotime HH:MM — Set morning todo time\n"
        "/reloadtemplates — Reload prompt templates\n\n"
        "Or just talk naturally.",
        parse_mode="HTML",
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_id = str(update.effective_user.id)
    user = get_user_with_defaults(telegram_id)
    if user is None:
        await update.effective_message.reply_text(_UNREGISTERED_MSG)
        return
    streak = user.get("streak_count", 0)
    longest = user.get("longest_streak", 0)
    failures = user.get("failure_count_this_week", 0)
    mode = user.get("current_mode", "NORMAL")
    await update.effective_message.reply_text(
        f"<b>Status</b>\n\nStreak: {streak} days (Best: {longest})\n"
        f"Failures this week: {failures}\nMode: {mode}",
        parse_mode="HTML",
    )


async def cmd_peace(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_id = str(update.effective_user.id)
    user = get_user_with_defaults(telegram_id)
    if user is None:
        await update.effective_message.reply_text(_UNREGISTERED_MSG)
        return
    users.update_one(
        {"_id": user["_id"]},
        {"$set": {"current_mode": "NORMAL", "war_mode_expires": None}},
    )
    await update.effective_message.reply_text("War Mode deactivated. You are now in NORMAL mode.")


async def cmd_settodotime(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_id = str(update.effective_user.id)
    user = get_user_with_defaults(telegram_id)
    if user is None:
        await update.effective_message.reply_text(_UNREGISTERED_MSG)
        return
    text = update.message.text or ""
    parts = text.split()
    if len(parts) < 2 or not re.match(r"^\d{2}:\d{2}$", parts[1]):
        await update.effective_message.reply_text(
            f"Invalid format. Use /settodotime HH:MM (e.g. /settodotime 08:30)"
        )
        return
    time_str = parts[1]
    try:
        hour, minute = map(int, time_str.split(":"))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError
    except ValueError:
        await update.effective_message.reply_text(
            f"Invalid format. Use /settodotime HH:MM (e.g. /settodotime 08:30)"
        )
        return
    users.update_one({"_id": user["_id"]}, {"$set": {"morning_todo_time": time_str}})
    await update.effective_message.reply_text(f"✅ Morning todo time set to {time_str}.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_id = str(update.effective_user.id)
    user_input = update.message.text or ""
    await _process_message(update, user_input, telegram_id)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_id = str(update.effective_user.id)
    caption = update.message.caption or ""
    
    # Get the largest photo
    photo_file = await update.message.photo[-1].get_file()
    media_url = photo_file.file_path # This is a temporary public URL from Telegram
    
    logger.info("📸 Photo received from %s. Caption: %s", telegram_id, caption)
    await _process_message(update, f"[PHOTO] {caption or ''}", telegram_id, media_url=media_url)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_id = str(update.effective_user.id)
    voice = update.message.voice
    
    # Download voice file
    voice_file = await voice.get_file()
    
    # Transcribe using OpenAI Whisper
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        await voice_file.download_to_drive(tmp.name)
        tmp_path = tmp.name

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        user_input = transcript.text
        logger.info("🎙️ Voice transcribed for %s: %s", telegram_id, user_input)
        await _process_message(update, user_input, telegram_id)
    except Exception as e:
        logger.error("Failed to transcribe voice for %s: %s", telegram_id, e)
        await update.message.reply_text("I heard your voice, but I couldn't understand it. Try text.")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)




def build_application() -> Application:
    app: Application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("peace", cmd_peace))
    app.add_handler(CommandHandler("settodotime", cmd_settodotime))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    return app
