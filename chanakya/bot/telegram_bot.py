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
from chanakya.db.mongo import get_user_with_defaults, interaction_logs, users
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
# Agent invocation
# ---------------------------------------------------------------------------

async def invoke_agent(
    user: dict,
    raw_input: str,
    interaction_type: str,
    media_url: str | None = None,
) -> LLMDecision | None:
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

    # Invoke agent
    try:
        llm_decision = await invoke_agent(
            user=user,
            raw_input=user_input,
            interaction_type="CHECKPOINT",
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

    # Update rolling conversation context (fire-and-forget)
    try:
        from chanakya.agent.context_assembler import update_conversation_context
        import asyncio
        asyncio.ensure_future(update_conversation_context(user, role="user", content=user_input, channel="text"))
        asyncio.ensure_future(update_conversation_context(user, role="assistant", content=llm_decision.response_text, channel="text"))
    except Exception:
        pass

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
        if tg_user.username and user.get("telegram_username") != tg_user.username:
            users.update_one({"_id": user["_id"]}, {"$set": {"telegram_username": tg_user.username}})

    await update.effective_message.reply_text(
        f"⚔️ Chanakya is watching, {user['name']}.\n\nJust talk naturally. No commands needed.",
        parse_mode="HTML",
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_id = str(update.effective_user.id)
    user_input = update.message.text or ""
    await _process_message(update, user_input, telegram_id)


async def perform_startup_audit(application: Application = None) -> None:
    """Proactive system check on boot. Waits 30s for server stability."""
    await _asyncio.sleep(30)
    try:
        user = users.find_one({"active": True})
        if not user or not user.get("telegram_id"):
            return

        telegram_id = user["telegram_id"]
        logger.info("🚀 Starting Guru's Awakening LITE Audit for %s", telegram_id)
        
        # LITE prompt: Skip heavy tools during boot spike
        lite_input = "SYSTEM: Perform a LITE startup temporal check. Do NOT call tools. Just check the time and scold if needed."
        
        response_text = await _asyncio.wait_for(
            generic_process_message(user, lite_input, channel="SYSTEM"),
            timeout=45.0
        )
        
        if application and response_text:
            await application.bot.send_message(chat_id=telegram_id, text=_md_to_html(response_text), parse_mode="HTML")
            logger.info("✅ Proactive startup alert sent.")
    except _asyncio.TimeoutError:
        logger.warning("Startup audit timed out. Chanakya will catch up in the next cycle.")
    except Exception as exc:
        logger.error("Startup audit failed: %s", exc)


def build_application() -> Application:
    app: Application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    return app
