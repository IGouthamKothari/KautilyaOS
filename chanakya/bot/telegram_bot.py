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
# One message at a time per user; eliminates duplicate streak increments.
# ---------------------------------------------------------------------------
import asyncio as _asyncio
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
    # Convert CAPS_UNDERSCORE identifiers to readable Title Case
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
# Image analysis — GPT-4o vision
# ---------------------------------------------------------------------------

async def _analyse_image(image_url: str, caption: str = "") -> str:
    """Send image to GPT-4o vision and return a plain-text description.

    Used for gym proof photos, food logs, screenshot submissions, etc.
    Returns a description string the agent can reason about.
    Falls back to a plain note on any failure.
    """
    context_hint = f'The user sent this image with caption: "{caption}".' if caption else "The user sent this image."
    prompt = (
        f"{context_hint} "
        "Describe what you see concisely and factually in 2-4 sentences. "
        "Focus on: what activity is shown, whether it looks genuine/complete, "
        "any visible details relevant to fitness, work, or accountability. "
        "Do not add opinions — just describe what is visible."
    )

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-5.4-nano-2026-03-17",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
                            ],
                        }
                    ],
                    "max_completion_tokens": 200,
                },
            )
            resp.raise_for_status()
            description = resp.json()["choices"][0]["message"]["content"].strip()
            logger.info("Image analysed: %s", description[:100])
            return description
    except Exception as exc:
        logger.warning("Image analysis failed: %s", exc)
        return f"[Image received — analysis unavailable: {exc}]"


# ---------------------------------------------------------------------------
# Audio transcription — Whisper
# ---------------------------------------------------------------------------

async def _transcribe_audio(file_bytes: bytes, filename: str = "audio.ogg") -> str:
    """Transcribe audio bytes via OpenAI Whisper API.

    Accepts any format Telegram sends (ogg/opus for voice, mp3/m4a for audio files).
    Returns transcribed text, or an error string on failure.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                files={"file": (filename, file_bytes, "audio/ogg")},
                data={"model": "gpt-4o-mini-transcribe", "language": "en"},
            )
            resp.raise_for_status()
            text = resp.json().get("text", "").strip()
            logger.info("Audio transcribed (%d bytes): %s", len(file_bytes), text[:100])
            return text
    except Exception as exc:
        logger.warning("Audio transcription failed: %s", exc)
        return f"[Audio received — transcription failed: {exc}]"


async def _download_telegram_file(bot, file_id: str) -> bytes:
    """Download a Telegram file by file_id and return raw bytes."""
    tg_file = await bot.get_file(file_id)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(tg_file.file_path)
        resp.raise_for_status()
        return resp.content


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
# Error handler
# ---------------------------------------------------------------------------

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception: %s", update, exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Something went wrong. Chanakya will be back shortly."
            )
        except Exception:
            logger.exception("Failed to send error reply.")


# ---------------------------------------------------------------------------
# /start — only slash command kept (needed for registration)
# ---------------------------------------------------------------------------

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
        # Keep username fresh
        if tg_user.username and user.get("telegram_username") != tg_user.username:
            users.update_one({"_id": user["_id"]}, {"$set": {"telegram_username": tg_user.username}})

    await update.effective_message.reply_text(
        f"⚔️ Chanakya is watching, {user['name']}.\n\n"
        "Just talk to me naturally. I'll handle everything:\n\n"
        "• <i>\"show my schedule\"</i>\n"
        "• <i>\"activate war mode\"</i>\n"
        "• <i>\"my number is +91XXXXXXXXXX\"</i>\n"
        "• <i>\"save mom's number +919876543210\"</i>\n"
        "• <i>\"call mom about dinner\"</i>\n"
        "• <i>\"call me\"</i>\n"
        "• <i>\"add mindset note: discipline over comfort\"</i>\n"
        "• <i>\"show my mindset notes\"</i>\n"
        "• <i>\"set my morning todo time to 8:30\"</i>\n"
        "• <i>\"what's my streak\"</i>\n\n"
        "No commands needed. Just send a message.",
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# Core message handler — routes everything through the agent
# ---------------------------------------------------------------------------

async def _process_message(
    update: Update,
    user_input: str,
    telegram_id: str,
    media_url: str | None = None,
) -> None:
    """Fetch user, log input, invoke agent, log output, reply."""
    # Acquire per-user lock — one message processed at a time per user
    if telegram_id not in _user_locks:
        _user_locks[telegram_id] = _asyncio.Lock()
    async with _user_locks[telegram_id]:
        await _process_message_inner(update, user_input, telegram_id, media_url)


async def _process_message_inner(
    update: Update,
    user_input: str,
    telegram_id: str,
    media_url: str | None = None,
) -> None:
    """Core message processing — always runs under the per-user lock."""
    user = get_user_with_defaults(telegram_id)
    if user is None:
        await update.effective_message.reply_text(_UNREGISTERED_MSG)
        return

    log_input("TELEGRAM", telegram_id, user_input)

    # Insert pending interaction_log
    now = datetime.utcnow()
    log_doc = {
        "user_id": user["_id"],
        "timestamp": now,
        "trigger_type": "MANUAL",
        "channel": "TELEGRAM",
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
        await update.effective_message.reply_text(
            "Something went wrong. Chanakya will be back shortly."
        )
        return

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

    logger.info("📤 [%s] verdict=%s: %s", telegram_id, llm_decision.verdict, llm_decision.response_text[:100])
    log_output("TELEGRAM", telegram_id, llm_decision.response_text, verdict=llm_decision.verdict)
    await _safe_reply(update.effective_message, llm_decision.response_text)

    # Update rolling conversation context (fire-and-forget)
    try:
        from chanakya.agent.context_assembler import update_conversation_context
        import asyncio
        asyncio.ensure_future(update_conversation_context(user, role="user", content=user_input, channel="text"))
        asyncio.ensure_future(update_conversation_context(user, role="assistant", content=llm_decision.response_text, channel="text"))
    except Exception:
        pass


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_id = str(update.effective_user.id)
    user_input = update.message.text or ""
    logger.info("📨 [%s] TEXT: %s", telegram_id, user_input[:100])
    await _process_message(update, user_input, telegram_id)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_id = str(update.effective_user.id)
    caption = update.message.caption or ""
    logger.info("📷 [%s] PHOTO caption: %s", telegram_id, caption[:100])

    # Get highest-res photo
    photo = update.message.photo[-1]
    tg_file = await context.bot.get_file(photo.file_id)
    image_url = tg_file.file_path  # Telegram CDN URL — valid for ~1 hour

    # Analyse image with GPT-4o vision before sending to agent
    description = await _analyse_image(image_url, caption)
    user_input = f"[PHOTO] {caption}\n[Image analysis: {description}]"

    logger.info("📷 [%s] Image analysed: %s", telegram_id, description[:100])
    await _process_message(update, user_input, telegram_id, media_url=image_url)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Telegram voice messages — transcribe via Whisper then process as text."""
    telegram_id = str(update.effective_user.id)
    logger.info("🎤 [%s] VOICE message received", telegram_id)

    try:
        file_bytes = await _download_telegram_file(context.bot, update.message.voice.file_id)
        transcript = await _transcribe_audio(file_bytes, filename="voice.ogg")
    except Exception as exc:
        logger.error("Voice handling failed for %s: %s", telegram_id, exc)
        await update.effective_message.reply_text("Couldn't process your voice message. Try again.")
        return

    if not transcript or transcript.startswith("[Audio received"):
        await update.effective_message.reply_text(
            "Couldn't transcribe that. Speak clearly or send a text message."
        )
        return

    logger.info("🎤 [%s] Transcribed: %s", telegram_id, transcript[:100])
    await _process_message(update, transcript, telegram_id)


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle audio file uploads (mp3, m4a, etc.) — transcribe via Whisper."""
    telegram_id = str(update.effective_user.id)
    audio = update.message.audio or update.message.document
    if not audio:
        return

    filename = getattr(audio, "file_name", None) or "audio.mp3"
    logger.info("🎵 [%s] AUDIO file: %s", telegram_id, filename)

    try:
        file_bytes = await _download_telegram_file(context.bot, audio.file_id)
        transcript = await _transcribe_audio(file_bytes, filename=filename)
    except Exception as exc:
        logger.error("Audio handling failed for %s: %s", telegram_id, exc)
        await update.effective_message.reply_text("Couldn't process that audio file.")
        return

    if not transcript or transcript.startswith("[Audio received"):
        await update.effective_message.reply_text(
            "Couldn't transcribe that audio. Try a clearer recording."
        )
        return

    logger.info("🎵 [%s] Transcribed: %s", telegram_id, transcript[:100])
    await _process_message(update, transcript, telegram_id)


# ---------------------------------------------------------------------------
# Application setup — /start + text + photo + voice + audio
# ---------------------------------------------------------------------------

def build_application() -> Application:
    app: Application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.AUDIO, handle_audio))
    # Audio files sent as documents (e.g. mp3 forwarded)
    app.add_handler(MessageHandler(
        filters.Document.MimeType("audio/mpeg")
        | filters.Document.MimeType("audio/mp4")
        | filters.Document.MimeType("audio/ogg")
        | filters.Document.MimeType("audio/wav")
        | filters.Document.MimeType("audio/x-m4a"),
        handle_audio,
    ))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_error_handler(error_handler)
    return app


async def perform_startup_audit(application: Application = None) -> None:
    """Proactive system check on boot. Scolds the user if they are awake at an unholy hour."""
    from chanakya.db.mongo import users
    from chanakya.bot.telegram_bot import invoke_agent
    
    # 1. Find the primary active user
    user = users.find_one({"active": True})
    if not user:
        logger.info("Startup Audit: No active user found. Skipping.")
        return
        
    telegram_id = user.get("telegram_id")
    if not telegram_id:
        logger.info("Startup Audit: Active user has no telegram_id. Skipping.")
        return

    logger.info("🚀 Starting Guru's Awakening Audit for user %s (%s)", user.get("name"), telegram_id)
    
    # 2. Invoke Agent with a system trigger
    try:
        # We use a dummy input that triggers the agent's temporal awareness rules
        llm_decision = await invoke_agent(
            user=user,
            raw_input="SYSTEM: Perform startup diagnostic and temporal check.",
            interaction_type="CHECKPOINT"
        )
        
        if llm_decision and llm_decision.response_text:
            # 3. If the agent decided to say something (likely a warning about the time), push it.
            if application:
                bot = application.bot
                try:
                    # Clean the response text (remove markdown for HTML)
                    from chanakya.bot.telegram_bot import _md_to_html
                    html_text = _md_to_html(llm_decision.response_text)
                    
                    await bot.send_message(
                        chat_id=telegram_id,
                        text=html_text,
                        parse_mode="HTML"
                    )
                    logger.info("✅ Proactive startup alert sent to %s", telegram_id)
                    
                    # Update context so he remembers this scolding
                    from chanakya.agent.context_assembler import update_conversation_context
                    await update_conversation_context(user, role="assistant", content=llm_decision.response_text, channel="text")
                    
                except Exception as e:
                    logger.error("Failed to send startup alert: %s", e)
    except Exception as exc:
        logger.error("Startup audit failed: %s", exc)
