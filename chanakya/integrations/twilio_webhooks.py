"""
twilio_webhooks.py — FastAPI router for Twilio call status and TwiML endpoints.

Endpoints:
  POST /twilio/status                      — Twilio call status callback
  GET  /twilio/twiml/{log_id}              — TwiML <Play> for scheduled one-way checkpoint calls
  GET  /twilio/audio/{session_id}          — Serve ElevenLabs opening audio
  GET  /twilio/audio/turn/{session_id}     — Serve ElevenLabs mid-call turn audio
  POST /twilio/voice/{session_id}          — TwiML entry point for two-way conversations
  POST /twilio/voice/respond/{session_id}  — Handle user speech and continue conversation

All spoken audio uses ElevenLabs — Polly is completely removed.
"""

import logging
import xml.sax.saxutils as saxutils
from typing import Optional

from fastapi import APIRouter, Form
from fastapi.responses import Response

from chanakya.io_logger import Timer, log_api_call, log_input, log_output

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/twilio/status")
async def twilio_call_status(
    CallSid: str = Form(...),
    CallStatus: str = Form(...),
    CallDuration: Optional[str] = Form(None),
):
    """Handle Twilio call status webhook.

    Marks interaction_log as FAILED on no-answer, busy, failed, canceled,
    or calls shorter than 10 seconds (voicemail). Always returns HTTP 200.
    """
    from bson import ObjectId
    from chanakya.db.mongo import interaction_logs, agent_tasks

    log_input("CALL", None, f"status={CallStatus}", extra={"call_sid": CallSid, "duration": CallDuration})

    duration = int(CallDuration) if CallDuration and CallDuration.isdigit() else None
    is_voicemail = duration is not None and duration < 10
    is_failed_status = CallStatus in ("no-answer", "busy", "failed", "canceled")

    log = interaction_logs.find_one({"twilio_call_sid": CallSid})
    task = agent_tasks.find_one({"result.call_sid": CallSid})

    if is_voicemail or is_failed_status:
        reason = "voicemail" if is_voicemail else CallStatus
        if log:
            interaction_logs.update_one(
                {"_id": log["_id"]},
                {
                    "$set": {
                        "ai_evaluation.verdict": "FAILED",
                        "ai_evaluation.reasoning": (
                            f"Call {reason}: duration={duration}s, status={CallStatus}"
                        ),
                    }
                },
            )
            logger.info("Call %s marked FAILED: reason=%s", CallSid, reason)
        else:
            logger.warning("No interaction_log found for call_sid=%s", CallSid)
            
        if task:
            from chanakya.scheduler.task_runner import _mark_task_failed
            _mark_task_failed(task, f"Twilio call failed: {reason}")
    else:
        if CallStatus == "completed":
            session_id = str(log["_id"]) if log else None
            if session_id:
                from chanakya.db.mongo import voice_sessions
                session = voice_sessions.find_one({"_id": session_id})
                if session:
                    # Clean up the session (ephemeral)
                    voice_sessions.delete_one({"_id": session_id})
                    
                    if session.get("proxy"):
                        import asyncio
                        asyncio.ensure_future(_send_proxy_call_summary(session_id, session))
                    
                    # Also mark the task as COMPLETED if it exists and wasn't handled by proxy summary
                    task_id = session.get("task_id")
                    if task_id and not session.get("proxy"):
                        agent_tasks.update_one(
                            {"_id": ObjectId(task_id)},
                            {"$set": {"status": "COMPLETED"}}
                        )
                        logger.info("Task %s marked COMPLETED after user call.", task_id)

    return {"status": "ok"}


@router.get("/twilio/twiml/{log_id}")
async def twilio_twiml(log_id: str):
    """Return TwiML <Play> for a scheduled checkpoint call (audio stored in interaction_log)."""
    from bson import ObjectId
    from chanakya.db.mongo import interaction_logs

    try:
        oid = ObjectId(log_id)
    except Exception:
        return Response(
            content=(
                '<?xml version="1.0" encoding="UTF-8"?>'
                "<Response><Say>Invalid log ID.</Say></Response>"
            ),
            media_type="application/xml",
            status_code=400,
        )

    log = interaction_logs.find_one({"_id": oid})
    audio_url = log.get("media_url") if log else None

    if audio_url:
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f"<Response><Play>{audio_url}</Play></Response>"
        )
    else:
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response><Say>Audio not available.</Say></Response>"
        )

    return Response(content=twiml, media_type="application/xml")


# ---------------------------------------------------------------------------
# Two-way voice conversation — session store and helpers
# ---------------------------------------------------------------------------

# Memory cache for ephemeral audio bytes (auto-evicts after 100 entries)
# Key: session_id or session_id_turn
_audio_cache = {}
_AUDIO_CACHE_MAX = 100


def _cache_audio(key: str, audio: bytes) -> None:
    """Store audio in cache with automatic eviction when full."""
    if len(_audio_cache) >= _AUDIO_CACHE_MAX:
        # Evict oldest entries (first 20)
        keys_to_remove = list(_audio_cache.keys())[:20]
        for k in keys_to_remove:
            del _audio_cache[k]
    _audio_cache[key] = audio

from chanakya.db.mongo import voice_sessions
from datetime import datetime


def create_voice_session(
    session_id: str,
    user_id: str,
    context: str,
    conversation_context: str = "",
    audio_bytes: bytes | None = None,
    proxy: bool = False,
    proxy_contact_name: str = "",
    proxy_topic: str = "",
    owner_telegram_id: str = "",
    owner_name: str = "",
    task_id: str = "",
) -> None:
    """Register a voice conversation session. Audio bytes are kept in memory."""
    session_doc = {
        "_id": session_id,
        "user_id": user_id,
        "context": context,
        "conversation_context": conversation_context,
        "history": [],
        "proxy": proxy,
        "proxy_contact_name": proxy_contact_name,
        "proxy_topic": proxy_topic,
        "owner_telegram_id": owner_telegram_id,
        "owner_name": owner_name,
        "task_id": task_id,
        "created_at": datetime.utcnow()
    }
    
    if audio_bytes:
        _cache_audio(session_id, audio_bytes)

    voice_sessions.replace_one({"_id": session_id}, session_doc, upsert=True)
    logger.info(
        "Voice session created: session_id=%s proxy=%s. Audio cached in memory.",
        session_id, proxy
    )


def _clean_for_speech(text: str) -> str:
    """Remove markdown and convert CAPS_UNDERSCORE identifiers to readable words."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^•\s*", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b",
        lambda m: m.group(1).replace("_", " ").title(),
        text,
    )
    text = re.sub(r"\n+", " ", text).strip()
    return text


def _synthesize_for_turn(text: str, session: dict) -> bytes | None:
    """Synthesize text via ElevenLabs using the session's user voice_id.

    Returns MP3 bytes on success, None on failure.
    Falls back gracefully — caller must handle None.
    """
    from chanakya.integrations.elevenlabs_client import ElevenLabsClient, ElevenLabsSynthesisError
    from chanakya.db.mongo import users as users_col
    from bson import ObjectId

    user_id = session.get("user_id", "")
    voice_id = None

    # Get voice_id from user doc
    if user_id:
        try:
            user_doc = users_col.find_one({"_id": ObjectId(user_id)})
            if user_doc:
                voice_id = user_doc.get("elevenlabs_voice_id", "")
        except Exception:
            pass

    # Fall back to config-level voice
    if not voice_id:
        try:
            from chanakya.config import ELEVENLABS_VOICE_ID
            voice_id = ELEVENLABS_VOICE_ID
        except Exception:
            pass

    if not voice_id:
        logger.warning("No ElevenLabs voice_id available for session %s", session.get("user_id"))
        return None

    try:
        client = ElevenLabsClient()
        audio_bytes = client.synthesise(text, voice_id)
        logger.info("ElevenLabs turn synthesis: %d bytes for user %s", len(audio_bytes), str(user_id))
        return audio_bytes
    except ElevenLabsSynthesisError as exc:
        logger.warning("ElevenLabs turn synthesis failed: %s", exc)
        return None


def _safe_xml(text: str) -> str:
    """Strict XML escaping for TwiML."""
    if not text: return ""
    return saxutils.escape(text).replace('"', "&quot;").replace("'", "&apos;")


def _build_gather_twiml(
    session_id: str,
    say_text: str,
    is_final: bool = False,
    audio_url: str | None = None,
    session: dict | None = None,
) -> str:
    """Build TwiML with ElevenLabs audio. Synthesizes on-the-fly if no audio_url."""
    from chanakya.config import WEBHOOK_URL
    base = (WEBHOOK_URL or "").rstrip("/")

    # Always use ElevenLabs — synthesize now if no pre-cached audio
    if not audio_url and say_text and session:
        audio_bytes = _synthesize_for_turn(say_text, session)
        if audio_bytes:
            _cache_audio(f"{session_id}_turn", audio_bytes)
            audio_url = f"{base}/twilio/audio/turn/{session_id}"

    if audio_url:
        speech_element = f"<Pause length=\"1\"/><Play>{_safe_xml(audio_url)}</Play>"
    else:
        # Absolute last resort — should rarely happen (ElevenLabs down)
        speech_element = f"<Say>{_safe_xml(say_text)}</Say>"

    if is_final:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f"<Response>{speech_element}<Hangup/></Response>"
        )

    respond_url = f"{base}/twilio/voice/respond/{session_id}"
    entry_url = f"{base}/twilio/voice/{session_id}"

    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"{speech_element}"
        f'<Gather input="speech" action="{_safe_xml(respond_url)}" method="POST" '
        f'speechTimeout="auto" language="en-IN" speechModel="phone_call">'
        "</Gather>"
        f'<Redirect method="POST">{_safe_xml(entry_url)}</Redirect>'
        "</Response>"
    )


@router.get("/twilio/audio/{session_id}")
async def twilio_audio(session_id: str):
    """Serve opening audio from memory cache."""
    audio = _audio_cache.get(session_id)
    if not audio:
        logger.warning("Audio cache MISS for session %s (opening)", session_id)
        return Response(status_code=404)

    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={"Cache-Control": "max-age=3600"},
    )


@router.get("/twilio/audio/turn/{session_id}")
async def twilio_audio_turn(session_id: str, farewell: Optional[str] = None, t: Optional[str] = None):
    """Serve mid-call turn audio from memory cache."""
    if farewell:
        cache_key = f"{session_id}_farewell"
    elif t:
        cache_key = f"{session_id}_turn_{t}"
    else:
        cache_key = f"{session_id}_turn"
    audio = _audio_cache.get(cache_key)
    if not audio:
        logger.warning("Audio cache MISS for session %s (key=%s)", session_id, cache_key)
        return Response(status_code=404)

    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={"Cache-Control": "max-age=3600"},
    )



@router.post("/twilio/voice/{session_id}")
async def twilio_voice_entry(session_id: str):
    """TwiML entry point for a two-way voice conversation.

    Plays the ElevenLabs-synthesized opening (if available) from DB.
    """
    session = voice_sessions.find_one({"_id": session_id})
    if not session:
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response><Say>Session not found. Goodbye.</Say></Response>"
        )
        return Response(content=twiml, media_type="application/xml")

    opening = session["context"] or "Chanakya here. What do you need to discuss?"
    opening = _clean_for_speech(opening)
    session["history"].append({"role": "assistant", "content": opening})

    log_input("CALL", session.get("user_id"), f"[CALL CONNECTED] session={session_id}")
    log_output("CALL", session.get("user_id"), opening, extra={"session_id": session_id, "turn": "opening"})

    # Use ElevenLabs audio if we have it in memory cache
    audio_url: str | None = None
    if _audio_cache.get(session_id):
        from chanakya.config import WEBHOOK_URL
        base = (WEBHOOK_URL or "").rstrip("/")
        audio_url = f"{base}/twilio/audio/{session_id}"

    twiml = _build_gather_twiml(session_id, opening, audio_url=audio_url, session=session)

    # Update session history in DB
    voice_sessions.update_one(
        {"_id": session_id},
        {"$set": {"history": session["history"]}}
    )

    return Response(content=twiml, media_type="application/xml")


@router.post("/twilio/voice/respond/{session_id}")
async def twilio_voice_respond(
    session_id: str,
    SpeechResult: Optional[str] = Form(None),
    Confidence: Optional[str] = Form(None),
):
    """Handle user's spoken response and continue the conversation."""
    session = voice_sessions.find_one({"_id": session_id})
    if not session:
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response><Say>Session expired. Goodbye.</Say></Response>"
        )
        return Response(content=twiml, media_type="application/xml")

    user_speech = (SpeechResult or "").strip()
    logger.info(
        "Voice respond: session=%s speech=%r confidence=%s",
        session_id,
        user_speech[:100],
        Confidence,
    )

    if not user_speech:
        retry_text = "I didn't catch that. Say it again."
        retry_audio = _synthesize_for_turn(retry_text, session)
        if retry_audio:
            _cache_audio(f"{session_id}_turn", retry_audio)
            from chanakya.config import WEBHOOK_URL
            base = (WEBHOOK_URL or "").rstrip("/")
            audio_url = f"{base}/twilio/audio/turn/{session_id}"
        else:
            audio_url = None
        twiml = _build_gather_twiml(session_id, retry_text, audio_url=audio_url, session=session)
        return Response(content=twiml, media_type="application/xml")

    # End-of-call keywords
    end_keywords = {"bye", "goodbye", "that's all", "thats all", "done", "end call", "stop"}
    if any(kw in user_speech.lower() for kw in end_keywords):
        session_copy = dict(session)
        voice_sessions.delete_one({"_id": session_id})
        
        if session_copy.get("proxy"):
            import asyncio
            asyncio.ensure_future(_send_proxy_call_summary(session_id, session_copy))
            
        owner = session_copy.get("owner_name", "the owner")
        farewell = (
            f"Thank you. I'll pass this along to {owner}. Goodbye."
            if session_copy.get("proxy")
            else "Understood. Execute the plan. No excuses. Goodbye."
        )
        farewell_audio = _synthesize_for_turn(farewell, session_copy)
        if farewell_audio:
            cache_key = f"{session_id}_farewell"
            _cache_audio(cache_key, farewell_audio)
            from chanakya.config import WEBHOOK_URL
            base = (WEBHOOK_URL or "").rstrip("/")
            audio_url = f"{base}/twilio/audio/turn/{session_id}?farewell=1"
        else:
            audio_url = None
        twiml = _build_gather_twiml(session_id, farewell, is_final=True, audio_url=audio_url, session=session_copy)
        return Response(content=twiml, media_type="application/xml")

    session["history"].append({"role": "user", "content": user_speech})

    try:
        reply = await _get_chanakya_voice_reply(session)
        if not reply:
            reply = "Continue. What else?"
    except Exception as exc:
        logger.error("Voice reply failed for session %s: %s", session_id, exc, exc_info=True)
        reply = "I encountered an issue. Reflect on what you said and act accordingly."

    reply = _clean_for_speech(reply)
    session["history"].append({"role": "assistant", "content": reply})

    if len(session["history"]) > 10:
        session["history"] = session["history"][-10:]

    # Update rolling conversation context (fire-and-forget)
    try:
        from chanakya.agent.context_assembler import update_conversation_context
        from chanakya.db.mongo import users as users_col
        from bson import ObjectId
        import asyncio

        user_id = session.get("user_id", "")
        if user_id:
            user_doc = users_col.find_one({"_id": ObjectId(user_id)})
            if user_doc:
                asyncio.ensure_future(update_conversation_context(
                    user_doc, role="user", content=user_speech, channel="call"
                ))
                asyncio.ensure_future(update_conversation_context(
                    user_doc, role="assistant", content=reply, channel="call"
                ))
    except Exception as exc:
        logger.warning("Failed to schedule context update for session %s: %s", session_id, exc)

    log_input("CALL", session.get("user_id"), user_speech, extra={"session_id": session_id, "confidence": Confidence})
    log_output("CALL", session.get("user_id"), reply, extra={"session_id": session_id})

    # Use turn index in cache key to avoid collision on Twilio retries
    turn_index = len(session["history"])
    turn_cache_key = f"{session_id}_turn_{turn_index}"

    # Synthesize reply via ElevenLabs
    turn_audio = _synthesize_for_turn(reply, session)
    if turn_audio:
        _cache_audio(turn_cache_key, turn_audio)
        from chanakya.config import WEBHOOK_URL
        base = (WEBHOOK_URL or "").rstrip("/")
        audio_url = f"{base}/twilio/audio/turn/{session_id}?t={turn_index}"
    else:
        audio_url = None
        logger.warning("ElevenLabs synthesis failed for turn in session %s", session_id)

    # Update session in DB
    voice_sessions.update_one(
        {"_id": session_id},
        {"$set": {"history": session["history"]}}
    )

    twiml = _build_gather_twiml(session_id, reply, audio_url=audio_url, session=session)
    return Response(content=twiml, media_type="application/xml")


async def _get_chanakya_voice_reply(session: dict) -> str:
    """Generate Chanakya's reply for a voice turn.

    Uses a lightweight direct OpenAI call with the full call history for context.
    The full ChanakyaAgent pipeline (12s+ latency) is too slow for live phone
    conversation — tool calls can be followed up on Telegram after the call.

    For proxy calls: uses a polite persona instead of the Chanakya persona.
    Returns a plain spoken reply (1-3 sentences, no markdown).
    """
    is_proxy = session.get("proxy", False)

    if is_proxy:
        return await _get_proxy_voice_reply(session)

    return await _get_simple_voice_reply(session)


async def _get_proxy_voice_reply(session: dict) -> str:
    """Lightweight OpenAI call for proxy calls (calling on behalf of user)."""
    from chanakya.config import OPENAI_API_KEY
    import httpx

    owner_name = session.get("owner_name", "Goutham")
    contact_name = session.get("proxy_contact_name", "")
    topic = session.get("proxy_topic", "")

    system_prompt = (
        f"You are Chanakya, an AI assistant calling on behalf of {owner_name}. "
        f"You are speaking with {contact_name}. "
        f"Your goal for this call: {topic}. "
        "Be polite, warm, and conversational. "
        "Keep replies SHORT — 1 to 3 sentences. Speak naturally as if on a phone call. "
        "Do NOT mention being an AI unless directly asked. "
        f"If asked, say: 'I'm calling on behalf of {owner_name}.' "
        "Gather the information needed and wrap up naturally.\n\n"

        "=== STRICT PRIVACY RULES — NEVER VIOLATE ===\n"
        f"1. NEVER reveal {owner_name}'s schedule, plans, tasks, checkpoints, or daily routine to anyone.\n"
        f"2. NEVER share personal details about {owner_name} — relationships, habits, goals, finances, health, or anything private.\n"
        f"3. If asked whether {owner_name} is free or available at a specific time: answer ONLY 'yes, he's available' or "
        f"'no, he's not available at that time' — nothing more. Do NOT explain why or what he is doing.\n"
        f"4. Exception: if the caller IS {contact_name} and {owner_name} has a scheduled meeting/plan specifically WITH {contact_name}, "
        f"you may confirm 'yes, {owner_name} has something scheduled with you at that time' — but still reveal no other details.\n"
        f"5. NEVER discuss other people in {owner_name}'s life — family, friends, partner — with this caller.\n"
        f"6. If the caller asks anything outside the call topic ('{topic}'), politely redirect: "
        f"'I'm only calling about {topic}. For anything else, {owner_name} will reach out directly.'\n"
        "7. Stay strictly on the call topic. Do not volunteer any information beyond what is needed to complete the task.\n"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(session["history"])

    _t = Timer()
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-5.4-nano-2026-03-17", "messages": messages, "max_completion_tokens": 150, "temperature": 0.7},
        )
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"].strip()
        log_api_call("OpenAI", "POST", "/v1/chat/completions",
                     user_id=session.get("user_id"),
                     request_preview=str(session["history"][-1:]),
                     response_preview=reply,
                     status_code=response.status_code,
                     latency_ms=_t.elapsed_ms())
        return reply


async def _get_simple_voice_reply(session: dict) -> str:
    """Fallback: simple OpenAI call without tool access."""
    from chanakya.config import OPENAI_API_KEY
    import httpx

    cross_channel_ctx = session.get("conversation_context", "")
    call_context = session.get("context", "")

    system_prompt = (
        "You are Chanakya, a strict accountability coach on a PHONE CALL. "
        "Keep replies SHORT — maximum 3 sentences. "
        "Be direct, sharp, actionable. No markdown, no bullets. Speak naturally.\n\n"
    )
    if cross_channel_ctx:
        system_prompt += f"RECENT CONTEXT:\n{cross_channel_ctx}\n\n"
    if call_context:
        system_prompt += f"CALL TOPIC:\n{call_context}"

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(session["history"])

    _t = Timer()
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-5.4-nano-2026-03-17", "messages": messages, "max_completion_tokens": 150, "temperature": 0.7},
        )
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"].strip()
        log_api_call("OpenAI", "POST", "/v1/chat/completions",
                     user_id=session.get("user_id"),
                     request_preview=str(session["history"][-1:]),
                     response_preview=reply,
                     status_code=response.status_code,
                     latency_ms=_t.elapsed_ms())
        return reply


async def _send_proxy_call_summary(session_id: str, session: dict) -> None:
    """Generate a summary of a proxy call and send it to the owner via Telegram."""
    from chanakya.config import OPENAI_API_KEY, TELEGRAM_BOT_TOKEN
    from chanakya.db.mongo import proxy_call_logs
    import httpx
    from datetime import datetime

    owner_telegram_id = session.get("owner_telegram_id", "")
    contact_name = session.get("proxy_contact_name", "")
    topic = session.get("proxy_topic", "")
    history = session.get("history", [])
    user_id = session.get("user_id", "")

    if not owner_telegram_id or not history:
        return

    # Build transcript text
    transcript_lines = []
    for turn in history:
        role = "Chanakya" if turn["role"] == "assistant" else contact_name
        transcript_lines.append(f"{role}: {turn['content']}")
    transcript = "\n".join(transcript_lines)

    # Ask OpenAI to summarise
    summary = transcript  # fallback
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
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
                            "role": "system",
                            "content": (
                                "You are summarising a phone call made on behalf of a user. "
                                "Be concise. Extract: what was discussed, any answers/info received, "
                                "and any action items. Use bullet points. Max 150 words."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Call topic: {topic}\n"
                                f"Called: {contact_name}\n\n"
                                f"Transcript:\n{transcript}"
                            ),
                        },
                    ],
                    "max_completion_tokens": 250,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            summary = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.error("Failed to generate proxy call summary: %s", exc)

    # Store in proxy_call_logs
    try:
        proxy_call_logs.insert_one({
            "user_id": user_id,
            "session_id": session_id,
            "contact_name": contact_name,
            "topic": topic,
            "transcript": transcript,
            "summary": summary,
            "on_behalf_of": True,
            "timestamp": datetime.utcnow(),
        })
    except Exception as exc:
        logger.error("Failed to store proxy call log: %s", exc)

    # Send summary to owner via Telegram
    try:
        from telegram import Bot
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        msg = (
            f"📞 <b>Call Summary — {contact_name}</b>\n"
            f"<i>Topic: {topic}</i>\n\n"
            f"{summary}\n\n"
            f"<i>Full transcript stored in DB.</i>"
        )
        await bot.send_message(
            chat_id=owner_telegram_id,
            text=msg,
            parse_mode="HTML",
        )
        logger.info("Proxy call summary sent to user %s", owner_telegram_id)
    except Exception as exc:
        logger.error("Failed to send proxy call summary to Telegram: %s", exc)

    # Mark the associated task as COMPLETED
    try:
        from chanakya.db.mongo import agent_tasks
        from bson import ObjectId
        task_id_str = session.get("task_id")
        if task_id_str:
            agent_tasks.update_one(
                {"_id": ObjectId(task_id_str)},
                {"$set": {"status": "COMPLETED"}}
            )
            logger.info("Task %s marked COMPLETED after proxy call.", task_id_str)
    except Exception as exc:
        logger.error("Failed to mark task as COMPLETED: %s", exc)


# ---------------------------------------------------------------------------
# Fallback helper
# ---------------------------------------------------------------------------


def log_twilio_fallback(user_id, checkpoint_id: str, reason: str) -> None:
    """Log that a Twilio call fell back to Telegram text."""
    logger.warning(
        "Twilio call fallback for user=%s, checkpoint=%s: %s",
        user_id,
        checkpoint_id,
        reason,
    )


def synthesize_call_opening(text: str) -> bytes | None:
    """Synthesize call opening text via ElevenLabs using ELEVENLABS_VOICE_ID from config.

    Returns MP3 bytes on success, None on any failure (caller falls back to <Say>).
    """
    from chanakya.config import ELEVENLABS_VOICE_ID
    from chanakya.integrations.elevenlabs_client import (
        ElevenLabsClient,
        ElevenLabsSynthesisError,
    )

    try:
        client = ElevenLabsClient()
        audio_bytes = client.synthesise(text, ELEVENLABS_VOICE_ID)
        logger.info("ElevenLabs synthesis succeeded for call opening (%d bytes)", len(audio_bytes))
        return audio_bytes
    except ElevenLabsSynthesisError as exc:
        logger.warning("ElevenLabs synthesis failed for call opening: %s — call will use Say fallback", exc)
        return None
