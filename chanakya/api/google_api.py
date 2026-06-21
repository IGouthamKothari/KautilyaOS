"""
google_api.py — Google OAuth + Calendar + Gmail REST endpoints.

Routes:
  GET  /auth/google              — start OAuth flow (redirect to Google)
  GET  /auth/google/callback     — OAuth callback, store tokens, redirect to dashboard
  GET  /auth/google/status       — is Google connected?
  GET  /auth/google/disconnect   — remove tokens
  GET  /api/google/calendar      — list upcoming events
  POST /api/google/calendar      — create event
  PATCH /api/google/calendar/{id} — update event
  DELETE /api/google/calendar/{id} — delete event
  GET  /api/google/gmail         — list inbox
  GET  /api/google/gmail/{id}    — get single email
  POST /api/google/gmail/{id}/read — mark as read
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter(tags=["google"])


def _active_user():
    from chanakya.db.mongo import users
    user = users.find_one({"active": True})
    if not user:
        raise HTTPException(status_code=404, detail="No active user found")
    return user


# ── OAuth ────────────────────────────────────────────────────────────────────

@router.get("/auth/google")
async def google_auth_start():
    """Redirect browser to Google consent screen."""
    from chanakya.integrations.google_auth import build_auth_url
    state = secrets.token_urlsafe(16)
    url = build_auth_url(state=state)
    return RedirectResponse(url)


@router.get("/auth/google/callback")
async def google_auth_callback(code: str = "", error: str = "", state: str = ""):
    """Handle OAuth callback from Google, store tokens, redirect to dashboard."""
    from chanakya.integrations.google_auth import exchange_code, save_google_tokens
    if error:
        return RedirectResponse(f"/?google_error={error}")
    if not code:
        raise HTTPException(status_code=400, detail="No code received from Google")

    user = _active_user()
    try:
        tokens = exchange_code(code, state=state)
        save_google_tokens(user["_id"], tokens)
        logger.info("Google connected for user %s", user.get("name"))
    except Exception as e:
        logger.error("Google OAuth callback failed: %s", e, exc_info=True)
        return RedirectResponse(f"/?google_error={str(e)[:120]}")

    return RedirectResponse("/?google_connected=1")


@router.get("/auth/google/status")
async def google_status():
    from chanakya.integrations.google_auth import is_connected
    user = _active_user()
    connected = is_connected(user["_id"])
    return {"connected": connected, "user": user.get("name")}


@router.get("/auth/google/disconnect")
async def google_disconnect():
    from chanakya.db.mongo import users
    user = _active_user()
    users.update_one(
        {"_id": user["_id"]},
        {"$unset": {"google_tokens": "", "google_connected": ""}}
    )
    return RedirectResponse("/")


# ── Calendar ─────────────────────────────────────────────────────────────────

class EventCreate(BaseModel):
    title: str
    start_iso: str
    end_iso: str
    description: str = ""
    location: str = ""
    reminder_minutes: int = 30


class EventUpdate(BaseModel):
    title: Optional[str] = None
    start_iso: Optional[str] = None
    end_iso: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None


@router.get("/api/google/calendar")
async def get_calendar(days: int = 7):
    from chanakya.integrations.google_calendar import list_events
    user = _active_user()
    try:
        return list_events(user["_id"], days_ahead=days)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/api/google/calendar")
async def create_calendar_event(body: EventCreate):
    from chanakya.integrations.google_calendar import create_event
    user = _active_user()
    try:
        return create_event(
            user["_id"], body.title, body.start_iso, body.end_iso,
            body.description, body.location, body.reminder_minutes
        )
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.patch("/api/google/calendar/{event_id}")
async def update_calendar_event(event_id: str, body: EventUpdate):
    from chanakya.integrations.google_calendar import update_event
    user = _active_user()
    kwargs = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        return update_event(user["_id"], event_id, **kwargs)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.delete("/api/google/calendar/{event_id}")
async def delete_calendar_event(event_id: str):
    from chanakya.integrations.google_calendar import delete_event
    user = _active_user()
    try:
        return {"result": delete_event(user["_id"], event_id)}
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))


# ── Gmail ────────────────────────────────────────────────────────────────────

@router.get("/api/google/gmail")
async def get_inbox(limit: int = 15, unread_only: bool = True):
    from chanakya.integrations.google_gmail import list_inbox
    user = _active_user()
    try:
        return list_inbox(user["_id"], max_results=limit, unread_only=unread_only)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("/api/google/gmail/{message_id}")
async def get_email(message_id: str):
    from chanakya.integrations.google_gmail import get_email as _get
    user = _active_user()
    try:
        return _get(user["_id"], message_id)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/api/google/gmail/{message_id}/read")
async def mark_email_read(message_id: str):
    from chanakya.integrations.google_gmail import mark_read
    user = _active_user()
    try:
        return {"result": mark_read(user["_id"], message_id)}
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
