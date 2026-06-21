"""
google_calendar.py — Google Calendar tools for Chanakya agents.

All functions take user_id and return plain dicts/strings so they can be
wrapped as LangChain tools.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from googleapiclient.discovery import build

from chanakya.integrations.google_auth import get_credentials

logger = logging.getLogger(__name__)


def _service(user_id):
    creds = get_credentials(user_id)
    if not creds:
        raise ValueError("Google Calendar not connected. Visit /auth/google to connect.")
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def list_events(user_id, days_ahead: int = 7, max_results: int = 20) -> list[dict]:
    """Return upcoming calendar events for the next N days."""
    svc = _service(user_id)
    from chanakya.db.mongo import users
    user = users.find_one({"_id": user_id})
    tz = user.get("timezone", "Asia/Kolkata") if user else "Asia/Kolkata"

    now = datetime.now(ZoneInfo(tz))
    time_max = now + timedelta(days=days_ahead)

    result = svc.events().list(
        calendarId="primary",
        timeMin=now.isoformat(),
        timeMax=time_max.isoformat(),
        maxResults=max_results,
        singleEvents=True,
        orderBy="startTime",
    ).execute()

    events = []
    for e in result.get("items", []):
        start = e.get("start", {})
        end = e.get("end", {})
        events.append({
            "id": e.get("id"),
            "title": e.get("summary", "(no title)"),
            "start": start.get("dateTime") or start.get("date"),
            "end": end.get("dateTime") or end.get("date"),
            "location": e.get("location", ""),
            "description": e.get("description", ""),
            "link": e.get("htmlLink", ""),
        })
    return events


def create_event(
    user_id,
    title: str,
    start_iso: str,
    end_iso: str,
    description: str = "",
    location: str = "",
    reminder_minutes: int = 30,
) -> dict:
    """Create a calendar event. start_iso and end_iso are ISO 8601 strings."""
    svc = _service(user_id)
    from chanakya.db.mongo import users
    user = users.find_one({"_id": user_id})
    tz = user.get("timezone", "Asia/Kolkata") if user else "Asia/Kolkata"

    body = {
        "summary": title,
        "description": description,
        "location": location,
        "start": {"dateTime": start_iso, "timeZone": tz},
        "end": {"dateTime": end_iso, "timeZone": tz},
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "popup", "minutes": reminder_minutes},
                {"method": "email", "minutes": reminder_minutes},
            ],
        },
    }
    event = svc.events().insert(calendarId="primary", body=body).execute()
    logger.info("Created calendar event '%s' for user %s", title, user_id)
    return {"id": event["id"], "title": title, "link": event.get("htmlLink", "")}


def update_event(user_id, event_id: str, **kwargs) -> dict:
    """Update fields of an existing event. Pass title, start_iso, end_iso, description, location."""
    svc = _service(user_id)
    event = svc.events().get(calendarId="primary", eventId=event_id).execute()

    if "title" in kwargs:
        event["summary"] = kwargs["title"]
    if "description" in kwargs:
        event["description"] = kwargs["description"]
    if "location" in kwargs:
        event["location"] = kwargs["location"]

    from chanakya.db.mongo import users
    user = users.find_one({"_id": user_id})
    tz = user.get("timezone", "Asia/Kolkata") if user else "Asia/Kolkata"

    if "start_iso" in kwargs:
        event["start"] = {"dateTime": kwargs["start_iso"], "timeZone": tz}
    if "end_iso" in kwargs:
        event["end"] = {"dateTime": kwargs["end_iso"], "timeZone": tz}

    updated = svc.events().update(calendarId="primary", eventId=event_id, body=event).execute()
    return {"id": updated["id"], "title": updated.get("summary"), "link": updated.get("htmlLink", "")}


def delete_event(user_id, event_id: str) -> str:
    """Delete a calendar event by ID."""
    svc = _service(user_id)
    svc.events().delete(calendarId="primary", eventId=event_id).execute()
    logger.info("Deleted calendar event %s for user %s", event_id, user_id)
    return f"Event {event_id} deleted."


def add_reminder(user_id, event_id: str, minutes_before: int = 15) -> str:
    """Add a popup + email reminder to an existing event."""
    svc = _service(user_id)
    event = svc.events().get(calendarId="primary", eventId=event_id).execute()
    event["reminders"] = {
        "useDefault": False,
        "overrides": [
            {"method": "popup", "minutes": minutes_before},
            {"method": "email", "minutes": minutes_before},
        ],
    }
    svc.events().update(calendarId="primary", eventId=event_id, body=event).execute()
    return f"Reminder set {minutes_before} minutes before event."


def find_event(user_id, query: str, days_ahead: int = 14) -> list[dict]:
    """Search upcoming events by keyword in title or description."""
    events = list_events(user_id, days_ahead=days_ahead, max_results=50)
    q = query.lower()
    return [e for e in events if q in e["title"].lower() or q in e.get("description", "").lower()]
