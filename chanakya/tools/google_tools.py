"""
google_tools.py — LangChain tools for Google Calendar and Gmail.

These are registered with Chanakya and Kautilya so they can read/create
calendar events and read/triage emails as part of their normal tool loop.
"""

from __future__ import annotations

import logging
from bson import ObjectId
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _uid(user_id: str):
    try:
        return ObjectId(user_id)
    except Exception:
        return user_id


# ── CALENDAR TOOLS ───────────────────────────────────────────────────────────

@tool
def google_list_events(user_id: str, days_ahead: int = 7) -> str:
    """List upcoming Google Calendar events for the next N days.
    Use this to see what's on the user's real schedule before making plans."""
    from chanakya.integrations.google_calendar import list_events
    from chanakya.integrations.google_auth import is_connected
    uid = _uid(user_id)
    if not is_connected(uid):
        return "Google Calendar not connected. User needs to visit /auth/google to connect."
    try:
        events = list_events(uid, days_ahead=days_ahead)
        if not events:
            return f"No events found in the next {days_ahead} days."
        lines = [f"Upcoming {len(events)} events:"]
        for e in events:
            lines.append(f"- [{e['start']}] {e['title']}" + (f" @ {e['location']}" if e.get('location') else ""))
        return "\n".join(lines)
    except Exception as exc:
        return f"Calendar error: {exc}"


@tool
def google_create_event(
    user_id: str,
    title: str,
    start_iso: str,
    end_iso: str,
    description: str = "",
    location: str = "",
    reminder_minutes: int = 30,
) -> str:
    """Create a Google Calendar event.
    start_iso and end_iso must be ISO 8601 format e.g. '2026-06-22T10:00:00+05:30'.
    Always include timezone offset. reminder_minutes sets popup + email reminder."""
    from chanakya.integrations.google_calendar import create_event
    from chanakya.integrations.google_auth import is_connected
    uid = _uid(user_id)
    if not is_connected(uid):
        return "Google Calendar not connected."
    try:
        result = create_event(uid, title, start_iso, end_iso, description, location, reminder_minutes)
        return f"Event created: '{title}' — {result.get('link', '')}"
    except Exception as exc:
        return f"Failed to create event: {exc}"


@tool
def google_update_event(
    user_id: str,
    event_id: str,
    title: str = "",
    start_iso: str = "",
    end_iso: str = "",
    description: str = "",
    location: str = "",
) -> str:
    """Update an existing Google Calendar event. Pass only fields you want to change.
    event_id comes from google_list_events or google_find_event."""
    from chanakya.integrations.google_calendar import update_event
    from chanakya.integrations.google_auth import is_connected
    uid = _uid(user_id)
    if not is_connected(uid):
        return "Google Calendar not connected."
    kwargs = {}
    if title: kwargs["title"] = title
    if start_iso: kwargs["start_iso"] = start_iso
    if end_iso: kwargs["end_iso"] = end_iso
    if description: kwargs["description"] = description
    if location: kwargs["location"] = location
    if not kwargs:
        return "No fields provided to update."
    try:
        result = update_event(uid, event_id, **kwargs)
        return f"Event updated: '{result.get('title')}'"
    except Exception as exc:
        return f"Failed to update event: {exc}"


@tool
def google_delete_event(user_id: str, event_id: str) -> str:
    """Delete a Google Calendar event by its event_id.
    event_id comes from google_list_events or google_find_event."""
    from chanakya.integrations.google_calendar import delete_event
    from chanakya.integrations.google_auth import is_connected
    uid = _uid(user_id)
    if not is_connected(uid):
        return "Google Calendar not connected."
    try:
        return delete_event(uid, event_id)
    except Exception as exc:
        return f"Failed to delete event: {exc}"


@tool
def google_find_event(user_id: str, query: str, days_ahead: int = 14) -> str:
    """Search Google Calendar events by keyword in title or description.
    Returns matching events with their IDs for further operations."""
    from chanakya.integrations.google_calendar import find_event
    from chanakya.integrations.google_auth import is_connected
    uid = _uid(user_id)
    if not is_connected(uid):
        return "Google Calendar not connected."
    try:
        events = find_event(uid, query, days_ahead=days_ahead)
        if not events:
            return f"No events found matching '{query}'."
        lines = [f"Found {len(events)} matching events:"]
        for e in events:
            lines.append(f"- ID:{e['id']} [{e['start']}] {e['title']}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Calendar search error: {exc}"


@tool
def google_add_reminder(user_id: str, event_id: str, minutes_before: int = 15) -> str:
    """Add a popup + email reminder to an existing Google Calendar event."""
    from chanakya.integrations.google_calendar import add_reminder
    from chanakya.integrations.google_auth import is_connected
    uid = _uid(user_id)
    if not is_connected(uid):
        return "Google Calendar not connected."
    try:
        return add_reminder(uid, event_id, minutes_before)
    except Exception as exc:
        return f"Failed to add reminder: {exc}"


# ── GMAIL TOOLS ──────────────────────────────────────────────────────────────

@tool
def gmail_list_inbox(user_id: str, max_results: int = 10, unread_only: bool = True) -> str:
    """List recent Gmail inbox messages. Returns subject, sender, date, and snippet.
    Use unread_only=True (default) to show only unread emails."""
    from chanakya.integrations.google_gmail import list_inbox
    from chanakya.integrations.google_auth import is_connected
    uid = _uid(user_id)
    if not is_connected(uid):
        return "Gmail not connected. User needs to visit /auth/google to connect."
    try:
        emails = list_inbox(uid, max_results=max_results, unread_only=unread_only)
        if not emails:
            return "No emails found."
        lines = [f"{'Unread' if unread_only else 'Recent'} emails ({len(emails)}):"]
        for e in emails:
            unread = "🔴 " if e.get("unread") else ""
            lines.append(f"- {unread}ID:{e['id']} | From:{e['from']} | {e['subject']} | {e['snippet'][:80]}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Gmail error: {exc}"


@tool
def gmail_read_email(user_id: str, message_id: str) -> str:
    """Read the full content of a Gmail email by its message ID.
    Use gmail_list_inbox first to get message IDs."""
    from chanakya.integrations.google_gmail import get_email
    from chanakya.integrations.google_auth import is_connected
    uid = _uid(user_id)
    if not is_connected(uid):
        return "Gmail not connected."
    try:
        email = get_email(uid, message_id)
        return (
            f"From: {email['from']}\n"
            f"Subject: {email['subject']}\n"
            f"Date: {email['date']}\n"
            f"---\n{email['body']}"
        )
    except Exception as exc:
        return f"Failed to read email: {exc}"


@tool
def gmail_search(user_id: str, query: str, max_results: int = 10) -> str:
    """Search Gmail using Gmail query syntax.
    Examples: 'from:boss@company.com', 'subject:invoice', 'is:unread label:important'"""
    from chanakya.integrations.google_gmail import search_emails
    from chanakya.integrations.google_auth import is_connected
    uid = _uid(user_id)
    if not is_connected(uid):
        return "Gmail not connected."
    try:
        emails = search_emails(uid, query, max_results=max_results)
        if not emails:
            return f"No emails found for query: '{query}'"
        lines = [f"Search results for '{query}' ({len(emails)}):"]
        for e in emails:
            lines.append(f"- ID:{e['id']} | From:{e['from']} | {e['subject']} | {e['snippet'][:80]}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Gmail search error: {exc}"


@tool
def gmail_mark_read(user_id: str, message_id: str) -> str:
    """Mark a Gmail email as read by its message ID."""
    from chanakya.integrations.google_gmail import mark_read
    from chanakya.integrations.google_auth import is_connected
    uid = _uid(user_id)
    if not is_connected(uid):
        return "Gmail not connected."
    try:
        return mark_read(uid, message_id)
    except Exception as exc:
        return f"Failed to mark email as read: {exc}"


@tool
def gmail_get_thread(user_id: str, thread_id: str) -> str:
    """Get all messages in a Gmail thread by thread_id.
    Useful for reading full email conversations."""
    from chanakya.integrations.google_gmail import get_thread
    from chanakya.integrations.google_auth import is_connected
    uid = _uid(user_id)
    if not is_connected(uid):
        return "Gmail not connected."
    try:
        messages = get_thread(uid, thread_id)
        if not messages:
            return "No messages found in thread."
        parts = []
        for i, m in enumerate(messages, 1):
            parts.append(f"[{i}] From:{m['from']} | {m['date']}\n{m['body'][:500]}")
        return "\n\n".join(parts)
    except Exception as exc:
        return f"Failed to get thread: {exc}"


@tool
def gmail_mark_important(user_id: str, message_id: str) -> str:
    """Star/mark a Gmail email as important by its message ID."""
    from chanakya.integrations.google_gmail import mark_important
    from chanakya.integrations.google_auth import is_connected
    uid = _uid(user_id)
    if not is_connected(uid):
        return "Gmail not connected."
    try:
        return mark_important(uid, message_id)
    except Exception as exc:
        return f"Failed to mark email as important: {exc}"


@tool
def gmail_send_email(user_id: str, to: str, subject: str, body: str) -> str:
    """Send an email via Gmail on the user's behalf.
    to: recipient email address. subject: email subject. body: plain text body."""
    from chanakya.integrations.google_gmail import send_email
    from chanakya.integrations.google_auth import is_connected
    uid = _uid(user_id)
    if not is_connected(uid):
        return "Gmail not connected."
    try:
        return send_email(uid, to=to, subject=subject, body=body)
    except Exception as exc:
        return f"Failed to send email: {exc}"


@tool
def gmail_reply_email(user_id: str, message_id: str, body: str) -> str:
    """Reply to an existing Gmail email thread.
    message_id: the ID of the email to reply to. body: plain text reply content."""
    from chanakya.integrations.google_gmail import reply_email
    from chanakya.integrations.google_auth import is_connected
    uid = _uid(user_id)
    if not is_connected(uid):
        return "Gmail not connected."
    try:
        return reply_email(uid, message_id=message_id, body=body)
    except Exception as exc:
        return f"Failed to reply to email: {exc}"


# ── TOOL LIST ────────────────────────────────────────────────────────────────

ALL_GOOGLE_TOOLS = [
    google_list_events,
    google_create_event,
    google_update_event,
    google_delete_event,
    google_find_event,
    google_add_reminder,
    gmail_list_inbox,
    gmail_read_email,
    gmail_search,
    gmail_mark_read,
    gmail_mark_important,
    gmail_get_thread,
    gmail_send_email,
    gmail_reply_email,
]
