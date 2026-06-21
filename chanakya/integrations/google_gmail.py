"""
google_gmail.py — Gmail tools for Chanakya agents.

Covers: read inbox, get thread, summarize, mark read, send, reply, triage for auto-checkpoints.
"""

from __future__ import annotations

import base64
import logging
import re
from datetime import datetime, timedelta
from email import message_from_bytes
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from googleapiclient.discovery import build

from chanakya.integrations.google_auth import get_credentials

logger = logging.getLogger(__name__)

# Labels considered high-priority for auto-checkpoint triage
_IMPORTANT_LABELS = {"IMPORTANT", "STARRED"}


def _service(user_id):
    creds = get_credentials(user_id)
    if not creds:
        raise ValueError("Gmail not connected. Visit /auth/google to connect.")
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _decode_body(payload: dict) -> str:
    """Recursively extract plain text body from a Gmail message payload."""
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
    for part in payload.get("parts", []):
        text = _decode_body(part)
        if text:
            return text
    return ""


def _parse_message(msg: dict) -> dict:
    headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
    body = _decode_body(msg.get("payload", {}))
    return {
        "id": msg["id"],
        "thread_id": msg.get("threadId"),
        "subject": headers.get("Subject", "(no subject)"),
        "from": headers.get("From", ""),
        "to": headers.get("To", ""),
        "date": headers.get("Date", ""),
        "snippet": msg.get("snippet", ""),
        "body": body[:2000],  # cap at 2k chars for LLM context
        "labels": msg.get("labelIds", []),
        "unread": "UNREAD" in msg.get("labelIds", []),
    }


def list_inbox(user_id, max_results: int = 15, unread_only: bool = True) -> list[dict]:
    """Return recent inbox messages."""
    svc = _service(user_id)
    query = "in:inbox"
    if unread_only:
        query += " is:unread"

    result = svc.users().messages().list(
        userId="me", q=query, maxResults=max_results
    ).execute()

    messages = []
    for item in result.get("messages", []):
        msg = svc.users().messages().get(
            userId="me", id=item["id"], format="full"
        ).execute()
        messages.append(_parse_message(msg))
    return messages


def get_email(user_id, message_id: str) -> dict:
    """Fetch a single email by ID."""
    svc = _service(user_id)
    msg = svc.users().messages().get(userId="me", id=message_id, format="full").execute()
    return _parse_message(msg)


def get_thread(user_id, thread_id: str) -> list[dict]:
    """Fetch all messages in a thread."""
    svc = _service(user_id)
    thread = svc.users().threads().get(userId="me", id=thread_id, format="full").execute()
    return [_parse_message(m) for m in thread.get("messages", [])]


def mark_read(user_id, message_id: str) -> str:
    """Mark an email as read."""
    svc = _service(user_id)
    svc.users().messages().modify(
        userId="me", id=message_id, body={"removeLabelIds": ["UNREAD"]}
    ).execute()
    return f"Message {message_id} marked as read."


def mark_important(user_id, message_id: str) -> str:
    """Star/mark an email as important."""
    svc = _service(user_id)
    svc.users().messages().modify(
        userId="me", id=message_id, body={"addLabelIds": ["STARRED"]}
    ).execute()
    return f"Message {message_id} marked as important."


def send_email(user_id, to: str, subject: str, body: str) -> str:
    """Send an email from the user's Gmail account."""
    svc = _service(user_id)
    msg = MIMEText(body, "plain")
    msg["to"] = to
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    svc.users().messages().send(userId="me", body={"raw": raw}).execute()
    return f"Email sent to {to}: {subject}"


def reply_email(user_id, message_id: str, body: str) -> str:
    """Reply to an existing email thread, preserving subject and thread."""
    svc = _service(user_id)
    original = svc.users().messages().get(userId="me", id=message_id, format="metadata",
                                           metadataHeaders=["Subject", "From", "Message-ID", "References"]).execute()
    headers = {h["name"]: h["value"] for h in original.get("payload", {}).get("headers", [])}
    thread_id = original.get("threadId")

    reply = MIMEText(body, "plain")
    subject = headers.get("Subject", "")
    reply["to"] = headers.get("From", "")
    reply["subject"] = subject if subject.startswith("Re:") else f"Re: {subject}"
    reply["In-Reply-To"] = headers.get("Message-ID", "")
    reply["References"] = headers.get("References", "") + " " + headers.get("Message-ID", "")

    raw = base64.urlsafe_b64encode(reply.as_bytes()).decode()
    svc.users().messages().send(userId="me", body={"raw": raw, "threadId": thread_id}).execute()
    return f"Reply sent to thread {thread_id}."


def search_emails(user_id, query: str, max_results: int = 10) -> list[dict]:
    """Search emails by Gmail query string (e.g. 'from:boss@co.com invoice')."""
    svc = _service(user_id)
    result = svc.users().messages().list(
        userId="me", q=query, maxResults=max_results
    ).execute()
    messages = []
    for item in result.get("messages", []):
        msg = svc.users().messages().get(userId="me", id=item["id"], format="full").execute()
        messages.append(_parse_message(msg))
    return messages


def triage_for_checkpoints(user_id, max_results: int = 20) -> list[dict]:
    """
    Fetch recent unread emails and return those that look action-requiring.
    Used by the background poller to create automatic checkpoints.
    Returns list of {subject, from, snippet, message_id} for high-priority items.
    """
    svc = _service(user_id)
    # Important + unread in inbox
    result = svc.users().messages().list(
        userId="me",
        q="in:inbox is:unread is:important",
        maxResults=max_results,
    ).execute()

    items = []
    for item in result.get("messages", []):
        msg = svc.users().messages().get(userId="me", id=item["id"], format="metadata",
                                          metadataHeaders=["Subject", "From", "Date"]).execute()
        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        items.append({
            "message_id": msg["id"],
            "thread_id": msg.get("threadId"),
            "subject": headers.get("Subject", "(no subject)"),
            "from": headers.get("From", ""),
            "date": headers.get("Date", ""),
            "snippet": msg.get("snippet", ""),
            "labels": msg.get("labelIds", []),
        })
    return items
