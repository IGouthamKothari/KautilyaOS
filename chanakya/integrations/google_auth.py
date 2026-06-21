"""
google_auth.py — Google OAuth2 flow + token storage in MongoDB.

Handles:
  - Building the authorization URL (redirect user to Google)
  - Exchanging the callback code for tokens
  - Storing/refreshing tokens in the users collection
  - Returning an authorized httplib2.Http or google credentials object
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

from chanakya.config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.modify",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
]


def build_auth_url(state: str) -> str:
    """Build Google OAuth2 authorization URL manually — no PKCE."""
    import urllib.parse
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
        "include_granted_scopes": "true",
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)


def exchange_code(code: str, state: str = "") -> dict:
    """Exchange authorization code for tokens via direct HTTP POST (no PKCE)."""
    import requests as _requests
    resp = _requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        },
        timeout=15,
    )
    if not resp.ok:
        body = resp.text
        logger.error("Google token exchange HTTP %s: %s", resp.status_code, body)
        raise ValueError(f"Google token exchange failed ({resp.status_code}): {body}")
    data = resp.json()
    if "error" in data:
        raise ValueError(f"Token exchange error: {data['error']} — {data.get('error_description', '')}")
    from datetime import timezone, timedelta
    expiry = None
    if "expires_in" in data:
        expiry = (datetime.now(timezone.utc) + timedelta(seconds=data["expires_in"])).isoformat()
    return {
        "token": data.get("access_token"),
        "refresh_token": data.get("refresh_token"),
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "scopes": SCOPES,
        "expiry": expiry,
    }


def _creds_to_dict(creds: Credentials) -> dict:
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else SCOPES,
        "expiry": creds.expiry.isoformat() if creds.expiry else None,
    }


def save_google_tokens(user_id, token_dict: dict) -> None:
    """Persist Google tokens to the user document."""
    from chanakya.db.mongo import users
    users.update_one(
        {"_id": user_id},
        {"$set": {"google_tokens": token_dict, "google_connected": True, "updated_at": datetime.utcnow()}}
    )
    logger.info("Google tokens saved for user %s", user_id)


def get_credentials(user_id) -> Optional[Credentials]:
    """Load + auto-refresh Google credentials for a user. Returns None if not connected."""
    from chanakya.db.mongo import users
    user = users.find_one({"_id": user_id})
    if not user or not user.get("google_tokens"):
        return None

    t = user["google_tokens"]
    expiry = datetime.fromisoformat(t["expiry"]) if t.get("expiry") else None

    creds = Credentials(
        token=t.get("token"),
        refresh_token=t.get("refresh_token"),
        token_uri=t.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=t.get("client_id", GOOGLE_CLIENT_ID),
        client_secret=t.get("client_secret", GOOGLE_CLIENT_SECRET),
        scopes=t.get("scopes", SCOPES),
    )
    if expiry:
        creds.expiry = expiry

    # Refresh if expired
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            save_google_tokens(user_id, _creds_to_dict(creds))
        except Exception as e:
            logger.warning("Google token refresh failed for user %s: %s", user_id, e)
            return None

    return creds


def is_connected(user_id) -> bool:
    from chanakya.db.mongo import users
    user = users.find_one({"_id": user_id}, {"google_connected": 1})
    return bool(user and user.get("google_connected"))
