"""
test_main.py — Smoke tests for chanakya/main.py.

Tests:
  1. create_app() returns a FastAPI instance with /health endpoint
  2. GET /health returns {"status": "ok", "service": "chanakya-bot"}
  3. Twilio router is included — /twilio/status and /twilio/twiml/{log_id} routes exist
"""

from __future__ import annotations

import os
import sys
import unittest.mock as mock
from unittest.mock import MagicMock

import mongomock
import pytest

# ---------------------------------------------------------------------------
# Environment setup — must happen before any chanakya import
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

# Remove any previously cached chanakya modules so patches take effect cleanly
for _mod in list(sys.modules.keys()):
    if _mod.startswith("chanakya"):
        del sys.modules[_mod]

# Build a mongomock client that will be returned by MongoClient()
_mock_client = mongomock.MongoClient()
_mock_client.admin.command = MagicMock(return_value={"ok": 1})

with mock.patch("pymongo.MongoClient", return_value=_mock_client):
    import chanakya.db.mongo as mongo_module

# Re-point the module's db handle to the mongomock database
_mock_db = _mock_client["chanakya"]
mongo_module.db = _mock_db
mongo_module.users = _mock_db["users"]
mongo_module.interaction_logs = _mock_db["interaction_logs"]

# ---------------------------------------------------------------------------
# Import create_app after env/mongo setup
# ---------------------------------------------------------------------------

from chanakya.main import create_fastapi_app as create_app  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


# ---------------------------------------------------------------------------
# Test 1: create_app() returns a FastAPI instance
# ---------------------------------------------------------------------------


class TestCreateApp:
    def test_returns_fastapi_instance(self):
        """create_app() must return a FastAPI instance."""
        app = create_app()
        assert isinstance(app, FastAPI)

    def test_app_title_is_chanakya_bot(self):
        """The app title should be 'Chanakya Bot'."""
        app = create_app()
        assert app.title == "Chanakya Bot"

    def test_app_version_is_set(self):
        """The app version should be set."""
        app = create_app()
        assert app.version == "1.0.0"


# ---------------------------------------------------------------------------
# Test 2: GET /health returns {"status": "ok", "service": "chanakya-bot"}
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def setup_method(self):
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_health_returns_200(self):
        """GET /health must return HTTP 200."""
        response = self.client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self):
        """GET /health must return {"status": "ok", "service": "chanakya-bot"}."""
        response = self.client.get("/health")
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "chanakya-bot"


# ---------------------------------------------------------------------------
# Test 3: Twilio router is included
# ---------------------------------------------------------------------------


class TestTwilioRouterIncluded:
    def setup_method(self):
        self.app = create_app()
        self.client = TestClient(self.app)

    def _get_route_paths(self):
        """Return a set of all registered route paths."""
        return {route.path for route in self.app.routes}

    def test_twilio_status_route_exists(self):
        """POST /twilio/status route must be registered."""
        paths = self._get_route_paths()
        assert "/twilio/status" in paths, (
            f"/twilio/status not found in routes: {paths}"
        )

    def test_twilio_twiml_route_exists(self):
        """GET /twilio/twiml/{log_id} route must be registered."""
        paths = self._get_route_paths()
        assert "/twilio/twiml/{log_id}" in paths, (
            f"/twilio/twiml/{{log_id}} not found in routes: {paths}"
        )

    def test_twilio_status_accepts_post(self):
        """POST /twilio/status must accept POST requests (returns 200 or 422, not 404/405)."""
        response = self.client.post(
            "/twilio/status",
            data={"CallSid": "CA123", "CallStatus": "completed", "CallDuration": "60"},
        )
        # 200 = success, 422 = validation error (acceptable — route exists)
        assert response.status_code in (200, 422), (
            f"Unexpected status {response.status_code} for POST /twilio/status"
        )

    def test_twilio_twiml_accepts_get(self):
        """GET /twilio/twiml/{log_id} must accept GET requests (returns 200/400, not 404/405)."""
        response = self.client.get("/twilio/twiml/not-a-valid-id")
        # 400 = invalid ObjectId (expected), 200 = valid log found
        assert response.status_code in (200, 400), (
            f"Unexpected status {response.status_code} for GET /twilio/twiml/..."
        )
