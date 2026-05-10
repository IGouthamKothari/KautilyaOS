"""
test_integrations.py — Unit tests for ElevenLabs and Twilio integrations.

Covers:
  ElevenLabs (Tasks 20.1–20.4):
    1.  synthesise returns audio bytes on successful API call
    2.  synthesise returns cached bytes on second call (no extra API call)
    3.  Cache expires after TTL — second call after TTL makes a new API request
    4.  synthesise raises ElevenLabsSynthesisError on HTTP 4xx/5xx
    5.  synthesise raises ElevenLabsSynthesisError on network error
    6.  Credit balance < 1000 → _low_credit_alert_pending set to True
    7.  Credit balance >= 1000 → _low_credit_alert_pending stays False
    8.  get_and_clear_low_credit_alert() returns True and resets flag to False

  Twilio client (Task 21.1–21.2):
    9.  send_sms returns message SID on success
    10. send_sms raises TwilioError on Twilio API failure
    11. make_call returns call SID on success
    12. make_call raises TwilioError on Twilio API failure

  Twilio webhooks (Task 21.3–21.4):
    13. POST /twilio/status with CallStatus=no-answer → verdict FAILED
    14. POST /twilio/status with CallDuration=5 (<10s) → verdict FAILED (voicemail)
    15. POST /twilio/status with CallStatus=completed and CallDuration=60 → no change
    16. GET /twilio/twiml/{log_id} with valid log → TwiML with <Play>
    17. GET /twilio/twiml/{log_id} with invalid ObjectId → 400 with <Say>
"""

import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Set required env vars and patch MongoClient BEFORE any chanakya import,
# because config.py validates env vars at import time and db/mongo.py
# connects at module level.
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

import mongomock

_mock_mongo_client = mongomock.MongoClient()
_mock_mongo_client.admin.command = MagicMock(return_value={"ok": 1})

with patch("pymongo.MongoClient", return_value=_mock_mongo_client):
    import chanakya.db.mongo as mongo_module

# Re-point the module's db handle to the mongomock database
_mock_db = _mock_mongo_client["chanakya"]
mongo_module.db = _mock_db
mongo_module.users = _mock_db["users"]
mongo_module.schedules = _mock_db["schedules"]
mongo_module.checkpoints = _mock_db["checkpoints"]
mongo_module.interaction_logs = _mock_db["interaction_logs"]
mongo_module.ai_tool_calls = _mock_db["ai_tool_calls"]
mongo_module.user_state_snapshots = _mock_db["user_state_snapshots"]
mongo_module.prompt_templates = _mock_db["prompt_templates"]


# ---------------------------------------------------------------------------
# Helpers to reset module-level cache and flag between tests
# ---------------------------------------------------------------------------


def _reset_elevenlabs_state():
    """Clear the audio cache and reset the low-credit flag."""
    import chanakya.integrations.elevenlabs_client as el_mod

    el_mod._audio_cache.clear()
    el_mod._low_credit_alert_pending = False


# ---------------------------------------------------------------------------
# ElevenLabs tests
# ---------------------------------------------------------------------------


class TestElevenLabsSynthesise(unittest.TestCase):
    """Tests for ElevenLabsClient.synthesise()."""

    def setUp(self):
        _reset_elevenlabs_state()

    def _make_mock_tts_response(self, content: bytes, status_code: int = 200):
        """Build a mock httpx.Response for the TTS endpoint."""
        import httpx

        mock_resp = MagicMock()
        mock_resp.content = content
        mock_resp.status_code = status_code
        mock_resp.text = "error body"
        if status_code >= 400:
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                message=f"HTTP {status_code}",
                request=MagicMock(),
                response=mock_resp,
            )
        else:
            mock_resp.raise_for_status.return_value = None
        return mock_resp

    def _make_mock_user_response(self, remaining: int):
        """Build a mock httpx.Response for the /user endpoint."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "subscription": {
                "character_limit": remaining + 500,
                "character_count": 500,
            }
        }
        return mock_resp

    @patch("chanakya.integrations.elevenlabs_client.httpx.get")
    @patch("chanakya.integrations.elevenlabs_client.httpx.post")
    def test_synthesise_returns_audio_bytes(self, mock_post, mock_get):
        """Test 1: synthesise returns audio bytes on a successful API call."""
        from chanakya.integrations.elevenlabs_client import ElevenLabsClient

        audio_data = b"fake-audio-bytes"
        mock_post.return_value = self._make_mock_tts_response(audio_data)
        mock_get.return_value = self._make_mock_user_response(5000)

        client = ElevenLabsClient()
        result = client.synthesise("Hello world", "voice-123")

        self.assertEqual(result, audio_data)
        mock_post.assert_called_once()

    @patch("chanakya.integrations.elevenlabs_client.httpx.get")
    @patch("chanakya.integrations.elevenlabs_client.httpx.post")
    def test_synthesise_returns_cached_bytes_on_second_call(self, mock_post, mock_get):
        """Test 2: synthesise returns cached bytes on second call (no extra API call)."""
        from chanakya.integrations.elevenlabs_client import ElevenLabsClient

        audio_data = b"cached-audio"
        mock_post.return_value = self._make_mock_tts_response(audio_data)
        mock_get.return_value = self._make_mock_user_response(5000)

        client = ElevenLabsClient()
        first = client.synthesise("Hello", "voice-abc")
        second = client.synthesise("Hello", "voice-abc")

        self.assertEqual(first, audio_data)
        self.assertEqual(second, audio_data)
        # API should only be called once; second call hits cache
        mock_post.assert_called_once()

    @patch("chanakya.integrations.elevenlabs_client.httpx.get")
    @patch("chanakya.integrations.elevenlabs_client.httpx.post")
    def test_cache_expires_after_ttl(self, mock_post, mock_get):
        """Test 3: Cache expires after TTL — second call after TTL makes a new API request."""
        import chanakya.integrations.elevenlabs_client as el_mod
        from chanakya.integrations.elevenlabs_client import ElevenLabsClient

        audio_data = b"audio-v1"
        mock_post.return_value = self._make_mock_tts_response(audio_data)
        mock_get.return_value = self._make_mock_user_response(5000)

        client = ElevenLabsClient()
        client.synthesise("Expire test", "voice-xyz")

        # Manually expire the cache entry by backdating its timestamp
        key = el_mod._cache_key("Expire test", "voice-xyz")
        old_bytes, _ = el_mod._audio_cache[key]
        el_mod._audio_cache[key] = (old_bytes, time.time() - el_mod._CACHE_TTL_SECONDS - 1)

        # Second call should hit the API again
        client.synthesise("Expire test", "voice-xyz")
        self.assertEqual(mock_post.call_count, 2)

    @patch("chanakya.integrations.elevenlabs_client.httpx.post")
    def test_synthesise_raises_on_http_error(self, mock_post):
        """Test 4: synthesise raises ElevenLabsSynthesisError on HTTP 4xx/5xx."""
        import httpx
        from chanakya.integrations.elevenlabs_client import (
            ElevenLabsClient,
            ElevenLabsSynthesisError,
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="401 Unauthorized",
            request=MagicMock(),
            response=mock_resp,
        )
        mock_post.return_value = mock_resp

        client = ElevenLabsClient()
        with self.assertRaises(ElevenLabsSynthesisError) as ctx:
            client.synthesise("Test", "voice-id")

        self.assertIn("401", str(ctx.exception))

    @patch("chanakya.integrations.elevenlabs_client.httpx.post")
    def test_synthesise_raises_on_network_error(self, mock_post):
        """Test 5: synthesise raises ElevenLabsSynthesisError on network error."""
        from chanakya.integrations.elevenlabs_client import (
            ElevenLabsClient,
            ElevenLabsSynthesisError,
        )

        mock_post.side_effect = Exception("Connection refused")

        client = ElevenLabsClient()
        with self.assertRaises(ElevenLabsSynthesisError) as ctx:
            client.synthesise("Test", "voice-id")

        self.assertIn("Connection refused", str(ctx.exception))

    @patch("chanakya.integrations.elevenlabs_client.httpx.get")
    @patch("chanakya.integrations.elevenlabs_client.httpx.post")
    def test_low_credit_sets_alert_flag(self, mock_post, mock_get):
        """Test 6: Credit balance < 1000 → _low_credit_alert_pending set to True."""
        import chanakya.integrations.elevenlabs_client as el_mod
        from chanakya.integrations.elevenlabs_client import ElevenLabsClient

        mock_post.return_value = self._make_mock_tts_response(b"audio")
        mock_get.return_value = self._make_mock_user_response(500)  # < 1000

        client = ElevenLabsClient()
        client.synthesise("Low credit test", "voice-id")

        self.assertTrue(el_mod._low_credit_alert_pending)

    @patch("chanakya.integrations.elevenlabs_client.httpx.get")
    @patch("chanakya.integrations.elevenlabs_client.httpx.post")
    def test_sufficient_credit_does_not_set_alert_flag(self, mock_post, mock_get):
        """Test 7: Credit balance >= 1000 → _low_credit_alert_pending stays False."""
        import chanakya.integrations.elevenlabs_client as el_mod
        from chanakya.integrations.elevenlabs_client import ElevenLabsClient

        mock_post.return_value = self._make_mock_tts_response(b"audio")
        mock_get.return_value = self._make_mock_user_response(5000)  # >= 1000

        client = ElevenLabsClient()
        client.synthesise("Sufficient credit test", "voice-id")

        self.assertFalse(el_mod._low_credit_alert_pending)

    @patch("chanakya.integrations.elevenlabs_client.httpx.get")
    @patch("chanakya.integrations.elevenlabs_client.httpx.post")
    def test_get_and_clear_low_credit_alert(self, mock_post, mock_get):
        """Test 8: get_and_clear_low_credit_alert() returns True and resets flag to False."""
        import chanakya.integrations.elevenlabs_client as el_mod
        from chanakya.integrations.elevenlabs_client import (
            ElevenLabsClient,
            get_and_clear_low_credit_alert,
        )

        mock_post.return_value = self._make_mock_tts_response(b"audio")
        mock_get.return_value = self._make_mock_user_response(100)  # < 1000

        client = ElevenLabsClient()
        client.synthesise("Alert test", "voice-id")

        # Flag should be True now
        self.assertTrue(el_mod._low_credit_alert_pending)

        # get_and_clear should return True and reset
        result = get_and_clear_low_credit_alert()
        self.assertTrue(result)
        self.assertFalse(el_mod._low_credit_alert_pending)

        # Calling again should return False
        result2 = get_and_clear_low_credit_alert()
        self.assertFalse(result2)


# ---------------------------------------------------------------------------
# Twilio client tests
# ---------------------------------------------------------------------------


class TestTwilioClient(unittest.TestCase):
    """Tests for TwilioClient.send_sms() and TwilioClient.make_call()."""

    def _make_client_with_mock(self):
        """Return a TwilioClient with a mocked underlying Twilio REST client."""
        from chanakya.integrations.twilio_client import TwilioClient

        client = TwilioClient.__new__(TwilioClient)
        client._client = MagicMock()
        return client

    def test_send_sms_returns_message_sid(self):
        """Test 9: send_sms returns message SID on success."""
        client = self._make_client_with_mock()
        mock_msg = MagicMock()
        mock_msg.sid = "SM123456"
        client._client.messages.create.return_value = mock_msg

        result = client.send_sms("+919999999999", "Hello")
        self.assertEqual(result, "SM123456")

    def test_send_sms_raises_twilio_error_on_failure(self):
        """Test 10: send_sms raises TwilioError on Twilio API failure."""
        from chanakya.integrations.twilio_client import TwilioError

        client = self._make_client_with_mock()
        client._client.messages.create.side_effect = Exception("Twilio API down")

        with self.assertRaises(TwilioError) as ctx:
            client.send_sms("+919999999999", "Hello")

        self.assertIn("SMS to", str(ctx.exception))

    def test_make_call_returns_call_sid(self):
        """Test 11: make_call returns call SID on success."""
        client = self._make_client_with_mock()
        mock_call = MagicMock()
        mock_call.sid = "CA987654"
        client._client.calls.create.return_value = mock_call

        result = client.make_call("+919999999999", "https://example.com/twiml")

        self.assertEqual(result, "CA987654")
        client._client.calls.create.assert_called_once_with(
            url="https://example.com/twiml",
            to="+919999999999",
            from_=os.environ["TWILIO_PHONE_NUMBER"],
        )

    def test_make_call_raises_twilio_error_on_failure(self):
        """Test 12: make_call raises TwilioError on Twilio API failure."""
        from chanakya.integrations.twilio_client import TwilioError

        client = self._make_client_with_mock()
        client._client.calls.create.side_effect = Exception("Call API down")

        with self.assertRaises(TwilioError) as ctx:
            client.make_call("+919999999999", "https://example.com/twiml")

        self.assertIn("Call to", str(ctx.exception))


# ---------------------------------------------------------------------------
# Twilio webhook tests
# ---------------------------------------------------------------------------


def _make_test_app():
    """Create a minimal FastAPI app with the Twilio webhooks router."""
    # Clear any cached module to avoid stale state from previous imports
    for mod_name in list(sys.modules.keys()):
        if "twilio_webhooks" in mod_name:
            del sys.modules[mod_name]

    from fastapi import FastAPI
    from chanakya.integrations.twilio_webhooks import router

    app = FastAPI()
    app.include_router(router)
    return app


class TestTwilioWebhooks(unittest.TestCase):
    """Tests for the Twilio FastAPI webhook endpoints."""

    def setUp(self):
        """Set up a TestClient with the Twilio webhooks router."""
        from fastapi.testclient import TestClient

        self.app = _make_test_app()
        self.http = TestClient(self.app)

    def _post_status(self, call_sid, call_status, call_duration=None):
        """Helper to POST to /twilio/status."""
        data = {"CallSid": call_sid, "CallStatus": call_status}
        if call_duration is not None:
            data["CallDuration"] = str(call_duration)
        return self.http.post("/twilio/status", data=data)

    def test_no_answer_sets_verdict_failed(self):
        """Test 13: POST /twilio/status with CallStatus=no-answer → verdict FAILED."""
        from bson import ObjectId
        import chanakya.integrations.twilio_webhooks as wh_mod

        fake_log = {"_id": ObjectId(), "twilio_call_sid": "CA_TEST_001"}
        mock_logs = MagicMock()
        mock_logs.find_one.return_value = fake_log
        mock_logs.update_one.return_value = MagicMock()

        with patch.object(wh_mod, "interaction_logs", mock_logs, create=True):
            # The webhook imports interaction_logs inside the function from chanakya.db.mongo
            # We need to patch it at the source
            with patch("chanakya.db.mongo.interaction_logs", mock_logs):
                resp = self._post_status("CA_TEST_001", "no-answer")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})
        mock_logs.update_one.assert_called_once()
        call_args = mock_logs.update_one.call_args
        update_doc = call_args[0][1]
        self.assertEqual(update_doc["$set"]["ai_evaluation.verdict"], "FAILED")

    def test_voicemail_sets_verdict_failed(self):
        """Test 14: POST /twilio/status with CallDuration=5 (<10s) → verdict FAILED (voicemail)."""
        from bson import ObjectId

        fake_log = {"_id": ObjectId(), "twilio_call_sid": "CA_TEST_002"}
        mock_logs = MagicMock()
        mock_logs.find_one.return_value = fake_log
        mock_logs.update_one.return_value = MagicMock()

        with patch("chanakya.db.mongo.interaction_logs", mock_logs):
            resp = self._post_status("CA_TEST_002", "completed", call_duration=5)

        self.assertEqual(resp.status_code, 200)
        mock_logs.update_one.assert_called_once()
        call_args = mock_logs.update_one.call_args
        update_doc = call_args[0][1]
        self.assertEqual(update_doc["$set"]["ai_evaluation.verdict"], "FAILED")
        self.assertIn("voicemail", update_doc["$set"]["ai_evaluation.reasoning"])

    def test_completed_long_call_no_verdict_change(self):
        """Test 15: POST /twilio/status with CallStatus=completed and CallDuration=60 → no change."""
        mock_logs = MagicMock()
        mock_logs.find_one.return_value = None

        with patch("chanakya.db.mongo.interaction_logs", mock_logs):
            resp = self._post_status("CA_TEST_003", "completed", call_duration=60)

        self.assertEqual(resp.status_code, 200)
        mock_logs.update_one.assert_not_called()

    def test_twiml_valid_log_returns_play_element(self):
        """Test 16: GET /twilio/twiml/{log_id} with valid log → TwiML with <Play>."""
        from bson import ObjectId

        oid = ObjectId()
        fake_log = {"_id": oid, "media_url": "https://example.com/audio.mp3"}
        mock_logs = MagicMock()
        mock_logs.find_one.return_value = fake_log

        with patch("chanakya.db.mongo.interaction_logs", mock_logs):
            resp = self.http.get(f"/twilio/twiml/{oid}")

        self.assertEqual(resp.status_code, 200)
        self.assertIn("application/xml", resp.headers["content-type"])
        self.assertIn("<Play>", resp.text)
        self.assertIn("https://example.com/audio.mp3", resp.text)

    def test_twiml_invalid_object_id_returns_400(self):
        """Test 17: GET /twilio/twiml/{log_id} with invalid ObjectId → 400 with <Say>."""
        resp = self.http.get("/twilio/twiml/not-a-valid-objectid")

        self.assertEqual(resp.status_code, 400)
        self.assertIn("application/xml", resp.headers["content-type"])
        self.assertIn("<Say>", resp.text)
        self.assertIn("Invalid log ID", resp.text)


if __name__ == "__main__":
    unittest.main()
