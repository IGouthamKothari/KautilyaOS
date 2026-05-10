"""
conftest.py — Shared pytest fixtures and Hypothesis profiles.

Provides:
  - Hypothesis settings profiles: ci (100 examples), thorough (500 examples)
  - SHARED_ENV_VARS dict for consistent test environment setup

Mock pattern used across all test files
----------------------------------------
All test files in this suite follow the same pattern for isolating MongoDB and
external service dependencies:

1. **MongoDB (mongomock)**:
   Each test file creates a ``mongomock.MongoClient()`` at module level and
   patches ``pymongo.MongoClient`` to return it *before* importing any
   ``chanakya.*`` module.  After import, the module-level collection handles
   (``mongo_module.users``, ``mongo_module.interaction_logs``, etc.) are
   re-pointed to the mongomock database so that all DB operations hit the
   in-memory store.  An ``autouse`` fixture re-applies these patches before
   every test to guard against cross-module contamination.

2. **ElevenLabsClient**:
   Tests that exercise code paths touching ElevenLabs use
   ``unittest.mock.patch("chanakya.integrations.elevenlabs_client.httpx.post")``
   and ``...httpx.get`` to intercept HTTP calls.  The module-level cache and
   ``_low_credit_alert_pending`` flag are reset in ``setUp`` / ``autouse``
   fixtures via ``_reset_elevenlabs_state()``.

3. **TwilioClient**:
   Tests patch ``chanakya.integrations.twilio_client.TwilioClient`` or
   directly replace ``client._client`` with a ``MagicMock`` to avoid real
   Twilio API calls.

4. **ChatOpenAI (LLM)**:
   Tests in ``test_chanakya_agent.py`` patch the ``_make_llm`` factory inside
   ``chanakya.agent.chanakya_agent`` with a side-effect function that returns
   pre-configured ``MagicMock`` instances, allowing precise control over which
   model succeeds or fails.
"""

import os
import sys
import unittest.mock as mock
from unittest.mock import MagicMock

import mongomock
import pytest
from hypothesis import HealthCheck, settings

# ---------------------------------------------------------------------------
# Hypothesis profiles
# ---------------------------------------------------------------------------

settings.register_profile(
    "ci",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "thorough",
    max_examples=500,
    suppress_health_check=[HealthCheck.too_slow],
)

# ---------------------------------------------------------------------------
# Shared env vars (set before any chanakya import in test files)
# ---------------------------------------------------------------------------

SHARED_ENV_VARS = {
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
