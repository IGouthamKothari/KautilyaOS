"""
config.py — Application configuration.

Loads environment variables from a .env file using python-dotenv and exposes
them as module-level constants.  All required variables are validated at import
time; a single ValueError is raised listing every missing variable so the
operator can fix them all in one go.

Required variables
------------------
TELEGRAM_BOT_TOKEN   — python-telegram-bot authentication token
MONGODB_URI          — MongoDB Atlas connection string (mongodb+srv://...)
OPENROUTER_API_KEY   — OpenRouter API key for LLM access
TWILIO_ACCOUNT_SID   — Twilio account SID
TWILIO_AUTH_TOKEN    — Twilio auth token
TWILIO_PHONE_NUMBER  — Twilio outbound phone number (E.164 format)
ELEVENLABS_API_KEY   — ElevenLabs API key for TTS synthesis

Optional variables (with defaults)
-----------------------------------
LOG_LEVEL            — Python logging level string (default: "INFO")
HOST                 — Uvicorn bind host (default: "0.0.0.0")
PORT                 — Uvicorn bind port (default: 8000)
"""

import logging
import os

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file (silently ignored if the file does not exist)
# ---------------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------------
# Required environment variables
# ---------------------------------------------------------------------------

_REQUIRED_VARS = [
    "TELEGRAM_BOT_TOKEN",
    "MONGODB_URI",
    "OPENAI_API_KEY",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_PHONE_NUMBER",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_VOICE_ID",
]

_missing = [var for var in _REQUIRED_VARS if not os.getenv(var)]
if _missing:
    raise ValueError(
        "Missing required environment variables: "
        + ", ".join(_missing)
        + ". Set them in your .env file or as OS environment variables."
    )

TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
MONGODB_URI: str = os.environ["MONGODB_URI"]
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")  # kept for backward compat
TWILIO_ACCOUNT_SID: str = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN: str = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_PHONE_NUMBER: str = os.environ["TWILIO_PHONE_NUMBER"]
ELEVENLABS_API_KEY: str = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID: str = os.environ["ELEVENLABS_VOICE_ID"]

# ---------------------------------------------------------------------------
# Optional environment variables
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-5-mini")
UTILITY_MODEL_NAME: str = os.getenv("UTILITY_MODEL_NAME", "gpt-5.4-nano")

# Council Models
KAUTILYA_MODEL: str = os.getenv("KAUTILYA_MODEL", "gpt-5.4-mini-2026-03-17")
CHARAKA_MODEL: str = os.getenv("CHARAKA_MODEL", "gpt-5-nano-2025-08-07")
VISHVAKARMA_MODEL: str = os.getenv("VISHVAKARMA_MODEL", "gpt-5-mini-2025-08-07")

# Darbar Multi-Agent System
DARBAR_ENABLED: bool = os.getenv("DARBAR_ENABLED", "false").lower() in ("true", "1", "yes")
ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", UTILITY_MODEL_NAME)

# Google OAuth (Calendar + Gmail)
GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI: str = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")

# WEBHOOK_URL: public HTTPS URL Telegram will POST updates to.
# Example: https://your-app.onrender.com  (no trailing slash, no /telegram path)
# Telegram requires HTTPS — use Render, EC2 with SSL, or ngrok for local testing.
# Required at runtime — main.py will raise if not set.
WEBHOOK_URL: str | None = os.getenv("WEBHOOK_URL")

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
