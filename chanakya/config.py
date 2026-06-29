"""
config.py — Application configuration.

Required variables
------------------
TELEGRAM_BOT_TOKEN   — python-telegram-bot authentication token
MONGODB_URI          — MongoDB Atlas connection string (mongodb+srv://...)
OPENAI_API_KEY       — OpenAI API key for LLM access

Optional variables (with defaults)
-----------------------------------
LOG_LEVEL            — Python logging level string (default: "INFO")
HOST                 — Uvicorn bind host (default: "0.0.0.0")
PORT                 — Uvicorn bind port (default: 8000)
WEBHOOK_URL          — Public HTTPS URL for Telegram webhook (Render URL)
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
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# Optional environment variables
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
# Primary model for LangChain tool-calling (chanakya_agent + specialists)
# call_with_fallback() uses Gemini native API first, then falls back here
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
UTILITY_MODEL_NAME: str = os.getenv("UTILITY_MODEL_NAME", "gpt-4o-mini")

# Council Models
KAUTILYA_MODEL: str = os.getenv("KAUTILYA_MODEL", "gpt-4o-mini")
CHARAKA_MODEL: str = os.getenv("CHARAKA_MODEL", "gpt-4o-mini")
VISHVAKARMA_MODEL: str = os.getenv("VISHVAKARMA_MODEL", "gpt-4o-mini")

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
