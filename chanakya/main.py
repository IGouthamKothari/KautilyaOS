import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
import signal
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from telegram import Update

from chanakya.config import HOST, PORT, TELEGRAM_BOT_TOKEN, WEBHOOK_URL
from chanakya.integrations.twilio_webhooks import router as twilio_router
from chanakya.io_logger import log_input
from chanakya.scheduler.checkpoint_runner import start_runner, stop_runner
from chanakya.scheduler.task_runner import start_task_runner, stop_task_runner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level application reference — set during startup, used by webhook handler
# ---------------------------------------------------------------------------
_telegram_app = None


def _auto_seed_schedules() -> None:
    """Seed goutham_base.json checkpoints for any active user with no checkpoints.

    Runs once at startup. Safe to call multiple times — write_schedule_to_db
    uses upsert so existing checkpoints are never duplicated.
    """
    from chanakya.db.mongo import checkpoints, users as users_col
    from chanakya.scripts.load_schedule import write_schedule_to_db

    try:
        active_users = list(users_col.find({"active": True}))
    except Exception as exc:
        logger.warning("Auto-seed: failed to fetch users: %s", exc)
        return

    for user in active_users:
        try:
            count = checkpoints.count_documents({"user_id": user["_id"], "active": True})
            if count == 0:
                inserted, updated = write_schedule_to_db(user["_id"])
                logger.info(
                    "Auto-seeded schedule for user %s: %d inserted, %d updated",
                    user["_id"], inserted, updated,
                )
            else:
                logger.debug(
                    "User %s already has %d checkpoints — skipping auto-seed",
                    user["_id"], count,
                )
        except Exception as exc:
            logger.warning("Auto-seed failed for user %s: %s", user.get("_id"), exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI to ensure proper event loop binding."""
    global _telegram_app

    if not WEBHOOK_URL:
        raise RuntimeError(
            "WEBHOOK_URL is not set. "
            "Set it to your public HTTPS URL (e.g. https://your-app.onrender.com) "
            "in your .env file. Telegram requires a public HTTPS endpoint to deliver updates."
        )

    # 1. Start Checkpoint runner
    start_runner()
    start_task_runner()
    logger.info("Checkpoint runner started.")

    # 2. Telegram bot — webhook mode
    from chanakya.bot.telegram_bot import build_application
    application = build_application()
    webhook_url = WEBHOOK_URL.rstrip("/") + "/telegram"

    try:
        await application.initialize()
        await application.bot.set_webhook(
            url=webhook_url,
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )
        await application.start()
        _telegram_app = application
        logger.info("✅ Telegram Webhook successfully registered: %s", webhook_url)
    except Exception as e:
        logger.error("❌ FAILED to register Telegram Webhook: %s", e)
        # We don't raise here to allow the rest of the app (Web UI, Scheduler) to function

    # Auto-seed schedule for any active users with no checkpoints
    _auto_seed_schedules()

    # Proactive Startup Audit: Check user status immediately on boot
    from chanakya.bot.telegram_bot import perform_startup_audit
    asyncio.create_task(perform_startup_audit(application))

    logger.info("Chanakya is watching. Webhook: %s", webhook_url)
    
    yield

    # Graceful teardown
    logger.info("Shutting down...")
    stop_runner()
    stop_task_runner()
    if _telegram_app:
        await _telegram_app.bot.delete_webhook()
        await _telegram_app.stop()
        await _telegram_app.shutdown()
    logger.info("Shutdown complete.")


def create_fastapi_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Mounts:
      GET  /health              — health check
      POST /twilio/status       — Twilio call status callback
      GET  /twilio/twiml/{id}   — TwiML response
      POST /telegram            — Telegram webhook endpoint
    """
    app = FastAPI(title="Chanakya Bot", version="1.0.0", lifespan=lifespan)
    app.include_router(twilio_router)

    # Static UI
    static_path = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_path):
        app.mount("/static", StaticFiles(directory=static_path), name="static")

    @app.api_route("/", methods=["GET", "HEAD"])
    async def index():
        """Serve the Dharma Dashboard."""
        return FileResponse(os.path.join(static_path, "index.html"))

    @app.api_route("/health", methods=["GET", "HEAD"])
    @app.api_route("/healthz", methods=["GET", "HEAD"])
    async def health():
        return {"status": "ok", "service": "chanakya-bot"}

    @app.post("/chat")
    async def web_chat(request: Request):
        """Handle chat from the web dashboard."""
        from chanakya.bot.telegram_bot import generic_process_message
        from chanakya.db.mongo import users
        
        data = await request.json()
        text = data.get("message")
        
        # Get the primary user (assumed to be you)
        user = users.find_one({"active": True})
        if not user:
            return {"response": "No active user found in Dharma database."}
            
        # Process through the universal agent engine
        response_text = await generic_process_message(user, text, channel="WEB")
        return {"response": response_text}

    @app.post("/telegram")
    async def telegram_webhook(request: Request) -> Response:
        """Receive Telegram update via webhook and dispatch to the bot."""
        if _telegram_app is None:
            logger.warning("Telegram app not initialised yet — dropping update.")
            return Response(status_code=503)
        data = await request.json()
        update = Update.de_json(data, _telegram_app.bot)
        # Log every incoming update so messages are visible in the console
        if update.message:
            user = update.message.from_user
            text = update.message.text or update.message.caption or "[media]"
            logger.info(
                "Telegram message from %s (@%s, id=%s): %s",
                user.full_name if user else "unknown",
                user.username if user else "?",
                user.id if user else "?",
                text[:200],
            )
            # Note: log_input is called inside _process_message_inner — not here
        await _telegram_app.process_update(update)
        return Response(status_code=200)

    return app

# ---------------------------------------------------------------------------
# Module-level app instance (Req 25.1)
# ---------------------------------------------------------------------------
app = create_fastapi_app()


def main() -> None:
    """Main entrypoint: start Uvicorn synchronously on the main thread."""
    logger.info("Starting Uvicorn on %s:%s", HOST, PORT)
    uvicorn.run("chanakya.main:app", host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()