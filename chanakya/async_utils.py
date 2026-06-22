"""
async_utils.py — Shared async/thread bridge utilities.

Single source of truth for scheduling coroutines from sync APScheduler
background threads onto the main FastAPI event loop.

Usage:
    from chanakya.async_utils import run_async, set_main_loop

Call set_main_loop() once during FastAPI lifespan startup.
Call run_async(coro) from any sync thread.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# The main event loop — set once at startup by set_main_loop()
_main_loop: asyncio.AbstractEventLoop | None = None


def set_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Store the main FastAPI event loop. Call once during lifespan startup."""
    global _main_loop
    _main_loop = loop
    logger.info("Main event loop registered for thread-safe async dispatch.")


def run_async(coro) -> None:
    """Schedule a coroutine from any thread onto the main event loop.

    - If called from within the main event loop thread: fire-and-forget via ensure_future.
    - If called from a background thread (APScheduler, Twilio): use run_coroutine_threadsafe.
    - Never calls asyncio.run() — that creates a new loop and crashes inside a running one.
    """
    global _main_loop

    # Case 1: we're already on a running event loop (e.g. called from async context)
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running is not None:
        asyncio.ensure_future(coro)
        return

    # Case 2: background thread — use the stored main loop
    if _main_loop is not None and _main_loop.is_running():
        asyncio.run_coroutine_threadsafe(coro, _main_loop)
        return

    # Case 3: last resort — main loop not set yet or already stopped
    # This should only happen during tests or very early startup
    logger.warning("run_async: main loop unavailable, falling back to asyncio.run()")
    try:
        asyncio.run(coro)
    except RuntimeError as exc:
        logger.error("run_async fallback failed: %s", exc)
