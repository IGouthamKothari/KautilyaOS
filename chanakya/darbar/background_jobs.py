"""
background_jobs.py — Periodic background jobs for the Darbar system.

Registers APScheduler jobs for:
  - Learning extractor (every 2 hours per active user)
  - Goal sentinel (every 6 hours per active user)

These never block user-facing responses. They run only when DARBAR_ENABLED=true.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler

from chanakya.db.mongo import users

logger = logging.getLogger(__name__)

_scheduler = BackgroundScheduler()


def start_darbar_jobs() -> None:
    """Start the Darbar background scheduler."""
    from chanakya.config import DARBAR_ENABLED

    if not DARBAR_ENABLED:
        logger.info("Darbar disabled — background jobs not started.")
        return

    _scheduler.add_job(
        _run_learning_for_all,
        "interval",
        hours=2,
        id="darbar_learning",
        replace_existing=True,
    )
    _scheduler.add_job(
        _run_goal_sentinel_for_all,
        "interval",
        hours=6,
        id="darbar_goal_sentinel",
        replace_existing=True,
    )

    if not _scheduler.running:
        _scheduler.start()

    logger.info("Darbar background jobs started (learning: 2h, sentinel: 6h).")


def stop_darbar_jobs() -> None:
    """Stop the Darbar background scheduler."""
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Darbar background jobs stopped.")


def _run_learning_for_all() -> None:
    """Run learning extraction for all active users who need it."""
    from chanakya.darbar.learning_extractor import should_run_learning, run_learning_cycle

    active_users = list(users.find({"active": True}))
    for user in active_users:
        uid = user["_id"]
        if should_run_learning(uid):
            try:
                _run_async(run_learning_cycle(uid))
            except Exception as exc:
                logger.warning("Learning cycle failed for user %s: %s", uid, exc)


def _run_goal_sentinel_for_all() -> None:
    """Run goal sentinel for all active users."""
    from chanakya.darbar.goal_sentinel import run_goal_sentinel

    active_users = list(users.find({"active": True}))
    for user in active_users:
        uid = user["_id"]
        try:
            _run_async(run_goal_sentinel(uid))
        except Exception as exc:
            logger.warning("Goal sentinel failed for user %s: %s", uid, exc)


def _run_async(coro):
    """Run an async coroutine from a sync scheduler thread."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(coro)
        else:
            asyncio.run(coro)
    except RuntimeError:
        asyncio.run(coro)
