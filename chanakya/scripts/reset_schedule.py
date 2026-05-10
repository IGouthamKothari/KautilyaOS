"""
reset_schedule.py — Clear all checkpoints for the active user and reingest from goutham_base.json.

Usage:
    python chanakya/scripts/reset_schedule.py
    python chanakya/scripts/reset_schedule.py --dry-run   # preview only
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset and reingest schedule for active user")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    from chanakya.db.mongo import checkpoints, users
    from chanakya.scripts.load_schedule import write_schedule_to_db, _load_base_schedule, format_schedule_preview

    user = users.find_one({"active": True})
    if not user:
        logger.error("No active user found.")
        sys.exit(1)

    logger.info("Active user: %s (id=%s)", user.get("name"), user["_id"])

    # Count existing
    existing = checkpoints.count_documents({"user_id": user["_id"]})
    logger.info("Existing checkpoints: %d", existing)

    if args.dry_run:
        schedule_data = _load_base_schedule()
        print(format_schedule_preview(schedule_data))
        ins, upd = write_schedule_to_db(user["_id"], dry_run=True)
        logger.info("DRY RUN — would insert %d checkpoints", ins)
        return

    # Delete all existing checkpoints for this user
    result = checkpoints.delete_many({"user_id": user["_id"]})
    logger.info("Deleted %d checkpoints.", result.deleted_count)

    # Reingest from JSON
    inserted, updated = write_schedule_to_db(user["_id"])
    logger.info("Reingested: %d inserted, %d updated.", inserted, updated)

    # Verify
    new_count = checkpoints.count_documents({"user_id": user["_id"]})
    logger.info("New checkpoint count: %d", new_count)

    # Show a sample
    sample = list(checkpoints.find({"user_id": user["_id"]}).sort("time", 1).limit(3))
    for cp in sample:
        logger.info(
            "  %s | %s | %s | %s",
            cp.get("time"), cp.get("display_name", cp.get("activity")),
            cp.get("action_type"), cp.get("priority"),
        )

    logger.info("Done. Restart the app to pick up changes.")


if __name__ == "__main__":
    main()
