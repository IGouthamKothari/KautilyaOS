"""
enrich_schedule.py — One-time script to AI-enrich goutham_base.json.

Calls gpt-5.4-nano with the full schedule context and asks it to generate:
  - display_name: short human-readable name (e.g. "Wake Up Call", "Gym Session")
  - description:  1-2 sentence plain-English description of what this block is for

Writes the enriched data back into goutham_base.json in-place.
Safe to re-run — only fills in missing display_name/description fields.

Usage:
    python chanakya/scripts/enrich_schedule.py
    python chanakya/scripts/enrich_schedule.py --force   # overwrite existing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCHEDULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "schedules", "goutham_base.json",
)


def _collect_unique_activities(schedule_data: dict) -> list[dict]:
    """Collect all unique activity entries across all days."""
    seen: dict[str, dict] = {}  # activity key → entry
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    for day in days:
        day_data = schedule_data["base_schedule"].get(day, [])
        if day_data == "COPY_MONDAY":
            day_data = schedule_data["base_schedule"]["monday"]
        if not isinstance(day_data, list):
            continue
        for entry in day_data:
            key = entry["activity"]
            if key not in seen:
                seen[key] = entry

    return list(seen.values())


def _build_prompt(activities: list[dict], force: bool) -> str:
    """Build the GPT prompt for enriching activities."""
    activity_lines = []
    for act in activities:
        if not force and act.get("display_name") and act.get("description"):
            continue  # already enriched
        notes = act.get("notes", "")
        activity_lines.append(
            f'- activity: "{act["activity"]}", time: {act["time"]}, '
            f'priority: {act["priority"]}, duration: {act["duration_min"]}min'
            + (f', notes: "{notes}"' if notes else "")
        )

    if not activity_lines:
        return ""

    activities_text = "\n".join(activity_lines)

    return f"""You are enriching a personal daily schedule for Goutham, a software engineer in India.
He follows a strict accountability system. His goals: DSA/LeetCode mastery, gym 5x/week, deep work at office, system design study, healthy sleep.

For each activity below, return a JSON array where each object has:
  - "activity": the original key (unchanged)
  - "display_name": short human-readable name, 2-4 words, Title Case, NO underscores, NO ALL CAPS
  - "description": 1-2 sentences describing what this block is for and why it matters. Conversational, not robotic.

Activities to enrich:
{activities_text}

Return ONLY a valid JSON array. No markdown, no explanation, no code fences. Example format:
[
  {{"activity": "WAKE_UP", "display_name": "Wake Up", "description": "Start the day on time. Missing this sets a bad tone for everything that follows."}},
  ...
]"""


def _call_gpt(prompt: str) -> list[dict]:
    """Call gpt-5.4-nano and parse the JSON response."""
    import httpx
    from chanakya.config import OPENAI_API_KEY

    logger.info("Calling gpt-5.4-nano to enrich schedule activities...")

    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-5.4-nano-2026-03-17",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "temperature": 0.4,
        },
        timeout=30.0,
    )
    response.raise_for_status()

    raw = response.json()["choices"][0]["message"]["content"].strip()
    logger.info("GPT response received (%d chars)", len(raw))

    # Strip markdown fences if present
    import re
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


def _apply_enrichments(schedule_data: dict, enrichments: list[dict], force: bool) -> int:
    """Apply enrichments to all matching activity entries in the schedule. Returns count updated."""
    enrichment_map = {e["activity"]: e for e in enrichments}
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    updated = 0

    for day in days:
        day_data = schedule_data["base_schedule"].get(day, [])
        if day_data == "COPY_MONDAY" or not isinstance(day_data, list):
            continue
        for entry in day_data:
            key = entry["activity"]
            if key in enrichment_map:
                enriched = enrichment_map[key]
                if force or not entry.get("display_name"):
                    entry["display_name"] = enriched["display_name"]
                    updated += 1
                if force or not entry.get("description"):
                    entry["description"] = enriched["description"]

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-enrich schedule display names and descriptions")
    parser.add_argument("--force", action="store_true", help="Overwrite existing display_name/description")
    parser.add_argument("--dry-run", action="store_true", help="Print enrichments without writing to file")
    args = parser.parse_args()

    with open(SCHEDULE_PATH, "r", encoding="utf-8") as f:
        schedule_data = json.load(f)

    activities = _collect_unique_activities(schedule_data)
    logger.info("Found %d unique activities in schedule", len(activities))

    # Filter to only unenriched ones unless --force
    to_enrich = [a for a in activities if args.force or not (a.get("display_name") and a.get("description"))]
    if not to_enrich:
        logger.info("All activities already enriched. Use --force to re-enrich.")
        return

    logger.info("%d activities need enrichment", len(to_enrich))

    prompt = _build_prompt(to_enrich, force=args.force)
    enrichments = _call_gpt(prompt)
    logger.info("Received %d enrichments from GPT", len(enrichments))

    if args.dry_run:
        print("\n--- DRY RUN — enrichments that would be applied ---")
        for e in enrichments:
            print(f"\n  {e['activity']}")
            print(f"    display_name: {e['display_name']}")
            print(f"    description:  {e['description']}")
        return

    updated = _apply_enrichments(schedule_data, enrichments, force=args.force)

    # Bump version
    old_version = schedule_data.get("version", "1.0")
    try:
        major, minor = old_version.split(".")
        schedule_data["version"] = f"{major}.{int(minor) + 1}"
    except Exception:
        schedule_data["version"] = old_version + "-enriched"

    with open(SCHEDULE_PATH, "w", encoding="utf-8") as f:
        json.dump(schedule_data, f, indent=2, ensure_ascii=False)

    logger.info(
        "Done. %d entries updated. Version: %s → %s. Written to %s",
        updated, old_version, schedule_data["version"], SCHEDULE_PATH,
    )


if __name__ == "__main__":
    main()
