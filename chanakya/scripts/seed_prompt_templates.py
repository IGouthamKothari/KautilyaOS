"""
seed_prompt_templates.py — Seed initial prompt templates into MongoDB.

Run once before first deployment:
  python scripts/seed_prompt_templates.py

Creates templates for all (activity_slot, interaction_type, tone) combinations
needed for the system to function. Templates use {variable} placeholders.
"""

import sys
import os
# Add workspace root (parent of the chanakya package) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime

ACTIVITY_SLOTS = [
    "OFFICE_WORK", "LEETCODE", "GYM", "COMMUTE", "MEAL",
    "FREE_TIME", "SLEEP", "STUDY", "GENERIC",
]

INTERACTION_TYPES = [
    "CHECKPOINT", "CHECK_IN", "EOD", "ESCALATION",
    "MENTOR_TALK", "COMMAND_RESPONSE", "MORNING_TODO",
]

TONES = ["HARSH", "MENTOR", "NEUTRAL", "CELEBRATORY"]

# Core templates — these are the ones the system actually uses
CORE_TEMPLATES = [
    # FREE_TIME × CHECKPOINT
    {
        "activity_slot": "FREE_TIME",
        "interaction_type": "CHECKPOINT",
        "tone": "HARSH",
        "template_text": "{name}, this checkpoint is due. You have been warned. Failure is not an option. Current streak: {streak} days. Do not break it.",
    },
    {
        "activity_slot": "FREE_TIME",
        "interaction_type": "CHECKPOINT",
        "tone": "MENTOR",
        "template_text": "{name}, time for your checkpoint. You are on a {streak}-day streak. Keep the momentum. What is your status?",
    },
    {
        "activity_slot": "FREE_TIME",
        "interaction_type": "CHECKPOINT",
        "tone": "NEUTRAL",
        "template_text": "{name}, checkpoint time. Streak: {streak} days. Please confirm your status.",
    },
    {
        "activity_slot": "FREE_TIME",
        "interaction_type": "CHECKPOINT",
        "tone": "CELEBRATORY",
        "template_text": "{name}, checkpoint! You are on fire — {streak} days strong. Keep going!",
    },
    # FREE_TIME × CHECK_IN
    {
        "activity_slot": "FREE_TIME",
        "interaction_type": "CHECK_IN",
        "tone": "MENTOR",
        "template_text": "{name}, quick check-in. How are you spending your free time? Are you working toward your goals or drifting?",
    },
    {
        "activity_slot": "FREE_TIME",
        "interaction_type": "CHECK_IN",
        "tone": "HARSH",
        "template_text": "{name}. Free time is not an excuse for laziness. What have you accomplished in the last hour?",
    },
    {
        "activity_slot": "FREE_TIME",
        "interaction_type": "CHECK_IN",
        "tone": "NEUTRAL",
        "template_text": "{name}, checking in. What are you currently working on?",
    },
    # FREE_TIME × EOD
    {
        "activity_slot": "FREE_TIME",
        "interaction_type": "EOD",
        "tone": "HARSH",
        "template_text": "{name}, the day is over. Streak: {streak} days. Mode: {current_mode}. Give me an honest account of today. What did you accomplish? What did you fail at? I will build your plan for tomorrow.",
    },
    {
        "activity_slot": "FREE_TIME",
        "interaction_type": "EOD",
        "tone": "MENTOR",
        "template_text": "{name}, end of day. Streak: {streak} days. Let us review today together and plan tomorrow. What went well? What needs improvement?",
    },
    {
        "activity_slot": "FREE_TIME",
        "interaction_type": "EOD",
        "tone": "NEUTRAL",
        "template_text": "{name}, daily review time. Streak: {streak}. Please summarize your day and I will generate tomorrow's plan.",
    },
    # FREE_TIME × MORNING_TODO
    {
        "activity_slot": "FREE_TIME",
        "interaction_type": "MORNING_TODO",
        "tone": "MENTOR",
        "template_text": "{name}, good morning. Streak: {streak} days. Here is your plan for today. Execute it without excuses.",
    },
    {
        "activity_slot": "FREE_TIME",
        "interaction_type": "MORNING_TODO",
        "tone": "HARSH",
        "template_text": "{name}. New day. No excuses. Your tasks for today are non-negotiable. Streak: {streak} days. Do not break it.",
    },
    {
        "activity_slot": "FREE_TIME",
        "interaction_type": "MORNING_TODO",
        "tone": "NEUTRAL",
        "template_text": "{name}, morning. Here is your todo list for today. Streak: {streak} days.",
    },
    # LEETCODE × CHECK_IN
    {
        "activity_slot": "LEETCODE",
        "interaction_type": "CHECK_IN",
        "tone": "MENTOR",
        "template_text": "{name}, you are solving LeetCode. What problem are you working on? What approach are you taking? Talk me through it.",
    },
    {
        "activity_slot": "LEETCODE",
        "interaction_type": "CHECK_IN",
        "tone": "HARSH",
        "template_text": "{name}. LeetCode session. How long have you been on this problem? If it has been more than 45 minutes, you need to look at the hint. Time is not infinite.",
    },
    # OFFICE_WORK × CHECK_IN
    {
        "activity_slot": "OFFICE_WORK",
        "interaction_type": "CHECK_IN",
        "tone": "MENTOR",
        "template_text": "{name}, you are at work. What task are you currently on? Any blockers? Stay focused.",
    },
    {
        "activity_slot": "OFFICE_WORK",
        "interaction_type": "CHECK_IN",
        "tone": "HARSH",
        "template_text": "{name}. Office hours. Are you actually working or are you distracted? What is your current task?",
    },
    # GYM × CHECKPOINT
    {
        "activity_slot": "GYM",
        "interaction_type": "CHECKPOINT",
        "tone": "HARSH",
        "template_text": "{name}, gym checkpoint. Send me a photo of the gym locker or equipment. No photo = no credit. Streak: {streak} days.",
    },
    {
        "activity_slot": "GYM",
        "interaction_type": "CHECKPOINT",
        "tone": "MENTOR",
        "template_text": "{name}, gym time. Prove you are there. Send a photo. Streak: {streak} days — protect it.",
    },
    # GENERIC × all interaction types (fallback)
    {
        "activity_slot": "GENERIC",
        "interaction_type": "CHECKPOINT",
        "tone": "NEUTRAL",
        "template_text": "{name}, checkpoint. Streak: {streak} days. Please confirm your status.",
    },
    {
        "activity_slot": "GENERIC",
        "interaction_type": "CHECK_IN",
        "tone": "NEUTRAL",
        "template_text": "{name}, quick check-in. How are things going?",
    },
    {
        "activity_slot": "GENERIC",
        "interaction_type": "EOD",
        "tone": "NEUTRAL",
        "template_text": "{name}, end of day review. Streak: {streak}. How did today go?",
    },
    {
        "activity_slot": "GENERIC",
        "interaction_type": "MORNING_TODO",
        "tone": "NEUTRAL",
        "template_text": "{name}, good morning. Streak: {streak}. Here is your plan for today.",
    },
    {
        "activity_slot": "GENERIC",
        "interaction_type": "ESCALATION",
        "tone": "HARSH",
        "template_text": "{name}, this is an escalation. You have failed repeatedly. Consequences are increasing.",
    },
    {
        "activity_slot": "GENERIC",
        "interaction_type": "MENTOR_TALK",
        "tone": "MENTOR",
        "template_text": "{name}, let us talk. I want to understand what is blocking you and help you break through it.",
    },
    {
        "activity_slot": "GENERIC",
        "interaction_type": "COMMAND_RESPONSE",
        "tone": "NEUTRAL",
        "template_text": "Acknowledged, {name}.",
    },
]


def seed_templates():
    """Seed prompt templates into MongoDB."""
    from chanakya.db.mongo import prompt_templates

    inserted = 0
    skipped = 0

    for template in CORE_TEMPLATES:
        # Check if already exists
        existing = prompt_templates.find_one({
            "activity_slot": template["activity_slot"],
            "interaction_type": template["interaction_type"],
            "tone": template["tone"],
        })

        if existing:
            skipped += 1
            continue

        prompt_templates.insert_one({
            **template,
            "version": 1,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        })
        inserted += 1

    print(f"Seeded {inserted} templates. Skipped {skipped} existing.")


if __name__ == "__main__":
    seed_templates()
