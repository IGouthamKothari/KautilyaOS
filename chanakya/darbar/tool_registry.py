"""
tool_registry.py — Maps tools to specialist agents.

Each specialist gets only its relevant tools, reducing token overhead
and improving tool selection accuracy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

# Tool names assigned to each specialist
TOOL_ASSIGNMENTS: dict[str, list[str]] = {
    "chanakya": [
        # Schedule/Calendar
        "fetch_schedule",
        "fetch_day_schedule",
        "add_day_event",
        "update_day_event",
        "delete_day_event",
        "reschedule_activity",
        "update_schedule_activity",
        "schedule_message",
        "cancel_scheduled_message",
        "get_day_log",
        # Contacts/Communication
        "save_contact",
        "list_contacts",
        "delete_contact",
        "place_proxy_call",
        "call_user",
        "send_telegram_message",
        "set_user_phone",
        # Mode/Schedule management
        "activate_war_mode",
        "deactivate_war_mode",
        "modify_wakeup_time",
        "add_daily_checkpoint",
        "update_morning_todo_time",
        "set_morning_todo_time",
        # Mindset
        "add_mindset_note",
        "add_mindset_entry",
        "get_mindset_notes",
        "remove_mindset_note",
        "clear_mindset_notes",
        # Status
        "get_user_status",
        # System
        "reload_prompt_templates",
        "send_emergency_alert",
        "escalate_punishment",
        # Council
        "consult_council",
        # Goals
        "set_goal",
        "update_goal",
        "list_goals",
        "abandon_goal_tool",
        # Google Calendar
        "google_list_events",
        "google_create_event",
        "google_update_event",
        "google_delete_event",
        "google_find_event",
        "google_add_reminder",
        # Gmail
        "gmail_list_inbox",
        "gmail_read_email",
        "gmail_search",
        "gmail_mark_read",
        "gmail_mark_important",
        "gmail_get_thread",
        "gmail_send_email",
        "gmail_reply_email",
    ],
    "kautilya": [
        "get_financial_ledger",
        "apply_financial_penalty",
        "set_commitment",
        "complete_commitment",
        "get_accountability_ledger",
        "update_warrior_streak",
        # Goals (finance-related goal tracking)
        "set_goal",
        "update_goal",
        "list_goals",
        # Google Calendar (to see financial deadlines, meetings)
        "google_list_events",
        "google_create_event",
        "google_find_event",
        # Gmail (to read financial emails)
        "gmail_list_inbox",
        "gmail_read_email",
        "gmail_search",
        # Shared read-only
        "get_user_status",
        "fetch_day_schedule",
    ],
    "charaka": [
        "log_ritual",
        "get_ritual_summary",
        # Shared read-only
        "get_user_status",
        "fetch_day_schedule",
        # Calendar — correlate rituals/health appointments with real schedule
        "google_list_events",
        "google_create_event",
        "google_find_event",
        "google_add_reminder",
        # Gmail — read health-related emails (appointment confirmations, lab results)
        "gmail_list_inbox",
        "gmail_read_email",
        "gmail_search",
    ],
    "vishvakarma": [
        # Future: code review, tech architecture tools
        "get_user_status",
        "fetch_day_schedule",
    ],
}


def get_tools_for_specialist(specialist: str) -> list[BaseTool]:
    """Return the actual LangChain tool objects for a given specialist."""
    from chanakya.tools.schedule_tools import ALL_TOOLS

    tool_map = {t.name: t for t in ALL_TOOLS}
    assigned_names = TOOL_ASSIGNMENTS.get(specialist, TOOL_ASSIGNMENTS["chanakya"])

    return [tool_map[name] for name in assigned_names if name in tool_map]


def get_all_specialist_names() -> list[str]:
    """Return all registered specialist IDs."""
    return list(TOOL_ASSIGNMENTS.keys())
