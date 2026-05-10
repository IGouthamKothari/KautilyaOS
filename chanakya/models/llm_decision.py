"""
llm_decision.py — Pydantic models for structured LLM output.

LLMDecision is the authoritative output of every agent invocation.
The server executes LLMDecision.actions in array order without modification.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    """A single side-effect action returned by the agent.

    type must be one of the known action types OR a tool name — if the LLM
    uses a tool name directly (e.g. "call_user") it is auto-promoted to
    call_tool with tool_name=type so the agent never hard-fails on format.
    """

    type: str   # flexible — validated/normalised in execute_actions
    params: dict = Field(default_factory=dict)


class LLMDecision(BaseModel):
    """Structured output returned by ChanakyaAgent on every invocation."""

    verdict: str | None = None  # "SUCCESS" | "FAILED" | "EXCUSED" | "WAR_MODE_OVERRIDE" | None
    actions: list[ActionItem] = Field(default_factory=list)
    tone: str = "NEUTRAL"  # "HARSH" | "MENTOR" | "NEUTRAL" | "CELEBRATORY"
    response_text: str = ""
    reasoning: str = ""
    streak_reset: bool = False
    model_used: str = ""
