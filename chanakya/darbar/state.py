"""
state.py — Shared state flowing through the Darbar pipeline.

DarbarState is the single object passed between router, specialist, synthesizer, and gate.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DarbarState(BaseModel):
    """Shared state for the Darbar multi-agent pipeline."""

    user_id: str
    raw_input: str
    scrubbed_input: str = ""
    media_url: str | None = None
    interaction_type: str = "MENTOR_TALK"

    # Router output
    intent: str = ""
    urgency: Literal["low", "medium", "high", "critical"] = "medium"
    specialist: str = "chanakya"
    context_tier_needed: int = 1
    router_reasoning: str = ""
    routed_via: str = ""  # "fast_path" | "llm" | "bypass"

    # Context (assembled after routing)
    context: dict = Field(default_factory=dict)
    system_prompt: str = ""

    # Specialist output
    specialist_response: str = ""
    verdict: str | None = None
    tone: str = "NEUTRAL"
    actions: list = Field(default_factory=list)
    reasoning: str = ""

    # Quality gate
    gate_passed: bool = True
    gate_issue: str = ""

    # Final
    final_response: str = ""
    model_used: str = ""
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
