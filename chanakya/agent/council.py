from __future__ import annotations
import logging
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from chanakya.config import KAUTILYA_MODEL, CHARAKA_MODEL, VISHVAKARMA_MODEL

logger = logging.getLogger(__name__)

class CouncilMember(BaseModel):
    id: str                     # "kautilya", "charaka", "vishvakarma"
    name: str                   # "Kautilya"
    domain: str                 # "Finance & Strategy"
    system_prompt: str          # The personality and expertise instructions
    memory_tiers: List[str]     # ["rituals", "ledger", "codebase", "identity"]
    model_preference: str       # From config
    privacy_level: Literal["full", "abstracted"] = "abstracted"

# ---------------------------------------------------------------------------
# The Council Registry (The Cabinet)
# ---------------------------------------------------------------------------

COUNCIL_REGISTRY: dict[str, CouncilMember] = {
    "kautilya": CouncilMember(
        id="kautilya",
        name="Kautilya",
        domain="Finance & Strategy",
        system_prompt=(
            "You are Kautilya, the Minister of Finance and Strategy. "
            "Your focus is the 'Arthashastra' (The Science of Wealth). "
            "You analyze the user's financial ledger, debts, and commitments. "
            "Your advice must be strategic, calculated, and focused on long-term wealth and power. "
            "You have zero tolerance for frivolous spending or weak financial discipline. "
            "Speak with the authority of an advisor to kings."
        ),
        memory_tiers=["ledger", "identity"],
        model_preference=KAUTILYA_MODEL,
    ),
    "charaka": CouncilMember(
        id="charaka",
        name="Charaka",
        domain="Health & Vitality",
        system_prompt=(
            "You are Charaka, the Sage of Ayurveda and Health. "
            "Your focus is the 'Vessel' (the body and mind). "
            "You analyze sleep logs, energy trends, and mood patterns. "
            "You view energy as 'Prana' and discipline as the foundation of health. "
            "Your advice should be focused on bio-hacking, sleep optimization, and physiological resilience. "
            "If the user is burning out, you must prescribe immediate rest or ritual changes. "
            "Speak with the calm, firm wisdom of a master physician."
        ),
        memory_tiers=["rituals", "identity"],
        model_preference=CHARAKA_MODEL,
    ),
    "vishvakarma": CouncilMember(
        id="vishvakarma",
        name="Vishvakarma",
        domain="Technology & Architecture",
        system_prompt=(
            "You are Vishvakarma, the Divine Architect. "
            "Your focus is the 'Shilpa Shastra' (The Science of Creation and Code). "
            "You analyze technical requirements, codebase structure, and engineering tasks. "
            "Your advice should be focused on clean architecture, performance, and technical discipline. "
            "You view code as a craft and system design as a sacred duty. "
            "Speak with the precision of a master engineer."
        ),
        memory_tiers=["codebase", "identity"],
        model_preference=VISHVAKARMA_MODEL,
    ),
}

def get_council_member(expert_id: str) -> Optional[CouncilMember]:
    """Return the CouncilMember from the registry or None."""
    return COUNCIL_REGISTRY.get(expert_id.lower())

def list_council_members() -> str:
    """Return a formatted list of all council members and their domains."""
    lines = ["🏛️ **The Council of Dharma**"]
    for member in COUNCIL_REGISTRY.values():
        lines.append(f"• **{member.name}** ({member.domain})")
    return "\n".join(lines)
