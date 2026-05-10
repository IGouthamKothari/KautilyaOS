from __future__ import annotations
import logging
from datetime import datetime, timedelta
from bson import ObjectId
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chanakya.agent.council import CouncilMember

from chanakya.db.mongo import users, rituals, checkpoints, interaction_logs
from chanakya.agent.privacy_scrubber import scrub_context

logger = logging.getLogger(__name__)

def build_expert_context(user_id: ObjectId, member: CouncilMember) -> str:
    """Assemble a domain-filtered context string for a council member."""
    context_parts = []
    
    # 1. Fetch User Base Data
    user_doc = users.find_one({"_id": user_id})
    if not user_doc:
        return "Error: User not found."

    # 2. Add Identity Context (Allowed for all ministers to know WHO they serve)
    if "identity" in member.memory_tiers:
        identity = user_doc.get("identity_context", "No identity context defined.")
        context_parts.append(f"--- USER IDENTITY ---\n{identity}")

    # 3. Add Health/Ritual Data (Charaka)
    if "rituals" in member.memory_tiers:
        cutoff = datetime.utcnow() - timedelta(days=7)
        recent_rituals = list(rituals.find({
            "user_id": user_id,
            "timestamp": {"$gte": cutoff}
        }).sort("timestamp", -1))
        
        if recent_rituals:
            lines = ["--- RECENT RITUALS (Last 7 Days) ---"]
            for r in recent_rituals:
                lines.append(f"• {r['timestamp'].strftime('%Y-%m-%d %H:%M')}: {r['category']} = {r['value']} ({r.get('note', '')})")
            context_parts.append("\n".join(lines))
        else:
            context_parts.append("--- RECENT RITUALS ---\nNo ritual data found for the last 7 days.")

    # 4. Add Financial Ledger (Kautilya)
    if "ledger" in member.memory_tiers:
        ledger = user_doc.get("accountability_ledger", {"balance": 0, "history": []})
        currency = user_doc.get("currency", "INR")
        lines = ["--- FINANCIAL LEDGER ---"]
        lines.append(f"Balance: {ledger['balance']} {currency}")
        if ledger.get("history"):
            lines.append("Recent History:")
            for h in ledger["history"][-10:]: # Last 10
                lines.append(f"• {h.get('at', datetime.utcnow()).strftime('%Y-%m-%d')}: {h['amount']} {currency} - {h['reason']}")
        context_parts.append("\n".join(lines))

    # 5. Add Codebase Context (Vishvakarma)
    if "codebase" in member.memory_tiers:
        # For MVP, we use the requirement.md as the codebase context
        try:
            with open("requirement.md", "r", encoding="utf-8") as f:
                reqs = f.read()[:5000] # Limit to 5k chars
            context_parts.append(f"--- PROJECT REQUIREMENTS ---\n{reqs}")
        except Exception as e:
            context_parts.append(f"--- PROJECT REQUIREMENTS ---\nError loading requirement.md: {e}")

    # 6. Final Anonymization (Privacy Fortress)
    raw_context = "\n\n".join(context_parts)
    
    if member.privacy_level == "abstracted":
        # Apply privacy scrubber to the assembled context
        return scrub_context(raw_context, user_id)
    
    return raw_context
