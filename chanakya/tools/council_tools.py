from __future__ import annotations
import logging
from bson import ObjectId
from langchain_core.tools import tool
from chanakya.agent.council import get_council_member
from chanakya.agent.context_filter import build_expert_context
from chanakya.agent.llm_provider import call_llm

logger = logging.getLogger(__name__)

async def _call_sub_agent(system_prompt: str, context: str, briefing: str, model_name: str) -> str:
    """Execute a call to a sub-agent via the centralized provider."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"USER CONTEXT (GATED):\n{context}\n\nCHANAKYA BRIEFING:\n{briefing}"}
    ]
    return await call_llm(messages, model_name)

@tool
async def consult_council(expert_id: str, briefing: str, user_id: str) -> str:
    """Ask a Council member (Kautilya, Charaka, Vishvakarma) for specialized advice.
    
    The briefing should be a clear, focused question or situation description.
    Chanakya will use the expert's report to provide final guidance to the user.
    """
    member = get_council_member(expert_id)
    if not member:
        return f"Error: '{expert_id}' is not a recognized member of the Council of Dharma."

    try:
        uid = ObjectId(user_id)
    except Exception:
        return f"Error: Invalid user_id {user_id}."

    # 1. Build the Gated Context
    context = build_expert_context(uid, member)
    
    # 2. Call the Sub-Agent
    logger.info(f"Consulting the Council: {member.name} regarding {briefing[:50]}...")
    report = await _call_sub_agent(member.system_prompt, context, briefing, member.model_preference)
    
    # 3. Audit log (Optional but recommended)
    try:
        from chanakya.db.mongo import ai_tool_calls
        from datetime import datetime
        ai_tool_calls.insert_one({
            "user_id": uid,
            "timestamp": datetime.utcnow(),
            "tool_name": "consult_council",
            "tool_input": {"expert_id": expert_id, "briefing": briefing},
            "tool_output": report,
            "expert_name": member.name
        })
    except Exception as e:
        logger.warning(f"Failed to audit council call: {e}")

    return f"--- COUNCIL REPORT FROM {member.name.upper()} ---\n{report}"

ALL_COUNCIL_TOOLS = [consult_council]
