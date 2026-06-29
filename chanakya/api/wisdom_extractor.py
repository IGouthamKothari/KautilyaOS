"""
wisdom_extractor.py — LLM-powered principle extraction from raw experiences.

Takes raw input (a story, reel description, quote, life event) and extracts:
1. A concise, actionable principle (1-2 sentences)
2. Auto-classified category
3. Trigger conditions (when to invoke this principle)
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """You extract life principles from raw experiences, stories, and insights.
Your job: distill the CORE LESSON into a sharp, actionable principle that can guide future decisions.

Rules:
- PRINCIPLE must be 1-2 sentences max. Make it punchy and memorable.
- Think: "What should I remember from this when making decisions?"
- TRIGGERS are situations where this principle should be actively invoked.
- CATEGORY must be exactly one of: quote, goal, trait, rule, reference, note

From this input, extract a JSON object:
{
  "principle": "The core lesson in 1-2 sharp sentences",
  "category": "rule|quote|goal|trait|reference|note",
  "triggers": ["situation 1", "situation 2", "situation 3"]
}

Return ONLY the JSON object. No explanation."""


async def extract_principle(
    raw_input: str,
    source: str = "",
    hint_category: str = "",
) -> dict:
    """Extract an actionable principle from raw experience using the utility LLM.

    Returns:
        {
            "principle": str,
            "category": str,
            "triggers": list[str],
        }

    On failure, returns a best-effort dict using the raw input directly.
    """
    user_message = f"Raw input: {raw_input}"
    if source:
        user_message += f"\nSource: {source}"
    if hint_category:
        user_message += f"\nSuggested category: {hint_category}"

    try:
        from chanakya.agent.llm_provider import call_with_fallback
        content = (await call_with_fallback(
            messages=[
                {"role": "system", "content": _EXTRACTION_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=300,
            timeout=15.0,
        )).strip()

            # Parse JSON from response
            result = _parse_extraction(content)
            if result:
                return result

    except Exception as exc:
        logger.warning("Principle extraction failed: %s", exc)

    # Fallback: use raw input as the principle
    return {
        "principle": raw_input[:200],
        "category": hint_category or "note",
        "triggers": [],
    }


def _parse_extraction(content: str) -> dict | None:
    """Parse LLM output into structured extraction result."""
    # Try direct JSON parse
    try:
        data = json.loads(content)
        return _validate_extraction(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try finding JSON object in text
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(content[start:end])
            return _validate_extraction(data)
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _validate_extraction(data: dict) -> dict:
    """Validate and normalize extraction result."""
    from chanakya.db.mongo import MINDSET_CATEGORIES

    principle = data.get("principle", "")
    if not principle:
        raise ValueError("Empty principle")

    category = data.get("category", "note").lower().strip()
    if category not in MINDSET_CATEGORIES:
        category = "note"

    triggers = data.get("triggers", [])
    if not isinstance(triggers, list):
        triggers = []
    triggers = [str(t).strip() for t in triggers if t][:5]

    return {
        "principle": principle.strip(),
        "category": category,
        "triggers": triggers,
    }
