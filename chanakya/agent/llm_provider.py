from __future__ import annotations
import logging
import httpx
from typing import List, Dict, Any, Optional
from chanakya.config import OPENAI_API_KEY, OPENROUTER_API_KEY

logger = logging.getLogger(__name__)

async def call_llm(
    messages: List[Dict[str, str]], 
    model_name: str, 
    temperature: float = 0.4,
    max_tokens: Optional[int] = None
) -> str:
    """Centralized LLM execution engine for all agents."""
    
    # Identify Provider
    if model_name.startswith("openrouter/"):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://chanakya.ai", # Optional for OpenRouter ranking
            "X-Title": "Chanakya Dharma Engine"
        }
        api_model = model_name.replace("openrouter/", "")
    else:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        api_model = model_name

    payload = {
        "model": api_model,
        "messages": messages,
    }
    
    # Some frontier/nano models don't support custom temperature
    if "nano" not in api_model.lower() and "mini" not in api_model.lower():
        payload["temperature"] = temperature
        
    if max_tokens:
        payload["max_tokens"] = max_tokens

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        logger.error(f"LLM API Error ({api_model}): {e.response.text}")
        return f"Error: The Guru's mind is clouded by API issues ({e.response.status_code})."
    except Exception as e:
        logger.error(f"Unexpected LLM Failure ({api_model}): {e}")
        return f"Error: Unexpected failure in LLM execution: {e}"
