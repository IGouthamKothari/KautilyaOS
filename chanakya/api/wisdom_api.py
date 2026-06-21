"""
wisdom_api.py — REST API for the Living Wisdom System.

Endpoints:
  POST   /api/wisdom          — Add raw insight, auto-extract principle
  GET    /api/wisdom          — List all wisdom entries
  GET    /api/wisdom/{index}  — Get specific entry by index
  PUT    /api/wisdom/{index}  — Update an entry
  DELETE /api/wisdom/{index}  — Remove an entry
  PATCH  /api/wisdom/{index}/toggle — Enable/disable an entry
  GET    /api/wisdom/categories — List entries grouped by category
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/wisdom", tags=["wisdom"])


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class WisdomInput(BaseModel):
    raw_input: str = Field(..., min_length=3, description="Raw experience, story, or insight")
    category: str = ""
    source: str = ""


class WisdomUpdate(BaseModel):
    text: Optional[str] = None
    category: Optional[str] = None
    source: Optional[str] = None
    triggers: Optional[list[str]] = None
    active: Optional[bool] = None


class WisdomEntry(BaseModel):
    id: Optional[str] = None  # stable UUID — use this for addressing in RN client
    index: int
    category: str
    text: str
    raw_input: str = ""
    source: str = ""
    triggers: list[str] = []
    active: bool = True
    times_invoked: int = 0
    added_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_active_user_id() -> ObjectId:
    """Get the primary active user's ID."""
    from chanakya.db.mongo import users
    user = users.find_one({"active": True})
    if not user:
        raise HTTPException(status_code=404, detail="No active user found")
    return user["_id"]


def _entry_to_response(entry: dict, index: int) -> WisdomEntry:
    """Convert a raw DB mindset entry to the API response model."""
    added_at = entry.get("added_at")
    return WisdomEntry(
        id=entry.get("_id"),
        index=index,
        category=entry.get("category", "note"),
        text=entry.get("text", ""),
        raw_input=entry.get("raw_input", ""),
        source=entry.get("source", ""),
        triggers=entry.get("triggers", []),
        active=entry.get("active", True),
        times_invoked=entry.get("times_invoked", 0),
        added_at=added_at.isoformat() if isinstance(added_at, datetime) else str(added_at) if added_at else None,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/", response_model=WisdomEntry, status_code=201)
async def add_wisdom(body: WisdomInput):
    """Add a new wisdom entry. Auto-extracts principle from raw input."""
    from chanakya.api.wisdom_extractor import extract_principle
    from chanakya.db.mongo import add_mindset_entry, get_mindset_entries

    user_id = _get_active_user_id()

    # Extract principle using LLM
    extraction = await extract_principle(
        raw_input=body.raw_input,
        source=body.source,
        hint_category=body.category,
    )

    # Store in DB
    count = add_mindset_entry(
        user_id=user_id,
        category=extraction["category"],
        text=extraction["principle"],
        source=body.source,
        raw_input=body.raw_input,
        triggers=extraction["triggers"],
        active=True,
    )

    # Return the newly created entry
    new_index = count - 1
    entries = get_mindset_entries(user_id)
    if new_index < len(entries):
        return _entry_to_response(entries[new_index], new_index)

    return WisdomEntry(
        index=new_index,
        category=extraction["category"],
        text=extraction["principle"],
        raw_input=body.raw_input,
        source=body.source,
        triggers=extraction["triggers"],
        active=True,
    )


@router.get("/", response_model=list[WisdomEntry])
async def list_wisdom(category: str = "", active_only: bool = False):
    """List all wisdom entries, optionally filtered."""
    from chanakya.db.mongo import get_mindset_entries

    user_id = _get_active_user_id()
    entries = get_mindset_entries(user_id)

    results = []
    for i, entry in enumerate(entries):
        if category and entry.get("category") != category:
            continue
        if active_only and not entry.get("active", True):
            continue
        results.append(_entry_to_response(entry, i))

    return results


@router.get("/categories")
async def list_by_category():
    """List entries grouped by category."""
    from chanakya.db.mongo import get_mindset_entries

    user_id = _get_active_user_id()
    entries = get_mindset_entries(user_id)

    grouped: dict[str, list[WisdomEntry]] = {}
    for i, entry in enumerate(entries):
        cat = entry.get("category", "note")
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(_entry_to_response(entry, i))

    return grouped


@router.get("/{index}", response_model=WisdomEntry)
async def get_wisdom(index: int):
    """Get a specific wisdom entry by index."""
    from chanakya.db.mongo import get_mindset_entry_by_index

    user_id = _get_active_user_id()
    entry = get_mindset_entry_by_index(user_id, index)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No entry at index {index}")
    return _entry_to_response(entry, index)


@router.put("/{index}", response_model=WisdomEntry)
async def update_wisdom(index: int, body: WisdomUpdate):
    """Update fields of a wisdom entry."""
    from chanakya.db.mongo import update_mindset_entry, get_mindset_entry_by_index

    user_id = _get_active_user_id()

    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    success = update_mindset_entry(user_id, index, updates)
    if not success:
        raise HTTPException(status_code=404, detail=f"No entry at index {index}")

    entry = get_mindset_entry_by_index(user_id, index)
    return _entry_to_response(entry, index)


@router.delete("/{index}", status_code=204)
async def delete_wisdom(index: int):
    """Delete a wisdom entry by index."""
    from chanakya.db.mongo import remove_mindset_entry

    user_id = _get_active_user_id()
    success = remove_mindset_entry(user_id, index)
    if not success:
        raise HTTPException(status_code=404, detail=f"No entry at index {index}")


@router.patch("/{index}/toggle", response_model=WisdomEntry)
async def toggle_wisdom(index: int, active: bool = True):
    """Enable or disable a wisdom entry without deleting it."""
    from chanakya.db.mongo import toggle_mindset_entry, get_mindset_entry_by_index

    user_id = _get_active_user_id()
    success = toggle_mindset_entry(user_id, index, active)
    if not success:
        raise HTTPException(status_code=404, detail=f"No entry at index {index}")

    entry = get_mindset_entry_by_index(user_id, index)
    return _entry_to_response(entry, index)


# ── Stable-ID endpoints (use these from the RN client) ───────────────────────

def _find_index_by_id(entries: list, entry_id: str) -> int:
    """Find the index of an entry by its stable _id. Returns -1 if not found."""
    for i, e in enumerate(entries):
        if e.get("_id") == entry_id:
            return i
    return -1


@router.delete("/by-id/{entry_id}", status_code=204)
async def delete_wisdom_by_id(entry_id: str):
    """Delete a wisdom entry by its stable ID (preferred over index-based delete)."""
    from chanakya.db.mongo import remove_mindset_entry, get_mindset_entries

    user_id = _get_active_user_id()
    entries = get_mindset_entries(user_id)
    idx = _find_index_by_id(entries, entry_id)
    if idx == -1:
        raise HTTPException(status_code=404, detail=f"No entry with id {entry_id}")
    from chanakya.db.mongo import remove_mindset_entry
    remove_mindset_entry(user_id, idx)


@router.patch("/by-id/{entry_id}/toggle", response_model=WisdomEntry)
async def toggle_wisdom_by_id(entry_id: str, active: bool = True):
    """Toggle a wisdom entry by its stable ID."""
    from chanakya.db.mongo import toggle_mindset_entry, get_mindset_entries

    user_id = _get_active_user_id()
    entries = get_mindset_entries(user_id)
    idx = _find_index_by_id(entries, entry_id)
    if idx == -1:
        raise HTTPException(status_code=404, detail=f"No entry with id {entry_id}")
    toggle_mindset_entry(user_id, idx, active)
    updated = get_mindset_entries(user_id)
    return _entry_to_response(updated[idx], idx)


@router.put("/by-id/{entry_id}", response_model=WisdomEntry)
async def update_wisdom_by_id(entry_id: str, body: WisdomUpdate):
    """Update a wisdom entry by its stable ID."""
    from chanakya.db.mongo import update_mindset_entry, get_mindset_entries

    user_id = _get_active_user_id()
    entries = get_mindset_entries(user_id)
    idx = _find_index_by_id(entries, entry_id)
    if idx == -1:
        raise HTTPException(status_code=404, detail=f"No entry with id {entry_id}")
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    update_mindset_entry(user_id, idx, updates)
    updated = get_mindset_entries(user_id)
    return _entry_to_response(updated[idx], idx)
