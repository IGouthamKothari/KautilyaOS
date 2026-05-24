"""
mongo.py — MongoDB Atlas connection and collection/index setup.

Establishes a PyMongo sync connection pool with exponential backoff on failure.
Provides helpers for all collections and the get_user_with_defaults() utility.
"""

import logging
import time
from urllib.parse import urlparse

import pymongo
from bson import ObjectId
from pymongo import MongoClient
from pymongo.database import Database

from chanakya.config import MONGODB_URI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field defaults applied at runtime when a field is absent (Req 22.4)
# ---------------------------------------------------------------------------

FIELD_DEFAULTS: dict = {
    "streak_count": 0,
    "longest_streak": 0,
    "morning_todo_time": None,
    "morning_todo_fallback_count": 0,
    "checkin_window_start": "09:00",
    "checkin_window_end": "21:00",
    "checkin_min_per_day": 2,
    "checkin_max_per_day": 4,
    "current_activity": "FREE_TIME",
    "activity_slot_updated_at": None,
    "next_day_plan": {},
    "timezone": "Asia/Kolkata",
    "eod_time": "21:00",
    "recurring_failure_patterns": [],
    "warrior_streak": 0,
    "accountability_ledger": {"balance": 0, "history": []},
    "currency": "INR",
}
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_db_name(uri: str, default: str = "chanakya") -> str:
    """Extract the database name from a MongoDB URI, falling back to *default*."""
    try:
        parsed = urlparse(uri)
        # The path component is "/<dbname>"; strip the leading slash.
        path = parsed.path.lstrip("/")
        # Strip any query-string fragment that may have been included in path.
        db_name = path.split("?")[0].strip()
        return db_name if db_name else default
    except Exception:
        return default


def _connect_with_backoff(uri: str) -> MongoClient:
    """
    Create a MongoClient with exponential backoff on connection failure.

    Retry schedule: 1s → 2s → 4s → 8s → 16s → 32s (capped).
    Each failed attempt is logged with the failure reason.
    """
    delay = 1  # seconds
    max_delay = 32
    attempt = 0

    while True:
        attempt += 1
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=5_000)
            # Force a real connection attempt so we catch errors here.
            client.admin.command("ping")
            logger.info("MongoDB connection established on attempt %d.", attempt)
            return client
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MongoDB connection attempt %d failed: %s. Retrying in %ds.",
                attempt,
                exc,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, max_delay)


def _apply_defaults(document: dict) -> dict:
    """
    Return a copy of *document* with FIELD_DEFAULTS applied for any absent key.

    Logs a WARNING for each field that was missing.
    Never raises KeyError.
    """
    result = dict(document)
    for field, default_value in FIELD_DEFAULTS.items():
        if field not in result:
            logger.warning(
                "User document (id=%s) is missing field '%s'; applying default: %r",
                result.get("_id", "unknown"),
                field,
                default_value,
            )
            # Use a copy for mutable defaults to avoid shared-state bugs.
            if isinstance(default_value, (dict, list)):
                result[field] = type(default_value)(default_value)
            else:
                result[field] = default_value
    return result


# ---------------------------------------------------------------------------
# Module-level connection and database handle
# ---------------------------------------------------------------------------

_client: MongoClient = _connect_with_backoff(MONGODB_URI)
_db_name: str = _extract_db_name(MONGODB_URI, default="chanakya")
db: Database = _client[_db_name]

# ---------------------------------------------------------------------------
# Collection handles (2.2)
# ---------------------------------------------------------------------------

users = db["users"]
schedules = db["schedules"]
checkpoints = db["checkpoints"]
interaction_logs = db["interaction_logs"]
ai_tool_calls = db["ai_tool_calls"]
user_state_snapshots = db["user_state_snapshots"]
prompt_templates = db["prompt_templates"]
personal_instructions = db["personal_instructions"]
contacts = db["contacts"]
proxy_call_logs = db["proxy_call_logs"]
daily_events = db["daily_events"]   # date-specific schedule entries & reminders
agent_tasks = db["agent_tasks"]
voice_sessions = db["voice_sessions"]
rituals = db["rituals"]
chat_messages = db["chat_messages"]  # per-user conversation history for LLM context
goals = db["goals"]  # GOAP-inspired goal tracking with milestones


# ---------------------------------------------------------------------------
# Public accessor
# ---------------------------------------------------------------------------


def get_db() -> Database:
    """Return the module-level database handle."""
    return db


# ---------------------------------------------------------------------------
# Index creation (2.3)
# ---------------------------------------------------------------------------


def create_indexes() -> None:
    """
    Create all required indexes for the Chanakya collections.

    This function is idempotent — calling it multiple times is safe.
    """
    # --- checkpoints ---
    checkpoints.create_index(
        [("user_id", pymongo.ASCENDING), ("time", pymongo.ASCENDING), ("active", pymongo.ASCENDING)],
        name="checkpoints_user_time_active",
    )
    checkpoints.create_index(
        [("user_id", pymongo.ASCENDING), ("last_triggered", pymongo.DESCENDING)],
        name="checkpoints_user_last_triggered",
    )

    # --- interaction_logs ---
    interaction_logs.create_index(
        [("user_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)],
        name="interaction_logs_user_timestamp",
    )
    interaction_logs.create_index(
        [
            ("user_id", pymongo.ASCENDING),
            ("ai_evaluation.verdict", pymongo.ASCENDING),
            ("timestamp", pymongo.DESCENDING),
        ],
        name="interaction_logs_user_verdict_timestamp",
    )

    # --- user_state_snapshots ---
    user_state_snapshots.create_index(
        [("user_id", pymongo.ASCENDING), ("date", pymongo.ASCENDING)],
        unique=True,
        name="user_state_snapshots_user_date_unique",
    )
    # NOTE: The Atlas Vector Search index on the `embeddings` field cannot be
    # created via PyMongo.  It must be created manually in the MongoDB Atlas UI
    # under "Search Indexes" using the following definition:
    #
    #   {
    #     "fields": [
    #       {
    #         "type": "vector",
    #         "path": "embeddings",
    #         "numDimensions": <your_embedding_dimension>,
    #         "similarity": "cosine"
    #       }
    #     ]
    #   }
    #
    # Index name suggestion: "user_state_snapshots_embeddings_vector"

    # --- prompt_templates ---
    prompt_templates.create_index(
        [
            ("activity_slot", pymongo.ASCENDING),
            ("interaction_type", pymongo.ASCENDING),
            ("tone", pymongo.ASCENDING),
        ],
        unique=True,
        name="prompt_templates_slot_type_tone_unique",
    )

    # --- daily_events ---
    daily_events.create_index(
        [("user_id", pymongo.ASCENDING), ("date", pymongo.ASCENDING), ("time", pymongo.ASCENDING)],
        name="daily_events_user_date_time",
    )

    # --- agent_tasks ---
    agent_tasks.create_index(
        [("status", pymongo.ASCENDING), ("last_attempted_at", pymongo.ASCENDING)],
        name="agent_tasks_status_last_attempted",
    )

    # --- voice_sessions ---
    # Auto-expire sessions after 1 hour (3600 seconds)
    voice_sessions.create_index("created_at", expireAfterSeconds=3600)

    # --- rituals ---
    rituals.create_index(
        [("user_id", pymongo.ASCENDING), ("category", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)],
        name="rituals_user_category_timestamp",
    )

    # --- chat_messages ---
    chat_messages.create_index(
        [("user_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)],
        name="chat_messages_user_timestamp",
    )

    logger.info("All MongoDB indexes created (or already exist).")


# ---------------------------------------------------------------------------
# Contact helpers
# ---------------------------------------------------------------------------


def get_contacts(user_id: ObjectId) -> list[dict]:
    """Return all contacts for a user."""
    return list(contacts.find({"user_id": user_id}).sort("name", 1))


def get_contact_by_name(user_id: ObjectId, name: str) -> dict | None:
    """Find a contact by name (case-insensitive prefix match)."""
    import re
    pattern = re.compile(re.escape(name.strip()), re.IGNORECASE)
    return contacts.find_one({"user_id": user_id, "name": pattern})


def add_contact(user_id: ObjectId, name: str, phone: str, relationship: str = "") -> dict:
    """Upsert a contact by name. Returns the contact doc."""
    from datetime import datetime as _dt
    now = _dt.utcnow()
    result = contacts.find_one_and_update(
        {"user_id": user_id, "name": name.strip()},
        {
            "$set": {
                "phone": phone.strip(),
                "relationship": relationship.strip(),
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
        return_document=True,
    )
    return result


def remove_contact(user_id: ObjectId, name: str) -> bool:
    """Delete a contact by name. Returns True if deleted."""
    import re
    pattern = re.compile(re.escape(name.strip()), re.IGNORECASE)
    result = contacts.delete_one({"user_id": user_id, "name": pattern})
    return result.deleted_count > 0


# ---------------------------------------------------------------------------
# User helpers (2.4)
# ---------------------------------------------------------------------------


def get_user_with_defaults(telegram_id: str) -> dict | None:
    """
    Fetch a user document by *telegram_id* and apply FIELD_DEFAULTS for any
    absent field.

    Also persists any missing fields back to MongoDB so they don't warn on
    every request.

    Returns:
        The user document with defaults filled in, or ``None`` if no user is found.
    """
    try:
        document = users.find_one({"telegram_id": telegram_id})
    except Exception as exc:  # noqa: BLE001
        logger.error("Error querying users by telegram_id=%r: %s", telegram_id, exc)
        return None

    if document is None:
        return None

    result = _apply_defaults(document)

    # Persist any fields that were missing so we don't warn every time
    missing = {
        field: result[field]
        for field in FIELD_DEFAULTS
        if field not in document
    }
    if missing:
        try:
            users.update_one(
                {"_id": document["_id"]},
                {"$set": missing},
            )
            logger.debug(
                "Persisted %d missing default fields for user %s",
                len(missing),
                document["_id"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist defaults for user %s: %s", document["_id"], exc)

    return result


def get_user_by_id(user_id: ObjectId) -> dict | None:
    """
    Fetch a user document by MongoDB *_id* and apply FIELD_DEFAULTS for any
    absent field.

    Returns:
        The user document with defaults filled in (in-memory only — the
        database document is NOT modified), or ``None`` if no user is found.

    Never raises ``KeyError`` under any circumstances.
    """
    try:
        document = users.find_one({"_id": user_id})
    except Exception as exc:  # noqa: BLE001
        logger.error("Error querying users by _id=%r: %s", user_id, exc)
        return None

    if document is None:
        return None

    return _apply_defaults(document)


# ---------------------------------------------------------------------------
# Personal instructions helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# In-process identity cache — eliminates MongoDB round-trip on every LLM call.
#
# Structure: { str(user_id): {"instructions": [...], "mindset": [...]} }
# Invalidated on every write (add, remove, clear).
# Single-process safe — works fine for one Uvicorn worker (Render free tier).
# If you ever scale to multiple workers, replace with Redis.
# ---------------------------------------------------------------------------
_identity_cache: dict[str, dict] = {}


def _cache_key(user_id: ObjectId) -> str:
    return str(user_id)


def _invalidate_identity_cache(user_id: ObjectId) -> None:
    """Drop the cached identity doc for this user. Called on every write."""
    _identity_cache.pop(_cache_key(user_id), None)
    logger.debug("Identity cache invalidated for user %s", user_id)


def get_personal_instructions(user_id: ObjectId) -> list[str]:
    """Return the user's personal instruction list (served from cache)."""
    return get_all_identity_context(user_id).get("instructions", [])


def add_personal_instruction(user_id: ObjectId, text: str) -> int:
    """Append an instruction. Invalidates cache. Returns the new total count."""
    from datetime import datetime as _dt
    result = personal_instructions.find_one_and_update(
        {"user_id": user_id},
        {
            "$push": {"instructions": text.strip()},
            "$set": {"updated_at": _dt.utcnow()},
            "$setOnInsert": {"created_at": _dt.utcnow()},
        },
        upsert=True,
        return_document=True,
    )
    _invalidate_identity_cache(user_id)
    return len(result.get("instructions", [text]))


def clear_personal_instructions(user_id: ObjectId) -> None:
    """Remove all personal instructions. Invalidates cache."""
    from datetime import datetime as _dt
    personal_instructions.update_one(
        {"user_id": user_id},
        {"$set": {"instructions": [], "updated_at": _dt.utcnow()}},
        upsert=True,
    )
    _invalidate_identity_cache(user_id)


def remove_personal_instruction(user_id: ObjectId, index: int) -> bool:
    """Remove instruction at 0-based index. Invalidates cache. Returns True if removed."""
    doc = personal_instructions.find_one({"user_id": user_id})
    if not doc:
        return False
    items = doc.get("instructions", [])
    if index < 0 or index >= len(items):
        return False
    items.pop(index)
    from datetime import datetime as _dt
    personal_instructions.update_one(
        {"user_id": user_id},
        {"$set": {"instructions": items, "updated_at": _dt.utcnow()}},
    )
    _invalidate_identity_cache(user_id)
    return True


# ---------------------------------------------------------------------------
# Mindset / identity helpers — typed runtime entries
# ---------------------------------------------------------------------------
# Each mindset entry is a dict:
#   {
#     "category": "quote" | "goal" | "trait" | "rule" | "reference" | "note",
#     "text": str,          # the content
#     "source": str,        # optional — "Harvey Specter", "Bhagavad Gita 2.47", etc.
#     "added_at": datetime,
#   }
#
# Categories:
#   quote     — a quote to embody or be reminded of
#   goal      — a life goal (billionaire, master DSA, etc.)
#   trait     — a character trait to build (discipline, fearlessness)
#   rule      — a personal rule / principle ("never negotiate with comfort")
#   reference — a story/person Chanakya should reference ("when I doubt, use Arjuna")
#   note      — anything else
# ---------------------------------------------------------------------------

MINDSET_CATEGORIES = {"quote", "goal", "trait", "rule", "reference", "note"}


def get_mindset_entries(user_id: ObjectId) -> list[dict]:
    """Return all typed mindset entries (served from cache)."""
    return get_all_identity_context(user_id).get("mindset", [])


def add_mindset_entry(
    user_id: ObjectId,
    category: str,
    text: str,
    source: str = "",
    raw_input: str = "",
    triggers: list[str] | None = None,
    active: bool = True,
) -> int:
    """Append a typed mindset entry. Invalidates cache. Returns new total count."""
    from datetime import datetime as _dt

    category = category.lower().strip()
    if category not in MINDSET_CATEGORIES:
        category = "note"

    entry = {
        "category": category,
        "text": text.strip(),
        "source": source.strip(),
        "raw_input": raw_input.strip() if raw_input else "",
        "triggers": triggers or [],
        "active": active,
        "times_invoked": 0,
        "added_at": _dt.utcnow(),
    }

    result = personal_instructions.find_one_and_update(
        {"user_id": user_id},
        {
            "$push": {"mindset": entry},
            "$set": {"updated_at": _dt.utcnow()},
            "$setOnInsert": {"created_at": _dt.utcnow()},
        },
        upsert=True,
        return_document=True,
    )
    _invalidate_identity_cache(user_id)
    return len(result.get("mindset", [entry]))


def remove_mindset_entry(user_id: ObjectId, index: int) -> bool:
    """Remove mindset entry at 0-based index. Invalidates cache. Returns True if removed."""
    # Read from DB directly to get freshest state before mutating
    doc = personal_instructions.find_one({"user_id": user_id})
    if not doc:
        return False
    items = doc.get("mindset", [])
    if index < 0 or index >= len(items):
        return False
    items.pop(index)
    from datetime import datetime as _dt
    personal_instructions.update_one(
        {"user_id": user_id},
        {"$set": {"mindset": items, "updated_at": _dt.utcnow()}},
    )
    _invalidate_identity_cache(user_id)
    return True


def clear_mindset_entries(user_id: ObjectId) -> None:
    """Remove all mindset entries. Invalidates cache."""
    from datetime import datetime as _dt
    personal_instructions.update_one(
        {"user_id": user_id},
        {"$set": {"mindset": [], "updated_at": _dt.utcnow()}},
        upsert=True,
    )
    _invalidate_identity_cache(user_id)


def get_mindset_entry_by_index(user_id: ObjectId, index: int) -> dict | None:
    """Return a single mindset entry by 0-based index, or None."""
    doc = personal_instructions.find_one({"user_id": user_id})
    if not doc:
        return None
    items = doc.get("mindset", [])
    if index < 0 or index >= len(items):
        return None
    return items[index]


def update_mindset_entry(user_id: ObjectId, index: int, updates: dict) -> bool:
    """Update specific fields of a mindset entry at given index. Returns True if updated."""
    from datetime import datetime as _dt

    doc = personal_instructions.find_one({"user_id": user_id})
    if not doc:
        return False
    items = doc.get("mindset", [])
    if index < 0 or index >= len(items):
        return False

    allowed_fields = {"text", "category", "source", "raw_input", "triggers", "active"}
    for key, value in updates.items():
        if key in allowed_fields:
            items[index][key] = value

    personal_instructions.update_one(
        {"user_id": user_id},
        {"$set": {"mindset": items, "updated_at": _dt.utcnow()}},
    )
    _invalidate_identity_cache(user_id)
    return True


def toggle_mindset_entry(user_id: ObjectId, index: int, active: bool) -> bool:
    """Enable or disable a mindset entry. Returns True if toggled."""
    return update_mindset_entry(user_id, index, {"active": active})


def get_all_identity_context(user_id: ObjectId) -> dict:
    """Return both flat instructions and typed mindset entries.

    Served from in-process cache after the first fetch.
    Cache is invalidated on every write so it's always consistent.
    """
    key = _cache_key(user_id)
    if key in _identity_cache:
        logger.debug("Identity cache hit for user %s", user_id)
        return _identity_cache[key]

    doc = personal_instructions.find_one({"user_id": user_id})
    result = {
        "instructions": doc.get("instructions", []) if doc else [],
        "mindset": doc.get("mindset", []) if doc else [],
    }
    _identity_cache[key] = result
    logger.debug("Identity cache populated for user %s (%d instructions, %d mindset entries)",
                 user_id, len(result["instructions"]), len(result["mindset"]))
    return result


# ---------------------------------------------------------------------------
# Chat history helpers
# ---------------------------------------------------------------------------


def store_chat_message(
    user_id: ObjectId,
    role: str,
    content: str,
    channel: str = "text",
) -> None:
    """Store a message in the chat history for later retrieval."""
    from datetime import datetime as _dt
    chat_messages.insert_one({
        "user_id": user_id,
        "role": role,
        "content": content,
        "channel": channel,
        "timestamp": _dt.utcnow(),
    })


def get_recent_messages(user_id: ObjectId, limit: int = 5) -> list[dict]:
    """Fetch the last N messages for a user, oldest-first."""
    docs = list(
        chat_messages.find(
            {"user_id": user_id},
            sort=[("timestamp", pymongo.DESCENDING)],
            limit=limit,
        )
    )
    docs.reverse()
    return [{"role": d["role"], "content": d["content"]} for d in docs]


def get_message_count(user_id: ObjectId) -> int:
    """Return total stored messages for a user."""
    return chat_messages.count_documents({"user_id": user_id})


def trim_old_messages(user_id: ObjectId, keep: int = 10) -> int:
    """Delete messages older than the most recent `keep` messages. Returns count deleted."""
    docs = list(
        chat_messages.find(
            {"user_id": user_id},
            sort=[("timestamp", pymongo.DESCENDING)],
            limit=keep,
            projection={"_id": 1},
        )
    )
    if len(docs) < keep:
        return 0
    keep_ids = [d["_id"] for d in docs]
    result = chat_messages.delete_many(
        {"user_id": user_id, "_id": {"$nin": keep_ids}}
    )
    return result.deleted_count


# ---------------------------------------------------------------------------
# Goals — GOAP-inspired tracking
# ---------------------------------------------------------------------------


def create_goal(
    user_id: ObjectId,
    title: str,
    description: str = "",
    category: str = "general",
    target_date: str | None = None,
    milestones: list[dict] | None = None,
) -> str:
    """Create a new goal with optional milestones. Returns the goal ID as string."""
    from datetime import datetime as _dt

    doc = {
        "user_id": user_id,
        "title": title,
        "description": description,
        "category": category,
        "status": "active",
        "progress": 0,
        "target_date": target_date,
        "milestones": milestones or [],
        "notes": [],
        "created_at": _dt.utcnow(),
        "updated_at": _dt.utcnow(),
    }
    result = goals.insert_one(doc)
    return str(result.inserted_id)


def get_goals(user_id: ObjectId, status: str | None = None) -> list[dict]:
    """Get all goals for a user, optionally filtered by status."""
    query: dict = {"user_id": user_id}
    if status:
        query["status"] = status
    docs = list(goals.find(query, sort=[("created_at", -1)]))
    for d in docs:
        d["_id"] = str(d["_id"])
        d.pop("user_id", None)
    return docs


def get_goal_by_id(user_id: ObjectId, goal_id: str) -> dict | None:
    """Get a specific goal by ID."""
    from bson import ObjectId as _OID
    try:
        doc = goals.find_one({"_id": _OID(goal_id), "user_id": user_id})
    except Exception:
        return None
    if doc:
        doc["_id"] = str(doc["_id"])
        doc.pop("user_id", None)
    return doc


def update_goal_progress(
    user_id: ObjectId,
    goal_id: str,
    progress: int | None = None,
    note: str | None = None,
    milestone_index: int | None = None,
    milestone_done: bool = True,
) -> bool:
    """Update goal progress, add a note, or mark a milestone complete."""
    from bson import ObjectId as _OID
    from datetime import datetime as _dt

    try:
        oid = _OID(goal_id)
    except Exception:
        return False

    updates: dict = {"$set": {"updated_at": _dt.utcnow()}}

    if progress is not None:
        updates["$set"]["progress"] = min(max(progress, 0), 100)
        if progress >= 100:
            updates["$set"]["status"] = "completed"
            updates["$set"]["completed_at"] = _dt.utcnow()

    if note:
        if "$push" not in updates:
            updates["$push"] = {}
        updates["$push"]["notes"] = {"text": note, "at": _dt.utcnow()}

    result = goals.update_one({"_id": oid, "user_id": user_id}, updates)

    if milestone_index is not None and result.modified_count > 0 or milestone_index is not None:
        goals.update_one(
            {"_id": oid, "user_id": user_id},
            {"$set": {f"milestones.{milestone_index}.done": milestone_done}},
        )

    return result.modified_count > 0 or milestone_index is not None


def abandon_goal(user_id: ObjectId, goal_id: str, reason: str = "") -> bool:
    """Mark a goal as abandoned."""
    from bson import ObjectId as _OID
    from datetime import datetime as _dt

    try:
        oid = _OID(goal_id)
    except Exception:
        return False

    result = goals.update_one(
        {"_id": oid, "user_id": user_id},
        {"$set": {"status": "abandoned", "abandon_reason": reason, "updated_at": _dt.utcnow()}},
    )
    return result.modified_count > 0


# ---------------------------------------------------------------------------
# Startup: create indexes once at module import time
# ---------------------------------------------------------------------------

create_indexes()
