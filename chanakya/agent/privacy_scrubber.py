import re
import logging
from typing import Any
from bson import ObjectId
from chanakya.db.mongo import users, contacts

logger = logging.getLogger(__name__)

# Regex for common PII
PHONE_REGEX = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')

def scrub_context(text: str, user_id: ObjectId) -> str:
    """De-identify sensitive information in the given text.
    
    Replaces:
    - User's own name -> [USER_NAME]
    - Partner's name -> [PARTNER_NAME]
    - Contact names -> [CONTACT_NAME]
    - Phone numbers -> [PHONE]
    - Email addresses -> [EMAIL]
    """
    if not text:
        return text

    # 1. Fetch user data for names
    user = users.find_one({"_id": user_id})
    if not user:
        return text

    replacements = []

    # User's name
    user_name = user.get("name")
    if user_name:
        replacements.append((user_name, "[USER_NAME]"))

    # Partner's name
    rel_config = user.get("relationship_config")
    if isinstance(rel_config, dict):
        partner_name = rel_config.get("partner_name")
        if partner_name:
            replacements.append((partner_name, "[PARTNER_NAME]"))

    # Contact names
    user_contacts = list(contacts.find({"user_id": user_id}))
    for contact in user_contacts:
        c_name = contact.get("name")
        if c_name:
            replacements.append((c_name, f"[CONTACT:{c_name}]"))

    # Sort replacements by length descending to avoid partial matches
    # e.g., if we have "John" and "John Smith", we want to match "John Smith" first.
    replacements.sort(key=lambda x: len(x[0]), reverse=True)

    scrubbed = text

    # Apply name replacements
    for original, token in replacements:
        # Use regex with word boundaries to avoid matching sub-strings
        # e.g., don't match "Priya" in "Priyanka"
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        scrubbed = pattern.sub(token, scrubbed)

    # Apply PII replacements
    scrubbed = PHONE_REGEX.sub("[PHONE]", scrubbed)
    scrubbed = EMAIL_REGEX.sub("[EMAIL]", scrubbed)

    return scrubbed

def unscrub_response(text: str, user_id: ObjectId) -> str:
    """Reverse the de-identification in the LLM response.
    
    Replaces tokens back with original names/values.
    """
    if not text:
        return text

    user = users.find_one({"_id": user_id})
    if not user:
        return text

    # 1. Prepare re-identification map
    re_map = {}

    # User
    user_name = user.get("name")
    if user_name:
        re_map["[USER_NAME]"] = user_name

    # Partner
    rel_config = user.get("relationship_config")
    if isinstance(rel_config, dict):
        partner_name = rel_config.get("partner_name")
        if partner_name:
            re_map["[PARTNER_NAME]"] = partner_name

    # Contacts
    user_contacts = list(contacts.find({"user_id": user_id}))
    for contact in user_contacts:
        c_name = contact.get("name")
        if c_name:
            re_map[f"[CONTACT:{c_name}]"] = c_name

    unscrubbed = text

    # Apply re-map
    for token, original in re_map.items():
        unscrubbed = unscrubbed.replace(token, original)

    # Note: We don't easily unscrub [PHONE] or [EMAIL] unless we stored them specifically
    # for the session. For v1, we assume the LLM won't need to generate specific new phones/emails.
    
    return unscrubbed


def scrub_recursive(data: Any, user_id: ObjectId) -> Any:
    """Recursively scrub all string values in a nested structure."""
    from typing import Any as _Any
    if isinstance(data, str):
        return scrub_context(data, user_id)
    if isinstance(data, dict):
        return {k: scrub_recursive(v, user_id) for k, v in data.items()}
    if isinstance(data, list):
        return [scrub_recursive(v, user_id) for v in data]
    return data


def unscrub_recursive(data: Any, user_id: ObjectId) -> Any:
    """Recursively unscrub all string values in a nested structure."""
    if isinstance(data, str):
        return unscrub_response(data, user_id)
    if isinstance(data, dict):
        return {k: unscrub_recursive(v, user_id) for k, v in data.items()}
    if isinstance(data, list):
        return [unscrub_recursive(v, user_id) for v in data]
    return data
def get_scrub_list(user_id: ObjectId) -> list[str]:
    """Return a list of all real names currently being scrubbed."""
    user = users.find_one({"_id": user_id})
    if not user:
        return []

    names = []
    if user.get("name"):
        names.append(user["name"])

    rel_config = user.get("relationship_config")
    if isinstance(rel_config, dict) and rel_config.get("partner_name"):
        names.append(rel_config["partner_name"])

    user_contacts = list(contacts.find({"user_id": user_id}))
    for contact in user_contacts:
        if contact.get("name"):
            names.append(contact["name"])

    return sorted(list(set(names)))
