"""Local data loading: evidence profile, motivation data, redacted web summary.

Default paths are resolved from the repository root (the package's parent
directory), NOT from ``__file__``, so a ``data_pool.json`` placed next to the
project keeps its ``.env`` discovery behavior identical to the original
monolithic app.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rie.web.store import APIError, _utc_now

# Limits shared by the CLI tools and the web layer.
MAX_REQUEST_BODY_BYTES = 1_048_576
MAX_JOB_DESCRIPTION_CHARS = 50_000

# Repository root = parent of the installed ``rie`` package directory.
# When the package is vendored elsewhere, callers should always pass explicit
# paths; these defaults only serve direct ``python app.py``-style usage.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILE_FILE = _PROJECT_ROOT / "data_pool.json"
MOTIVATION_FILE = _PROJECT_ROOT / "motivation_data.json"


def _read_json_file(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty object for missing/bad local data."""
    try:
        if not path.is_file():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def load_profile(path: str | Path | None = None) -> dict[str, Any]:
    """Load the local evidence profile without raising for missing/bad data."""
    return _read_json_file(Path(path) if path is not None else PROFILE_FILE)


def load_motivation_data(path: str | Path | None = None) -> dict[str, Any]:
    return _read_json_file(Path(path) if path is not None else MOTIVATION_FILE)


def profile_summary(profile: dict[str, Any] | None) -> dict[str, Any]:
    """Return a deliberately small, non-secret profile view for the browser."""
    if not isinstance(profile, dict) or not profile:
        return {"available": False, "name": None, "location": None, "target_roles": [], "skill_categories": {}}

    personal_value = profile.get("personal_info")
    positioning_value = profile.get("positioning")
    preferences_value = profile.get("candidate_preferences")
    skills_value = profile.get("skills")
    personal: dict[str, Any] = personal_value if isinstance(personal_value, dict) else {}
    positioning: dict[str, Any] = positioning_value if isinstance(positioning_value, dict) else {}
    preferences: dict[str, Any] = preferences_value if isinstance(preferences_value, dict) else {}
    skills: dict[str, Any] = skills_value if isinstance(skills_value, dict) else {}
    safe_skills: dict[str, list[str]] = {}
    for category, values in skills.items():
        if isinstance(values, list):
            safe_skills[str(category)] = [str(value) for value in values[:30] if isinstance(value, (str, int, float))]

    education_value = profile.get("education")
    education: dict[str, Any] = education_value if isinstance(education_value, dict) else {}
    return {
        "available": True,
        "name": personal.get("name") or profile.get("profile_owner"),
        "location": preferences.get("current_location") or personal.get("location"),
        "positioning": positioning.get("core_identity"),
        "current_level": positioning.get("current_level"),
        "target_roles": list(positioning.get("target_roles", []))[:20] if isinstance(positioning.get("target_roles"), list) else [],
        "skill_categories": safe_skills,
        "education": {
            "degree": education.get("degree"),
            "university": education.get("university"),
            "year": education.get("year"),
        },
    }


def parse_json_body(
    body: bytes | bytearray | str,
    content_type: str | None = "application/json",
    max_bytes: int = MAX_REQUEST_BODY_BYTES,
) -> dict[str, Any]:
    """Strictly parse one JSON object with an application/json content type."""
    if not content_type or content_type.split(";", 1)[0].strip().lower() != "application/json":
        raise APIError(415, "unsupported_media_type", "Content-Type must be application/json.")
    raw = body.encode("utf-8") if isinstance(body, str) else bytes(body)
    if len(raw) > max_bytes:
        raise APIError(413, "request_too_large", "Request body exceeds the maximum allowed size.")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise APIError(400, "invalid_json", "Request body must contain valid UTF-8 JSON.") from None
    if not isinstance(value, dict):
        raise APIError(400, "invalid_json_shape", "Request JSON must be an object.")
    return value


def validate_job_description(value: Any) -> str:
    if not isinstance(value, str):
        raise APIError(400, "invalid_job_description", "job_description must be a string.")
    description = value.strip()
    if not description:
        raise APIError(400, "empty_job_description", "job_description must not be empty.")
    if len(description) > MAX_JOB_DESCRIPTION_CHARS:
        raise APIError(413, "job_description_too_large", "job_description exceeds the maximum allowed length.")
    return description
