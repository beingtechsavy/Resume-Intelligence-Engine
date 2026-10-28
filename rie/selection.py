"""Deterministic, role-aware selection of resume evidence.

The profile is intentionally much larger than any one resume.  This module
turns the full evidence pool into a small, explainable, job-specific profile
before a provider sees it.  The selection policy is conservative:

* technical JDs are allowed to use technical experience and relevant projects;
* content/community experiences are not allowed to masquerade as technical
  experience merely because they are recent or called an internship;
* ambiguous, copied, or unverified material is excluded unless the profile's
  explicit policy says otherwise; and
* the model receives only selected evidence, not the raw pool it could
  mis-prioritize.

This is not an ATS score.  It is a local evidence-routing layer that keeps
resume drafting tied to the job description and the candidate's claim policy.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rie.matching import (
    SKILL_RULES,
    _alias_in_text,
    _detected_role_families,
    _detected_skills,
    _normalize,
)


TECHNICAL_ROLE_FAMILIES = frozenset(
    {
        "ai_support",
        "technical_support",
        "implementation_customer",
        "software_development",
        "data_operations",
        "research_iot",
    }
)

TECHNICAL_ADJACENT_ROLE_FAMILIES = frozenset(
    {"project_delivery", "technical_communication"}
)

CONTENT_ROLE_FAMILIES = frozenset(
    {"content_community", "sales_business"}
)

NON_TECHNICAL_SKILL_SIGNALS = frozenset(
    {"documentation", "customer_communication", "project_coordination"}
)

PROVISIONAL_SKILL_CATEGORIES = frozenset(
    {"historical_self_report_only", "learning_exposure", "not_established_from_copied_material"}
)

GENERIC_SELECTION_SKILLS = frozenset({"rest_http_json", "git", "documentation"})

USER_PRIORITY_VALUES = frozenset({"primary", "secondary", "keyword_dependent", "exclude"})
USER_PRIORITY_WEIGHTS = {
    "primary": 90,
    "secondary": 42,
    "keyword_dependent": 8,
    "exclude": -120,
}
USER_PRIORITY_ORDER = {
    "primary": 0,
    "secondary": 1,
    "keyword_dependent": 2,
    "exclude": 3,
}

ONE_PAGE_SELECTION_LIMITS = {
    "experience": 2,
    "projects": 3,
    "learning_projects": 0,
    "documented_concepts": 0,
    "open_source_contributions": 0,
    "certifications": 2,
    "achievements": 0,
    "learning_activities": 0,
}

SELECTION_LIMITS = ONE_PAGE_SELECTION_LIMITS

_COMPACT_FIELDS = (
    "id",
    "name",
    "role",
    "organization",
    "repository",
    "duration",
    "location",
    "type",
    "status",
    "classification",
    "description",
    "tech_stack",
    "responsibilities_reported",
    "resume_ready_bullets",
    "impact",
    "data_work",
    "tags",
    "details",
    "scope",
    "issuer",
    "completion_year",
    "completion_period",
    "completed_on",
    "issued_on",
    "overall_band",
    "cefr_level",
    "activity",
    "source_ids",
)


def _flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _flatten_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _flatten_strings(child)
    elif isinstance(value, str):
        yield value


def _item_key(item: dict[str, Any]) -> str:
    for key in ("id", "name", "activity", "repository", "role"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return "unknown"


def _item_label(item: dict[str, Any]) -> str:
    for key in ("name", "role", "activity", "repository", "id"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return "Unnamed evidence"


def _item_text(item: dict[str, Any]) -> str:
    return " ".join(_flatten_strings(item))


def _index_for(profile: dict[str, Any], section: str, item: dict[str, Any]) -> dict[str, Any]:
    index = profile.get("resume_selection_index")
    if not isinstance(index, dict):
        return _fallback_classification(section, item)
    section_index = index.get(section)
    if not isinstance(section_index, dict):
        return _fallback_classification(section, item)
    metadata = section_index.get(_item_key(item))
    if isinstance(metadata, dict):
        return dict(metadata)
    # Certificates and achievements are keyed by their display name.  This
    # fallback also supports profiles that use a different stable identifier.
    metadata = section_index.get(_item_label(item))
    if isinstance(metadata, dict):
        return dict(metadata)
    return _fallback_classification(section, item)


def _fallback_classification(section: str, item: dict[str, Any]) -> dict[str, Any]:
    """Keep older/example profiles usable when they lack the new index."""
    text = _normalize(_item_text(item))
    if section == "experience":
        if any(term in text for term in ("podcast", "social media", "marketing", "content")):
            return {
                "category": "content_operations",
                "priority": "conditional",
                "technicality": "low",
                "role_families": ["content_community"],
            }
        if any(term in text for term in ("research", "software", "python", "iot", "technical", "cloud", "data")):
            return {
                "category": "technical_experience",
                "priority": "primary",
                "technicality": "high",
                "role_families": ["software_development", "data_operations", "research_iot"],
            }
        return {
            "category": "general_experience",
            "priority": "conditional",
            "technicality": "low",
            "role_families": [],
        }
    if section == "projects":
        return {
            "category": "technical_project",
            "priority": "primary",
            "technicality": "high" if item.get("tech_stack") else "medium",
            "role_families": ["software_development"],
        }
    if section in {"documented_concepts", "learning_projects"}:
        return {
            "category": "learning_or_concept",
            "priority": "learning",
            "technicality": "low",
            "role_families": ["software_development"],
        }
    if section == "certifications":
        return {
            "category": "technical_credential",
            "priority": "supporting",
            "technicality": "medium",
            "role_families": ["software_development", "data_operations"],
        }
    return {
        "category": "supporting_evidence",
        "priority": "conditional",
        "technicality": "low",
        "role_families": [],
    }


def _matched_skills(text: str, detected_skills: list[str]) -> list[str]:
    normalized = _normalize(text)
    return [
        skill
        for skill in detected_skills
        if any(_alias_in_text(alias, normalized) for alias in SKILL_RULES.get(skill, ()))
    ]


def _matched_roles(metadata: dict[str, Any], detected_roles: list[str]) -> list[str]:
    roles = metadata.get("role_families", [])
    if not isinstance(roles, list):
        return []
    return [role for role in detected_roles if role in roles]


def _technical_demand(job_text: str, roles: list[str], skills: list[str]) -> bool:
    normalized = _normalize(job_text)
    explicit_terms = (
        "technical",
        "engineering",
        "developer",
        "software",
        "api",
        "database",
        "debug",
        "troubleshoot",
        "implementation",
        "automation",
        "data",
        "cloud",
        "iot",
        "ai",
        "llm",
    )
    specific_skills = set(skills) - NON_TECHNICAL_SKILL_SIGNALS
    explicit_technical = any(_alias_in_text(term, normalized) for term in explicit_terms)
    technical_roles = set(roles) & TECHNICAL_ROLE_FAMILIES
    if set(roles) & CONTENT_ROLE_FAMILIES and not technical_roles and not specific_skills and not explicit_technical:
        return False
    return bool(technical_roles or specific_skills or explicit_technical)


def _content_demand(job_text: str, roles: list[str]) -> bool:
    normalized = _normalize(job_text)
    explicit_terms = (
        "podcast",
        "content",
        "social media",
        "marketing",
        "communications",
        "community",
        "media production",
        "brand",
    )
    return bool(set(roles) & CONTENT_ROLE_FAMILIES or any(term in normalized for term in explicit_terms))


def _score(
    metadata: dict[str, Any],
    matched_roles: list[str],
    matched_skills: list[str],
    technical_demand: bool,
    content_demand: bool,
    section: str,
) -> int:
    priority_weight = {
        "primary": 55,
        "supporting": 30,
        "conditional": 15,
        "learning": 5,
        "exclude": -100,
    }
    technicality_weight = {"high": 24, "medium": 12, "low": 2, "none": -10}
    score = priority_weight.get(str(metadata.get("priority", "conditional")), 10)
    user_priority = str(metadata.get("user_priority", ""))
    score += USER_PRIORITY_WEIGHTS.get(user_priority, 0)
    score += min(36, len(matched_roles) * 18)
    score += min(30, len(matched_skills) * 10)
    score += technicality_weight.get(str(metadata.get("technicality", "low")), 0)

    category = str(metadata.get("category", ""))
    if technical_demand and section == "experience":
        if category in {"technical_experience", "technical_community", "technical_communication"}:
            score += 18
        elif category in {"content_operations", "community_creative", "community_membership", "general_experience"}:
            score -= 42
    if technical_demand and section in {"projects", "learning_projects", "documented_concepts"}:
        if category in {"technical_project", "technical_application", "technical_data_project"}:
            score += 12
        elif category in {"learning_or_concept", "copied_material", "documentation_only_concept"}:
            score -= 18
    if content_demand and category in {"content_operations", "technical_content", "community_creative"}:
        score += 28
    if metadata.get("evidence_status") in {"ambiguous", "unverified", "self_report_only"}:
        score -= 30
    if metadata.get("default_action") == "exclude":
        score -= 100
    return score


def _eligible(
    section: str,
    metadata: dict[str, Any],
    matched_roles: list[str],
    matched_skills: list[str],
    detected_skills: list[str],
    technical_demand: bool,
    content_demand: bool,
    score: int,
) -> tuple[bool, str]:
    category = str(metadata.get("category", ""))
    priority = str(metadata.get("priority", "conditional"))
    user_priority = str(metadata.get("user_priority", ""))
    meaningful_skill_matches = set(matched_skills) - NON_TECHNICAL_SKILL_SIGNALS
    direct_match = bool(matched_roles or matched_skills or content_demand)
    if priority == "exclude" or metadata.get("default_action") == "exclude":
        return False, "Excluded by the evidence classification policy."
    if user_priority == "exclude":
        return False, "Excluded by the user's resume inclusion policy."
    if user_priority in {"secondary", "keyword_dependent"} and not direct_match:
        label = "keyword-dependent" if user_priority == "keyword_dependent" else "secondary"
        return False, f"Suppressed: {label} evidence requires a direct role, skill, or content match."

    if content_demand and not technical_demand:
        if section == "experience" and category == "technical_experience" and not meaningful_skill_matches:
            return False, "Suppressed: technical experience has no direct content or communication bridge for this JD."
        if section in {"projects", "learning_projects", "documented_concepts"} and not (
            matched_roles or meaningful_skill_matches
        ):
            return False, "Suppressed: technical project evidence has no direct content or communication bridge."

    if section == "experience" and technical_demand and not content_demand:
        if category not in {"technical_experience", "technical_communication", "technical_community"}:
            return False, "Suppressed: this is non-technical experience for a technical JD."
        if category == "technical_experience" and detected_skills:
            specific_jd_skills = set(detected_skills) - GENERIC_SELECTION_SKILLS
            specific_matches = set(matched_skills) - GENERIC_SELECTION_SKILLS
            direct_role_matches = set(matched_roles) - {"software_development"}
            if specific_jd_skills and not specific_matches and not direct_role_matches:
                return False, "Suppressed: technical experience has no direct overlap with the JD's specific skills."
        if category in {"technical_communication", "technical_community"} and not (
            matched_roles or matched_skills
        ):
            return False, "Suppressed: communication/community evidence has no direct JD bridge."

    if section in {"learning_projects", "documented_concepts"}:
        if priority == "learning" and not (matched_roles or matched_skills):
            return False, "Suppressed: learning/concept evidence is not a direct JD match."

    if section == "projects" and technical_demand and detected_skills and user_priority != "primary":
        specific_jd_skills = set(detected_skills) - GENERIC_SELECTION_SKILLS
        specific_matches = set(matched_skills) - GENERIC_SELECTION_SKILLS
        if specific_jd_skills and not specific_matches:
            return False, "Suppressed: the project has no specific overlap with the JD's technical skills."
        if not specific_jd_skills and not matched_skills:
            return False, "Suppressed: the project has no explicit overlap with the JD's detected technical skills."

    if section == "experience" and metadata.get("evidence_status") in {"ambiguous", "unverified"}:
        return False, "Suppressed: role scope or employment status still needs confirmation."

    if priority == "conditional" and not (matched_roles or matched_skills or content_demand):
        return False, "Suppressed: no direct role or skill match was found."

    minimum = 34 if section in {"experience", "projects"} else 26
    if score < minimum and not (matched_roles or matched_skills):
        return False, "Suppressed: relevance did not clear the conservative selection threshold."
    return True, "Selected by role family, skill overlap, and evidence classification."


def _record(
    section: str,
    item: dict[str, Any],
    metadata: dict[str, Any],
    detected_roles: list[str],
    detected_skills: list[str],
    technical_demand: bool,
    content_demand: bool,
) -> dict[str, Any]:
    text = _item_text(item)
    matched_roles = _matched_roles(metadata, detected_roles)
    matched_skills = _matched_skills(text, detected_skills)
    score = _score(
        metadata,
        matched_roles,
        matched_skills,
        technical_demand,
        content_demand,
        section,
    )
    selected, reason = _eligible(
        section,
        metadata,
        matched_roles,
        matched_skills,
        detected_skills,
        technical_demand,
        content_demand,
        score,
    )
    return {
        "section": section,
        "key": _item_key(item),
        "label": _item_label(item),
        "category": metadata.get("category", "supporting_evidence"),
        "priority": metadata.get("priority", "conditional"),
        "user_priority": metadata.get("user_priority", ""),
        "technicality": metadata.get("technicality", "low"),
        "role_families": metadata.get("role_families", []),
        "matched_role_families": matched_roles,
        "matched_skills": matched_skills,
        "score": score,
        "selected": selected,
        "reason": reason,
        "item": item,
    }


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "section",
            "key",
            "label",
            "category",
            "priority",
            "user_priority",
            "technicality",
            "matched_role_families",
            "matched_skills",
            "score",
            "reason",
        )
    }


def _choose(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    eligible = [record for record in records if record["selected"]]
    eligible.sort(
        key=lambda record: (
            USER_PRIORITY_ORDER.get(str(record.get("user_priority", "")), 2),
            record["priority"] != "primary",
            -int(record["score"]),
            record["label"].casefold(),
        )
    )
    return eligible[:limit] if limit > 0 else []


def _section_items(profile: dict[str, Any], section: str) -> list[dict[str, Any]]:
    value = profile.get(section, [])
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def select_resume_evidence(profile: dict[str, Any], job_text: str) -> dict[str, Any]:
    """Return a deterministic selection report for one job description."""
    if not isinstance(profile, dict):
        raise ValueError("profile must be an object")
    if not isinstance(job_text, str) or not job_text.strip():
        raise ValueError("job_text must not be empty")

    roles = _detected_role_families(job_text)
    skills = _detected_skills(job_text)
    technical_demand = _technical_demand(job_text, roles, skills)
    content_demand = _content_demand(job_text, roles)

    all_records: list[dict[str, Any]] = []
    for section in SELECTION_LIMITS:
        for item in _section_items(profile, section):
            all_records.append(
                _record(
                    section,
                    item,
                    _index_for(profile, section, item),
                    roles,
                    skills,
                    technical_demand,
                    content_demand,
                )
            )

    selected: dict[str, list[dict[str, Any]]] = {}
    suppressed: list[dict[str, Any]] = []
    for section, limit in SELECTION_LIMITS.items():
        section_records = [record for record in all_records if record["section"] == section]
        selected_records = _choose(section_records, limit)
        selected[section] = [_public_record(record) for record in selected_records]
        selected_keys = {record["key"] for record in selected_records}
        suppressed.extend(
            _public_record(record)
            for record in section_records
            if record["key"] not in selected_keys
        )

    selected_experience = selected.get("experience", [])
    selected_projects = selected.get("projects", [])
    has_technical_experience = any(
        item.get("category") == "technical_experience" for item in selected_experience
    )
    section_order = ["experience", "projects"] if has_technical_experience else ["projects", "experience"]
    section_order.extend(["education", "skills", "certifications", "achievements"])

    selected_count = sum(len(items) for items in selected.values())
    suppressed_count = len(suppressed)
    return {
        "schema_version": "resume-selection-2",
        "role_families": roles,
        "detected_skills": skills,
        "technical_demand": technical_demand,
        "content_demand": content_demand,
        "section_order": section_order,
        "selected": selected,
        "suppressed": suppressed,
        "summary": {
            "selected_items": selected_count,
            "suppressed_items": suppressed_count,
            "selected_experience": len(selected_experience),
            "selected_projects": len(selected_projects),
            "technical_experience_present": has_technical_experience,
        },
        "page_budget": {
            "target_pages": 1,
            "max_experience_entries": ONE_PAGE_SELECTION_LIMITS["experience"],
            "max_project_entries": ONE_PAGE_SELECTION_LIMITS["projects"],
            "max_certifications": ONE_PAGE_SELECTION_LIMITS["certifications"],
            "max_skill_rows": 3,
            "max_bullets_per_entry": 2,
            "instruction": "If space is tight, omit lower-scoring selected evidence. Never add a second page.",
        },
        "guardrails": [
            "Only selected evidence may be used in the resume or cover letter.",
            "Primary user-priority projects and experience may be used by default when evidence status permits.",
            "Secondary user-priority evidence requires a direct role, skill, or content bridge.",
            "Keyword-dependent user-priority evidence requires a direct job-description signal and remains rare.",
            "Excluded or omitted user-priority evidence must not be reintroduced by the drafting model.",
            "Do not promote suppressed experience into a different category because it is recent or called an internship.",
            "Do not use provisional, ambiguous, copied, or unverified material as established technical experience.",
            "Use detected job terms only when the selected evidence supports the claim.",
            "The resume has a one-page budget: keep only the highest-scoring selected evidence.",
        ],
    }


def _selected_keys(selection: dict[str, Any], section: str) -> set[str]:
    keys: set[str] = set()
    for item in selection.get("selected", {}).get(section, []):
        if isinstance(item, dict) and isinstance(item.get("key"), str):
            keys.add(item["key"])
    return keys


def _compact_item(item: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {key: item[key] for key in _COMPACT_FIELDS if key in item}
    if isinstance(item.get("limitations"), list):
        compact["claim_limits"] = item["limitations"][:4]
    compact["selection"] = {
        "category": record.get("category"),
        "priority": record.get("priority"),
        "user_priority": record.get("user_priority"),
        "technicality": record.get("technicality"),
        "matched_role_families": record.get("matched_role_families", []),
        "matched_skills": record.get("matched_skills", []),
        "reason": record.get("reason"),
    }
    return compact


def _selected_items(profile: dict[str, Any], selection: dict[str, Any], section: str) -> list[dict[str, Any]]:
    keys = _selected_keys(selection, section)
    records = {
        record["key"]: record
        for record in selection.get("selected", {}).get(section, [])
        if isinstance(record, dict)
    }
    return [
        _compact_item(item, records[_item_key(item)])
        for item in _section_items(profile, section)
        if _item_key(item) in keys and _item_key(item) in records
    ]


def _skill_value_matches(value: str, job_text: str, selected_text: str) -> bool:
    normalized_value = _normalize(value)
    if normalized_value in _normalize(selected_text):
        return True
    return normalized_value in _normalize(job_text)


def _selected_skills(
    profile: dict[str, Any],
    selection: dict[str, Any],
    job_text: str,
) -> dict[str, list[str]]:
    skills = profile.get("skills", {})
    if not isinstance(skills, dict):
        return {}
    selected_text = " ".join(
        _item_text(item)
        for section in ("experience", "projects", "learning_projects", "open_source_contributions")
        for item in _selected_items(profile, selection, section)
    )
    roles = set(selection.get("role_families", []))
    relevant_categories = {
        "languages",
        "frameworks",
        "tools",
        "engineering",
        "cloud_and_integrations",
        "web_platform",
        "data",
        "ml",
        "android",
    }
    if roles == {"data_operations"} or "data_operations" in roles:
        relevant_categories.update({"data", "ml"})
    if "android" in roles:
        relevant_categories.add("android")

    result: dict[str, list[str]] = {}
    for category, values in skills.items():
        if category in PROVISIONAL_SKILL_CATEGORIES or not isinstance(values, list):
            continue
        kept = [
            value
            for value in values
            if isinstance(value, str) and _skill_value_matches(value, job_text, selected_text)
        ]
        if not kept and category in relevant_categories and selection.get("technical_demand"):
            # If the JD has no exact alias for a category, keep only a small
            # baseline rather than dumping the entire skills inventory.
            kept = [value for value in values if isinstance(value, str)][:6]
        if kept:
            result[category] = kept[:18]

    # Keep a small truthful baseline when the JD has no recognizable aliases.
    if not result:
        for category in ("languages", "frameworks", "tools"):
            values = skills.get(category)
            if isinstance(values, list) and values:
                result[category] = [value for value in values if isinstance(value, str)][:10]
    return result


def _compact_education(profile: dict[str, Any], job_text: str) -> dict[str, Any]:
    education = profile.get("education")
    if not isinstance(education, dict):
        return {}
    result = {
        key: education[key]
        for key in ("degree", "university", "year", "academic_year_of_completion", "cgpa", "cgpa_scale")
        if key in education
    }
    coursework = education.get("relevant_coursework")
    if isinstance(coursework, list):
        selected_coursework = []
        normalized_job = _normalize(job_text)
        detected_skills = _detected_skills(job_text)
        for course in coursework:
            text = " ".join(_flatten_strings(course)) if isinstance(course, dict) else str(course)
            normalized = _normalize(text)
            skill_match = any(
                any(_alias_in_text(alias, normalized) for alias in SKILL_RULES.get(skill, ()))
                for skill in detected_skills
            )
            token_match = any(
                token in normalized for token in normalized_job.split() if len(token) > 3
            )
            if skill_match or token_match:
                selected_coursework.append(course)
        result["relevant_coursework"] = selected_coursework[:8]
    return result


def _compact_languages(profile: dict[str, Any], job_text: str) -> dict[str, Any]:
    languages = profile.get("languages")
    if not isinstance(languages, dict):
        return {}
    normalized_job = _normalize(job_text)
    result: dict[str, Any] = {}
    for name, value in languages.items():
        normalized_name = _normalize(str(name))
        # English is broadly relevant to the target roles. Other languages are
        # included only when the JD explicitly asks for them, preventing old
        # learning-only language notes from appearing on technical resumes.
        if normalized_name == "english" or _alias_in_text(normalized_name, normalized_job):
            result[str(name)] = value
    return result


def build_curated_profile(profile: dict[str, Any], job_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the compact profile that is safe to send to a drafting model."""
    selection = select_resume_evidence(profile, job_text)
    personal_value = profile.get("personal_info")
    positioning_value = profile.get("positioning")
    claim_policy_value = profile.get("claim_policy")
    personal: dict[str, Any] = personal_value if isinstance(personal_value, dict) else {}
    positioning: dict[str, Any] = positioning_value if isinstance(positioning_value, dict) else {}
    claim_policy: dict[str, Any] = claim_policy_value if isinstance(claim_policy_value, dict) else {}

    curated: dict[str, Any] = {
        "schema_version": profile.get("schema_version", "resume-profile"),
        "profile_owner": profile.get("profile_owner") or personal.get("name"),
        "personal_info": {
            key: personal[key]
            for key in ("name", "location", "email", "linkedin", "github")
            if key in personal and personal[key] is not None
        },
        "positioning": {
            key: positioning[key]
            for key in ("core_identity", "current_level", "public_current_focus", "target_roles", "strengths")
            if key in positioning
        },
        "claim_policy": {
            "resume_rules": claim_policy.get("resume_rules", []),
            "evidence_levels": claim_policy.get("evidence_levels", {}),
        },
        "education": _compact_education(profile, job_text),
        "skills": _selected_skills(profile, selection, job_text),
        "projects": _selected_items(profile, selection, "projects"),
        "experience": _selected_items(profile, selection, "experience"),
        "learning_projects": _selected_items(profile, selection, "learning_projects"),
        "open_source_contributions": _selected_items(profile, selection, "open_source_contributions"),
        "certifications": _selected_items(profile, selection, "certifications"),
        "achievements": _selected_items(profile, selection, "achievements"),
        "languages": _compact_languages(profile, job_text),
        "selection_context": {
            "selection_version": selection["schema_version"],
            "instruction": "Use only the selected evidence in this compact profile. Do not reintroduce suppressed items from memory or from the job description.",
            "role_families": selection["role_families"],
            "detected_skills": selection["detected_skills"],
            "technical_demand": selection["technical_demand"],
            "content_demand": selection["content_demand"],
            "section_order": selection["section_order"],
            "page_budget": selection["page_budget"],
            "selected_item_ids": {
                section: [record["key"] for record in records]
                for section, records in selection["selected"].items()
                if records
            },
            "suppressed_item_count": selection["summary"]["suppressed_items"],
            "guardrails": selection["guardrails"],
        },
    }
    return curated, selection


__all__ = [
    "TECHNICAL_ROLE_FAMILIES",
    "build_curated_profile",
    "select_resume_evidence",
]
