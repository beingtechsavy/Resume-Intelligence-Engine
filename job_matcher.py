"""Deterministic, explainable first-pass job-to-evidence matcher.

This module deliberately does not produce an ATS score.  It identifies explicit
requirements, maps them to the verified evidence pool, and flags eligibility
questions for human review before any AI drafting call is made.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_POOL = ROOT / "data_pool.json"


SKILL_RULES: dict[str, tuple[str, ...]] = {
    "python": ("python",),
    "django": ("django",),
    "postgresql_sql": ("postgresql", "postgres", "sql"),
    "aws": ("aws", "ec2", "s3"),
    "rest_http_json": ("rest", "http", "json", "api"),
    "git": ("git", "github"),
    "linux": ("linux", "ubuntu", "wsl"),
    "ai_llm": ("artificial intelligence", "ai", "llm", "machine learning", "openrouter"),
    "prompt_testing": ("prompt", "prompts", "instruction", "transcript", "iterative testing"),
    "debugging": ("debug", "troubleshoot", "incident", "root cause", "error handling"),
    "customer_communication": ("customer", "customers", "client", "clients", "user", "users", "communication", "support"),
    "project_coordination": ("project management", "project coordinator", "scope", "timeline", "budget", "risk"),
    "documentation": ("documentation", "technical writing", "research", "reporting"),
    "typescript_react": ("typescript", "react", "next.js", "nextjs"),
    "kotlin_android": ("kotlin", "android", "jetpack compose", "compose"),
}


ROLE_RULES: dict[str, tuple[str, ...]] = {
    "ai_support": ("ai support", "ai applications", "prompt", "transcript", "agent", "llm"),
    "technical_support": ("technical support", "support engineer", "troubleshoot", "incident", "debugging"),
    "project_delivery": ("project manager", "project coordinator", "delivery", "scope", "timeline", "budget"),
    "implementation_customer": ("implementation", "customer success", "technical account", "onboarding", "solutions"),
    "software_development": ("software engineer", "developer", "full-stack", "backend", "frontend", "android"),
    "data_operations": ("data engineer", "data analyst", "operations analyst", "sql"),
    "research_iot": ("research intern", "iot", "smart hub", "device communication", "computer vision"),
    "technical_communication": ("technical writing", "technical documentation", "developer advocate", "student ambassador", "technical content"),
    "content_community": ("podcast", "podcaster", "content creator", "social media", "marketing", "community", "communications", "media production"),
    "sales_business": ("sales", "sales representative", "business development", "account executive", "partnerships"),
}


def _normalize(text: str) -> str:
    """Normalize text for conservative phrase matching."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _alias_in_text(alias: str, normalized_text: str) -> bool:
    """Match a phrase or whole word without treating ``ai`` as part of ``training``."""
    normalized_alias = _normalize(alias)
    if " " in normalized_alias or "/" in normalized_alias or "-" in normalized_alias:
        return normalized_alias in normalized_text
    return re.search(rf"\b{re.escape(normalized_alias)}\b", normalized_text) is not None


def _flatten_strings(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    """Return string values with a stable JSON-like evidence path."""
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            found.extend(_flatten_strings(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_flatten_strings(child, f"{prefix}[{index}]"))
    elif isinstance(value, str):
        found.append((prefix, value))
    return found


def _evidence_buckets(profile: dict[str, Any]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Separate stronger implementation evidence from provisional exposure."""
    strong: list[tuple[str, str]] = []
    provisional: list[tuple[str, str]] = []

    skills = profile.get("skills", {})
    for key, values in skills.items():
        pairs = [(f"skills.{key}[{index}]", value) for index, value in enumerate(values)] if isinstance(values, list) else []
        if key in {"historical_self_report_only", "learning_exposure", "not_established_from_copied_material"}:
            provisional.extend(pairs)
        else:
            strong.extend(pairs)

    for project in profile.get("projects", []):
        project_id = project.get("id", "project")
        for field in ("name", "type", "description", "tech_stack", "features", "data_work", "impact", "resume_ready_bullets", "tags"):
            value = project.get(field)
            if isinstance(value, list):
                strong.extend((f"projects.{project_id}.{field}[{index}]", item) for index, item in enumerate(value) if isinstance(item, str))
            elif isinstance(value, str):
                strong.append((f"projects.{project_id}.{field}", value))

    for experience in profile.get("experience", []):
        experience_id = experience.get("id", "experience")
        for field in ("role", "organization", "responsibilities_reported", "resume_use"):
            value = experience.get(field)
            if isinstance(value, list):
                strong.extend((f"experience.{experience_id}.{field}[{index}]", item) for index, item in enumerate(value) if isinstance(item, str))
            elif isinstance(value, str):
                strong.append((f"experience.{experience_id}.{field}", value))

    strong.extend(_flatten_strings(profile.get("education", {}), "education"))
    return strong, provisional


def _matching_evidence(
    canonical_skill: str,
    strong: list[tuple[str, str]],
    provisional: list[tuple[str, str]],
) -> dict[str, Any]:
    aliases = SKILL_RULES[canonical_skill]

    def hits(entries: list[tuple[str, str]]) -> list[dict[str, str]]:
        results = []
        for path, value in entries:
            normalized = _normalize(value)
            if any(_alias_in_text(alias, normalized) for alias in aliases):
                results.append({"path": path, "excerpt": value[:180]})
        return results[:6]

    strong_hits = hits(strong)
    provisional_hits = hits(provisional)
    if strong_hits:
        status = "matched"
    elif provisional_hits:
        status = "partial"
    else:
        status = "missing"
    return {
        "requirement": canonical_skill,
        "status": status,
        "evidence": strong_hits,
        "provisional_evidence": provisional_hits,
    }


def _detected_skills(job_text: str) -> list[str]:
    normalized = _normalize(job_text)
    detected = []
    for skill, aliases in SKILL_RULES.items():
        if any(_alias_in_text(alias, normalized) for alias in aliases):
            detected.append(skill)
    return detected


def _detected_role_families(job_text: str) -> list[str]:
    normalized = _normalize(job_text)
    ranked = []
    for role, aliases in ROLE_RULES.items():
        count = sum(1 for alias in aliases if _alias_in_text(alias, normalized))
        if count:
            ranked.append((count, role))
    return [role for _, role in sorted(ranked, key=lambda item: (-item[0], item[1]))]


def _experience_gate(job_text: str) -> dict[str, Any]:
    normalized = _normalize(job_text)
    if re.search(r"no (prior )?experience|without (commercial )?experience|0 years?", normalized):
        return {"status": "accessible", "reason": "The listing explicitly presents a no-experience path."}
    match = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", normalized)
    if match:
        return {"status": "review", "years_stated": float(match.group(1)), "reason": "The listing states an experience requirement."}
    if "internship" in normalized or "projects" in normalized or "course" in normalized:
        return {"status": "accessible_with_evidence", "reason": "The listing mentions projects, courses or internships as an evidence path."}
    return {"status": "unknown", "reason": "No explicit experience gate was detected."}


def _eligibility_review(job_text: str, profile: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize(job_text)
    preferences = profile.get("candidate_preferences", {})
    current_location = preferences.get("current_location", profile.get("personal_info", {}).get("location", ""))
    if "remote" in normalized or "worldwide" in normalized or "work from anywhere" in normalized:
        return {
            "status": "human_review",
            "mode": "remote",
            "reason": "Remote scope is visible, but payroll, contract and India hiring eligibility still need employer confirmation.",
            "candidate_location": current_location,
        }
    if any(term in normalized for term in ("relocation", "visa", "sponsorship", "europe", "eu member")):
        return {
            "status": "human_review",
            "mode": "relocation_or_sponsorship",
            "reason": "Relocation or sponsorship signals require destination-country and employer checks.",
            "candidate_location": current_location,
        }
    return {
        "status": "unknown",
        "mode": "unspecified",
        "reason": "The job's location or hiring-country model was not explicit enough for a decision.",
        "candidate_location": current_location,
    }


def _compensation(job_text: str) -> list[dict[str, str]]:
    patterns = [
        r"(?:[$€£]\s?\d[\d,]*(?:\.\d+)?(?:\s*[-–]\s*[$€£]?\s?\d[\d,]*(?:\.\d+)?)?\s*(?:/\s*)?(?:month|year|hour|day)?)",
        r"(?:(?:USD|EUR|GBP)\s+\d[\d,]*(?:\.\d+)?(?:\s*[-–]\s*(?:USD|EUR|GBP)?\s?\d[\d,]*(?:\.\d+)?)?\s*(?:/\s*)?(?:month|year|hour|day)?)",
        r"(?:\d[\d,]*(?:\.\d+)?\s*(?:USD|EUR|GBP)(?:\s*[-–]\s*\d[\d,]*(?:\.\d+)?\s*(?:USD|EUR|GBP))?\s*(?:/\s*)?(?:month|year|hour|day)?)",
    ]
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, job_text, flags=re.IGNORECASE))
    return [{"text": match.strip()} for match in dict.fromkeys(matches)]


def analyze_job(job_text: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Create a local explainable report for one raw job description."""
    if not job_text.strip():
        raise ValueError("job_text must not be empty")

    strong, provisional = _evidence_buckets(profile)
    detected_skills = _detected_skills(job_text)
    requirement_matches = [
        _matching_evidence(skill, strong, provisional) for skill in detected_skills
    ]
    matched = sum(item["status"] == "matched" for item in requirement_matches)
    partial = sum(item["status"] == "partial" for item in requirement_matches)
    missing = sum(item["status"] == "missing" for item in requirement_matches)

    # Selection is deliberately kept separate from requirement matching: the
    # report can explain both what the JD asks for and what the resume writer is
    # allowed to use.  Import lazily to keep this module usable on its own and
    # avoid a module cycle for the prompt builders.
    from resume_selection import select_resume_evidence

    resume_selection = select_resume_evidence(profile, job_text)

    return {
        "schema_version": "job-match-report-1",
        "role_families": _detected_role_families(job_text),
        "requirements": requirement_matches,
        "resume_selection": resume_selection,
        "experience_gate": _experience_gate(job_text),
        "eligibility": _eligibility_review(job_text, profile),
        "compensation_mentions": _compensation(job_text),
        "summary": {
            "detected_requirements": len(requirement_matches),
            "matched": matched,
            "partial": partial,
            "missing": missing,
            "decision": "human_review_required",
            "reason": "This first pass is evidence and eligibility triage, not an ATS score or an automatic application decision.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jd-file", type=Path, required=True, help="Text file containing one job description")
    parser.add_argument("--profile", type=Path, default=DEFAULT_POOL, help="Candidate evidence JSON")
    parser.add_argument("--output", type=Path, help="Optional JSON report destination")
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    report = analyze_job(args.jd_file.read_text(encoding="utf-8"), profile)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"[OK] Match report saved -> {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
