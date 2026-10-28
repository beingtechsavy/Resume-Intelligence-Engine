"""Local ATS-oriented checks for generated LaTeX resumes.

This is intentionally a static, explainable validator rather than a vendor
score.  ATS implementations differ, and no local check can guarantee a hiring
outcome.  The validator checks the document properties that are controllable
in this project: text extraction, standard structure, one-column layout,
parser-hostile elements, and truthful overlap with detected JD requirements.

Usage:
    python ats_validator.py --resume output.tex --jd-file job.txt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from rie.latex import FORBIDDEN_PACKAGES
from rie.matching import SKILL_RULES, _alias_in_text, _detected_skills, _normalize


STANDARD_HEADINGS = {
    "experience",
    "relevant experience",
    "technical experience",
    "professional experience",
    "projects",
    "education",
    "skills",
    "certifications",
    "certifications & achievements",
    "achievements",
    "additional experience",
}

_HOSTILE_LAYOUT_PATTERNS = {
    "multi_column_layout": r"\\(?:twocolumn|onecolumn|multicol|begin\{multicols\})",
    "table_layout": r"\\(?:begin\{tabular|begin\{tabularx|begin\{table|begin\{longtable)",
    "text_box_layout": r"\\(?:begin\{(?:minipage|parbox|textblock)|makebox|fbox)",
    "image_or_icon": r"\\(?:includegraphics|includegraphics|qrcode|tikz|begin\{tikzpicture\})",
    "header_or_footer_content": r"\\(?:lhead|chead|rhead|lfoot|cfoot|rfoot|fancyhf|fancyhead|fancyfoot)",
}


def _check(check_id: str, label: str, status: str, details: str) -> dict[str, str]:
    return {"id": check_id, "label": label, "status": status, "details": details}


def _plain_text(latex: str) -> str:
    """Approximate selectable resume text without trying to compile TeX."""
    text = re.sub(r"(?m)^\s*%.*$", "", latex)
    text = re.sub(r"\\href\{[^{}]*\}\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:textbf|textit|emph|underline|texttt)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:section|subsection|subsubsection)\*?\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+(?:\[[^]]*\])?\s*", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = text.replace("\\", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _section_headings(latex: str) -> list[str]:
    return [
        _normalize(match.group(1).replace("\\&", "&"))
        for match in re.finditer(r"\\section\*?\{([^{}]+)\}", latex)
    ]


def validate_one_page_budget(latex: str) -> dict[str, Any]:
    """Check the deterministic content budget used by the one-page prompt."""
    body_start = latex.find("\\begin{document}")
    body_end = latex.find("\\end{document}")
    body = latex[body_start:body_end] if body_start >= 0 and body_end > body_start else latex

    counts = {
        "experience_entries": len(re.findall(r"\\resumeEntry\b", body)),
        "project_entries": len(re.findall(r"\\projectEntry\b", body)),
        "resume_bullets": len(re.findall(r"\\resumeItem\b", body)),
        "skill_rows": len(re.findall(r"\\skillRow\b", body)),
        "sections": len(re.findall(r"\\section\*?\{", body)),
        "page_break_commands": len(re.findall(r"\\(?:newpage|pagebreak|clearpage|vfill)\b", body)),
    }
    limits = {
        "experience_entries": 2,
        "project_entries": 3,
        "resume_bullets": 13,
        "skill_rows": 3,
        "sections": 5,
        "page_break_commands": 0,
    }
    violations = [
        f"{key}={counts[key]} exceeds {limits[key]}"
        for key in limits
        if counts[key] > limits[key]
    ]
    return {
        "status": "fail" if violations else "pass",
        "counts": counts,
        "limits": limits,
        "details": "One-page content budget passed." if not violations else "; ".join(violations),
    }


def _keyword_check(latex: str, job_description: str) -> tuple[dict[str, Any], list[str]]:
    if not job_description.strip():
        return (
            _check("jd_keyword_alignment", "Detected JD skills appear in the resume", "not_evaluable", "No job description was supplied."),
            [],
        )
    text = _normalize(_plain_text(latex))
    detected = _detected_skills(job_description)
    matched: list[str] = []
    missing: list[str] = []
    for skill in detected:
        aliases = SKILL_RULES.get(skill, ())
        if any(_alias_in_text(alias, text) for alias in aliases):
            matched.append(skill)
        else:
            missing.append(skill)
    if not detected:
        status = "not_evaluable"
        details = "The local conservative matcher detected no supported skill aliases in the JD."
    elif not missing:
        status = "pass"
        details = f"All {len(matched)} detected skill requirement(s) appear in the text layer."
    else:
        status = "review"
        details = f"Matched {len(matched)} of {len(detected)} detected skill requirement(s); review missing terms without adding unsupported claims."
    return _check("jd_keyword_alignment", "Detected JD skills appear in the resume", status, details), missing


def validate_resume_ats(latex: str, job_description: str = "") -> dict[str, Any]:
    """Run local, explainable ATS-oriented checks on one LaTeX resume."""
    if not isinstance(latex, str) or not latex.strip():
        raise ValueError("latex must not be empty")

    checks: list[dict[str, str]] = []
    has_document = "\\documentclass" in latex and "\\begin{document}" in latex and "\\end{document}" in latex
    checks.append(
        _check(
            "document_structure",
            "Complete text document structure",
            "pass" if has_document else "fail",
            "The LaTeX document has the expected documentclass and document boundaries." if has_document else "The document is missing a required LaTeX boundary.",
        )
    )

    page_budget = validate_one_page_budget(latex)
    checks.append(
        _check(
            "one_page_budget",
            "One-page content budget",
            page_budget["status"],
            page_budget["details"],
        )
    )

    for check_id, pattern in _HOSTILE_LAYOUT_PATTERNS.items():
        found = re.search(pattern, latex, flags=re.IGNORECASE) is not None
        labels = {
            "multi_column_layout": "One-column layout",
            "table_layout": "No table-based layout",
            "text_box_layout": "No text boxes or positioned blocks",
            "image_or_icon": "No images or icons in resume content",
            "header_or_footer_content": "No header/footer content",
        }
        checks.append(
            _check(
                check_id,
                labels[check_id],
                "fail" if found else "pass",
                "Parser-hostile layout command detected." if found else "No parser-hostile command detected.",
            )
        )

    forbidden_packages = FORBIDDEN_PACKAGES
    package_hits = sorted(
        package
        for package in forbidden_packages
        if re.search(rf"\\usepackage(?:\[[^]]*\])?\{{[^}}]*\b{re.escape(package)}\b[^}}]*\}}", latex)
    )
    checks.append(
        _check(
            "text_safe_packages",
            "Text-safe LaTeX packages",
            "fail" if package_hits else "pass",
            f"Forbidden package(s): {', '.join(package_hits)}." if package_hits else "No known XeLaTeX-only package was detected.",
        )
    )

    headings = _section_headings(latex)
    unknown_headings = [heading for heading in headings if heading not in STANDARD_HEADINGS]
    standard_ok = len(headings) >= 2 and not unknown_headings
    checks.append(
        _check(
            "standard_section_headings",
            "Recognizable section headings",
            "pass" if standard_ok else "review",
            f"Detected headings: {', '.join(headings) or 'none'}." + (f" Unrecognized: {', '.join(unknown_headings)}." if unknown_headings else ""),
        )
    )

    body_start = latex.find("\\begin{document}")
    body_end = latex.find("\\end{document}")
    body = latex[body_start:body_end] if body_start >= 0 and body_end > body_start else latex
    email_present = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", body, re.IGNORECASE) is not None
    checks.append(
        _check(
            "contact_in_text_body",
            "Contact details in the text body",
            "pass" if email_present else "review",
            "An email address is present in the selectable document body." if email_present else "No email address was detected in the document body.",
        )
    )

    keyword_check, missing_skills = _keyword_check(latex, job_description)
    checks.append(keyword_check)

    text = _normalize(_plain_text(latex))
    repeated_terms = []
    for skill in _detected_skills(job_description):
        aliases = [alias for alias in SKILL_RULES.get(skill, ()) if " " not in alias]
        count = max((len(re.findall(rf"\b{re.escape(alias)}\b", text)) for alias in aliases), default=0)
        if count > 4:
            repeated_terms.append(skill)
    checks.append(
        _check(
            "natural_keyword_use",
            "Natural keyword use",
            "review" if repeated_terms else "pass",
            f"Repeated skill terms may need a human edit: {', '.join(repeated_terms)}." if repeated_terms else "No obvious repeated-keyword pattern was detected.",
        )
    )

    failed = sum(check["status"] == "fail" for check in checks)
    review = sum(check["status"] == "review" for check in checks)
    passed = sum(check["status"] == "pass" for check in checks)
    status = "fail" if failed else "review" if review else "pass"
    return {
        "schema_version": "ats-validation-1",
        "status": status,
        "summary": {"passed": passed, "review": review, "failed": failed, "missing_detected_skills": missing_skills},
        "checks": checks,
        "note": "This is a local parser-safety and relevance check, not a universal ATS score or hiring prediction.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True, help="Generated LaTeX resume")
    parser.add_argument("--jd-file", type=Path, help="Optional text file containing the job description")
    parser.add_argument("--output", type=Path, help="Optional JSON report destination")
    args = parser.parse_args()

    latex = args.resume.read_text(encoding="utf-8")
    job_description = args.jd_file.read_text(encoding="utf-8") if args.jd_file else ""
    report = validate_resume_ats(latex, job_description)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"[OK] ATS validation report saved -> {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
