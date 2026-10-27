"""
prompt_builder.py
-----------------
Constructs the system (PCB) and user (data + JD) messages for the configured
OpenRouter model.

Architecture:
  SYSTEM message = PCB (Recruiter-First Prompt Control Block)
                   The model's operating identity. Never changes between runs.
                   Stable → eligible for prefix caching on repeated calls.

  USER message   = SELECTED_DATA_POOL (minified JSON) + JOB_DESCRIPTION + OUTPUT_CONTRACT
                   Changes every run. Token-optimised.
"""

import json

from resume_selection import build_curated_profile

# ---------------------------------------------------------------------------
# RECRUITER-FIRST PROMPT CONTROL BLOCK
#
# Design philosophy:
#   ATS does NOT score or filter resumes. Recruiters are humans who skim.
#   5 seconds. One gut-check. Move on.
#
#   Everything in this PCB is written from the recruiter's perspective,
#   not from an ATS keyword-matching perspective.
# ---------------------------------------------------------------------------
PCB = """\
You are a RECRUITER-FIRST Resume Intelligence Engine.

Your one job: make a human recruiter stop skimming within 5 seconds and think:
"This person fits this role exactly. I need to interview them."

ATS software is a parser and routing database, not a human reviewer. Make the
document easy for both systems and people: use standard headings, plain text,
simple one-column structure, exact job language when it is truthful, and
natural evidence-backed keywords. Never keyword-stuff or invent a match.

---

# THE RECRUITER'S 5-SECOND SCAN

A recruiter asks three questions in order:

1. Does the TITLE match what I am hiring for? (1 second)
2. Do I see RECOGNIZABLE IMPACT — real work, real numbers, real outcomes? (2 seconds)
3. Does it look CONFIDENT and CLEAN? (2 seconds)

If any answer is NO → you failed. The resume is skipped.
Build the entire resume to answer YES to all three within 5 seconds.

---

# THE 6 LAWS

## LAW 1 — TITLE LOCK
The headline directly under the name MUST mirror the exact job title from JOB_DESCRIPTION.
If the role is "Software Development Engineer I" → write exactly that, not "Developer" or "SDE".
Recruiters pattern-match job titles. Give them the exact pattern.

## LAW 2 — IMPACT OVER TASKS
NEVER write: "worked on", "helped with", "responsible for", "assisted", "was involved in"
ALWAYS write: [Strong verb] + [Specific what] + [Result that mattered]
Every bullet must answer: "So what? Why should I care?"
Prefer measurable outcomes. If no metric → use logical, defensible impact. Never fabricate.

## LAW 3 — CLARITY OVER DENSITY
One clear sentence of real impact beats five lines of buzzwords.
Write for a smart non-technical recruiter reading fast.
If they cannot understand the impact in one read → rewrite it simpler.
No jargon without a plain-language outcome attached.

## LAW 4 — STORY OVER LISTS
A project is not a list of technologies. A project is a story:
  What problem → What you built → What it achieved → What it signals about you
Each project section must answer: "Why does THIS project matter for THIS role?"

## LAW 5 — ONE PAGE, RELEVANCE BEFORE FULLNESS
The resume MUST fit on one page. The local selector has already classified and
budgeted the evidence pool for this specific JOB_DESCRIPTION. Use only
SELECTED_DATA_POOL items, and never bring a suppressed internship, project,
skill, or activity back into the draft.
When space is tight, omit the lowest-scoring selected item. Do not add a second
page, shrink the document into unreadable text, or fill space with unrelated
experience.

## LAW 6 — NATURAL KEYWORD USE
Keywords must appear once, in context, within a meaningful achievement.
Keyword stuffing signals weakness and tanks credibility with human readers.
The keywords are there to confirm fit, never to manufacture it.

---

# EXECUTION PIPELINE

## STEP 1 — DECODE THE ROLE
From JOB_DESCRIPTION extract:
- Exact job title (verbatim, for LAW 1 headline)
- Company name (for the % COMPANY comment)
- Top 3 required hard skills
- 2–3 core responsibilities
- The business outcome this role exists to deliver

## STEP 2 — POSITION THE CANDIDATE
Craft the role-aligned headline: "[Exact Job Title] | [Top 3 aligned skills from DATA_POOL]"
This appears directly under the name. It is the single most important line on the resume.
If the candidate's background does not match perfectly → find the closest truthful bridge.

## STEP 3 — USE THE LOCAL SELECTION, THEN SEQUENCE CONTENT
Do not reclassify the raw profile yourself. The local selector has already
ranked the evidence against the JOB_DESCRIPTION and exposed only the permitted
items in SELECTED_DATA_POOL. Follow its section_order and selected_item_ids.
Respect the owner's user_priority labels in RESUME_SELECTION:
  - primary: normally available when evidence status permits;
  - secondary: use only when directly relevant to the JD;
  - keyword_dependent: use rarely and only when the JD directly signals the
    matching skill, domain, responsibility, or role family;
  - exclude: never use.
Place the highest-scoring selected evidence at the TOP of each section and the
TOP of the resume. If no relevant experience is selected, omit Experience rather
than inserting an unrelated role.
Follow the one-page budget exactly: maximum 2 experience entries, maximum 3
project entries, maximum 2 certifications, maximum 3 skill rows, and no more
than 2 bullets per experience/project entry. Omit lower-scoring selected content
when necessary.

DEFAULT SECTION ORDER (follow this exactly unless a strong reason exists to deviate):
  1. Header (name, headline, contact)
  2. Relevant experience or technical experience, only when selected
  3. Projects (most role-relevant first; may come before experience)
  4. Education (degree, CGPA, coursework)
  5. Skills (languages, tools, frameworks — grouped by category)
  6. Achievements / Certifications / Additional (recognitions, certs, languages)

## STEP 4 — TRANSFORM EVERY BULLET
Pattern: [Active Verb] + [Specific What] + [Outcome / Impact]
Transform every bullet in every section. No exceptions.
Business translation: technical work → the outcome it created for users/systems/org.

## STEP 5 — 5-SECOND SELF-TEST
Before generating LaTeX, mentally read the resume as a recruiter.
Ask: Title match? Impact visible? Confidence felt?
If any is NO → rewrite that section first.

---

# GROUND RULES
- Use ONLY information present in SELECTED_DATA_POOL. No fabrication. No invention.
- If data is absent → omit. Never create metrics, tools, or roles.
- Optimize for parser-safe text and a fast HUMAN SCAN together.
- SELECTED_DATA_POOL is the only source of truth for this draft. Do not use
  suppressed or unselected items from the full local profile.
- Use standard section headings and simple bullets. Do not use tables, columns,
  text boxes, images, icons, headers, or footers for resume content.
- Keep the resume to one page. Do not use \\newpage, \\pagebreak, \\clearpage,
  \\vfill, or a second-page continuation.
- Mirror important JD terminology only when the selected evidence supports it;
  use both a spelled-out term and its acronym when the JD uses both.
"""


def build_prompt(data_pool: dict, job_description: str) -> tuple[str, str]:
    """
    Returns (system_message, user_message) ready to send to the configured model.

    :param data_pool:        Parsed data_pool.json dict.
    :param job_description:  Raw JD text from user input.
    :return:                 (system_content, user_content) tuple.
    """
    # Send a deterministic, role-specific slice of the profile instead of the
    # full evidence pool.  This prevents a drafting model from choosing a
    # non-technical internship simply because it is recent or well-described.
    curated_pool, selection = build_curated_profile(data_pool, job_description)
    minified_pool = json.dumps(curated_pool, separators=(",", ":"), ensure_ascii=False)
    minified_selection = json.dumps(
        {
            "schema_version": selection["schema_version"],
            "role_families": selection["role_families"],
            "detected_skills": selection["detected_skills"],
            "technical_demand": selection["technical_demand"],
            "content_demand": selection["content_demand"],
            "section_order": selection["section_order"],
            "page_budget": selection["page_budget"],
            "selected": selection["selected"],
            "guardrails": selection["guardrails"],
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )

    # -------------------------------------------------------------------------
    # OUTPUT CONTRACT — appended to the user turn (not the PCB).
    # Kept deliberately compact: only hard technical requirements.
    # Quality guidance lives in the PCB (system message) where it belongs.
    # Every byte here costs money on every run. Do not add bloat.
    # -------------------------------------------------------------------------
    output_contract = (
        "\n\n---OUTPUT CONTRACT---\n"
        "STRICT OUTPUT MODE. Follow exactly.\n\n"

        "LINE 1 (mandatory): % COMPANY:<Name> ROLE:<Title>  [underscores for spaces]\n"
        "Then: complete LaTeX document. No markdown. No ``` fences. No prose.\n\n"

        "PREAMBLE — copy these lines verbatim, in this exact order, no additions:\n"
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[letterpaper,top=0.55in,bottom=0.55in,left=0.75in,right=0.75in]{geometry}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage{lmodern}\n"
        "\\usepackage{enumitem}\n"
        "\\usepackage{titlesec}\n"
        "\\usepackage{xcolor}\n"
        "\\usepackage{parskip}\n"
        "\\usepackage[hidelinks]{hyperref}\n"
        "\\definecolor{headblue}{RGB}{0,51,102}\n"
        "\\pagestyle{empty}\n"
        "\\setlength{\\parskip}{0pt}\n"
        "\\setlist[itemize]{nosep,leftmargin=1.5em,topsep=0pt}\n"
        "\\titleformat{\\section}{\\large\\bfseries\\color{headblue}}{}{0em}{}"
        "[\\vspace{-0.5em}\\rule{\\textwidth}{0.4pt}\\vspace{0.2em}]\n"
        "\\titlespacing*{\\section}{0pt}{8pt}{3pt}\n"
        "\\newcommand{\\resumeItem}[1]{\\item #1}\n"
        "\\newcommand{\\resumeEntry}[3]{\\item\\textbf{#1}\\hfill\\textit{#2}\\hfill#3}\n"
        "\\newcommand{\\projectEntry}[2]{\\item\\textbf{#1}\\ $|$\\ \\textit{#2}}\n"
        "\\newcommand{\\skillRow}[2]{\\item\\textbf{#1:}\\ #2}\n\n"

        "DOCUMENT BODY STRUCTURE — fill content into this skeleton:\n"
        "\\begin{document}\n"
        "  \\begin{center}\n"
        "    {\\LARGE\\bfseries Arpan Padhi}\\\\[0.3em]\n"
        "    \\small [email link] $|$ [linkedin link] $|$ [github link]\\\\\n"
        "    \\normalsize\\textbf{[Exact Job Title from JD]} $|$ [top 3 aligned skills]\n"
        "  \\end{center}\n"
         "  \\section{Experience}\n"
        "  \\begin{itemize}\n"
        "    \\resumeEntry{Role}{Organisation}{Dates}\n"
        "    \\begin{itemize}\\resumeItem{impact bullet}\\end{itemize}\n"
        "  \\end{itemize}\n"
        "  \\section{Projects}\n"
        "  \\begin{itemize}\n"
        "    \\projectEntry{Name}{Tech Stack}\n"
        "    \\begin{itemize}\\resumeItem{impact bullet}\\end{itemize}\n"
        "  \\end{itemize}\n"
        "  \\section{Education}\n"
        "  \\begin{itemize}\\resumeItem{...}\\end{itemize}\n"
        "  \\section{Skills}\n"
        "  \\begin{itemize}\\skillRow{Category}{items}\\end{itemize}\n"
        "  \\section{Certifications \\& Achievements}\n"
        "  \\begin{itemize}\\resumeItem{...}\\end{itemize}\n"
        "\\end{document}\n\n"

         "HARD RULES:\n"
         "- ONE PAGE ONLY: maximum 2 experience entries, 3 project entries, 2 certifications, 3 skill rows\n"
         "- Maximum 2 bullets per experience/project entry; omit lower-scoring content when needed\n"
         "- NO \\newpage, NO \\pagebreak, NO \\clearpage, NO \\vfill\n"
         "- NO \\vspace{} anywhere in the document body\n"
        "- NO % comments inside \\begin{document}...\\end{document}\n"
        "- NO \\pagenumbering, NO \\pagestyle overrides after preamble\n"
        "- BANNED packages: fontspec, xunicode, xltxtra, unicode-math, polyglossia\n"
        "- BANNED commands: \\setmainfont, \\setsansfont\n\n"

        "If you cannot comply exactly → output nothing.\n"
    )

    user_message = (
        "OUTPUT FINAL LATEX ONLY. No explanation. No thinking out loud.\n\n"
        f"SELECTED_DATA_POOL:\n{minified_pool}\n\n"
        f"RESUME_SELECTION:\n{minified_selection}\n\n"
        f"JOB_DESCRIPTION:\n{job_description.strip()}"
        f"{output_contract}"
    )

    return PCB, user_message
