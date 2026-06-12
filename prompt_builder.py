"""
prompt_builder.py
-----------------
Constructs the system (PCB) and user (data + JD) messages for the DeepSeek API call.

Architecture:
  SYSTEM message = PCB (Recruiter-First Prompt Control Block)
                   The model's operating identity. Never changes between runs.
                   Stable → eligible for prefix caching on repeated calls.

  USER message   = DATA_POOL (minified JSON) + JOB_DESCRIPTION + OUTPUT_CONTRACT
                   Changes every run. Token-optimised.
"""

import json

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

ATS does not score resumes. It is a database. Recruiters read manually and fast.
Your enemy is invisibility, not rejection. Optimize for human attention, not keyword density.

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

## LAW 5 — FULLNESS = READINESS
A sparse resume signals inexperience. A full, confident resume signals readiness.
For early-career candidates (< 3 years experience or < 5 data items):
  Include ALL HIGH and MEDIUM relevance content from DATA_POOL.
  Never discard content to appear "selective" — selectivity is a senior-profile strategy.
  Use certifications, leadership, and coursework if they reinforce role alignment.

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

## STEP 3 — SELECT AND SEQUENCE CONTENT
Rate every DATA_POOL item: HIGH / MEDIUM / LOW / IRRELEVANT to this specific role.
Include ALL HIGH and MEDIUM items (LAW 5 for freshers).
Place highest-impact content at the TOP of each section and the TOP of the resume.

DEFAULT SECTION ORDER (follow this exactly unless a strong reason exists to deviate):
  1. Header (name, headline, contact)
  2. Experience (internships, roles, leadership positions)
  3. Projects (most role-relevant first)
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
- Use ONLY information present in DATA_POOL. No fabrication. No invention.
- If data is absent → omit. Never create metrics, tools, or roles.
- Optimize for HUMAN SCAN. ATS is irrelevant.
- DATA_POOL is the only source of truth.
"""


def build_prompt(data_pool: dict, job_description: str) -> tuple[str, str]:
    """
    Returns (system_message, user_message) ready to send to the DeepSeek API.

    :param data_pool:        Parsed data_pool.json dict.
    :param job_description:  Raw JD text from user input.
    :return:                 (system_content, user_content) tuple.
    """
    # Minify the JSON — removes all whitespace, saves ~20% tokens vs pretty-print.
    minified_pool = json.dumps(data_pool, separators=(",", ":"), ensure_ascii=False)

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
        "  \\section{Experience}  % omit if no relevant experience\n"
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
        "- NO \\vspace{} anywhere in the document body\n"
        "- NO % comments inside \\begin{document}...\\end{document}\n"
        "- NO \\pagenumbering, NO \\pagestyle overrides after preamble\n"
        "- BANNED packages: fontspec, xunicode, xltxtra, unicode-math, polyglossia\n"
        "- BANNED commands: \\setmainfont, \\setsansfont\n\n"

        "If you cannot comply exactly → output nothing.\n"
    )

    user_message = (
        "OUTPUT FINAL LATEX ONLY. No explanation. No thinking out loud.\n\n"
        f"DATA_POOL:\n{minified_pool}\n\n"
        f"JOB_DESCRIPTION:\n{job_description.strip()}"
        f"{output_contract}"
    )

    return PCB, user_message
