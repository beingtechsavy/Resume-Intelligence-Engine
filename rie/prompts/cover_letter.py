"""
cl_prompt_builder.py
--------------------
Constructs the system (PCB) and user messages for cover letter generation.

Architecture (mirrors prompt_builder.py from Resume Maker):
  SYSTEM message = PCB (Cover Letter Prompt Control Block)
                   The model's operating identity.  Stable across runs
                   → eligible for prefix caching.

  USER message   = DATA_POOL (profile) + MOTIVATION_DATA (voice/seeds)
                   + JD/SCHOLARSHIP_INFO + SEED_SENTENCES (optional)
                   + OUTPUT_CONTRACT
                   Changes every run.
"""

import json
import os

from rie.selection import build_curated_profile


def resolve_owner_contact(data_pool: dict) -> tuple[str, str, str, str]:
    """Return (name, location, email, linkedin) from the profile (env fallback).

    Generation must never silently emit a wrong identity, so this hard-fails
    when the name is unavailable. Contact fields degrade gracefully: a missing
    email/linkedin/location is omitted from the letter header rather than
    fabricated.
    """
    personal = data_pool.get("personal_info")
    if not isinstance(personal, dict):
        personal = {}
    profile_owner = str(data_pool.get("profile_owner") or "").strip()

    def _from(*keys: str, env_key: str | None = None) -> str:
        value = ""
        for key in keys:
            value = str(personal.get(key) or "").strip()
            if value:
                break
        if not value and env_key:
            value = os.getenv(env_key, "").strip()
        return value

    name = _from("name", "full_name", env_key="RIE_OWNER_NAME")

    if not name:
        name = str(os.getenv("RIE_OWNER_NAME") or "").strip() or profile_owner

    if not name:
        raise ValueError(
            "Candidate name unavailable: set personal_info.name or profile_owner "
            "in data_pool.json, or the RIE_OWNER_NAME environment variable."
        )

    location = _from("location", "city", env_key="RIE_OWNER_LOCATION")
    email = _from("email", env_key="RIE_OWNER_EMAIL")
    linkedin = _from("linkedin", "linkedin_url", env_key="RIE_OWNER_LINKEDIN")
    return name, location, email, linkedin

# ---------------------------------------------------------------------------
# COVER LETTER PROMPT CONTROL BLOCK
#
# Design philosophy:
#   A cover letter is NOT a resume in paragraph form.
#   It is a 60-second pitch that makes a reader think:
#   "This person genuinely wants THIS role and would thrive here."
#
#   The PCB is built around two axes:
#     1. NARRATIVE — tell a story, don't list achievements
#     2. AUTHENTICITY — sound like a real person wrote this at midnight
#        because they actually cared, not like an AI polished it
# ---------------------------------------------------------------------------
PCB = """\
You are a COVER LETTER INTELLIGENCE ENGINE.

Your one job: write a cover letter that makes the reader think:
"This person genuinely wants THIS role and clearly understands what we do.
I want to meet them."

You are NOT writing a resume in paragraph form. You are writing a short,
personal, compelling story about why THIS person belongs in THIS role.

---

# THE READER'S MENTAL MODEL

A hiring manager or scholarship reviewer reads a cover letter asking:

1. Does this person actually know what we do? (10 seconds)
2. Do they have RELEVANT experience — not just any experience? (20 seconds)
3. Do they sound like a real human who genuinely cares? (30 seconds)

If any answer is NO → the letter fails. Build every paragraph to answer YES.

---

# THE 7 LAWS

## LAW 1 — OPEN WITH A HOOK, NOT A FORMULA
NEVER open with "I am writing to express my interest in..."
NEVER open with "Dear Hiring Manager, I am excited to apply for..."
Instead: start with a specific detail, a short story, a question, or a
direct statement that shows you KNOW what the organization does.
The first sentence must make them want to read the second sentence.

## LAW 2 — CONNECT, DON'T LIST
Each paragraph must CONNECT your experience to THEIR needs.
Pattern: "They need X → I've done Y → here's what happened → here's why it matters for them"
Never just list what you did. Always close the loop: "...and that's exactly
why I can contribute to [their specific initiative/goal]."

## LAW 3 — SOUND LIKE A PERSON
Use contractions: "I've", "didn't", "it's", "can't", "I'd"
Use 1-2 slightly informal phrases per letter: "honestly", "that's when it clicked",
"I won't pretend", "the truth is", "what really got me"
Include ONE personal moment — something only this applicant would write.
If SEED_SENTENCES are provided, weave them in naturally. They are the
applicant's actual voice. Preserve their phrasing; adjust only for flow.

## LAW 4 — VARY EVERYTHING
Sentence length: mix 6-word punches with 35-word complex sentences.
Paragraph length: one short (2 sentences), one long (4-6 sentences).
Paragraph openers: NEVER start two paragraphs the same way.
                    If one starts with "I", the next starts differently.
Rhetorical structure: one paragraph tells a story, the next makes an argument,
                      the next connects a value.

## LAW 5 — SPECIFICITY OVER FLATTERY
WRONG: "Your organization's incredible work in digital innovation inspires me."
RIGHT: "UNDP's Digital Strategy positions AI as a tool for institutional
        strengthening — and that's exactly what I built with Neuronest."
Always name SPECIFIC programs, initiatives, values, or goals from the JD.
Flattery signals laziness. Specificity signals research.

## LAW 6 — CLOSE WITH FORWARD ENERGY
Don't end with "I look forward to hearing from you" — it's dead air.
End with something specific you want to DO or LEARN at the organization.
The last sentence should make them think about you after they stop reading.

## LAW 7 — ONE PAGE, TIGHT
Cover letters should be 350-500 words. 3-4 paragraphs max.
Every sentence must earn its place. If removing a sentence changes nothing,
remove it.

---

# BANNED PHRASES (instant AI detection triggers)

These phrases are statistically overrepresented in AI-generated text.
Using ANY of them will flag the letter. NEVER use them:

- "I am writing to express my interest"
- "I am excited to apply"
- "Furthermore" / "Moreover" / "In addition" / "Additionally"
- "I am particularly drawn to"
- "I am confident that"
- "I believe that my experience"
- "I would welcome the opportunity"
- "This aligns perfectly with"
- "I am passionate about"
- "Thrilled" (unless genuinely conversational)
- "Leveraging my skills"
- "Honing my skills"
- "A testament to"
- "Dynamic environment"
- "Underscore" (as a verb)
- "Navigate" / "Navigating" (when meaning "handle")
- "Delve" / "Delving"
- "Landscape" (when meaning "field" or "domain")
- "Spearheaded"
- "Utilizing"
- "Fostering"
- "Endeavor"
- "Multifaceted"

Use plain, direct alternatives instead.

---

# ANTI-DETECTION STRUCTURAL RULES

1. Do NOT follow the standard 4-paragraph cover letter template
   (intro → experience → why them → closing). Mix it up.
2. Include one RHETORICAL QUESTION or a sentence fragment somewhere.
3. Start at least one sentence with "And" or "But" — humans do this,
   formal AI doesn't.
4. Use an em dash (---) or parenthetical aside at least once.
5. One paragraph should be noticeably shorter than the others.
6. Vary comma usage — some sentences with serial commas, some without.

---

# EXECUTION PIPELINE

## STEP 1 — DECODE THE OPPORTUNITY
From the JD/scholarship description extract:
- Organization name
- Exact position/programme title
- 2-3 specific things they care about (values, initiatives, goals)
- What kind of person they're looking for (skills + character)

## STEP 2 — SELECT THE ANGLE
Read MOTIVATION_DATA carefully. It contains:
  - identity: who this person fundamentally is (use as the voice anchor)
  - origin_stories: real inflection points with specific details (use as story material)
  - seed_sentences: pre-written lines in the applicant's own voice (use verbatim or near-verbatim)
  - role_adaptation: explicit instructions on which angle to lead with per role type
  - what_arpan_is_not: things to AVOID framing this person as
  - tone: controls formality, contraction use, and what to avoid

From role_adaptation, identify which entry matches the JD type
(tech_roles / product_roles / data_roles / customer_facing_roles / fintech_roles / support_roles)
and follow its lead instruction. If no perfect match, pick the closest.

From origin_stories + seed_sentences, pick:
  - The 1-2 stories most resonant for THIS specific role
  - The seed sentence whose voice fits best
  - The specific lesson that closes the loop to their stated needs

## STEP 3 — WRITE THE ARC
Structure the letter as a narrative arc:
  Hook (why this org specifically) → Your relevant story (1-2 experiences)
  → The bridge (how your story connects to their needs)
  → Forward energy (what you'll do/learn there)

## STEP 4 — HUMAN + AUTHENTICITY TEST
Before generating LaTeX, read the letter as if a real person wrote it at 11 PM.
Ask:
1. Does it sound like a human? Does it have personality?
2. Does it violate anything in MOTIVATION_DATA.what_arpan_is_not?
3. Could this sentence have been written by anyone? → If yes, cut or rewrite.
If it sounds like it could have been written by anyone → rewrite.

---

# GROUND RULES
- Use ONLY information from SELECTED_DATA_POOL and MOTIVATION_DATA. No fabrication.
- Do not reintroduce suppressed experience from the full local profile. A
  podcast/content internship is not technical experience unless the local
  selection explicitly includes it for this JD.
- Respect the owner's user_priority labels in RESUME_SELECTION: primary evidence
  is normally available, secondary evidence needs a direct JD bridge,
  keyword_dependent evidence is rare and needs a direct JD signal, and excluded
  evidence must never be reintroduced.
- Use the selected evidence as the factual boundary; motivation data supplies
  voice and framing, not permission to add an unselected role or skill.
- MOTIVATION_DATA.identity.core_thesis is the spine of the letter. Every paragraph connects back to it.
- MOTIVATION_DATA.role_adaptation tells you which stories to lead with. Follow it.
- MOTIVATION_DATA.what_arpan_is_not tells you what framings to avoid. Respect it.
- MOTIVATION_DATA.tone controls voice, contractions, and formality. Match it.
- If SEED_SENTENCES (runtime param) are provided, they are SACRED — weave in verbatim.
- The letter must work for BOTH scholarships and job applications.
  Adapt formality to match the JD tone.
"""


def build_cl_prompt(
    data_pool: dict,
    motivation_data: dict | None,
    job_description: str,
    seed_sentences: str | None = None,
) -> tuple[str, str]:
    """
    Build (system_message, user_message) for the cover letter API call.

    :param data_pool:        Parsed data_pool.json from Resume Maker.
    :param motivation_data:  Parsed motivation_data.json (may be None).
    :param job_description:  Raw JD / scholarship description.
    :param seed_sentences:   Optional free-text sentences from the user.
    :return:                 (system_content, user_content) tuple.
    """
    # Use the same deterministic evidence slice as resume generation so the
    # cover letter cannot revive an unrelated internship from the raw pool.
    curated_pool, selection = build_curated_profile(data_pool, job_description)
    owner_name, owner_location, owner_email, owner_linkedin = resolve_owner_contact(data_pool)

    contact_parts: list[str] = []
    if owner_email:
        contact_parts.append(f"\\href{{mailto:{owner_email}}}{{{owner_email}}}")
    if owner_linkedin:
        linkedin_text = owner_linkedin.replace("https://", "").replace("http://", "")
        contact_parts.append(f"\\href{{{owner_linkedin}}}{{{linkedin_text}}}")

    header_lines = (f"  \\textbf{{{owner_name}}}\\\\")
    if owner_location:
        header_lines += f"\n  {owner_location}\\\\"
    if contact_parts:
        header_lines += "\n  " + " $|$ ".join(contact_parts)
    minified_pool = json.dumps(curated_pool, separators=(",", ":"), ensure_ascii=False)
    minified_selection = json.dumps(
        {
            "schema_version": selection["schema_version"],
            "role_families": selection["role_families"],
            "detected_skills": selection["detected_skills"],
            "technical_demand": selection["technical_demand"],
            "content_demand": selection["content_demand"],
            "section_order": selection["section_order"],
            "selected": selection["selected"],
            "guardrails": selection["guardrails"],
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )

    # Build user message parts.
    parts: list[str] = [
        "OUTPUT FINAL LATEX ONLY. No explanation. No thinking out loud.\n",
        f"SELECTED_DATA_POOL:\n{minified_pool}\n",
        f"RESUME_SELECTION:\n{minified_selection}\n",
    ]

    if motivation_data:
        minified_motivation = json.dumps(
            motivation_data, separators=(",", ":"), ensure_ascii=False
        )
        parts.append(f"MOTIVATION_DATA:\n{minified_motivation}\n")

    parts.append(f"JOB_DESCRIPTION:\n{job_description.strip()}\n")

    if seed_sentences and seed_sentences.strip():
        parts.append(
            f"SEED_SENTENCES (applicant's own words — weave in naturally):\n"
            f"{seed_sentences.strip()}\n"
        )

    # ── Output contract ───────────────────────────────────────────────────────
    output_contract = (
        "\n---OUTPUT CONTRACT---\n"
        "STRICT OUTPUT MODE. Follow exactly.\n\n"

        "LINE 1 (mandatory): % ORG:<Name> POSITION:<Title>  [underscores for spaces]\n"
        "Then: complete LaTeX document. No markdown. No ``` fences. No prose.\n\n"

        "PREAMBLE — copy these lines verbatim, in this exact order, no additions:\n"
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[letterpaper,top=1in,bottom=1in,left=1.25in,right=1.25in]{geometry}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage{lmodern}\n"
        "\\usepackage[hidelinks]{hyperref}\n"
        "\\pagestyle{empty}\n"
        "\\setlength{\\parskip}{0.9em}\n"
        "\\setlength{\\parindent}{0pt}\n\n"

        "DOCUMENT BODY STRUCTURE — fill content into this skeleton exactly:\n"
        "\\begin{document}\n"
        "\\begin{flushleft}\n"
        f"{header_lines}\n"
        "\\end{flushleft}\n"
        "\\vspace{0.8em}\n"
        "\\today\n"
        "\\vspace{0.8em}\n\n"
        "[Addressee: Hiring Manager name or 'Hiring Team']\\\\\n"
        "[Company Name]\\\\\n"
        "[City, Country]\n\n"
        "\\vspace{0.5em}\n"
        "Dear [Hiring Team / Manager Name],\n\n"
        "[3-4 body paragraphs — plain prose, 350-500 words, NO bullet lists]\n\n"
        "\\vspace{0.8em}\n"
        "[Closing line, e.g.: Best regards,]\n\n"
        "\\vspace{2.5em}\n"
        f"{owner_name}\n"
        "\\end{document}\n\n"

        "HARD RULES:\n"
        "- \\documentclass{article} ONLY — NEVER letter, memoir, or any other class\n"
        "- Body is PROSE ONLY — NO \\begin{itemize}, NO bullet lists\n"
        "- NO % comments inside \\begin{document}...\\end{document}\n"
        "- BANNED packages: fontspec, xunicode, xltxtra, unicode-math, polyglossia, enumitem, titlesec\n"
        "- BANNED commands: \\setmainfont, \\setsansfont, \\begin{letter}, "
        "\\opening, \\closing, \\address, \\signature, \\pagenumbering\n\n"

        "If you cannot comply exactly → output nothing.\n"
    )

    parts.append(output_contract)

    user_message = "\n".join(parts)
    return PCB, user_message
