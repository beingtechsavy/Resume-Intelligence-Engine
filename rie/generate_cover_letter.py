"""
generate_cover_letter.py
------------------------
Cover Letter generator — streaming API orchestrator.

Loads .env + data_pool.json + motivation_data.json, collects the JD/scholarship
description (and optional seed sentences), builds the cover-letter prompt, and
streams the model's LaTeX to output.<doc>.tex, optionally compiling a PDF.
Consolidates the former ``Cover Letter Maker/cover_letter_generator.py``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from rie.llm import AIConfig, load_ai_config
from rie.prompts.cover_letter import build_cl_prompt
from rie.latex import (
    extract_org_position,
    is_valid_latex,
    strip_markdown_fences,
    validate_latex_packages,
)
from rie.generate_resume import call_openrouter, print_usage_summary

# Quiet the openai/httpx libraries like the other orchestrators do.
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
POOL_FILE = BASE_DIR / "data_pool.json"
MOTIVATION_FILE = BASE_DIR / "motivation_data.json"
OUTPUT_TEX = BASE_DIR / "output.tex"
OUTPUTS_DIR = BASE_DIR / "Cover letters"

# LaTeX output token ceiling. A full cover letter ~1,500-2,500 tokens.
MAX_OUTPUT_TOKENS = 4_000

# JD word limit — truncate extremely long JDs.
_JD_WORD_LIMIT = 1_200

# Max auto-retries if the model returns non-LaTeX output.
_MAX_RETRIES = 2


def load_env() -> AIConfig:
    """Load and validate the local provider configuration."""
    return load_ai_config(ENV_FILE)


def load_data_pool() -> dict:
    """Parse data_pool.json from disk."""
    if not POOL_FILE.exists():
        sys.exit(f"\n[ERROR] '{POOL_FILE}' not found. Please create it first.\n")
    with POOL_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_motivation_data() -> dict | None:
    """Load motivation_data.json (optional). Returns None if not found."""
    if not MOTIVATION_FILE.exists():
        print("[INFO] No motivation_data.json found -- running without personal seeds.\n")
        return None
    with MOTIVATION_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_job_description() -> str:
    """Read a multiline JD / scholarship description; END on a new line finishes."""
    print("\n" + "=" * 60)
    print("  COVER LETTER INTELLIGENCE ENGINE")
    print("=" * 60)
    print("\nPaste the Job Description / Scholarship info below.")
    print("When done, type  END  on a new line and press Enter.\n")

    lines: list[str] = []
    try:
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
    except EOFError:
        pass

    jd = "\n".join(lines).strip()
    if not jd:
        sys.exit("\n[ERROR] Description was empty. Aborting.\n")

    words = jd.split()
    if len(words) > _JD_WORD_LIMIT:
        jd = " ".join(words[:_JD_WORD_LIMIT])
        print(
            f"\n[INFO] Description truncated from {len(words)} -> {_JD_WORD_LIMIT} words "
            f"(~{len(words) - _JD_WORD_LIMIT} words removed).\n"
        )

    return jd


def collect_seed_sentences(motivation_data: dict | None) -> str | None:
    """Ask for optional seed sentences only when motivation_data lacks them."""
    if motivation_data and motivation_data.get("seed_sentences"):
        seeds = motivation_data["seed_sentences"]
        print(f"[INFO] Using {len(seeds)} pre-written seed sentences from motivation_data.json\n")
        return None  # seeds are passed via motivation_data directly

    print("-" * 60)
    print("(Optional) Write 2-3 sentences IN YOUR OWN WORDS about")
    print("why this position/scholarship matters to you.")
    print("Type  END  when done, or  SKIP  to skip.\n")

    lines: list[str] = []
    try:
        while True:
            line = input()
            upper = line.strip().upper()
            if upper == "END":
                break
            if upper == "SKIP":
                print("[INFO] Skipping seed sentences.\n")
                return None
            lines.append(line)
    except EOFError:
        pass

    text = "\n".join(lines).strip()
    if not text:
        print("[INFO] No seed sentences provided.\n")
        return None

    print(f"[OK] Captured {len(text.split())} words of personal voice\n")
    return text


def main() -> None:
    config = load_env()
    data_pool = load_data_pool()
    motivation_data = load_motivation_data()
    job_description = collect_job_description()
    seed_sentences = collect_seed_sentences(motivation_data)

    system_msg, user_msg = build_cl_prompt(
        data_pool, motivation_data, job_description, seed_sentences
    )

    # -- API call with auto-retry on non-LaTeX output --------------------------
    latex = ""
    usage: dict = {}
    retry_suffix = ""

    for attempt in range(1 + _MAX_RETRIES):
        if attempt > 0:
            print(f"\n[RETRY] Attempt {attempt + 1}/{1 + _MAX_RETRIES} -- stricter reminder ...")
            retry_suffix = (
                "\n\nCRITICAL REMINDER: Your previous response was NOT valid LaTeX. "
                "Output ONLY raw LaTeX starting with \\documentclass. "
                "Zero prose. Zero markdown. Zero backticks."
            )

        raw_output, usage = call_openrouter(
            config, system_msg, user_msg + retry_suffix
        )
        latex = strip_markdown_fences(raw_output)

        if is_valid_latex(latex):
            break
        if attempt < _MAX_RETRIES:
            print("[WARN] Response does not look like LaTeX  --  will retry.")
        else:
            print("[WARN] Max retries reached. Saving raw output for manual inspection.")

    # -- Validate packages ------------------------------------------------------
    latex_is_clean = validate_latex_packages(latex, "CL")

    # -- Save & compile ---------------------------------------------------------
    OUTPUT_TEX.write_text(latex, encoding="utf-8")
    print(f"[OK] LaTeX saved -> {OUTPUT_TEX}")

    org, position = extract_org_position(latex)
    print(f"[INFO] Detected -> Org: {org}  |  Position: {position}")

    if latex_is_clean:
        from rie.latex import compile_pdf

        compile_pdf(
            latex,
            pdf_name=f"{org}_{position}.pdf",
            output_dir=OUTPUTS_DIR,
            project_dir=BASE_DIR,
            build_name="cover_letter.tex",
        )
    else:
        print("[SKIP] PDF compilation skipped due to package errors above.\n")

    print_usage_summary(usage, config)
    print("[OK] Done. Your cover letter is ready.\n")


if __name__ == "__main__":
    main()
