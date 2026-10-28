"""
generate_resume.py
------------------
Standalone CV generator — streaming API orchestrator.

Loads .env + data_pool.json, collects the JD from stdin, builds the recruiter-
first prompt, streams the model's LaTeX to output.tex, and optionally compiles
a PDF. The CLI entry point is the root ``resume_generator.py`` wrapper.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from rie.llm import AIConfig, load_ai_config
from rie.ats import validate_one_page_budget, validate_resume_ats
from rie.prompts.cv import build_prompt
from rie.latex import (
    extract_company_role,
    is_valid_latex,
    strip_markdown_fences,
    validate_latex_packages,
)

# Suppress the openai + httpx libraries' own progress/debug output.
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
POOL_FILE = BASE_DIR / "data_pool.json"
OUTPUT_TEX = BASE_DIR / "output.tex"
OUTPUTS_DIR = BASE_DIR / "outputs"

# One-page LaTeX resume output ceiling. Keep enough headroom for valid LaTeX
# while preventing the model from expanding into a two-page document.
MAX_OUTPUT_TOKENS = 3_500

# JD word limit — truncate extremely long JDs to keep prompt readable.
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


def collect_job_description() -> str:
    """Read a multiline JD from stdin; END on a new line finishes."""
    print("\n" + "=" * 60)
    print("  RESUME INTELLIGENCE ENGINE")
    print("=" * 60)
    print("\nPaste the Job Description below.")
    print("When done, type  END  on a new line and press Enter.\n")

    lines: list[str] = []
    try:
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
    except EOFError:
        pass  # Ctrl+Z / piped input

    jd = "\n".join(lines).strip()
    if not jd:
        sys.exit("\n[ERROR] Job Description was empty. Aborting.\n")

    words = jd.split()
    if len(words) > _JD_WORD_LIMIT:
        jd = " ".join(words[:_JD_WORD_LIMIT])
        print(
            f"\n[INFO] JD truncated from {len(words)} → {_JD_WORD_LIMIT} words "
            f"to reduce token cost (~{len(words) - _JD_WORD_LIMIT} words removed).\n"
        )

    return jd


def call_openrouter(
    config: AIConfig,
    system_msg: str,
    user_msg: str,
) -> tuple[str, dict]:
    """Send the prompt to the configured provider, streaming the response."""
    client = config.client()

    print(f"\n[INFO] Sending prompt to '{config.model}' | temperature=0.7")

    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        stream=True,
        temperature=0.7,
        stream_options={"include_usage": True},
        **config.request_kwargs(),
    )

    print("\n[API] Connected. Streaming response in real-time...\n")
    print("-" * 60)

    content_chunks = []
    usage = {}

    for chunk in response:
        # Usage info arrives in the final chunk when include_usage=True.
        if hasattr(chunk, "usage") and chunk.usage:
            raw_usage = chunk.usage

            # Track reasoning-token fields when the selected model reports them.
            thinking_count = 0
            details = getattr(raw_usage, "completion_tokens_details", None)
            if details and hasattr(details, "reasoning_tokens"):
                thinking_count = details.reasoning_tokens or 0

            # Prefix cache hit/miss fields are provider-dependent.
            prompt_details = getattr(raw_usage, "prompt_tokens_details", None)
            cache_hit_tokens = 0
            cache_miss_tokens = 0
            if prompt_details:
                cache_hit_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
                cache_miss_tokens = getattr(prompt_details, "cache_miss_tokens", 0) or 0

            # Fallback: some SDK versions expose these directly on usage.
            if cache_hit_tokens == 0:
                cache_hit_tokens = getattr(raw_usage, "prompt_cache_hit_tokens", 0) or 0
            if cache_miss_tokens == 0:
                cache_miss_tokens = getattr(raw_usage, "prompt_cache_miss_tokens", 0) or 0

            usage = {
                "prompt_tokens": raw_usage.prompt_tokens,
                "completion_tokens": raw_usage.completion_tokens,
                "total_tokens": raw_usage.total_tokens,
                "thinking_tokens": thinking_count,
                "cache_hit_tokens": cache_hit_tokens,
                "cache_miss_tokens": cache_miss_tokens,
            }

        # Accumulate streamed content.
        if chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                content_chunks.append(delta.content)

    print("Done!\n" + "-" * 60)
    content = "".join(content_chunks)

    # Fallback if usage wasn't returned in the stream.
    if not usage:
        usage = {
            "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "thinking_tokens": 0,
            "cache_hit_tokens": 0, "cache_miss_tokens": 0,
        }

    return content, usage


def print_usage_summary(usage: dict, config: AIConfig) -> None:
    """Print token usage without hard-coding provider/model pricing."""
    p_tok = usage["prompt_tokens"]
    c_tok = usage["completion_tokens"]
    t_tok = usage["total_tokens"]
    th_tok = usage.get("thinking_tokens", 0)
    hit_tok = usage.get("cache_hit_tokens", 0)
    miss_tok = usage.get("cache_miss_tokens", 0)

    print("\n" + "-" * 48)
    print("  Provider         : configured endpoint")
    print(f"  Model            : {config.model}")
    print(f"  Prompt tokens    : {p_tok:,}")
    if hit_tok or miss_tok:
        print(f"    |- Cache hit     : {hit_tok:,}")
        _miss_label = "\\- Cache miss"
        print(f"    {_miss_label}    : {miss_tok:,}")
    if th_tok:
        print(f"  Thinking tokens  : {th_tok:,}  (!) reasoning cost (should be 0)")
    print(f"  Completion tokens: {c_tok:,}")
    print(f"  Total tokens     : {t_tok:,}")
    print("  Est. cost        : See your provider's usage/pricing for the selected model")
    print("-" * 48 + "\n")


def main() -> None:
    config = load_env()
    data_pool = load_data_pool()
    job_description = collect_job_description()

    system_msg, user_msg = build_prompt(data_pool, job_description)

    # ── API call with auto-retry on non-LaTeX output ──────────────────────────
    latex = ""
    usage: dict = {}
    retry_suffix = ""

    for attempt in range(1 + _MAX_RETRIES):
        if attempt > 0:
            print(f"\n[RETRY] Attempt {attempt + 1}/{1 + _MAX_RETRIES} — injecting stricter reminder …")
            retry_suffix = (
                "\n\nCRITICAL REMINDER: Your previous response violated the output contract "
                "or exceeded the one-page content budget. Output ONLY raw LaTeX "
                "starting with \\documentclass, with maximum 2 experience entries, "
                "3 projects, 2 certifications, 3 skill rows, and 2 bullets per "
                "experience/project entry. Zero prose. Zero markdown. Zero backticks."
            )

        raw_output, usage = call_openrouter(config, system_msg, user_msg + retry_suffix)
        latex = strip_markdown_fences(raw_output)

        if is_valid_latex(latex) and validate_one_page_budget(latex)["status"] != "fail":
            break
        if attempt < _MAX_RETRIES:
            print("[WARN] Response does not look like LaTeX — will retry.")
        else:
            print("[WARN] Max retries reached. Saving raw output for manual inspection.")

    # ── Validate packages ──────────────────────────────────────────────────────
    latex_is_clean = validate_latex_packages(latex, "CV")
    ats_report = validate_resume_ats(latex, job_description)

    # -- Save & compile ---------------------------------------------------------
    OUTPUT_TEX.write_text(latex, encoding="utf-8")
    print(f"[OK] LaTeX saved -> {OUTPUT_TEX}")

    company, role = extract_company_role(latex)
    print(f"[INFO] Detected -> Company: {company}  |  Role: {role}")

    if latex_is_clean:
        from rie.latex import compile_pdf

        compile_pdf(
            latex,
            pdf_name=f"{company}_{role}.pdf",
            output_dir=OUTPUTS_DIR,
            project_dir=BASE_DIR,
            build_name="resume.tex",
        )
    else:
        print("[SKIP] PDF compilation skipped due to package errors above.\n")

    ats_summary = ats_report["summary"]
    print(
        "[ATS] Local validation -> "
        f"{ats_report['status']} "
        f"({ats_summary['passed']} pass, {ats_summary['review']} review, {ats_summary['failed']} fail)"
    )

    print_usage_summary(usage, config)
    print("[OK] Done. Your resume is ready.\n")


if __name__ == "__main__":
    main()
