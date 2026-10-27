"""
generate.py
-----------
Unified Application Packet Generator
AI-Powered CV + Cover Letter  --  One JD, One Command

Usage:
    python generate.py

Flow:
    1.  Load .env            -> API key + model
    2.  Load data_pool.json  -> your full profile
    3.  Load motivation_data.json  -> your voice / seed sentences for CL
    4.  Collect Job Description from terminal (type END to finish)
    5.  Fire CV + Cover Letter generation in PARALLEL (ThreadPoolExecutor)
    6.  Both buffer silently; live status updates in terminal
    7.  Write docs/CV_Output.tex  (always overwritten)
    8.  Write docs/CL_Output.tex  (always overwritten)
    9.  Print combined cost + timing summary table

Why parallel?
    Both API calls are fully independent I/O-bound network requests.
    Sequential: ~14s + ~13s = ~27s total.
    Parallel:   max(14s, 13s) = ~14s total.  Nearly half the wait time.
    OpenRouter provides a single OpenAI-compatible endpoint for the selected model.
    Cost is identical (you pay per token, not per second).

Output files:
    docs/CV_Output.tex  -- LaTeX resume (overwritten each run)
    docs/CL_Output.tex  -- LaTeX cover letter (overwritten each run)
"""

import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ai_config import AIConfig, load_ai_config
from ats_validator import validate_one_page_budget, validate_resume_ats
from prompt_builder import build_prompt
from cl_prompt_builder import build_cl_prompt

# Suppress SDK noise from interleaving with our output.
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DOCS_DIR        = BASE_DIR / "docs"
CV_OUTPUT       = DOCS_DIR / "CV_Output.tex"
CL_OUTPUT       = DOCS_DIR / "CL_Output.tex"
POOL_FILE       = BASE_DIR / "data_pool.json"
MOTIVATION_FILE = BASE_DIR / "motivation_data.json"
ENV_FILE        = BASE_DIR / ".env"

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
# CV is constrained to one page; CL remains a tight 350-500 word letter.
MAX_CV_TOKENS = 3_500
MAX_CL_TOKENS = 4_000

# Max auto-retries if the model returns non-LaTeX output.
_MAX_RETRIES = 2

# JD word limit -- truncate extremely long JDs to keep prompt cost in check.
_JD_WORD_LIMIT = 1_200

# Packages that crash pdflatex (XeLaTeX/LuaLaTeX-only).
_FORBIDDEN_PACKAGES = {"fontspec", "xunicode", "xltxtra", "unicode-math", "polyglossia"}


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_env() -> AIConfig:
    """Load and validate the local OpenRouter configuration."""
    return load_ai_config(ENV_FILE)


def load_data_pool() -> dict:
    """Load data_pool.json -- your full profile."""
    if not POOL_FILE.exists():
        sys.exit(f"\n[ERROR] '{POOL_FILE}' not found.\n")
    with POOL_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_motivation_data() -> dict:
    """
    Load motivation_data.json -- personal voice, seed sentences, values, tone.
    Required for Cover Letter generation. No interactive seed input is asked;
    the file provides everything the CL engine needs.
    """
    if not MOTIVATION_FILE.exists():
        sys.exit(
            f"\n[ERROR] '{MOTIVATION_FILE}' not found.\n"
            f"  -> Expected at project root: {MOTIVATION_FILE}\n"
            f"  -> Copy it from 'Cover Letter Maker/motivation_data.json' if needed.\n"
        )
    with MOTIVATION_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_job_description() -> str:
    """
    Read a multiline Job Description from stdin.
    User types END on a new line to finish.
    Truncates at _JD_WORD_LIMIT to keep prompt cost predictable.
    """
    print("\n" + "=" * 62)
    print("  APPLICATION PACKET GENERATOR")
    print("  Resume + Cover Letter  --  One JD, One Command")
    print("=" * 62)
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
        pass

    jd = "\n".join(lines).strip()
    if not jd:
        sys.exit("\n[ERROR] Job Description was empty. Aborting.\n")

    words = jd.split()
    if len(words) > _JD_WORD_LIMIT:
        jd = " ".join(words[:_JD_WORD_LIMIT])
        print(
            f"\n[INFO] JD truncated from {len(words)} -> {_JD_WORD_LIMIT} words "
            f"(~{len(words) - _JD_WORD_LIMIT} words removed).\n"
        )

    return jd


# ─────────────────────────────────────────────────────────────────────────────
# Shared post-processing helpers (mirrors resume_generator.py)
# ─────────────────────────────────────────────────────────────────────────────
def strip_markdown_fences(text: str) -> str:
    """Remove accidental ```latex / ``` wrappers the model may emit."""
    closed = re.search(r"```(?:latex|tex)?\s*\n(.*?)\n?```", text, re.DOTALL)
    if closed:
        return closed.group(1).strip()
    open_fence = re.search(r"```(?:latex|tex)?\s*\n(.*)", text, re.DOTALL)
    if open_fence:
        return open_fence.group(1).strip()
    return text.strip()


def is_valid_latex(text: str) -> bool:
    r"""Returns True if the text looks like compilable LaTeX (starts with \)."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        return stripped.startswith("\\")
    return False


def validate_latex_packages(latex: str, label: str) -> bool:
    """Scan for XeLaTeX-only packages that crash pdflatex. Returns True if clean."""
    pkg_pattern = re.compile(r"\\usepackage(?:\[.*?\])?\{([^}]+)\}", re.DOTALL)
    issues: list[str] = []
    for lineno, line in enumerate(latex.splitlines(), start=1):
        for match in pkg_pattern.finditer(line):
            for pkg in [p.strip() for p in match.group(1).split(",")]:
                if pkg in _FORBIDDEN_PACKAGES:
                    issues.append(f"  Line {lineno}: \\usepackage{{{pkg}}}")
    if issues:
        print(f"\n[WARN] {label} -- forbidden packages detected:")
        for issue in issues:
            print(issue)
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# API call (batch mode -- cleaner for parallel execution)
# ─────────────────────────────────────────────────────────────────────────────
def call_api(
    config: AIConfig,
    system_msg: str,
    user_msg: str,
    max_tokens: int,
) -> tuple[str, dict]:
    """
    Non-streaming API call with thinking explicitly disabled.

    Why non-streaming (batch) here instead of streaming like the standalone scripts?
    ─────────────────────────────────────────────────────────────────────────────
    Both CV and CL run simultaneously in separate threads.  Interleaving two
    streaming outputs into one terminal produces garbled text.  Batch mode
    buffers the complete response before returning -- the threads finish
    independently and we print results cleanly after both complete.

    Both approaches cost identical tokens.  The UX difference is that instead
    of watching text appear character-by-character, you see a clean
    "[OK] CV complete (14.2s)" line when each finishes.

    Provider-specific reasoning controls are intentionally omitted so the
    configured OpenRouter model can come from any supported provider.
    """
    client = config.client()

    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
        **config.request_kwargs(),
    )

    content = response.choices[0].message.content or ""
    raw     = response.usage

    # Thinking tokens (should be 0).
    thinking = 0
    details  = getattr(raw, "completion_tokens_details", None)
    if details:
        thinking = getattr(details, "reasoning_tokens", 0) or 0

    # Cache hit / miss fields are provider-dependent.
    prompt_details = getattr(raw, "prompt_tokens_details", None)
    hit_tok  = 0
    miss_tok = 0
    if prompt_details:
        hit_tok  = getattr(prompt_details, "cached_tokens",    0) or 0
        miss_tok = getattr(prompt_details, "cache_miss_tokens", 0) or 0
    # Fallback: some SDK versions expose these directly on usage.
    if hit_tok == 0:
        hit_tok  = getattr(raw, "prompt_cache_hit_tokens",  0) or 0
    if miss_tok == 0:
        miss_tok = getattr(raw, "prompt_cache_miss_tokens", 0) or 0

    usage = {
        "prompt_tokens":     raw.prompt_tokens,
        "completion_tokens": raw.completion_tokens,
        "total_tokens":      raw.total_tokens,
        "thinking_tokens":   thinking,
        "cache_hit_tokens":  hit_tok,
        "cache_miss_tokens": miss_tok,
    }

    return content, usage


# ─────────────────────────────────────────────────────────────────────────────
# Task runners -- called by ThreadPoolExecutor workers
# ─────────────────────────────────────────────────────────────────────────────
def run_cv(config: AIConfig, data_pool: dict, jd: str) -> tuple[str, dict, float]:
    """
    Generate CV LaTeX.
    Returns (latex_string, usage_dict, elapsed_seconds).
    """
    t0 = time.time()
    system_msg, user_msg = build_prompt(data_pool, jd)

    latex        = ""
    usage: dict  = {}
    retry_suffix = ""

    for attempt in range(1 + _MAX_RETRIES):
        if attempt > 0:
            retry_suffix = (
                "\n\nCRITICAL REMINDER: Your previous response violated the output contract "
                "or exceeded the one-page content budget. Output ONLY raw LaTeX "
                "starting with \\documentclass, with maximum 2 experience entries, "
                "3 projects, 2 certifications, 3 skill rows, and 2 bullets per "
                "experience/project entry. Zero prose. Zero markdown. Zero backticks."
            )
        content, usage = call_api(
            config, system_msg, user_msg + retry_suffix, MAX_CV_TOKENS
        )
        latex = strip_markdown_fences(content)
        if is_valid_latex(latex) and validate_one_page_budget(latex)["status"] != "fail":
            break

    return latex, usage, time.time() - t0


def run_cl(
    config: AIConfig,
    data_pool: dict,
    motivation_data: dict,
    jd: str,
) -> tuple[str, dict, float]:
    """
    Generate Cover Letter LaTeX.
    Returns (latex_string, usage_dict, elapsed_seconds).
    seed_sentences=None -- motivation_data.json provides all personal voice.
    """
    t0 = time.time()
    system_msg, user_msg = build_cl_prompt(
        data_pool, motivation_data, jd, seed_sentences=None
    )

    latex        = ""
    usage: dict  = {}
    retry_suffix = ""

    for attempt in range(1 + _MAX_RETRIES):
        if attempt > 0:
            retry_suffix = (
                "\n\nCRITICAL REMINDER: Your previous response was NOT valid LaTeX. "
                "Output ONLY raw LaTeX starting with \\documentclass. "
                "Zero prose. Zero markdown. Zero backticks."
            )
        content, usage = call_api(
            config, system_msg, user_msg + retry_suffix, MAX_CL_TOKENS
        )
        latex = strip_markdown_fences(content)
        if is_valid_latex(latex):
            break

    return latex, usage, time.time() - t0


def run_parallel_generation(
    config: AIConfig,
    data_pool: dict,
    motivation_data: dict,
    jd: str,
) -> tuple[tuple[str, dict, float], tuple[str, dict, float], float]:
    """Run the resume and cover-letter calls concurrently for one JD."""
    wall_start = time.time()
    cv_result = None
    cl_result = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_cv = executor.submit(run_cv, config, data_pool, jd)
        fut_cl = executor.submit(run_cl, config, data_pool, motivation_data, jd)

        for future in as_completed({fut_cv, fut_cl}):
            result = future.result()
            elapsed = result[2]
            if future is fut_cv:
                cv_result = result
                print(f"  [OK] CV   complete  ({elapsed:.1f}s)")
            else:
                cl_result = result
                print(f"  [OK] CL   complete  ({elapsed:.1f}s)")

    if cv_result is None or cl_result is None:
        raise RuntimeError("parallel generation did not return both artifacts")
    return cv_result, cl_result, time.time() - wall_start


# ─────────────────────────────────────────────────────────────────────────────
# Metadata extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_company_role(latex: str) -> tuple[str, str]:
    """
    Read the % COMPANY:<name> ROLE:<title> comment the CV LLM places on line 1.
    Used for display only (no archive naming -- output files are always overwritten).
    """
    pattern  = re.compile(r"%\s*COMPANY:(\S+)\s+ROLE:(\S+)", re.IGNORECASE)
    sanitize = lambda t: re.sub(r'[\\/*?:"<>|]', "", t)
    for line in (latex or "").splitlines()[:10]:
        m = pattern.search(line)
        if m:
            return sanitize(m.group(1)), sanitize(m.group(2))
    return "Unknown_Company", "Unknown_Role"


# ─────────────────────────────────────────────────────────────────────────────
# Combined usage summary
# ─────────────────────────────────────────────────────────────────────────────
def print_combined_summary(
    cv_usage: dict,
    cl_usage: dict,
    cv_elapsed: float,
    cl_elapsed: float,
    wall_time: float,
    config: AIConfig,
) -> None:
    """Print a side-by-side CV | CL | TOTAL token and timing table."""

    cv_p  = cv_usage.get("prompt_tokens",     0)
    cl_p  = cl_usage.get("prompt_tokens",     0)
    cv_c  = cv_usage.get("completion_tokens", 0)
    cl_c  = cl_usage.get("completion_tokens", 0)
    cv_h  = cv_usage.get("cache_hit_tokens",  0)
    cl_h  = cl_usage.get("cache_hit_tokens",  0)
    cv_m  = cv_usage.get("cache_miss_tokens", 0) or cv_p
    cl_m  = cl_usage.get("cache_miss_tokens", 0) or cl_p
    cv_th = cv_usage.get("thinking_tokens",   0)
    cl_th = cl_usage.get("thinking_tokens",   0)

    W = 62
    print("\n" + "=" * W)
    print(f"  APPLICATION PACKET -- USAGE SUMMARY  |  OpenRouter / {config.model}")
    print("=" * W)
    print(f"  {'':24}{'CV':>10}  {'CL':>10}  {'TOTAL':>10}")
    print(f"  {'-'*24}{'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'Prompt tokens':<24}{cv_p:>10,}  {cl_p:>10,}  {cv_p+cl_p:>10,}")
    _miss = "\\- Cache miss"
    print(f"    {'|- Cache hit':<22}{cv_h:>10,}  {cl_h:>10,}  {cv_h+cl_h:>10,}")
    print(f"    {_miss:<22}{cv_m:>10,}  {cl_m:>10,}  {cv_m+cl_m:>10,}")
    if cv_th or cl_th:
        print(f"  {'Thinking tokens':<24}{cv_th:>10,}  {cl_th:>10,}  {cv_th+cl_th:>10,}  (!) should be 0")
    print(f"  {'Completion tokens':<24}{cv_c:>10,}  {cl_c:>10,}  {cv_c+cl_c:>10,}")
    print(f"  {'Est. cost':<24}{'see OpenRouter dashboard':>10}  {'see OpenRouter dashboard':>10}  {'see OpenRouter dashboard':>10}")
    print(f"  {'-'*24}{'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'API time':<24}{cv_elapsed:>9.1f}s  {cl_elapsed:>9.1f}s")
    sequential = cv_elapsed + cl_elapsed
    saved      = sequential - wall_time
    print(f"  {'Wall time (parallel)':<24}{wall_time:>9.1f}s  (saved {saved:.1f}s vs {sequential:.1f}s sequential)")
    print("=" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    config          = load_env()
    data_pool       = load_data_pool()
    motivation_data = load_motivation_data()
    jd              = collect_job_description()

    # Ensure docs/ exists.
    DOCS_DIR.mkdir(exist_ok=True)

    print(f"\n[INFO] Provider: OpenRouter | Model: {config.model} | parallel=True")
    if config.provider_options:
        print(f"[INFO] Provider routing: {config.provider_options}")
    if config.reasoning_options:
        print(f"[INFO] Reasoning policy: {config.reasoning_options}")
    print("[...] Generating CV and Cover Letter simultaneously...\n")

    cv_result, cl_result, wall_time = run_parallel_generation(
        config, data_pool, motivation_data, jd
    )

    cv_latex, cv_usage, cv_elapsed = cv_result
    cl_latex, cl_usage, cl_elapsed = cl_result

    # Validate packages (warn but don't block save).
    cv_clean = validate_latex_packages(cv_latex, "CV")
    cl_clean = validate_latex_packages(cl_latex, "CL")
    ats_report = validate_resume_ats(cv_latex, jd)

    # Save -- always overwrite.
    CV_OUTPUT.write_text(cv_latex, encoding="utf-8")
    CL_OUTPUT.write_text(cl_latex, encoding="utf-8")

    company, role = extract_company_role(cv_latex)

    print(f"\n[OK] docs/CV_Output.tex  saved")
    print(f"[OK] docs/CL_Output.tex  saved")
    print(f"[INFO] Detected -> Company: {company}  |  Role: {role}")

    if not cv_clean:
        print("[WARN] CV has forbidden packages -- review output manually.")
    if not cl_clean:
        print("[WARN] CL has forbidden packages -- review output manually.")

    ats_summary = ats_report["summary"]
    print(
        "[ATS] Local CV validation -> "
        f"{ats_report['status']} "
        f"({ats_summary['passed']} pass, {ats_summary['review']} review, {ats_summary['failed']} fail)"
    )

    print_combined_summary(
        cv_usage, cl_usage,
        cv_elapsed, cl_elapsed,
        wall_time, config,
    )
    print("[OK] Application packet ready.")
    print(f"     Upload docs/CV_Output.tex and docs/CL_Output.tex to Overleaf.\n")


if __name__ == "__main__":
    main()
