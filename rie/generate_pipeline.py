"""
generate_pipeline.py
--------------------
Unified Application Packet Generator pipeline.

One JD -> CV + Cover Letter generated in parallel (ThreadPoolExecutor).
This module holds the reusable pipeline logic; the CLI lives in the root
``generate.py`` wrapper.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from rie.llm import AIConfig, load_ai_config
from rie.ats import validate_one_page_budget, validate_resume_ats
from rie.prompts.cv import build_prompt
from rie.prompts.cover_letter import build_cl_prompt
from rie.latex import strip_markdown_fences, is_valid_latex

# Suppress SDK noise from interleaving with our output.
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
CV_OUTPUT = DOCS_DIR / "CV_Output.tex"
CL_OUTPUT = DOCS_DIR / "CL_Output.tex"
POOL_FILE = BASE_DIR / "data_pool.json"
MOTIVATION_FILE = BASE_DIR / "motivation_data.json"
ENV_FILE = BASE_DIR / ".env"

# CV is constrained to one page; CL remains a tight 350-500 word letter.
MAX_CV_TOKENS = 3_500
MAX_CL_TOKENS = 4_000

# Max auto-retries if the model returns non-LaTeX output.
_MAX_RETRIES = 2

# JD word limit -- truncate extremely long JDs to keep prompt cost in check.
_JD_WORD_LIMIT = 1_200


def load_env() -> AIConfig:
    """Load and validate the local provider configuration."""
    return load_ai_config(ENV_FILE)


def load_data_pool() -> dict:
    """Load data_pool.json -- your full profile."""
    if not POOL_FILE.exists():
        sys.exit(f"\n[ERROR] '{POOL_FILE}' not found.\n")
    with POOL_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_motivation_data() -> dict:
    """Load motivation_data.json -- personal voice, seed sentences, values, tone."""
    if not MOTIVATION_FILE.exists():
        sys.exit(f"\n[ERROR] '{MOTIVATION_FILE}' not found.\n")
    with MOTIVATION_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_job_description() -> str:
    """Read a multiline Job Description from stdin; END on a new line finishes."""
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


def call_api(
    config: AIConfig,
    system_msg: str,
    user_msg: str,
    max_tokens: int,
) -> tuple[str, dict]:
    """Non-streaming API call.

    Batch mode is used for parallel execution: interleaving two streaming
    outputs into one terminal garbles the text. Costs are identical.
    """
    client = config.client()

    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
        **config.request_kwargs(),
    )

    content = response.choices[0].message.content or ""
    raw = response.usage

    # Thinking tokens (should be 0).
    thinking = 0
    details = getattr(raw, "completion_tokens_details", None)
    if details:
        thinking = getattr(details, "reasoning_tokens", 0) or 0

    # Cache hit / miss fields are provider-dependent.
    prompt_details = getattr(raw, "prompt_tokens_details", None)
    hit_tok = 0
    miss_tok = 0
    if prompt_details:
        hit_tok = getattr(prompt_details, "cached_tokens", 0) or 0
        miss_tok = getattr(prompt_details, "cache_miss_tokens", 0) or 0
    # Fallback: some SDK versions expose these directly on usage.
    if hit_tok == 0:
        hit_tok = getattr(raw, "prompt_cache_hit_tokens", 0) or 0
    if miss_tok == 0:
        miss_tok = getattr(raw, "prompt_cache_miss_tokens", 0) or 0

    usage = {
        "prompt_tokens": raw.prompt_tokens,
        "completion_tokens": raw.completion_tokens,
        "total_tokens": raw.total_tokens,
        "thinking_tokens": thinking,
        "cache_hit_tokens": hit_tok,
        "cache_miss_tokens": miss_tok,
    }

    return content, usage


def run_cv(config: AIConfig, data_pool: dict, jd: str) -> tuple[str, dict, float]:
    """Generate CV LaTeX. Returns (latex_string, usage_dict, elapsed_seconds)."""
    t0 = time.time()
    system_msg, user_msg = build_prompt(data_pool, jd)

    latex = ""
    usage: dict = {}
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
        content, usage = call_api(config, system_msg, user_msg + retry_suffix, MAX_CV_TOKENS)
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
    """Generate Cover Letter LaTeX. Returns (latex, usage, elapsed_seconds)."""
    t0 = time.time()
    system_msg, user_msg = build_cl_prompt(data_pool, motivation_data, jd, seed_sentences=None)

    latex = ""
    usage: dict = {}
    retry_suffix = ""

    for attempt in range(1 + _MAX_RETRIES):
        if attempt > 0:
            retry_suffix = (
                "\n\nCRITICAL REMINDER: Your previous response was NOT valid LaTeX. "
                "Output ONLY raw LaTeX starting with \\documentclass. "
                "Zero prose. Zero markdown. Zero backticks."
            )
        content, usage = call_api(config, system_msg, user_msg + retry_suffix, MAX_CL_TOKENS)
        latex = strip_markdown_fences(content)
        if is_valid_latex(latex):
            break

    return latex, usage, time.time() - t0


def run_parallel_generation(
    config: AIConfig,
    data_pool: dict,
    motivation_data: dict,
    jd: str,
    *,
    run_cv_fn: Callable | None = None,
    run_cl_fn: Callable | None = None,
) -> tuple[tuple[str, dict, float], tuple[str, dict, float], float]:
    """Run the resume and cover-letter calls concurrently for one JD.

    The runner functions are injectable so callers (e.g. tests patching the
    root ``generate.py`` wrapper) can substitute fakes without monkey-patching
    this module's internal bindings.
    """
    cv_runner = run_cv_fn if run_cv_fn is not None else run_cv
    cl_runner = run_cl_fn if run_cl_fn is not None else run_cl
    if run_cv_fn is None or run_cl_fn is None:
        # Honor patches applied to the root ``generate.py`` wrapper (tests
        # patch generate.run_cv / generate.run_cl and call this function
        # without explicit runner arguments).
        try:
            import generate as _root_wrapper
        except ImportError:
            _root_wrapper = None
        if _root_wrapper is not None:
            if run_cv_fn is None:
                cv_runner = _root_wrapper.run_cv
            if run_cl_fn is None:
                cl_runner = _root_wrapper.run_cl
    wall_start = time.time()
    cv_result = None
    cl_result = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_cv = executor.submit(cv_runner, config, data_pool, jd)
        fut_cl = executor.submit(cl_runner, config, data_pool, motivation_data, jd)

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


def print_combined_summary(
    cv_usage: dict,
    cl_usage: dict,
    cv_elapsed: float,
    cl_elapsed: float,
    wall_time: float,
    config: AIConfig,
) -> None:
    """Print a side-by-side CV | CL | TOTAL token and timing table."""
    cv_p = cv_usage.get("prompt_tokens", 0)
    cl_p = cl_usage.get("prompt_tokens", 0)
    cv_c = cv_usage.get("completion_tokens", 0)
    cl_c = cl_usage.get("completion_tokens", 0)
    cv_h = cv_usage.get("cache_hit_tokens", 0)
    cl_h = cl_usage.get("cache_hit_tokens", 0)
    cv_m = cv_usage.get("cache_miss_tokens", 0) or cv_p
    cl_m = cl_usage.get("cache_miss_tokens", 0) or cl_p
    cv_th = cv_usage.get("thinking_tokens", 0)
    cl_th = cv_usage.get("thinking_tokens", 0)

    W = 62
    print("\n" + "=" * W)
    print(f"  APPLICATION PACKET -- USAGE SUMMARY  |  {config.model}")
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
    print(f"  {'Est. cost':<24}{'see provider dashboard':>10}  {'see provider dashboard':>10}  {'see provider dashboard':>10}")
    print(f"  {'-'*24}{'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'API time':<24}{cv_elapsed:>9.1f}s  {cl_elapsed:>9.1f}s")
    sequential = cv_elapsed + cl_elapsed
    saved = sequential - wall_time
    print(f"  {'Wall time (parallel)':<24}{wall_time:>9.1f}s  (saved {saved:.1f}s vs {sequential:.1f}s sequential)")
    print("=" * W + "\n")
