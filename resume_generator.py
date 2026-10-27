"""
resume_generator.py
-------------------
AI-Powered Resume Intelligence Engine — Main Orchestrator

Usage:
    python resume_generator.py

Flow:
    1. Load .env → API key + model config
    2. Load data_pool.json
    3. Collect Job Description (multiline, type END on a new line to finish)
    4. Build prompt via prompt_builder
    5. Call OpenRouter through the OpenAI-compatible API
    6. Strip any accidental markdown fences
    7. Write output.tex
    8. Extract Company + Role from the LaTeX comment header
    9. Optionally compile to PDF via pdflatex → outputs/Company_Role.pdf
   10. Print token usage summary
"""

import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ai_config import AIConfig, load_ai_config
from ats_validator import validate_one_page_budget, validate_resume_ats
from prompt_builder import build_prompt

# Suppress the openai + httpx libraries' own progress/debug output.
# Without this, their carriage-return spinner interleaves with our print() calls
# and garbles the terminal display on Windows PowerShell.
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
ENV_FILE    = BASE_DIR / ".env"
POOL_FILE   = BASE_DIR / "data_pool.json"
PMOTIVATION_FILE = BASE_DIR / "motivation_data.json"
OUTPUT_TEX  = BASE_DIR / "output.tex"
OUTPUTS_DIR = BASE_DIR / "outputs"

# One-page LaTeX resume output ceiling. Keep enough headroom for valid LaTeX
# while preventing the model from expanding into a two-page document.
MAX_OUTPUT_TOKENS = 3_500

# JD word limit — truncate extremely long JDs to keep prompt readable.
_JD_WORD_LIMIT = 1_200

# Max auto-retries if the model returns non-LaTeX output.
_MAX_RETRIES = 2


def load_env() -> AIConfig:
    """Load and validate the local OpenRouter configuration."""
    return load_ai_config(ENV_FILE)


def load_data_pool() -> dict:
    """Parse data_pool.json from disk."""
    if not POOL_FILE.exists():
        sys.exit(f"\n[ERROR] '{POOL_FILE}' not found. Please create it first.\n")
    with POOL_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_job_description() -> str:
    """
    Prompt the user for a multiline Job Description.
    Input ends when the user types END on a new line (or hits Ctrl+Z on Windows).
    Long JDs are truncated at _JD_WORD_LIMIT words to control prompt token cost.
    """
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

    # Truncate excessively long JDs to keep prompt cost in check.
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
    """
    Send the prompt through OpenRouter's OpenAI-compatible API.

    Provider-specific reasoning controls are intentionally not sent because the
    configured OpenRouter model may come from any supported provider.
    """
    client = config.client()

    print(f"\n[INFO] Sending prompt to '{config.model}' via OpenRouter | temperature=0.7")

    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
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
            cache_hit_tokens  = 0
            cache_miss_tokens = 0
            if prompt_details:
                cache_hit_tokens  = getattr(prompt_details, "cached_tokens",     0) or 0
                cache_miss_tokens = getattr(prompt_details, "cache_miss_tokens",  0) or 0

            # Fallback: some SDK versions expose these directly on usage.
            if cache_hit_tokens == 0:
                cache_hit_tokens  = getattr(raw_usage, "prompt_cache_hit_tokens",  0) or 0
            if cache_miss_tokens == 0:
                cache_miss_tokens = getattr(raw_usage, "prompt_cache_miss_tokens", 0) or 0

            usage = {
                "prompt_tokens":      raw_usage.prompt_tokens,
                "completion_tokens":  raw_usage.completion_tokens,
                "total_tokens":       raw_usage.total_tokens,
                "thinking_tokens":    thinking_count,
                "cache_hit_tokens":   cache_hit_tokens,
                "cache_miss_tokens":  cache_miss_tokens,
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
            "total_tokens": 0,  "thinking_tokens": 0,
            "cache_hit_tokens": 0, "cache_miss_tokens": 0,
        }

    return content, usage


def strip_markdown_fences(text: str) -> str:
    """
    Remove accidental ```latex / ``` wrappers the model may emit.

    Uses re.search (not fullmatch) so it works even when the closing fence
    is missing due to a truncated response. Extracts content from the first
    opening fence to either the closing fence or end-of-string.
    """
    # Case 1: well-formed fence with closing ```
    closed = re.search(r"```(?:latex|tex)?\s*\n(.*?)\n?```", text, re.DOTALL)
    if closed:
        return closed.group(1).strip()

    # Case 2: opening fence present but no closing (truncated output)
    open_fence = re.search(r"```(?:latex|tex)?\s*\n(.*)", text, re.DOTALL)
    if open_fence:
        return open_fence.group(1).strip()

    return text.strip()


def is_valid_latex(text: str) -> bool:
    r"""
    Returns True if the text looks like compilable LaTeX.
    Skips comment lines (%) and checks for \documentclass.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue          # skip blank lines and LaTeX comments
        return stripped.startswith("\\")
    return False


def extract_company_role(latex: str) -> tuple[str, str]:
    """
    Read the % COMPANY:<name> ROLE:<title> comment the LLM is instructed to
    place on the first non-blank line.  Scans all lines (not just line 1) as
    a fallback in case the model inserts a blank line before it.
    Returns (company, role) sanitized for use in the filename.
    """
    pattern = re.compile(r"%\s*COMPANY:(\S+)\s+ROLE:(\S+)", re.IGNORECASE)
    
    def sanitize(text: str) -> str:
        return re.sub(r'[\\/*?:"<>|]', "", text)

    for line in (latex or "").splitlines()[:10]:   # check first 10 lines only
        match = pattern.search(line)
        if match:
            return sanitize(match.group(1)), sanitize(match.group(2))
    return "Unknown_Company", "Unknown_Role"


# Packages that are XeLaTeX/LuaLaTeX-only and will crash pdflatex.
_FORBIDDEN_PACKAGES = {"fontspec", "xunicode", "xltxtra", "unicode-math", "polyglossia"}


def validate_latex_packages(latex: str) -> bool:
    """
    Scan the generated LaTeX for packages that are incompatible with pdflatex.
    Returns True if the output is clean, False if forbidden packages are found.
    Prints a clear warning for each offending line.
    """
    issues: list[str] = []
    pkg_pattern = re.compile(r"\\usepackage(?:\[.*?\])?\{([^}]+)\}", re.DOTALL)

    for lineno, line in enumerate(latex.splitlines(), start=1):
        for match in pkg_pattern.finditer(line):
            # A single \usepackage can load multiple comma-separated packages
            packages = [p.strip() for p in match.group(1).split(",")]
            for pkg in packages:
                if pkg in _FORBIDDEN_PACKAGES:
                    issues.append(f"  Line {lineno}: \\usepackage{{{pkg}}}  ← XeLaTeX-only, crashes pdflatex")

    if issues:
        print("\n[ERROR] Generated LaTeX contains pdflatex-incompatible packages:")
        for issue in issues:
            print(issue)
        print(
            "\n  → Re-running resume_generator.py should fix this automatically.\n"
            "    The output contract now explicitly bans these packages.\n"
            "    If it persists, check prompt_builder.py's output_contract section.\n"
        )
        return False
    return True


def save_latex(latex: str) -> None:
    """Write output.tex to the project root."""
    OUTPUT_TEX.write_text(latex, encoding="utf-8")
    print(f"[OK] LaTeX saved -> {OUTPUT_TEX}")


def compile_pdf(latex: str, company: str, role: str) -> Path | None:
    """
    Compile output.tex → PDF via pdflatex (if available).
    Runs in an isolated temp directory inside the project to avoid cluttering.
    On success, copies the PDF to outputs/Company_Role.pdf.
    Returns the PDF path on success, None on failure or if pdflatex is absent.
    """
    if not shutil.which("pdflatex"):
        print(
            "\n[WARN] pdflatex not found in PATH.\n"
            "       Install a LaTeX distribution (TeX Live / MiKTeX) to enable\n"
            "       auto PDF compilation, or upload output.tex to Overleaf.\n"
        )
        return None

    OUTPUTS_DIR.mkdir(exist_ok=True)
    pdf_name = f"{company}_{role}.pdf"
    dest_pdf  = OUTPUTS_DIR / pdf_name

    # Use a temp directory inside the project so pdflatex aux files stay local.
    tmp_dir = BASE_DIR / ".pdfbuild"
    tmp_dir.mkdir(exist_ok=True)
    tmp_tex = tmp_dir / "resume.tex"
    tmp_tex.write_text(latex, encoding="utf-8")

    print(f"[INFO] Compiling PDF …")

    try:
        # Run pdflatex twice for proper cross-references / hyperlinks.
        for _pass in range(2):
            result = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",   # Never pause for user input
                    "-halt-on-error",             # Exit immediately on error
                    str(tmp_tex.name),
                ],
                cwd=str(tmp_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                _print_latex_error(result.stdout + result.stderr)
                return None

        # Both passes succeeded — copy the PDF out.
        compiled_pdf = tmp_dir / "resume.pdf"
        if not compiled_pdf.exists():
            print("[ERROR] pdflatex exited 0 but resume.pdf was not created.")
            return None

        shutil.copy2(compiled_pdf, dest_pdf)
        print(f"[OK] PDF saved    -> {dest_pdf}")
        return dest_pdf

    except subprocess.TimeoutExpired:
        print("[ERROR] pdflatex timed out after 60 seconds.")
        return None
    except Exception as exc:
        print(f"[ERROR] PDF compilation failed: {exc}")
        return None
    finally:
        # Always clean up the temp build dir (keep outputs/ clean).
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _print_latex_error(log_text: str) -> None:
    """Print the first relevant LaTeX error line from the pdflatex log."""
    print("\n[ERROR] pdflatex compilation failed.")
    for line in log_text.splitlines():
        if line.startswith("!") or "Error" in line:
            print(f"  → {line}")
    print(
        "\n  The LaTeX source has been saved to output.tex.\n"
        "  You can debug it manually or upload to Overleaf.\n"
    )


def print_usage_summary(usage: dict, config: AIConfig) -> None:
    """Print token usage without hard-coding provider/model pricing."""
    p_tok        = usage["prompt_tokens"]
    c_tok        = usage["completion_tokens"]
    t_tok        = usage["total_tokens"]
    th_tok       = usage.get("thinking_tokens",   0)
    hit_tok      = usage.get("cache_hit_tokens",  0)
    miss_tok     = usage.get("cache_miss_tokens", 0)

    print("\n" + "-" * 48)
    print("  Provider         : OpenRouter")
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
    print("  Est. cost        : See OpenRouter usage/pricing for the selected model")
    print("-" * 48 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    config          = load_env()
    data_pool       = load_data_pool()
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

        raw_output, usage = call_openrouter(
            config,
            system_msg,
            user_msg + retry_suffix,
        )
        latex = strip_markdown_fences(raw_output)

        if is_valid_latex(latex) and validate_one_page_budget(latex)["status"] != "fail":
            break
        if attempt < _MAX_RETRIES:
            print("[WARN] Response does not look like LaTeX — will retry.")
        else:
            print("[WARN] Max retries reached. Saving raw output for manual inspection.")

    # ── Validate packages ──────────────────────────────────────────────────────
    latex_is_clean = validate_latex_packages(latex)
    ats_report = validate_resume_ats(latex, job_description)

    # -- Save & compile ---------------------------------------------------------
    save_latex(latex)

    company, role = extract_company_role(latex)
    print(f"[INFO] Detected -> Company: {company}  |  Role: {role}")

    if latex_is_clean:
        compile_pdf(latex, company, role)
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

