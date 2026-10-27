"""
cover_letter_generator.py
-------------------------
AI-Powered Cover Letter Intelligence Engine  --  Main Orchestrator

Usage:
    python cover_letter_generator.py

Flow:
    1.  Load .env  ->  OpenRouter API key + model config
    2.  Load data_pool.json (from ../Resume Maker/)
    3.  Load motivation_data.json (local, optional)
    4.  Collect JD / scholarship description (multiline, END to finish)
    5.  Optionally collect seed sentences (user's own words)
    6.  Build prompt via cl_prompt_builder
    7.  Call OpenRouter through the OpenAI-compatible API
    8.  Strip any accidental markdown fences
    9.  Write output.tex
    10. Extract Org + Position from the LaTeX comment header
    11. Optionally compile to PDF via pdflatex  ->  outputs/Org_Position.pdf
    12. Print token usage summary
"""

import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

# The shared provider config lives at the project root while this script keeps
# its dedicated local prompt builder and output files. Append the root so the
# local ``cl_prompt_builder.py`` remains the import that wins.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_config import AIConfig, load_ai_config
from cl_prompt_builder import build_cl_prompt

# Suppress the openai + httpx libraries' own progress/debug output.
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
BASE_DIR         = Path(__file__).parent
ENV_FILE         = BASE_DIR / ".env"
# motivation_data.json lives at project root (shared with generate.py).
MOTIVATION_FILE  = BASE_DIR.parent / "motivation_data.json"
OUTPUT_TEX       = BASE_DIR / "output.tex"
# Cover letters archive folder lives at project root.
OUTPUTS_DIR      = BASE_DIR.parent / "Cover letters"

# data_pool.json lives at project root  --  single source of truth.
POOL_FILE = BASE_DIR.parent / "data_pool.json"

# LaTeX output token ceiling.
# A full cover letter: ~1,500-2,500 tokens.  4,000 gives plenty of room.
MAX_OUTPUT_TOKENS = 4_000

# JD word limit  --  truncate extremely long JDs.
_JD_WORD_LIMIT = 1_200

# Max auto-retries if the model returns non-LaTeX output.
_MAX_RETRIES = 2


# ------------------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------------------
def load_env() -> AIConfig:
    """Load and validate the local or project-root OpenRouter configuration."""
    return load_ai_config(ENV_FILE, BASE_DIR.parent / ".env")


def load_data_pool() -> dict:
    """Load data_pool.json from the Resume Maker folder."""
    if not POOL_FILE.exists():
        sys.exit(
            f"\n[ERROR] '{POOL_FILE}' not found.\n"
            f"  The cover letter generator reads your profile from the Resume Maker's\n"
            f"  data_pool.json. Make sure it exists at:\n"
            f"  {POOL_FILE}\n"
        )
    with POOL_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_motivation_data() -> dict | None:
    """Load motivation_data.json (optional). Returns None if not found."""
    if not MOTIVATION_FILE.exists():
        print("[INFO] No motivation_data.json found -- running without personal seeds.")
        print("       (Create one to improve authenticity and reduce AI detection.)\n")
        return None
    with MOTIVATION_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[OK] Loaded motivation data ({len(data.get('seed_sentences', {}))} seed sentences)")
    return data


# ------------------------------------------------------------------------------
# Input collection
# ------------------------------------------------------------------------------
def collect_job_description() -> str:
    """
    Prompt the user for a multiline JD / scholarship description.
    Input ends when the user types END on a new line (or Ctrl+Z on Windows).
    """
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

    # Truncate excessively long JDs.
    words = jd.split()
    if len(words) > _JD_WORD_LIMIT:
        jd = " ".join(words[:_JD_WORD_LIMIT])
        print(
            f"\n[INFO] Description truncated from {len(words)} -> {_JD_WORD_LIMIT} words "
            f"(~{len(words) - _JD_WORD_LIMIT} words removed).\n"
        )

    return jd


def collect_seed_sentences(motivation_data: dict | None) -> str | None:
    """
    Ask the user for optional seed sentences IF motivation_data doesn't
    already have them.  If motivation_data has seeds, skip this step
    to keep the flow fast for mass production.
    """
    # If we already have seed sentences in motivation_data, skip.
    if motivation_data and motivation_data.get("seed_sentences"):
        seeds = motivation_data["seed_sentences"]
        print(f"[INFO] Using {len(seeds)} pre-written seed sentences from motivation_data.json")
        print("       (To write fresh ones for this application, rename/remove the file.)\n")
        return None  # seeds are passed via motivation_data directly

    # No pre-written seeds  --  ask the user.
    print("-" * 60)
    print("(Optional) Write 2-3 sentences IN YOUR OWN WORDS about")
    print("why this position/scholarship matters to you.")
    print("These will be woven into the letter to make it authentically yours.")
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


# ------------------------------------------------------------------------------
# API call
# ------------------------------------------------------------------------------
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

    print(f"[INFO] Sending prompt to '{config.model}' via OpenRouter | temperature=0.7")

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


# ------------------------------------------------------------------------------
# Post-processing & validation (mirrors resume_generator.py)
# ------------------------------------------------------------------------------
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
    """Returns True if the text looks like compilable LaTeX."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        return stripped.startswith("\\")
    return False


def extract_org_position(latex: str) -> tuple[str, str]:
    """
    Read the % ORG:<name> POSITION:<title> comment the LLM places
    on the first non-blank line.
    Returns (org, position) sanitized for use in the filename.
    """
    pattern = re.compile(r"%\s*ORG:(\S+)\s+POSITION:(\S+)", re.IGNORECASE)
    
    def sanitize(text: str) -> str:
        return re.sub(r'[\\/*?:"<>|]', "", text)

    for line in (latex or "").splitlines()[:10]:
        match = pattern.search(line)
        if match:
            return sanitize(match.group(1)), sanitize(match.group(2))
    return "Unknown_Org", "Unknown_Position"


# Packages that are XeLaTeX/LuaLaTeX-only and will crash pdflatex.
_FORBIDDEN_PACKAGES = {"fontspec", "xunicode", "xltxtra", "unicode-math", "polyglossia"}


def validate_latex_packages(latex: str) -> bool:
    """
    Scan for packages incompatible with pdflatex.
    Returns True if clean, False if forbidden packages found.
    """
    issues: list[str] = []
    pkg_pattern = re.compile(r"\\usepackage(?:\[.*?\])?\{([^}]+)\}", re.DOTALL)

    for lineno, line in enumerate(latex.splitlines(), start=1):
        for match in pkg_pattern.finditer(line):
            packages = [p.strip() for p in match.group(1).split(",")]
            for pkg in packages:
                if pkg in _FORBIDDEN_PACKAGES:
                    issues.append(
                        f"  Line {lineno}: \\usepackage{{{pkg}}}  <- XeLaTeX-only"
                    )

    if issues:
        print("\n[ERROR] Generated LaTeX contains pdflatex-incompatible packages:")
        for issue in issues:
            print(issue)
        print(
            "\n  -> Re-running should fix this automatically.\n"
            "    The output contract bans these packages.\n"
        )
        return False
    return True


# ------------------------------------------------------------------------------
# Output
# ------------------------------------------------------------------------------
def save_latex(latex: str) -> None:
    """Write output.tex to the project root."""
    OUTPUT_TEX.write_text(latex, encoding="utf-8")
    print(f"[OK] LaTeX saved -> {OUTPUT_TEX}")


def compile_pdf(latex: str, org: str, position: str) -> Path | None:
    """
    Compile output.tex  ->  PDF via pdflatex (if available).
    On success, copies the PDF to outputs/Org_Position.pdf.
    """
    if not shutil.which("pdflatex"):
        print(
            "\n[WARN] pdflatex not found in PATH.\n"
            "       Install TeX Live / MiKTeX or upload output.tex to Overleaf.\n"
        )
        return None

    OUTPUTS_DIR.mkdir(exist_ok=True)
    pdf_name = f"{org}_{position}.pdf"
    dest_pdf = OUTPUTS_DIR / pdf_name

    tmp_dir = BASE_DIR / ".pdfbuild"
    tmp_dir.mkdir(exist_ok=True)
    tmp_tex = tmp_dir / "cover_letter.tex"
    tmp_tex.write_text(latex, encoding="utf-8")

    print("[INFO] Compiling PDF ...")

    try:
        for _pass in range(2):
            result = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
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

        compiled_pdf = tmp_dir / "cover_letter.pdf"
        if not compiled_pdf.exists():
            print("[ERROR] pdflatex exited 0 but cover_letter.pdf was not created.")
            return None

        shutil.copy2(compiled_pdf, dest_pdf)
        print(f"[OK] PDF saved   -> {dest_pdf}")
        return dest_pdf

    except subprocess.TimeoutExpired:
        print("[ERROR] pdflatex timed out after 60 seconds.")
        return None
    except Exception as exc:
        print(f"[ERROR] PDF compilation failed: {exc}")
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _print_latex_error(log_text: str) -> None:
    """Print the first relevant LaTeX error line."""
    print("\n[ERROR] pdflatex compilation failed.")
    for line in log_text.splitlines():
        if line.startswith("!") or "Error" in line:
            print(f"  -> {line}")
    print(
        "\n  The LaTeX source has been saved to output.tex.\n"
        "  You can debug it manually or upload to Overleaf.\n"
    )


def print_usage_summary(usage: dict, config: AIConfig) -> None:
    """Print token usage without hard-coding provider/model pricing."""
    p_tok   = usage["prompt_tokens"]
    c_tok   = usage["completion_tokens"]
    t_tok   = usage["total_tokens"]
    th_tok  = usage.get("thinking_tokens",   0)
    hit_tok = usage.get("cache_hit_tokens",  0)
    mis_tok = usage.get("cache_miss_tokens", 0)

    print("\n" + "-" * 48)
    print("  Provider         : OpenRouter")
    print(f"  Model            : {config.model}")
    print(f"  Prompt tokens    : {p_tok:,}")
    if hit_tok or mis_tok:
        print(f"    |- Cache hit     : {hit_tok:,}")
        _miss_label = "\\- Cache miss"
        print(f"    {_miss_label}    : {mis_tok:,}")
    if th_tok:
        print(f"  Thinking tokens  : {th_tok:,}  (!) reasoning cost (should be 0)")
    print(f"  Completion tokens: {c_tok:,}")
    print(f"  Total tokens     : {t_tok:,}")
    print("  Est. cost        : See OpenRouter usage/pricing for the selected model")
    print("-" * 48 + "\n")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
def main() -> None:
    config            = load_env()
    data_pool         = load_data_pool()
    motivation_data   = load_motivation_data()
    job_description   = collect_job_description()
    seed_sentences    = collect_seed_sentences(motivation_data)

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
            config,
            system_msg,
            user_msg + retry_suffix,
        )
        latex = strip_markdown_fences(raw_output)

        if is_valid_latex(latex):
            break
        if attempt < _MAX_RETRIES:
            print("[WARN] Response does not look like LaTeX  --  will retry.")
        else:
            print("[WARN] Max retries reached. Saving raw output for manual inspection.")

    # -- Validate packages ------------------------------------------------------
    latex_is_clean = validate_latex_packages(latex)

    # -- Save & compile ---------------------------------------------------------
    save_latex(latex)

    org, position = extract_org_position(latex)
    print(f"[INFO] Detected -> Org: {org}  |  Position: {position}")

    if latex_is_clean:
        compile_pdf(latex, org, position)
    else:
        print("[SKIP] PDF compilation skipped due to package errors above.\n")

    print_usage_summary(usage, config)
    print("[OK] Done. Your cover letter is ready.\n")


if __name__ == "__main__":
    main()
