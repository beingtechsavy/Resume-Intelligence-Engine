"""LaTeX post-processing helpers — the single source shared by all generators.

Every CLI orchestrator previously maintained its own copy of these helpers
(``generate.py``, ``resume_generator.py``, ``Cover Letter Maker/``).  They now
live here once: markdown-fence stripping, LaTeX validity checks, company/role
extraction from the header comment, the forbidden-packages contract, and
optional pdflatex compilation.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

# Packages that only compile under XeLaTeX/LuaLaTeX and crash pdflatex.
# Keep in sync with the "BANNED packages" lines in the prompt output contracts
# (rie.prompts.cv / rie.prompts.cover_letter).
FORBIDDEN_PACKAGES = {"fontspec", "xunicode", "xltxtra", "unicode-math", "polyglossia"}


def strip_markdown_fences(text: str) -> str:
    """Remove accidental ```latex / ``` wrappers the model may emit.

    Uses re.search (not fullmatch) so it also works when the closing fence
    is missing due to a truncated response.
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
    r"""Returns True if the text looks like compilable LaTeX.

    Skips comment lines (%) and checks whether the first meaningful line
    starts with a backslash (e.g. a header comment, then \documentclass).
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue  # skip blank lines and LaTeX comments
        return stripped.startswith("\\")
    return False


def _sanitize_component(text: str) -> str:
    """Strip filesystem-hostile characters from a filename component."""
    return re.sub(r'[\\/*?:"<>|]', "", text)


def extract_header_pair(
    latex: str,
    *,
    first_label: str,
    second_label: str,
    default: tuple[str, str],
) -> tuple[str, str]:
    """Read a ``% KEY:<value> KEY2:<value>`` header comment.

    Scans the first 10 lines so a leading blank line does not hide it.
    """
    pattern = re.compile(
        rf"%\s*{first_label}:(\S+)\s+{second_label}:(\S+)", re.IGNORECASE
    )
    for line in (latex or "").splitlines()[:10]:
        match = pattern.search(line)
        if match:
            return _sanitize_component(match.group(1)), _sanitize_component(match.group(2))
    return default


def extract_company_role(latex: str) -> tuple[str, str]:
    """Read the ``% COMPANY:<name> ROLE:<title>`` header of a generated CV."""
    return extract_header_pair(
        latex,
        first_label="COMPANY",
        second_label="ROLE",
        default=("Unknown_Company", "Unknown_Role"),
    )


def extract_org_position(latex: str) -> tuple[str, str]:
    """Read the ``% ORG:<name> POSITION:<title>`` header of a generated CL."""
    return extract_header_pair(
        latex,
        first_label="ORG",
        second_label="POSITION",
        default=("Unknown_Org", "Unknown_Position"),
    )


def find_forbidden_packages(latex: str) -> list[tuple[int, str]]:
    """Scan for XeLaTeX-only packages that crash pdflatex.

    Returns ``(line_number, package)`` pairs in document order.
    """
    pkg_pattern = re.compile(r"\\usepackage(?:\[.*?\])?\{([^}]+)\}", re.DOTALL)
    issues: list[tuple[int, str]] = []
    for lineno, line in enumerate(latex.splitlines(), start=1):
        for match in pkg_pattern.finditer(line):
            # A single \usepackage can load multiple comma-separated packages.
            for pkg in (p.strip() for p in match.group(1).split(",")):
                if pkg in FORBIDDEN_PACKAGES:
                    issues.append((lineno, pkg))
    return issues


def validate_latex_packages(latex: str, label: str) -> bool:
    """Report forbidden packages for ``label``. Returns True if clean."""
    issues = find_forbidden_packages(latex)
    if not issues:
        return True
    print(f"\n[ERROR] {label} contains pdflatex-incompatible packages:")
    for lineno, pkg in issues:
        print(f"  Line {lineno}: \\usepackage{{{pkg}}}  <- XeLaTeX-only, crashes pdflatex")
    print(
        "\n  -> Re-running should fix this automatically."
        "\n     The output contract bans these packages.\n"
    )
    return False


def print_latex_error(log_text: str, source_hint: str) -> None:
    """Print the first relevant LaTeX error lines from a pdflatex log."""
    print("\n[ERROR] pdflatex compilation failed.")
    for line in log_text.splitlines():
        if line.startswith("!") or "Error" in line:
            print(f"  -> {line}")
    print(f"\n  The LaTeX source has been saved to {source_hint}."
          "\n  You can debug it manually or upload to Overleaf.\n")


def compile_pdf(
    latex: str,
    pdf_name: str,
    output_dir: Path,
    project_dir: Path,
    build_name: str = "resume.tex",
) -> Path | None:
    """Compile ``latex`` to ``output_dir/pdf_name`` via pdflatex.

    Runs in an isolated ``.pdfbuild`` temp directory inside ``project_dir`` so
    pdflatex aux files never clutter the repo.  Two passes for cross-references.
    ``build_name`` is the internal .tex filename used during compilation.
    Returns the PDF path on success, None on failure or if pdflatex is absent.
    """
    if not shutil.which("pdflatex"):
        print(
            "\n[WARN] pdflatex not found in PATH."
            "\n       Install a LaTeX distribution (TeX Live / MiKTeX) to enable"
            "\n       auto PDF compilation, or upload the .tex file to Overleaf.\n"
        )
        return None

    if not pdf_name.endswith(".pdf"):
        pdf_name += ".pdf"
    output_dir.mkdir(exist_ok=True)
    dest_pdf = output_dir / pdf_name

    tmp_dir = project_dir / ".pdfbuild"
    tmp_dir.mkdir(exist_ok=True)
    tmp_tex = tmp_dir / build_name
    tmp_tex.write_text(latex, encoding="utf-8")

    print("[INFO] Compiling PDF …")

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
                print_latex_error(
                    result.stdout + result.stderr,
                    source_hint=tmp_tex.name,
                )
                return None

        compiled_pdf = tmp_dir / build_name.replace(".tex", ".pdf")
        if not compiled_pdf.exists():
            print(f"[ERROR] pdflatex exited 0 but {compiled_pdf.name} was not created.")
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
        shutil.rmtree(tmp_dir, ignore_errors=True)
