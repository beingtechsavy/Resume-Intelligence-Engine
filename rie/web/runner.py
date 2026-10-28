"""Generation worker: runs one web-submitted job and archives its artifacts."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from rie.data import (
    load_motivation_data,
    load_profile,
    validate_job_description,
)
from rie.matching import analyze_job
from rie.ats import validate_resume_ats
from rie.web.store import APIError, JobStore, _utc_now

# Keep importing the web app possible when optional SDK dependencies are absent.
try:
    from generate import run_cl, run_cv
except Exception:  # pragma: no cover - exercised only in minimal installations.
    run_cl = None  # type: ignore[assignment]
    run_cv = None  # type: ignore[assignment]

MAX_GENERATED_ARTIFACT_BYTES = 8 * 1024 * 1024
DEFAULT_GENERATION_TIMEOUT_SECONDS = 300.0
_SAFE_RUN_ID = re.compile(r"^run-\d{8}T\d{6}Z-[0-9a-f]{12}$")
_ARTIFACT_NAMES = frozenset({"resume.tex", "cover_letter.tex", "match_report.json", "manifest.json"})
_PLAIN_TEXT_ARTIFACT_NAMES = frozenset({"resume.tex", "cover_letter.tex"})


def safe_error_message(exc: BaseException) -> str:
    if isinstance(exc, APIError):
        return exc.message
    return "Generation failed; check the local profile, motivation data, and provider configuration."


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def generation_timeout() -> float:
    try:
        return max(1.0, float(os.getenv("RIE_GENERATION_TIMEOUT_SECONDS", DEFAULT_GENERATION_TIMEOUT_SECONDS)))
    except ValueError:
        return DEFAULT_GENERATION_TIMEOUT_SECONDS


def provider_configured(env_file: Path) -> bool:
    """Check configuration presence without loading dotenv or revealing values."""

    def usable(value: str) -> bool:
        normalized = value.strip().strip("\"'").casefold()
        return normalized not in {
            "",
            "your_openrouter_api_key_here",
            "your_openrouter_model_here",
            "your_model_here",
        }

    if usable(os.getenv("RIE_API_KEY", "")) and usable(os.getenv("RIE_MODEL", "")):
        return True
    if usable(os.getenv("OPENROUTER_API_KEY", "")) and usable(os.getenv("OPENROUTER_MODEL", "")):
        return True
    try:
        configured: set[str] = set()
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            name = name.removeprefix("export ").strip()
            if name in {"RIE_API_KEY", "RIE_MODEL", "OPENROUTER_API_KEY", "OPENROUTER_MODEL"} and usable(value):
                configured.add(name)
        # Either the neutral pair or the legacy pair must be complete.
        if {"RIE_API_KEY", "RIE_MODEL"}.issubset(configured):
            return True
        return {"OPENROUTER_API_KEY", "OPENROUTER_MODEL"}.issubset(configured)
    except (OSError, UnicodeError):
        return False


def run_artifact_urls(run_id: str, artifacts: Any, runs_dir: Path) -> dict[str, str]:
    """Return URLs only for known, existing artifacts inside one run directory."""
    if not isinstance(artifacts, dict) or not _SAFE_RUN_ID.fullmatch(run_id):
        return {}
    urls: dict[str, str] = {}
    run_dir = (runs_dir / run_id).resolve()
    try:
        run_dir.relative_to(runs_dir)
    except ValueError:
        return {}
    for name, relative_path in artifacts.items():
        if not isinstance(name, str) or relative_path not in _ARTIFACT_NAMES:
            continue
        candidate = (run_dir / relative_path).resolve()
        try:
            candidate.relative_to(run_dir)
        except ValueError:
            continue
        if candidate.is_file() and candidate.stat().st_size <= MAX_GENERATED_ARTIFACT_BYTES:
            urls[name] = f"/api/runs/{run_id}/{relative_path}"
    return urls


def restore_completed_runs(store: JobStore, runs_dir: str | Path) -> int:
    """Hydrate the bounded job list from valid local run manifests."""
    root = Path(runs_dir).resolve()
    if not root.is_dir():
        return 0
    restored = 0
    for run_dir in sorted(root.iterdir(), key=lambda path: path.name, reverse=True):
        if not run_dir.is_dir() or not _SAFE_RUN_ID.fullmatch(run_dir.name):
            continue
        from rie.data import _read_json_file

        manifest = _read_json_file(run_dir / "manifest.json")
        job_id = manifest.get("job_id")
        if manifest.get("kind") != "generate" or not isinstance(job_id, str):
            continue
        artifact_urls = run_artifact_urls(run_dir.name, manifest.get("artifacts"), root)
        if not artifact_urls:
            continue
        match_report = _read_json_file(run_dir / "match_report.json")
        record = {
            "id": job_id,
            "kind": "generate",
            "status": "succeeded",
            "created_at": manifest.get("created_at") or manifest.get("generated_at") or _utc_now(),
            "updated_at": manifest.get("generated_at") or manifest.get("created_at") or _utc_now(),
            "finished_at": manifest.get("generated_at"),
            "include_cover_letter": bool(manifest.get("include_cover_letter", "cover_letter.tex" in artifact_urls)),
            "result": {
                "run_id": run_dir.name,
                "artifacts": {name: name for name in artifact_urls},
                "artifact_urls": artifact_urls,
                "match_report": match_report,
                "manifest": "manifest.json",
            },
        }
        if store.restore(record):
            restored += 1
    return restored


def call_generator(function: Callable[..., Any] | None, *args: Any) -> tuple[str, dict[str, Any], float]:
    if function is None:
        try:
            from generate import run_cl as imported_cl, run_cv as imported_cv

            function = imported_cl if args and len(args) == 4 else imported_cv
        except Exception as exc:
            raise RuntimeError("generation dependencies unavailable") from exc
    result = function(*args)
    if not isinstance(result, tuple) or len(result) < 3:
        raise RuntimeError("generator returned an invalid result")
    latex, usage, elapsed = result[:3]
    if not isinstance(latex, str) or not latex.strip():
        raise RuntimeError("generator returned empty output")
    if len(latex.encode("utf-8")) > MAX_GENERATED_ARTIFACT_BYTES:
        raise RuntimeError("generator output exceeded the artifact limit")
    return latex, usage if isinstance(usage, dict) else {}, float(elapsed)


def run_generation(
    store: JobStore,
    job_id: str,
    profile_path: Path,
    motivation_path: Path,
    runs_dir: Path,
) -> None:
    payload = store.payload(job_id) or {}
    if store.get(job_id) is None:
        return
    store.update(job_id, status="running", started_at=_utc_now())
    started = time.monotonic()
    try:
        profile = load_profile(profile_path)
        if not profile:
            raise RuntimeError("profile unavailable")
        job_description = validate_job_description(payload.get("job_description"))
        include_cover_letter = bool(payload.get("include_cover_letter", True))
        match_report = analyze_job(job_description, profile)

        from rie.llm import cl_model_override, load_ai_config

        env_file = profile_path.parent / ".env"
        try:
            config = load_ai_config(env_file)
        except SystemExit as exc:
            raise RuntimeError("provider configuration unavailable") from exc
        cv_latex, cv_usage, cv_elapsed = call_generator(run_cv, config, profile, job_description)
        if time.monotonic() - started > generation_timeout():
            raise TimeoutError("generation deadline exceeded")

        match_report["ats_validation"] = validate_resume_ats(cv_latex, job_description)

        cl_latex = ""
        cl_usage: dict[str, Any] = {}
        cl_elapsed = 0.0
        if include_cover_letter:
            motivation = load_motivation_data(motivation_path)
            if not motivation:
                raise RuntimeError("motivation data unavailable")
            # The cover-letter slot can run its own model (RIE_CL_MODEL).
            cl_model = cl_model_override(env_file)
            cl_config = config.with_model(cl_model) if cl_model else config
            cl_latex, cl_usage, cl_elapsed = call_generator(
                run_cl, cl_config, profile, motivation, job_description
            )
            if time.monotonic() - started > generation_timeout():
                raise TimeoutError("generation deadline exceeded")

        run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{job_id[4:]}"
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        artifacts: dict[str, str] = {}
        atomic_write(run_dir / "resume.tex", cv_latex)
        artifacts["resume"] = "resume.tex"
        if include_cover_letter:
            atomic_write(run_dir / "cover_letter.tex", cl_latex)
            artifacts["cover_letter"] = "cover_letter.tex"
        atomic_write(run_dir / "match_report.json", json.dumps(match_report, indent=2, ensure_ascii=False) + "\n")
        artifacts["match_report"] = "match_report.json"
        artifacts["manifest"] = "manifest.json"
        existing_record = store.get(job_id)
        artifact_urls = {
            name: f"/api/runs/{run_id}/{relative_path}"
            for name, relative_path in artifacts.items()
        }
        manifest = {
            "schema_version": "resume-engine-run-1",
            "job_id": job_id,
            "kind": "generate",
            "include_cover_letter": include_cover_letter,
            "created_at": existing_record.get("created_at") if existing_record else None,
            "generated_at": _utc_now(),
            "artifacts": artifacts,
            "artifact_urls": artifact_urls,
            "match_report": "match_report.json",
            "usage": {"cv": cv_usage, "cover_letter": cl_usage},
            "elapsed_seconds": {"cv": cv_elapsed, "cover_letter": cl_elapsed},
        }
        atomic_write(run_dir / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        result = {
            "run_id": run_id,
            "artifacts": artifacts,
            "artifact_urls": artifact_urls,
            "match_report": match_report,
            "manifest": "manifest.json",
        }
        store.update(job_id, status="succeeded", finished_at=_utc_now(), result=result)
    except Exception as exc:  # Never let worker exceptions escape or leak provider details.
        store.update(
            job_id,
            status="failed",
            finished_at=_utc_now(),
            error={"code": "generation_timeout" if isinstance(exc, TimeoutError) else "generation_failed", "message": safe_error_message(exc)},
        )
