"""Local HTTP API for the Resume Intelligence Engine.

The command-line generators remain the source of truth for document generation;
this module adds a small, dependency-light local web backend around them.  It
does not load provider configuration or make network calls until a generation
job is explicitly submitted.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlsplit

try:  # Keep importing the web app possible when optional SDK dependencies are absent.
    from generate import run_cl, run_cv
except Exception:  # pragma: no cover - exercised only in minimal installations.
    run_cl = None  # type: ignore[assignment]
    run_cv = None  # type: ignore[assignment]

from job_matcher import analyze_job
from ats_validator import validate_resume_ats


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
PROFILE_FILE = BASE_DIR / "data_pool.json"
MOTIVATION_FILE = BASE_DIR / "motivation_data.json"
RUNS_DIR = BASE_DIR / "runs"

MAX_REQUEST_BODY_BYTES = 1_048_576
MAX_JOB_DESCRIPTION_CHARS = 50_000
MAX_JOB_RECORDS = 20
MAX_WORKERS = 2
MAX_GENERATED_ARTIFACT_BYTES = 8 * 1024 * 1024
DEFAULT_GENERATION_TIMEOUT_SECONDS = 300.0
_SAFE_ID = re.compile(r"^job-[0-9a-f]{12}$")
_SAFE_RUN_ID = re.compile(r"^run-\d{8}T\d{6}Z-[0-9a-f]{12}$")
_ARTIFACT_NAMES = frozenset({"resume.tex", "cover_letter.tex", "match_report.json", "manifest.json"})
_PLAIN_TEXT_ARTIFACT_NAMES = frozenset({"resume.tex", "cover_letter.tex"})


class APIError(Exception):
    """An expected request/application error with a safe public message."""

    def __init__(self, status: int, code: str, message: str, details: Any = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details


class JobStoreFullError(RuntimeError):
    """Raised when every retained job is still active."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json_file(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty object for missing/bad local data."""
    try:
        if not path.is_file():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def load_profile(path: str | Path | None = None) -> dict[str, Any]:
    """Load the local evidence profile without raising for missing/bad data."""
    return _read_json_file(Path(path) if path is not None else PROFILE_FILE)


def _load_motivation_data(path: str | Path | None = None) -> dict[str, Any]:
    return _read_json_file(Path(path) if path is not None else MOTIVATION_FILE)


def profile_summary(profile: dict[str, Any] | None) -> dict[str, Any]:
    """Return a deliberately small, non-secret profile view for the browser."""
    if not isinstance(profile, dict) or not profile:
        return {"available": False, "name": None, "location": None, "target_roles": [], "skill_categories": {}}

    personal_value = profile.get("personal_info")
    positioning_value = profile.get("positioning")
    preferences_value = profile.get("candidate_preferences")
    skills_value = profile.get("skills")
    personal: dict[str, Any] = personal_value if isinstance(personal_value, dict) else {}
    positioning: dict[str, Any] = positioning_value if isinstance(positioning_value, dict) else {}
    preferences: dict[str, Any] = preferences_value if isinstance(preferences_value, dict) else {}
    skills: dict[str, Any] = skills_value if isinstance(skills_value, dict) else {}
    safe_skills: dict[str, list[str]] = {}
    for category, values in skills.items():
        if isinstance(values, list):
            safe_skills[str(category)] = [str(value) for value in values[:30] if isinstance(value, (str, int, float))]

    education_value = profile.get("education")
    education: dict[str, Any] = education_value if isinstance(education_value, dict) else {}
    return {
        "available": True,
        "name": personal.get("name") or profile.get("profile_owner"),
        "location": preferences.get("current_location") or personal.get("location"),
        "positioning": positioning.get("core_identity"),
        "current_level": positioning.get("current_level"),
        "target_roles": list(positioning.get("target_roles", []))[:20] if isinstance(positioning.get("target_roles"), list) else [],
        "skill_categories": safe_skills,
        "education": {
            "degree": education.get("degree"),
            "university": education.get("university"),
            "year": education.get("year"),
        },
    }


def parse_json_body(
    body: bytes | bytearray | str,
    content_type: str | None = "application/json",
    max_bytes: int = MAX_REQUEST_BODY_BYTES,
) -> dict[str, Any]:
    """Strictly parse one JSON object with an application/json content type."""
    if not content_type or content_type.split(";", 1)[0].strip().lower() != "application/json":
        raise APIError(415, "unsupported_media_type", "Content-Type must be application/json.")
    raw = body.encode("utf-8") if isinstance(body, str) else bytes(body)
    if len(raw) > max_bytes:
        raise APIError(413, "request_too_large", "Request body exceeds the maximum allowed size.")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise APIError(400, "invalid_json", "Request body must contain valid UTF-8 JSON.") from None
    if not isinstance(value, dict):
        raise APIError(400, "invalid_json_shape", "Request JSON must be an object.")
    return value


def _validate_job_description(value: Any) -> str:
    if not isinstance(value, str):
        raise APIError(400, "invalid_job_description", "job_description must be a string.")
    description = value.strip()
    if not description:
        raise APIError(400, "empty_job_description", "job_description must not be empty.")
    if len(description) > MAX_JOB_DESCRIPTION_CHARS:
        raise APIError(413, "job_description_too_large", "job_description exceeds the maximum allowed length.")
    return description


class JobStore:
    """Thread-safe bounded store for public job state and private job inputs."""

    STATES = frozenset({"queued", "running", "succeeded", "failed"})

    def __init__(self, max_records: int = MAX_JOB_RECORDS):
        self.max_records = max(1, int(max_records))
        self._records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._payloads: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def create(self, kind: str = "generate", payload: dict[str, Any] | None = None, **metadata: Any) -> dict[str, Any]:
        now = _utc_now()
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        record = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            **{key: value for key, value in metadata.items() if key not in {"error", "result"}},
        }
        with self._lock:
            if len(self._records) >= self.max_records and not any(
                record.get("status") in {"succeeded", "failed"}
                for record in self._records.values()
            ):
                raise JobStoreFullError("all retained jobs are still active")
            self._records[job_id] = record
            self._payloads[job_id] = dict(payload or {})
            self._trim_locked()
            return dict(record)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(job_id)
            return dict(record) if record is not None else None

    def payload(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._payloads.get(job_id)
            return dict(value) if value is not None else None

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in reversed(self._records.values())]

    def restore(self, record: dict[str, Any]) -> bool:
        """Restore one completed local run without restoring its private input."""
        job_id = record.get("id")
        status = record.get("status")
        if not isinstance(job_id, str) or not _SAFE_ID.fullmatch(job_id):
            return False
        if status not in {"succeeded", "failed"}:
            return False
        with self._lock:
            if job_id in self._records:
                return False
            self._records[job_id] = dict(record)
            self._payloads[job_id] = {}
            self._trim_locked()
            return job_id in self._records

    def update(self, job_id: str, status: str | None = None, **fields: Any) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return None
            if status is not None:
                if status not in self.STATES:
                    raise ValueError(f"unsupported job state: {status}")
                record["status"] = status
            record.update(fields)
            record["updated_at"] = _utc_now()
            return dict(record)

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        """Mark a not-yet-successful job failed without exposing a new state."""
        return self.update(job_id, status="failed", error={"code": "cancelled", "message": "Job cancelled."})

    def _trim_locked(self) -> None:
        while len(self._records) > self.max_records:
            removable = next(
                (job_id for job_id, record in self._records.items() if record.get("status") in {"succeeded", "failed"}),
                next(iter(self._records)),
            )
            self._records.pop(removable, None)
            self._payloads.pop(removable, None)


class _AppServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], executor: ThreadPoolExecutor):
        super().__init__(address, handler)
        self.executor = executor

    def server_close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        super().server_close()


def _safe_error_message(exc: BaseException) -> str:
    if isinstance(exc, APIError):
        return exc.message
    return "Generation failed; check the local profile, motivation data, and provider configuration."


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _generation_timeout() -> float:
    try:
        return max(1.0, float(os.getenv("RIE_GENERATION_TIMEOUT_SECONDS", DEFAULT_GENERATION_TIMEOUT_SECONDS)))
    except ValueError:
        return DEFAULT_GENERATION_TIMEOUT_SECONDS


def _provider_configured(env_file: Path) -> bool:
    """Check configuration presence without loading dotenv or revealing values."""
    def usable(value: str) -> bool:
        normalized = value.strip().strip("\"'").casefold()
        return normalized not in {
            "",
            "your_openrouter_api_key_here",
            "your_openrouter_model_here",
            "your_model_here",
        }

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
            if name in {"OPENROUTER_API_KEY", "OPENROUTER_MODEL"} and usable(value):
                configured.add(name)
        return {"OPENROUTER_API_KEY", "OPENROUTER_MODEL"}.issubset(configured)
    except (OSError, UnicodeError):
        return False


def _run_artifact_urls(run_id: str, artifacts: Any, runs_dir: Path) -> dict[str, str]:
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
        manifest = _read_json_file(run_dir / "manifest.json")
        job_id = manifest.get("job_id")
        if manifest.get("kind") != "generate" or not isinstance(job_id, str):
            continue
        artifact_urls = _run_artifact_urls(run_dir.name, manifest.get("artifacts"), root)
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


def _call_generator(function: Callable[..., Any] | None, *args: Any) -> tuple[str, dict[str, Any], float]:
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


def _run_generation(
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
        job_description = _validate_job_description(payload.get("job_description"))
        include_cover_letter = bool(payload.get("include_cover_letter", True))
        match_report = analyze_job(job_description, profile)

        from ai_config import load_ai_config

        try:
            config = load_ai_config(profile_path.parent / ".env")
        except SystemExit as exc:
            raise RuntimeError("provider configuration unavailable") from exc
        cv_latex, cv_usage, cv_elapsed = _call_generator(run_cv, config, profile, job_description)
        if time.monotonic() - started > _generation_timeout():
            raise TimeoutError("generation deadline exceeded")

        match_report["ats_validation"] = validate_resume_ats(cv_latex, job_description)

        cl_latex = ""
        cl_usage: dict[str, Any] = {}
        cl_elapsed = 0.0
        if include_cover_letter:
            motivation = _load_motivation_data(motivation_path)
            if not motivation:
                raise RuntimeError("motivation data unavailable")
            cl_latex, cl_usage, cl_elapsed = _call_generator(run_cl, config, profile, motivation, job_description)
            if time.monotonic() - started > _generation_timeout():
                raise TimeoutError("generation deadline exceeded")

        run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{job_id[4:]}"
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        artifacts: dict[str, str] = {}
        _atomic_write(run_dir / "resume.tex", cv_latex)
        artifacts["resume"] = "resume.tex"
        if include_cover_letter:
            _atomic_write(run_dir / "cover_letter.tex", cl_latex)
            artifacts["cover_letter"] = "cover_letter.tex"
        _atomic_write(run_dir / "match_report.json", json.dumps(match_report, indent=2, ensure_ascii=False) + "\n")
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
        _atomic_write(run_dir / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
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
            error={"code": "generation_timeout" if isinstance(exc, TimeoutError) else "generation_failed", "message": _safe_error_message(exc)},
        )


def build_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    web_dir: str | Path | None = None,
    profile_path: str | Path | None = None,
    motivation_path: str | Path | None = None,
    runs_dir: str | Path | None = None,
    max_jobs: int = MAX_JOB_RECORDS,
    max_workers: int = MAX_WORKERS,
) -> ThreadingHTTPServer:
    """Build, but do not start, the local API server."""
    root = (Path(web_dir) if web_dir is not None else WEB_DIR).resolve()
    profile_file = Path(profile_path) if profile_path is not None else PROFILE_FILE
    motivation_file = Path(motivation_path) if motivation_path is not None else MOTIVATION_FILE
    runs_root = (Path(runs_dir) if runs_dir is not None else RUNS_DIR).resolve()
    store = JobStore(max_jobs)
    restore_completed_runs(store, runs_root)
    executor = ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), MAX_WORKERS)), thread_name_prefix="resume-job")

    class Handler(BaseHTTPRequestHandler):
        server_version = "ResumeIntelligenceEngine/1"
        protocol_version = "HTTP/1.1"

        @property
        def app_server(self) -> _AppServer:
            return self.server  # type: ignore[return-value]

        def _json(self, status: int, payload: Any) -> None:
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(encoded)
            self.close_connection = True

        def _error(self, error: APIError | None = None, status: int = 500, code: str = "internal_error", message: str = "Internal server error.") -> None:
            if error is not None:
                status, code, message = error.status, error.code, error.message
                payload: dict[str, Any] = {"error": {"code": code, "message": message}}
                if error.details is not None:
                    payload["error"]["details"] = error.details
            else:
                payload = {"error": {"code": code, "message": message}}
            self._json(status, payload)

        def _read_body(self) -> bytes:
            header = self.headers.get("Content-Length")
            if header is None:
                raise APIError(411, "length_required", "Content-Length is required.")
            try:
                length = int(header)
            except ValueError:
                raise APIError(400, "invalid_content_length", "Content-Length must be an integer.") from None
            if length < 0:
                raise APIError(400, "invalid_content_length", "Content-Length must not be negative.")
            if length > MAX_REQUEST_BODY_BYTES:
                raise APIError(413, "request_too_large", "Request body exceeds the maximum allowed size.")
            return self.rfile.read(length)

        def _request_json(self) -> dict[str, Any]:
            return parse_json_body(self._read_body(), self.headers.get("Content-Type"))

        def _static(self, request_path: str) -> None:
            relative = unquote(request_path.split("?", 1)[0]).lstrip("/") or "index.html"
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                raise APIError(404, "not_found", "Resource not found.") from None
            if not candidate.is_file():
                raise APIError(404, "not_found", "Resource not found.")
            data = candidate.read_bytes()
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
            self.close_connection = True

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            try:
                path = urlsplit(self.path).path
                if path == "/api/health":
                    profile = load_profile(profile_file)
                    self._json(200, {"status": "ok", "profile_available": bool(profile), "provider_configured": _provider_configured(profile_file.parent / ".env"), "active_jobs": sum(item["status"] in {"queued", "running"} for item in store.list())})
                elif path == "/api/profile":
                    self._json(200, profile_summary(load_profile(profile_file)))
                elif path == "/api/jobs":
                    self._json(200, {"jobs": store.list()})
                elif path.startswith("/api/jobs/"):
                    job_id = path.rsplit("/", 1)[-1]
                    if not _SAFE_ID.fullmatch(job_id):
                        raise APIError(404, "not_found", "Job not found.")
                    job = store.get(job_id)
                    if job is None:
                        raise APIError(404, "not_found", "Job not found.")
                    self._json(200, job)
                elif path.startswith("/api/runs/"):
                    parts = [unquote(part) for part in path.split("/") if part]
                    if len(parts) != 4 or parts[0:2] != ["api", "runs"]:
                        raise APIError(404, "not_found", "Artifact not found.")
                    run_id, artifact_name = parts[2], parts[3]
                    if not _SAFE_RUN_ID.fullmatch(run_id) or artifact_name not in _ARTIFACT_NAMES:
                        raise APIError(404, "not_found", "Artifact not found.")
                    candidate = (runs_root / run_id / artifact_name).resolve()
                    try:
                        candidate.relative_to(runs_root)
                    except ValueError:
                        raise APIError(404, "not_found", "Artifact not found.") from None
                    if not candidate.is_file():
                        raise APIError(404, "not_found", "Artifact not found.")
                    data = candidate.read_bytes()
                    if len(data) > MAX_GENERATED_ARTIFACT_BYTES:
                        raise APIError(413, "artifact_too_large", "Artifact exceeds the maximum allowed size.")
                    content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
                    # Browsers commonly download application/x-tex even when the
                    # response is marked inline. Serve generated LaTeX as plain
                    # text so "Open raw" opens a readable browser tab instead.
                    if artifact_name in _PLAIN_TEXT_ARTIFACT_NAMES:
                        content_type = "text/plain"
                    if content_type.startswith("text/") or content_type == "application/json":
                        content_type += "; charset=utf-8"
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Disposition", f'inline; filename="{artifact_name}"')
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(data)
                    self.close_connection = True
                else:
                    self._static(path)
            except APIError as exc:
                self._error(exc)
            except (OSError, ValueError):
                self._error(status=404, code="not_found", message="Resource not found.")
            except Exception:
                self._error()

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            try:
                path = urlsplit(self.path).path
                if path not in {"/api/analyze", "/api/jobs"}:
                    raise APIError(404, "not_found", "Endpoint not found.")
                body = self._request_json()
                if path == "/api/analyze":
                    unknown = set(body) - {"job_description"}
                    if unknown:
                        raise APIError(400, "unknown_fields", "Unexpected request fields.", sorted(unknown))
                    description = _validate_job_description(body.get("job_description"))
                    report = analyze_job(description, load_profile(profile_file))
                    # Keep the canonical nested shape while also exposing the
                    # report fields at the top level for compatibility with
                    # older locally cached frontend bundles.
                    self._json(
                        200,
                        {
                            "job_description_length": len(description),
                            "match_report": report,
                            **report,
                        },
                    )
                    return

                unknown = set(body) - {"kind", "job_description", "include_cover_letter"}
                if unknown:
                    raise APIError(400, "unknown_fields", "Unexpected request fields.", sorted(unknown))
                if body.get("kind") != "generate":
                    raise APIError(400, "invalid_kind", "kind must be 'generate'.")
                description = _validate_job_description(body.get("job_description"))
                include = body.get("include_cover_letter", True)
                if not isinstance(include, bool):
                    raise APIError(400, "invalid_include_cover_letter", "include_cover_letter must be boolean.")
                try:
                    record = store.create(
                        "generate",
                        {"job_description": description, "include_cover_letter": include},
                        include_cover_letter=include,
                    )
                except JobStoreFullError:
                    raise APIError(429, "job_queue_full", "The local generation queue is full; wait for an active job to finish.") from None
                try:
                    self.app_server.executor.submit(
                        _run_generation,
                        store,
                        record["id"],
                        profile_file,
                        motivation_file,
                        runs_root,
                    )
                except Exception:
                    store.update(record["id"], status="failed", error={"code": "queue_failed", "message": "Job could not be queued."})
                    raise APIError(503, "queue_failed", "Job could not be queued.") from None
                self._json(202, record)
            except APIError as exc:
                self._error(exc)
            except Exception:
                self._error()

        def log_message(self, format: str, *args: Any) -> None:
            # Avoid default request-body/header logging; paths and status remain enough for local debugging.
            return

    server = _AppServer((host, int(port)), Handler, executor)
    server.job_store = store  # type: ignore[attr-defined]
    server.profile_path = profile_file  # type: ignore[attr-defined]
    server.web_dir = root  # type: ignore[attr-defined]
    server.runs_dir = runs_root  # type: ignore[attr-defined]
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the local Resume Intelligence Engine web API.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="TCP port (default: 8000).")
    args = parser.parse_args(argv)
    server = build_server(args.host, args.port)
    print(f"Resume Intelligence Engine listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
