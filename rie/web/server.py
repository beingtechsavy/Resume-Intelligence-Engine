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
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from rie.data import (
    MAX_REQUEST_BODY_BYTES,
    load_profile,
    parse_json_body,
    profile_summary,
    validate_job_description,
)
from rie.matching import analyze_job
from rie.web.runner import (
    _PLAIN_TEXT_ARTIFACT_NAMES,
    _SAFE_RUN_ID,
    _ARTIFACT_NAMES,
    MAX_GENERATED_ARTIFACT_BYTES,
    provider_configured,
    restore_completed_runs,
    run_generation,
)
from rie.web.store import (
    MAX_JOB_RECORDS,
    APIError,
    JobStore,
    JobStoreFullError,
    _SAFE_ID,
)

_WEB_ROOT = Path(__file__).resolve().parent.parent.parent / "web"
_PROFILE_FILE = _WEB_ROOT.parent / "data_pool.json"
_MOTIVATION_FILE = _WEB_ROOT.parent / "motivation_data.json"
_RUNS_DIR = _WEB_ROOT.parent / "runs"
MAX_WORKERS = 2


class _AppServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], executor: ThreadPoolExecutor):
        super().__init__(address, handler)
        self.executor = executor

    def server_close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        super().server_close()


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
    root = (Path(web_dir) if web_dir is not None else _WEB_ROOT).resolve()
    profile_file = Path(profile_path) if profile_path is not None else _PROFILE_FILE
    motivation_file = Path(motivation_path) if motivation_path is not None else _MOTIVATION_FILE
    runs_root = (Path(runs_dir) if runs_dir is not None else _RUNS_DIR).resolve()
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
                    self._json(200, {"status": "ok", "profile_available": bool(profile), "provider_configured": provider_configured(profile_file.parent / ".env"), "active_jobs": sum(item["status"] in {"queued", "running"} for item in store.list())})
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
                    description = validate_job_description(body.get("job_description"))
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
                description = validate_job_description(body.get("job_description"))
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
                        run_generation,
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
