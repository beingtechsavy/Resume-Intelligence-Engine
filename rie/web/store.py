"""Bounded thread-safe job registry for the local web API."""

from __future__ import annotations

import re
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

MAX_JOB_RECORDS = 20


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


# Matches the ``job-<12 hex>`` identifiers produced by JobStore.create.
_SAFE_ID = re.compile(r"^job-[0-9a-f]{12}$")
