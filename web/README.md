# Resume Intelligence Engine web frontend

This directory contains the dependency-free local frontend for the Resume Intelligence Engine. It has no build step and uses relative API calls so it can be served by the local Python application.

## API routes used

- `GET /api/health`
- `GET /api/profile`
- `POST /api/analyze` with `{ "job_description": "…" }`
- `POST /api/jobs` with `{ "kind": "generate", "job_description": "…", "include_cover_letter": true }`
- `GET /api/jobs`
- `GET /api/jobs/{id}`
- `GET /api/runs/{run_id}/{artifact}` for generated local artifacts

The renderer keeps the UI intentionally tolerant of small response-shape differences while preserving the explainable matcher fields (`requirements`, evidence paths, role families, eligibility and experience review).

Completed runs are rehydrated from valid manifests under `runs/` on server
startup. The backend never rehydrates the original job-description payload into
the browser-facing job list. Resume and cover-letter `.tex` artifacts expose an
inline preview and copy control in the workbench, with a raw-artifact link kept
as a fallback. Raw `.tex` responses are served as `text/plain` so the fallback
opens in the browser rather than downloading the file.
