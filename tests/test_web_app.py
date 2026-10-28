import json
import tempfile
import threading
import unittest
from pathlib import Path
from typing import cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from rie.data import (
    load_profile,
    parse_json_body,
    profile_summary,
)
from rie.web.runner import restore_completed_runs
from rie.web.server import build_server
from rie.web.store import (
    APIError,
    JobStore,
    JobStoreFullError,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "data_pool.json"
WEB_ROOT = ROOT / "web"


class WebAppHelperTests(unittest.TestCase):
    def test_profile_load_and_summary_are_minimal_and_redacted(self):
        profile = load_profile(PROFILE_PATH)
        summary = profile_summary(profile)

        self.assertIsInstance(profile, dict)
        self.assertIsInstance(summary, dict)
        rendered = json.dumps(summary, ensure_ascii=False).casefold()
        for sensitive in (
            "arpanpadhi2004@gmail.com",
            "linkedin.com/in/",
            "github.com/beingtechsavy",
            "social_links",
            "source_ids",
            "claim_policy",
        ):
            self.assertNotIn(sensitive.casefold(), rendered)

        # The summary should retain useful positioning context without exposing
        # the full evidence pool.
        self.assertTrue(
            any(key in summary for key in ("name", "profile_owner", "candidate"))
        )
        self.assertTrue(
            any(key in summary for key in ("location", "current_location"))
        )
        self.assertLess(len(rendered), 8_000)

    def test_parse_json_body_accepts_json_and_rejects_bad_or_oversized_input(self):
        self.assertEqual(parse_json_body(b'{"job_description":"Python role"}'), {
            "job_description": "Python role"
        })

        for payload in (b"{not-json", b"[]", b'"a string"'):
            with self.subTest(payload=payload):
                with self.assertRaises(APIError):
                    parse_json_body(payload)

        with self.assertRaises(APIError) as context:
            parse_json_body(b'{"job_description":"' + (b"x" * 100) + b'"}', max_bytes=32)
        self.assertEqual(context.exception.status, 413)


class JobStoreTests(unittest.TestCase):
    def test_job_transitions_are_recorded_and_retention_is_bounded(self):
        store = JobStore(max_records=2)

        first = store.create(payload={"job_description": "first job"})
        second = store.create(payload={"job_description": "second job"})
        self.assertEqual(first["status"], "queued")
        self.assertEqual(second["status"], "queued")

        updated = store.update(first["id"], status="running")
        self.assertIsNotNone(updated)
        updated = cast(dict, updated)
        self.assertEqual(updated["status"], "running")
        completed = store.update(
            first["id"],
            status="succeeded",
            result={"schema_version": "job-match-report-1"},
        )
        self.assertIsNotNone(completed)
        completed = cast(dict, completed)
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(completed["result"]["schema_version"], "job-match-report-1")
        stored = store.get(first["id"])
        self.assertIsNotNone(stored)
        stored = cast(dict, stored)
        self.assertEqual(stored["status"], "succeeded")

        third = store.create(payload={"job_description": "third job"})
        self.assertEqual(len(store.list()), 2)
        self.assertIsNone(store.get(first["id"]))
        self.assertEqual(
            {job["id"] for job in store.list()},
            {second["id"], third["id"]},
        )

    def test_unknown_job_update_is_rejected(self):
        store = JobStore(max_records=2)
        self.assertIsNone(store.update("missing", status="succeeded"))

    def test_restore_only_accepts_completed_records(self):
        store = JobStore(max_records=2)
        self.assertTrue(store.restore({"id": "job-aaaaaaaaaaaa", "status": "succeeded"}))
        self.assertFalse(store.restore({"id": "job-bbbbbbbbbbbb", "status": "running"}))
        self.assertFalse(store.restore({"id": "not-a-job", "status": "succeeded"}))
        restored = store.get("job-aaaaaaaaaaaa")
        self.assertIsNotNone(restored)
        self.assertEqual(cast(dict, restored)["status"], "succeeded")

    def test_active_job_capacity_is_not_silently_evicted(self):
        store = JobStore(max_records=1)
        store.create(payload={"job_description": "active"})
        with self.assertRaises(JobStoreFullError):
            store.create(payload={"job_description": "would overflow"})


class StaticAssetTests(unittest.TestCase):
    def test_frontend_assets_exist_and_expose_accessible_landmarks(self):
        index = WEB_ROOT / "index.html"
        styles = WEB_ROOT / "styles.css"
        script = WEB_ROOT / "app.js"
        for asset in (index, styles, script):
            self.assertTrue(asset.is_file(), asset)
            self.assertGreater(asset.stat().st_size, 0, asset)

        html = index.read_text(encoding="utf-8").casefold()
        css = styles.read_text(encoding="utf-8").casefold()
        js = script.read_text(encoding="utf-8").casefold()
        self.assertIn("<main", html)
        self.assertIn("<header", html)
        self.assertRegex(html, r"aria-(label|live)")
        self.assertIn("@media", css)
        self.assertIn("/api/analyze", js)
        self.assertIn("fetch", js)
        self.assertIn("localstorage", js)
        self.assertIn("extractmatchreport", js)
        self.assertIn("preview code", js)
        self.assertIn("copy code", js)
        self.assertIn("createartifactcard", js)
        self.assertIn("code-preview", css)


class WebAppHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs_tempdir = tempfile.TemporaryDirectory()
        run_dir = Path(cls.runs_tempdir.name) / "run-20261024T000000Z-000000000000"
        run_dir.mkdir(parents=True)
        (run_dir / "resume.tex").write_text("% test resume\n", encoding="utf-8")
        (run_dir / "cover_letter.tex").write_text("% test cover letter\n", encoding="utf-8")
        cls.server = build_server(
            "127.0.0.1",
            0,
            profile_path=PROFILE_PATH,
            runs_dir=cls.runs_tempdir.name,
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.server.server_address[:2]
        cls.base_url = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.runs_tempdir.cleanup()

    @classmethod
    def request_json(cls, method, path, payload=None):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            cls.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    @classmethod
    def request_error_json(cls, method, path, raw_body):
        request = Request(
            cls.base_url + path,
            data=raw_body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            urlopen(request, timeout=5)
        except HTTPError as response:
            return response.code, json.loads(response.read().decode("utf-8"))
        raise AssertionError("request unexpectedly succeeded")

    def test_health_profile_and_analyze_routes(self):
        status, health = self.request_json("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["status"], "ok")

        status, profile = self.request_json("GET", "/api/profile")
        self.assertEqual(status, 200)
        rendered_profile = json.dumps(profile, ensure_ascii=False).casefold()
        self.assertNotIn("arpanpadhi2004@gmail.com", rendered_profile)
        self.assertNotIn("social_links", rendered_profile)

        payload = {
            "job_description": (
                "Junior AI Support Specialist. Remote worldwide. "
                "Troubleshoot Python and PostgreSQL API issues, write prompts, "
                "and communicate with customers."
            )
        }
        status, report = self.request_json("POST", "/api/analyze", payload)
        self.assertEqual(status, 200)
        self.assertEqual(report["match_report"]["schema_version"], "job-match-report-1")
        self.assertIn("requirements", report["match_report"])
        self.assertIn("resume_selection", report["match_report"])
        selected_experience = {
            item["key"]
            for item in report["match_report"]["resume_selection"]["selected"]["experience"]
        }
        self.assertNotIn("cogent_podcast_internship", selected_experience)
        self.assertIn("eligibility", report["match_report"])
        self.assertEqual(report["summary"], report["match_report"]["summary"])

        _, second_report = self.request_json("POST", "/api/analyze", payload)
        self.assertEqual(report, second_report)

        status, jobs = self.request_json("GET", "/api/jobs")
        self.assertEqual(status, 200)
        self.assertEqual(jobs, {"jobs": []})

    def test_static_path_cannot_escape_web_root(self):
        request = Request(self.base_url + "/../data_pool.json", method="GET")
        with self.assertRaises(HTTPError) as context:
            urlopen(request, timeout=5)
        self.assertEqual(context.exception.code, 404)

    def test_missing_artifact_returns_json_error(self):
        request = Request(
            self.base_url + "/api/runs/run-20261024T000000Z-111111111111/resume.tex",
            method="GET",
        )
        with self.assertRaises(HTTPError) as context:
            urlopen(request, timeout=5)
        self.assertEqual(context.exception.code, 404)
        body = json.loads(context.exception.read().decode("utf-8"))
        self.assertEqual(body["error"]["code"], "not_found")

    def test_safe_artifact_route_serves_versioned_local_output(self):
        expected = {
            "resume.tex": b"% test resume\n",
            "cover_letter.tex": b"% test cover letter\n",
        }
        for artifact_name, expected_body in expected.items():
            with self.subTest(artifact_name=artifact_name):
                request = Request(
                    self.base_url + f"/api/runs/run-20261024T000000Z-000000000000/{artifact_name}",
                    method="GET",
                )
                with urlopen(request, timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.headers.get("Content-Type"), "text/plain; charset=utf-8")
                    self.assertEqual(
                        response.headers.get("Content-Disposition"),
                        f'inline; filename="{artifact_name}"',
                    )
                    self.assertEqual(response.read().replace(b"\r\n", b"\n"), expected_body)

    def test_completed_run_manifests_are_restored_on_server_start(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_id = "run-20261024T000000Z-222222222222"
            run_dir = root / run_id
            run_dir.mkdir()
            (run_dir / "resume.tex").write_text("% restored resume\n", encoding="utf-8")
            (run_dir / "match_report.json").write_text(
                json.dumps({"schema_version": "job-match-report-1"}), encoding="utf-8"
            )
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "resume-engine-run-1",
                        "job_id": "job-222222222222",
                        "kind": "generate",
                        "include_cover_letter": False,
                        "created_at": "2026-10-24T00:00:00Z",
                        "generated_at": "2026-10-24T00:01:00Z",
                        "artifacts": {
                            "resume": "resume.tex",
                            "match_report": "match_report.json",
                            "manifest": "manifest.json",
                        },
                    }
                ),
                encoding="utf-8",
            )

            store = JobStore(max_records=5)
            restored = restore_completed_runs(store, root)

        self.assertEqual(restored, 1)
        record = store.get("job-222222222222")
        self.assertIsNotNone(record)
        restored_record = cast(dict, record)
        self.assertEqual(restored_record["status"], "succeeded")
        self.assertIn(
            "/api/runs/run-20261024T000000Z-222222222222/resume.tex",
            restored_record["result"]["artifact_urls"].values(),
        )

    def test_malformed_request_and_unknown_route_return_json_errors(self):
        status, error = self.request_error_json("POST", "/api/analyze", b"{bad")
        self.assertEqual(status, 400)
        self.assertIsInstance(error["error"]["message"], str)

        request = Request(self.base_url + "/api/does-not-exist", method="GET")
        with self.assertRaises(HTTPError) as context:
            urlopen(request, timeout=5)
        self.assertEqual(context.exception.code, 404)
        body = json.loads(context.exception.read().decode("utf-8"))
        self.assertIsInstance(body["error"]["message"], str)


if __name__ == "__main__":
    unittest.main()
