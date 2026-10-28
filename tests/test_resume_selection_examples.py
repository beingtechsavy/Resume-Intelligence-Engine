"""Selection-engine checks that run against the sanitized example pool.

These keep the public clone green without the personal data_pool.json:
they verify the mechanics (deterministic report shape, prompt contract,
key derivation) on data_pool.example.json rather than real entry keys.
"""

import json
import unittest
from pathlib import Path

from rie.prompts.cv import build_prompt
from rie.selection import build_curated_profile, select_resume_evidence


ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / "data_pool.example.json").read_text(encoding="utf-8"))

_TECH_JD = (
    "Software Engineer building Python and TypeScript web applications "
    "with PostgreSQL and REST APIs."
)


class ExamplePoolSelectionTests(unittest.TestCase):
    def test_selection_report_shape_is_complete(self):
        report = select_resume_evidence(PROFILE, _TECH_JD)
        self.assertEqual(report["schema_version"], "resume-selection-2")
        self.assertEqual(report["page_budget"]["target_pages"], 1)
        for section in ("experience", "projects", "certifications"):
            self.assertIn(section, report["selected"])

    def test_selection_is_deterministic_for_identical_inputs(self):
        first = select_resume_evidence(PROFILE, _TECH_JD)
        second = select_resume_evidence(PROFILE, _TECH_JD)
        self.assertEqual(first, second)

    def test_curated_prompt_never_leaks_raw_catalog_markers(self):
        curated, _ = build_curated_profile(PROFILE, _TECH_JD)
        rendered = json.dumps(curated, ensure_ascii=False)
        self.assertNotIn("source_catalog", rendered)

        _, prompt = build_prompt(PROFILE, _TECH_JD)
        self.assertIn("SELECTED_DATA_POOL:", prompt)
        self.assertIn("RESUME_SELECTION:", prompt)
        self.assertIn("ONE PAGE ONLY", prompt)
        self.assertNotIn("source_catalog", prompt)

    def test_limits_are_enforced_even_for_small_pools(self):
        report = select_resume_evidence(PROFILE, _TECH_JD)
        self.assertLessEqual(len(report["selected"]["experience"]), 2)
        self.assertLessEqual(len(report["selected"]["projects"]), 3)
        self.assertLessEqual(len(report["selected"]["certifications"]), 2)


if __name__ == "__main__":
    unittest.main()
