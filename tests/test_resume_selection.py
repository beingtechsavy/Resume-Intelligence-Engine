import json
import unittest
from pathlib import Path

from rie.prompts.cv import build_prompt
from rie.selection import build_curated_profile, select_resume_evidence


ROOT = Path(__file__).resolve().parents[1]

# The real data_pool.json is personal and gitignored. These tests assert on
# real entry keys (neuronest, cogent_podcast_internship, ...) from the
# resume_selection_index, so they run only when the private pool exists;
# public clones skip them, and test_resume_selection_examples.py covers the
# sanitized example pool instead.
_POOL_PATH = ROOT / "data_pool.json"
_PROFILE = json.loads(_POOL_PATH.read_text(encoding="utf-8")) if _POOL_PATH.exists() else None


@unittest.skipUnless(_PROFILE is not None, "requires the private, gitignored data_pool.json")
class ResumeSelectionTests(unittest.TestCase):
    def test_technical_jd_suppresses_podcast_internship(self):
        report = select_resume_evidence(
            _PROFILE,
            "Junior AI Support Specialist. Troubleshoot Python and PostgreSQL API issues, "
            "analyze transcripts, test prompts and communicate with customers.",
        )

        selected_experience = {item["key"] for item in report["selected"]["experience"]}
        suppressed_experience = {
            item["key"]: item["reason"]
            for item in report["suppressed"]
            if item["section"] == "experience"
        }
        self.assertIn("iiser_research_internship", selected_experience)
        self.assertNotIn("cogent_podcast_internship", selected_experience)
        self.assertIn("cogent_podcast_internship", suppressed_experience)
        self.assertIn("user's resume inclusion policy", suppressed_experience["cogent_podcast_internship"])

    def test_omitted_podcast_internship_stays_out_even_for_content_jds(self):
        report = select_resume_evidence(
            _PROFILE,
            "Podcast content producer internship supporting episode research, guest coordination, "
            "social media promotion and analytics reporting.",
        )
        selected_experience = {item["key"] for item in report["selected"]["experience"]}
        self.assertNotIn("cogent_podcast_internship", selected_experience)
        podcast = next(
            item for item in report["suppressed"]
            if item["key"] == "cogent_podcast_internship"
        )
        self.assertEqual(podcast["user_priority"], "exclude")

    def test_specific_jd_skills_prevent_generic_android_project_leakage(self):
        report = select_resume_evidence(
            _PROFILE,
            "Software Engineer building Python and TypeScript web applications with "
            "PostgreSQL and REST APIs.",
        )
        selected_projects = {item["key"] for item in report["selected"]["projects"]}
        self.assertIn("neuronest", selected_projects)
        self.assertIn("resume_engine", selected_projects)
        self.assertLessEqual(len(selected_projects), 3)
        self.assertNotIn("bawarchi", selected_projects)

    def test_curated_prompt_does_not_send_raw_pool_or_podcast(self):
        curated, selection = build_curated_profile(
            _PROFILE,
            "Software Engineer building Python and TypeScript web applications with PostgreSQL.",
        )
        rendered = json.dumps(curated, ensure_ascii=False)
        self.assertNotIn("Podcast Intern", rendered)
        self.assertNotIn("source_catalog", rendered)
        self.assertEqual(selection["schema_version"], "resume-selection-2")

        _, prompt = build_prompt(
            _PROFILE,
            "Software Engineer building Python and TypeScript web applications with PostgreSQL.",
        )
        self.assertIn("SELECTED_DATA_POOL:", prompt)
        self.assertIn("RESUME_SELECTION:", prompt)
        self.assertIn("ONE PAGE ONLY", prompt)
        self.assertNotIn("Podcast Intern", prompt)
        self.assertNotIn("source_catalog", prompt)

    def test_owner_priorities_control_project_and_experience_selection(self):
        report = select_resume_evidence(
            _PROFILE,
            "Junior AI Support Specialist. Troubleshoot Python and PostgreSQL API issues, "
            "review prompts and communicate with customers.",
        )
        selected_projects = {item["key"]: item for item in report["selected"]["projects"]}
        selected_experience = {item["key"]: item for item in report["selected"]["experience"]}

        for key in ("neuronest", "resume_engine", "learnmate"):
            self.assertIn(key, selected_projects)
            self.assertEqual(selected_projects[key]["user_priority"], "primary")
        self.assertLessEqual(len(selected_projects), 3)
        self.assertIn("iiser_research_internship", selected_experience)
        self.assertIn("mlsa", selected_experience)
        self.assertLessEqual(len(selected_experience), 2)

        suppressed = {
            item["key"]: item
            for item in report["suppressed"]
            if item["section"] == "experience"
        }
        self.assertEqual(suppressed["cogent_podcast_internship"]["user_priority"], "exclude")

    def test_keyword_dependent_projects_require_a_direct_job_signal(self):
        report = select_resume_evidence(_PROFILE, "AI Support Specialist with prompt testing and customer communication.")
        suppressed = {
            item["key"]: item
            for item in report["suppressed"]
            if item["section"] == "projects"
        }
        self.assertIn("finola", suppressed)
        self.assertEqual(suppressed["finola"]["user_priority"], "keyword_dependent")
        self.assertIn("keyword-dependent", suppressed["finola"]["reason"])

    def test_one_page_budget_is_exposed_and_limits_selected_evidence(self):
        report = select_resume_evidence(
            _PROFILE,
            "Software Engineer building Python and TypeScript web applications with PostgreSQL.",
        )
        self.assertEqual(report["page_budget"]["target_pages"], 1)
        self.assertLessEqual(len(report["selected"]["experience"]), 2)
        self.assertLessEqual(len(report["selected"]["projects"]), 3)
        self.assertLessEqual(len(report["selected"]["certifications"]), 2)


if __name__ == "__main__":
    unittest.main()
