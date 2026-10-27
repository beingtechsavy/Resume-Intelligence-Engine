import unittest

from job_matcher import analyze_job, _detected_skills


PROFILE = {
    "personal_info": {"location": "Bengaluru, India"},
    "candidate_preferences": {"current_location": "Bengaluru, India"},
    "skills": {
        "languages": ["Python"],
        "data": ["PostgreSQL", "SQL CTEs"],
        "ml": ["LLM API integration", "Structured JSON prompting"],
        "engineering": ["Retry logic", "LaTeX output validation"],
        "historical_self_report_only": ["Django"],
    },
    "projects": [
        {
            "id": "support_demo",
            "name": "Support Demo",
            "tech_stack": ["Python", "PostgreSQL", "REST APIs"],
            "features": ["Debugging API errors and writing technical documentation"],
            "resume_ready_bullets": ["Built a small AI workflow with structured prompt tests."],
            "tags": ["ai-integration"],
        }
    ],
    "experience": [],
    "education": {"relevant_coursework": [{"name": "Business Communication"}]},
}


class JobMatcherTests(unittest.TestCase):
    def test_ai_alias_does_not_match_inside_unrelated_words(self):
        self.assertNotIn("ai_llm", _detected_skills("training and onboarding responsibilities"))
        self.assertIn("customer_communication", _detected_skills("customer communication"))

    def test_matches_strong_evidence_and_flags_provisional_evidence(self):
        report = analyze_job(
            "Junior AI Support Specialist. Remote worldwide. "
            "Write prompts, troubleshoot Python/PostgreSQL/API issues, "
            "communicate with customers. AWS is helpful. No prior experience required. Salary USD 700/month.",
            PROFILE,
        )

        statuses = {item["requirement"]: item["status"] for item in report["requirements"]}
        self.assertEqual(statuses["python"], "matched")
        self.assertEqual(statuses["postgresql_sql"], "matched")
        self.assertEqual(statuses["prompt_testing"], "matched")
        self.assertEqual(statuses["aws"], "missing")
        self.assertEqual(report["experience_gate"]["status"], "accessible")
        self.assertEqual(report["eligibility"]["status"], "human_review")
        self.assertIn("ai_support", report["role_families"])
        self.assertEqual(report["compensation_mentions"], [{"text": "USD 700/month"}])

    def test_historical_only_skill_is_partial_when_required(self):
        report = analyze_job("Technical support role requiring Django.", PROFILE)
        django = next(item for item in report["requirements"] if item["requirement"] == "django")
        self.assertEqual(django["status"], "partial")
        self.assertTrue(django["provisional_evidence"])


if __name__ == "__main__":
    unittest.main()
