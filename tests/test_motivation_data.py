import json
import unittest
from pathlib import Path

from rie.prompts.cover_letter import build_cl_prompt


ROOT = Path(__file__).resolve().parents[1]


class MotivationDataTests(unittest.TestCase):
    def test_saved_motivation_data_has_required_voice_sections(self):
        path = ROOT / "motivation_data.json"
        data = json.loads(path.read_text(encoding="utf-8"))

        for key in (
            "identity",
            "origin_stories",
            "seed_sentences",
            "values",
            "what_you_are_not",
            "role_adaptation",
            "tone",
            "personal_story",
        ):
            self.assertIn(key, data)

        self.assertIn("mechanically manage", data["personal_story"])
        self.assertIn("Altum", data["personal_story"])
        self.assertIn("for_project_roles", data["role_adaptation"])

    def test_prompt_includes_user_voice_and_current_location(self):
        pool = json.loads((ROOT / "data_pool.json").read_text(encoding="utf-8"))
        motivation = json.loads(
            (ROOT / "motivation_data.json").read_text(encoding="utf-8")
        )
        _, prompt = build_cl_prompt(
            pool,
            motivation,
            "Project Manager — Web / Software Projects at Altum",
        )

        self.assertIn("MOTIVATION_DATA:", prompt)
        self.assertIn("mechanically manage", prompt)
        self.assertIn("Bengaluru, India", prompt)
        self.assertNotIn("Bhubaneswar, India", prompt)


if __name__ == "__main__":
    unittest.main()
