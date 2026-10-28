import json
import unittest
from pathlib import Path

from rie.prompts.cover_letter import build_cl_prompt


ROOT = Path(__file__).resolve().parents[1]

# The real data files are personal and gitignored; the .example copies ship
# in the repo. Tests run against whichever exists, so the public clone stays
# green while a local checkout can still be exercised against real data.
POOL_PATH = ROOT / "data_pool.json"
MOTIVATION_PATH = ROOT / "motivation_data.json"
POOL_EXAMPLE_PATH = ROOT / "data_pool.example.json"
MOTIVATION_EXAMPLE_PATH = ROOT / "motivation_data.example.json"

_USE_REAL = POOL_PATH.exists() and MOTIVATION_PATH.exists()
_POOL_SRC = POOL_PATH if _USE_REAL else POOL_EXAMPLE_PATH
_MOTIVATION_SRC = MOTIVATION_PATH if _USE_REAL else MOTIVATION_EXAMPLE_PATH


class MotivationDataTests(unittest.TestCase):
    def test_data_files_exist(self):
        self.assertTrue(
            _POOL_SRC.exists(),
            f"Neither {POOL_PATH.name} (local) nor {POOL_EXAMPLE_PATH.name} found.",
        )
        self.assertTrue(
            _MOTIVATION_SRC.exists(),
            f"Neither {MOTIVATION_PATH.name} (local) nor "
            f"{MOTIVATION_EXAMPLE_PATH.name} found.",
        )

    def test_saved_motivation_data_has_required_voice_sections(self):
        data = _motivation_data()
        for key in ("identity", "origin_stories", "seed_sentences", "tone"):
            self.assertIn(key, data)

    @unittest.skipUnless(
        _USE_REAL, "voice-content assertions require the private motivation_data.json"
    )
    def test_saved_motivation_data_sections_have_real_voice_content(self):
        data = json.loads(MOTIVATION_PATH.read_text(encoding="utf-8"))
        self.assertIn("mechanically manage", data["personal_story"])
        self.assertIn("Altum", data["personal_story"])
        self.assertIn("for_project_roles", data["role_adaptation"])

    def test_prompt_includes_motivation_data_and_pool_identity(self):
        prompt = _built_prompt()
        self.assertIn("MOTIVATION_DATA:", prompt)

    @unittest.skipUnless(
        _USE_REAL, "voice/location assertions require the private data files"
    )
    def test_prompt_includes_user_voice_and_current_location(self):
        _, prompt = build_cl_prompt(
            _pool(),
            _motivation_data(),
            "Project Manager — Web / Software Projects at Altum",
        )
        self.assertIn("mechanically manage", prompt)
        self.assertIn("Bengaluru, India", prompt)
        self.assertNotIn("Bhubaneswar, India", prompt)


def _pool():
    return json.loads(_POOL_SRC.read_text(encoding="utf-8"))


def _motivation_data():
    return json.loads(_MOTIVATION_SRC.read_text(encoding="utf-8"))


def _built_prompt():
    _, user_message = build_cl_prompt(_pool(), _motivation_data(), "Sample role")
    return user_message


if __name__ == "__main__":
    unittest.main()
