import threading
import unittest
from typing import Any, cast
from unittest.mock import patch

import generate


class ParallelGenerationTests(unittest.TestCase):
    def test_resume_and_cover_letter_tasks_start_concurrently(self):
        barrier = threading.Barrier(2, timeout=2)

        def fake_cv(config, profile, jd):
            barrier.wait()
            return "\\documentclass{article}", {"total_tokens": 1}, 0.01

        def fake_cl(config, profile, motivation, jd):
            barrier.wait()
            return "\\documentclass{article}", {"total_tokens": 1}, 0.01

        with patch.object(generate, "run_cv", side_effect=fake_cv), patch.object(
            generate, "run_cl", side_effect=fake_cl
        ):
            cv_result, cl_result, wall_time = generate.run_parallel_generation(
                cast(Any, object()), {}, {}, "Software Engineer"
            )

        self.assertEqual(cv_result[0], "\\documentclass{article}")
        self.assertEqual(cl_result[0], "\\documentclass{article}")
        self.assertLess(wall_time, 2)


if __name__ == "__main__":
    unittest.main()
