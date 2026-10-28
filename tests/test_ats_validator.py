import unittest

from rie.ats import validate_resume_ats


GOOD_LATEX = r"""
\documentclass{article}
\usepackage{hyperref}
\begin{document}
Sample Person\\ arpan@example.com
\section{Experience}
\begin{itemize}
\item Built Python and PostgreSQL services with REST APIs.
\end{itemize}
\section{Projects}
\begin{itemize}
\item Developed a TypeScript application.
\end{itemize}
\section{Skills}
Python, PostgreSQL, TypeScript
\end{document}
"""


class ATSValidatorTests(unittest.TestCase):
    def test_passes_text_safe_one_column_resume_and_checks_jd_terms(self):
        report = validate_resume_ats(
            GOOD_LATEX,
            "Software Engineer requiring Python, PostgreSQL and TypeScript.",
        )
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["summary"]["failed"], 0)
        self.assertEqual(report["summary"]["missing_detected_skills"], [])
        budget = next(check for check in report["checks"] if check["id"] == "one_page_budget")
        self.assertEqual(budget["status"], "pass")

    def test_flags_table_and_image_layout(self):
        latex = GOOD_LATEX.replace(
            r"\section{Experience}",
            r"\includegraphics{photo}\begin{tabular}{ll}\section{Experience}",
        )
        report = validate_resume_ats(latex)
        failed = {check["id"] for check in report["checks"] if check["status"] == "fail"}
        self.assertIn("table_layout", failed)
        self.assertIn("image_or_icon", failed)

    def test_reports_missing_jd_skill_for_human_review_not_fabrication(self):
        report = validate_resume_ats(
            GOOD_LATEX,
            "Software Engineer requiring Django and Python.",
        )
        self.assertEqual(report["status"], "review")
        self.assertIn("django", report["summary"]["missing_detected_skills"])

    def test_fails_when_resume_exceeds_one_page_content_budget(self):
        latex = GOOD_LATEX.replace(
            r"\section{Experience}",
            r"\section{Experience}\resumeEntry{A}{B}{C}\resumeEntry{D}{E}{F}\resumeEntry{G}{H}{I}",
        )
        report = validate_resume_ats(latex)
        failed = {check["id"] for check in report["checks"] if check["status"] == "fail"}
        self.assertIn("one_page_budget", failed)


if __name__ == "__main__":
    unittest.main()
