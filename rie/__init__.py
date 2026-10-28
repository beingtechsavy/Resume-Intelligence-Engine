"""Resume Intelligence Engine — provider-agnostic local resume toolkit.

Package layout:

    rie.data               profile/motivation loading + redacted web summary
    rie.matching           job-description -> evidence matching
    rie.selection          deterministic evidence selection / curated profile
    rie.ats                local ATS-oriented LaTeX validator
    rie.llm                provider configuration (OpenAI-compatible endpoints)
    rie.latex              LaTeX post-processing helpers (single source)
    rie.prompts.cv         recruiter-first resume prompt (PCB + output contract)
    rie.prompts.cover_letter  cover letter prompt (PCB + output contract)
    rie.generate_pipeline  unified CV + cover-letter generation pipeline
    rie.generate_resume    standalone CV generator (streaming CLI)
    rie.generate_cover_letter  standalone cover-letter generator (streaming CLI)
    rie.web.server         local HTTP API + static web workbench
    rie.web.store          bounded thread-safe job registry
    rie.web.runner         generation worker + run archiving

The command-line entry points (``generate.py``, ``resume_generator.py``,
``app.py``) remain thin wrappers at the repository root for compatibility.
"""

__version__ = "1.0.0"
