"""
resume_generator.py
-------------------
AI-Powered Resume Intelligence Engine — CLI entry point.

Thin wrapper: the implementation lives in ``rie.generate_resume``.
Usage is unchanged:  python resume_generator.py
"""

from rie.generate_resume import (  # noqa: F401 - re-exported API
    call_openrouter,
    collect_job_description,
    extract_company_role,
    load_data_pool,
    load_env,
    main,
    print_usage_summary,
)

if __name__ == "__main__":
    main()
