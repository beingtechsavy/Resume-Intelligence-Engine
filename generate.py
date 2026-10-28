"""
generate.py
-----------
Unified Application Packet Generator — CLI entry point.

One JD, one command: fires CV + Cover Letter generation in parallel and
writes docs/CV_Output.tex + docs/CL_Output.tex, always overwritten.

All pipeline logic lives in ``rie.generate_pipeline``; this wrapper keeps
the historical ``python generate.py`` entry point working.
"""

from __future__ import annotations

from rie.generate_pipeline import (  # noqa: F401 - re-exported pipeline API
    CV_OUTPUT,
    CL_OUTPUT,
    call_api,
    collect_job_description,
    load_data_pool,
    load_env,
    load_motivation_data,
    print_combined_summary,
    run_cl,
    run_cv,
    run_parallel_generation,
)
from rie.ats import validate_resume_ats
from rie.latex import extract_company_role, validate_latex_packages


def main() -> None:
    config = load_env()
    data_pool = load_data_pool()
    motivation_data = load_motivation_data()
    jd = collect_job_description()

    # Ensure docs/ exists.
    from rie.generate_pipeline import DOCS_DIR

    DOCS_DIR.mkdir(exist_ok=True)

    print(f"\n[INFO] Provider: configured endpoint | Model: {config.model} | parallel=True")
    if config.provider_options:
        print(f"[INFO] Provider routing: {config.provider_options}")
    if config.reasoning_options:
        print(f"[INFO] Reasoning policy: {config.reasoning_options}")
    print("[...] Generating CV and Cover Letter simultaneously...\n")

    # Pass this module's runner attributes explicitly so test patches on
    # generate.run_cv / generate.run_cl are honored.
    cv_result, cl_result, wall_time = run_parallel_generation(
        config, data_pool, motivation_data, jd,
        run_cv_fn=run_cv, run_cl_fn=run_cl,
    )

    cv_latex, cv_usage, cv_elapsed = cv_result
    cl_latex, cl_usage, cl_elapsed = cl_result

    # Validate packages (warn but don't block save).
    cv_clean = validate_latex_packages(cv_latex, "CV")
    cl_clean = validate_latex_packages(cl_latex, "CL")
    ats_report = validate_resume_ats(cv_latex, jd)

    # Save -- always overwrite.
    CV_OUTPUT.write_text(cv_latex, encoding="utf-8")
    CL_OUTPUT.write_text(cl_latex, encoding="utf-8")

    company, role = extract_company_role(cv_latex)

    print(f"\n[OK] docs/CV_Output.tex  saved")
    print(f"[OK] docs/CL_Output.tex  saved")
    print(f"[INFO] Detected -> Company: {company}  |  Role: {role}")

    if not cv_clean:
        print("[WARN] CV has forbidden packages -- review output manually.")
    if not cl_clean:
        print("[WARN] CL has forbidden packages -- review output manually.")

    ats_summary = ats_report["summary"]
    print(
        "[ATS] Local CV validation -> "
        f"{ats_report['status']} "
        f"({ats_summary['passed']} pass, {ats_summary['review']} review, {ats_summary['failed']} fail)"
    )

    print_combined_summary(
        cv_usage, cl_usage,
        cv_elapsed, cl_elapsed,
        wall_time, config,
    )
    print("[OK] Application packet ready.")
    print(f"     Upload docs/CV_Output.tex and docs/CL_Output.tex to Overleaf.\n")


if __name__ == "__main__":
    main()
