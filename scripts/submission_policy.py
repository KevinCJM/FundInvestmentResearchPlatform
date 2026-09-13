"""Pure branch and CI evidence rules, shared by the trusted workflows and tests."""
from pathlib import PurePosixPath
import re

REPOSITORY = "KevinCJM/FundInvestmentResearchPlatform"
QUALITY_WORKFLOW = ".github/workflows/quality-gate.yml"
QUALITY_JOBS = {"prepare", "governance", "policy-tests", "protected-policy-tests", "frontend", "backend", "e2e", "quality-result"}
SHA = re.compile(r"[0-9a-f]{40}\Z")


def branch_decision(pr, *, base_is_ancestor, latest_base, latest_source, duplicate_heads=(), latest_main=None, main_is_ancestor=False):
    if pr.get("state") != "open" or pr.get("draft"):
        return "failure", "PR must be open and ready for review"
    if any((pr[side].get("repo") or {}).get("full_name") != REPOSITORY for side in ("head", "base")):
        return "failure", "Only same-repository pull requests are supported"
    base, head = pr["base"], pr["head"]
    if base["ref"] not in {"main", "Dev"}:
        return "failure", "Unsupported target branch"
    if base["ref"] == "main" and head["ref"] != "Dev":
        return "failure", "Only Dev may enter main"
    if base["ref"] == "Dev" and (head["ref"] in {"main", "Dev"} or head["ref"].startswith("release/")):
        return "failure", "Use a development or dedicated synchronization branch for Dev"
    if any(not SHA.fullmatch(value or "") for value in [head["sha"], base["sha"], latest_base, latest_source]):
        return "failure", "Invalid commit identity"
    if base["sha"] != latest_base or head["sha"] != latest_source:
        return "failure", "The source or target ref changed; revalidate the latest commits"
    if not base_is_ancestor:
        return "failure", "Source must include the latest target branch"
    if head["ref"].startswith("codex/sync-main-") and (not SHA.fullmatch(latest_main or "") or not main_is_ancestor):
        return "failure", "Synchronization source must include the current main commit"
    if duplicate_heads:
        return "failure", "Multiple open protected-target PRs share the same HEAD"
    return "success", "Repository, branch direction and current source/target commits verified"


def quality_title(number, head, base):
    return f"quality PR {number} head:{head} base:{base}"


def quality_plan(paths, base_ref):
    """Unknown executable paths expand to the full suite; only documented docs/CI paths are exempt."""
    frontend = backend = False
    for path in paths:
        if path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise ValueError("Invalid repository-relative path")
        if path == ".github/requirements-ci.txt":
            backend = True
            continue
        if (path.endswith(".md") or path.startswith("docs/images/") or
                path in {"docs/repo_map.json", "docs/task_routes.json", "docs/pitfalls.json", "docs/ai_routing_evolution_policy.json"} or
                path.startswith((".github/", "skills/", "scripts/tests/")) or
                path in {"scripts/check_ai_review.py", "scripts/check_submission.py", "scripts/submission_policy.py", "scripts/ci_quality.py", ".gitignore"}):
            continue
        if path.startswith(("frontend/", "locales/")) or path in {"scripts/check_frontend_design.mjs", "scripts/check_i18n.mjs"}:
            frontend = True
        elif path.startswith("backend/"):
            backend = True
        else:
            frontend = backend = True
    # A business release validates the integrated backend and frontend together.
    if base_ref == "main" and (frontend or backend):
        frontend = backend = True
    return {"frontend": frontend, "backend": backend, "e2e": frontend or backend}


def trusted_quality_run(run, pr):
    if ((run.get("repository") or {}).get("full_name") != REPOSITORY
            or run.get("path") != QUALITY_WORKFLOW
            or run.get("display_title") != quality_title(pr["number"], pr["head"]["sha"], pr["base"]["sha"])):
        return False
    if run.get("event") == "workflow_dispatch":
        return run.get("head_branch") == pr["base"]["ref"] and run.get("head_sha") == pr["base"]["sha"]
    if run.get("event") == "pull_request" and run.get("head_sha") == pr["head"]["sha"]:
        return any(p.get("number") == pr["number"] and p.get("head", {}).get("sha") == pr["head"]["sha"]
                   and p.get("base", {}).get("sha") == pr["base"]["sha"] for p in run.get("pull_requests", []))
    return False


def quality_decision(runs, jobs_by_run, pr):
    candidates = [run for run in runs if trusted_quality_run(run, pr) and run.get("workflow_verified") is True]
    if not candidates:
        return "pending", "Waiting for trusted quality workflow on the current HEAD/base", None
    # A newer attempt replaces an earlier success, including while queued or running.
    run = max(candidates, key=lambda item: (item["id"], item.get("run_attempt", 1)))
    if run.get("status") != "completed":
        return "pending", "Current quality validation is running", run
    if run.get("conclusion") != "success":
        return "failure", "Current quality validation did not succeed", run
    jobs = jobs_by_run.get(run["id"], [])
    if len(jobs) != len(QUALITY_JOBS) or {job["name"] for job in jobs} != QUALITY_JOBS:
        return "failure", "Trusted quality job set is incomplete or ambiguous", run
    if any(job.get("status") != "completed" or job.get("run_attempt") != run.get("run_attempt", 1) for job in jobs):
        return "failure", "Quality jobs are not from the completed current attempt", run
    if any(job.get("conclusion") != "success" for job in jobs if job["name"] in {"prepare", "governance", "policy-tests", "protected-policy-tests", "quality-result"}):
        return "failure", "Required quality aggregation or governance did not succeed", run
    # The trusted quality-result job verifies each optional job against the trusted plan.
    if any(job.get("conclusion") not in {"success", "skipped"} for job in jobs):
        return "failure", "A quality child job failed or was cancelled", run
    return "success", "Trusted quality jobs and aggregate succeeded on the current HEAD/base", run
