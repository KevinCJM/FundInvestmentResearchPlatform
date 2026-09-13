#!/usr/bin/env python3
"""Publish three fail-closed PR gates using protected code and a dedicated App."""
import argparse
import json
import os
from urllib.parse import quote

from check_ai_review import GitHub, REPOSITORY, TRUSTED_EVENTS, evaluate, outcome
from submission_policy import QUALITY_WORKFLOW, branch_decision, quality_decision, quality_title, trusted_quality_run


class PolicyChanged(RuntimeError):
    """An obsolete main workflow must stop without overwriting newer decisions."""


def ensure_current_policy(gh, policy_sha):
    if gh.api(f"repos/{REPOSITORY}/git/ref/heads/main")["object"]["sha"] != policy_sha:
        raise PolicyChanged("Publisher is no longer the current protected main commit")


def current_pr(gh, number):
    pr = gh.api(f"repos/{REPOSITORY}/pulls/{number}")
    # PR metadata can lag a base push. Resolve the actual branch ref explicitly.
    latest_base = gh.api(f"repos/{REPOSITORY}/git/ref/heads/{quote(pr['base']['ref'], safe='')}")["object"]["sha"]
    pr["base"]["sha"] = latest_base
    latest_source = None
    if (pr["head"].get("repo") or {}).get("full_name") == REPOSITORY:
        latest_source = gh.api(f"repos/{REPOSITORY}/git/ref/heads/{quote(pr['head']['ref'], safe='')}")["object"]["sha"]
    return pr, latest_base, latest_source


def quality_evidence(gh, pr):
    runs = []
    for sha in {pr["head"]["sha"], pr["base"]["sha"]}:
        runs.extend(gh.pages(f"repos/{REPOSITORY}/actions/workflows/quality-gate.yml/runs?head_sha={sha}", "workflow_runs"))
    candidates = [run for run in runs if trusted_quality_run(run, pr)]
    # Ordinary PR workflows run without privileged credentials. Their definition
    # is accepted only when its Git blob matches the protected base definition.
    # Workflow-changing PRs are instead tested by a dispatch on the protected base.
    expected = gh.api(f"repos/{REPOSITORY}/contents/{QUALITY_WORKFLOW}?ref={pr['base']['sha']}")["sha"]
    verified = []
    for run in sorted(candidates, key=lambda item: item["id"], reverse=True):
        definition = gh.api(f"repos/{REPOSITORY}/contents/{QUALITY_WORKFLOW}?ref={run['head_sha']}")
        if definition.get("type") == "file" and definition["sha"] == expected:
            verified.append({**run, "workflow_verified": True})
            break
    jobs = {}
    if verified:
        run = verified[0]
        jobs[run["id"]] = gh.pages(f"repos/{REPOSITORY}/actions/runs/{run['id']}/attempts/{run.get('run_attempt', 1)}/jobs", "jobs")
    return quality_decision(verified, jobs, pr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if not args.publish:
        parser.error("Use the pure policy tests for local validation; remote publication requires --publish")
    if (os.getenv("GITHUB_ACTIONS") != "true" or os.getenv("GITHUB_REPOSITORY") != REPOSITORY
            or os.getenv("GITHUB_EVENT_NAME") not in TRUSTED_EVENTS | {"workflow_run"}
            or os.getenv("GITHUB_REF") != "refs/heads/main"
            or not os.getenv("AI_REVIEW_CHECKS_TOKEN") or not os.getenv("AI_REVIEW_APP_ID", "").isdigit()):
        parser.error("Protected GitHub workflow and dedicated publisher identity required")
    gh = GitHub()
    policy_sha = os.environ["SUBMISSION_POLICY_SHA"]
    ensure_current_policy(gh, policy_sha)
    errors = 0
    prs = gh.pages(f"repos/{REPOSITORY}/pulls?state=open")
    for listed in prs:
        if listed["base"]["ref"] not in {"main", "Dev"}:
            continue
        context, check_ids = None, {}
        results = {}
        try:
            ensure_current_policy(gh, policy_sha)
            pr, latest_base, latest_source = current_pr(gh, listed["number"])
            context = gh.context(pr, publish=True)
            for name in ("branch-policy", "ai-review", "quality-gate"):
                check_ids[name] = gh.publish(context, outcome("pending", "Revalidating current HEAD/base"), name=name)
            snapshot = gh.collect(pr)
            duplicate = [p["number"] for p in prs if p["number"] != pr["number"]
                         and p["head"]["sha"] == pr["head"]["sha"] and p["base"]["ref"] in {"main", "Dev"}]
            state, reason = branch_decision(pr, base_is_ancestor=snapshot["base_is_ancestor"],
                latest_base=latest_base, latest_source=latest_source, duplicate_heads=duplicate)
            results["branch-policy"] = outcome(state, reason)
            if state != "success":
                results.update({name: outcome("failure", "Branch policy must pass first") for name in ("ai-review", "quality-gate")})
            else:
                results["ai-review"] = evaluate(snapshot, context)
                qstate, qreason, run = quality_evidence(gh, pr)
                results["quality-gate"] = outcome(qstate, qreason, [run["html_url"]] if run else [])
                if run is None:
                    gh.api(f"repos/{REPOSITORY}/actions/workflows/quality-gate.yml/dispatches", {
                        "ref": pr["base"]["ref"], "inputs": {"pr": str(pr["number"]),
                            "head_sha": pr["head"]["sha"], "base_sha": pr["base"]["sha"]}})
            final_pr, final_base, final_source = current_pr(gh, pr["number"])
            if (final_pr["state"] != "open" or final_pr.get("draft")
                    or final_pr["head"]["sha"] != context["head_sha"] or final_source != context["head_sha"]
                    or final_base != context["base_sha"] or final_pr["base"]["ref"] != context["base_ref"]):
                results = {name: outcome("failure", "PR changed during validation; rerun on the latest version") for name in check_ids}
            ensure_current_policy(gh, policy_sha)
            for name, result in results.items():
                gh.publish(context, result, check_ids[name], name=name)
        except PolicyChanged:
            print(json.dumps({"pr": listed["number"], "error": "Protected policy advanced; publication stopped"}))
            return 1
        except Exception as exc:
            # Do not expose HTTP bodies, user-controlled text or credentials in logs.
            errors += 1
            results = {"error": type(exc).__name__}
            # An API failure must not turn a superseded workflow into a writer.
            try:
                ensure_current_policy(gh, policy_sha)
            except Exception:
                return 1
            for name, check_id in check_ids.items():
                try:
                    gh.publish(context, outcome("failure", "Evidence collection failed; inspect the protected workflow"), check_id, name=name)
                except Exception:
                    pass  # An already-created pending check fails closed if publication is unavailable.
        print(json.dumps({"pr": listed["number"], "context": context, "results": results}, ensure_ascii=False))
    return int(errors > 0)


if __name__ == "__main__":
    raise SystemExit(main())
