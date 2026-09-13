#!/usr/bin/env python3
"""Publish three fail-closed PR gates using protected code and the built-in Actions token."""
import argparse
import json
import os
from urllib.parse import quote

from check_ai_review import ACTIONS_APP_ID, GitHub, REPOSITORY, TRUSTED_EVENTS, evaluate, outcome
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


def verify_submission(gh, number, expected_head, expected_base):
    """Read-only terminal verification using current-main code and real evidence."""
    import subprocess
    from pathlib import Path

    policy_sha = gh.api(f"repos/{REPOSITORY}/git/ref/heads/main")["object"]["sha"]
    # The documented launcher materializes an isolated git archive of main.
    # Reject a modified/candidate validator even when it claims the same SHA.
    for name in ("check_ai_review.py", "check_submission.py", "submission_policy.py"):
        path = Path(__file__).with_name(name)
        blob = subprocess.run(["git", "hash-object", str(path)], capture_output=True, text=True, check=True).stdout.strip()
        remote = gh.api(f"repos/{REPOSITORY}/contents/scripts/{name}?ref={policy_sha}")
        if remote.get("type") != "file" or remote["sha"] != blob:
            raise ValueError("Run the unmodified verifier from current origin/main")
    pr, base, source = current_pr(gh, number)
    if pr["head"]["sha"] != expected_head or base != expected_base:
        raise ValueError("HEAD/base changed; refresh and verify again")
    record = gh.published_records(policy_sha).get(number, {})
    context = record.get("context", {})
    if any(context.get(k) != v for k, v in {"head_sha": source, "base_sha": base,
            "base_ref": pr["base"]["ref"], "policy_sha": policy_sha, "pr_number": number}.items()):
        raise ValueError("No authentic publisher record for the current version")
    names = ("branch-policy", "ai-review", "quality-gate")
    if any(record.get("results", {}).get(name, {}).get("state") != "success" for name in names):
        raise ValueError("Protected publisher has not passed all three gates")
    snapshot = gh.collect(pr)
    state, reason = branch_decision(pr, base_is_ancestor=snapshot["base_is_ancestor"],
        latest_base=base, latest_source=source, duplicate_heads=snapshot["other_prs_with_same_head"],
        latest_main=snapshot.get("latest_main"), main_is_ancestor=snapshot.get("main_is_ancestor", False))
    ai = evaluate(snapshot, context)
    quality, qreason, _ = quality_evidence(gh, pr)
    if state != "success" or ai["state"] != "success" or quality != "success":
        raise ValueError(f"Fresh evidence failed: {reason}; {ai['reason']}; {qreason}")
    checks = gh.pages(f"repos/{REPOSITORY}/commits/{source}/check-runs?filter=all", "check_runs")
    for name in names:
        matching = [check for check in checks if check["name"] == name and check.get("app", {}).get("id") == ACTIONS_APP_ID]
        latest = max(matching, key=lambda check: check["id"], default={})
        if latest.get("status") != "completed" or latest.get("conclusion") != "success":
            raise ValueError(f"Required check is not successful: {name}")
    final, final_base, final_source = current_pr(gh, number)
    ensure_current_policy(gh, policy_sha)
    if (final["state"] != "open" or final.get("draft") or final["base"]["ref"] != pr["base"]["ref"]
            or final["head"]["sha"] != source or final_base != base or final_source != source):
        raise ValueError("PR changed during verification")
    return {"pr": number, "head_sha": source, "base_sha": base, "policy_sha": policy_sha, "state": "success"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--publish", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--pr", type=int)
    parser.add_argument("--expected-head")
    parser.add_argument("--expected-base")
    args = parser.parse_args()
    if args.verify:
        if not args.pr or not args.expected_head or not args.expected_base:
            parser.error("--verify requires --pr, --expected-head and --expected-base")
        try:
            print(json.dumps(verify_submission(GitHub(), args.pr, args.expected_head, args.expected_base)))
            return 0
        except Exception as exc:
            print(json.dumps({"pr": args.pr, "state": "failure", "error": type(exc).__name__}))
            return 1
    if (os.getenv("GITHUB_ACTIONS") != "true" or os.getenv("GITHUB_REPOSITORY") != REPOSITORY
            or os.getenv("GITHUB_EVENT_NAME") not in TRUSTED_EVENTS | {"workflow_run"}
            or os.getenv("GITHUB_REF") != "refs/heads/main"
            or not os.getenv("GH_TOKEN")):
        parser.error("Protected GitHub workflow and built-in token required")
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
                latest_base=latest_base, latest_source=latest_source, duplicate_heads=duplicate, latest_main=snapshot.get("latest_main"),
                main_is_ancestor=snapshot.get("main_is_ancestor", False))
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
