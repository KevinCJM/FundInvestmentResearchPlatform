#!/usr/bin/env python3
"""Turn authenticated Codex review evidence into a check on the actual PR HEAD.

Only GitHub metadata is read. No code from the candidate PR is executed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import time
from urllib.parse import quote

REPOSITORY = "KevinCJM/FundInvestmentResearchPlatform"
BOT_ID = 199175422
BOT_LOGIN = "chatgpt-codex-connector[bot]"
CODEX_APP_ID = 1144995
WORKFLOW = ".github/workflows/ai-review.yml"
REQUEST_MARKER = "<!-- codex-review-request/v1 "
SCHEMA = "codex-review/v1"
TRUSTED_EVENTS = {"pull_request_target", "issue_comment", "push", "schedule", "workflow_dispatch", "workflow_run"}
NATIVE_CLEAN = re.compile(
    r"\ACodex Review: Didn't find any major issues\. You're on a roll\.\s*"
    r"\*\*Reviewed commit:\*\* `([0-9a-f]{7,40})`\s*"
    r"(?:<details>\s*<summary>ℹ️ About Codex in GitHub</summary>.*?</details>\s*)?\Z", re.S)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def timestamp(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def is_bot(item):
    user = item.get("user") or {}
    return user.get("id") == BOT_ID and user.get("login") == BOT_LOGIN and user.get("type") == "Bot"


def is_app_comment(item):
    return is_bot(item) and (item.get("performed_via_github_app") or {}).get("id") == CODEX_APP_ID


def request_body(pr):
    pair = {"repository": REPOSITORY, "pr_number": pr["number"],
            "head_sha": pr["head"]["sha"], "base_sha": pr["base"]["sha"], "base_ref": pr["base"]["ref"]}
    return ("@codex review\n\nPlease review the complete PR diff against branch_submission_rules.md, "
            "including P0/P1/P2 findings. Review exactly the HEAD/base recorded below; "
            "if either changes, report INCOMPLETE. Do not modify code.\n\n"
            + REQUEST_MARKER + json.dumps(pair, sort_keys=True) + " -->")


def bound_request(comment, pr, observed_at):
    # A later edit could otherwise turn a thumbs-up on an old request into a new verdict.
    return (comment.get("body") == request_body(pr)
            and comment.get("created_at") == comment.get("updated_at")
            and timestamp(comment["created_at"]) >= timestamp(observed_at))


def outcome(state, reason, evidence=None):
    return {"state": state, "reason": reason, "evidence": evidence or []}


def evaluate(snapshot, context):
    """Pure, fail-closed decision; context must predate the reviewed evidence."""
    pr = snapshot["pr"]
    if pr["state"] != "open" or pr.get("draft"):
        return outcome("failure", "PR is closed or still a draft")
    if any((pr[side].get("repo") or {}).get("full_name") != REPOSITORY for side in ("head", "base")):
        return outcome("failure", "Only same-repository PRs are supported")
    if snapshot["other_prs_with_same_head"]:
        return outcome("failure", "Multiple open PRs share this HEAD; keep only one before reviewing")
    head, base = pr["head"]["sha"], pr["base"]["sha"]
    if (context["head_sha"], context["base_sha"], context["base_ref"]) != (head, base, pr["base"]["ref"]):
        return outcome("failure", "HEAD or base changed; a new review is required")
    if pr["base"]["ref"] not in {"main", "Dev"}:
        return outcome("failure", "Unsupported target branch")
    if pr["base"]["ref"] == "main" and pr["head"]["ref"] != "Dev":
        return outcome("failure", "Only Dev may enter main")
    if pr["base"]["ref"] == "Dev" and pr["head"]["ref"] in {"Dev", "main"}:
        return outcome("failure", "Use a development or dedicated synchronization branch")
    if not snapshot["base_is_ancestor"]:
        return outcome("failure", "Update the source branch to include the latest target branch")
    started = timestamp(context["observed_at"])
    # Resolving a conversation is not a substitute for a subsequent independent review.
    findings = snapshot["findings"]
    unresolved = [f for f in findings if not f["resolved"]]
    if unresolved:
        return outcome("failure", "Unresolved Codex findings", [f["url"] for f in unresolved])
    latest_finding = max([timestamp(f["updated_at"]) for f in findings] + [started])
    records = [r for r in snapshot["reviews"] if is_bot(r) and r.get("commit_id") == head
               and r.get("state") not in {"DISMISSED", "PENDING"}]
    latest_review = max(records, key=lambda r: r["submitted_at"]) if records else None
    changes_requested = latest_review and latest_review.get("state") == "CHANGES_REQUESTED"
    comments = [c for c in snapshot["comments"] if is_app_comment(c)]
    requests = [c for c in snapshot["comments"] if c.get("body") == request_body(pr)
                and timestamp(c["created_at"]) >= started]
    request = max(requests, key=lambda c: (c["created_at"], c["id"])) if requests else None
    requested_at = timestamp(request["created_at"]) if request else started
    # A structured verdict can carry exact HEAD/base and explicit limitations.
    reports = []
    for record in records + comments:
        body = record.get("body", "")
        blocks = re.findall(r"```json\s*\n(.*?)\n```", body, re.S)
        for block in blocks:
            try:
                report = json.loads(block)
            except (ValueError, TypeError):
                continue
            if not isinstance(report, dict) or report.get("schema") != SCHEMA:
                continue
            if any(report.get(k) != v for k, v in {
                "repository": REPOSITORY, "pr_number": pr["number"], "head_sha": head,
                "base_sha": base, "base_ref": pr["base"]["ref"],
            }.items()):
                continue
            # A rewritten issue comment must not retroactively attest a new base.
            if "submitted_at" not in record and record.get("created_at") != record.get("updated_at"):
                continue
            at = timestamp(record.get("submitted_at") or record["created_at"])
            if at < max(started, requested_at):
                continue
            # A PASS quoted in an explanation is not a verdict. Exactly one
            # top-level report is accepted; ambiguous reports fail closed.
            if len(blocks) != 1 or not re.fullmatch(r"\s*```json\s*\n.*?\n```\s*", body, re.S):
                report = {"conclusion": "INCOMPLETE", "limitations": ["Ambiguous or quoted structured report"]}
            reports.append((at, report, record["html_url"], record.get("state")))
    if reports:
        at, report, url, review_state = max(reports, key=lambda r: (r[0], r[1].get("conclusion") != "PASS"))
        if (report.get("conclusion") == "PASS" and report.get("findings") == [] and report.get("limitations") == []
                and review_state != "CHANGES_REQUESTED" and at >= latest_finding
                and not any(timestamp(r["submitted_at"]) > at or
                            (r.get("state") == "CHANGES_REQUESTED" and timestamp(r["submitted_at"]) == at)
                            for r in records)):
            return outcome("success", "Current Codex structured review passed", [url])
        return outcome("failure", "Codex report is blocked, incomplete, or contains findings/limitations", [url])
    review_pending = "failure" if changes_requested else "pending"
    if not requests:
        return outcome(review_pending, "Waiting for an exact HEAD/base review request")
    if not bound_request(request, pr, context["observed_at"]):
        return outcome(review_pending, "Latest version-bound request was edited; submit a fresh request")
    # Native no-findings review: require ALL of current authenticated summary,
    # uniquely resolved commit, fresh official thumbs-up and no unresolved finding.
    summaries = [c for c in comments if "<!-- codex-pull-request-review-summary -->" in c.get("body", "")]
    if not summaries:
        return outcome(review_pending, "Waiting for a current Codex review summary")
    summary = max(summaries, key=lambda c: c["updated_at"])
    row = next((line for line in summary["body"].splitlines()
                if line.startswith("|") and "**Code Review**" in line), "")
    if not re.search(r"\*\*Completed\*\*", row):
        return outcome(review_pending, "Codex review has not completed")
    commit = re.search(r"`([0-9a-f]{7,40})`", row)
    completed = re.search(r'datetime="([^\"]+)"', row)
    if not commit or snapshot["resolved_commits"].get(commit[1]) != head:
        return outcome(review_pending, "Review summary does not identify the current full HEAD")
    if not completed or timestamp(completed[1]) < max(latest_finding, requested_at) or timestamp(summary["updated_at"]) < started:
        return outcome(review_pending, "Review predates this version or the latest finding")
    positives = [r for r in snapshot["reactions"] if is_bot(r) and r.get("content") == "+1"
                 and r.get("request_comment_id") == request["id"]
                 and timestamp(r["created_at"]) >= max(latest_finding, requested_at)]
    clean_comments = []
    for comment in comments:
        match = NATIVE_CLEAN.fullmatch(comment.get("body", ""))
        if (match and comment.get("created_at") == comment.get("updated_at")
                and snapshot["resolved_commits"].get(match[1]) == head
                and timestamp(comment["created_at"]) >= max(latest_finding, requested_at)):
            clean_comments.append(comment)
    positives += clean_comments
    if not positives:
        return outcome(review_pending, "Waiting for an authenticated Codex no-findings verdict after the latest request")
    # A later review with suggestions must never be overridden by an old reaction.
    newest_positive = max(timestamp(r["created_at"]) for r in positives)
    if any(timestamp(r["submitted_at"]) > newest_positive or
           (r.get("state") == "CHANGES_REQUESTED" and timestamp(r["submitted_at"]) == newest_positive)
           for r in records):
        return outcome(review_pending, "A newer Codex review requires a fresh no-findings verdict")
    return outcome("success", "Current Codex summary and fresh no-findings verdict verified",
                   [summary["html_url"], request.get("html_url", "")] + [c["html_url"] for c in clean_comments])


class GitHub:
    def api(self, path, body=None, method="POST", publisher=False):
        args = ["gh", "api", path]
        if body is not None:
            args += ["--method", method, "--input", "-"]
        env = dict(os.environ)
        if publisher:
            env["GH_TOKEN"] = os.environ["AI_REVIEW_CHECKS_TOKEN"]
        result = subprocess.run(args, env=env, input=json.dumps(body) if body is not None else None,
                                text=True, capture_output=True, timeout=45)
        if result.returncode:
            # Do not echo arbitrary API bodies, user text, or credentials into Actions logs.
            raise RuntimeError("GitHub API request failed: " + path.split("?")[0])
        return json.loads(result.stdout) if result.stdout.strip() else None

    def pages(self, path, key=None):
        items = []
        for page in range(1, 101):
            data = self.api(path + ("&" if "?" in path else "?") + f"per_page=100&page={page}")
            batch = data[key] if key else data
            items.extend(batch)
            if len(batch) < 100:
                return items
        raise RuntimeError("Pagination limit reached; refusing incomplete evidence")

    def findings(self, number):
        query = '''query($owner:String!,$name:String!,$number:Int!,$cursor:String){
          repository(owner:$owner,name:$name){pullRequest(number:$number){reviewThreads(first:100,after:$cursor){
            pageInfo{hasNextPage endCursor} nodes{isResolved comments(first:100){
              pageInfo{hasNextPage} nodes{author{login} databaseId url updatedAt}}}}}}}'''
        cursor, findings = None, []
        for _ in range(100):
            data = self.api("graphql", {"query": query, "variables": {"owner": "KevinCJM", "name": "FundInvestmentResearchPlatform", "number": number, "cursor": cursor}})
            if data.get("errors"):
                raise RuntimeError("Incomplete review-thread query")
            threads = data["data"]["repository"]["pullRequest"]["reviewThreads"]
            for thread in threads["nodes"]:
                if thread["comments"]["pageInfo"]["hasNextPage"]:
                    raise RuntimeError("Truncated review thread; manual investigation required")
                for comment in thread["comments"]["nodes"]:
                    if (comment.get("author") or {}).get("login") == "chatgpt-codex-connector":
                        # Resolve the REST identity; a matching display name alone is insufficient.
                        detail = self.api(f"repos/{REPOSITORY}/pulls/comments/{comment['databaseId']}")
                        if is_bot(detail):
                            findings.append({"resolved": thread["isResolved"], "url": detail["html_url"], "updated_at": detail["updated_at"]})
            if not threads["pageInfo"]["hasNextPage"]:
                return findings
            cursor = threads["pageInfo"]["endCursor"]
        raise RuntimeError("Review-thread pagination limit reached")

    def collect(self, pr):
        prefix = f"repos/{REPOSITORY}"
        number, head, base = pr["number"], pr["head"]["sha"], pr["base"]["sha"]
        comments = self.pages(f"{prefix}/issues/{number}/comments")
        reactions = []
        # Manual review requests may receive the positive reaction instead of the PR.
        for comment in comments:
            if comment.get("body") == request_body(pr):
                for reaction in self.pages(f"{prefix}/issues/comments/{comment['id']}/reactions"):
                    reactions.append({**reaction, "request_comment_id": comment["id"]})
        resolved = {}
        for comment in comments:
            if is_app_comment(comment) and ("<!-- codex-pull-request-review-summary -->" in comment.get("body", "")
                                            or NATIVE_CLEAN.fullmatch(comment.get("body", ""))):
                for short in re.findall(r"`([0-9a-f]{7,40})`", comment["body"]):
                    resolved[short] = self.api(f"{prefix}/commits/{short}")["sha"]
        comparison = self.api(f"{prefix}/compare/{base}...{head}")
        shared_head = [p["number"] for p in self.pages(f"{prefix}/pulls?state=open")
                       if p["number"] != number and p["head"]["sha"] == head
                       and p["base"]["ref"] in {"main", "Dev"}]
        return {"pr": pr, "comments": comments, "reactions": reactions, "resolved_commits": resolved,
                "reviews": self.pages(f"{prefix}/pulls/{number}/reviews"), "findings": self.findings(number),
                "base_is_ancestor": comparison["merge_base_commit"]["sha"] == base,
                "other_prs_with_same_head": shared_head}

    def trusted_check(self, check):
        if check.get("app", {}).get("id") != int(os.environ["AI_REVIEW_APP_ID"]):
            return False
        match = re.fullmatch(rf"https://github.com/{re.escape(REPOSITORY)}/actions/runs/(\d+)", check.get("details_url", ""))
        if not match:
            return False
        run = self.api(f"repos/{REPOSITORY}/actions/runs/{match[1]}")
        # pull_request_target run.head_branch may identify the PR source; the
        # dedicated App and protected Environment enforce the executable ref.
        return (run.get("path") == WORKFLOW and run.get("event") in TRUSTED_EVENTS
                and (run.get("event") == "pull_request_target" or run.get("head_branch") in {"main", "Dev"})
                and (run.get("repository") or {}).get("full_name") == REPOSITORY)

    def context(self, pr, publish, observed_at=None):
        context = {"schema": SCHEMA, "pr_number": pr["number"], "base_ref": pr["base"]["ref"],
                   "head_sha": pr["head"]["sha"], "base_sha": pr["base"]["sha"], "observed_at": observed_at or now()}
        if not publish:
            return context
        context["policy_sha"] = os.environ["SUBMISSION_POLICY_SHA"]
        checks = self.pages(f"repos/{REPOSITORY}/commits/{context['head_sha']}/check-runs?check_name=ai-review&filter=all", "check_runs")
        for check in sorted(checks, key=lambda c: c["id"], reverse=True):
            if not self.trusted_check(check):
                continue
            try:
                prior = json.loads(check["output"]["text"])["context"]
            except (ValueError, TypeError, KeyError):
                continue
            if all(prior.get(k) == context[k] for k in context if k != "observed_at"):
                context["observed_at"] = prior["observed_at"]
                break
        return context

    def publish(self, context, result, check_id=None, name="ai-review"):
        state = result["state"]
        body = {"name": name, "head_sha": context["head_sha"],
                "external_id": f"{SCHEMA}:{context['pr_number']}:{context['base_sha']}:{context['head_sha']}",
                "details_url": f"https://github.com/{REPOSITORY}/actions/runs/{os.environ['GITHUB_RUN_ID']}",
                "status": "in_progress" if state == "pending" else "completed",
                "output": {"title": result["reason"], "summary": result["reason"],
                           "text": json.dumps({"context": context, "result": result}, ensure_ascii=False)}}
        if state != "pending":
            body.update(conclusion="success" if state == "success" else "failure", completed_at=now())
        if check_id:
            body.pop("head_sha")
        return self.api(f"repos/{REPOSITORY}/check-runs" + (f"/{check_id}" if check_id else ""), body, method="PATCH" if check_id else "POST", publisher=True)["id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr", type=int)
    parser.add_argument("--request-body", action="store_true", help="Print the exact version-bound review request; does not post it")
    parser.add_argument("--observed-at", help="Read-only bootstrap evidence boundary; never used for publishing")
    parser.add_argument("--expected-head")
    parser.add_argument("--expected-base")
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 900:
        parser.error("Wait must be between 0 and 900 seconds")
    gh = GitHub()
    if args.request_body:
        if not args.pr:
            parser.error("--request-body requires --pr")
        pr = gh.api(f"repos/{REPOSITORY}/pulls/{args.pr}")
        pr["base"]["sha"] = gh.api(f"repos/{REPOSITORY}/git/ref/heads/{quote(pr['base']['ref'], safe='')}")["object"]["sha"]
        print(request_body(pr))
        return 0
    numbers = [args.pr] if args.pr else [p["number"] for p in gh.pages(f"repos/{REPOSITORY}/pulls?state=open") if p["base"]["ref"] in {"main", "Dev"}]
    exit_code = 0
    for number in numbers:
        context = None
        try:
            pr = gh.api(f"repos/{REPOSITORY}/pulls/{number}")
            if pr["state"] != "open" or pr["base"]["ref"] not in {"main", "Dev"}:
                continue
            pr["base"]["sha"] = gh.api(f"repos/{REPOSITORY}/git/ref/heads/{quote(pr['base']['ref'], safe='')}")["object"]["sha"]
            if (args.expected_head and args.expected_head != pr["head"]["sha"]
                    or args.expected_base and args.expected_base != pr["base"]["sha"]):
                raise RuntimeError("Bootstrap event SHA no longer matches the PR")
            context = gh.context(pr, False, args.observed_at)
            deadline = time.monotonic() + args.wait_seconds
            while True:
                snapshot = gh.collect(pr)
                snapshot["pr"] = gh.api(f"repos/{REPOSITORY}/pulls/{number}")
                snapshot["pr"]["base"]["sha"] = gh.api(f"repos/{REPOSITORY}/git/ref/heads/{quote(snapshot['pr']['base']['ref'], safe='')}")["object"]["sha"]
                result = evaluate(snapshot, context)
                if result["state"] != "pending" or time.monotonic() >= deadline:
                    break
                time.sleep(min(20, max(0, deadline - time.monotonic())))
        except (RuntimeError, ValueError, KeyError, TypeError, OSError, subprocess.TimeoutExpired):
            result = outcome("failure", f"Evidence collection failed for PR #{number}; inspect the workflow")
        print(json.dumps({"context": context, "result": result}, ensure_ascii=False))
        exit_code = max(exit_code, {"success": 0, "failure": 1, "pending": 2}[result["state"]])
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
