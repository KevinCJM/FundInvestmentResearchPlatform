"""Regression tests for decisions which must never accidentally allow a merge."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(os.environ.get("FIRP_GATE_ROOT", Path(__file__).parents[2]))
SPEC = importlib.util.spec_from_file_location("check_ai_review", ROOT / "scripts/check_ai_review.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
HEAD, BASE = "a" * 40, "b" * 40
REVIEW_URL = f"https://github.com/{gate.REPOSITORY}/pull/12#pullrequestreview-1"
START, END = "2026-09-11T10:00:00Z", "2026-09-11T10:10:00Z"
BOT = {"id": gate.BOT_ID, "login": gate.BOT_LOGIN, "type": "Bot"}


def valid_report(context):
    return {**context, "repository": gate.REPOSITORY, "conclusion": "PASS", "findings": [], "limitations": [],
            "reviewer_identity": gate.BOT_LOGIN, "review_run_id": "codex-review-1",
            "reviewed_scope": "Complete PR diff and necessary callers, contracts and tests",
            "evidence": [REVIEW_URL]}


def fixture():
    context = {"schema": gate.SCHEMA, "pr_number": 12, "head_sha": HEAD, "base_sha": BASE, "base_ref": "Dev", "observed_at": START}
    repo = {"full_name": gate.REPOSITORY}
    summary = {"id": 1, "user": BOT, "performed_via_github_app": {"id": gate.CODEX_APP_ID}, "updated_at": END, "html_url": "https://github.com/review",
               "body": '<!-- codex-pull-request-review-summary -->\n| 📝 **Code Review** | ✅ **Completed** <relative-time datetime="2026-09-11T10:10:00Z">done</relative-time> | `aaaaaaa` | PR opened |'}
    snapshot = {"pr": {"number": 12, "state": "open", "draft": False, "head": {"sha": HEAD, "ref": "codex/task", "repo": repo}, "base": {"sha": BASE, "ref": "Dev", "repo": repo}},
                "comments": [summary], "reviews": [], "findings": [], "resolved_commits": {"aaaaaaa": HEAD}, "base_is_ancestor": True, "other_prs_with_same_head": [],
                "reactions": [{"user": BOT, "content": "+1", "created_at": END, "request_comment_id": 2}]}
    snapshot["comments"].append({"id": 2, "body": gate.request_body(snapshot["pr"]), "created_at": START, "updated_at": START})
    return copy.deepcopy(snapshot), copy.deepcopy(context)


class ReviewDecisionTests(unittest.TestCase):
    def state(self, snapshot, context):
        return gate.evaluate(snapshot, context)["state"]

    def test_structured_pass_requires_complete_review_attestation(self):
        for key in ["reviewer_identity", "review_run_id", "reviewed_scope", "evidence"]:
            for value in [None, "", [], "<placeholder>"]:
                s, c = fixture(); report = valid_report(c); report[key] = value
                s["comments"].append({"user": BOT, "performed_via_github_app": {"id": gate.CODEX_APP_ID},
                    "created_at": END, "updated_at": END, "html_url": "report",
                    "body": "```json\n" + json.dumps(report) + "\n```"})
                self.assertEqual(self.state(s, c), "failure", (key, value))

    def test_structured_report_rejects_placeholder_or_malformed_evidence_urls(self):
        for value in ['https://', 'https://<evidence>', 'https://example.com:bad', 'https://example.com:99999',
                      'http://example.com/run', 'https://bad host/run', 'https://a..com/run', 'https://-bad.com/run', None]:
            s, c = fixture(); report = valid_report(c); report['evidence'] = [value]
            s['comments'].append({'user': BOT, 'performed_via_github_app': {'id': gate.CODEX_APP_ID},
                'created_at': END, 'updated_at': END, 'html_url': 'report',
                'body': '```json\n' + json.dumps(report) + '\n```'})
            self.assertEqual(self.state(s, c), 'failure', value)

    def test_report_evidence_must_resolve_to_authenticated_current_pr_record(self):
        for url in ["https://example.com/run", "https://does-not-exist.invalid/run",
                    f"https://github.com/{gate.REPOSITORY}/pull/99#pullrequestreview-1",
                    f"https://github.com/{gate.REPOSITORY}/pull/12#pullrequestreview-999"]:
            s, c = fixture(); report = valid_report(c); report["evidence"] = [url]
            s["reviews"] = [{"user": BOT, "commit_id": HEAD, "state": "COMMENTED", "submitted_at": END,
                             "html_url": REVIEW_URL, "body": "```json\n" + json.dumps(report) + "\n```"}]
            self.assertEqual(self.state(s, c), "failure")

    def test_generic_pr_thumb_is_not_version_bound(self):
        s, c = fixture(); del s["reactions"][0]["request_comment_id"]
        self.assertEqual(self.state(s, c), "pending")

    def test_edited_request_cannot_rebind_old_approval(self):
        s, c = fixture(); s["comments"][1]["updated_at"] = END
        self.assertEqual(self.state(s, c), "pending")

    def test_old_request_cannot_attest_new_base(self):
        s, c = fixture(); s["comments"][1]["created_at"] = s["comments"][1]["updated_at"] = "2026-09-10T10:00:00Z"
        self.assertEqual(self.state(s, c), "pending")

    def test_reaction_cannot_predate_bound_request(self):
        s, c = fixture(); s["comments"][1]["created_at"] = s["comments"][1]["updated_at"] = "2026-09-11T10:11:00Z"
        self.assertEqual(self.state(s, c), "pending")

    def test_dismissed_pass_is_not_evidence(self):
        s, c = fixture(); s["reactions"] = []
        report = valid_report(c)
        s["reviews"] = [{"user": BOT, "commit_id": HEAD, "state": "DISMISSED", "submitted_at": END, "html_url": REVIEW_URL, "body": "```json\n" + json.dumps(report) + "\n```"}]
        self.assertNotEqual(self.state(s, c), "success")

    def test_changes_requested_after_structured_pass_is_blocked(self):
        s, c = fixture()
        report = valid_report(c)
        s["reviews"] = [{"user": BOT, "commit_id": HEAD, "state": "COMMENTED", "submitted_at": END, "html_url": REVIEW_URL, "body": "```json\n" + json.dumps(report) + "\n```"},
                        {"user": BOT, "commit_id": HEAD, "state": "CHANGES_REQUESTED", "submitted_at": "2026-09-11T10:11:00Z", "body": "blocking", "html_url": "y"}]
        self.assertEqual(self.state(s, c), "failure")

    def test_new_official_report_can_supersede_old_changes_requested(self):
        s, c = fixture()
        report = valid_report(c)
        s["reviews"] = [{"user": BOT, "commit_id": HEAD, "state": "CHANGES_REQUESTED", "submitted_at": START, "body": "blocking", "html_url": "old"}]
        report["evidence"] = [f"https://github.com/{gate.REPOSITORY}/pull/12#issuecomment-3"]
        s["comments"].append({"user": BOT, "performed_via_github_app": {"id": gate.CODEX_APP_ID}, "created_at": END, "updated_at": END, "html_url": report["evidence"][0], "body": "```json\n" + json.dumps(report) + "\n```"})
        self.assertEqual(self.state(s, c), "success")

    def test_new_bound_native_verdict_can_supersede_old_changes_requested(self):
        s, c = fixture()
        s["reviews"] = [{"user": BOT, "commit_id": HEAD, "state": "CHANGES_REQUESTED", "submitted_at": START, "body": "blocking", "html_url": "old"}]
        self.assertEqual(self.state(s, c), "success")

    def test_current_native_no_findings(self):
        self.assertEqual(self.state(*fixture()), "success")

    def clean_comment(self):
        return {"id": 3, "user": BOT, "performed_via_github_app": {"id": gate.CODEX_APP_ID},
                "created_at": END, "updated_at": END, "html_url": "clean",
                "body": "Codex Review: Didn't find any major issues. You're on a roll.\n\n**Reviewed commit:** `aaaaaaa`"}

    def test_official_pr12_no_findings_template(self):
        s, c = fixture(); s["reactions"] = []; s["comments"].append(self.clean_comment())
        self.assertEqual(self.state(s, c), "success")
        for field, value in [("body", "Quoted: " + self.clean_comment()["body"]),
                             ("updated_at", "2026-09-11T10:11:00Z"), ("user", {**BOT, "id": 1})]:
            bad = copy.deepcopy(s); bad["comments"][-1][field] = value
            self.assertNotEqual(self.state(bad, c), "success")

    def test_clean_comment_wrong_head_or_old_request(self):
        s, c = fixture(); s["reactions"] = []; s["comments"].append(self.clean_comment())
        s["resolved_commits"]["aaaaaaa"] = "c" * 40
        self.assertNotEqual(self.state(s, c), "success")

    def test_latest_request_invalidates_previous_thumb(self):
        s, c = fixture()
        s["comments"].append({**s["comments"][1], "id": 4, "created_at": "2026-09-11T10:11:00Z", "updated_at": "2026-09-11T10:11:00Z"})
        self.assertNotEqual(self.state(s, c), "success")

    def test_summary_before_request_cannot_pass(self):
        s, c = fixture()
        s["comments"][1]["created_at"] = s["comments"][1]["updated_at"] = "2026-09-11T10:11:00Z"
        s["reactions"][0]["created_at"] = "2026-09-11T10:12:00Z"
        self.assertNotEqual(self.state(s, c), "success")

    def test_contradictory_structured_pass_is_blocked(self):
        s, c = fixture()
        report = valid_report(c)
        s["reviews"] = [{"user": BOT, "commit_id": HEAD, "state": "CHANGES_REQUESTED", "submitted_at": END,
                         "html_url": REVIEW_URL, "body": "```json\n" + json.dumps(report) + "\n```"}]
        self.assertEqual(self.state(s, c), "failure")

    def test_quoted_or_multiple_structured_reports_never_pass(self):
        s, c = fixture()
        report = valid_report(c)
        block = "```json\n" + json.dumps(report) + "\n```"
        for body in ["This is an unsafe example:\n" + block, block + "\n" + block,
                     block + "\n```json\n" + json.dumps({**report,"conclusion":"BLOCKED"}) + "\n```"]:
            s["reviews"] = [{"user": BOT, "commit_id": HEAD, "state": "COMMENTED", "submitted_at": END, "html_url": REVIEW_URL, "body": body}]
            self.assertEqual(self.state(s, c), "failure")

    def test_same_head_on_another_pr_cannot_share_success(self):
        s, c = fixture(); s["other_prs_with_same_head"] = [13]
        self.assertEqual(self.state(s, c), "failure")

    def test_completed_without_positive_is_pending(self):
        s, c = fixture(); s["reactions"] = []
        self.assertEqual(self.state(s, c), "pending")

    def test_old_thumb_cannot_approve_new_review(self):
        s, c = fixture(); s["reactions"][0]["created_at"] = "2026-09-10T10:00:00Z"
        self.assertEqual(self.state(s, c), "pending")

    def test_spoofed_bot_name_rejected(self):
        s, c = fixture(); s["comments"][0]["user"]["id"] = 99
        self.assertNotEqual(self.state(s, c), "success")

    def test_wrong_app_rejected(self):
        s, c = fixture(); s["comments"][0]["performed_via_github_app"]["id"] = 99
        self.assertEqual(self.state(s, c), "pending")

    def test_same_short_sha_resolves_to_different_commit(self):
        s, c = fixture(); s["resolved_commits"]["aaaaaaa"] = "a" * 39 + "c"
        self.assertEqual(self.state(s, c), "pending")

    def test_pr10_completed_with_p2_is_blocked(self):
        s, c = fixture(); s["findings"] = [{"resolved": False, "url": "https://github.com/discussion_r3987603881", "updated_at": END}]
        self.assertEqual(self.state(s, c), "failure")

    def test_author_resolving_thread_without_new_review_does_not_pass(self):
        s, c = fixture(); s["findings"] = [{"resolved": True, "url": "x", "updated_at": "2026-09-11T10:11:00Z"}]
        self.assertNotEqual(self.state(s, c), "success")

    def test_new_head_invalidates_context(self):
        s, c = fixture(); s["pr"]["head"]["sha"] = "c" * 40
        self.assertEqual(self.state(s, c), "failure")

    def test_new_base_invalidates_context(self):
        s, c = fixture(); s["pr"]["base"]["sha"] = "c" * 40
        self.assertEqual(self.state(s, c), "failure")

    def test_wrong_main_source(self):
        s, c = fixture(); s["pr"]["base"]["ref"] = c["base_ref"] = "main"
        self.assertEqual(self.state(s, c), "failure")

    def test_fork_is_rejected(self):
        s, c = fixture(); s["pr"]["head"]["repo"] = {"full_name": "other/repo"}
        self.assertEqual(self.state(s, c), "failure")

    def test_behind_base_is_rejected(self):
        s, c = fixture(); s["base_is_ancestor"] = False
        self.assertEqual(self.state(s, c), "failure")

    def test_pending_summary_cannot_reuse_positive(self):
        s, c = fixture(); s["comments"][0]["body"] = s["comments"][0]["body"].replace("Completed", "In progress")
        self.assertEqual(self.state(s, c), "pending")

    def test_draft_is_blocked(self):
        s, c = fixture(); s["pr"]["draft"] = True
        self.assertEqual(self.state(s, c), "failure")

    def test_structured_pass_must_match_both_shas(self):
        s, c = fixture(); s["comments"] = []; s["reactions"] = []
        report = valid_report(c)
        review = {"user": BOT, "commit_id": HEAD, "state": "COMMENTED", "submitted_at": END, "html_url": REVIEW_URL, "body": "```json\n" + json.dumps(report) + "\n```"}
        s["reviews"] = [review]
        self.assertEqual(self.state(s, c), "success")
        report["base_sha"] = "c" * 40
        review["body"] = "```json\n" + json.dumps(report) + "\n```"
        self.assertNotEqual(self.state(s, c), "success")

    def test_structured_incomplete_never_falls_back_to_thumb(self):
        s, c = fixture()
        report = {**c, "repository": gate.REPOSITORY, "conclusion": "INCOMPLETE", "findings": [], "limitations": ["truncated"]}
        s["reviews"] = [{"user": BOT, "commit_id": HEAD, "submitted_at": END, "html_url": REVIEW_URL, "body": "```json\n" + json.dumps(report) + "\n```"}]
        self.assertEqual(self.state(s, c), "failure")

    def test_later_suggestions_invalidate_earlier_positive(self):
        s, c = fixture(); s["reviews"] = [{"user": BOT, "commit_id": HEAD, "submitted_at": "2026-09-11T10:11:00Z", "body": "suggestion", "html_url": "x"}]
        self.assertNotEqual(self.state(s, c), "success")


class GitHubContractTests(unittest.TestCase):
    def test_null_author_does_not_hide_later_bot_finding(self):
        gh = gate.GitHub()
        threads = {"pageInfo": {"hasNextPage": False}, "nodes": [
            {"isResolved": False, "comments": {"pageInfo": {"hasNextPage": False}, "nodes": [
                {"author": None}, {"author": {"login": "chatgpt-codex-connector"}, "databaseId": 10}]}}]}
        with mock.patch.object(gh, "api", side_effect=[
                {"data": {"repository": {"pullRequest": {"reviewThreads": threads}}}},
                {"user": BOT, "html_url": "finding", "updated_at": END}]):
            self.assertEqual(gh.findings(12), [{"resolved": False, "url": "finding", "updated_at": END}])

    def test_sync_collection_reads_current_main_and_its_ancestry(self):
        gh = gate.GitHub(); s, _ = fixture(); pr = s["pr"]; pr["head"]["ref"] = "codex/sync-main-release"
        for included in [True, False]:
            def response(path):
                if '/git/ref/heads/main' in path:
                    return {"object": {"sha": "c" * 40}}
                if '/compare/' + 'c' * 40 in path:
                    return {"merge_base_commit": {"sha": "c" * 40 if included else BASE}}
                return {"merge_base_commit": {"sha": BASE}}
            with mock.patch.object(gh, "api", side_effect=response), \
                    mock.patch.object(gh, "pages", return_value=[]), mock.patch.object(gh, "findings", return_value=[]):
                snapshot = gh.collect(pr)
                self.assertEqual(snapshot["latest_main"], "c" * 40)
                self.assertEqual(snapshot["main_is_ancestor"], included)

    def test_later_evidence_pages_are_read(self):
        gh = gate.GitHub()
        with mock.patch.object(gh, "api", side_effect=[[{}] * 100, [{"blocking": True}]]) as api:
            self.assertEqual(len(gh.pages("example?state=open")), 101)
            self.assertIn("&per_page=100&page=2", api.call_args.args[0])

    def test_pending_check_is_completed_in_place(self):
        gh = gate.GitHub(); _, context = fixture()
        with mock.patch.dict(gate.os.environ, {"GITHUB_RUN_ID": "42"}), mock.patch.object(gh, "api", return_value={"id": 9}) as api:
            gh.publish(context, gate.outcome("failure", "unresolved"), 9)
            self.assertTrue(api.call_args.args[0].endswith("/9"))
            self.assertNotIn("head_sha", api.call_args.args[1])
            self.assertEqual(api.call_args.kwargs["method"], "PATCH")
            self.assertNotIn("publisher", api.call_args.kwargs)

    def test_only_current_protected_publisher_run_is_accepted(self):
        run = {"id": 1, "path": gate.WORKFLOW, "event": "pull_request_target", "head_sha": BASE,
               "head_branch": "codex/task", "status": "completed", "repository": {"full_name": gate.REPOSITORY}}
        self.assertTrue(gate.GitHub.trusted_publisher_run(run, BASE))
        for key, value in [("event", "pull_request"), ("path", "fake.yml"), ("head_sha", HEAD), ("status", "in_progress")]:
            self.assertFalse(gate.GitHub.trusted_publisher_run({**run, key: value}, BASE))
        self.assertFalse(gate.GitHub.trusted_publisher_run({**run, "event": "workflow_dispatch"}, BASE))

    def test_context_is_read_from_protected_job_logs_not_check_output(self):
        gh = gate.GitHub(); s, c = fixture(); c["policy_sha"] = BASE
        record = {"pr": 12, "context": c, "results": {"ai-review": {"state": "pending"}}}
        run = {"id": 42, "path": gate.WORKFLOW, "event": "push", "head_branch": "main", "head_sha": BASE,
               "status": "completed", "repository": {"full_name": gate.REPOSITORY}}
        with mock.patch.dict(gate.os.environ, {"SUBMISSION_POLICY_SHA": BASE}), \
                mock.patch.object(gh, "pages", side_effect=[[run], [{"id": 8, "name": "publish", "conclusion": "success"}]]) as pages, \
                mock.patch.object(gate.subprocess, "run", return_value=mock.Mock(returncode=0, stdout="publish\tstep\t2026-09-11T10:10:00Z " + json.dumps(record))) as logs:
            self.assertEqual(gh.context(s["pr"], True)["observed_at"], START)
            self.assertNotIn("check-runs", str(pages.call_args_list))
            self.assertIn("--job", logs.call_args.args[0])

    def test_one_pr_api_failure_does_not_skip_other_prs(self):
        gh = gate.GitHub(); s, c = fixture()
        with mock.patch.object(gate, "GitHub", return_value=gh), mock.patch("sys.argv", ["check_ai_review.py"]), \
                mock.patch.object(gh, "pages", return_value=[{"number": n, "base": {"ref": "Dev"}} for n in (11, 12)]), \
                mock.patch.object(gh, "api", side_effect=[RuntimeError("offline"), s["pr"], {"object": {"sha": c["base_sha"]}}, s["pr"], {"object": {"sha": c["base_sha"]}}]), \
                mock.patch.object(gh, "context", return_value=c), mock.patch.object(gh, "collect", return_value=s) as collect, \
                mock.patch("builtins.print"):
            self.assertEqual(gate.main(), 1)
            collect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
