"""Regression tests for decisions which must never accidentally allow a merge."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("check_ai_review", Path(__file__).parents[1] / "check_ai_review.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
HEAD, BASE = "a" * 40, "b" * 40
START, END = "2026-09-11T10:00:00Z", "2026-09-11T10:10:00Z"
BOT = {"id": gate.BOT_ID, "login": gate.BOT_LOGIN, "type": "Bot"}


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
        report = {**c, "repository": gate.REPOSITORY, "conclusion": "PASS", "findings": [], "limitations": []}
        s["reviews"] = [{"user": BOT, "commit_id": HEAD, "state": "DISMISSED", "submitted_at": END, "html_url": "x", "body": "```json\n" + json.dumps(report) + "\n```"}]
        self.assertNotEqual(self.state(s, c), "success")

    def test_changes_requested_after_structured_pass_is_blocked(self):
        s, c = fixture()
        report = {**c, "repository": gate.REPOSITORY, "conclusion": "PASS", "findings": [], "limitations": []}
        s["reviews"] = [{"user": BOT, "commit_id": HEAD, "state": "COMMENTED", "submitted_at": END, "html_url": "x", "body": "```json\n" + json.dumps(report) + "\n```"},
                        {"user": BOT, "commit_id": HEAD, "state": "CHANGES_REQUESTED", "submitted_at": "2026-09-11T10:11:00Z", "body": "blocking", "html_url": "y"}]
        self.assertEqual(self.state(s, c), "failure")

    def test_new_official_report_can_supersede_old_changes_requested(self):
        s, c = fixture()
        report = {**c, "repository": gate.REPOSITORY, "conclusion": "PASS", "findings": [], "limitations": []}
        s["reviews"] = [{"user": BOT, "commit_id": HEAD, "state": "CHANGES_REQUESTED", "submitted_at": START, "body": "blocking", "html_url": "old"}]
        s["comments"].append({"user": BOT, "performed_via_github_app": {"id": gate.CODEX_APP_ID}, "updated_at": END, "html_url": "new", "body": "```json\n" + json.dumps(report) + "\n```"})
        self.assertEqual(self.state(s, c), "success")

    def test_new_bound_native_verdict_can_supersede_old_changes_requested(self):
        s, c = fixture()
        s["reviews"] = [{"user": BOT, "commit_id": HEAD, "state": "CHANGES_REQUESTED", "submitted_at": START, "body": "blocking", "html_url": "old"}]
        self.assertEqual(self.state(s, c), "success")

    def test_current_native_no_findings(self):
        self.assertEqual(self.state(*fixture()), "success")

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
        report = {**c, "repository": gate.REPOSITORY, "conclusion": "PASS", "findings": [], "limitations": []}
        review = {"user": BOT, "commit_id": HEAD, "state": "COMMENTED", "submitted_at": END, "html_url": "x", "body": "```json\n" + json.dumps(report) + "\n```"}
        s["reviews"] = [review]
        self.assertEqual(self.state(s, c), "success")
        report["base_sha"] = "c" * 40
        review["body"] = "```json\n" + json.dumps(report) + "\n```"
        self.assertNotEqual(self.state(s, c), "success")

    def test_structured_incomplete_never_falls_back_to_thumb(self):
        s, c = fixture()
        report = {**c, "repository": gate.REPOSITORY, "conclusion": "INCOMPLETE", "findings": [], "limitations": ["truncated"]}
        s["reviews"] = [{"user": BOT, "commit_id": HEAD, "submitted_at": END, "html_url": "x", "body": "```json\n" + json.dumps(report) + "\n```"}]
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
            self.assertTrue(api.call_args.kwargs["publisher"])

    def test_wrong_check_publisher_cannot_supply_context(self):
        with mock.patch.dict(gate.os.environ, {"AI_REVIEW_APP_ID": "999"}):
            self.assertFalse(gate.GitHub().trusted_check({"app": {"id": 15368}}))

    def test_target_run_can_record_development_head_branch(self):
        gh = gate.GitHub()
        check = {"app": {"id": 999}, "details_url": f"https://github.com/{gate.REPOSITORY}/actions/runs/42"}
        run = {"path": gate.WORKFLOW, "event": "pull_request_target", "head_branch": "codex/task", "repository": {"full_name": gate.REPOSITORY}}
        with mock.patch.dict(gate.os.environ, {"AI_REVIEW_APP_ID": "999"}), mock.patch.object(gh, "api", return_value=run):
            self.assertTrue(gh.trusted_check(check))
            run["event"] = "workflow_dispatch"
            self.assertFalse(gh.trusted_check(check))

    def test_one_pr_api_failure_does_not_skip_other_prs(self):
        gh = gate.GitHub(); s, c = fixture()
        with mock.patch.object(gate, "GitHub", return_value=gh), mock.patch("sys.argv", ["check_ai_review.py"]), \
                mock.patch.object(gh, "pages", return_value=[{"number": n, "base": {"ref": "Dev"}} for n in (11, 12)]), \
                mock.patch.object(gh, "api", side_effect=[RuntimeError("offline"), s["pr"], s["pr"]]), \
                mock.patch.object(gh, "context", return_value=c), mock.patch.object(gh, "collect", return_value=s) as collect, \
                mock.patch("builtins.print"):
            self.assertEqual(gate.main(), 1)
            collect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
