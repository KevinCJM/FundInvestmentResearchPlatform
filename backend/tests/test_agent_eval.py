"""AC-5: deterministic fixed-model evaluation of the harness on reusable tasks.

This is engineering evidence against ``FixtureLLMClient`` and temporary data
(zero result, missing result, full-catalog tail, prohibited paste) with explicit
scorers and bounded repeated trials.  It is not a real-model success rate and
makes no token/cost/speed claims.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_eval_fixtures import TRIALS, TASKS, run_all

ARTIFACT = Path(__file__).resolve().parents[2] / ".run" / "harness-upgrade-20260921" / "p2-eval-report.json"


def test_fixed_tasks_pass_with_explicit_scorers_and_stable_trials(tmp_path):
    reports = run_all(tmp_path)
    payload = {"kind": "engineering-fixed-model-evidence", "trials_per_task": TRIALS,
               "reports": reports}
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    assert len(reports) == len(TASKS) * TRIALS
    by_case: dict[str, list[dict]] = {}
    for report in reports:
        by_case.setdefault(report["case"], []).append(report)
    assert set(by_case) == {case.id for case in TASKS}
    for case_id, trials in by_case.items():
        assert len(trials) == TRIALS
        for trial in trials:
            assert trial["status"] == "pass", {case_id: trial}
            assert trial["violations"] == [] and trial["duration_ms"] >= 0
        # Repeated attempts of a deterministic fixture must agree exactly.
        signatures = {(trial["status"], tuple(trial["violations"])) for trial in trials}
        assert len(signatures) == 1, {case_id: signatures}
