"""Repeated engineering scenarios; scores actual state, not fixture prose."""
import json
import time
from pathlib import Path

import pytest

from test_agent_task_state import (
    test_middle_negative_quote_and_later_correction_survive_twenty_turns_and_three_compactions as long_scenario,
    test_memory_confirmation_scope_replace_revoke_and_new_session_isolation as memory_scenario,
    test_stopped_edited_turn_removes_plan_memory_sources_and_keeps_prior_state as edit_scenario,
)


CASES = {
    'long-correction-three-compactions': long_scenario,
    'human-memory-save-boundary': memory_scenario,
    'stop-edit-source-exclusion': edit_scenario,
}


def test_repeated_multiturn_scenarios(tmp_path, monkeypatch):
    results = []
    for name, scenario in CASES.items():
        for attempt in range(3):
            path = tmp_path / name / str(attempt); path.mkdir(parents=True)
            started = time.monotonic()
            violations = []
            with monkeypatch.context() as patch:
                patch.setenv('CUSTOM_INDICATOR_DATA_DIR', str(path))
                try:
                    if name.startswith('long-'):
                        scenario(path, patch)
                    else:
                        scenario(path)
                except (AssertionError, Exception) as exc:
                    # Never put the failed payload into the report.
                    violations.append(type(exc).__name__)
            results.append({'case': name, 'attempt': attempt, 'status': 'pass' if not violations else 'violation',
                            'violations': violations, 'duration_ms': int((time.monotonic()-started)*1000),
                            'tokens': None, 'cost': None, 'model': 'FixtureLLMClient', 'scorer': scenario.__name__})
    artifact = Path(__file__).resolve().parents[2] / '.run/harness-upgrade-20260921/p2-multiturn-eval-report.json'
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps({'kind': 'offline-engineering', 'trials': 3, 'results': results}, indent=2))
    assert all(item['status'] == 'pass' for item in results), results
