import threading
from datetime import datetime, timedelta, timezone

from historical_regimes.v2_service import RegimeGraphV2Service
from scenario_stress.service import ScenarioStressService
from scenario_stress.numba_kernels import warm_scenario_numba_kernels
from research_access.contracts import PageContext
from research_access.scenarios import DefinitionArgs, execute
from integrations.portable_agent.service import TOOL_NAMES
from test_scenario_stress import _template


def test_stress_tool_uses_the_real_service_without_publishing(tmp_path):
    assert warm_scenario_numba_kernels()['fully_warmed']
    service = ScenarioStressService(tmp_path, tmp_path)
    definition = _template('factor_path')
    context = PageContext.model_validate({'page': 'scenario-algorithms', 'page_instance_id': 'advanced',
        'calculation': {'context_kind': 'scenario', 'workspace': 'stress'}, 'view_state': 'inherit'})
    snapshot = {'sections': {'editing': {'definition': definition}}}
    calls = []
    service.publish = lambda *args, **kwargs: calls.append('published')
    checked = execute('scenarios.validate', DefinitionArgs(), context, snapshot, {'stress': service}, checkpoint=lambda **kw: None, cancelled=lambda: False)
    assert checked['result']['valid'] is True
    preview = execute('scenarios.preview', DefinitionArgs(), context, snapshot, {'stress': service}, checkpoint=lambda **kw: None, cancelled=lambda: False)
    assert preview['result']['status'] == 'completed'
    assert preview['_scenario_payload']['result']['id'] == preview['result']['run_id']
    assert 'values' not in preview['result'] and 'paths' not in preview['result']
    assert calls == [] and 'scenarios_publish' not in TOOL_NAMES


def test_published_scenario_preview_keeps_full_paths_in_host_artifact(tmp_path):
    from test_published_risk_models import environment
    from backend.sensitivity.kernels import warm_sensitivity_kernels
    warm_sensitivity_kernels()
    env = environment.__wrapped__(tmp_path)
    definition = {'name': '仅试算市场冲击', 'entry': 'market', 'frequency': 'monthly',
                  'input_ids': [env.factor['id']], 'rows': [[-8.0], [0.0], [0.0]]}
    context = PageContext.model_validate({'page': 'published-scenarios', 'page_instance_id': 'published',
        'calculation': {'context_kind': 'scenario', 'workspace': 'published'}, 'view_state': 'inherit'})
    result = execute('scenarios.preview', DefinitionArgs(), context,
        {'sections': {'editing': {'definition': definition}}}, {'published': env.scenarios},
        checkpoint=lambda **kw: None, cancelled=lambda: False)
    assert result['result']['status'] == 'completed'
    assert result['result']['validation_scope'] == 'parameter_schema'
    artifact = result['_scenario_payload']['result']
    assert artifact['transient'] and artifact['preview_hash']
    assert artifact['definition']['rows'] == definition['rows']
    assert 'rows' not in result['result']
    assert env.scenarios.artifacts.list('release') == []


def test_graph_cancelled_status_does_not_claim_thread_completion_or_prune_live_work():
    service = object.__new__(RegimeGraphV2Service)
    release = threading.Event()
    thread = threading.Thread(target=lambda: release.wait(5), daemon=True)
    thread.start()
    job = {'id': 'probe', 'status': 'cancelled', 'thread': thread, 'expires_at': datetime.now(timezone.utc)-timedelta(seconds=1)}
    service._jobs = {'probe': job}
    try:
        assert service._public_job(job)['execution_finished'] is False
        service._prune_jobs()
        assert 'probe' in service._jobs
    finally:
        release.set(); thread.join(5)
    assert service._public_job(job)['execution_finished'] is True
    service._prune_jobs()
    assert service._jobs == {}
