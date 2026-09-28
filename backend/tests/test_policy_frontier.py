"""Frontier diagnostics stay visible when the unchanged policy gate rejects."""
import copy
from datetime import date, timedelta

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import frontier_moments
from backend.custom_indicators.errors import ValidationError
from backend.strategic_allocation.return_targets import requirements
from backend.strategic_allocation.contracts import MandateRequest, PolicyRequest
from backend.strategic_allocation.routes import build_router
from backend.tests.test_strategic_allocation import workspace, warm, saved_inputs, confirmed_mandate
from backend.tests.test_multi_cma import setup, warm_multi


@pytest.fixture(scope="module", autouse=True)
def warmed_frontier():
    frontier_moments.warm()
    from backend.strategic_allocation import compatibility_kernels
    compatibility_kernels.warm()


def test_unreachable_target_keeps_both_curves_and_unchanged_gate(workspace):
    service, _ = workspace
    _, cma, original = saved_inputs(service)
    mandate = confirmed_mandate(service, MandateRequest(name="不可达目标", as_of=date.today(),
        review_date=date.today() + timedelta(days=90), target_return=.15, max_volatility=.095))
    request = original.model_copy(update={"mandate_id": mandate["id"]})
    app = FastAPI(); app.include_router(build_router(service)); client = TestClient(app)
    result = client.post('/api/strategic-allocation/policy/frontier', json=request.model_dump(mode='json'))
    assert result.status_code == 200
    assert result.json()['target_check']['status'] == 'infeasible'
    view = result.json()['views'][0]
    assert view['target_return'] == .15 and view['volatility_cap'] == .095
    assert view['reference']['complete'] and view['configured']['complete']
    assert view['configured']['max_return'] == pytest.approx(.07)
    with pytest.raises(ValidationError) as exc:
        service.preview_policy(request)
    assert exc.value.code == 'SAA_NO_FEASIBLE_CANDIDATE'
    assert result.json()['execution']['complete'] and result.json()['execution']['python_fallback'] == 0


def test_configured_frontier_uses_same_limits_and_frozen_moments(workspace):
    service, _ = workspace
    _, cma, request = saved_inputs(service)
    request = PolicyRequest.model_validate({**request.model_dump(),
        'constraints': {'股票': {'min_weight': 0, 'max_weight': .3}}})
    before = copy.deepcopy(cma)
    result = service.policy_frontier(request)
    assert result['target_check']['status'] == 'feasible'
    view = result['views'][0]
    assert view['reference']['max_return'] == pytest.approx(.07)
    assert view['configured']['max_return'] == pytest.approx(.3*.07 + .7*.025)
    for p in view['configured']['points']:
        assert p['status'] == 'optimal_to_tolerance'
        w = np.array([p['weights'][x] for x in ['股票', '债券']])
        assert w[0] <= .3 + 1e-7
        assert w.sum() == pytest.approx(1.)
        assert p['expected_return'] == pytest.approx(w @ np.array([.07, .025]))
        assert p['volatility'] == pytest.approx(np.sqrt(w @ np.array(cma['covariance']) @ w))
    assert cma == before
    assert service.preview_policy(request)['candidates']


def test_conflicting_weights_preserve_reference_and_target(workspace):
    service, _ = workspace
    _, _, request = saved_inputs(service)
    request = PolicyRequest.model_validate({**request.model_dump(), 'constraints': {
        '股票': {'min_weight': .8, 'max_weight': 1}, '债券': {'min_weight': .8, 'max_weight': 1}}})
    view = service.policy_frontier(request)['views'][0]
    assert view['reference']['complete']
    assert view['configured']['status'] == 'infeasible_certified'
    assert not view['configured']['complete']
    assert all(p['expected_return'] is None for p in view['configured']['points'])


def test_parameter_average_and_common_models_keep_correct_coordinates(workspace):
    service, _ = workspace
    _, sources, request = setup(service)
    blended = service.policy_frontier(request)
    assert len(blended['views']) == 1
    assert blended['views'][0]['configured']['max_return'] == pytest.approx(.7*.07+.3*.09)
    common = PolicyRequest.model_validate({**request.model_dump(), 'mode': 'compatible_all_models',
        'cma_refs': [{**ref.model_dump(), 'weight': None} for ref in request.cma_refs]})
    result = service.policy_frontier(common)
    assert result['additional_checks']['all_models']
    assert [v['id'] for v in result['views']] == [c['id'] for c in sources]
    assert [v['configured']['max_return'] for v in result['views']] == pytest.approx([.07, .09])


def test_frontier_requires_own_process_warmup(workspace, monkeypatch):
    service, _ = workspace
    _, _, request = saved_inputs(service)
    monkeypatch.setattr(frontier_moments, '_WARMED_PID', None)
    app = FastAPI(); app.include_router(build_router(service))
    response = TestClient(app).post('/api/strategic-allocation/policy/frontier', json=request.model_dump(mode='json'))
    assert response.status_code == 503


def test_frontier_coordinates_match_exact_source_or_blended_moments(workspace):
    service, _ = workspace
    _, sources, request = setup(service)
    means = [np.array([a['annual_return'] for a in c['definition']['assets']]) for c in sources]
    covariances = [np.array(c['covariance']) for c in sources]
    blended = service.policy_frontier(request)['views'][0]
    assert blended['moment_basis'] == 'parameter_average'
    assert blended['cma_hash'] == service.preview_policy(request)['multi_cma']['content_hash']
    common = PolicyRequest.model_validate({**request.model_dump(), 'mode': 'compatible_all_models',
        'cma_refs': [{**ref.model_dump(), 'weight': None} for ref in request.cma_refs]})
    views = service.policy_frontier(common)['views']
    assert [v['cma_hash'] for v in views] == [c['content_hash'] for c in sources]
    assert all(v['moment_basis'] == 'source_model' for v in views)
    for view, mu, covariance in [*zip(views, means, covariances),
            (blended, .7*means[0]+.3*means[1], .7*covariances[0]+.3*covariances[1])]:
        for point in view['configured']['points']:
            assert point['status'] == 'optimal_to_tolerance'
            weights = np.array([point['weights'][a['id']] for a in sources[0]['definition']['assets']])
            assert point['expected_return'] == pytest.approx(weights @ mu)
            assert point['volatility'] == pytest.approx(np.sqrt(weights @ covariance @ weights))


@pytest.mark.parametrize('target,expected', [(.04695, 'feasible'), (.047, 'feasible'), (.0471, 'infeasible')])
def test_target_region_uses_continuous_solution_not_grid_intersections(target, expected):
    from backend.strategic_allocation.policy_frontier import _target_check
    from backend.strategic_allocation import compatibility_kernels
    means, risk = np.array([0., .1]), np.diag([0., .04])
    means.flags.writeable = False
    risk.flags.writeable = False
    before = compatibility_kernels.execution_audit()['kernel_signatures']
    view = {'limits': {x: {'min_weight': 0., 'max_weight': 1.} for x in ['cash', 'stock']},
            'groups': [], 'constraint_error': None, 'target_return': target, 'volatility_cap': .094,
            # Nearest plotted samples straddle the region. They are not the feasibility test.
            'configured': {'status': 'optimal_to_tolerance', 'points': []}}
    mandate = {'objective_kind': 'absolute_return', 'target_return': view['target_return'], 'max_volatility': view['volatility_cap']}
    view['return_requirements'] = requirements(mandate)
    request = PolicyRequest(mandate_id='m', cma_id='c')
    check = _target_check(request, [view], [means], [risk], ['cash', 'stock'], mandate)
    assert check['status'] == expected
    if expected == 'infeasible':
        assert check['solver']['phase_one_lower_bound'] > 0
    assert check['execution']['complete']
    assert check['execution']['request_time_compilation'] == 0
    assert check['execution']['kernel_signatures'] == before
    np.testing.assert_array_equal(means, [0., .1])
    np.testing.assert_array_equal(risk, np.diag([0., .04]))


def test_common_target_needs_one_weight_vector_and_unresolved_is_not_infeasible(monkeypatch):
    from backend.strategic_allocation import policy_frontier as module
    view = {'limits': {x: {'min_weight': 0., 'max_weight': 1.} for x in ['a', 'b']},
            'groups': [], 'constraint_error': None, 'target_return': .075, 'volatility_cap': 1.,
            'configured': {'status': 'optimal_to_tolerance', 'points': []}}
    mandate = {'objective_kind': 'absolute_return', 'target_return': view['target_return'], 'max_volatility': view['volatility_cap']}
    view['return_requirements'] = requirements(mandate)
    request = PolicyRequest(mandate_id='m', cma_id='c')
    means = [np.array([.1, 0.]), np.array([0., .1])]
    risks = [np.eye(2)*.04, np.eye(2)*.04]
    for mean in means:
        assert module._target_check(request, [view], [mean], risks[:1], ['a', 'b'], mandate)['status'] == 'feasible'
    assert module._target_check(request, [view, view], means, risks, ['a', 'b'], mandate)['status'] == 'infeasible'
    monkeypatch.setattr(module, 'solve', lambda *args, **kwargs: {'weights': None, 'status': 'time_budget'})
    assert module._target_check(request, [view], means[:1], risks[:1], ['a', 'b'], mandate)['status'] == 'undetermined'

@pytest.mark.parametrize('point_status,risk,passes,expected', [
    ('optimal_to_tolerance', .04, True, 'feasible'),
    ('optimal_to_tolerance', .06, True, 'undetermined'),
    ('optimal_to_tolerance', .04, False, 'undetermined'),
    ('iteration_limit', .04, True, 'undetermined'),
])
def test_verified_frontier_weight_can_prove_feasibility_when_separate_solver_stops(
        monkeypatch, point_status, risk, passes, expected):
    from backend.strategic_allocation import policy_frontier as module
    monkeypatch.setattr(module, 'solve', lambda *a, **k: {'weights': None, 'status': 'phase_one_unresolved'})
    mandate = {'objective_kind': 'absolute_return', 'target_return': .02,
               'target_return_basis': 'annual_compound', 'max_volatility': .05}
    view = {'limits': {x: {'min_weight': 0., 'max_weight': 1.} for x in ['a', 'b']},
        'groups': [], 'constraint_error': None, 'volatility_cap': .05,
        'return_requirements': requirements(mandate),
        'configured': {'status': 'optimal_to_tolerance', 'points': [
            {'status': point_status, 'volatility': risk, 'return_check': {'within_limits': passes}}]}}
    request = PolicyRequest(mandate_id='m', cma_id='c')
    means, risks = [np.array([.06, .025])], [np.diag([.15**2, .01**2])]
    result = module._target_check(request, [view], means, risks, ['a', 'b'], mandate)
    assert result['status'] == expected
    assert result['solver']['status'] == 'phase_one_unresolved'
    # Per-model plotted weights may differ; they cannot certify common weights.
    assert module._target_check(request, [view, view], means*2, risks*2, ['a', 'b'], mandate)['status'] == 'undetermined'
