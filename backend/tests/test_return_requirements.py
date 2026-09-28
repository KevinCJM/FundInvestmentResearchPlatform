"""Return bases and handoff gates: independent formulas and prior audit counterexamples."""
from copy import deepcopy
import numpy as np
import pytest
from pydantic import ValidationError as InputError
from backend.tests.test_strategic_allocation import warm, workspace, saved_inputs
from backend.tests.test_policy_frontier import warmed_frontier
from backend.tests.test_mandate_boundary_contracts import new_definition, budget, stream
from backend.strategic_allocation.contracts import MandateRequest, MandateStudyRequest, PolicyRequest
from backend.strategic_allocation import goal_kernels, kernels
from backend.strategic_allocation.return_targets import requirements, required_mean, check_return
from backend.strategic_allocation.mandate_inputs import require_resolved_authorization
from backend.custom_indicators.errors import ValidationError


@pytest.mark.parametrize('method,periods', [(0, 1), (1, 252)])
@pytest.mark.parametrize('growth', [-.4, 0., .04, 1., 5.])
@pytest.mark.parametrize('risk', [0., .06, .2, 2.])
def test_inverse_matches_independent_lognormal_moment_formula(method, periods, growth, risk):
    before = goal_kernels.execution_audit()['kernel_signatures']
    mean = goal_kernels.compound_required_mean_kernel(growth, risk, method, periods)
    n = periods if method else 1
    base = 1 + mean/n
    implied = np.exp(n*(np.log(base) - .5*np.log1p(risk**2/n/base**2)))-1
    assert implied == pytest.approx(growth, abs=2e-12)
    assert goal_kernels.execution_audit()['kernel_signatures'] == before
    assert goal_kernels.execution_audit()['request_time_compilation'] == 0


@pytest.mark.parametrize('args', [(np.nan,.1,0,1),(.04,np.inf,0,1),(-1.,.1,0,1),(.04,-.1,0,1),(.04,.1,3,252)])
def test_inverse_rejects_invalid_inputs(args):
    with pytest.raises(ValueError): goal_kernels.compound_required_mean_kernel(*args)


def test_compound_is_not_an_arithmetic_floor_and_fees_are_not_deducted_twice():
    definition = MandateRequest.model_validate(new_definition(target_return=.02,
        cash_budget=budget(flows=[stream()], annual_fee=.01))).model_dump(mode='json')
    before = deepcopy(definition)
    result = requirements(definition)
    assert result['arithmetic_floor'] == .02
    compound = result['compound_floor']
    assert compound == result['funding']['cashflow_required_return']
    assert not check_return(result, compound, .2)['within_limits']
    needed = required_mean(result, .2)
    assert check_return(result, needed, .2)['within_limits']
    assert goal_kernels.funding_compound_return_kernel(needed, .2, 0, 1) == pytest.approx(compound)
    assert definition == before


def test_cumulative_target_and_relative_target_remain_typed():
    data = new_definition(target_return=.04, target_return_basis='annual_compound')
    r = requirements(MandateRequest.model_validate(data).model_dump(mode='json'))
    assert r['arithmetic_floor'] is None and r['compound_floor'] == .04
    assert not check_return(r, .04, .2)['within_limits']
    relative = new_definition(objective_kind='benchmark_relative', target_excess_return=.01,
        benchmark={'name':'benchmark','alloc_name':'test','weights':{'a':.5,'b':.5},
                   'target_excess_return':.01,'max_tracking_error':.2})
    r = requirements(relative, means=np.array([.07,.025]), ids=['a','b'])
    assert r['benchmark_return'] == pytest.approx(.0475)
    assert r['arithmetic_floor'] == pytest.approx(.0575)
    assert not check_return(r, .055, .1)['within_limits']
    assert check_return(r, .06, .1)['within_limits']


def test_missing_relative_benchmark_is_rejected_on_input_and_frozen_gate():
    data = new_definition(objective_kind='benchmark_relative', target_excess_return=.99)
    with pytest.raises(InputError, match='基准'): MandateRequest.model_validate(data)
    with pytest.raises(ValidationError, match='基准'): require_resolved_authorization(data)


def test_real_candidate_search_and_frontier_respect_compound_target(workspace):
    service, _ = workspace
    _, cma, request = saved_inputs(service)
    data = new_definition(target_return=.04, target_return_basis='annual_compound', max_volatility=.2)
    preview = service.preview_mandate(MandateStudyRequest(definition=MandateRequest.model_validate(data), cma_id=cma['id']))
    assert preview['candidates']
    for c in preview['candidates']:
        mean, risk = c['metrics']['expected_return'], c['metrics']['volatility']
        assert (1+mean)/np.sqrt(1+(risk/(1+mean))**2)-1 >= .04-1e-10
        assert c['return_check']['within_limits']
    from backend.strategic_allocation.contracts import ConfirmMandateRequest
    saved = service.confirm_mandate(ConfirmMandateRequest(
        request=MandateStudyRequest(definition=MandateRequest.model_validate(data), cma_id=cma['id']),
        preview_hash=preview['preview_hash'], acknowledge_limits=True))
    assert service.get_mandate(saved['id'])['definition']['target_return_basis'] == 'annual_compound'
    from backend.strategic_allocation.policy_frontier import diagnose
    view = diagnose(service, request, preview['definition'], cma)['views'][0]
    assert view['target_return'] is None
    assert view['return_requirements']['compound_floor'] == .04
    assert view['target_curve'][0]['expected_return'] == pytest.approx(.04)
    assert view['target_curve'][-1]['expected_return'] > .04
    impossible = {**data, 'target_return': .9}
    preview = service.preview_mandate(MandateStudyRequest(definition=MandateRequest.model_validate(impossible), cma_id=cma['id']))
    assert preview['status'] == 'needs_revision' and not preview['candidates']


def test_taa_fixed_weight_gate_rechecks_compound_instead_of_arithmetic(workspace):
    from backend.strategic_allocation.contracts import PublishPolicyRequest
    from backend.strategic_allocation.policy_gate import check_policy
    service, _ = workspace
    _, _, request = saved_inputs(service)
    preview = service.preview_policy(request)
    baseline = service.publish_policy(PublishPolicyRequest(request=request,
        preview_hash=preview['preview_hash'], candidate_id='nominal-utility', name='复利门禁基准', reason='离线数值门禁回归'))
    frozen = deepcopy(baseline)
    definition = frozen['policy']['mandate']
    definition.update(target_return=.06, target_return_basis='annual_compound')
    weights = {a['id']: float(i == 0) for i, a in enumerate(frozen['assets'])}
    outcome = check_policy(frozen, weights, .04, str(definition['as_of']))
    assert outcome['expected_return'] >= .06  # The old arithmetic-only check would pass.
    assert not outcome['return_check']['within_limits']
    definition['target_return_basis'] = 'annual_arithmetic'
    assert check_policy(frozen, weights, .04, str(definition['as_of']))['return_check']['within_limits']


def test_all_original_models_recompute_relative_and_compound_requirements(workspace):
    from backend.tests.test_multi_cma import setup
    from backend.strategic_allocation import cma_model_kernels, cma_statistical_kernels, multi_cma_kernels
    from backend.strategic_allocation.multi_cma import cross_model_results
    cma_model_kernels.warm(); cma_statistical_kernels.warm(); multi_cma_kernels.warm()
    service, _ = workspace
    mandate, _, request = setup(service)
    multi = service.preview_policy(request)['multi_cma']
    ids = [a['id'] for a in multi['assumptions']['assets']]
    weights = {x: .5 for x in ids}
    definition = {**mandate['definition'], 'target_return': .04, 'target_return_basis': 'annual_compound'}
    rows = cross_model_results(multi, weights, definition)
    for row in rows:
        mu, sigma = row['metrics']['expected_return'], row['metrics']['volatility']
        growth = (1+mu)/(1+sigma**2/(1+mu)**2)**.5 - 1
        assert row['return_check']['within_limits'] == (growth >= .04 - 1e-10)
    definition.update(objective_kind='benchmark_relative', target_return_basis='annual_arithmetic',
        target_return=0., target_excess_return=.01, benchmark={'name':'半股半债', 'weights':weights,
        'target_excess_return':.01,'max_tracking_error':.2})
    rows = cross_model_results(multi, weights, definition)
    for row in rows:
        assert row['return_check']['required_arithmetic_return'] == pytest.approx(row['metrics']['expected_return']+.01)
        assert not row['return_check']['within_limits']
    assert rows[0]['return_check']['required_arithmetic_return'] != rows[1]['return_check']['required_arithmetic_return']
