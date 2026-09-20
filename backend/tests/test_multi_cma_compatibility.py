"""Independent numerical references and all-model policy lifecycle tests."""
import copy
import time
from datetime import date, timedelta

import numpy as np
import pytest
from pydantic import ValidationError as InputError
from scipy.optimize import linprog, minimize

from backend.qp_numba import bounded_lp_kernel
from backend.strategic_allocation import compatibility_kernels as numeric
from backend.strategic_allocation.compatibility_solver import solve
from backend.strategic_allocation.contracts import PolicyRequest, PublishPolicyRequest, MandateRequest
from backend.strategic_allocation.policy_gate import check_policy, require_policy_application
from backend.custom_indicators.errors import ValidationError
from backend.tests.test_multi_cma import publish_v2, warm_multi
from backend.tests.test_strategic_allocation import workspace, warm, confirmed_mandate


@pytest.fixture(scope='module', autouse=True)
def ready():
    numeric.warm()


def problem(means=None, risks=None, *, floor=.058, cap=.16, benchmark=None, te=.1, excess=0., references=None, objective=0, **kwargs):
    means = np.array([[.1,.02],[.02,.1]]) if means is None else np.asarray(means, dtype=np.float64)
    m,n = means.shape
    risks = np.array([np.eye(n)*.04]*m) if risks is None else np.asarray(risks, dtype=np.float64)
    inputs = (means, risks, np.tile([0.,1.], (n,1)), np.empty((0,n)), np.empty(0), np.empty(0),
              np.asarray([] if benchmark is None else benchmark, dtype=np.float64), floor, cap, te, excess,
              np.zeros(m) if references is None else np.asarray(references, dtype=np.float64), objective)
    return solve(*inputs, **kwargs)


def test_document_joint_solution_and_model_anchors():
    q = .5 + np.sqrt((.16**2/.04-.5)/2)
    anchor = problem([[.1,.02]])
    assert anchor['status'] == 'converged'
    assert anchor['support_cuts'] > 0
    np.testing.assert_allclose(anchor['weights'], [q,1-q], atol=2e-8)
    assert -anchor['lower_bound'] >= .02+.08*q-1e-10
    regret = problem(references=[.02+.08*q]*2, objective=1)
    assert regret['status'] == 'converged'
    np.testing.assert_allclose(regret['weights'], [.5,.5], atol=1e-9)
    assert regret['objective_value'] == pytest.approx(.02+.08*q-.06, abs=1e-8)
    assert regret['objective_gap'] <= 1e-7


def test_document_conflict_has_phase_one_positive_lower_bound():
    result = problem(floor=.061)
    assert result['status'] == 'infeasible'
    assert result['weights'] is None
    assert result['phase_one_lower_bound'] > 0


def test_quadratic_infeasible_never_calls_a_cut_solution_feasible():
    result = problem(floor=0., cap=.13)
    assert result['status'] == 'infeasible'
    assert result['support_cuts'] > 0
    assert result['phase_one_lower_bound'] > 0


def test_budget_exhaustion_and_timeout_are_not_infeasibility():
    limited = problem([[.1,.02]], max_iterations=1)
    assert limited['status'] == 'iteration_limit' and limited['weights'] is None
    timed = problem(deadline=time.monotonic()-1)
    assert timed['status'] == 'time_budget' and timed['weights'] is None
    assert timed['phase_one_lower_bound'] is None


@pytest.mark.parametrize('seed', range(8))
def test_maximin_matches_independent_scipy_qcqp(seed):
    rng=np.random.default_rng(seed)
    means=rng.uniform(.02,.15,(3,3))
    raw=rng.normal(0,.06,(3,3,3)); risks=np.array([x@x.T+np.eye(3)*.001 for x in raw])
    result=problem(means, risks, floor=0., cap=.12)
    constraints=[{'type':'eq','fun':lambda x: np.sum(x[:3])-1., 'jac':lambda x: np.array([1.,1.,1.,0.])}]
    for mu,cov in zip(means,risks):
        constraints += [dict(type='ineq',fun=lambda x,mu=mu: mu@x[:3]+x[3], jac=lambda x,mu=mu: np.r_[mu,1.]),
                        dict(type='ineq',fun=lambda x,cov=cov: .12**2-x[:3]@cov@x[:3], jac=lambda x,cov=cov: np.r_[-2*cov@x[:3],0.])]
    reference=minimize(lambda x:x[3],np.array([1/3,1/3,1/3,0.]),method='SLSQP',
                       bounds=[(0,1)]*3+[(-1,1)],constraints=constraints,
                       jac=lambda x: np.array([0.,0.,0.,1.]), options={'ftol':1e-11,'maxiter':1000})
    assert reference.success, reference.message
    assert result['weights'] is not None, result
    assert result['objective_value'] == pytest.approx(reference.fun,abs=2e-7)
    assert result['lower_bound'] <= reference.fun+1e-8
    for cov in risks: assert np.sqrt(result['weights']@cov@result['weights']) <= .12+1e-10


def test_shifted_benchmark_te_and_relative_objective():
    result=problem([[.1,.02]],floor=-np.inf,cap=1.,benchmark=[.5,.5],te=.03,excess=.001)
    expected=.5 + .03/np.sqrt(.08)
    assert result['status']=='converged'
    np.testing.assert_allclose(result['weights'],[expected,1-expected],atol=1e-8)
    assert -result['objective_value']==pytest.approx(.08*(expected-.5),abs=1e-8)


@pytest.mark.parametrize('cov', [np.zeros((2,2)),np.full((2,2),.01),np.diag([0.,.04])])
def test_semidefinite_zero_risk_and_cash(cov):
    result=problem([[.02,.06]], [cov],floor=0.,cap=.15)
    assert result['weights'] is not None, result
    assert np.sqrt(result['weights']@cov@result['weights']) <= .15+1e-10


@pytest.mark.parametrize('iterations',[1,500])
def test_lp_lower_bound_matches_reference_even_before_convergence(iterations):
    f=np.array([-.1,.03,.02]);a=np.vstack([np.ones(3),np.eye(3),-np.eye(3),[0,1,1]])
    b=np.array([1,0,0,0,-1,-1,-1,.4],dtype=float)
    x,status,used,checks=bounded_lp_kernel(f,a,b,np.ones(3)/3,np.zeros(3),np.ones(3),iterations,1e-11)
    ref=linprog(f,A_ub=-a[1:],b_ub=-b[1:],A_eq=a[:1],b_eq=b[:1],bounds=[(0,1)]*3,method='highs')
    assert ref.success
    assert checks[4] <= ref.fun+1e-12
    if status==0: assert f@x==pytest.approx(ref.fun,abs=1e-9)


def test_readonly_strides_share_memory_and_no_request_compilation():
    base=np.array([[.02,.1],[.1,.02]])
    means=base[:,::-1];risks=np.array([np.eye(2)*.04]*2)[:,::-1,::-1]
    means.flags.writeable=risks.flags.writeable=False
    assert np.shares_memory(means,base)
    before={k.__name__:list(k.signatures) for k in numeric.KERNELS}
    result=problem(means,risks)
    assert result['status']=='converged'
    np.testing.assert_array_equal(base,[[.02,.1],[.1,.02]])
    assert before=={k.__name__:list(k.signatures) for k in numeric.KERNELS}
    assert numeric.execution_audit()['python_fallback']==0


@pytest.mark.parametrize('bad', [np.nan,np.inf])
def test_invalid_numeric_input_fails_closed(bad):
    with pytest.raises(ValueError): problem([[bad,.02]])


def setup_common(service, *, floor=.058, cap=.16, funding=False):
    args=dict(name='共同模型目标',as_of=date.today(),review_date=date.today()+timedelta(days=90),
              target_return=floor,max_volatility=cap,max_tracking_error=.1)
    if funding:
        args.update(objective_kind='funding_goal',funding_plan=dict(total_capital=100000.,terminal_target=10000.,
                    required_probability=.5,liquidity_months=12,flows=[]))
    mandate=confirmed_mandate(service,MandateRequest(**args))
    def zero_corr(raw): raw['correlation']=[[1.,0.],[0.,1.]]
    sources=[publish_v2(service,'common-a',means=(.1,.02),vol=(.2,.2),patch=zero_corr),
             publish_v2(service,'common-b',means=(.02,.1),vol=(.2,.2),patch=zero_corr)]
    request=PolicyRequest(mandate_id=mandate['id'],mode='compatible_all_models',
                         cma_refs=[dict(cma_id=s['id'],content_hash=s['content_hash']) for s in sources])
    return sources,request


def publish_common(service,request,preview):
    return service.publish_policy(PublishPolicyRequest(request=request,preview_hash=preview['preview_hash'],
        candidate_id='compatible',name='共同约束政策',reason='所有冻结模型均已通过风险与目标检查'))


def test_service_anchors_joint_publication_and_taa_cannot_bypass_model(workspace):
    service,_=workspace
    sources,request=setup_common(service)
    preview=service.preview_policy(request)
    assert preview['execution']['backend'] == 'numba_njit_fixed_signature'
    assert preview['execution']['nopython'] and preview['execution']['object_mode'] == 0
    candidate=preview['candidates'][0]
    np.testing.assert_allclose(list(candidate['weights'].values()),[.5,.5],atol=1e-8)
    assert candidate['all_models_pass']
    assert all(not all(r['within_limits'] for r in a['cross_model_results']) for a in preview['compatibility']['anchors'])
    saved=publish_common(service,request,preview)
    assert saved['policy']['mode']=='compatible_all_models'
    assert service.baselines.get_baseline(saved['id'])==saved
    okay=check_policy(saved,candidate['weights'],.1,str(date.today()))
    assert okay['within_limits'] and okay['risk_evaluation_mode']=='compatible_all_models'
    # Equal-weight display moments still pass the return floor at 75/25. The
    # second original model does not; the production gate must reject it.
    bad={'股票':.75,'债券':.25}
    check=check_policy(saved,bad,.1,str(date.today()))
    assert not check['within_limits']
    assert not check['cross_model_results'][1]['within_limits']
    with pytest.raises(ValidationError): require_policy_application(saved,bad,.1,str(date.today()))


def test_infeasible_study_is_visible_and_cannot_be_published(workspace):
    service,_=workspace
    _,request=setup_common(service,floor=.061)
    preview=service.preview_policy(request)
    assert preview['candidates']==[]
    assert preview['compatibility']['joint_solver']['status']=='infeasible'
    with pytest.raises(ValidationError): publish_common(service,request,preview)


def test_each_model_gets_independent_funding_validation(workspace):
    service,_=workspace
    _,request=setup_common(service,floor=0.,funding=True)
    preview=service.preview_policy(request)
    saved=publish_common(service,request,preview)
    validation=saved['policy']['funding_validation']
    assert validation['within_limits'] and len(validation['models'])==2
    assert validation['distribution']=='each_source_annual_moment_proxy'
    assert all(row['goal_check'] is None for row in check_policy(saved,saved['policy']['selection']['weights'],.1,str(date.today()))['cross_model_results'])


@pytest.mark.parametrize('field,value',[('mode','parameter_average'),('compatibility',{}),('multi_cma',None)])
def test_frozen_common_policy_cannot_be_downgraded(workspace,field,value):
    service,_=workspace
    _,request=setup_common(service)
    saved=publish_common(service,request,service.preview_policy(request))
    broken=copy.deepcopy(saved);broken['policy'][field]=value
    with pytest.raises(ValidationError): check_policy(broken,saved['policy']['selection']['weights'],.1,str(date.today()))


def test_common_request_cannot_silently_ignore_model_weight_or_risk_budget():
    raw=dict(mandate_id='mandate',mode='compatible_all_models',cma_refs=[dict(cma_id='cma',content_hash='a'*64)])
    assert PolicyRequest(**raw).cma_refs[0].weight is None
    with pytest.raises(InputError): PolicyRequest(**{**raw,'risk_budget':{'股票':1.}})
    raw['cma_refs'][0]['weight']=1.
    with pytest.raises(InputError): PolicyRequest(**raw)


@pytest.mark.parametrize('seed', range(4))
def test_minimax_regret_against_independent_reference(seed):
    rng = np.random.default_rng(seed+30)
    means = rng.uniform(.01, .12, (3, 3))
    risks = np.array([np.diag(rng.uniform(.01, .04, 3)) for _ in range(3)])
    references = []
    for mu, cov in zip(means, risks):
        ref = minimize(lambda w: -mu@w, np.ones(3)/3, jac=lambda w: -mu, method='SLSQP',
            bounds=[(0, 1)]*3, constraints=[{'type': 'eq', 'fun': lambda w: w.sum()-1, 'jac': lambda w: np.ones(3)},
                {'type': 'ineq', 'fun': lambda w: .13**2-w@cov@w, 'jac': lambda w: -2*cov@w}],
            options={'ftol': 1e-11, 'maxiter': 1000})
        assert ref.success
        references.append(-ref.fun)
    constraints = [{'type': 'eq', 'fun': lambda x: x[:3].sum()-1, 'jac': lambda x: np.r_[np.ones(3), 0.]}]
    for mu, cov, reference in zip(means, risks, references):
        constraints.extend([
            {'type': 'ineq', 'fun': lambda x, mu=mu, reference=reference: mu@x[:3]+x[3]-reference,
             'jac': lambda x, mu=mu: np.r_[mu, 1.]},
            {'type': 'ineq', 'fun': lambda x, cov=cov: .13**2-x[:3]@cov@x[:3],
             'jac': lambda x, cov=cov: np.r_[-2*cov@x[:3], 0.]}])
    reference = minimize(lambda x: x[3], np.r_[np.ones(3)/3, .1], jac=lambda x: np.r_[np.zeros(3), 1.],
        method='SLSQP', bounds=[(0, 1)]*4, constraints=constraints, options={'ftol': 1e-11, 'maxiter': 1000})
    assert reference.success
    result = problem(means, risks, floor=0., cap=.13, references=references, objective=1)
    assert result['status'] == 'converged', result
    assert result['objective_value'] == pytest.approx(reference.fun, abs=2e-7)


def test_overlapping_group_and_asset_bounds_with_zero_tracking_error():
    means = np.array([[.08, .03, .06], [.04, .09, .05]])
    risks = np.array([np.eye(3)*.04]*2)
    benchmark = np.array([.3, .3, .4])
    result = solve(means, risks, np.array([[.1, .5]]*3), np.array([[1., 1., 0.], [0., 1., 1.]]),
        np.array([.5, .5]), np.array([.8, .8]), benchmark, -np.inf, .2, 0., 0., np.zeros(2), 0)
    assert result['weights'] is not None, result
    np.testing.assert_allclose(result['weights'], benchmark, atol=1e-8)


def test_common_readiness_is_bound_to_worker_pid(monkeypatch):
    monkeypatch.setattr(numeric, '_WARMED_PID', -1)
    with pytest.raises(RuntimeError): problem()
