"""M3: real common-date clocks, causal signals, independent intervals and ABI."""
from datetime import date, timedelta
import copy
import numpy as np
import pytest
from pydantic import ValidationError as ContractError
from backend.tactical_allocation import numeric
from backend.tactical_allocation.contracts import DecisionPolicy, PreviewRequest, SaveDecisionRequest, SignalComponent
from backend.tactical_allocation.clocks import clock_plan, threshold_kernel
from backend.tactical_allocation.signals import build_composite
from backend.custom_indicators.errors import ValidationError
from backend.tests.test_tactical_allocation_service import workspace


@pytest.fixture(scope='module', autouse=True)
def warm():
    from backend.tactical_allocation.data import warm_tactical_data
    from backend.tactical_allocation.walk_forward import warm_walk_forward_kernels
    warm_tactical_data()
    assert numeric.warm_tactical_allocation_kernels()['complete']
    warm_walk_forward_kernels()


def path(returns, decisions, executions, lag=1, holding=0, threshold=0., cost=10.):
    n = len(returns)
    return numeric._checked_path(returns, np.ones((n, 1)), np.ones(n, dtype=np.uint8),
                                 np.array([[.1, -.1]]), np.array([.5, .5]), cost,
                                 {'decisions': np.array(decisions, dtype=np.uint8),
                                  'executions': np.array(executions, dtype=np.uint8)},
                                 DecisionPolicy(cost_basis="gross_traded_weight", execution_lag=lag, min_holding_periods=holding, deviation_threshold=threshold))


def test_recursive_lag_drift_cost_matches_independent_holdings_reference():
    r = np.array([[.1, 0], [0, .1], [.2, 0], [0, -.1]])
    p = path(r, [1,0,0,0], [1,1,0,0])
    current, nav = np.array([.5,.5]), 1.
    for t in range(4):
        target = np.array([.6,.4]) if t == 1 else current
        gross_trade = abs(target-current).sum()
        gross = target @ r[t]
        nav *= (1-gross_trade*.001)*(1+gross)
        np.testing.assert_allclose(p[t,:2], target, rtol=0, atol=1e-14)
        assert p[t,5] == pytest.approx(gross_trade/2)
        assert p[t,13] == pytest.approx(nav)
        current = target*(1+r[t])/(1+gross)
    assert p[0,20] == 0 and p[1,20] == 1
    assert p[2,5] == p[3,5] == 0


def test_threshold_equality_and_minimum_holding_use_actual_trades():
    assert threshold_kernel(np.array([.125,-.125]), .125)
    assert not threshold_kernel(np.zeros(2), 0.)
    p = path(np.array([[0.,0.],[.1,0.],[0.,.1],[0.,.1],[0.,.1]]), [1]*5, [1]*5, holding=3)
    assert p[1,20] == 1
    assert p[2,21] == p[3,21] == 3
    assert p[4,20] == 1


@pytest.mark.parametrize('frequency,expected', [('daily',[1,1,1,1,1]),('weekly',[1,1,0,1,1]),('monthly',[1,0,1,1,0]),('quarterly',[1,0,1,0,0])])
def test_calendar_start_keys_do_not_consult_future(frequency, expected):
    days=['2025-03-28','2025-03-31','2025-04-01','2025-05-05','2025-05-12']
    policy=DecisionPolicy(decision_frequency=frequency)
    full=clock_plan(days,policy)['decisions']
    assert full.tolist()==expected
    np.testing.assert_array_equal(full[:3],clock_plan(days[:3],policy)['decisions'])


def test_readonly_strided_inputs_and_fixed_signature_no_mutation():
    owner=np.zeros((12,4)); owner[::2,::2]=.001
    view=owner[::2,::2]; view.setflags(write=False)
    before=owner.copy()
    p=path(view,[1]*6,[1]*6)
    q=path(view,[1]*6,[1]*6)
    assert np.shares_memory(view,owner)
    np.testing.assert_array_equal(owner,before)
    np.testing.assert_array_equal(p,q)
    audit=numeric.execution_audit()
    assert audit['complete'] and audit['python_fallback']==audit['request_time_compilation']==0
    assert all(len(s)==1 for s in audit['kernel_signatures'].values())


def external(request, weight=1.):
    return SignalComponent(id='research',kind='value',weight=weight,source='Fixed offline researched fixture',
                           methodology='Explicit normalized cross-asset research scores; no NAV valuation proxy',max_age_days=365,
                           observations=[{'observed_on':request.start_date,'available_on':request.start_date+timedelta(days=25),
                                          'expires_on':request.start_date+timedelta(days=130),'values':{'股票':.2,'债券':-.2}}])


def test_composite_availability_expiry_neutral_and_no_missing_renormalization(workspace):
    service,baseline,request=workspace
    data=service.data.load_data(baseline,str(request.start_date),str(request.end_date),str(request.as_of))
    component=external(request,.5)
    momentum=SignalComponent(id='trend',kind='momentum',weight=.5,source='frozen NAV',methodology='centered normalized window return',lookback=5)
    req=request.model_copy(update={'signal_mode':'composite','signal_components':[component,momentum]})
    signals=build_composite(req,data,['股票','债券'])
    assert not signals['use_signal'][:25].any()
    assert signals['use_signal'][25:131].all()
    assert not signals['use_signal'][131:].any()
    np.testing.assert_array_equal(signals['direct_tilts'][:25],0)
    neutral=component.model_copy(update={'weight':1.,'observations':[component.observations[0].model_copy(update={'values':{'股票':0.,'债券':0.}})]})
    zero=build_composite(req.model_copy(update={'signal_components':[neutral]}),data,['股票','债券'])
    assert zero['knowledge_verified'][25] and not zero['use_signal'][25]
    changed=copy.deepcopy(data); changed['returns']=data['returns'].copy();changed['returns'][100:]=-.1
    perturb=build_composite(req,changed,['股票','债券'])
    np.testing.assert_array_equal(signals['direct_tilts'][:100],perturb['direct_tilts'][:100])


def test_new_service_preview_frozen_inputs_and_clock_export_gate(workspace):
    service,baseline,request=workspace
    req=request.model_copy(update={'signal_mode':'composite','signal_components':[external(request)],
                                   'decision_policy':DecisionPolicy(decision_frequency='weekly'), 'search':False,
                                   'walk_forward':__import__('backend.tactical_allocation.contracts',fromlist=['WalkForwardConfig']).WalkForwardConfig(training_periods=60,validation_periods=30)})
    root = service.repository.artifacts.root
    before = {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
    preview=service.preview(req)
    assert before == {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
    assert not preview['application']['eligible']
    assert preview['application']['threshold_triggered'] is None
    assert any(r['turnover']==0 for r in preview['weight_path'])
    assert preview['walk_forward']['completed_folds']>0
    assert service.repository.list_decisions()==[]
    saved=service.save_decision(SaveDecisionRequest(request=req,preview_hash=preview['preview_hash'],name='clock research'))
    assert saved['preview']['request']['signal_components'][0]['observations']
    with pytest.raises(ValidationError,match='实际持仓'):
        service.product_allocation(saved['id'])
    from backend.tactical_allocation.portfolio_bridge import validate_decision_application
    with pytest.raises(ValidationError, match='实际持仓'):
        validate_decision_application(saved, service.data)
    assert service.repository.get_decision(saved['id'])['preview']==preview


def test_expired_decision_target_cannot_trade(workspace):
    service,baseline,request=workspace
    req=request.model_copy(update={'signal_mode':'composite','signal_components':[external(request)],
                                   'decision_policy':DecisionPolicy(decision_frequency='quarterly'), 'search':False})
    result=service.preview(req)
    assert result['recommendation']['is_saa']


@pytest.mark.parametrize('scheduled', [False, True])
@pytest.mark.parametrize('explicit_days,max_age,expected_days', [(60, 365, 30), (20, 365, 20), (60, 165, 5), (0, 365, 0)])
def test_composite_recommendation_uses_component_expiry(workspace, scheduled, explicit_days, max_age, expected_days):
    from backend.tactical_allocation.portfolio_bridge import validate_decision_application
    service, _, request = workspace
    component = SignalComponent(
        id='long-lived', kind='value', weight=1, source='Offline dated research',
        methodology='Explicit normalized scores', max_age_days=max_age,
        observations=[{'observed_on': request.start_date, 'available_on': request.start_date,
                       'expires_on': request.as_of + timedelta(days=explicit_days),
                       'values': {'股票': .5, '债券': -.5}}])
    req = request.model_copy(update={
        'signal_mode': 'composite', 'signal_components': [component], 'search': False,
        'decision_policy': DecisionPolicy(decision_frequency='daily') if scheduled else None,
        'current_weights': {'股票': .6, '债券': .4}, 'current_weights_as_of': request.as_of,
    })
    assert req.max_signal_age_days == 31  # The old global default is irrelevant to components.
    preview = service.preview(req)
    assert not preview['recommendation']['is_saa']
    assert preview['recommendation']['expires_on'] == str(req.as_of + timedelta(days=expected_days))
    if scheduled:
        assert preview['application']['eligible'], preview['application']['reasons']
    saved = service.save_decision(SaveDecisionRequest(request=req, preview_hash=preview['preview_hash'], name='Composite expiry'))
    # Date eligibility must pass, while the real missing-product-domain gate stays enforced.
    with pytest.raises(ValidationError) as error:
        validate_decision_application(saved, service.data)
    assert error.value.code == 'TAA_APPLICATION_UNIVERSE'
    expired = copy.deepcopy(saved)
    expired['preview']['recommendation']['expires_on'] = str(req.as_of - timedelta(days=1))
    with pytest.raises(ValidationError) as error:
        validate_decision_application(expired, service.data)
    assert error.value.code == 'TAA_DECISION_EXPIRED'


@pytest.mark.parametrize('latest_expiry_days', [2, 60])
def test_scheduled_composite_expiry_tracks_adopted_decision_not_pending_signal(workspace, latest_expiry_days):
    service, _, request = workspace
    component = SignalComponent(
        id='dated', kind='value', weight=1, source='Offline dated research',
        methodology='Explicit normalized scores', max_age_days=365,
        observations=[
            {'observed_on': request.start_date, 'available_on': request.start_date,
             'expires_on': request.as_of + timedelta(days=10), 'values': {'股票': .5, '债券': -.5}},
            {'observed_on': request.as_of, 'available_on': request.as_of,
             'expires_on': request.as_of + timedelta(days=latest_expiry_days), 'values': {'股票': -.5, '债券': .5}},
        ])
    req = request.model_copy(update={
        'signal_mode': 'composite', 'signal_components': [component], 'search': False,
        'decision_policy': DecisionPolicy(decision_frequency='daily', execution_lag=2),
        'current_weights': {'股票': .6, '债券': .4}, 'current_weights_as_of': request.as_of,
    })
    preview = service.preview(req)
    assert preview['application']['eligible'] and preview['application']['pending_decision']
    assert preview['recommendation']['weights']['股票'] > .6
    assert preview['recommendation']['signal_date'] == str(request.start_date)
    assert preview['recommendation']['expires_on'] == str(request.as_of + timedelta(days=10))


def test_complete_holdout_is_independent_and_future_changes_do_not_select(workspace):
    service,baseline,request=workspace
    req=request.model_copy(update={'decision_policy':DecisionPolicy(decision_frequency='weekly')})
    first=service.preview(req)
    load=service.data.load_data
    def altered(*args):
        data=load(*args);data['returns']=data['returns'].copy(); data['returns'][100:]=-.01;return data
    service.data.load_data=altered
    second=service.preview(req)
    assert first['selected_id']==second['selected_id']
    assert [x['train'] for x in first['candidates']]==[x['train'] for x in second['candidates']]
    for result in (first,second):
        assert not result['audit']['selection']['holdout_used_for_selection']
        row=next(x for x in result['weight_path'] if x['segment']=='validation')
        assert row['weights']=={'股票':.6,'债券':.4}


@pytest.mark.parametrize('patch', [{'execution_lag':True},{'execution_lag':0},{'deviation_threshold':float('nan')},{'deviation_threshold':True}])
def test_strict_policy_contracts(patch):
    with pytest.raises(ContractError):DecisionPolicy(**patch)


def test_external_contracts_axis_dates_and_provenance(workspace):
    service,baseline,request=workspace
    c=external(request)
    with pytest.raises(ContractError):SignalComponent.model_validate({**c.model_dump(),'source':' '})
    bad=c.model_dump();bad['observations'][0]['values']['股票']=True
    with pytest.raises(ContractError):SignalComponent.model_validate(bad)
    body=request.model_dump();body.update(signal_mode='composite',signal_components=[c.model_dump()])
    body['signal_components'][0]['observations'][0]['values'].pop('债券')
    with pytest.raises(ValidationError,match='资产轴'):service.preview(PreviewRequest.model_validate(body))


def test_cold_worker_rejects_preview_and_preflight(workspace, monkeypatch):
    service, _, request = workspace
    monkeypatch.setattr(numeric, '_WARMED_PID', None)
    for fn in (service.preview, service.preflight):
        with pytest.raises(RuntimeError, match='not warmed'):
            fn(request)


def test_new_unknown_training_clock_blocks_selection_but_keeps_fixed_research(workspace):
    service, _, request = workspace
    load = service.data.load_data
    def unknown(*args):
        data = load(*args); data['available_at'] = data['available_at'].copy(); data['available_at'][0] = -1
        return data
    service.data.load_data = unknown
    req = request.model_copy(update={'decision_policy': DecisionPolicy(), 'signal_mode': 'manual',
                                      'manual_tilts': {'股票': .05, '债券': -.05}})
    assert not service.preflight(req)['can_calculate']
    with pytest.raises(ValidationError, match='未知可得日期'):
        service.preview(req)
    assert service.preview(req.model_copy(update={'search': False}))['selected_id'] == 'scale-1'


def test_clock_application_uses_actual_threshold_and_holding_not_simulated_weights(workspace):
    service, _, request = workspace
    req = request.model_copy(update={'signal_mode': 'manual', 'manual_tilts': {'股票': .05, '债券': -.05},
                                      'search': False, 'decision_policy': DecisionPolicy(decision_frequency='daily', deviation_threshold=.01),
                                      'current_weights': {'股票': .6, '债券': .4}, 'current_weights_as_of': request.as_of})
    preview = service.preview(req)
    assert preview['application']['eligible']
    assert preview['application']['state'] == 'adjustment_proposal'
    assert preview['application']['threshold_triggered'] is True
    assert preview['application']['decision_date'] < str(request.as_of)
    close = req.model_copy(update={'current_weights': preview['recommendation']['weights']})
    result = service.preview(close)
    assert not result['application']['eligible'] and result['application']['state'] == 'maintain'
    held = req.model_copy(update={'decision_policy': DecisionPolicy(min_holding_periods=30), 'last_execution_date': request.as_of})
    result = service.preview(held)
    assert not result['application']['eligible'] and result['application']['state'] == 'waiting_execution'


def test_nan_and_empty_returns_are_rejected_by_new_core():
    with pytest.raises(ValueError):
        path(np.empty((0,2)), [], [])
    with pytest.raises(ValueError):
        path(np.array([[np.nan,0.],[0.,0.]]),[1,1],[1,1])


def test_cost_basis_is_explicit_and_independent_of_decision_frequency():
    r=np.zeros((4,2)); base=np.array([.5,.5]); tilt=np.array([[.1,-.1]])
    flags=np.ones(4,dtype=np.uint8); probs=np.ones((4,1)); clock={'decisions':flags,'executions':flags}
    half=numeric._checked_path(r,probs,flags,tilt,base,10.,clock,DecisionPolicy(decision_frequency='daily'))
    gross=numeric._checked_path(r,probs,flags,tilt,base,10.,clock,DecisionPolicy(decision_frequency='daily',cost_basis='gross_traded_weight'))
    assert gross[1,6] == pytest.approx(half[1,6]*2)
    np.testing.assert_array_equal(half[:,:2],gross[:,:2])
    scenario=numeric.stress_compare(r,base,np.array([.6,.4]),10.,cost_basis='gross_traded_weight')
    assert scenario['target_cost'] == pytest.approx(gross[:,7].sum())


def test_actual_drift_breaches_are_not_hidden_by_valid_targets():
    r=np.tile(np.array([.005,0.]),(60,1)); base=np.array([.5,.5]); flags=np.ones(60,dtype=np.uint8)
    decisions=np.zeros(60,dtype=np.uint8);decisions[0]=decisions[30]=1
    result=numeric.evaluate_candidates(r,np.ones((60,1)),flags,base,np.array([[.05,-.05]]),
                                      np.zeros(2),np.array([.55,1.]),np.ones(2),30,
                                      strengths=np.array([0.,1.]), selected_candidate_id='scale-1',allow_infeasible_selected=True,
                                      decision_policy=DecisionPolicy(),clock={'decisions':decisions,'executions':flags})
    candidate=result['candidates'][1]
    assert candidate['holding_budget_breaches']['train']>0
    assert candidate['holding_budget_breaches']['validation']>0
    assert not candidate['feasible'] and not candidate['validation_feasible']
