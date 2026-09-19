"""Continuous modeled states, independent confidence and explicit invalid-data failure."""
import time

import numpy as np
import pytest

from custom_indicators.errors import ValidationError
from historical_regimes.authoring import AuthoringRequest, resolve_authoring
from historical_regimes.condition_numba import continuous_state_kernel
from historical_regimes.v2_contracts import parse_definition_v2
from historical_regimes.v2_numba import KERNELS, regime_graph_numba_status
from historical_regimes.v2_service import PortValue, RegimeGraphV2Service
from historical_regimes.v2_registry import node_catalog
from test_regime_granularity import source


def run(candidates, initial=0, confirmation=2):
    values = np.asarray(candidates, dtype=np.int64)
    return continuous_state_kernel(values, np.full(values.size, initial, dtype=np.int64), np.ones(values.size), confirmation, 2)


def test_no_signal_and_pending_reversal_never_open_a_gap_or_backfill():
    states, evidence, pending = run([-1, 0, -1, 1, -1, 1, 1, -1, 0, 0])
    np.testing.assert_array_equal(states, [0, 0, 0, 0, 0, 0, 1, 1, 1, 0])
    np.testing.assert_array_equal(evidence, [0, 1, 2, 3, 2, 3, 1, 2, 3, 1])
    np.testing.assert_array_equal(pending, [0, 0, 0, 1, 0, 1, 0, 0, 1, 0])
    assert run([-1]*100, initial=1)[0].tolist() == [1]*100
    assert run([])[0].size == 0


@pytest.mark.parametrize('confirmation', [1, 2, 5, 252])
def test_prefix_future_perturbation_and_independent_reference(confirmation):
    candidates = np.random.default_rng(91).integers(-1, 2, size=900, dtype=np.int64)
    expected = []
    active, pending = 1, []
    for value in candidates:
        if value == -1 or value == active:
            pending = []
        else:
            pending = pending + [value] if pending and pending[-1] == value else [value]
            if len(pending) >= confirmation:
                active, pending = value, []
        expected.append(active)
    actual = run(candidates, initial=1, confirmation=confirmation)
    np.testing.assert_array_equal(actual[0], expected)
    for end in [0, 1, 20, 300, 650]:
        for full, prefix in zip(actual, run(candidates[:end], initial=1, confirmation=confirmation)):
            np.testing.assert_array_equal(full[:end], prefix)
    changed = candidates.copy(); changed[650:] = 0
    np.testing.assert_array_equal(actual[0][:650], run(changed, initial=1, confirmation=confirmation)[0][:650])


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf, 0, -1])
def test_invalid_observations_fail_instead_of_becoming_a_market_state(invalid):
    with pytest.raises(ValueError, match='OBSERVATION_MISSING'):
        continuous_state_kernel(np.array([0, -1, 1]), np.zeros(3, np.int64), np.array([100., invalid, 101.]), 1, 2)


def test_invalid_initial_codes_axes_parameters_and_fixed_readonly_views():
    with pytest.raises(ValueError, match='INITIAL_REQUIRED'):
        run([0], initial=-1)
    with pytest.raises(ValueError, match='CODE_INVALID'):
        run([2])
    with pytest.raises(ValueError, match='CONTRACT'):
        run([0], confirmation=0)
    with pytest.raises(ValueError, match='CONTRACT'):
        continuous_state_kernel(np.ones(2, np.int64), np.ones(1, np.int64), np.ones(2), 1, 2)
    base = np.array([0, -1, 1, 1, 0, -1]*10, np.int64)
    view = base[::-2]; view.flags.writeable = False
    initial = np.zeros(len(view), np.int64); initial.flags.writeable = False
    prices = np.ones(len(view)); prices.flags.writeable = False
    before = base.copy(); signatures = tuple(continuous_state_kernel.signatures)
    states = continuous_state_kernel(view, initial, prices, 2, 2)[0]
    assert np.shares_memory(view, base) and not np.shares_memory(states, base)
    np.testing.assert_array_equal(base, before)
    assert tuple(continuous_state_kernel.signatures) == signatures
    assert continuous_state_kernel.nopython_signatures and not continuous_state_kernel._can_compile
    with pytest.raises(TypeError):
        continuous_state_kernel(view, initial, prices.astype(np.float32), 1, 2)


def definition():
    def ref(n, p='value'): return {'node_id': n, 'port': p}
    nodes = [source('market', [100., 102, 103, 100, 97, 96, 100, 101]),
        {'id':'initial_check','type':'condition.compare','parameters':{'operator':'ge','threshold':100.},'inputs':{'value':ref('market')}},
        {'id':'initial','type':'state.select','parameters':{'true_code':0,'false_code':1},'inputs':{'condition':ref('initial_check','condition')}},
        {'id':'high','type':'condition.compare','parameters':{'operator':'gt','threshold':101.},'inputs':{'value':ref('market')}},
        {'id':'low','type':'condition.compare','parameters':{'operator':'lt','threshold':99.},'inputs':{'value':ref('market')}},
        {'id':'bear','type':'state.select','parameters':{'true_code':1,'false_code':-1},'inputs':{'condition':ref('low','condition')}},
        {'id':'candidate','type':'state.select','parameters':{'true_code':0,'false_code':-1},'inputs':{'condition':ref('high','condition'),'when_false':ref('bear','state')}},
        {'id':'continuous','type':'state.continuous','parameters':{'confirmation':2},'inputs':{'candidate':ref('candidate','state'),'initial':ref('initial','state'),'value':ref('market')}}]
    return {'schema_version':'2.0','name':'Continuous state fixture','graph':{'nodes':nodes,'outputs':{'state':ref('continuous','state'),'basis':ref('continuous','evidence')},'channel_metadata':{'basis':{'label':'State evidence'}}},
            'states':[{'id':'bull','label':'Bull','role':'positive','color':'#16a34a','order':1},{'id':'bear','label':'Bear','role':'negative','color':'#dc2626','order':2}]}


def test_catalog_formula_real_preview_and_njit_execution(tmp_path, monkeypatch):
    from historical_regimes import granular_runtime
    catalog = next(n for n in node_catalog()['items'] if n['id']=='state.continuous')
    assert catalog['granularity']['kind']=='coupled' and catalog['temporal_contract']['rule']=='history'
    raw = definition()
    first = resolve_authoring(AuthoringRequest(definition=raw, source_kind='graph', mode='realtime'))
    assert first['valid'], first
    again = resolve_authoring(AuthoringRequest(definition=raw, source_kind='formula', source=first['source'], mode='realtime'))
    assert again['valid'] and again['source']==first['source']
    service = RegimeGraphV2Service(tmp_path,tmp_path)
    signatures = regime_graph_numba_status()['kernel_signatures']
    calls = []
    def observed(*args):
        calls.append(True)
        return continuous_state_kernel(*args)
    monkeypatch.setattr(granular_runtime, 'continuous_state_kernel', observed)
    plan = service.prepare(raw)
    assert 'continuous_state' in plan['kernel_ids'] and KERNELS['continuous_state'] is continuous_state_kernel
    job = service.create_preview(raw, mode='realtime', compile_token=plan['compile_token'], audit_temporal=True)
    for _ in range(300):
        job = service.get_preview(job['id'])
        if job['status'] in {'completed','failed'}: break
        time.sleep(.02)
    assert job['status']=='completed', job
    rows = service.preview_series(job['id'])['items']
    assert [row['state_code'] for row in rows]==[0,0,0,0,0,1,1,1]
    assert all(row['state_id'] in {'bull','bear'} for row in rows)
    assert '延续' in rows[-1]['reasons'][0]
    assert calls and regime_graph_numba_status()['kernel_signatures']==signatures
    target = {'node_id':'continuous','port':'state'}
    assert 'continuous_state' in service.prepare(raw, preview_target=target)['kernel_ids']


def test_service_available_time_accumulates_and_missing_is_structured_error(tmp_path):
    service = RegimeGraphV2Service(tmp_path,tmp_path)
    node = parse_definition_v2(definition()).graph.nodes[-1]
    dates = np.arange(3,dtype=np.int64)
    available = np.array([1,10,3],np.int64)
    ports = {'candidate':{'state':PortValue(np.array([0,-1,-1],np.int64),dates,available)},
             'initial':{'state':PortValue(np.zeros(3,np.int64),dates,dates)},
             'market':{'value':PortValue(np.ones(3),dates,dates)}}
    result = service._execute_numeric_node(node,ports,2,'realtime',None,None,{},{},{})
    np.testing.assert_array_equal(result['state'].available,[1,10,10])
    assert result['state'].dates is dates
    ports['market']['value'].values[1]=np.nan
    with pytest.raises(ValidationError) as caught:
        service._execute_numeric_node(node,ports,2,'realtime',None,None,{},{},{})
    assert caught.value.code=='CONTINUOUS_STATE_INPUT_INVALID'


def test_continuity_does_not_remove_hindsight_gate(tmp_path, monkeypatch):
    raw=definition()
    raw['graph']['nodes'].insert(1,{'id':'future','type':'filter.butterworth_zero_phase','parameters':{'period':20},'inputs':{'value':{'node_id':'market','port':'value'}}})
    for n in raw['graph']['nodes']:
        if n['id'] in {'high','low'}: n['inputs']['value']['node_id']='future'
    service=RegimeGraphV2Service(tmp_path,tmp_path)
    monkeypatch.setattr(service,'_resolve_sources',lambda *a,**k: pytest.fail('must reject before I/O'))
    with pytest.raises(ValidationError) as caught:
        service._execute_graph(None,parse_definition_v2(raw),'realtime',None)
    assert caught.value.code=='NON_CAUSAL_REALTIME_GRAPH'
