"""Unmodified recommended graphs through registration, source succession and TAA.

Synthetic data and injected clocks prove protocol mechanics, not investment
accuracy. The reference uses the identical deterministic recipe deliberately:
perfect future labels are a controlled oracle for qualification/consumption.
No model, calibration, capture, maturity, signature or qualification gate is mocked.
"""
import copy
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from historical_regimes.v2_service import RegimeGraphV2Service
from historical_regimes.v2_contracts import parse_definition_v2
from historical_regimes.reliability.prospective import ProspectiveService, default_forward_policy
from historical_regimes.reliability import source_versions as sv
from historical_regimes.reliability.consumer import attach_calibration, calibrated_output, allocation_probabilities
from historical_regimes.taa import run_taa_backtest
from test_regime_prospective import Clock
from test_regime_source_continuation import publish_updated

TEMPLATES = [
    'csi300-maintrend-sma9-realtime-v3',
    'csi300-volatility-hmm-reference-recognition-v1',
    'csi300-drawdown-cycle-realtime-v1',
]


def prices_for(identity, dates):
    if 'volatility' in identity:
        indices = np.arange(len(dates))
        amplitude = np.array([.004, .011, .025])[(indices // 30) % 3]
        return 100 * np.cumprod(1 + np.where(indices % 2, amplitude, -amplitude))
    months = (dates.year - dates[0].year) * 12 + dates.month - dates[0].month
    if 'drawdown' in identity:
        monthly = np.array([100., 100., 80., 70., 80., 90., 98., 100., 100., 80., 70., 80.])
        return monthly[months % len(monthly)]
    # Long rising, flat and falling phases produce each real SMA9 state.
    monthly_returns = np.tile(np.r_[np.full(12, .08), np.zeros(12), np.full(12, -.08)], 20)
    monthly = 100 * np.cumprod(1 + monthly_returns)
    return monthly[months]


@pytest.mark.parametrize('identity', TEMPLATES)
def test_recommended_template_full_forward_workflow(tmp_path, identity):
    frequency = 'daily' if 'volatility' in identity else 'monthly'
    registration = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None).to_period('M').end_time.normalize() + pd.Timedelta(days=1)
    start = registration - pd.DateOffset(years=18)
    capture_dates = (pd.date_range(registration + pd.Timedelta(days=1), periods=60, freq='D') if frequency == 'daily'
                     else pd.period_range(registration.to_period('M'), periods=60, freq='M').end_time.normalize())
    dates = pd.date_range(start, capture_dates[-1] + pd.DateOffset(months=2), freq='D')
    rows = pd.DataFrame({'ts_code': '000300.SH', 'trade_date': dates.strftime('%Y-%m-%d'),
                         'available_at': dates.strftime('%Y-%m-%d'), 'close': prices_for(identity, dates)})
    market = tmp_path / 'market'; market.mkdir()
    original = rows.loc[dates < registration]
    original.to_parquet(market / 'index_daily_df.parquet', index=False)
    before_bytes = (market / 'index_daily_df.parquet').read_bytes()
    graph = RegimeGraphV2Service(tmp_path / 'workspace', market)
    draft = graph.instantiate_template(identity)['definition']
    frozen_recipe = copy.deepcopy(draft['graph'])
    historical = copy.deepcopy(draft)
    historical['study']['purpose'] = 'historical_reference'
    historical['default_mode'] = 'retrospective'
    saved_reference = graph.create_definition(historical)
    clock = Clock(registration.to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(hours=12))
    case = {'graph': graph, 'clock': clock}
    reference = publish_updated(case, {'definition_id': saved_reference['id'], 'revision': 1}, (registration - pd.Timedelta(days=1)).date())
    draft['study']['reference'] = reference
    model = graph.create_definition(draft)
    graph.prepare(model)
    # Preserve actual source, all editable calculation nodes, parameters and evaluation targets.
    assert model['evaluation_targets'] and sv.same_source_targets(parse_definition_v2(model))
    for actual, original_node in zip(model['graph']['nodes'], frozen_recipe['nodes']):
        for key, value in original_node['parameters'].items():
            assert actual['parameters'][key] == value
        assert actual['type'] == original_node['type']
    preview = graph.reliability.preview({'definition_id': model['id'], 'revision': 1, 'reference': reference,
        'policy': {'calibration_end': (registration - pd.DateOffset(years=8)).date().isoformat(),
                   'validation_end': (registration - pd.DateOffset(years=4)).date().isoformat(),
                   'test_end': (registration - pd.Timedelta(days=1)).date().isoformat(),
                   'minimum_samples': 30, 'minimum_class_samples': 5, 'minimum_segments': 3}})
    assert preview['report']['calibration']['fitted'], preview['report']['calibration']
    artifact = graph.reliability.confirm({'request': preview['request'], 'preview_hash': preview['preview_hash']})
    forward = ProspectiveService(graph, graph.reliability, clock=clock); forward.warm()
    graph.prospective = forward
    policy = default_forward_policy(frequency).model_dump()
    policy.update(observation_window=60, minimum_observations=60, minimum_class_observations=5,
                  minimum_class_complete_regimes=1, block_size=5, minimum_complete_blocks=3)
    protocol = forward.register({'calibration_id': artifact['id'], 'policy': policy})
    assert protocol['status'] == 'pending'
    assert forward.capture(protocol['id'])['status'] == 'pending'
    # Append-only immutable successor; the original file and model are untouched.
    successor = market / 'successor'; successor.mkdir()
    rows.to_parquet(successor / 'index_daily_df.parquet', index=False)
    (market / 'tushare_active.json').write_text(json.dumps({'schema_version': 1,
        'snapshot_dir': 'successor', 'snapshot_id': 'successor', 'generation': 'successor'}))
    clock.value = (capture_dates[0] + pd.Timedelta(days=1 if frequency == 'daily' else 2)).to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(hours=12)
    preview_source = sv.preview(forward, protocol['id'])
    accepted = sv.confirm(forward, protocol['id'], {'preview_hash': preview_source['preview_hash']})
    for day in capture_dates:
        clock.value = (day + pd.Timedelta(days=1 if frequency == 'daily' else 2)).to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(hours=12)
        captured = forward.capture(protocol['id'])
        assert captured['status'] == 'captured', captured
        assert captured['observation']['observation_date'] == day.date().isoformat()
        assert captured['observation']['temporal_audit']['verified']
    pending = forward.assess(protocol['id'], {'reference': reference})
    assert pending['status'] != 'qualified'  # old labels cannot mature future evidence
    clock.value += timedelta(seconds=1)
    matured = publish_updated(case, accepted['reference_definition'], capture_dates[-1].date())
    clock.value += timedelta(seconds=1)
    qualified = forward.assess(protocol['id'], {'reference': matured})
    assert qualified['status'] == 'qualified', qualified
    assert qualified['metrics']['paired_samples'] == 60
    assert qualified['qualified_states']
    clock.value += timedelta(seconds=1)
    assert forward.verify_qualification(qualified['id'], artifact['id'], accepted['model_binding_hash'])['status'] == 'qualified'
    assert (market / 'index_daily_df.parquet').read_bytes() == before_bytes
    assert graph.get_definition(model['id'], 1) == model
    adopted = sv.rebind(parse_definition_v2(model), accepted['model_bindings']).model_dump(mode='json')
    adopted['study'].update(calibration_id=artifact['id'], qualification_id=qualified['id'])
    # One further real output boundary supplies effective_date for the prior signal.
    decision = (capture_dates[-1] + pd.Timedelta(days=3) if frequency == 'daily'
                else (capture_dates[-1].to_period('M') + 1).end_time.normalize() + pd.Timedelta(days=2))
    clock.value = decision.to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(hours=12)
    plan = graph.prepare(adopted)
    parsed = parse_definition_v2(adopted)
    execution = graph._execute_graph(None, parsed, 'realtime', decision.date().isoformat(), plan=graph._validate_plan(parsed, plan['compile_token']))
    point = next(p for p in reversed(execution['series']) if p.get('effective_date') and p['state_id'] in qualified['qualified_states'])
    source_run = {'id': 'fixture-current-run', 'definition': adopted, 'definition_id': model['id'], 'states': adopted['states'], 'series': [point]}
    attached = attach_calibration(source_run, graph.reliability)
    assert attached['_reliability']['error'] is None
    states = [s['id'] for s in adopted['states']]
    probabilities, confidence, reason = calibrated_output(attached, point, decision.date().isoformat(), states)
    assert reason is None and confidence == 1.
    assert sum(allocation_probabilities(attached, probabilities).values()) == 1.
    taa = run_taa_backtest(attached, {
        'asset_returns': [{'date': (decision - pd.Timedelta(days=1)).date().isoformat(), 'equity': .0, 'bond': 0.}, {'date': decision.date().isoformat(), 'equity': .01, 'bond': 0.}],
        'base_weights': {'equity': .5, 'bond': .5},
        'state_tilts': {state: {'equity': .1, 'bond': -.1} for state in states},
        'limits': {'min_weight': 0., 'max_weight': 1., 'max_abs_tilt': .2},
        'confidence_floor': .1, 'max_signal_age_days': 3650,
    }, {'passed': True})
    assert taa['weights'][-1]['fallback_to_base'] is False
    assert taa['weights'][-1]['weights']['equity'] == pytest.approx(.6)
