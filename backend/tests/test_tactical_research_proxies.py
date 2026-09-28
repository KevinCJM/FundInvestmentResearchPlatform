"""Real local series -> unmapped SAA -> TAA research, never product authority."""
from copy import deepcopy
from datetime import date
import json

import numpy as np
import pandas as pd
import pytest

from backend.custom_indicators.errors import ConflictError, ValidationError
from backend.tactical_allocation.contracts import PreviewRequest, SaveDecisionRequest, ScenarioRequest
from backend.tactical_allocation.service import TacticalAllocationService
from backend.tests import test_bettersaataa_scope_journey as journey
from backend.tests.test_bettersaataa_scope_journey import warmed
from backend.tests.risk_scale_app import seed_sources


@pytest.fixture
def research(tmp_path, monkeypatch):
    seed_sources(tmp_path / 'market')
    definition = journey.universe_request()
    definition['assets'][0]['research_proxy'] = {
        'asset_type': 'market', 'cash_return': None, 'rebalance': 'daily', 'components': [
            {'kind': 'index', 'series_id': 'index:index_daily:900003.SH', 'field': 'close', 'weight': .6},
            {'kind': 'etf', 'series_id': 'etf:fund_daily:900002.SH', 'field': 'adj_nav', 'weight': .4}]}
    definition['assets'][1]['research_proxy'] = {
        'asset_type': 'cash', 'cash_return': .02, 'rebalance': None, 'components': []}
    monkeypatch.setattr(journey, 'universe_request', lambda: deepcopy(definition))
    _, _, baseline, _, _ = journey.setup_journey(tmp_path)
    service = TacticalAllocationService(tmp_path / 'research', tmp_path / 'market')
    assert service.warm()['research_proxies']['complete']
    frame = pd.read_parquet(tmp_path / 'market' / 'synthetic-test-only' / 'index_daily_df.parquet')
    days = sorted(frame.trade_date.unique())
    request = PreviewRequest(baseline_id=baseline['id'], start_date=pd.Timestamp(days[0]).date(),
        end_date=pd.Timestamp(days[-1]).date(), as_of=date.today(),
        train_end_date=pd.Timestamp(days[180]).date(), lookback=20,
        signal_mode='manual', manual_tilts={'growth': -.02, 'cash': .02}, max_tracking_error=.1, search=False)
    return service, baseline, request


def test_real_proxies_enable_preflight_calculation_scenarios_save_and_reopen(research, monkeypatch):
    service, baseline, request = research
    assert baseline['implementation_status'] == 'incomplete'
    assert not (service.data.data_dir / 'asset_alloc_info.parquet').exists()
    assert service.preflight(request)['can_calculate']
    result = service.preview(request)
    assert result['data']['observations'] == 300
    assert result['data']['lineage']['source_kind'] == 'strategic_research_proxies'
    assert result['data']['lineage']['proxy_definition'][0]['components'][0]['weight'] == .6
    assert result['policy_check']['current_application_eligible'] is False
    assert not result['audit']['formal_pit_eligible']
    shock = service.scenario(ScenarioRequest(preview_request=request,
        scenario={'kind': 'shock', 'name': '股跌现金不变', 'shocks': {'growth': -.2, 'cash': 0.}}))
    assert np.isfinite(shock['taa_return'])
    saved = service.save_decision(SaveDecisionRequest(request=request, preview_hash=result['preview_hash'], name='无产品的大类研究'))
    frozen = service.repository.decision_arrays(saved['id'])
    assert frozen['returns'].shape == (300, 2)
    np.testing.assert_allclose(frozen['returns'][:, 1], np.expm1(np.log1p(.02) * frozen['period_years']))
    np.testing.assert_allclose(frozen['period_years'], (frozen['period_end_days'] - frozen['period_start_days']) / 365.25)
    assert not frozen['returns'].flags.writeable
    with pytest.raises(ValidationError):
        service.product_allocation(saved['id'])
    with pytest.raises(ValidationError, match='实际交易产品'):
        service.data.validate_application({**baseline, 'apply_eligible': True, 'implementation_status': 'complete'})
    monkeypatch.setattr(service.data, 'load_data', lambda *_: pytest.fail('reopening must use frozen results'))
    assert service.repository.get_decision(saved['id'])['preview'] == result


def test_proxy_returns_match_explicit_weighted_index_and_fund_returns(research):
    service, baseline, request = research
    loaded = service.data.load_data(baseline, str(request.start_date), str(request.end_date), str(request.as_of))
    x = np.arange(300)
    expected = .6 * (.0004 + .010 * np.sin(x * .43)) + .4 * (.00025 + .003 * np.cos(x * .19))
    np.testing.assert_allclose(loaded['returns'][:, 0], expected, atol=1e-15)
    assert loaded['available_at'].shape == loaded['returns'].shape
    assert not loaded['available_at'].flags.writeable


def test_proxy_snapshot_tampering_is_rejected(research):
    service, baseline, request = research
    changed = deepcopy(baseline)
    changed['strategic_universe_snapshot']['definition']['assets'][0]['research_proxy']['components'][0]['weight'] = .5
    with pytest.raises(ValidationError, match='哈希'):
        service.data.load_data(changed, str(request.start_date), str(request.end_date), str(request.as_of))


def test_source_change_invalidates_preview_but_does_not_rewrite_saved_saa(research):
    service, baseline, request = research
    result = service.preview(request)
    path = service.data.data_dir / 'synthetic-test-only' / 'index_daily_df.parquet'
    frame = pd.read_parquet(path)
    index = frame.index[frame.ts_code == '900003.SH'][70]
    frame.loc[index, 'close'] *= 1.001
    frame.to_parquet(path, index=False)
    with pytest.raises(ConflictError, match='重新预览'):
        service.save_decision(SaveDecisionRequest(request=request, preview_hash=result['preview_hash'], name='过期预览'))
    assert service.repository.get_baseline(baseline['id']) == baseline


@pytest.mark.parametrize('issue', ['invalid_value', 'missing_announcement', 'pit_cutoff'])
def test_missing_or_unavailable_evidence_is_not_silently_filled(research, issue):
    service, _, request = research
    path = service.data.data_dir / 'synthetic-test-only' / 'etf_daily_df.parquet'
    frame = pd.read_parquet(path)
    index = frame.index[frame.ts_code == '900002.SH'][80]
    if issue == 'invalid_value':
        frame.loc[index, 'adj_nav'] = -1.
        frame.to_parquet(path, index=False)
        code = 'TAA_PROXY_INVALID'
    elif issue == 'missing_announcement':
        frame.loc[index, 'ann_date'] = pd.NaT
        frame.to_parquet(path, index=False)
        code = 'REFERENCE_INFORMATION_CLOCK'
    else:
        (service.data.data_dir / 'pit_settings.json').write_text(json.dumps({'as_of': str(request.train_end_date)}))
        code = 'TAA_KNOWLEDGE_CUTOFF'
    with pytest.raises(ValidationError) as error:
        service.preflight(request)
    assert error.value.code == code


def test_late_announcements_cannot_enter_training_selection(research):
    service, _, request = research
    path = service.data.data_dir / 'synthetic-test-only' / 'etf_daily_df.parquet'
    frame = pd.read_parquet(path)
    frame.loc[frame.ts_code == '900002.SH', 'ann_date'] = pd.Timestamp(request.as_of)
    frame.to_parquet(path, index=False)
    searched = request.model_copy(update={'search': True})
    preflight = service.preflight(searched)
    assert not preflight['can_calculate']
    assert preflight['training']['unavailable_count'] > 0
    with pytest.raises(ValidationError):
        service.preview(searched)
    assert service.preflight(request)['can_calculate']  # Explicit fixed hypothesis remains research.


def test_different_observation_dates_preserve_all_interval_returns(research, monkeypatch):
    service, baseline, request = research
    path = service.data.data_dir / 'synthetic-test-only' / 'etf_daily_df.parquet'
    frame = pd.read_parquet(path)
    product = frame.loc[frame.ts_code == '900002.SH']
    original_dates = sorted(product.nav_date.unique())
    frame.drop(index=product.index[[80, 81, 190]]).to_parquet(path, index=False)
    from backend.strategic_allocation import cma_evidence
    monkeypatch.setattr(cma_evidence, '_calendar', lambda *a, **kw: pytest.fail('No SSE calendar may gate cross-market proxies'))
    loaded = service.data.load_data(baseline, str(request.start_date), str(request.end_date), str(request.as_of))
    assert loaded['alignment']['non_common_dates'] == 3
    assert loaded['alignment']['multi_observation_periods'] == 2
    assert loaded['alignment']['calendar_verified'] is False
    assert len(loaded['returns']) == 297
    positions = {pd.Timestamp(day).date().isoformat(): i for i, day in enumerate(original_dates)}
    x = np.arange(300)
    index_returns = .0004 + .010 * np.sin(x * .43)
    fund_returns = .00025 + .003 * np.cos(x * .19)
    expected = []
    for start, end in zip(loaded['period_starts'], loaded['dates']):
        left, right = positions[start], positions[end]
        expected.append(.6 * (np.prod(1 + index_returns[left:right]) - 1)
                        + .4 * (np.prod(1 + fund_returns[left:right]) - 1))
    np.testing.assert_allclose(loaded['returns'][:, 0], expected, atol=1e-14)
    elapsed = (date.fromisoformat(loaded['dates'][-1]) - date.fromisoformat(loaded['period_starts'][0])).days
    assert np.prod(1 + loaded['returns'][:, 1]) == pytest.approx(1.02 ** (elapsed / 365.25))
    assert service.preflight(request)['can_calculate']
    result = service.preview(request)
    assert result['data']['alignment'] == loaded['alignment']
    assert result['audit']['periods_per_year'] is None
    saved = service.save_decision(SaveDecisionRequest(request=request, preview_hash=result['preview_hash'], name='共同区间'))
    assert service.repository.get_decision(saved['id'])['preview'] == result
    np.testing.assert_array_equal(service.repository.decision_arrays(saved['id'])['returns'], loaded['returns'])
    from backend.tactical_allocation import numeric
    original = numeric.evaluate_candidates
    seen = []
    def capture(*args, **kwargs):
        seen.append(kwargs['period_years'])
        return original(*args, **kwargs)
    monkeypatch.setattr(numeric, 'evaluate_candidates', capture)
    from backend.tactical_allocation.contracts import WalkForwardConfig
    config = WalkForwardConfig(training_periods=100, validation_periods=60)
    folded = service.preview(request.model_copy(update={'walk_forward': config}))
    assert folded['walk_forward']['completed_folds'] >= 1
    for fold, duration in zip(folded['walk_forward']['folds'], seen[1:]):
        assert len(duration) == fold['training_observations'] + fold['validation_observations'] + fold['purged_training_periods']
    assert all(duration is not None for duration in seen)
    history = ScenarioRequest(preview_request=request, scenario={'kind': 'historical', 'name': '区间重演',
        'start_date': loaded['period_starts'][40], 'end_date': loaded['dates'][100]})
    scenario = service.scenario(history)
    assert scenario['evidence']['annualization'] == 'elapsed_time_arithmetic_diffusion'
    momentum = request.model_copy(update={'signal_mode': 'momentum', 'search': True})
    signals = service._signals(momentum, loaded, ['growth', 'cash'])
    for row in signals['audit']['signal_timing']:
        if row['active']:
            assert row['window_end'] < row['period_start']
            assert row['available_at'] <= row['period_start']


def test_no_overlap_still_fails_instead_of_filling_prices(research):
    service, _, request = research
    path = service.data.data_dir / 'synthetic-test-only' / 'etf_daily_df.parquet'
    frame = pd.read_parquet(path)
    product = frame.loc[frame.ts_code == '900002.SH']
    frame.drop(index=product.index[2:]).to_parquet(path, index=False)
    with pytest.raises(ValidationError) as error:
        service.preflight(request)
    assert error.value.code == 'TAA_DATA_SHORT'
