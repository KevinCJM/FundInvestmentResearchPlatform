"""Class-level TAA evidence from the research proxies frozen with an SAA scope.

This adapter never creates an implementation mapping. Proxy return arithmetic
uses the same prewarmed kernels as LTCMA; availability remains source evidence.
"""
from datetime import date
import hashlib
from types import SimpleNamespace

import numpy as np
from pydantic import ValidationError as ContractError

from backend.custom_indicators.errors import ValidationError
from backend.sensitivity.repository import digest_json
from backend.strategic_allocation import reference_evidence_kernels as kernels
from backend.strategic_allocation.reference_contracts import ReferenceAsset
from backend.strategic_allocation.reference_inputs import (
    _rebalance_reset_flags, automatic_research_day, common_return_periods,
)
from backend.strategic_allocation.reference_sources import ReferenceSources


def load_research_data(data_dir, baseline, start, end, cutoff):
    if date.fromisoformat(cutoff) > automatic_research_day(data_dir):
        raise ValidationError('TAA_KNOWLEDGE_CUTOFF', '研究日晚于平台 PIT 截止日，请调整研究日期。')
    # The caller verifies this immutable snapshot, never a live scope or UI hint.
    scope = baseline['strategic_universe_snapshot']['definition']
    if scope['currency'] != 'CNY':
        raise ValidationError('TAA_PROXY_CURRENCY', '研究代理暂只支持人民币口径，不会自动换汇。')
    assets, missing = [], []
    for asset in scope['assets']:
        proxy = asset.get('research_proxy') or {}
        try:
            assets.append(ReferenceAsset.model_validate({
                'id': asset['id'], 'name': asset['name'],
                **{key: value for key, value in proxy.items() if key != 'source_labels'},
            }))
        except ContractError:
            missing.append(asset['name'])
    if missing:
        raise ValidationError('TAA_RESEARCH_PROXY_REQUIRED',
            f"{'、'.join(missing)}尚未设置研究代理或现金收益率。请在研究范围中补充指数、基金代理或现金收益率，再确认新的 SAA；无需先绑定实际交易产品。")
    if [asset.id for asset in assets] != [asset['id'] for asset in baseline['assets']]:
        raise ValidationError('TAA_PROXY_AXIS', '研究代理与已保存 SAA 的大类顺序不一致，请检查研究范围。')
    kernels.require_ready()

    sources = ReferenceSources(data_dir)
    request = SimpleNamespace(as_of=date.fromisoformat(cutoff))
    loaded = {}
    snapshot = manifest = None
    for asset in assets:
        for component in asset.components:
            key = digest_json(component.model_dump(exclude={'weight'}))
            if key in loaded:
                continue
            if snapshot is None:
                snapshot, manifest = sources.active_snapshot_context()
            raw = sources.load(component, request, snapshot=snapshot, manifest=manifest)
            days = np.asarray(raw['dates'], dtype='datetime64[D]').astype(np.int64)
            values = np.asarray(raw['values'], dtype=np.float64)
            if (days.ndim != 1 or len(days) != len(values) or len(days) != len(raw['available_at'])
                    or np.any(days[1:] <= days[:-1]) or not np.isfinite(values).all() or np.any(values <= 0)):
                raise ValidationError('TAA_PROXY_INVALID', f'{asset.name}的研究代理日期或数值无效，请检查数据来源。')
            known = raw['available_at']
            if any(not day or day > cutoff for day in known):
                raise ValidationError('TAA_PROXY_AVAILABILITY', f'{asset.name}的代理数据在研究日尚不可得，请检查公布日期。')
            available = np.asarray(known, dtype='datetime64[D]').astype(np.int64)
            if np.any(available < days):
                raise ValidationError('TAA_PROXY_AVAILABILITY', f'{asset.name}的代理公布日期早于观察日期，请检查数据。')
            loaded[key] = (raw, days, values, available)

    from . import numeric
    numeric._require_ready()
    low, high = (int(np.datetime64(day, 'D').astype(np.int64)) for day in (start, end))
    axes = [days[(days >= low) & (days <= high)] for _, days, _, _ in loaded.values()]
    if axes:
        common, adjacent = common_return_periods(axes)
    else:
        # Only a cash-only scope needs a synthetic observation grid.
        from backend.strategic_allocation.cma_evidence import _calendar
        calendar = _calendar(data_dir, through=date.fromisoformat(end))
        common = calendar[(calendar >= low) & (calendar <= high)]
        adjacent = np.ones(max(len(common) - 1, 0), dtype=bool)
    if len(common) < 3:
        raise ValidationError('TAA_DATA_SHORT', '所选区间的大类代理共同数据不足，请调整日期或检查代理数据。')
    if len(common) > 10000:
        raise ValidationError('TAA_DATA_BUDGET', '单次最多读取 10000 个观察日，请缩短回测区间。')
    # Intersection happens on levels, before returns. Every interval's full
    # price movement is retained, even when only one market traded in between.
    years = numeric.interval_years_kernel(common)
    union = np.unique(np.concatenate(axes)) if axes else common
    within = union[(union >= common[0]) & (union <= common[-1])]
    excluded = np.setdiff1d(within, common)
    alignment = {'method': 'common_observation_intervals/1.0.0',
        'common_observations': len(common), 'return_periods': len(common) - 1,
        'non_common_dates': len(excluded), 'multi_observation_periods': int((~adjacent).sum()),
        'max_calendar_days': int(np.diff(common).max()), 'day_count': 'ACT/365.25',
        'calendar_verified': False, 'sources': []}
    for (raw, _, _, _), axis in zip(loaded.values(), axes):
        missing = np.setdiff1d(within, axis)
        alignment['sources'].append({'series_id': raw['identity']['series_id'],
            'name': raw['identity'].get('name', raw['identity']['series_id']),
            'observations': len(axis), 'not_observed_dates': len(missing),
            'start_date': str(np.datetime64(int(axis[0]), 'D')),
            'end_date': str(np.datetime64(int(axis[-1]), 'D')),
            'date_examples': missing[:5].astype('datetime64[D]').astype(str).tolist()})
    days = common.astype('datetime64[D]').astype(str).tolist()
    panel = np.empty((len(common) - 1, len(assets)), dtype=np.float64)
    availability = np.empty(panel.shape, dtype=np.int64)
    for j, asset in enumerate(assets):
        if asset.asset_type == 'cash':
            panel[:, j] = numeric.cash_interval_returns_kernel(years, float(asset.cash_return))
            # A deterministic scenario assumption, not observed cash interest.
            # Its clock is declared separately below; no historical PIT claim.
            availability[:, j] = common[1:]
            continue
        levels = np.empty((len(common), len(asset.components)), dtype=np.float64)
        clocks = []
        for k, component in enumerate(asset.components):
            key = digest_json(component.model_dump(exclude={'weight'}))
            _, observed, values, available = loaded[key]
            positions = np.searchsorted(observed, common)
            levels[:, k] = values[positions]
            if component.weight > 0:
                clocks.append(available[positions])
        levels.flags.writeable = False
        panel[:, j] = kernels.proxy_returns(kernels.adjacent_returns(levels),
            np.asarray([component.weight for component in asset.components]), _rebalance_reset_flags(days, asset.rebalance))
        # Timestamp propagation is I/O metadata, not return arithmetic. Held
        # weights depend on earlier observations, so keep their knowledge clock.
        clock = np.maximum.reduce(clocks)
        if asset.rebalance != 'daily':
            clock = np.maximum.accumulate(clock)
        availability[:, j] = np.maximum(clock[:-1], clock[1:])
    if not np.isfinite(panel).all():
        raise ValidationError('TAA_PROXY_INVALID', '代理收益包含无效值，请检查代理与权重。')
    # Date-only close evidence cannot support a trade at another market's
    # earlier close on that date. Signal use waits until the following date;
    # training labels retain their actual publication dates (end-of-day cutoff).
    signal_available = availability + 1
    for array in (panel, availability, signal_available, adjacent, years):
        array.flags.writeable = False
    reasons = ['使用 SAA 研究范围中的指数或基金代理进行大类研究；代理不等于最终交易产品。',
               '历史回放使用当前可读取的数据版本及已选范围，不构成历史 PIT 认证。']
    if any(asset.asset_type == 'cash' for asset in assets):
        reasons.append('现金按研究范围中的年化收益率生成固定假设曲线，不代表历史实际利息。')
    lineage = {'source_kind': 'strategic_research_proxies', 'strategic_universe_id': baseline['strategic_universe_id'],
        'strategic_universe_hash': baseline['lineage']['strategic_universe_hash'],
        'proxy_definition': [asset.model_dump(mode='json') for asset in assets],
        'sources': [raw['identity'] for raw, _, _, _ in loaded.values()],
        'requested_as_of': cutoff, 'requested_start': start, 'requested_end': end,
        'start_date': days[0], 'end_date': days[-1], 'observation_count': len(days),
        'cash_basis': 'fixed_scenario_assumption', 'cash_assumption_as_of': scope['as_of'],
        'alignment': alignment['method'], 'alignment_details': alignment,
        'annualization': 'elapsed_time_arithmetic_diffusion', 'cash_day_count': 'ACT/365.25',
        'signal_availability': 'after_publication_date', 'knowledge_time_granularity': 'day',
        'intraday_execution_verified': False, 'missing_availability_rows': 0}
    source_hash = hashlib.sha256(digest_json(lineage).encode() + panel.tobytes() + availability.tobytes()).hexdigest()
    return {'dates': days[1:], 'period_starts': days[:-1], 'returns': panel, 'available_at': availability,
        'period_years': years, 'signal_available_at': signal_available, 'alignment': alignment,
        'period_complete': adjacent, 'excluded_dates': excluded.astype('datetime64[D]').astype(str).tolist(),
        'lineage': lineage, 'reasons': reasons,
        'pit': {'status': 'research_only', 'reasons': reasons}, 'source_hash': source_hash, 'execution': kernels.audit()}
