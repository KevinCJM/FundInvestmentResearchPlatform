"""Read-only adapter from Risk Scale reference proxies to real research series."""
from pathlib import Path
import os

import numpy as np

from backend.research_series.service import ResearchSeriesService
from backend.research_series.numba_kernels import warm_research_series_numba_kernels
from backend.sensitivity.repository import file_hash
from backend.historical_regimes.data import resolve_target
from custom_indicators.errors import IndicatorDomainError as ResolverError
from backend.custom_indicators.errors import ValidationError


_WARMED_PID = None


class ReferenceSources:
    def __init__(self, data_dir: Path):
        self.series = ResearchSeriesService(data_dir, workspace_data_dir=data_dir)
        self.data_dir = Path(data_dir)

    def active_snapshot(self) -> Path:
        snapshot, _ = self.series._active_snapshot()
        return snapshot

    def warm(self):
        global _WARMED_PID
        result = warm_research_series_numba_kernels()
        # Existing raw resolver imports its product boundary via the direct package.
        from research_series.numba_kernels import warm_research_series_numba_kernels as warm_raw
        from backend.instrument_analytics_numba import count_true_kernel
        warm_raw()
        count_true_kernel(np.zeros(1, dtype=np.uint8))
        _WARMED_PID = os.getpid()
        return result

    def audit(self):
        return {'complete': _WARMED_PID == os.getpid(), 'warmed_pid': _WARMED_PID,
                'backend': 'existing_raw_resolver_fixed_njit', 'python_fallback': 0, 'request_time_compilation': 0}

    @staticmethod
    def _capability(item):
        kind = item.get('kind')
        fields = [x.get('name', x.get('id')) if isinstance(x, dict) else x for x in item.get('fields', [])
                  if not isinstance(x, dict) or x.get('available', True)]
        allowed = ('close',) if kind == 'index' else ('adj_nav', 'close_hfq') if kind in ('etf', 'fund') else ()
        fields = [field for field in fields if field in allowed]
        available = kind in ('index', 'etf', 'fund') and item.get('status') == 'available' and bool(fields)
        if available:
            reason = None
        elif kind == 'index':
            reason = '缺少可读取的指数值序列。'
        elif kind in ('etf', 'fund'):
            reason = '缺少可读取的复权价格或复权净值。'
        else:
            reason = '当前来源类型不支持作为风险标尺代理。'
        return {'available': available, 'supported_fields': fields, 'reason': reason,
                'selection_basis': 'daily_return_series_only'}

    def catalog(self, kind='index', query='', offset=0, limit=50):
        if _WARMED_PID != os.getpid():
            raise RuntimeError('REFERENCE_SOURCE_NOT_READY')
        payload = self.series.catalog(kind=kind, query=query, offset=offset, limit=limit)
        items = []
        for item in payload['items']:
            items.append({key: item.get(key) for key in ('id', 'kind', 'name', 'code', 'status', 'coverage')}
                         | {'reference_capability': self._capability(item)})
        return {'items': items, 'total': payload['total'], 'offset': offset, 'limit': limit, 'problems': []}

    def load(self, component, request, *, snapshot=None):
        code = component.series_id.split(':')[-1]
        catalog = self.series.catalog(kind=component.kind, query=code, offset=0, limit=200)
        item = next((x for x in catalog['items'] if x['id'] == component.series_id), None)
        capability = self._capability(item) if item is not None else {'available': False, 'supported_fields': []}
        if item is None or not capability['available'] or component.field not in capability['supported_fields']:
            raise ValidationError('REFERENCE_SOURCE_UNAVAILABLE', '来源或所需历史字段不可用，请在数据中心核验或更换来源。')
        if _WARMED_PID != os.getpid():
            raise RuntimeError('REFERENCE_SOURCE_NOT_READY')
        snapshot = snapshot or self.active_snapshot()
        field = next((x for x in item['fields'] if isinstance(x, dict) and x.get('name', x.get('id')) == component.field), {})
        binding = {**item.get('binding_parameters', {}), **field.get('binding_parameters', {}),
                   'kind': component.kind, 'field': component.field, 'ts_code': code,
                   'availability_mode': 'latest'}
        try:
            bundle = resolve_target(binding, 'retrospective', str(request.as_of), snapshot, resolved_data_dir=True)
        except ResolverError as exc:
            raise ValidationError(exc.code, '来源解析失败，请检查相应字段、覆盖区间和可得日期。') from exc
        frame = bundle.frame
        if len(frame) > 10000:
            raise ValidationError('REFERENCE_SOURCE_CAPACITY', '单一参考序列超过 10000 个观测，当前风险标尺不支持更长历史。')
        if 'availability_unknown' in frame and frame['availability_unknown'].any():
            raise ValidationError('REFERENCE_INFORMATION_CLOCK', '来源公告时间未知，无法冻结可得日期；请补齐来源证据。')
        known = [x.date().isoformat() for x in frame['available_at']]
        dates = [x.date().isoformat() for x in frame['observation_date']]
        if not known or len(known) != len(dates) or any(not x or x > str(request.as_of) for x in known):
            raise ValidationError('REFERENCE_INFORMATION_CLOCK', '来源可得日期未知或晚于研究日，无法冻结参考输入。')
        expected_checksum = binding.get('file_checksum')
        source_file = bundle.snapshot['file']
        actual_checksum = 'sha256:' + file_hash(snapshot / source_file)
        if expected_checksum != actual_checksum:
            raise ValidationError('REFERENCE_SOURCE_CHANGED', '数据源在读取期间变化，请重新检查来源。')
        values = np.asarray(frame['value'].to_numpy(dtype=np.float64), dtype=np.float64)
        values.flags.writeable = False
        return {'dates': dates, 'values': values, 'available_at': known,
                'identity': {'series_id': component.series_id, 'kind': component.kind, 'field': component.field,
                             'name': str(item.get('name') or code), 'code': str(item.get('code') or code),
                             'frequency': 'daily', 'content_hash': actual_checksum,
                             **{k: binding[k] for k in ('snapshot_id', 'snapshot_generation') if k in binding},
                             'adjustment_checksum': binding.get('adjustment_checksum')},
                'historical_pit_proven': False}
