"""Standalone frozen reference inputs and direct historical parameter orchestration for Risk Scale."""
from __future__ import annotations
import hashlib
import json
from datetime import date
import numpy as np
import pandas as pd

from backend.custom_indicators.errors import ConflictError, ValidationError
from backend.pit.context import view_override
from backend.sensitivity.repository import ArtifactRepository, digest_json
from backend.factor_research.repository import clean
from . import reference_evidence_kernels as evidence
from .reference_contracts import ReferenceInputRequest


def problem(code, message, field=None, action='请核验来源并重新预览。'):
    return {'code': code, 'message': message, 'field': field, 'suggested_action': action}


def array_digest(value):
    return hashlib.sha256(value.dtype.str.encode() + repr(value.shape).encode() + value.tobytes(order='C')).hexdigest()


def freeze_hash(payload):
    return {**payload, 'preview_hash': digest_json(payload)}


def automatic_research_day(data_dir):
    """Resolve PIT/today without creating PIT stores or lock files."""
    override = view_override()
    if override is not None:
        return date.fromisoformat(override.as_of) if override.as_of else date.today()
    settings_path = data_dir / 'pit_settings.json'
    if not settings_path.exists():
        return date.today()
    try:
        settings = json.loads(settings_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError('PIT_SETTINGS_UNREADABLE', 'PIT 设置无法只读解析，不能确定参考研究日。') from exc
    stated = str(settings.get('as_of') or '').strip()
    if stated:
        return date.fromisoformat(stated)
    release_id = str(settings.get('active_release_id') or '').strip()
    if not release_id:
        return date.today()
    release_path = data_dir / 'data_releases.json'
    if not release_path.exists():
        return date.today()
    try:
        payload = json.loads(release_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError('PIT_RELEASE_UNREADABLE', 'PIT 数据版本无法只读解析，不能确定参考研究日。') from exc
    release = next((item for item in payload.get('releases', []) if item.get('id') == release_id), None)
    if not release:
        return date.today()
    research_day = str(release.get('as_of') or (release.get('summary') or {}).get('available_through') or '').strip()
    return date.fromisoformat(research_day) if research_day else date.today()


def _rebalance_reset_flags(days: list[str], rule: str) -> np.ndarray:
    """Reset at the close of the last common-value date in each selected period."""
    rows = max(0, len(days) - 1)
    if rule == 'daily':
        return np.ones(rows, dtype=np.int64)
    if rule == 'buy_and_hold':
        return np.zeros(rows, dtype=np.int64)
    result = np.zeros(rows, dtype=np.int64)
    for t in range(rows):
        current, following = date.fromisoformat(days[t]), date.fromisoformat(days[t + 1])
        if rule == 'monthly':
            changed = (current.year, current.month) != (following.year, following.month)
        elif rule == 'quarterly':
            changed = (current.year, (current.month - 1) // 3) != (following.year, (following.month - 1) // 3)
        elif rule == 'yearly':
            changed = current.year != following.year
        else:
            raise ValidationError('REFERENCE_REBALANCE_RULE', '不支持的代理再平衡规则。')
        result[t] = int(changed)
    return result


def _validate_sse_calendar(snapshot_dir, observed_dates):
    path = snapshot_dir / "trade_day_df.parquet"
    if not path.is_file():
        raise ValidationError("REFERENCE_SSE_CALENDAR_REQUIRED", "缺少 SSE 交易日日历，无法证明日频参考样本连续。")
    try:
        calendar = pd.read_parquet(path, columns=["exchange", "cal_date", "is_open"])
    except (OSError, ValueError, KeyError) as exc:
        raise ValidationError("REFERENCE_SSE_CALENDAR_INVALID", "SSE 交易日日历无法读取，不能继续构建日频参考样本。") from exc
    calendar = calendar.loc[(calendar["exchange"].astype(str).str.upper() == "SSE")
                            & (pd.to_numeric(calendar["is_open"], errors="coerce") == 1)]
    raw_dates = calendar["cal_date"]
    if pd.api.types.is_datetime64_any_dtype(raw_dates):
        parsed = pd.to_datetime(raw_dates, errors="coerce").dt.normalize()
    else:
        compact = raw_dates.astype(str).str.replace("-", "", regex=False).str[:8]
        parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce").dt.normalize()
    observed = pd.DatetimeIndex(np.asarray(observed_dates).astype("datetime64[D]")).normalize().unique().sort_values()
    expected = pd.DatetimeIndex(parsed.dropna().unique()).sort_values()
    expected = expected[(expected >= observed[0]) & (expected <= observed[-1])]
    missing = expected.difference(observed)
    unexpected = observed.difference(expected)
    if len(missing) or len(unexpected):
        diagnostics = ([{"code": "missing_trading_day", "date": stamp.strftime("%Y-%m-%d")} for stamp in missing[:20]]
                       + [{"code": "unexpected_observation_day", "date": stamp.strftime("%Y-%m-%d")} for stamp in unexpected[:20]])
        raise ValidationError("REFERENCE_SSE_CALENDAR_GAP",
            f"共同参考样本与 SSE 开放日不连续：缺少 {len(missing)} 个开放日，含 {len(unexpected)} 个非开放日观察。",
            diagnostics=diagnostics)


def confirm_warnings(preview, acknowledged):
    required = {x['code'] for x in preview['warnings']}
    if not required.issubset(set(acknowledged)):
        raise ValidationError('WARNINGS_NOT_ACKNOWLEDGED', '请确认本次预览的全部限制与警告。', 'acknowledged_warnings')


class ReferenceInputs:
    def __init__(self, artifacts: ArtifactRepository, sources):
        self.artifacts, self.sources = artifacts, sources

    def _with_research_day(self, request: ReferenceInputRequest) -> ReferenceInputRequest:
        day = automatic_research_day(self.sources.data_dir)
        return request if request.as_of == day else request.model_copy(update={'as_of': day})

    def warm(self):
        self.sources.warm()
        evidence.warm()
        return evidence.audit()

    def get(self, reference, artifact_type=None, *, verify_arrays=True):
        item = self.artifacts.get(reference.id, 'series')
        if item['content_hash'] != reference.content_hash:
            raise ConflictError('REFERENCE_HASH_MISMATCH', '所选冻结来源与校验值不一致，请重新选择。')
        if artifact_type and item.get('artifact_type') != artifact_type:
            raise ValidationError('REFERENCE_TYPE_MISMATCH', '所选引用不是要求的冻结来源。')
        if verify_arrays:
            self.artifacts.arrays(item['id'])
        return item

    def version(self, identifier):
        item = self.artifacts.get(identifier, 'series')
        if item.get('artifact_type') != 'reference_inputs':
            raise ValidationError('REFERENCE_TYPE_MISMATCH', '不是风险标尺的冻结参考资产版本。')
        self.artifacts.arrays(identifier)
        return {**item['reference_preview'], **{k: item[k] for k in ('id', 'content_hash', 'created_at', 'artifact_type', 'immutable')}}

    def _input_calculation(self, request: ReferenceInputRequest):
        evidence.require_ready()
        source_cache = {}
        common_dates = None
        for asset in request.assets:
            if asset.asset_type == 'cash':
                continue
            for component in asset.components:
                source_key = digest_json(component.model_dump(exclude={'weight'}))
                if source_key not in source_cache:
                    source = self.sources.load(component, request)
                    dates = np.asarray(source['dates'], dtype='datetime64[D]').astype(np.int64)
                    if dates.ndim != 1 or dates.size < 21 or np.any(dates[1:] <= dates[:-1]):
                        raise ValidationError('REFERENCE_SOURCE_DATES', '参考序列日期不足、重复或未按时间递增。')
                    source_cache[source_key] = {**source, '_date_ints': dates}
                dates = source_cache[source_key]['_date_ints']
                common_dates = dates if common_dates is None else np.intersect1d(common_dates, dates, assume_unique=True)
        if common_dates is None or common_dates.size < 21:
            raise ValidationError('REFERENCE_INTERSECTION_TOO_SHORT', '所选非现金代理的历史数据交集不足 21 个观测日，请更换代理。')
        if common_dates.size > 10000:
            raise ValidationError('REFERENCE_INTERSECTION_TOO_LONG', '共同历史区间超过 10000 个观测日，当前风险标尺不支持更长历史。')
        _validate_sse_calendar(self.sources.active_snapshot(), common_dates)
        days = common_dates.astype('datetime64[D]').astype(str).tolist()
        panel = np.empty((common_dates.size - 1, len(request.assets)), dtype=np.float64)
        provenance = []
        information_clocks = []
        for j, asset in enumerate(request.assets):
            if asset.asset_type == 'cash':
                panel[:, j] = float(asset.cash_return) / request.periods_per_year
                information_clocks.append(str(request.as_of))
                provenance.append({'asset_id': asset.id, 'asset_type': 'cash', 'cash_return': float(asset.cash_return),
                    'sources': [], 'rebalance': None, 'information_available_at': str(request.as_of)})
                continue
            values = np.empty((common_dates.size, len(asset.components)), dtype=np.float64)
            identities = []
            for k, component in enumerate(asset.components):
                source_key = digest_json(component.model_dump(exclude={'weight'}))
                source = source_cache[source_key]
                source_dates = source['_date_ints']
                positions = np.searchsorted(source_dates, common_dates)
                if np.any(positions >= source_dates.size) or not np.array_equal(source_dates[positions], common_dates):
                    raise ValidationError('REFERENCE_ALIGNMENT', '参考序列无法按共同历史日期严格对齐。', f'assets.{j}.components.{k}')
                identity = source['identity']
                if (identity.get('kind') != component.kind or identity.get('series_id') != component.series_id
                    or identity.get('field') != component.field or identity.get('frequency') != request.frequency
                    or not identity.get('content_hash')):
                    raise ValidationError('REFERENCE_SOURCE_IDENTITY', '真实来源与请求的类型、口径或身份不一致。')
                selected_available = [source['available_at'][int(position)] for position in positions]
                if any(not value or value > str(request.as_of) for value in selected_available):
                    raise ValidationError('REFERENCE_INFORMATION_CLOCK', '来源可得日期未知或晚于参考研究日。')
                values[:, k] = np.asarray(source['values'], dtype=np.float64)[positions]
                identities.append({**identity, 'information_available_at': max(selected_available),
                    'historical_pit_proven': bool(source.get('historical_pit_proven', False))})
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValidationError('REFERENCE_MISSING_VALUES', '共同区间内价格/净值存在缺失或无效值，不会填充或重分配权重。', f'assets.{j}')
            values.flags.writeable = False
            component_returns = evidence.adjacent_returns(values)
            reset = _rebalance_reset_flags(days, asset.rebalance)
            result = evidence.proxy_returns(component_returns, np.asarray([x.weight for x in asset.components], dtype=np.float64), reset)
            if not np.isfinite(result).all():
                raise ValidationError('REFERENCE_PROXY_PATH', '代理持有路径中断，不会归一化剩余资产。', f'assets.{j}')
            panel[:, j] = result
            information_clocks.extend(x['information_available_at'] for x in identities)
            provenance.append({'asset_id': asset.id, 'asset_type': 'market', 'sources': identities, 'rebalance': asset.rebalance})
        panel.flags.writeable = False
        mean, covariance, observed_volatility, correlation = evidence.annual_moments(panel, 0., np.int64(252))
        fingerprints = {'return_panel': array_digest(panel), 'effective_returns': array_digest(mean),
                        'covariance': array_digest(covariance)}
        warnings = [problem('RETROSPECTIVE_REFERENCE', '非现金参考资产使用截至参考研究日可得的历史序列；结果是研究标尺，不代表未来收益。')]
        if any(asset.asset_type == 'cash' for asset in request.assets):
            warnings.append(problem('CASH_ASSUMPTION', '现金收益率由用户直接输入；现金波动率和与其它资产协方差按定义为 0。'))
        if any(component.kind == 'index' for asset in request.assets for component in asset.components):
            warnings.append(problem('INDEX_SERIES_SEMANTICS', '指数按用户选择的指数值序列直接计算；价格指数或全收益指数的经济含义由所选指数本身决定。'))
        quality = {'observed_annual_volatility': observed_volatility.tolist(),
                   'actual_start': days[0], 'data_as_of': days[-1],
                   'intersection_start': days[0], 'intersection_end': days[-1],
                   'observations': common_dates.size - 1, 'complete_intersection': True, 'missing': 0,
                   'historical_pit_proven': False, 'information_available_at': max(information_clocks),
                   'cash_asset_ids': [asset.id for asset in request.assets if asset.asset_type == 'cash'],
                   'unit': 'decimal', 'moment_period': 'annual', 'periods_per_year': 252,
                   'annualization_method': 'arithmetic_mean_and_covariance_times_periods',
                   'return_semantics': request.return_basis,
                   'quality_policy': 'strict-common-date-intersection-adjusted-products/3.0.0'}
        moments = {'annual_returns': mean.tolist(), 'covariance': covariance.tolist(),
                   'annual_volatilities': observed_volatility.tolist(), 'correlation': clean(correlation),
                   'moment_semantics': request.return_basis, 'moment_period': 'annual', 'unit': 'decimal'}
        payload = {'definition': request.model_dump(mode='json'), 'ordered_asset_ids': [a.id for a in request.assets],
                   'quality': quality, 'warnings': warnings, 'moments': moments,
                   'provenance': {'assets': provenance, 'array_hashes': fingerprints,
                                  'kernel_version': evidence.VERSION, 'kernel_fingerprint': evidence.audit()['fingerprint']}}
        arrays = {'returns': panel, 'dates': common_dates[1:], 'effective_returns': mean, 'covariance': covariance}
        return freeze_hash(payload), arrays

    def preview(self, request):
        return self._input_calculation(self._with_research_day(request))[0]

    def confirm(self, body):
        key = 'reference-input:' + body.idempotency_key
        effective_request = self._with_research_day(body.request)
        effective_body = body.model_copy(update={'request': effective_request})
        request_hash = digest_json(effective_body.model_dump(mode='json'))
        replay = self.artifacts.idempotent_result(key, request_hash)
        if replay:
            return self.version(replay['id'])
        preview, arrays = self._input_calculation(effective_request)
        if preview['preview_hash'] != body.preview_hash:
            raise ConflictError('PREVIEW_STALE', '来源或输入已变化，请重新检查参考资产。')
        confirm_warnings(preview, body.acknowledged_warnings)
        item = self.artifacts.save('series', {'artifact_type': 'reference_inputs', 'name': effective_request.name,
            'reference_preview': preview, 'definition': preview['definition'], 'base_currency': effective_request.currency,
            'as_of': str(effective_request.as_of)}, arrays, idempotency_key=key, request_hash=request_hash)
        return self.version(item['id'])
