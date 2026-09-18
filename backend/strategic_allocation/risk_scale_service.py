"""Risk scale orchestration; no private numerical implementation or market refit."""
from __future__ import annotations
from collections import OrderedDict
from contextlib import contextmanager
from datetime import date, timedelta
import threading
import time
import numpy as np
from backend import frontier_moments as frontier
from backend.custom_indicators.errors import ConflictError, IndicatorDomainError, ValidationError
from backend.sensitivity.repository import digest_json
from backend.factor_research.repository import clean
from . import risk_scale_kernels as numeric, reference_evidence_kernels as evidence
from . import risk_scale_registry as registry
from .common_contracts import FrozenRef
from .reference_inputs import problem, array_digest, confirm_warnings, automatic_research_day
from .risk_scale_contracts import PreviewResponse, PreviewRequest


class FrontierCache:
    """One bounded process-local owner of immutable small frontier arrays."""
    def __init__(self, max_bytes=registry.LIMITS['cache_bytes'], ttl=registry.LIMITS['cache_ttl_seconds']):
        self.items = OrderedDict()
        self.bytes = 0
        self.max_bytes, self.ttl = max_bytes, ttl
        self.hits = 0
        self.lock = threading.RLock()

    def get(self, key):
        with self.lock:
            now = time.monotonic()
            for k in list(self.items):
                if self.items[k][0] <= now:
                    self.bytes -= self.items.pop(k)[1]
            value = self.items.get(key)
            if value:
                self.items.move_to_end(key)
                self.hits += 1
                return value[2]
            return None

    def put(self, key, value):
        size = sum(a.nbytes for grid in value for a in grid if isinstance(a, np.ndarray))
        for grid in value:
            for a in grid:
                if isinstance(a, np.ndarray):
                    a.flags.writeable = False
        with self.lock:
            if size > self.max_bytes:
                return
            if key in self.items:
                self.bytes -= self.items.pop(key)[1]
            while self.items and (self.bytes + size > self.max_bytes or len(self.items) >= registry.LIMITS['cache_entries']):
                self.bytes -= self.items.popitem(last=False)[1][1]
            self.items[key] = (time.monotonic() + self.ttl, size, value)
            self.bytes += size


class RiskScaleService:
    def __init__(self, artifacts, store, references):
        self.artifacts, self.store, self.references = artifacts, store, references
        self.cache = FrontierCache()
        self._slots = threading.BoundedSemaphore(registry.LIMITS['concurrent_computations'])

    def warm(self):
        self.references.warm()
        frontier.warm()
        numeric.warm()
        return self.execution()

    def execution(self):
        parts = {'frontier': frontier.execution_audit(), 'segmentation': numeric.execution_audit(), 'reference': evidence.audit(), 'sources': self.references.sources.audit()}
        return {'complete': all(x['complete'] for x in parts.values()), 'backend': 'numba_njit_fixed_signature',
                'python_fallback': 0, 'request_time_compilation': 0, **parts}

    @contextmanager
    def compute_slot(self):
        if not self.execution()['complete']:
            raise IndicatorDomainError('COMPUTE_NOT_READY', '本进程计算预热尚未完成，请稍后重试。', status_code=503)
        if not self._slots.acquire(blocking=False):
            raise IndicatorDomainError('COMPUTE_BUSY', '计算容量暂满，请稍后重试；只读查询仍可用。', status_code=503)
        try:
            yield
        finally:
            self._slots.release()

    def reference_call(self, operation, request):
        with self.compute_slot():
            return getattr(self.references, operation)(request)

    def capabilities(self):
        execution = self.execution()
        return {'ready': execution['complete'], 'execution': execution, 'algorithms': registry.algorithms(),
            'trust_mode': 'single_local_trusted_workspace', 'limits': registry.LIMITS,
            'reference': {'frequencies': ['daily'],
                'fields': {'index': ['close'], 'etf': ['close_hfq', 'adj_nav'], 'fund': ['adj_nav']},
                'parameter_method': 'historical_common_intersection',
                'historical_pit_claim': False, 'risk_unit': 'decimal',
                'source_selection': 'daily_return_series_only',
                'source_basis': 'index close = selected index level; product close_hfq/adj_nav = adjusted return proxy'},
            'templates': [{'id': 'multi-asset', 'name': '多资产参考模板', 'status': 'unconfigured',
                           'assets': [], 'assumptions': None, 'published_default': False}]}

    def _draft(self, request, state=None):
        if not request.draft_id:
            return
        state = self.store.read() if state is None else state
        draft = next((x for x in state['drafts'] if x['id'] == request.draft_id), None)
        if draft is None or draft['revision'] != request.draft_revision:
            raise ConflictError('REVISION_CONFLICT', '草稿已修改或删除，请重新加载。')
        # Revision identifies the saved edit base; preview_hash binds current unsaved edits.
        if draft['scheme_id'] != request.definition.scheme_id:
            raise ConflictError('DRAFT_SCHEME_CHANGED', '提交方案与所选草稿不一致，请重新加载。')

    def _research_day(self):
        return automatic_research_day(self.references.sources.data_dir)

    def _resolve(self, definition, *, current=False, state=None):
        source = self.references.get(definition.reference_input_ref, 'reference_inputs', verify_arrays=False)
        raw = source['definition']
        q = source['reference_preview']['quality']
        ids = source['reference_preview']['ordered_asset_ids']
        if (raw.get('currency') != definition.base_currency
                or raw.get('return_basis') != 'selected_index_and_adjusted_product_total_return'):
            raise ValidationError('REFERENCE_BASIS_MISMATCH', '参考资产与风险标尺的本位币或收益语义不兼容。')
        if not raw.get('as_of') or raw['as_of'] > str(definition.research_as_of):
            raise ValidationError('REFERENCE_INFORMATION_CLOCK', '参考资产版本晚于本次参考研究日。')
        if not 1 <= len(ids) <= registry.LIMITS['assets'] or len(set(ids)) != len(ids):
            raise ValidationError('REFERENCE_ASSET_AXIS', '冻结参考资产轴无效。')
        state = state if state is not None else self.store.read()
        clock = str(date.today() if current else definition.research_as_of)
        retired_refs = {x.get('release_id') for x in self.artifacts.list('retirement') if x.get('created_at', '')[:10] <= clock}
        if source['id'] in state['retired'] or source['id'] in retired_refs or source.get('retired'):
            raise ValidationError('REFERENCE_RETIRED', '参考资产版本已停用，不能新引用。')
        arrays = self.artifacts.arrays(source['id'])
        required = ('returns', 'dates', 'effective_returns', 'covariance')
        if any(name not in arrays for name in required):
            raise ValidationError('REFERENCE_ARRAY_IDENTITY', '冻结参考资产缺少完整历史参数数组。')
        panel, dates = arrays['returns'], arrays['dates']
        mu, covariance = arrays['effective_returns'], arrays['covariance']
        moments = source['reference_preview'].get('moments') or {}
        if (panel.dtype != np.dtype('float64') or panel.ndim != 2 or panel.shape[1] != len(ids)
                or not 20 <= panel.shape[0] < 10000 or dates.dtype != np.dtype('int64')
                or dates.shape != (panel.shape[0],) or q.get('observations') != panel.shape[0]
                or q.get('periods_per_year') != 252 or not np.isfinite(panel).all()
                or np.any(panel <= -1) or np.any(dates[1:] <= dates[:-1])
                or mu.shape != (len(ids),) or covariance.shape != (len(ids), len(ids))
                or not np.isfinite(mu).all() or not np.isfinite(covariance).all()
                or not np.array_equal(mu, np.asarray(moments.get('annual_returns'), dtype=np.float64))
                or not np.array_equal(covariance, np.asarray(moments.get('covariance'), dtype=np.float64))):
            raise ValidationError('REFERENCE_FROZEN_MOMENTS', '冻结参考收益、日期或历史参数不一致。')
        hashes = source['reference_preview']['provenance']['array_hashes']
        if (array_digest(panel) != hashes.get('return_panel') or array_digest(mu) != hashes.get('effective_returns')
                or array_digest(covariance) != hashes.get('covariance')):
            raise ValidationError('REFERENCE_PANEL_HASH', '冻结参考数据与参数指纹不一致。')
        if q.get('information_available_at') is None or q['information_available_at'] > str(definition.research_as_of):
            raise ValidationError('REFERENCE_INFORMATION_CLOCK', '参考数据在本次参考研究日尚不可得。')
        assets = []
        for index, asset in enumerate(raw['assets']):
            assets.append({**asset, 'annual_return': float(mu[index]),
                'annual_volatility': float(np.sqrt(max(0.0, covariance[index, index]))), 'mean_uncertainty': 0.0})
        parameters = {'currency': raw['currency'], 'return_basis': raw['return_basis'], 'as_of': raw['as_of'],
            'assets': assets, 'method_identity': {'id': 'historical_common_intersection', 'version': evidence.VERSION,
                'intersection_start': q.get('intersection_start'), 'intersection_end': q.get('intersection_end'),
                'observations': q.get('observations')}}
        mu, covariance = mu.view(), covariance.view()
        mu.flags.writeable = covariance.flags.writeable = False
        source = {**source, '_dates': dates}
        return source, parameters, mu, covariance, source, panel

    @staticmethod
    def _constraints(definition, ids):
        c = definition.constraint_profile
        if set(c.asset_limits) - set(ids):
            raise ValidationError('CONSTRAINT_ASSET_UNKNOWN', '权重约束含有不属于冻结资产轴的标识。')
        bounds = np.asarray([[c.asset_limits[x].min_weight, c.asset_limits[x].max_weight] if x in c.asset_limits else [0., 1.] for x in ids], dtype=np.float64)
        groups = np.zeros((len(c.group_limits), len(ids)), dtype=np.float64)
        seen = set()
        for i, g in enumerate(c.group_limits):
            if g.id in seen or len(set(g.assets)) != len(g.assets) or not g.assets or set(g.assets) - set(ids) or g.lo > g.hi:
                raise ValidationError('CONSTRAINT_GROUP_INVALID', '分组约束的标识、成员或上下界无效。')
            seen.add(g.id)
            for member in g.assets:
                groups[i, ids.index(member)] = 1.
        return bounds, groups, np.asarray([g.lo for g in c.group_limits], dtype=np.float64), np.asarray([g.hi for g in c.group_limits], dtype=np.float64)

    def _default_blockers(self, definition, source, raw, *, clock=None):
        blockers = []
        if source is None:
            blockers.append(problem('CASH_ANCHOR_REQUIRED', '系统默认标尺须来自冻结参考资产，并包含明确的纯现金零风险锚。'))
            return blockers
        observed = source['reference_preview']['quality'].get('observed_annual_volatility', [])
        cash = [i for i, asset in enumerate(source['definition']['assets']) if asset.get('asset_type') == 'cash']
        qualified = [i for i in cash if i < len(raw['assets']) and i < len(observed)
                     and raw['assets'][i].get('asset_type') == 'cash'
                     and abs(float(raw['assets'][i].get('annual_volatility', np.nan))) <= numeric.BOUNDARY_TOL
                     and abs(float(observed[i])) <= numeric.BOUNDARY_TOL
                     and abs(float(raw['assets'][i].get('annual_return', np.nan))
                             - float(source['definition']['assets'][i].get('cash_return', np.nan))) <= numeric.BOUNDARY_TOL]
        if len(qualified) != 1:
            blockers.append(problem('CASH_ANCHOR_INVALID', '系统默认标尺须且只能包含一个有效现金大类；现金波动率与协方差必须为 0。'))
        return blockers

    def _preview(self, request):
        effective_day = self._research_day()
        if request.definition.research_as_of != effective_day:
            request = request.model_copy(update={'definition': request.definition.model_copy(update={'research_as_of': effective_day})})
        self._draft(request)
        d = request.definition
        reference, raw, mu, covariance, source, panel = self._resolve(d)
        ids = [a['id'] for a in raw['assets']]
        constraints = self._constraints(d, ids)
        key = digest_json({'reference': d.reference_input_ref.model_dump(), 'mu': array_digest(mu), 'covariance': array_digest(covariance),
            'constraints': d.constraint_profile.model_dump(), 'currency': d.base_currency, 'basis': d.risk_basis_id,
            'as_of': str(d.research_as_of), 'frontier': frontier.VERSION,
            'fingerprint': frontier.execution_audit()['fingerprint'], 'primary': registry.PRIMARY_POINTS,
            'dense': registry.STABILITY_POINTS, 'iterations': registry.LIMITS['max_iterations']})
        grids = self.cache.get(key)
        numeric_error = None
        if grids is None:
            try:
                grids = tuple(frontier.solve_frontier(mu, covariance, *constraints, point_count=count,
                    max_iterations=registry.LIMITS['max_iterations']) for count in (registry.PRIMARY_POINTS, registry.STABILITY_POINTS))
                self.cache.put(key, grids)
            except (ValueError, np.linalg.LinAlgError):
                numeric_error = problem('FRONTIER_INPUT_INVALID', '冻结风险矩或约束无法形成可验证前沿；不会修补矩阵或放宽约束。')
        warnings = [problem('RESEARCH_SCALE', 'C1–C5 是本工作区研究标尺，不是跨机构适当性评级。')]
        blockers = [numeric_error] if numeric_error else []
        default_blockers = self._default_blockers(d, source, raw)
        result = {'ordered_asset_ids': ids, 'frontier': [], 'levels': [], 'applied_boundaries': [],
                  'algorithm_id': d.segmentation.algorithm_id, 'algorithm_version': numeric.VERSION,
                  'stability': {}, 'parameter_evidence': {'method_identity': raw.get('method_identity'),
                      'as_of': raw['as_of'], 'currency': raw['currency'], 'return_basis': raw['return_basis'],
                      'data_quality': source['reference_preview']['quality'],
                      'provenance': reference.get('reference_preview', {}).get('provenance'),
                      'historical_pit_proven': False}, 'diagnostics': {'frontier_key': key, 'frontier_version': frontier.VERSION,
                      'segmentation_parameters': registry.algorithms()[0]['parameters'],
                      'asset_names': {a['id']: a.get('name', a['id']) for a in raw['assets']}}, 'fallback_reason': None}
        arrays = {}
        if grids is not None:
            base, dense = grids
            manual = None if d.segmentation.manual_caps is None else np.asarray(d.segmentation.manual_caps, dtype=np.float64)
            seg = numeric.segment_frontier(base[2], base[3], d.segmentation.algorithm_id, manual_caps=manual)
            dense_seg = numeric.segment_frontier(dense[2], dense[3], d.segmentation.algorithm_id, manual_caps=manual)
            algorithm_seg, algorithm_dense_seg = seg, dense_seg
            boundary_adjusted = d.segmentation.adjusted_caps is not None
            if boundary_adjusted and seg['status'] == 0:
                adjusted = np.asarray(d.segmentation.adjusted_caps, dtype=np.float64)
                seg = numeric.segment_frontier(base[2], base[3], 'manual_volatility_bands_v1', manual_caps=adjusted)
                dense_seg = numeric.segment_frontier(dense[2], dense[3], 'manual_volatility_bands_v1', manual_caps=adjusted)
            result['frontier'] = [{'node_id': i, 'volatility': clean(base[2][i, 0]), 'expected_return': clean(base[2][i, 1]),
                'weights': clean(base[1][i]) if base[3][i] == 0 else None, 'status': frontier.STATUS_NAMES[int(base[3][i])]}
                for i in range(base[3].size)]
            result['diagnostics'].update({'phase_status': frontier.STATUS_NAMES[int(base[11])],
                'point_statuses': base[3].tolist(), 'endpoint_statuses': base[8].tolist(),
                'endpoint_metrics': clean(base[7]), 'point_diagnostics': clean(base[5]), 'constraint_bounds': constraints[0].tolist(),
                'curve_status': int(seg['curve_status']), 'geometry_status': int(seg['geometry_status']),
                'boundary_adjusted': boundary_adjusted,
                'algorithm_boundaries': algorithm_seg['risk_caps'].tolist() if algorithm_seg['status'] == 0 else None})
            if base[11] != 0 or any(x != 0 for x in base[3]) or any(x != 0 for x in dense[3]):
                blockers.append(problem('FRONTIER_UNVERIFIED', '前沿存在未验证点或求解失败，不能发布；请查看诊断。'))
            if seg['status'] != 0:
                blockers.append(problem(numeric.SEGMENTATION_STATUS[int(seg['status'])].upper(), '当前前沿不能形成五个有效自动区间。'))
            else:
                caps = seg['risk_caps']
                result['applied_boundaries'] = caps.tolist()
                result['fallback_reason'] = algorithm_seg['fallback_reason']
                if algorithm_seg['fallback_reason']:
                    warnings.append(problem('NEAR_LINEAR_FRONTIER', '形状算法检测到近直线，按其内部规则采用弧长切分。'))
                if d.segmentation.algorithm_id == 'manual_volatility_bands_v1':
                    result['stability'] = {'status': 'manual_policy_thresholds', 'maximum_normalized_shift': None}
                    warnings.append(problem('MANUAL_POLICY_THRESHOLDS', '人工阈值依据由研究者提供，超出参考范围的部分未校准。'))
                elif boundary_adjusted:
                    if algorithm_dense_seg['status'] == 0:
                        shift, status = numeric.boundary_stability_kernel(algorithm_seg['risk_caps'], algorithm_dense_seg['risk_caps'],
                            base[7][[1, 3]].reshape(4), dense[7][[1, 3]].reshape(4), registry.ENDPOINT_TOLERANCE, registry.STABILITY_TOLERANCE)
                        result['stability'] = {'status': 'manual_adjustment', 'maximum_normalized_shift': None,
                            'algorithm_stability_status': ('stable', 'unstable_calibration', 'endpoint_mismatch', 'degenerate')[status],
                            'algorithm_maximum_normalized_shift': clean(shift),
                            'algorithm_dense_boundaries': algorithm_dense_seg['risk_caps'].tolist()}
                        if status:
                            warnings.append(problem('UNSTABLE_CALIBRATION', '原始算法的 101/200 点复核未稳定，请核对后再采用人工微调边界。'))
                            if status >= 2:
                                blockers.append(problem('ENDPOINT_MISMATCH', '两次前沿端点不一致，无法完成稳定性验证。'))
                    else:
                        blockers.append(problem('STABILITY_UNAVAILABLE', '200 点分档验证失败，当前自动标尺不能发布。'))
                    warnings.append(problem('BOUNDARY_ADJUSTED', '风险档位边界已在算法结果上人工微调；代表组合、收益和风险画像已按调整后的前开后闭区间重新计算。'))
                elif dense_seg['status'] == 0:
                    shift, status = numeric.boundary_stability_kernel(caps, dense_seg['risk_caps'],
                        base[7][[1, 3]].reshape(4), dense[7][[1, 3]].reshape(4), registry.ENDPOINT_TOLERANCE, registry.STABILITY_TOLERANCE)
                    result['stability'] = {'status': ('stable', 'unstable_calibration', 'endpoint_mismatch', 'degenerate')[status],
                                          'maximum_normalized_shift': clean(shift), 'dense_boundaries': dense_seg['risk_caps'].tolist()}
                    if status:
                        warnings.append(problem('UNSTABLE_CALIBRATION', '101/200 点复核未稳定，请核对差异后再采用。'))
                        if status >= 2:
                            blockers.append(problem('ENDPOINT_MISMATCH', '两次前沿端点不一致，无法完成稳定性验证。'))
                else:
                    blockers.append(problem('STABILITY_UNAVAILABLE', '200 点分档验证失败，当前自动标尺不能发布。'))
                resets = None
                if panel is not None:
                    texts = source['_dates'].astype('datetime64[D]').astype(str).tolist()
                    start = source['reference_preview']['quality']['intersection_start']
                    previous = [start, *texts[:-1]]
                    resets = np.asarray([int(a[:7] != b[:7]) for a,b in zip(previous, texts)], dtype=np.int64)
                for i, cap in enumerate(caps):
                    node = int(seg['representative_node_indices'][i])
                    available = node >= 0 and base[3][node] == 0
                    missing = {'value': None, 'status': 'unavailable', 'reason': '该档内没有有效参考组合。', 'unit': 'decimal'}
                    metric = lambda v: {'value': float(v), 'status': 'available', 'reason': None, 'unit': 'annual_decimal'}
                    portrait = {'value': None, 'status': 'unavailable', 'reason': '没有有效真实冻结历史面板。', 'unit': 'daily_decimal'}
                    mdd = {**portrait, 'unit': 'path_drawdown_decimal'}
                    if available and panel is not None:
                        replay = evidence.proxy_returns(panel, base[1][node], resets)
                        _, es, _, status = numeric.historical_tail_kernel(replay, .95, 5.)
                        loss, loss_status = numeric.historical_drawdown_kernel(replay)
                        portrait = {'value': clean(es) if status == 0 else None, 'status': 'available' if status == 0 else 'unavailable',
                            'reason': None if status == 0 else '经验尾部样本不足或历史路径无效。', 'unit': 'daily_es_95_decimal'}
                        mdd = {'value': clean(loss), 'status': 'available' if loss_status == 0 else 'unavailable',
                            'reason': None if loss_status == 0 else '历史路径无效。', 'unit': 'path_drawdown_decimal'}
                    result['levels'].append({'level_code': 'C'+str(i+1), 'lower_bound': float(caps[i-1]) if i else 0.,
                        'lower_inclusive': i == 0, 'upper_bound': float(cap), 'upper_inclusive': True,
                        'authorized_volatility_cap': float(cap), 'calibration_status': 'not_calibrated' if seg['not_calibrated'][i] or not available else 'calibrated',
                        'representative_node_id': node if available else None,
                        'representative_weights': base[1][node].tolist() if available else None,
                        'expected_return': metric(base[2][node, 1]) if available else missing,
                        'volatility': metric(base[2][node, 0]) if available else missing,
                        'historical_es': portrait if available else missing, 'historical_mdd': mdd if available else missing})
            arrays = {'frontier_weights': base[1], 'frontier_metrics': base[2], 'frontier_statuses': base[3],
                      'frontier_diagnostics': base[5], 'dense_metrics': dense[2], 'dense_statuses': dense[3],
                      'endpoint_metrics': base[7], 'endpoint_statuses': base[8], 'target_returns': base[0],
                      'dense_weights': dense[1], 'dense_diagnostics': dense[5], 'dense_endpoint_metrics': dense[7],
                      'dense_endpoint_statuses': dense[8], 'dense_target_returns': dense[0]}
        if panel is not None:
            observed = source['reference_preview']['quality']['observed_annual_volatility']
            cash = [i for i, asset in enumerate(source['definition']['assets']) if asset.get('asset_type') == 'cash'
                    and i < len(observed) and abs(float(observed[i])) <= numeric.BOUNDARY_TOL
                    and raw['assets'][i].get('asset_type') == 'cash'
                    and abs(float(raw['assets'][i]['annual_volatility'])) <= numeric.BOUNDARY_TOL]
            result['diagnostics']['cash_asset_ids'] = [ids[i] for i in cash]
            if cash and grids is not None and (grids[0][3][0] != 0 or grids[0][2][0, 0] > numeric.BOUNDARY_TOL):
                default_blockers.append(problem('CASH_CONSTRAINT_COVERAGE', '当前参考约束排除了零风险现金端点，不能设为全范围系统默认标尺。'))
            warnings.append(problem('RETROSPECTIVE_PORTRAITS', '非现金代表组合按各代理配置的再平衡规则回放，仅是事后画像；ES 为日频经验值，不是年度损失保证。'))
        default_blockers.extend(self._stability_default_blockers(result))
        payload = {'request_echo': request.model_dump(mode='json'),
            'resolved_refs': {'reference_inputs': d.reference_input_ref.model_dump()},
            'data_fingerprints': {'reference_inputs': reference['content_hash'], 'means': array_digest(mu), 'covariance': array_digest(covariance),
                                  **{'result.' + name: array_digest(value) for name, value in arrays.items()}},
            'result': clean(result), 'mathematical_status': 'blocked' if blockers else 'valid',
            'publication_eligibility': {'eligible': not blockers, 'blockers': blockers},
            'default_eligibility': {'eligible': not blockers and not default_blockers, 'blockers': blockers + default_blockers},
            'warnings': warnings, 'limitations': ['受信任的单一本地工作区；未提供用户/租户隔离。', '预测均值与风险不是投资回报承诺。'],
            'execution_audit': self.execution()}
        stable = {k: v for k,v in payload.items() if k != 'execution_audit'}
        stable['versions'] = {'service': registry.VERSION, 'numeric': numeric.VERSION, 'evidence': evidence.VERSION, 'evidence_fingerprint': evidence.audit()['fingerprint'],
                              'frontier_fingerprint': frontier.execution_audit()['fingerprint'], 'segmentation_fingerprint': numeric.execution_audit()['fingerprint']}
        payload['preview_hash'] = digest_json(stable)
        return PreviewResponse.model_validate(payload).model_dump(mode='json'), arrays

    def preview(self, request):
        with self.compute_slot():
            return self._preview(request)[0]

    def confirm(self, body):
        effective_request = body.request.model_copy(update={'definition': body.request.definition.model_copy(update={'research_as_of': self._research_day()})})
        effective_body = body.model_copy(update={'request': effective_request})
        request_hash = digest_json(effective_body.model_dump(mode='json'))
        key = 'risk-scale:' + body.idempotency_key
        replay = self.artifacts.idempotent_result(key, request_hash)
        if replay:
            return self.get_version(replay['id'])
        with self.compute_slot():
            preview, arrays = self._preview(effective_request)
            if preview['preview_hash'] != body.preview_hash:
                raise ConflictError('PREVIEW_STALE', '预览与当前输入或算法不一致，请重新预览。')
            if not preview['publication_eligibility']['eligible']:
                raise ValidationError('PUBLICATION_BLOCKED', '当前研究仍有阻止发布的问题。')
            confirm_warnings(preview, body.acknowledged_warnings)
            resolved_request = PreviewRequest.model_validate(preview['request_echo'])
            d = resolved_request.definition
            versions = [x for x in self.artifacts.list('series') if x.get('artifact_type') == 'risk_scale' and x.get('scheme_id') == d.scheme_id]
            version_number = self.store.reserve_version(d.scheme_id, d.name, digest_json(key), request_hash,
                                                       max([x['version_number'] for x in versions] or [0]))
            with self.store.transaction() as state:
                self._draft(body.request, state)
                self._resolve(d, current=True, state=state)
                item = self.artifacts.save('series', {'artifact_type': 'risk_scale', 'name': d.name,
                    'scheme_id': d.scheme_id, 'version_number': version_number,
                    'base_currency': d.base_currency, 'risk_basis_id': d.risk_basis_id,
                    'research_as_of': str(d.research_as_of), 'review_due_at': str(d.review_due_at) if d.review_due_at else None,
                    'preview': preview, 'acknowledged_warnings': body.acknowledged_warnings, 'research_only': True}, arrays,
                    idempotency_key=key, request_hash=request_hash)
                state['schemes'].setdefault(d.scheme_id, {'name': d.name, 'revision': 1})
        return self.get_version(item['id'])

    def _item(self, identifier):
        item = self.artifacts.get(identifier, 'series')
        if item.get('artifact_type') != 'risk_scale':
            raise ValidationError('RISK_SCALE_VERSION_TYPE', '所选版本不是风险标尺。')
        self.artifacts.arrays(identifier)
        return item

    @staticmethod
    def _stability_default_blockers(result):
        stability = result.get('stability') or {}
        if (stability.get('status') == 'unstable_calibration'
                or stability.get('algorithm_stability_status') == 'unstable_calibration'):
            return [problem('UNSTABLE_CALIBRATION', '101/200 点分档复核不稳定，不能设为系统默认；请调整研究输入并重新验证。')]
        return []

    def _current(self, item, state=None, default=False):
        state = self.store.read() if state is None else state
        blockers = []
        d = PreviewRequest.model_validate(item['preview']['request_echo']).definition
        if item['id'] in state['retired']:
            blockers.append(problem('VERSION_RETIRED', '版本已退休，仅供历史读取。'))
        if d.valid_until and d.valid_until <= date.today():
            blockers.append(problem('VERSION_EXPIRED', '版本已超过硬失效日。'))
        try:
            _, raw, _, _, source, _ = self._resolve(d, current=True, state=state)
            if default:
                blockers.extend(self._default_blockers(d, source, raw, clock=date.today()))
                blockers.extend(item['preview']['default_eligibility']['blockers'])
                # Older immutable versions may have recorded instability only as a warning.
                blockers.extend(self._stability_default_blockers(item['preview']['result']))
        except IndicatorDomainError:
            blockers.append(problem('FROZEN_REFERENCE_UNAVAILABLE', '冻结参考来源不可读、失效或校验失败。'))
        return {'eligible': not blockers, 'blockers': blockers}

    @staticmethod
    def _review_status(review_due_at):
        if not review_due_at:
            return 'none'
        due = date.fromisoformat(str(review_due_at)[:10])
        remaining = (due - date.today()).days
        if remaining <= 0:
            return 'due'
        if remaining <= 30:
            return 'upcoming'
        return 'scheduled'

    def get_version(self, identifier):
        item = self._item(identifier)
        state = self.store.read()
        definition = PreviewRequest.model_validate(item['preview']['request_echo']).definition
        return {**{k: item[k] for k in ('id', 'name', 'artifact_type', 'content_hash', 'scheme_id', 'version_number', 'created_at', 'immutable', 'preview')},
                'current_eligibility': self._current(item, state), 'retired': identifier in state['retired'],
                'current_default_eligibility': self._current(item, state, default=True),
                'review_due_at': definition.review_due_at, 'review_status': self._review_status(definition.review_due_at)}

    def catalog(self, offset=0, limit=50, base_currency=None, risk_basis_id=None, include_retired=False):
        state = self.store.read()
        items = [x for x in self.artifacts.list('series') if x.get('artifact_type') == 'risk_scale'
                 and (include_retired or x.get('id') not in state['retired'])
                 and (base_currency is None or x.get('base_currency') == base_currency)
                 and (risk_basis_id is None or x.get('risk_basis_id') == risk_basis_id)]
        page = []
        for item in items[offset:offset+limit]:
            due = item.get('review_due_at')
            if 'review_due_at' not in item:
                # Older index summaries predate review metadata. Read the immutable
                # manifest only for those rows; never rewrite historical artifacts.
                frozen = self.artifacts.get(item['id'], 'series')
                due = frozen.get('review_due_at')
                if due is None:
                    due = PreviewRequest.model_validate(frozen['preview']['request_echo']).definition.review_due_at
            page.append({**item, 'review_due_at': due, 'review_status': self._review_status(due), 'retired': item['id'] in state['retired']})
        return {'items': page, 'drafts': state['drafts'], 'total': len(items),
                'next_offset': offset+limit if offset+limit < len(items) else None}

    def study_options(self, as_of):
        """List immutable scales that were valid at one research date; never refit them."""
        study_day = date.fromisoformat(str(as_of))
        if study_day > date.today():
            raise ValidationError('RISK_SCALE_STUDY_DATE', '目标研究日不能位于未来。')
        state = self.store.read()
        items = []
        for summary in self.artifacts.list('series'):
            if summary.get('artifact_type') != 'risk_scale':
                continue
            item = self._item(summary['id'])
            definition = PreviewRequest.model_validate(item['preview']['request_echo']).definition
            if definition.risk_basis_id != 'annualized-periodic-volatility-v1' or definition.research_as_of > study_day:
                continue
            if definition.valid_until and study_day >= definition.valid_until:
                continue
            retirement = state['retired'].get(item['id'])
            if retirement and retirement.get('retired_on') and retirement['retired_on'] <= str(study_day):
                continue
            if not item['preview']['publication_eligibility']['eligible']:
                continue
            try:
                self.frozen_context(item['id'], str(study_day))
            except IndicatorDomainError:
                continue
            items.append({'id': item['id'], 'name': item['name'], 'content_hash': item['content_hash'],
                          'version_number': item['version_number'], 'base_currency': definition.base_currency,
                          'risk_basis_id': definition.risk_basis_id, 'research_as_of': definition.research_as_of,
                          'valid_until': definition.valid_until})
        items.sort(key=lambda row: (str(row['research_as_of']), row['version_number'], row['id']), reverse=True)
        return {'as_of': study_day, 'items': items}

    def defaults(self):
        state = self.store.read()
        items = []
        for value in state['defaults'].values():
            eligibility = None
            if value['version_id']:
                try:
                    item = self._item(value['version_id'])
                    if item['content_hash'] != value['content_hash']:
                        raise ValueError('POINTER_HASH')
                    eligibility = self._current(item, state, default=True)
                except (IndicatorDomainError, ValueError):
                    eligibility = {'eligible': False, 'blockers': [problem('DEFAULT_UNAVAILABLE', '默认版本不可读或已不具备引用资格。')]}
            items.append({**value, 'eligibility': eligibility})
        return {'items': items}

    def activate(self, identifier, body):
        with self.store.transaction() as state:
            item = self._item(identifier)
            eligibility = self._current(item, state, default=True)
            if not eligibility['eligible']:
                raise ValidationError('DEFAULT_INELIGIBLE', '此版本不满足默认资格，请核对参考锚、来源与有效日期。')
            key = self.store.key(item['base_currency'], item['risk_basis_id'])
            binding = self.store.set_default(state, key, item, body.expected_revision)
        return binding

    def retire(self, identifier, body):
        with self.store.transaction() as state:
            item = self._item(identifier)
            key = self.store.key(item['base_currency'], item['risk_basis_id'])
            previous = self.store.binding(state, key)
            if previous['revision'] != body.expected_revision:
                raise ConflictError('DEFAULT_CHANGED', '默认修订已变化，请重新加载。')
            if identifier in state['retired']:
                raise ConflictError('VERSION_ALREADY_RETIRED', '此版本已经退休。')
            if previous['version_id'] == identifier:
                if not body.clear_default and not body.replacement_id:
                    raise ValidationError('DEFAULT_RETIRE_ACTION_REQUIRED', '退休当前默认须明确清空或指定替代版本。')
                replacement = self._item(body.replacement_id) if body.replacement_id else None
                if replacement and (replacement['id'] == identifier or self.store.key(replacement['base_currency'], replacement['risk_basis_id']) != key
                        or not self._current(replacement, state, default=True)['eligible']):
                    raise ValidationError('DEFAULT_REPLACEMENT_INVALID', '替代版本须同口径且符合默认资格。')
                binding = self.store.set_default(state, key, replacement, body.expected_revision, 'retire', body.reason)
            else:
                if body.replacement_id or body.clear_default:
                    raise ValidationError('RETIRE_NONDEFAULT_ACTION', '此版本不是当前默认，无需清空或替换默认。')
                binding = previous
            state['retired'][identifier] = {'reason': body.reason, 'retired_on': str(date.today())}
            state['history'].append({'action': 'retirement', 'version_id': identifier, **state['retired'][identifier]})
        return {**binding, 'eligibility': None}

    def compare(self, body):
        left, right = self.get_version(body.left_id), self.get_version(body.right_id)
        ld, rd = left['preview']['request_echo']['definition'], right['preview']['request_echo']['definition']
        compatible = (ld['base_currency'], ld['risk_basis_id']) == (rd['base_currency'], rd['risk_basis_id'])
        differences = [k for k in ld if ld[k] != rd[k]]
        if left['preview']['result']['parameter_evidence'] != right['preview']['result']['parameter_evidence']:
            differences.append('historical_parameters_or_provenance')
        delta = None
        if compatible:
            with self.compute_slot():
                delta = evidence.boundary_difference(np.asarray(left['preview']['result']['applied_boundaries'], dtype=np.float64),
                    np.asarray(right['preview']['result']['applied_boundaries'], dtype=np.float64)).tolist()
        return {'compatible': compatible, 'left': left, 'right': right, 'boundary_differences': delta, 'differences': differences}

    def frozen_context(self, identifier, research_as_of):
        """Read verified frozen reference arrays for a downstream study, without refitting."""
        item = self._item(identifier)
        definition = PreviewRequest.model_validate(item['preview']['request_echo']).definition
        study_day = date.fromisoformat(research_as_of)
        if study_day < definition.research_as_of:
            raise ValidationError('REFERENCE_INFORMATION_CLOCK', '研究日不能早于标尺的信息日期。')
        definition = definition.model_copy(update={'research_as_of': study_day})
        reference, raw, means, covariance, source, panel = self._resolve(definition)
        return {'version': item, 'reference': reference, 'assumptions': raw, 'means': means,
                'covariance': covariance, 'source': source, 'returns': panel}

    def classify(self, identifier, body):
        item = self._item(identifier)
        caps = np.asarray(item['preview']['result']['applied_boundaries'], dtype=np.float64)
        valid = [p for p in item['preview']['result']['frontier'] if p['status'] == 'optimal_to_tolerance' and p['volatility'] is not None]
        minimum = valid[0]['volatility'] if valid else 0.
        risk = np.nan if body.volatility is None else body.volatility
        with self.compute_slot():
            level, flags = numeric.classify_risk_kernel(risk, caps, minimum, numeric.BOUNDARY_TOL)
            within = numeric.risk_within_cap_kernel(risk, caps[body.authorized_level-1], numeric.BOUNDARY_TOL)
        return {'status': 'unavailable' if level == 0 else 'above_scale' if level == 6 else 'classified',
            'level_code': 'C'+str(level) if 1 <= level <= 5 else None, 'cap_satisfied': None if within == -1 else bool(within),
            'authorized_cap': float(caps[body.authorized_level-1]),
            'flags': [name for bit,name in ((1,'below_reference_range'),(2,'within_boundary_tolerance')) if flags & bit]}
