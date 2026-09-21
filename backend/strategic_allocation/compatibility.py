"""Common-model policy study: independent anchors, joint solve, exact model gates."""
from __future__ import annotations

import copy
import time
import numpy as np
from backend.custom_indicators.errors import ValidationError
from . import compatibility_kernels as numeric
from .compatibility_solver import solve, MAX_SECONDS
from .multi_cma import METRICS, cross_model_results, require_calculation_budget
from .planning import funding_inputs
from .mandate_inputs import cash_success_required
from . import goal_kernels


def _evidence(result):
    return {key: value for key, value in result.items() if key != 'weights'}


def calculate(service, request, mandate, cma, *, paths, simulation_seed):
    numeric.require_ready()
    budget = require_calculation_budget(request, mandate, paths)
    multi = cma['multi_cma']
    definition = multi['assumptions']
    names = [a['id'] for a in definition['assets']]
    groups, limits = service._constraints(request, definition, mandate)
    bounds = np.asarray([[limits[n]['min_weight'], limits[n]['max_weight']] for n in names], dtype=np.float64)
    membership = np.asarray([[int(n in g['assets']) for n in names] for g in groups], dtype=np.float64).reshape(len(groups), len(names))
    lows = np.asarray([g['lo'] for g in groups], dtype=np.float64)
    highs = np.asarray([g['hi'] for g in groups], dtype=np.float64)
    means = np.asarray([[a['annual_return'] for a in s['assumptions']['assets']] for s in multi['sources']], dtype=np.float64)
    risks = np.asarray([s['covariance'] for s in multi['sources']], dtype=np.float64)
    benchmark = mandate.get('benchmark')
    if benchmark and (set(benchmark['weights']) != set(names) or
            benchmark.get('source', 'explicit') == 'explicit' and benchmark['alloc_name'] != definition['alloc_name']):
        raise ValidationError('MANDATE_BENCHMARK_AXIS', '授权基准须与所选模型使用一致的资产轴与大类方案。')
    benchmark_weights = np.asarray([benchmark['weights'][n] for n in names] if benchmark else [], dtype=np.float64)
    floor = mandate.get('effective_target_return')
    if floor is None:
        floor = mandate['target_return'] if mandate.get('objective_kind', 'absolute_return') == 'absolute_return' else -np.inf
    for array in (bounds, membership, lows, highs, means, risks, benchmark_weights):
        array.flags.writeable = False
    deadline = time.monotonic() + MAX_SECONDS
    seed = request.seed if simulation_seed is None else simulation_seed
    common = (bounds, membership, lows, highs, benchmark_weights, floor, mandate['max_volatility'],
              benchmark['max_tracking_error'] if benchmark else 1., benchmark['target_excess_return'] if benchmark else 0.)
    anchors = []
    references = []
    joint = None
    for m, source in enumerate(multi['sources']):
        result = solve(means[m:m+1], risks[m:m+1], *common, np.zeros(1), 0,
                       max_iterations=request.solver_max_iterations, deadline=deadline)
        anchor = {'cma_id': source['cma_id'], 'name': source['name'], 'solver': _evidence(result),
                  'weights': None, 'cross_model_results': [], 'reference_value': None, 'reference_upper_bound': None}
        if result['weights'] is not None:
            anchor['weights'] = dict(zip(names, result['weights'].tolist(), strict=True))
            anchor['reference_value'] = -result['objective_value']
            anchor['reference_upper_bound'] = -result['lower_bound'] if result['lower_bound'] is not None else None
            anchor['cross_model_results'] = cross_model_results(multi, anchor['weights'], mandate,
                penalty=request.uncertainty_penalty, paths=paths, seed=seed)
            references.append(anchor['reference_value'])
        anchors.append(anchor)
        if result['weights'] is None:
            # A proven empty individual set also proves the intersection empty.
            # Otherwise the anchor/return reference is simply unresolved.
            joint = {**_evidence(result), 'blocked_by_anchor': source['cma_id']}
            break
    candidates = []
    if len(references) == len(multi['sources']):
        references_array = np.asarray(references, dtype=np.float64)
        references_array.flags.writeable = False
        result = solve(means, risks, *common, references_array,
                       int(request.compatibility_objective == 'minimax_regret'),
                       max_iterations=request.solver_max_iterations, deadline=deadline)
        joint = _evidence(result)
        if result['weights'] is not None:
            weights = dict(zip(names, result['weights'].tolist(), strict=True))
            rows = cross_model_results(multi, weights, mandate, penalty=request.uncertainty_penalty, paths=paths, seed=seed)
            metrics_array = np.asarray([[r['metrics'][k] for k in METRICS] for r in rows], dtype=np.float64)
            probabilities = np.asarray([r['goal_check']['central']['probability_lower'] for r in rows]
                                       if cash_success_required(mandate) else [], dtype=np.float64)
            worst_metrics, worst_funding = numeric.worst_model_summary_kernel(metrics_array, probabilities)
            passing = all(row['within_limits'] for row in rows)
            candidate = {'id': 'compatible', 'name': '共同约束：最小最大收益机会损失' if request.compatibility_objective == 'minimax_regret' else '共同约束：最大化最坏模型收益',
                         'weights': weights, 'metrics': dict(zip(METRICS, worst_metrics.tolist(), strict=True)),
                         'metric_basis': 'worst_per_metric_not_one_distribution', 'risk_contributions': {},
                         'cross_model_results': rows, 'all_models_pass': passing,
                         'solver': joint, 'available': passing,
                         'unavailable_reason': None if passing else '至少一个原模型未通过收益、风险或资金检查，不能采纳。'}
            if worst_funding >= 0:
                candidate['goal_check'] = copy.deepcopy(rows[worst_funding]['goal_check'])
                candidate['goal_check']['summary_source_cma_id'] = rows[worst_funding]['cma_id']
            candidates.append(candidate)
    prepared = funding_inputs(mandate)
    exact_anchors = len(anchors) == len(multi['sources']) and all(a['solver']['status'] == 'converged' for a in anchors)
    compatibility = {'objective': request.compatibility_objective,
                     'regret_basis': 'bounded_continuous_anchor_optima' if exact_anchors else 'approximate_regret',
                     'anchors': anchors, 'joint_solver': joint,
                     'gate': 'all_frozen_models', 'funding_search_domain': 'one_joint_candidate_with_anchor_cross_diagnostics',
                     'enforcement': {'linear_and_risk_constraints': 'solver_and_gate',
                                     'funding_success': 'candidate_filter_and_gate' if cash_success_required(mandate) else 'not_applicable'},
                     'limitations': ['资金成功率只检查已生成的共同配置；失败不等于资金目标在全部权重空间无解。',
                                     '汇总指标分别取各模型最不利值，不代表一个联合分布；具体资金模拟使用各自原始矩。']}
    return {'constraints': limits, 'group_limits': groups, 'covariance': multi['effective_covariance'],
            'candidates': candidates, 'accepted_candidates': int(any(c['available'] for c in candidates)),
            'funding': prepared[0] if prepared else None,
            'funding_model': {'distribution': 'each_source_annual_moment_proxy', 'paths': paths, 'seed': seed} if cash_success_required(mandate) else None,
            'funding_execution': goal_kernels.execution_audit() if cash_success_required(mandate) else None,
            'multi_cma_budget': budget, 'compatibility': compatibility,
            'method': 'continuous_common_model_outer_approximation', 'execution': numeric.execution_audit()}
