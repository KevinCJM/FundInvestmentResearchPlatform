"""Scenario CMA orchestration over frozen historical and recognition evidence.

Long-run occupancy and conditional forecasts share exactly one empirical joint
asset distribution. Recognition qualification never becomes forecast authority.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from backend.custom_indicators.errors import ValidationError
from backend.factor_research.repository import clean
from . import cma_scenario_kernels as numeric
from . import cma_statistical_kernels as statistics
from .cma_model_contracts import ConditionalScenarioCmaRequest


_NUMERIC_MESSAGES = {
    "LTCMA_FORECAST_MISSING_TRANSITIONS": "历史情景缺少完整的状态转换记录，暂时无法推演后续情景。请增加历史覆盖。",
    "LTCMA_FORECAST_UNOBSERVED_STATE": "所选情景包含尚无资产收益样本的状态，无法推演未来收益。",
    "LTCMA_FORECAST_PROBABILITY": "实时情景尚未提供可用的完整概率，请先完成情景识别校准。",
    "LTCMA_FORECAST_TRANSITION": "历史状态转换证据不完整，暂时无法计算条件情景。",
    "LTCMA_SCENARIO_BOOTSTRAP_SAMPLE": "有效情景样本过少，无法估计结果的稳定性。请增加历史覆盖。",
    "LTCMA_STATE_SAMPLE": "已识别情景的共同收益样本不足 20 个，暂时无法计算。",
    "LTCMA_STATE_RETURN": "历史收益包含缺失值或无效价格变化，请检查研究代理数据。",
    "LTCMA_FORECAST_NONFINITE": "当前期限下的推演数值超出计算范围，请缩短期限或检查异常行情。",
}


def _seed(model):
    # Descriptive names and notes cannot change numerical results.
    payload = {"asset_ids": model.asset_ids, "as_of": str(model.as_of),
               "run_ref": model.run_ref.model_dump(mode="json"),
               "window": model.window.model_dump(mode="json"), "method": model.method}
    if isinstance(model, ConditionalScenarioCmaRequest):
        payload.update(realtime_ref=model.realtime_ref.model_dump(mode="json"), horizon_days=model.horizon_days)
    return int(hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8], 16)


def _transition_diagnostics(states, state_ids, regime_audit, evidence, horizon):
    """Date decoding only; scores, splits and maturity gates stay in NJIT."""
    dates = np.asarray(evidence["dates"], dtype="datetime64[D]").astype(np.int64)
    available = regime_audit.get("label_available_dates")
    missing_date = np.iinfo(np.int64).min
    availability = np.asarray([np.datetime64(value, "D").astype(np.int64) if value else missing_date
                               for value in available], dtype=np.int64) if available else np.full(len(dates), missing_date, np.int64)
    scores, cutoff = numeric.transition_holdout_scores(states, dates, availability, len(state_ids), horizon,
                                                      evidence["period_contiguous"])
    names = ("markov", "occupancy", "persistence")
    result = {"status": "diagnostic_only", "horizon_days": horizon,
              "training_end": evidence["dates"][cutoff - 1], "overlap_purged": True,
              "origin_step_days": horizon, "gapped_horizons_excluded": True, "reference_definition_refitted": False,
              "realtime_probability_stream_evaluated": False, "model_fitting_history_proven": False}
    for i, kind in enumerate(("retrospective_oracle_state", "label_availability_checked")):
        result[kind] = {"samples": int(scores[i, 0]),
            "brier": {name: float(scores[i, 1 + j]) for j, name in enumerate(names)},
            "log_loss": {name: float(scores[i, 4 + j]) for j, name in enumerate(names)}}
    if scores[0, 0] == 0:
        result["reason"] = "留出的历史区间不足以比较该期限的后续情景，或训练转换样本不足。"
    elif scores[1, 0] == 0:
        result["reason"] = "历史标签在当时尚未成熟，无法完成满足时点条件的未来比较。"
    else:
        result["reason"] = "已检查标签成熟时点；仍缺少当时已发布的实时概率与模型拟合证据，不能认定未来预测已验证。"
    return clean(result)


def scenario_result(model, evidence):
    numeric.require_ready()
    statistics.require_ready()
    if evidence is None or "regime" not in evidence:
        raise ValidationError("LTCMA_SCENARIO_EVIDENCE", "请先选择已保存且适用于当前研究日的历史情景。")
    try:
        return _scenario_result(model, evidence)
    except ValueError as exc:
        code = str(exc)
        if code in _NUMERIC_MESSAGES:
            raise ValidationError(code, _NUMERIC_MESSAGES[code]) from exc
        raise


def _scenario_result(model, evidence):
    returns = evidence["returns"]
    states, state_ids, regime_audit = evidence["regime"]
    if returns.shape[1] != len(model.asset_ids) or states.size != returns.shape[0]:
        raise ValidationError("LTCMA_SCENARIO_AXIS", "情景与资产收益的日期或资产范围不一致，请重新选择。")
    counts, raw_means, raw_covariance = statistics.conditional_state_moments(returns, states, len(state_ids), 0.0)
    occupancy = statistics.occupancy_probabilities(counts)
    weights, validation_support = numeric.empirical_regularization_weights(returns, states, len(state_ids))
    means, covariance, pooled_mean, pooled_covariance = numeric.regularized_state_moments(
        counts, raw_means, raw_covariance, weights)
    baseline_mean, baseline_covariance = statistics.annualize_moments(pooled_mean, pooled_covariance, 252)
    contiguous = evidence["period_contiguous"]
    episodes = numeric.complete_state_episodes(states, len(state_ids), contiguous)
    applied = occupancy
    half_width = estimation_covariance = None
    seed = _seed(model)
    audit = {
        "scenario_mode": model.method,
        "evidence": evidence["metadata"], "regime": regime_audit,
        "return_basis": model.return_basis, "currency": model.currency,
        "moment_semantics": "annualized_periodic_arithmetic", "covariance_role": "asset_return",
        "included_uncertainty_components": [], "observations": int(returns.shape[0]),
        "periods_per_year": 252, "state_ids": state_ids, "counts": counts.tolist(),
        "unestimated_states": [state_ids[i] for i in range(len(state_ids)) if counts[i] < 2],
        "complete_episodes": episodes.tolist(), "covariance_ddof": 0,
        "mean_method": "regularized_joint_empirical_scenario_distribution",
        "risk_method": "base_period_total_covariance_then_annualize",
        "uncertainty_status": "not_estimated", "model_policy_version": numeric.VERSION,
        "regularization": {"method": "chronological_energy_score_empirical_mixture",
            "weights": weights.tolist(), "validation_samples": validation_support.tolist(),
            "fallback": "pooled_distribution_if_less_than_10_validation_samples",
            "training_atoms_per_group": 64, "validation_cap_per_state_fold": 128,
            "folds": 3, "reference_definition_refitted": False},
        "raw_conditional_base_means": raw_means.tolist(),
        "raw_conditional_base_covariances": raw_covariance.tolist(),
        "conditional_base_means": means.tolist(), "conditional_base_covariances": covariance.tolist(),
        "historical_baseline": {"annual_returns": baseline_mean.tolist(),
            "annual_covariance": baseline_covariance.tolist()},
        "scenario_probabilities": {"state_ids": state_ids,
            "state_labels": [regime_audit.get("state_labels", {}).get(s, s) for s in state_ids],
            "historical": occupancy.tolist()},
        "model_validation": {"status": "historical_research", "downstream_eligible": True,
            "reason": "长期历史情景研究；确认后可供 SAA 使用，不代表已验证未来预测能力。"},
        "limitations": [*evidence["metadata"].get("warnings", []),
            "状态按时间占用率汇总，不把完整行情片段的数量当作出现概率。",
            "自动收缩只调整历史估计稳定性；事后情景划分与历史样本不是未来预测证明。",
            "基础期均值与协方差按独立增量近似年化；累计收益和复利收益另行计算。",
            "联合经验分布保留样本中的共同涨跌，但不能生成历史中从未发生的危机。",
            "当前模型只支持同轴 CNY/SSE 日频证据，不自动扩展月频标签或补齐缺失状态。"],
    }
    if isinstance(model, ConditionalScenarioCmaRequest):
        forecast = evidence.get("forecast_evidence")
        if not forecast or not forecast.get("calibration") or "current_probabilities" not in forecast:
            raise ValidationError("LTCMA_SCENARIO_CALIBRATION", "请先在情景研究中心完成实时情景的概率校准，再生成条件情景研究。")
        if forecast.get("state_ids", state_ids) != state_ids:
            raise ValidationError("LTCMA_SCENARIO_REFERENCE", "实时识别与历史情景的状态定义不一致，请选择同一情景参考。")
        if any(count < 20 for count in counts):
            raise ValidationError("LTCMA_SCENARIO_STATE_SAMPLE", "至少一个情景的资产收益样本不足 20 个，暂时无法推演完整情景路径。")
        if model.horizon_days * len(state_ids) ** 2 * len(model.asset_ids) ** 2 > 50_000_000:
            raise ValidationError("LTCMA_SCENARIO_FORECAST_BUDGET", "当前资产和情景数量对应的推演期限过长，请缩短期限后重试。")
        initial = np.asarray(forecast["current_probabilities"], dtype=np.float64)
        transition, strength, training_pairs = numeric.estimated_transition_matrix(states, len(state_ids), contiguous)
        horizon_mean, horizon_covariance, path, applied = numeric.markov_compound_moments(
            initial, transition, means, covariance, model.horizon_days)
        # Memory/work caps are policy, not a new human form field. Adaptive doubling
        # uses the same seed and reports unmet precision rather than false success.
        max_paths = min(8192, 20_000_000 // (model.horizon_days * returns.shape[1]))
        paths = min(1024, max_paths)
        simulation_work = 0
        work_per_path = model.horizon_days * returns.shape[1]
        while True:
            samples = numeric.simulate_joint_horizon(returns, states, weights, initial, transition,
                model.horizon_days, paths, seed)
            simulation_work += paths * work_per_path
            quantiles, loss, mean_se, loss_se, quantile_bounds = numeric.horizon_sample_summary(samples)
            converged = bool(np.all(mean_se <= .001))
            next_paths = min(paths * 2, max_paths, (20_000_000 - simulation_work) // work_per_path)
            if converged or next_paths <= paths:
                break
            paths = next_paths
        audit["scenario_probabilities"].update(current=initial.tolist(), endpoint=path[-1].tolist(), average=applied.tolist())
        audit["forecast_evidence"] = clean({k: v for k, v in forecast.items() if k != "current_probabilities"})
        audit["transition_model"] = {"matrix": transition.tolist(), "method": "training_selected_dirichlet_markov",
            "prior_strength": float(strength), "training_validation_pairs": int(training_pairs),
            "forecast_used": True, "future_skill_validated": False,
            "adjacency": "contiguous_return_intervals_and_known_states",
            "excluded_gap_transitions": int(np.count_nonzero(contiguous[1:] == 0))}
        audit["forecast_validation"] = _transition_diagnostics(states, state_ids, regime_audit, evidence, model.horizon_days)
        audit["horizon_distribution"] = {
            "horizon_days": model.horizon_days, "basis": "cumulative_compounded_simple_return",
            "annualized": False, "expected_returns": horizon_mean.tolist(),
            "covariance": clean(horizon_covariance.tolist()), "quantile_levels": [.05, .5, .95],
            "quantiles": quantiles.tolist(), "loss_probabilities": loss.tolist(),
            "moment_method": "exact_markov_reward_recursion", "tail_method": "joint_empirical_simulation",
            "paths": paths, "seed": seed, "mean_monte_carlo_standard_error": mean_se.tolist(),
            "loss_monte_carlo_standard_error": loss_se.tolist(),
            "quantile_monte_carlo_intervals": quantile_bounds.tolist(),
            "quantile_interval_method": "approximate_95_percent_order_statistic_binomial_normal",
            "simulation_precision": "target_met" if converged else "budget_reached",
            "simulation_precision_scope": "mean_only_not_tail_or_model_uncertainty",
            "simulation_mean_standard_error_target": .001,
            "simulation_work_budget": 20_000_000,
            "simulation_return_draws": simulation_work,
        }
        audit["model_validation"] = {"status": "research_only", "downstream_eligible": False,
            "reason": "已完成条件情景推演；未来预测尚未独立验证，暂不用于 SAA。"}
        audit["limitations"].extend([
            "当前状态概率来自实时识别校准；历史转换拟合不代表未来预测已通过验证。",
            "给定状态后，逐期收益独立抽取；状态转换保留时序影响，状态内剩余序列相关尚未建模。",
            "年化结果是未来窗口内随机一期的矩摘要，不是该窗口累计风险，也不是 CAGR。",
            "分位数与亏损概率来自条件模型模拟，不是保证；模型估计不确定性尚未计入路径分布。",
            "条件结果仅供研究比较；在完成未来验证和期限适配前，不交给长期 SAA。",
        ])
    else:
        estimation_covariance, half_width, block_length, max_sampled_length, block_count = numeric.block_mean_uncertainty(
            returns, states, weights, seed, contiguous)
        gapped = bool(np.any(contiguous[1:] == 0))
        audit.update(uncertainty_status="estimated_fixed_definition_block_bootstrap",
            mean_uncertainty_method=("joint_contiguous_block_bootstrap_marginal_95" if gapped
                                     else "joint_circular_block_bootstrap_marginal_95"),
            mean_estimation_covariance=estimation_covariance.tolist(),
            mean_estimation_covariance_method="bootstrap_annual_mean_fixed_definition_and_regularizer",
            bootstrap={"draws": 256, "seed": seed, "block_length": int(max_sampled_length),
                "target_block_length": int(block_length), "blocks_per_draw": int(block_count),
                "sampling": "disjoint_contiguous_blocks" if gapped else "circular_blocks",
                "fallback": "independent_observations" if max_sampled_length == 1 else None,
                "gap_count": int(np.count_nonzero(contiguous[1:] == 0)),
                "lag_one_pairs": "contiguous_return_intervals_and_known_states",
                "normalization": "resampled_known_observation_count",
                "block_policy": "cube_root_n_inflated_by_positive_lag_one_return_dependence"})
        audit["limitations"].extend([
            "均值区间来自联合时间块重采样，不计入资产协方差，避免重复计算风险。",
            "区间以固定情景定义及已选收缩权重为条件；未覆盖情景模型选择或标签修订不确定性。",
            "自动块长度使用明确的依赖性启发式，不宣称最优块长或完整的长期预测覆盖率。",
        ])
    base_mean, base_cov, within, between = numeric.scenario_mixture_moments(applied, means, covariance)
    annual_mean, annual_cov = statistics.annualize_moments(base_mean, base_cov, 252)
    volatility, correlation, minimum = statistics.statistical_covariance_diagnostics(annual_cov)
    audit["scenario_probabilities"]["applied"] = applied.tolist()
    audit.update(detected_probabilities=occupancy.tolist(), applied_probabilities=applied.tolist(),
        within_base_covariance=within.tolist(), between_base_covariance=between.tolist(),
        effective_volatility=volatility.tolist(), effective_correlation=clean(correlation.tolist()),
        min_correlation_eigenvalue=float(minimum), covariance_repaired=False,
        correlation_semantics="undefined_for_zero_variance_assets")
    return annual_mean, annual_cov, half_width, clean(audit), estimation_covariance
