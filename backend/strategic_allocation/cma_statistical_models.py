"""Statistical model orchestration over an already resolved, read-only evidence panel."""
from __future__ import annotations

import numpy as np

from backend.custom_indicators.errors import ValidationError
from backend.factor_research.repository import clean
from . import cma_statistical_kernels as numeric
from .cma_model_contracts import HistoricalCmaRequest, BayesianCmaRequest, RegimeCmaRequest
from .cma_model_kernels import mixture_moments_kernel


def statistical_result(model, evidence):
    numeric.require_ready()
    if evidence is None:
        raise ValidationError("LTCMA_EVIDENCE_REQUIRED", "统计模型必须由服务端解析真实历史证据。")
    returns = evidence["returns"]
    metadata = evidence["metadata"]
    posterior, half_width, estimation_covariance = None, None, None
    audit = {"evidence": metadata, "return_basis": model.return_basis,
             "currency": model.currency, "moment_semantics": "annualized_periodic_arithmetic",
             "covariance_role": "asset_return", "included_uncertainty_components": [],
             "observations": int(returns.shape[0]), "periods_per_year": 252,
             "uncertainty_status": "not_estimated", "limitations": [
                *metadata.get("warnings", []),
                "历史数据与模型假设不保证未来结果；不构成经认证的历史可交易记录。",
                "日频均值和协方差采用独立增量年化近似；预测期限不等于历史估计窗口。",
                "后续资金测算仍须声明分布适配，两个矩不能唯一确定资金成功率或尾部损失。"]}
    if isinstance(model, HistoricalCmaRequest):
        means, covariance, estimation_covariance, half_width = numeric.historical_estimate(returns, float(model.shrinkage))
        audit["mean_estimation_covariance"] = estimation_covariance.tolist()
        audit.update(mean_method="historical_sample_baseline", risk_method="fixed_diagonal_shrinkage",
            shrinkage=float(model.shrinkage), covariance_ddof=1,
            mean_uncertainty_method="iid_marginal_student_t_95", uncertainty_status="estimated_iid",
            mean_estimation_covariance_method="iid_sample_mean_covariance_with_selected_shrinkage")
        audit["limitations"].extend(["均值区间是 iid 假设下的逐资产边际参考，不是联合置信域或长期稳定性证明。",
            "对角收缩保留各资产样本方差并减弱相关性；组合风险相对原样本可能升高或降低，不是风险上界。"])
        audit["shrinkage_target"] = "sample_diagonal_not_portfolio_risk_bound"
    elif isinstance(model, BayesianCmaRequest):
        prior = evidence.get("prior")
        if prior is None:
            raise ValidationError("LTCMA_PRIOR_REQUIRED", "贝叶斯更新缺少已核验的先验版本。")
        prior_state = prior.get("posterior")
        previous_end = prior.get("evidence_end")
        overlap = previous_end is not None and evidence["dates"][0] <= previous_end
        if model.prior_mode == "continue":
            if not prior_state or prior_state.get("frequency") != "daily" or prior_state.get("periods_per_year") != 252:
                raise ValidationError("LTCMA_NIW_CONTINUATION", "后验续更需要同频、同轴的 NIW 后验及信息量。")
            if not previous_end or evidence["dates"][0] <= previous_end:
                raise ValidationError("LTCMA_NIW_OVERLAP", "后验续更只能使用原样本截止日之后的新收益证据。")
            base_mean = np.asarray(prior_state["mean"], dtype=np.float64)
            psi = np.asarray(prior_state["psi"], dtype=np.float64)
            kappa, nu = float(prior_state["kappa"]), float(prior_state["nu"])
        else:
            if overlap and not model.data_reuse_acknowledged:
                raise ValidationError("LTCMA_NIW_DATA_REUSE", "先验与样本存在重叠；请调整窗口或明确确认经验贝叶斯数据复用。")
            base_mean, psi, kappa, nu = numeric.recenter_niw_prior(
                prior["means"], prior["covariance"], float(model.mean_prior_observations),
                float(model.covariance_prior_observations))
        means, covariance, posterior, half_width, base_mean, psi, kappa, nu = numeric.niw_update(
            returns, base_mean, psi, kappa, nu)
        audit.update(mean_method="normal_inverse_wishart_conjugate_update", risk_method="posterior_asset_covariance",
            uncertainty_status="estimated_under_niw", mean_uncertainty_method="marginal_student_t_95",
            posterior_mean_covariance_method="annual_mean_covariance_times_periods_squared",
            prior_ref=model.prior_ref.model_dump(mode="json"), prior_mode=model.prior_mode,
            prior_evidence_overlap=overlap if previous_end else None,
            niw_posterior={"mean": base_mean.tolist(), "psi": psi.tolist(), "kappa": float(kappa),
                "nu": float(nu), "frequency": "daily", "periods_per_year": 252,
                "last_observation": evidence["dates"][-1]})
        audit["limitations"].append("NIW 以同频正态似然与声明的先验为条件；先验强度不是胜率，均值不确定性不重复计入资产风险。")
        if overlap:
            audit["limitations"].append("先验与似然复用了历史数据，本次为已确认的经验贝叶斯研究，不视为两份独立证据。")
    elif isinstance(model, RegimeCmaRequest):
        states, state_ids, regime_audit = evidence["regime"]
        counts, conditional_means, risks = numeric.conditional_state_moments(
            returns, states, len(state_ids), float(model.shrinkage))
        detected = numeric.occupancy_probabilities(counts)
        if model.probabilities is not None and set(model.probabilities) != set(state_ids):
            raise ValidationError("LTCMA_REGIME_PROBABILITY_AXIS", "应用概率必须完整对应所选运行的状态轴。")
        applied = (np.asarray([model.probabilities[s] for s in state_ids], dtype=np.float64)
                   if model.probabilities is not None else detected)
        if any(applied[i] > 0 and counts[i] < 20 for i in range(len(state_ids))):
            raise ValidationError("LTCMA_REGIME_SAMPLE", "正概率状态至少需要 20 个共同收益样本；不能为缺失状态编造收益或风险。")
        # Only small S×N moments are materialized; the T×N panel is never copied per state.
        estimated = np.flatnonzero(counts >= 2)
        probabilities = applied[estimated]
        base_mean, base_cov, within, between = mixture_moments_kernel(
            probabilities, conditional_means[estimated], risks[estimated], False)
        means, covariance = numeric.annualize_moments(base_mean, base_cov, 252)
        audit.update(mean_method="historical_regime_occupancy", risk_method="base_period_mixture_then_annualize",
            annualization_method="historical_occupancy_iid_annualization", covariance_ddof=0,
            state_ids=state_ids, counts=counts.tolist(), detected_probabilities=detected.tolist(),
            applied_probabilities=applied.tolist(), probability_reason=model.probability_reason,
            estimated_states=[state_ids[i] for i in estimated],
            unestimated_states=[state_ids[i] for i in range(len(state_ids)) if counts[i] < 2],
            conditional_base_means=conditional_means[estimated].tolist(),
            conditional_base_covariances=risks[estimated].tolist(),
            within_base_covariance=within.tolist(), between_base_covariance=between.tolist(),
            regime=regime_audit, shrinkage=float(model.shrinkage))
        transition_counts, transition, duration, stationary, transition_status = numeric.regime_transition_diagnostics_kernel(states, len(state_ids))
        statuses = ("unique_irreducible_stationary", "missing_outgoing_observations", "reducibility_not_certified", "stationary_solve_unavailable")
        audit["transition_diagnostics"] = {"counts": transition_counts.tolist(),
            "matrix": clean(transition.tolist()), "markov_duration_observations": clean(duration.tolist()),
            "stationary_probabilities": clean(stationary.tolist()), "stationary_status": statuses[transition_status],
            "adjacency": "adjacent_known_return_end_states", "code_vector_abi_copy_bytes": states.nbytes,
            "forecast_used": False}
        audit["limitations"].extend(["状态占用率不是状态转移概率；本次未建立多年 Markov 路径。",
            "转移与持续期仅为历史诊断；状态持续不必然意味着收益正自相关，不自动上调长期风险。",
            "状态条件协方差使用 ML 分母 n；与历史方法的样本分母 n-1 不同，不直接替换以免破坏总体矩重构。",
            "原样按历史占用率混合、且无收缩时等同于共同分类样本的 ML 矩；不自动增加预测信息。",
            "本方法未估计均值置信半宽，数值 0 仅表示未施加额外半宽惩罚。"])
    else:
        raise ValidationError("LTCMA_METHOD_UNSUPPORTED", "当前统计生成器不支持所选方法。")
    if not np.isfinite(means).all() or not np.isfinite(covariance).all():
        raise ValidationError("LTCMA_NONFINITE_RESULT", "统计结果包含非有限值，未生成假设。")
    vol, corr, minimum = numeric.statistical_covariance_diagnostics(covariance)
    audit.update(effective_volatility=vol.tolist(), effective_correlation=clean(corr.tolist()),
                 min_correlation_eigenvalue=float(minimum), covariance_repaired=False,
                 correlation_semantics="undefined_for_zero_variance_assets")
    return means, covariance, posterior, half_width, audit, estimation_covariance
