"""Pure CMA boundary orchestration, ready for the existing service to consume."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np
from pydantic import BaseModel

from . import cma_model_kernels as kernels
from .kernels import mean_covariance_diagnostics_kernel
from .cma_model_contracts import (
    BlackLittermanRequest, CMA_MODEL_ADAPTER, CmaModelRequest, ScenarioMixtureRequest,
    HistoricalCmaRequest, BayesianCmaRequest, RegimeCmaRequest,
)


def readonly_float64(values, ndim: int) -> np.ndarray:
    """One boundary conversion; float64 arrays keep their owner and strides."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError("CMA_MODEL_ARRAY_DIMENSION")
    view = array.view()
    view.flags.writeable = False
    return view


@dataclass(frozen=True)
class CmaModelResult:
    asset_ids: tuple[str, ...]
    method: str
    effective_returns: np.ndarray
    effective_covariance: np.ndarray
    posterior_mean_covariance: np.ndarray | None
    model_audit: dict[str, Any]
    execution: dict[str, Any]
    definition: dict[str, Any]
    mean_uncertainty: np.ndarray | None = None
    mean_estimation_covariance: np.ndarray | None = None

    def to_payload(self) -> dict[str, Any]:
        """JSON boundary only; this does not save or hash an artifact."""
        return {"asset_ids": list(self.asset_ids), "method": self.method,
                "effective_returns": self.effective_returns.tolist(),
                "effective_covariance": self.effective_covariance.tolist(),
                "posterior_mean_covariance": None if self.posterior_mean_covariance is None else self.posterior_mean_covariance.tolist(),
                "model_audit": self.model_audit, "execution": self.execution, "definition": self.definition,
                **({"mean_uncertainty": self.mean_uncertainty.tolist()} if self.mean_uncertainty is not None else {}),
                **({"mean_estimation_covariance": self.mean_estimation_covariance.tolist()}
                   if self.mean_estimation_covariance is not None else {})}


def evaluate_cma_model(
    request: CmaModelRequest | Mapping[str, Any], *, asset_ids: Sequence[str] | None = None,
    as_of: date | str | None = None, currency: str | None = None,
    evidence: dict | None = None,
) -> CmaModelResult:
    """Compute only after startup warm(); optional context guards serve integration.

    The enclosing service remains responsible for research-clock/PIT, source
    identity, immutable confirmation/hash, and publication/application gates.
    Model instances are revalidated because their nested lists may have changed.
    """
    kernels.require_ready()
    raw = request.model_dump(mode="python") if isinstance(request, BaseModel) else request
    model = CMA_MODEL_ADAPTER.validate_python(raw)
    if asset_ids is not None and list(asset_ids) != model.asset_ids:
        raise ValueError("CMA_MODEL_CONTEXT_AXIS")
    if as_of is not None and str(as_of) != str(model.as_of):
        raise ValueError("CMA_MODEL_CONTEXT_DATE")
    if currency is not None and currency != model.currency:
        raise ValueError("CMA_MODEL_CONTEXT_CURRENCY")
    if isinstance(model, (HistoricalCmaRequest, BayesianCmaRequest, RegimeCmaRequest)):
        from .cma_statistical_models import statistical_result
        from .cma_statistical_kernels import execution_audit
        means, covariance, posterior, half_width, audit, estimation_covariance = statistical_result(model, evidence)
        mean_covariance = posterior if posterior is not None else estimation_covariance
        if mean_covariance is not None:
            standard_error, _, _ = mean_covariance_diagnostics_kernel(mean_covariance)
            audit["mean_standard_error"] = standard_error.tolist()
        return CmaModelResult(tuple(model.asset_ids), model.method, readonly_float64(means, 1),
            readonly_float64(covariance, 2), None if posterior is None else readonly_float64(posterior, 2),
            audit, execution_audit(), model.model_dump(mode="json"),
            None if half_width is None else readonly_float64(half_width, 1),
            None if estimation_covariance is None else readonly_float64(estimation_covariance, 2))
    n = len(model.asset_ids)
    posterior = None
    if isinstance(model, BlackLittermanRequest):
        covariance = readonly_float64(model.covariance, 2)
        weights = readonly_float64([model.market_weights[a] for a in model.asset_ids], 1)
        picks = np.zeros((len(model.views), n), dtype=np.float64)
        axis = {asset: i for i, asset in enumerate(model.asset_ids)}
        for v, view in enumerate(model.views):
            if view.kind == "basket":
                for leg in view.legs:
                    picks[v, axis[leg.asset_id]] = leg.coefficient
            else:
                picks[v, axis[view.asset_id]] = 1
                if view.relative_to is not None:
                    picks[v, axis[view.relative_to]] = -1
        picks.flags.writeable = False
        means, posterior, prior, gain, residual, prior_view_variance, omega = kernels.black_litterman_update_kernel(
            covariance, weights, picks,
            readonly_float64([v.annual_return for v in model.views], 1),
            readonly_float64([v.view_std for v in model.views], 1),
            float(model.delta), float(model.tau), float(model.risk_free_rate))
        market_excess, market_vol, sharpe, ratio, contributions, shift = kernels.black_litterman_diagnostics_kernel(
            covariance, weights, prior, gain, residual, prior_view_variance, omega)
        audit = {"mean_method": "black_litterman_gaussian_update", "prior_basis": "excess_return",
                 "diagnostics_version": "bl-diagnostics/1.0", "uncertainty_status": "declared_not_estimated",
                 "mean_covariance_status": "gaussian_posterior",
                 "implied_market_excess_return": float(market_excess),
                 "implied_market_volatility": float(market_vol),
                 "implied_market_sharpe": float(sharpe) if np.isfinite(sharpe) else None,
                 "view_prior_variance": prior_view_variance.tolist(), "view_error_variance": omega.tolist(),
                 "view_precision_ratio": [float(x) if np.isfinite(x) else None for x in ratio],
                 "view_return_contributions": contributions.tolist(), "posterior_shift": shift.tolist(),
                 "prior_excess_returns": prior.tolist(), "view_count": len(model.views),
                 "view_basis": "absolute_total_or_relative_total_difference",
                 "view_uncertainty": "diagonal_variance_from_positive_std",
                 "risk_method": "explicit_input_covariance_unchanged",
                 "posterior_mean_covariance_method": "joseph_form",
                 "solver": "linear_solve_no_inverse", "no_views_equals_prior": not model.views,
                 "limitations": ["观点标准差不是收益实现概率。", "均值后验协方差不作为资产风险，也不自动转为稳健半宽。",
                     "隐含 Sharpe 和观点方差比仅供复核，没有通用合理阈值；相关观点的实际影响以联合更新贡献为准。"]}
    else:
        means_by_scenario = readonly_float64([[s.annual_returns[a] for a in model.asset_ids] for s in model.scenarios], 2)
        probabilities = readonly_float64([s.probability for s in model.scenarios], 1)
        if model.risk_mode == "shared":
            risks = readonly_float64(model.shared_covariance, 2)[None, :, :]
        else:
            risks = readonly_float64([s.covariance for s in model.scenarios], 3)
        means, covariance, within, between = kernels.scenario_mixture_kernel(
            probabilities, means_by_scenario, risks, model.risk_mode == "shared")
        audit = {"mean_method": "explicit_probability_mixture", "risk_method": "total_covariance",
                 "uncertainty_status": "declared_not_estimated",
                 "risk_mode": model.risk_mode, "probabilities_normalized": False,
                 "scenario_ids": [s.id for s in model.scenarios],
                 "probabilities": probabilities.tolist(), "within_covariance": within.tolist(),
                 "between_mean_covariance": between.tolist(),
                 "limitations": ["用户假设的单期矩匹配，不是多期校准或尾部分布预测。"]}
    volatility, correlation, min_eigenvalue = kernels.covariance_diagnostics_kernel(covariance)
    if posterior is not None:
        standard_error, _, _ = mean_covariance_diagnostics_kernel(posterior)
        audit["mean_standard_error"] = standard_error.tolist()
    audit.update({"return_basis": model.return_basis, "currency": model.currency,
                  "effective_volatility": volatility.tolist(), "effective_correlation": correlation.tolist(),
                  "min_correlation_eigenvalue": float(min_eigenvalue), "covariance_repaired": False})
    return CmaModelResult(tuple(model.asset_ids), model.method, readonly_float64(means, 1),
                          readonly_float64(covariance, 2),
                          None if posterior is None else readonly_float64(posterior, 2),
                          audit, kernels.execution_audit(), model.model_dump(mode="json"))
