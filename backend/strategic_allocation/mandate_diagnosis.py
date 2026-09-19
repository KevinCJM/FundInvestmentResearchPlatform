"""Reference-only mandate research: constrained frontier, search, then fixed-candidate validation."""
from __future__ import annotations

import numpy as np

from backend import frontier_moments
from backend.custom_indicators.errors import ValidationError
from backend.sensitivity.repository import digest_json
from .contracts import PolicyRequest
from . import goal_kernels as goals, mandate_kernels as numeric
from .mandate_inputs import effective_funding_plan, cash_success_required, effective_cash_floor, effective_return_floor
from .planning import funding_inputs, _funding_metrics

LIMITATIONS = [
    "这是冻结全局参考下的有限候选研究，不是本次实际SAA的采纳证明。",
    "未找到候选不证明数学无解；提高授权上限不排除原有低风险候选。",
    "搜索与验证使用不同随机样本；验证失败不使用同一验证集重新挑选候选。",
    "Wilson区间只衡量独立模拟样本误差，不包含市场假设、现金计划与分布模型误差。",
    "参考资金成功不代表未来保证，也不证明历史时点可投资。",
]


def distribution_spec(context: dict) -> dict:
    reference = context["reference"]
    preview = reference.get("reference_preview", {})
    quality = preview.get("quality", {})
    periodic = (quality.get("annualization_method") == "arithmetic_mean_and_covariance_times_periods"
                and quality.get("periods_per_year") == 252)
    return {"engine": "base_period_moment_match_v2" if periodic else "annual_moment_proxy_approximation",
            "method_code": int(periodic), "periods_per_year": 252 if periodic else 1,
            "source_reference_ref": {k: reference[k] for k in ("id", "content_hash")},
            "version": "mandate-distribution/1.1.0", "frequency": "equal_model_month",
            "serial_dependence_assumption": "independent_log_growth_increments",
            "fee_basis": "source_embedded_plus_explicit_additional_model_fee",
            "historical_pit_proven": False}


def _reachability(metrics, levels, cap: float, return_floor: float) -> dict:
    """Name the binding bound so the user sees why a target is or is not reachable."""
    best, best_row, least, least_row = numeric.reference_reachability_kernel(metrics, levels, cap, return_floor)
    target = float(return_floor) if np.isfinite(return_floor) else None
    reachable = float(best) if best_row >= 0 else None
    binding = "none"
    if target is not None:
        if reachable is None or reachable < target - 1e-10:
            binding = "volatility_cap" if least_row >= 0 else "unreachable_at_any_level"
    return {"volatility_cap": float(cap), "max_return_under_cap": reachable,
            "max_return_candidate_id": f"frontier-{best_row}" if best_row >= 0 else None,
            "max_return_risk_level": int(levels[best_row]) if best_row >= 0 and 1 <= levels[best_row] <= 5 else None,
            "target_return": target,
            "min_volatility_for_target": float(least) if least_row >= 0 else None,
            "required_risk_level": int(levels[least_row]) if least_row >= 0 and 1 <= levels[least_row] <= 5 else None,
            "binding": binding,
            "basis": "frozen_reference_constrained_frontier_not_an_adoption_proof"}


def _constraints(service, definition, context):
    raw = {"alloc_name": None, "strategic_universe_id": None, **context["assumptions"]}
    standard = context["version"]["preview"]["request_echo"]["definition"]["constraint_profile"]
    request = PolicyRequest(mandate_id="reference-study", cma_id=context["reference"]["id"],
        constraints=standard["asset_limits"], group_limits=[{**g, "id": f"reference-standard-{i}"}
                                                         for i, g in enumerate(standard["group_limits"])])
    # Risk Scale only carries cash/non-cash. Mandate's shared constraint consumer
    # still expects role/liquidity fields, so adapt them only at this boundary.
    constraint_raw = {**raw, "assets": [{**asset,
        "role": "liquidity" if asset.get("asset_type") == "cash" else "diversifier",
        "liquidity": "liquid"} for asset in raw["assets"]]}
    groups, limits = service._constraints(request, constraint_raw, definition)
    ids = [a["id"] for a in raw["assets"]]
    bounds = np.asarray([[limits[x]["min_weight"], limits[x]["max_weight"]] for x in ids], dtype=np.float64)
    membership = np.asarray([[float(x in g["assets"]) for x in ids] for g in groups], dtype=np.float64).reshape(len(groups), len(ids))
    lows = np.asarray([g["lo"] for g in groups], dtype=np.float64)
    highs = np.asarray([g["hi"] for g in groups], dtype=np.float64)
    for array in (bounds, membership, lows, highs):
        array.flags.writeable = False
    return raw, ids, groups, bounds, membership, lows, highs


def validate_fixed_candidate(metrics, definition, prepared, request, spec):
    summary, inflows, outflows = prepared
    plan = effective_funding_plan(definition)
    draws, _ = goals.seeded_factor_draws_kernel(inflows.size, request.simulation_paths, 1, request.validation_seed, 0, 5.)
    draws.flags.writeable = False
    drift, scale = goals.funding_monthly_parameters_kernel(float(metrics[0]), float(metrics[1]), plan["annual_fee"],
                                                           spec["method_code"], spec["periods_per_year"])
    values, fan = goals.funding_paths_from_monthly_kernel(draws, drift, scale, summary["investable_capital"],
        inflows, outflows, summary["nominal_terminal_target"], plan["required_probability"], 1., 1.)
    central = _funding_metrics(values)
    central["drawdown_alert_probability"] = None
    central["annual_fan"] = [{"year": i, "p05": float(x[0]), "median": float(x[1]), "p95": float(x[2])} for i, x in enumerate(fan)]
    return {"seed": request.validation_seed, "paths": request.simulation_paths,
            "within_limits": central["probability_lower"] >= plan["required_probability"],
            "threshold": plan["required_probability"], "gate_basis": "wilson_95pct_lower_bound",
            "central": central, "candidate_frozen_before_validation": True}


def diagnose_reference(service, request, definition: dict, decision: dict) -> dict:
    """No writes or policy authorization. A recommendation remains research evidence."""
    numeric.require_ready()
    result = {"status": "reference_pending", "minimum_tested_feasible_level": None,
              "selected_candidate": None, "validation": None, "candidates": [],
              "limitations": list(LIMITATIONS), "blockers": [], "search_seed": request.seed,
              "validation_seed": request.validation_seed, "paths": request.simulation_paths,
              "execution": numeric.execution_audit()}
    ref = decision.get("risk_scale_ref")
    if ref is None:
        return result
    context = service.risk_scales.frozen_context(ref["id"], definition["as_of"])
    scale = context["version"]["preview"]["request_echo"]["definition"]
    raw = context["assumptions"]
    result["risk_scale_ref"] = ref
    benchmark = definition.get("benchmark")
    explicit_benchmark = benchmark and benchmark.get("source", "explicit") == "explicit"
    if ((definition.get("allocation_scope") and definition["allocation_scope"] != raw.get("alloc_name"))
            or (definition.get("strategic_universe_id") and definition["strategic_universe_id"] != raw.get("strategic_universe_id"))
            or (explicit_benchmark and benchmark["alloc_name"] != raw.get("alloc_name"))):
        result.update(status="awaiting_actual_scope", blockers=["本次资产授权或基准不能直接映射全局参考，请在实际CMA下诊断；不按同名资产猜测。"])
        return result
    try:
        raw, ids, groups, bounds, membership, lows, highs = _constraints(service, definition, context)
    except ValidationError as exc:
        if exc.code not in {"MANDATE_LIQUIDITY_CONFLICT", "MANDATE_LIMIT_CONFLICT", "SAA_CASH_ASSETS_MISSING", "SAA_LIQUID_ASSETS_MISSING"}:
            raise
        result.update(status="constraint_conflict", blockers=[str(exc)])
        return result
    spec = distribution_spec(context)
    prepared = funding_inputs(definition)
    funding_floor = prepared[0]["required_liquid_weight"] if prepared else 0.0
    cash_ids = [asset["id"] for asset in raw["assets"] if asset.get("asset_type") == "cash"]
    reference_result = context["version"]["preview"]["result"]
    result.update(distribution=spec, group_limits=groups, ordered_asset_ids=ids,
                  reference_input_ref={k: context["reference"][k] for k in ("id", "content_hash")},
                  reference_frontier=[{key: point.get(key) for key in ("node_id", "expected_return", "volatility", "status")}
                                      for point in reference_result.get("frontier", [])],
                  risk_boundaries=list(reference_result.get("applied_boundaries", [])),
                  cash_constraint={"cash_asset_ids": cash_ids,
                      "requested_min_cash_weight": float(definition.get("min_cash_weight", 0.0)),
                      "cashflow_derived_weight": float(funding_floor),
                      "effective_min_cash_weight": float(effective_cash_floor(definition, funding_floor))})
    means, covariance = context["means"], context["covariance"]
    uncertainty = np.asarray([a["mean_uncertainty"] for a in raw["assets"]], dtype=np.float64)
    uncertainty.flags.writeable = False
    benchmark = definition.get("benchmark")
    if benchmark and set(benchmark["weights"]) != set(ids):
        result.update(status="awaiting_actual_scope", blockers=["基准缺少完整同轴权重，需实际范围验证。"])
        return result
    benchmark_weights = np.asarray([benchmark["weights"][x] for x in ids] if benchmark else [], dtype=np.float64)
    caps = np.asarray(context["version"]["preview"]["result"]["applied_boundaries"], dtype=np.float64)
    with service.risk_scales.compute_slot():
        solved = frontier_moments.solve_frontier(means, covariance, bounds, membership, lows, highs)
        weights, statuses = solved[1], solved[3]
        result["constrained_frontier"] = [{"node_id": i, "volatility": float(solved[2][i, 0]) if np.isfinite(solved[2][i, 0]) else None,
            "expected_return": float(solved[2][i, 1]) if np.isfinite(solved[2][i, 1]) else None,
            "status": frontier_moments.STATUS_NAMES[int(statuses[i])]} for i in range(statuses.size)]
        result["solver"] = {"version": frontier_moments.VERSION, "phase_status": frontier_moments.STATUS_NAMES[int(solved[11])],
            "point_statuses": [frontier_moments.STATUS_NAMES[int(s)] for s in statuses],
            "scope": "finite_constrained_reference_frontier"}
        if solved[11] != 0:
            result.update(status="solver_failed", blockers=["参考约束前沿未完成数值验证，不能判定目标无解。"])
            return result
        floor = effective_return_floor(definition, prepared[0]["cashflow_required_return"] if prepared else None)
        return_floor = floor if floor is not None else -np.inf
        metrics, levels, eligible = numeric.reference_candidate_checks_kernel(weights, statuses, means, covariance, uncertainty,
            bounds, membership, lows, highs, benchmark_weights, definition["max_volatility"],
            return_floor,
            benchmark["max_tracking_error"] if benchmark else 1., benchmark["target_excess_return"] if benchmark else 0.,
            definition["risk_aversion"], request.uncertainty_penalty, caps, float(solved[2][0, 0]))
        result["reachability"] = _reachability(metrics, levels, definition["max_volatility"], return_floor)
        outcomes = None
        if cash_success_required(definition):
            summary, inflows, outflows = prepared
            plan = effective_funding_plan(definition)
            draws, _ = goals.seeded_factor_draws_kernel(inflows.size, request.simulation_paths, 1, request.seed, 0, 5.)
            draws.flags.writeable = False
            outcomes, selected = numeric.reference_funding_search_kernel(metrics, eligible, draws, summary["investable_capital"],
                inflows, outflows, summary["nominal_terminal_target"], plan["annual_fee"], plan["required_probability"],
                spec["method_code"], spec["periods_per_year"])
        else:
            selected = numeric.minimum_reference_candidate_kernel(metrics, eligible)
        result["candidates"] = [{"id": f"frontier-{i}", "solver_status": frontier_moments.STATUS_NAMES[int(statuses[i])],
            "hard_constraints_pass": bool(eligible[i]), "risk_level": int(levels[i]) if 1 <= levels[i] <= 5 else None,
            "expected_return": float(metrics[i, 0]) if np.isfinite(metrics[i, 0]) else None,
            "volatility": float(metrics[i, 1]) if np.isfinite(metrics[i, 1]) else None,
            "search_probability": float(outcomes[i, 0]) if outcomes is not None and np.isfinite(outcomes[i, 0]) else None,
            "search_probability_lower": float(outcomes[i, 1]) if outcomes is not None and np.isfinite(outcomes[i, 1]) else None}
            for i in range(statuses.size)]
        if selected < 0:
            result.update(status="no_validated_candidate_in_search", blockers=["当前授权和有限参考搜索中未找到达标候选；可调整预算或继续实际CMA研究。"])
            if outcomes is not None:
                diagnostic = numeric.diagnostic_reference_candidate_kernel(metrics, eligible, outcomes)
                if diagnostic >= 0:
                    candidate = {**result["candidates"][diagnostic], "weights": dict(zip(ids, weights[diagnostic].tolist(), strict=True))}
                    candidate["content_hash"] = digest_json(candidate)
                    result["adjustment_diagnosis"] = {"candidate": candidate,
                        "validation": validate_fixed_candidate(metrics[diagnostic], definition, prepared, request, spec),
                        "purpose": "fixed_candidate_capital_diagnostic_only",
                        "selection_rule": "maximum_search_probability_then_minimum_risk_then_stable_id"}
            return result
        chosen = {**result["candidates"][selected], "weights": dict(zip(ids, weights[selected].tolist(), strict=True))}
        chosen["content_hash"] = digest_json(chosen)
        result["selected_candidate"] = chosen
        if outcomes is not None:
            result["validation"] = validate_fixed_candidate(metrics[selected], definition, prepared, request, spec)
            if not result["validation"]["within_limits"]:
                result.update(status="validation_failed", blockers=["搜索选定候选未通过独立样本验证；不能改用同一验证集重选。"])
                return result
        result.update(status="validated", minimum_tested_feasible_level=chosen["risk_level"])
        result["selection_rule"] = "minimum_volatility_then_search_probability_then_stable_node_id"
        return result
