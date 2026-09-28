"""Read-only SAA frontier diagnostics; never changes candidate/adoption gates.

Reuse the warmed matrix frontier. The target is an overlay, not a constraint
on this plot: an unreachable target must not erase the opportunity set.
"""
import numpy as np

from backend import frontier_moments as frontier
from backend.custom_indicators.errors import ValidationError
from .cma_application import frozen_assumptions, frozen_numeric_inputs
from .mandate_inputs import cash_success_required, effective_cash_floor, effective_return_floor
from .return_targets import requirements, required_mean, check_return, target_curve
from . import kernels
from .planning import funding_inputs
from .scope_facts import cma_scope_weight_limits
from .compatibility_solver import solve
from . import compatibility_kernels


def _curve(means, covariance, ids, limits, groups):
    bounds = np.asarray([[limits[x]["min_weight"], limits[x]["max_weight"]] for x in ids], dtype=np.float64)
    members = np.asarray([[float(x in g["assets"]) for x in ids] for g in groups], dtype=np.float64).reshape(len(groups), len(ids))
    lows = np.asarray([g["lo"] for g in groups], dtype=np.float64)
    highs = np.asarray([g["hi"] for g in groups], dtype=np.float64)
    result = frontier.solve_frontier(means, covariance, bounds, members, lows, highs, point_count=41)
    points = [{"volatility": float(result[2][i, 0]), "expected_return": float(result[2][i, 1]),
               "weights": dict(zip(ids, result[1][i].tolist())), "status": "optimal_to_tolerance"}
              if status == 0 else {"volatility": None, "expected_return": None, "weights": {},
                                    "status": frontier.STATUS_NAMES[int(status)]}
              for i, status in enumerate(result[3])]
    return {"points": points, "status": frontier.STATUS_NAMES[int(result[11])],
            "complete": bool(np.all(result[3] == 0)),
            "max_return": float(result[7][2, 1]) if result[8][2] == 0 else None}


def diagnose(service, request, mandate, cma):
    frontier.require_ready()
    common = request.mode == "compatible_all_models"
    # A common-model study has no single risk coordinate. Plot each original
    # model explicitly; equal-weight moments cannot represent the common gate.
    models = [source["artifact"] for source in cma["multi_cma"]["sources"]] if common else [cma]
    views, model_means, model_risks = [], [], []
    for model in models:
        definition = frozen_assumptions(model)
        ids = [a["id"] for a in definition["assets"]]
        means, covariance, _ = frozen_numeric_inputs(model, service.artifacts)
        model_means.append(means)
        model_risks.append(covariance)
        base_limits = {x: {"min_weight": 0., "max_weight": 1.} for x in ids}
        reference = _curve(means, covariance, ids, base_limits, [])
        constraint_error = None
        try:
            groups, limits = service._constraints(request, definition, mandate, cma_scope_weight_limits(cma))
            configured = _curve(means, covariance, ids, limits, groups)
        except ValidationError as exc:
            if exc.code not in {"MANDATE_LIMIT_CONFLICT", "MANDATE_LIQUIDITY_CONFLICT",
                                "SAA_CASH_ASSETS_MISSING", "SAA_LIQUID_ASSETS_MISSING"}:
                raise
            groups, limits = [], {}
            constraint_error = exc.message
            configured = {"points": [], "status": "constraint_conflict", "complete": False, "max_return": None}
        returns = requirements(mandate, means=means, ids=ids)
        target = returns["arithmetic_floor"]
        for point in configured["points"]:
            if point["status"] == "optimal_to_tolerance":
                point["return_check"] = check_return(returns, point["expected_return"], point["volatility"])
        funding = funding_inputs(mandate)
        cash_floor = effective_cash_floor(mandate, funding[0]["required_liquid_weight"] if funding else 0.)
        views.append({"id": model.get("id") or "parameter-average", "name": model["definition"]["name"],
                      "cma_hash": model["content_hash"],
                      "moment_basis": "parameter_average" if request.mode == "parameter_average" else "source_model",
                      "reference": reference, "configured": configured, "constraint_error": constraint_error,
                      "limits": limits, "groups": groups, "cash_floor": cash_floor,
                      "target_return": target, "volatility_cap": mandate["max_volatility"],
                      "return_requirements": returns,
                      "target_curve": target_curve(returns, reference["points"] + configured["points"])})
    target_check = _target_check(request, views, model_means, model_risks, ids, mandate)
    return {"mode": request.mode, "views": views, "target_check": target_check, "execution": frontier.execution_audit(),
            "basis": "frozen_cma_arithmetic_moments_weight_and_liquidity_constraints",
            "additional_checks": {"benchmark": bool(mandate.get("benchmark")),
                                  "funding": cash_success_required(mandate), "all_models": common},
            "research_only": True}


def _target_check(request, views, model_means, model_risks, ids, mandate):
    """Check the continuous target region, never infer failure from plot samples.

    The common mode uses one weight vector across every frozen model. Benchmark
    and funding remain separate candidate checks, as on the displayed frontier.
    """
    if any(v["constraint_error"] or v["configured"]["status"] == "infeasible_certified" for v in views):
        return {"status": "infeasible", "reason": "constraint_conflict"}
    view = views[0]
    limits, groups = view["limits"], view["groups"]
    means = model_means[0][None, :] if len(views) == 1 else np.asarray(model_means, dtype=np.float64)
    risks = model_risks[0][None, :, :] if len(views) == 1 else np.asarray(model_risks, dtype=np.float64)
    bounds = np.asarray([[limits[x]["min_weight"], limits[x]["max_weight"]] for x in ids], dtype=np.float64)
    members = np.asarray([[float(x in g["assets"]) for x in ids] for g in groups], dtype=np.float64).reshape(len(groups), len(ids))
    lows = np.asarray([g["lo"] for g in groups], dtype=np.float64)
    highs = np.asarray([g["hi"] for g in groups], dtype=np.float64)
    if any(v["return_requirements"]["status"] != "resolved" for v in views):
        return {"status": "undetermined", "reason": "return_requirement_pending"}
    benchmark = mandate.get("benchmark")
    benchmark_weights = np.asarray([benchmark["weights"][x] for x in ids] if benchmark else [], dtype=np.float64)
    necessary = required_mean({**view["return_requirements"],
        "arithmetic_floor": effective_return_floor(mandate, None)}, 0.)
    answer = solve(means, risks, bounds, members, lows, highs, benchmark_weights,
        necessary if necessary is not None else -np.inf,
        view["volatility_cap"], benchmark["max_tracking_error"] if benchmark else 1.,
        benchmark["target_excess_return"] if benchmark else 0., np.zeros(len(views)), 0,
        max_iterations=request.solver_max_iterations)
    status = "feasible" if answer["weights"] is not None else "infeasible" if answer["status"] == "infeasible" else "undetermined"
    if answer["weights"] is not None:
        for index, v in enumerate(views):
            metrics, _ = kernels.portfolio_moments_kernel(answer["weights"], means[index], risks[index],
                np.zeros(len(ids)), 1., 0.)
            if not check_return(v["return_requirements"], metrics[0], metrics[1])["within_limits"]:
                status = "undetermined"
    # A verified plotted weight is a witness even if the separate continuous
    # search stopped without a solution. It never proves joint feasibility.
    if status == "undetermined" and len(views) == 1 and not benchmark and any(
            p.get("status") == "optimal_to_tolerance"
            and p.get("return_check", {}).get("within_limits")
            and p["volatility"] <= view["volatility_cap"] + 1e-10
            for p in view["configured"]["points"]):
        status = "feasible"
    return {"status": status, "reason": "target_outside" if status == "infeasible" else None,
            "solver": {k: v for k, v in answer.items() if k != "weights"},
            "execution": compatibility_kernels.execution_audit()}
