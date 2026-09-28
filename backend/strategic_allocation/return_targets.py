"""One return-requirement contract for objectives, frontiers and policy gates.

Required returns describe the goal, never a market forecast. Arithmetic and
compound requirements stay separate; only like-for-like numbers are compared.
"""
from __future__ import annotations

import numpy as np
from backend.custom_indicators.errors import ValidationError
from . import goal_kernels as goals
from .mandate_inputs import effective_return_floor
from .planning import funding_inputs


def requirements(definition, *, means=None, ids=None, method=0, periods=1):
    goals.require_ready()
    arithmetic = effective_return_floor(definition, None)
    compound = (float(definition["target_return"])
                if definition.get("objective_kind", "absolute_return") == "absolute_return"
                and definition.get("target_return_basis") == "annual_compound" else None)
    prepared = funding_inputs(definition)
    funding = prepared[0] if prepared else None
    status = "resolved"
    if funding:
        cash = funding["cashflow_required_return"]
        if funding["cashflow_required_return_status"] == "above_search_bound" or cash is None:
            status = "funding_return_unresolved"
        else:
            compound = cash if compound is None else max(compound, cash)
    benchmark_return = None
    if definition.get("objective_kind") == "benchmark_relative":
        benchmark = definition.get("benchmark")
        if not benchmark:
            status = "benchmark_required" if status == "resolved" else status
        elif means is None or ids is None:
            status = "benchmark_moments_required" if status == "resolved" else status
        elif set(benchmark["weights"]) != set(ids):
            raise ValidationError("MANDATE_BENCHMARK_AXIS", "基准与本次模型须使用完整一致的资产轴。")
        else:
            from .kernels import expected_excess_return_kernel
            weights = np.asarray([benchmark["weights"][x] for x in ids], dtype=np.float64)
            benchmark_return = float(expected_excess_return_kernel(weights, np.zeros(len(ids)), means))
            arithmetic = benchmark_return + float(benchmark["target_excess_return"])
    return {"arithmetic_floor": arithmetic, "compound_floor": compound,
            "benchmark_return": benchmark_return,
            "target_excess_return": (definition.get("benchmark") or {}).get("target_excess_return"),
            "volatility_cap": definition.get("max_volatility"), "status": status,
            "distribution": {"method": method, "periods_per_year": periods},
            "compound_basis": "annual_median_path_growth_before_additional_fee",
            "funding": funding, "probability_validated": False}


def required_mean(requirement, volatility):
    """The minimum arithmetic mean at this risk, not at an assumed zero risk."""
    floor = requirement["arithmetic_floor"]
    if requirement["compound_floor"] is not None:
        spec = requirement["distribution"]
        converted = float(goals.compound_required_mean_kernel(float(requirement["compound_floor"]),
            float(volatility), spec["method"], spec["periods_per_year"]))
        floor = converted if floor is None else max(floor, converted)
    return floor


def check_return(requirement, mean, volatility):
    floor = required_mean(requirement, volatility)
    valid = requirement["status"] == "resolved" and np.isfinite(mean) and np.isfinite(volatility)
    return {"within_limits": bool(valid and (floor is None or mean >= floor - 1e-10)),
            "required_arithmetic_return": floor, "arithmetic_return": float(mean),
            "compound_floor": requirement["compound_floor"], "status": requirement["status"]}


def target_curve(requirement, points):
    """Use actual frontier coordinates plus the authorized risk boundary."""
    risks = sorted({0., *(float(p["volatility"]) for p in points if p.get("volatility") is not None),
                    *([float(requirement["volatility_cap"])] if requirement["volatility_cap"] is not None else [])})
    return [{"volatility": risk, "expected_return": required_mean(requirement, risk)} for risk in risks]
