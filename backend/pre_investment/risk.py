"""Shared historical fit plus full joint residual covariance and CMA bridges."""

import numpy as np
import pandas as pd
from backend.custom_indicators.errors import ValidationError
from backend.custom_indicators.series_provider import load_adjusted_product_series
from backend.sensitivity.service import _fit
from backend.sensitivity.kernels import validation_status_kernel
from backend.sensitivity.repository import digest_json
from backend.strategic_allocation.reference_evidence_kernels import adjacent_returns
from backend.strategic_allocation.contracts import RiskReferenceRequest
from . import risk_kernels as kernels


def load_panel(strategic, source, candidate):
    kernels.require_ready()
    baseline = source["baseline"]
    loaded = strategic.data.load_data(
        baseline, str(candidate.start_date), str(candidate.as_of), str(candidate.as_of)
    )
    strategic._validate_daily_risk_axis(
        RiskReferenceRequest(
            alloc_name=baseline["alloc_name"],
            as_of=candidate.as_of,
            start_date=candidate.start_date,
            end_date=candidate.as_of,
            periods_per_year=252,
        ),
        loaded,
    )
    allowed = {
        (p["kind"], p["product_id"].upper()): a["id"]
        for a in baseline["assets"]
        for p in a.get("products", [])
    }
    for p in candidate.products:
        if (
            p.kind != "cash"
            and allowed.get((p.kind, p.product_id.upper())) != p.asset_class_id
        ):
            raise ValidationError(
                "IMPLEMENTATION_PRODUCT_MEMBERSHIP",
                f"{p.product_id} 不属于冻结映射的大类，请先确认产品归属。",
            )
        if p.kind == "cash":
            asset = next(
                (
                    a
                    for a in source["assumptions"]["assets"]
                    if a["id"] == p.asset_class_id
                ),
                None,
            )
            if not asset or asset["role"] != "liquidity":
                raise ValidationError(
                    "IMPLEMENTATION_CASH_CLASS", "现金必须归属于政策中的流动性大类。"
                )
    starts, ends = loaded["period_starts"], loaded["dates"]
    product_arrays, fingerprints = [], {}
    common = set(zip(starts, ends))
    for p in candidate.products:
        if p.kind == "cash":
            product_arrays.append(None)
            continue
        series = load_adjusted_product_series(
            p.kind, p.product_id, strategic.data.data_dir
        )
        if series is None:
            raise ValidationError(
                "IMPLEMENTATION_NAV_MISSING", f"{p.product_id} 缺少真实复权净值。"
            )
        frame = series.frame[["date", "value"]].sort_values("date")
        frame = frame.loc[
            (frame["date"] >= pd.Timestamp(candidate.start_date))
            & (frame["date"] <= pd.Timestamp(candidate.as_of))
        ]
        if frame["date"].duplicated().any():
            raise ValidationError("IMPLEMENTATION_NAV_DUPLICATE", "产品净值日期重复。")
        dates = frame["date"].dt.strftime("%Y-%m-%d").tolist()
        values = np.ascontiguousarray(
            frame["value"].to_numpy(dtype=np.float64).reshape(-1, 1)
        )
        values.flags.writeable = False
        returns = adjacent_returns(values)
        index = {key: i for i, key in enumerate(zip(dates[:-1], dates[1:]))}
        common.intersection_update(index)
        product_arrays.append((returns, index))
        fingerprints[f"{p.kind}:{p.product_id}"] = series.fingerprint
    intervals = [key for key in zip(starts, ends) if key in common]
    if len(intervals) < 60 or len(intervals) > 5000:
        raise ValidationError(
            "IMPLEMENTATION_COMMON_SAMPLE",
            "需要 60–5000 个产品与代理共享的相同日收益区间；缺失不补零。",
        )
    positions = {key: i for i, key in enumerate(zip(starts, ends))}
    # One explicit alignment allocation. Shared panels remain read-only afterwards.
    factors = np.empty((len(intervals), len(baseline["assets"])))
    products = np.empty((len(intervals), len(candidate.products)))
    for t, key in enumerate(intervals):
        factors[t] = loaded["returns"][positions[key]]
        for j, item in enumerate(product_arrays):
            products[t, j] = 0.0 if item is None else item[0][item[1][key], 0]
    factors.flags.writeable = False
    products.flags.writeable = False
    metadata = {
        "class_source_hash": loaded["source_hash"],
        "product_fingerprints": fingerprints,
        "dates": [x[1] for x in intervals],
        "period_starts": [x[0] for x in intervals],
        "alignment": "strict_identical_adjacent_intervals",
        "frequency": "SSE_daily",
        "currency": source["assumptions"]["currency"],
        "price_basis": "adjusted_nav_return_proxy",
        "historical_pit_eligible": False,
        "alignment_copy_bytes": factors.nbytes + products.nbytes,
    }
    metadata["content_hash"] = digest_json(metadata)
    return factors, products, metadata


def fit_joint(factors, products, candidate, source):
    # Cash is deterministic, outside the regression intercept. No ridge fallback.
    factor_cov = kernels.covariance_kernel(factors, 0, factors.shape[0] * 2 // 3, 252.0)
    noncash_factors = [
        i
        for i, a in enumerate(source["assumptions"]["assets"])
        if a["role"] != "liquidity" or factor_cov[i, i] > 1e-16
    ]
    noncash_products = [i for i, p in enumerate(candidate.products) if p.kind != "cash"]
    if not noncash_factors or not noncash_products:
        raise ValidationError(
            "IMPLEMENTATION_FACTOR_AXIS", "至少需要一个非现金研究因子与产品。"
        )
    split = factors.shape[0] * 2 // 3
    # Existing OLS expects writable contiguous arrays: one documented boundary copy.
    x = np.ascontiguousarray(factors[:, noncash_factors])
    y = np.ascontiguousarray(products[:, noncash_products])
    rank, condition = kernels.design_diagnostics_kernel(
        kernels.covariance_kernel(x, 0, split, 1.0)
    )
    coefficients, stats, _, _ = _fit(x, y, split)
    status = validation_status_kernel(
        coefficients,
        stats,
        max(30, len(noncash_factors) * 5),
        20,
        float(candidate.min_validation_r2),
    )
    if np.any(status):
        raise ValidationError(
            "IMPLEMENTATION_EXPOSURE_UNAVAILABLE",
            "联合暴露的训练或留出验证未通过；请扩大样本或复核代理，不会填零或加岭修补。",
        )
    residuals = kernels.residuals_kernel(x, y, coefficients)
    small_cov = kernels.covariance_kernel(residuals, 0, split, 252.0)
    beta = np.zeros((len(candidate.products), factors.shape[1]))
    residual_cov = np.zeros((len(candidate.products), len(candidate.products)))
    for i, p in enumerate(noncash_products):
        for j, c in enumerate(noncash_factors):
            beta[p, c] = coefficients[i, j]
        for j, q in enumerate(noncash_products):
            residual_cov[p, q] = small_cov[i, j]
    # Physical cash has zero modeled return/risk, rather than inheriting a risky
    # liquidity proxy. Its budget class is still explicit and independently gated.
    return (
        beta,
        residual_cov,
        {
            "train_observations": split,
            "validation_observations": factors.shape[0] - split,
            "validation_r2": stats[:, 3].tolist(),
            "min_validation_r2": candidate.min_validation_r2,
            "training_design_rank": rank,
            "training_design_condition": condition,
            "design_eigenvalue_relative_cutoff": 1e-12,
            "residual_assumption": "D=0_forward_declared_not_estimated_orthogonality",
            "residual_covariance": "full_joint_training_covariance",
            "alpha_forecast": "zero_active_premium",
            "historical_intercepts_not_forecasts": coefficients[:, -1].tolist(),
            "fit_boundary_copy_bytes": x.nbytes + y.nbytes,
        },
    )


def evaluate_models(source, candidate, beta, residual_cov):
    policy = source["policy"]
    assumptions = source["assumptions"]
    mode = policy.get("mode", "single")
    names = [a["id"] for a in assumptions["assets"]]
    definitions = []
    if mode != "compatible_all_models":
        definitions.append(
            {
                "id": policy.get("cma_id") or "parameter_average",
                "name": "主研究假设",
                "assumptions": assumptions,
                "covariance": policy["covariance"],
                "enforced": True,
            }
        )
    for row in policy.get("multi_cma", {}).get("sources", []):
        definitions.append(
            {
                "id": row["cma_id"],
                "name": row["name"],
                "assumptions": row["assumptions"],
                "covariance": row["covariance"],
                "enforced": mode == "compatible_all_models",
            }
        )
    weights = np.asarray([p.weight for p in candidate.products], dtype=np.float64)
    saa = np.asarray(
        [a["base_weight"] for a in source["baseline"]["assets"]], dtype=np.float64
    )
    target = np.asarray([source["target"][a] for a in names], dtype=np.float64)
    mandate = policy["mandate"]
    results = []
    arrays = {}
    for index, model in enumerate(definitions):
        means = np.asarray(
            [a["annual_return"] for a in model["assumptions"]["assets"]],
            dtype=np.float64,
        )
        cov = np.asarray(model["covariance"], dtype=np.float64)
        metrics, product_cov, product_means, exposure = kernels.product_risk_kernel(
            beta, residual_cov, cov, means, weights, saa, target
        )
        violations = []
        if metrics[1] > mandate["max_volatility"] + 1e-10:
            violations.append("产品组合的实际预期波动超过授权上限。")
        if metrics[2] > mandate["max_tracking_error"] + 1e-10:
            violations.append("相对 SAA 的总主动风险超过授权上限。")
        floor = mandate.get("effective_target_return")
        if (
            floor is None
            and mandate.get("objective_kind", "absolute_return") == "absolute_return"
        ):
            floor = mandate["target_return"]
        if (
            candidate.return_basis_confirmed
            and floor is not None
            and metrics[0] < floor - 1e-10
        ):
            violations.append("产品桥接预期收益低于授权下限。")
        benchmark = mandate.get("benchmark")
        benchmark_result = None
        if benchmark:
            bench = np.asarray(
                [benchmark["weights"][a] for a in names], dtype=np.float64
            )
            bm = kernels.product_risk_kernel(
                beta, residual_cov, cov, means, weights, bench, target
            )[0]
            # A one-product deterministic class vector reuses the same mean operator.
            bench_mean = kernels.product_risk_kernel(
                bench.reshape(1, -1),
                np.zeros((1, 1)),
                cov,
                means,
                np.ones(1),
                bench,
                bench,
            )[0][0]
            excess = float(metrics[0] - bench_mean)
            benchmark_result = {
                "tracking_error": float(bm[2]),
                "expected_excess_return": excess,
            }
            if bm[2] > benchmark["max_tracking_error"] + 1e-10:
                violations.append("产品相对授权基准的主动风险超过上限。")
            if (
                candidate.return_basis_confirmed
                and excess < benchmark["target_excess_return"] - 1e-10
            ):
                violations.append("产品相对授权基准的收益低于目标。")
        results.append(
            {
                "model_id": model["id"],
                "name": model["name"],
                "enforced": model["enforced"],
                "status": "failed" if violations else "passed",
                "violations": violations,
                "expected_return": (
                    float(metrics[0]) if candidate.return_basis_confirmed else None
                ),
                "volatility": float(metrics[1]),
                "total_active_risk": float(metrics[2]),
                "implementation_tracking_error": float(metrics[3]),
                "residual_risk": float(metrics[4]),
                "exposure": exposure.tolist(),
                "benchmark": benchmark_result,
                "max_volatility": mandate["max_volatility"],
                "max_tracking_error": mandate["max_tracking_error"],
            }
        )
        arrays[f"product_covariance_{index}"] = product_cov
        arrays[f"product_means_{index}"] = product_means
    return results, arrays
