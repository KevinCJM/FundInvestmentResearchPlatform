"""Historical scope screening, using the same evidence, constraints and solvers as SAA."""
import numpy as np
from pydantic import Field, ValidationError as InputError

from backend import frontier_moments
from backend.custom_indicators.errors import IndicatorDomainError, ValidationError
from backend.product_pools.errors import ProductPoolError
from backend.product_pools.repository import ProductPoolRepository
from backend.product_pools.service import ProductPoolService, version_data_as_of
from backend.sensitivity.repository import digest_json
from . import compatibility_kernels, goal_kernels, kernels, reference_evidence_kernels
from .compatibility_solver import solve
from .contracts import PolicyRequest
from .cma_model_contracts import HistoricalCmaRequest
from .mandate_inputs import cash_success_required, has_cash_budget, require_resolved_authorization, effective_cash_floor
from .planning import funding_inputs
from .return_targets import requirements, check_return, target_curve
from .mandate_inputs import effective_return_floor
from .reference_contracts import ReferenceAsset, ReferenceInputRequest
from .reference_inputs import automatic_research_day
from .scope_facts import scope_difference, scope_facts, scope_weight_limits, SCOPE_MESSAGES


class ScopeReferenceInputs(ReferenceInputRequest):
    # A one-asset opportunity set is a point, and is valid for screening.
    assets: list[ReferenceAsset] = Field(min_length=1, max_length=30)


def _issue(code, message):
    return {"code": code, "message": message}


def _reference_comparison(service, artifact, as_of):
    """Display the exact saved scale; never refit it or use it to pass screening."""
    mandate = artifact["definition"]
    ref = (mandate.get("risk_authorization") or {}).get("risk_scale_ref")
    if not ref:
        return {"status": "not_linked"}
    try:
        # Only chart coordinates and provenance are needed, not source arrays or
        # today's default/eligibility. The repository verifies the manifest hash.
        scale = service._get(ref["id"], "risk_scale")
        if scale["content_hash"] != ref["content_hash"]:
            return {"status": "unavailable"}
        preview = scale["preview"]
        definition = preview["request_echo"]["definition"]
        evidence = preview["result"]["parameter_evidence"]
        quality = evidence.get("data_quality") or {}
        if (definition["base_currency"] != mandate["currency"]
                or definition["risk_basis_id"] != "annualized-periodic-volatility-v1"
                or definition["research_as_of"] > str(as_of)
                or mandate["as_of"] > str(as_of)
                or quality.get("annualization_method") != "arithmetic_mean_and_covariance_times_periods"
                or quality.get("periods_per_year") != 252):
            return {"status": "incompatible"}
        diagnosis = artifact.get("assessment", {}).get("reference_diagnosis") or {}
        constrained = diagnosis.get("constrained_frontier", []) if diagnosis.get("risk_scale_ref") == ref else []
        def points(values):
            return [{key: point.get(key) for key in ("volatility", "expected_return", "status")} for point in values]
        return {"status": "available", "risk_scale_ref": ref, "name": scale["name"],
                "as_of": definition["research_as_of"], "currency": definition["base_currency"],
                "sample_start": quality.get("intersection_start"), "sample_end": quality.get("intersection_end"),
                "points": points(preview["result"]["frontier"]), "constrained_points": points(constrained)}
    except (IndicatorDomainError, OSError, KeyError, TypeError):
        # An unavailable comparison must not erase the current scope's result.
        return {"status": "unavailable"}


def _product_inputs(service, request):
    pools = ProductPoolService(ProductPoolRepository(service.data.universe_dir / "product_pools.json"),
                               None, strategic_root=service.artifacts.root.parent.parent)
    if request.universe_snapshot_id:
        snapshot = pools.get_universe_snapshot(request.universe_snapshot_id)
        if snapshot.get("mandate_id") != request.mandate_id:
            raise ValidationError("SCOPE_MANDATE_MISMATCH", "此产品范围绑定了其他目标，或尚未关联目标。")
        if snapshot["research_date"] != str(request.as_of):
            raise ValidationError("SCOPE_RESEARCH_DATE", "产品范围与初筛研究日不同，请使用该范围的研究日。")
    else:
        snapshot = pools.preview_universe_snapshot({"name": "范围历史初筛", "research_date": str(request.as_of),
            "version_ids": request.product_version_ids, "excluded_product_keys": request.product_excluded_keys})
    for identifier in snapshot.get("version_ids", []):
        version = pools.get_version(identifier)
        known = version_data_as_of(version)
        if known is None or known > str(request.as_of):
            raise ValidationError("SCOPE_SELECTION_CLOCK", "产品池的筛选证据未知或晚于研究日，不能据此判断当时可选范围。")
    products = snapshot.get("products") or snapshot.get("members") or []
    if not 1 <= len(products) <= 30:
        raise ValidationError("SCOPE_ASSET_CAPACITY", "历史初筛支持 1 至 30 个产品；请显式调整范围，系统不会自动删选产品。")
    assets, proxies, limits, used_sources = [], [], {}, set()
    sources = service.cma.evidence.sources
    for product in products:
        eligible, _ = pools._member_is_eligible(product, str(request.as_of))
        if not eligible:
            raise ValidationError("SCOPE_PRODUCT_INELIGIBLE", f"产品 {product.get('name') or product['product_id']} 在研究日不可新增配置。")
        kind, identity = product["kind"], product["product_id"]
        code = product.get("code") or identity
        if kind not in ("etf", "fund"):
            raise ValidationError("SCOPE_PRODUCT_UNSUPPORTED", "当前产品历史初筛支持 ETF 和基金的复权收益，所选范围包含尚不支持的产品。")
        matches = [item for item in sources.catalog(kind=kind, query=code, limit=200)["items"]
                   if item.get("code") == code and item.get("reference_capability", {}).get("available")]
        if len(matches) != 1:
            raise ValidationError("SCOPE_PRODUCT_SOURCE", f"产品 {product.get('name') or code} 缺少唯一可用的复权收益来源。")
        item = matches[0]
        fields = item["reference_capability"]["supported_fields"]
        field = next((key for key in ("adj_nav", "close_hfq") if key in fields), None)
        if field is None:
            raise ValidationError("SCOPE_PRODUCT_SOURCE", f"产品 {product.get('name') or code} 缺少复权收益字段。")
        source_identity = (kind, item["id"], field)
        if source_identity in used_sources:
            raise ValidationError("SCOPE_PRODUCT_SOURCE_DUPLICATE", "不同产品身份指向了同一收益来源，请先核对产品代码与别名；不会重复计算或合并限额。")
        used_sources.add(source_identity)
        identifier = "product-" + digest_json([kind, identity])[:20]
        name = product.get("name") or code
        assets.append({"id": identifier, "name": name, "role": "growth", "liquidity": "liquid"})
        proxies.append(ReferenceAsset(id=identifier, name=name, asset_type="market", cash_return=None,
            rebalance="daily", components=[{"kind": kind, "series_id": item["id"], "field": field, "weight": 1.}]))
        cap = product.get("max_weight")
        limits[identifier] = {"min_weight": 0., "max_weight": 1. if cap is None else cap}
    return assets, proxies, limits


def _point(weights, means, covariance, ids):
    metrics, _ = kernels.portfolio_moments_kernel(weights, means, covariance, np.zeros(means.size), 0., 0.)
    return {"volatility": float(metrics[1]), "expected_return": float(metrics[0]),
            "weights": dict(zip(ids, weights.tolist(), strict=True))}


def _calculate(service, request, result, mandate):
    from .service import _policy_expiry
    if request.as_of > automatic_research_day(service.data.data_dir):
        raise ValidationError("SCOPE_KNOWLEDGE_CUTOFF", "研究日晚于平台知识截止日。")
    require_resolved_authorization(mandate)
    if str(request.as_of) < mandate["as_of"] or str(request.as_of) >= _policy_expiry(mandate):
        raise ValidationError("SCOPE_MANDATE_DATE", "研究日不在关联目标的有效期内。")
    if has_cash_budget(mandate) and str(request.as_of) != mandate["as_of"]:
        raise ValidationError("SCOPE_CASH_BUDGET_DATE", "金额计划与范围研究日须一致；更新时需要显式滚动现金流计划。")
    if mandate["currency"] != "CNY":
        raise ValidationError("SCOPE_CURRENCY_UNSUPPORTED", "历史初筛目前只支持人民币口径，不自动折汇。")
    funding = funding_inputs(mandate)
    if funding:
        # Reuse step 01's deterministic cash-flow calculation. Its compound
        # return requirement is display evidence, not an arithmetic frontier floor.
        result["mandate"]["funding_requirement"] = {
            "required_return": funding[0]["cashflow_required_return"],
            "status": funding[0]["cashflow_required_return_status"],
            "basis": funding[0]["required_return_basis"],
        }
    result["mandate"]["min_cash_weight"] = effective_cash_floor(mandate, funding[0]["required_liquid_weight"] if funding else 0.)
    if request.strategic_definition:
        definition = request.strategic_definition.model_dump(mode="json")
        if definition["currency"] != mandate["currency"]:
            raise ValidationError("SCOPE_CURRENCY_MISMATCH", "范围与投资目标的计价币种不同。")
        assets, proxies, limits = definition["assets"], [], {}
        for asset in assets:
            proxy = asset.get("research_proxy")
            if not proxy or (proxy["asset_type"] == "market" and not proxy["components"]):
                raise ValidationError("SCOPE_PROXY_REQUIRED", f"大类 {asset['name']} 尚未配置完整研究代理，暂时无法判断。")
            proxies.append(ReferenceAsset(id=asset["id"], name=asset["name"],
                **{k: v for k, v in proxy.items() if k != "source_labels"}))
        if mandate.get("strategic_universe_id"):
            authorised = service.scopes.get_universe(mandate["strategic_universe_id"])["definition"]
            mismatch = scope_difference(scope_facts(authorised), scope_facts(definition))
            if mismatch:
                raise ValidationError("SCOPE_MANDATE_AXIS", SCOPE_MESSAGES[mismatch])
    else:
        assets, proxies, limits = _product_inputs(service, request)
        result["additional_checks"]["product_liquidity"] = mandate["min_liquid_weight"] > 0 or mandate["max_illiquid_weight"] < 1
    ids = [a["id"] for a in assets]
    result["scope"] = {"kind": "strategic" if request.strategic_definition else "product",
                       "asset_ids": ids, "assets": [{"id": a["id"], "name": a["name"]} for a in assets]}
    if not any(proxy.asset_type == "market" for proxy in proxies):
        raise ValidationError("SCOPE_MARKET_PROXY_REQUIRED", "纯现金范围没有历史风险前沿；请增加非现金研究资产后进行历史初筛。")
    reference = ScopeReferenceInputs(name="范围历史初筛", as_of=request.as_of, assets=proxies)
    model = HistoricalCmaRequest(method="historical_statistics", asset_ids=ids, as_of=request.as_of,
        currency="CNY", source="范围历史初筛", window=request.window, proxy_inputs=reference, shrinkage=0.)
    evidence = service.cma.evidence._proxies(model)
    result["sample"] = {key: evidence["metadata"][key] for key in (
        "requested_start", "requested_end", "actual_start", "actual_end", "observations",
        "common_days", "excluded_return_periods", "missing_trading_days")}
    result["sample"]["window"] = request.window.model_dump(mode="json")
    result["provenance"] = {"source_hash": evidence["metadata"]["source_hash"],
                            "sources": evidence["metadata"]["sources"], "historical_pit_proven": False}
    reference_evidence_kernels.require_ready()
    returns = evidence["returns"]
    returns.flags.writeable = False
    means, covariance, _, _ = reference_evidence_kernels.annual_moments(returns, 0., np.int64(252))
    policy = PolicyRequest(mandate_id=request.mandate_id, cma_id="scope-preview", constraints=limits)
    # A scoped mandate is usable only after the draft facts matched above.
    raw = {"assets": assets, "alloc_name": None,
           "strategic_universe_id": mandate.get("strategic_universe_id") if request.strategic_definition else None}
    constraint_error = None
    try:
        groups, bounds_by_id = service._constraints(policy, raw, mandate, scope_weight_limits(assets))
    except ValidationError as exc:
        constraint_error = exc
        groups = []
        bounds_by_id = {identifier: limits.get(identifier, {"min_weight": 0., "max_weight": 1.}) for identifier in ids}
    bounds = np.asarray([[bounds_by_id[x]["min_weight"], bounds_by_id[x]["max_weight"]] for x in ids], dtype=np.float64)
    members = np.asarray([[float(x in group["assets"]) for x in ids] for group in groups], dtype=np.float64).reshape(len(groups), len(ids))
    lows = np.asarray([g["lo"] for g in groups], dtype=np.float64)
    highs = np.asarray([g["hi"] for g in groups], dtype=np.float64)
    with service.risk_scales.compute_slot():
        solved = frontier_moments.solve_frontier(means, covariance, bounds, members, lows, highs, point_count=41)
        result["frontier"] = {"points": [
            {"volatility": float(solved[2][i, 0]), "expected_return": float(solved[2][i, 1]),
             "weights": dict(zip(ids, solved[1][i].tolist(), strict=True)), "status": "optimal_to_tolerance"}
            if status == 0 else {"volatility": None, "expected_return": None, "weights": {},
                                 "status": frontier_moments.STATUS_NAMES[int(status)]}
            for i, status in enumerate(solved[3])],
            "status": frontier_moments.STATUS_NAMES[int(solved[11])], "complete": bool(np.all(solved[3] == 0)),
            "constraints_applied": constraint_error is None and not result["additional_checks"].get("product_liquidity", False)}
        result["constraints"] = {"asset_limits": bounds_by_id, "group_limits": groups,
            "cash_floor": next((g["lo"] for g in groups if g["id"] == "policy-cash-reserve"), 0.)}
        if constraint_error:
            if constraint_error.code in {"MANDATE_LIMIT_CONFLICT", "MANDATE_LIQUIDITY_CONFLICT", "SAA_CASH_ASSETS_MISSING", "SAA_LIQUID_ASSETS_MISSING"}:
                result["status"] = "infeasible" if request.strategic_definition else "undetermined"
            if not request.strategic_definition and constraint_error.code == "SAA_CASH_ASSETS_MISSING":
                raise ValidationError("SCOPE_PRODUCT_CASH_CLASSIFICATION", "产品范围尚无明确现金角色，无法核验现金下限；不会把可交易 ETF 或基金自动当作现金。")
            raise constraint_error
        benchmark = mandate.get("benchmark")
        if benchmark and (set(benchmark["weights"]) != set(ids) or benchmark.get("source", "explicit") == "explicit"):
            raise ValidationError("SCOPE_BENCHMARK_AXIS", "当前范围与冻结基准的资产轴或所属方案不一致，暂时只能展示历史前沿。")
        returns = requirements(mandate, means=means, ids=ids, method=1, periods=252)
        result["mandate"]["return_requirements"] = returns
        result["mandate"]["target_return"] = returns["arithmetic_floor"]
        result["target_check"]["target_return"] = returns["arithmetic_floor"]
        result["mandate"]["target_curve"] = target_curve(returns, result["frontier"]["points"])
        if returns["compound_floor"] is not None:
            result["mandate"]["funding_requirement"] = {"required_return": returns["compound_floor"],
                "status": "solved" if returns["status"] == "resolved" else returns["status"],
                "basis": "annual_effective_gross_of_model_fee"}
        benchmark_weights = np.asarray([benchmark["weights"][x] for x in ids] if benchmark else [], dtype=np.float64)
        # Solve over the continuous set. Plot samples never certify infeasibility.
        answer = solve(means[None, :], covariance[None, :, :], bounds, members, lows, highs,
            benchmark_weights, -np.inf, float(mandate["max_volatility"]),
            benchmark["max_tracking_error"] if benchmark else 1., benchmark["target_excess_return"] if benchmark else 0.,
            np.zeros(1), 0)
    result["target_check"]["solver"] = {k: v for k, v in answer.items() if k != "weights"}
    if answer["weights"] is None:
        if answer["status"] == "infeasible":
            result["status"] = "infeasible"
            raise ValidationError("SCOPE_CONSTRAINTS_INFEASIBLE", "按所选历史样本，范围无法同时满足风险上限与权重、流动性等约束。")
        raise ValidationError("SCOPE_SOLVER_UNRESOLVED", "数值求解尚未完成验证，暂时无法判断；这不表示范围不可行。")
    candidate = _point(answer["weights"], means, covariance, ids)
    offset = _point(benchmark_weights, means, covariance, ids)["expected_return"] if benchmark else 0.
    upper = -answer["lower_bound"] + offset if answer["lower_bound"] is not None else None
    result["target_check"].update(candidate=candidate, max_return_upper_bound=upper,
        max_return_under_cap=candidate["expected_return"] if answer["status"] == "converged" else None)
    floor = result["mandate"]["target_return"]
    if floor is not None and candidate["expected_return"] < floor - 1e-8:
        if upper is not None and upper < floor - 1e-8:
            result["status"] = "infeasible"
            raise ValidationError("SCOPE_RETURN_SHORTFALL", "按所选历史样本，在当前风险上限及其他约束下，收益目标超出该范围可达到的水平。")
        raise ValidationError("SCOPE_SOLVER_UNRESOLVED", "收益边界尚未完成数值验证，暂时无法判断目标是否可达到。")
    result["target_check"]["status"] = "feasible"
    candidate["return_check"] = check_return(returns, candidate["expected_return"], candidate["volatility"])
    # Preserve the verified risk/weight screen while reporting compound success
    # separately. A finite miss cannot certify the full target impossible.
    if not candidate["return_check"]["within_limits"] and not benchmark:
        witness = next((p for p in result["frontier"]["points"]
            if p.get("status") == "optimal_to_tolerance" and p.get("volatility") is not None
            and p["volatility"] <= mandate["max_volatility"] + 1e-10
            and check_return(returns, p["expected_return"], p["volatility"])["within_limits"]), None)
        if witness is not None:
            candidate = {**witness, "return_check": check_return(returns, witness["expected_return"], witness["volatility"])}
            result["target_check"]["candidate"] = candidate
    if not candidate["return_check"]["within_limits"]:
        result["target_check"]["return_status"] = "undetermined"
        raise ValidationError("SCOPE_RETURN_CHECK_REQUIRED", "风险与权重约束存在可行点；尚需核验同口径收益要求。")
    result["target_check"]["return_status"] = "passed"
    if cash_success_required(mandate):
        raise ValidationError("SCOPE_FUNDING_CHECK_REQUIRED", "历史收益与风险约束存在可行点；资金支付及期末目标成功率仍需后续验证。")
    if result["additional_checks"].get("product_liquidity"):
        raise ValidationError("SCOPE_PRODUCT_LIQUIDITY_REQUIRED", "历史收益与风险约束存在可行点；产品的现金角色、赎回与流动性限制尚未完整分类，仍需后续核验。")
    result.update(status="feasible", reasons=[_issue("SCOPE_HISTORICAL_FEASIBLE", "按所选历史样本，存在满足当前收益、风险及权重约束的配置。")])


def _funding_comparison(result):
    """Project all curves onto the same compound basis; never certify probability.

    Keep the source arithmetic frontier and full feasibility verdict intact.
    Frozen reference curves are visual context and cannot pass the current scope.
    """
    requirement = result["mandate"].get("funding_requirement")
    if result["mandate"]["target_return"] is not None or not requirement:
        return None
    goal_kernels.require_ready()

    def project(point):
        value = None
        if (point.get("status", "optimal_to_tolerance") == "optimal_to_tolerance"
                and point.get("expected_return") is not None and point.get("volatility") is not None):
            value = float(goal_kernels.funding_compound_return_kernel(
                float(point["expected_return"]), float(point["volatility"]), 1, 252))
        return {**point, "expected_return": value}

    frontier = result.get("frontier") or {}
    reference = result.get("reference_comparison") or {}
    check = result["target_check"]
    points = [project(point) for point in frontier.get("points", [])]
    candidate = project(check["candidate"]) if check.get("candidate") else None
    target = requirement["required_return"] if requirement["status"] == "solved" else None
    status = "unavailable"
    # The continuous candidate includes benchmark constraints; plotted samples do
    # not, so never promote a plot sample when benchmark constraints apply.
    eligible = ([candidate] if candidate else []) + ([] if result["additional_checks"]["benchmark"] else points)
    if (target is not None and check["status"] == "feasible" and frontier.get("constraints_applied")):
        passing = [point for point in eligible if point.get("expected_return") is not None
                   and point["volatility"] <= result["mandate"]["volatility_cap"] + 1e-8
                   and point["expected_return"] >= target - 1e-8]
        status = "passed" if passing else "no_candidate"
        if passing:
            candidate = min(passing, key=lambda point: point["volatility"])
    return {"basis": "annual_compound_median_gross_of_model_fee", "status": status,
            "target_return": target, "points": points, "candidate": candidate,
            "reference_points": [project(point) for point in reference.get("points", [])],
            "constrained_points": [project(point) for point in reference.get("constrained_points", [])],
            "distribution": {"method": "base_period_moment_match_v2", "periods_per_year": 252,
                             "serial_independence": True},
            "probability_validated": False}


def diagnose(service, request):
    artifact = service._require_active_mandate(request.mandate_id)
    mandate = artifact["definition"]
    target = effective_return_floor(mandate, None)
    result = {"status": "undetermined", "reasons": [], "research_only": True,
        "basis": "historical_annualized_periodic_arithmetic", "scope": None, "sample": None, "frontier": None,
        "mandate": {"id": artifact["id"], "content_hash": artifact["content_hash"], "target_return": target,
            "volatility_cap": mandate.get("max_volatility"), "min_cash_weight": mandate.get("min_cash_weight", 0.)},
        "target_check": {"status": "undetermined", "target_return": target, "volatility_cap": mandate.get("max_volatility"),
            "candidate": None, "max_return_under_cap": None, "max_return_upper_bound": None},
        "additional_checks": {"funding": cash_success_required(mandate), "benchmark": bool(mandate.get("benchmark"))},
        "limitations": ["历史均值与协方差初筛，不是未来收益承诺，也不替代 LTCMA、SAA 及资金路径验证。",
                        "指数使用所选指数值；ETF/基金使用复权历史收益，不自动补分红或转换币种。"],
        "execution": {"frontier": frontier_moments.execution_audit(), "feasibility": compatibility_kernels.execution_audit()}}
    try:
        _calculate(service, request, result, mandate)
    except (IndicatorDomainError, ProductPoolError) as exc:
        result["reasons"] = [_issue(exc.code, exc.message)]
    except InputError:
        result["reasons"] = [_issue("SCOPE_PROXY_INVALID", "研究代理或产品权重不完整，请核对配置后重试。")]
    except (ValueError, np.linalg.LinAlgError):
        result["reasons"] = [_issue("SCOPE_NUMERICAL_UNRESOLVED", "历史矩阵或求解结果未通过数值验证，暂时无法判断。")]
    result["reason_code"] = result["reasons"][0]["code"] if result["reasons"] else None
    if result["status"] == "infeasible":
        result["target_check"]["status"] = "infeasible"
    result["reference_comparison"] = _reference_comparison(service, artifact, request.as_of)
    try:
        result["funding_comparison"] = _funding_comparison(result)
    except (ValueError, np.linalg.LinAlgError):
        # A failed display projection does not invalidate the original evidence.
        result["funding_comparison"] = None
    return result
