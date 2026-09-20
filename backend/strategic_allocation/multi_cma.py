"""Parameter averaging of immutable CMA versions and source-model diagnostics.

The derived moments belong to one policy study; they are never published as a
standalone CMA or represented by the first source's identity. Only the small
M x N moments are materialized at this boundary; historical panels stay mapped.
"""
from __future__ import annotations

import copy
import json
import numpy as np

from backend.custom_indicators.errors import ConflictError, ValidationError
from backend.factor_research.repository import clean
from backend.sensitivity.repository import digest_json
from . import cma_model_kernels, cma_statistical_kernels, kernels, multi_cma_kernels
from .cma_application import frozen_assumptions, frozen_model_lineage, frozen_numeric_inputs
from .planning import diagnose_funding, require_goal_checks
from .reference_inputs import automatic_research_day
from .mandate_inputs import cash_success_required

METRICS = ("expected_return", "volatility", "conservative_return", "nominal_utility", "robust_utility")
# Explicit admission bounds, not a promise about runtime. The ordinary maximum
# 20-source/5-candidate/2,000-path/10-year study uses 25.2M path-month units.
# Funding evaluates central and conservative paths, at most twice these units.
MAX_DIAGNOSTIC_PATH_MONTHS = 50_000_000
# ArtifactRepository's read ceiling is 8 MB. Reserve 1 MB for the envelope,
# names and immutable repository metadata; reject before any baseline write.
MAX_FROZEN_PAYLOAD_BYTES = 7_000_000
LIMITATIONS = [
    "参数平均只按融合后的收益和协方差判断政策约束；原模型失败是交叉诊断，不代表所有模型均通过。",
    "模型分歧单独展示，不加到参数平均的资产风险中；模型权重不是经过校准的预测概率。",
    "融合均值不确定性未联合校准；加权半宽沿用各来源声明，缺失估计不表示没有不确定性。",
    "资金测算使用融合矩的既有对数正态代理，不随机抽取原模型，不能视为原生混合分布模拟。",
]
COMPATIBILITY_LIMITATIONS = [
    "共同约束仅覆盖所选有限模型，不保证未知市场模型或未来实际结果。",
    "逐模型风险与收益全部通过才可采纳；资金目标另用各模型独立验证，未通过不证明所有权重都无解。",
    "政策内等权矩只作展示参考，不能替代任何原模型的风险门禁。研究权重不适用于共同约束模式。",
]


def request_payload(request):
    # Preserve the exact legacy request representation and its preview hash.
    excluded = {"compatibility_objective", "solver_max_iterations"} if request.mode != "compatible_all_models" else set()
    if request.mode == "single":
        excluded.update({"mode", "cma_refs"})
    if request.uncertainty_set == "box":
        excluded.update({"uncertainty_set", "uncertainty_confidence", "uncertainty_approximation_acknowledged"})
    return request.model_dump(mode="json", exclude=excluded)


def require_payload_budget(payload):
    size = len(json.dumps(clean(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    if size > MAX_FROZEN_PAYLOAD_BYTES:
        raise ValidationError("SAA_MULTI_CMA_PAYLOAD_BUDGET", "融合研究的冻结证据超过 7 MB 保存预算；请减少所选来源或缩短来源说明。不会写入无法读取的政策。")
    return size


def require_calculation_budget(request, mandate, paths):
    sources = len(request.cma_refs)
    common = request.mode == "compatible_all_models"
    candidates = sources + 2 if common else 5 if request.risk_budget is not None else 4
    models = sources if common else sources + 1
    months = mandate["horizon_years"] * 12
    units = models * candidates * paths * months if cash_success_required(mandate) else 0
    if units > MAX_DIAGNOSTIC_PATH_MONTHS:
        raise ValidationError("SAA_MULTI_CMA_CALCULATION_BUDGET",
            "融合及原模型资金交叉诊断超过 5000 万路径月计算预算；请显式减少来源，或重新确认较小路径预算的目标。系统不会自动减少路径或丢弃模型。")
    return {"source_models": sources, "evaluated_models": models, "candidate_slots": candidates,
            "paths_per_candidate": paths, "months": months, "diagnostic_path_months": units,
            "maximum_diagnostic_path_months": MAX_DIAGNOSTIC_PATH_MONTHS,
            "maximum_central_and_conservative_path_months": units * 2}


def _compatibility(artifacts):
    first = artifacts[0]
    basis = first["definition"]
    axis = [(a["id"], a["role"], a["liquidity"]) for a in basis["assets"]]
    proxy = None
    for item in artifacts:
        definition = item["definition"]
        semantics = item.get("semantics", {})
        if definition.get("schema_version") != "2.0" or any(definition.get(key) is None for key in
                ("moment_semantics", "fee_basis", "fx_hedging_basis")):
            raise ValidationError("SAA_MULTI_CMA_SEMANTICS", "参数融合只接受已明确收益矩、费用和汇率口径的 LTCMA 2.0；旧版本须先复制确认。")
        if (semantics.get("covariance_role") != "asset_return"
                or semantics.get("moment_semantics") != definition["moment_semantics"]
                or semantics.get("fee_basis") != definition["fee_basis"]
                or semantics.get("fx_hedging_basis") != definition["fx_hedging_basis"]):
            raise ValidationError("SAA_MULTI_CMA_SEMANTICS", "冻结协方差或收益语义不完整，不能猜测或混合不同风险口径。")
        if any(definition.get(key) != basis.get(key) for key in
               ("currency", "horizon_years", "return_basis", "moment_semantics", "fee_basis", "fx_hedging_basis", "as_of")):
            raise ValidationError("SAA_MULTI_CMA_BASIS", "融合来源须具有相同研究日、币种、预测期限、收益矩、费用和汇率口径；本版不自动延展旧预测。")
        if ([(a["id"], a["role"], a["liquidity"]) for a in definition["assets"]] != axis
                or any(definition.get(key) != basis.get(key) for key in ("alloc_name", "strategic_universe_id"))):
            raise ValidationError("SAA_MULTI_CMA_AXIS", "融合来源的资产定义、顺序及经济角色须完全一致，不能按同名自动对齐或补零。")
        if any(item["source_snapshot"]["lineage"].get(key) != first["source_snapshot"]["lineage"].get(key)
               for key in ("config_hash", "strategic_universe_hash", "implementation_mapping_hash")):
            raise ValidationError("SAA_MULTI_CMA_SOURCE", "融合来源绑定不同的大类定义、战略范围或实施映射，不能转移资产含义。")
        audit = item.get("model_result", {}).get("model_audit", {})
        if audit.get("covariance_role", "asset_return") != "asset_return" or "mean_estimation" in audit.get("included_uncertainty_components", []):
            raise ValidationError("SAA_MULTI_CMA_RISK_ROLE", "含均值估计风险的预测协方差不能与资产收益协方差直接平均。")
        inputs = (definition.get("model") or {}).get("proxy_inputs")
        if inputs is not None:
            # Windows may differ; asset proxy definitions may not. Direct manual
            # assumptions refer to the same economic scope without a fitted proxy.
            identity = inputs.get("assets")
            if proxy is not None and proxy != identity:
                raise ValidationError("SAA_MULTI_CMA_PROXY", "统计来源采用不同研究代理定义，不能因战略资产同名而融合。")
            proxy = identity


def resolve(service, request):
    common = request.mode == "compatible_all_models"
    cma_model_kernels.require_ready()
    cma_statistical_kernels.require_ready()
    multi_cma_kernels.require_ready()
    artifacts = []
    for ref in request.cma_refs:
        item = service.cma.require_selectable(ref.cma_id)
        if ref.content_hash != item["content_hash"]:
            raise ConflictError("SAA_MULTI_CMA_HASH", "融合来源的版本指纹已变化，请重新选择完整版本。")
        artifacts.append(item)
        # Bound retained metadata before loading another potentially large source.
        require_payload_budget({"artifacts": artifacts})
    _compatibility(artifacts)
    if artifacts[0]["definition"]["as_of"] > str(automatic_research_day(service.data.data_dir)):
        raise ValidationError("SAA_MULTI_CMA_KNOWLEDGE_CUTOFF", "融合来源晚于当前知识截止日。")
    arrays, sources = [], []
    for ref, item in zip(request.cma_refs, artifacts, strict=True):
        current = service._definition_source(item["definition"])
        if any(current["lineage"].get(key) != item["source_snapshot"]["lineage"].get(key)
               for key in ("config_hash", "nav_hash", "strategic_universe_hash", "implementation_mapping_hash")):
            raise ConflictError("SAA_CMA_SOURCE_CHANGED", "融合来源的大类或净值已变化，请重新确认长期假设；旧政策仍保留。")
        effective = frozen_assumptions(item)
        arrays.append(frozen_numeric_inputs(item, service.artifacts))
        sources.append({"cma_id": item["id"], "content_hash": item["content_hash"], "weight": None if common else float(ref.weight),
                        "name": item["name"], "as_of": item["definition"]["as_of"],
                        "definition": copy.deepcopy(item["definition"]), "assumptions": copy.deepcopy(effective),
                        "covariance": copy.deepcopy(item["covariance"]),
                        "artifact": copy.deepcopy(item), **frozen_model_lineage(item)})
        require_payload_budget({"sources": sources})
    probabilities = np.full(len(sources), 1./len(sources)) if common else np.asarray([r.weight for r in request.cma_refs], dtype=np.float64)
    means = np.stack([a[0] for a in arrays])
    covariances = np.stack([a[1] for a in arrays])
    uncertainties = np.stack([a[2] for a in arrays])
    for value in (probabilities, means, covariances, uncertainties):
        value.flags.writeable = False
    # E2 consumes the within-model covariance. The mixture's total covariance
    # (second result) belongs to E3 and is deliberately not the policy risk.
    mean, _, covariance, between = cma_model_kernels.mixture_moments_kernel(probabilities, means, covariances, False)
    uncertainty = multi_cma_kernels.weighted_half_width_kernel(probabilities, uncertainties)
    vol, corr, _ = cma_statistical_kernels.statistical_covariance_diagnostics(covariance)
    assumptions = copy.deepcopy(frozen_assumptions(artifacts[0]))
    assumptions.update(name="多 CMA 参数平均", source="所选冻结长期假设的显式参数加权", model=None,
                       risk_origin="manual", risk_reference=None, risk_reference_hash=None)
    if common:
        assumptions.update(name="共同约束的展示参考矩", source="等权矩仅用于展示；各原模型分别执行授权门禁")
    for i, asset in enumerate(assumptions["assets"]):
        asset.update(annual_return=float(mean[i]), annual_volatility=float(vol[i]), mean_uncertainty=float(uncertainty[i]))
    assumptions["correlation"] = clean(corr.tolist())
    limitations = list(COMPATIBILITY_LIMITATIONS if common else LIMITATIONS)
    uncertainty_sources = [{"cma_id": item["id"], "status": item.get("model_result", {}).get("model_audit", {}).get(
        "uncertainty_status", "declared_not_estimated")} for item in artifacts]
    if len({item["status"] for item in uncertainty_sources}) > 1:
        limitations.append("来源混合了不同的均值不确定性语义；加权半宽只用于区间稳健研究，不代表统一校准的概率置信区间。")
    multi = {"mode": request.mode, "aggregation_semantics": "all_models_required" if common else "parameter_average",
             "refs": [r.model_dump(mode="json") for r in request.cma_refs], "sources": sources,
             "effective_returns": mean.tolist(), "effective_covariance": covariance.tolist(),
             "effective_mean_uncertainty": uncertainty.tolist(), "model_disagreement": between.tolist(),
             "assumptions": assumptions, "uncertainty_status": "not_jointly_calibrated",
             "uncertainty_sources": uncertainty_sources,
             "uncertainty_rule": "weighted_declared_marginal_half_widths_not_joint_confidence",
             "covariance_role": "asset_return", "distribution_adapter": "annual_moment_proxy_approximation",
             "execution": {"moments": cma_model_kernels.execution_audit(),
                           "uncertainty": multi_cma_kernels.execution_audit()}, "warnings": limitations}
    if common:
        multi["primary_evaluation_spec"] = "each_frozen_source_model"
        multi["effective_moments_role"] = "display_reference_only"
    multi["content_hash"] = digest_json(multi)
    return {"id": None, "content_hash": multi["content_hash"], "definition": assumptions,
            "covariance": multi["effective_covariance"], "multi_cma": multi,
            "source_snapshot": artifacts[0]["source_snapshot"],
            "warnings": list(dict.fromkeys([*limitations, *(w for item in artifacts for w in item["warnings"])]))}


def validate_frozen(multi):
    modes = {"parameter_average": "parameter_average", "compatible_all_models": "all_models_required"}
    if not isinstance(multi, dict) or multi.get("mode") not in modes or multi.get("aggregation_semantics") != modes[multi["mode"]]:
        raise ValidationError("SAA_MULTI_CMA_MODE", "未知的冻结多 CMA 模式，不能按单模型继续计算。")
    if multi["mode"] == "compatible_all_models" and (multi.get("primary_evaluation_spec") != "each_frozen_source_model"
            or multi.get("effective_moments_role") != "display_reference_only"):
        raise ValidationError("SAA_MULTI_CMA_MODE", "共同约束政策缺少逐模型风险契约，不能降级为参数平均。")
    if multi.get("content_hash") != digest_json({k: v for k, v in multi.items() if k != "content_hash"}):
        raise ValidationError("SAA_MULTI_CMA_LINEAGE", "冻结融合参数与来源指纹不一致，不能补算历史。")
    sources, refs = multi.get("sources", []), multi.get("refs", [])
    if not sources or len(sources) != len(refs) or len(sources) > 20:
        raise ValidationError("SAA_MULTI_CMA_LINEAGE", "冻结融合来源不完整。")
    for source, ref in zip(sources, refs, strict=True):
        artifact = source.get("artifact", {})
        if (source.get("cma_id") != ref.get("cma_id") or source.get("weight") != ref.get("weight")
                or source.get("content_hash") != ref.get("content_hash")
                or artifact.get("id") != source["cma_id"] or artifact.get("content_hash") != source["content_hash"]
                or digest_json({k: v for k, v in artifact.items() if k != "content_hash"}) != source["content_hash"]
                or source.get("definition") != artifact.get("definition")
                or source.get("assumptions") != frozen_assumptions(artifact)
                or source.get("covariance") != artifact.get("covariance")):
            raise ValidationError("SAA_MULTI_CMA_LINEAGE", "冻结原 CMA 及有效假设不一致，已停止使用。")
    return multi["assumptions"]


def cross_model_results(multi, weights, mandate, *, penalty=1., paths=None, seed=42,
                        policy_weights=None, tracking_error_limit=None):
    """Diagnose one final weight vector under every frozen source, without refits."""
    validate_frozen(multi)
    kernels.require_ready()
    rows = []
    names = [a["id"] for a in multi["assumptions"]["assets"]]
    values = np.asarray([weights[name] for name in names], dtype=np.float64)
    for source in multi["sources"]:
        assumptions = source["assumptions"]
        means = np.asarray([a["annual_return"] for a in assumptions["assets"]], dtype=np.float64)
        uncertainty = np.asarray([a["mean_uncertainty"] for a in assumptions["assets"]], dtype=np.float64)
        covariance = np.asarray(source["covariance"], dtype=np.float64)
        metrics, contributions = kernels.portfolio_moments_kernel(values, means, covariance, uncertainty,
                                                                 float(mandate["risk_aversion"]), float(penalty))
        row = {"cma_id": source["cma_id"], "cma_hash": source["content_hash"], "name": source["name"],
               "weight": source["weight"], "metrics": dict(zip(METRICS, metrics.tolist(), strict=True)),
               "risk_contributions": clean(dict(zip(names, contributions.tolist(), strict=True))),
               "benchmark_check": None, "goal_check": None, "expected_tracking_error": None,
               "diagnostic_scope": "strategic_moments_and_funding" if paths is not None else "tactical_moments_only"}
        violations = []
        if metrics[1] > mandate["max_volatility"] + 1e-10:
            violations.append("原 CMA 下预期波动超过授权上限。")
        floor = mandate.get("effective_target_return")
        if floor is None and mandate.get("objective_kind", "absolute_return") == "absolute_return":
            floor = mandate["target_return"]
        if floor is not None and metrics[0] < floor - 1e-10:
            violations.append("原 CMA 下预期收益低于授权下限。")
        benchmark = mandate.get("benchmark")
        if benchmark:
            base = np.asarray([benchmark["weights"][name] for name in names], dtype=np.float64)
            te = kernels.expected_active_risk_kernel(values, base, covariance)
            excess = kernels.expected_excess_return_kernel(values, base, means)
            row["benchmark_check"] = {"name": benchmark["name"], "tracking_error": float(te),
                "expected_excess_return": float(excess), "target_excess_return": benchmark["target_excess_return"],
                "max_tracking_error": benchmark["max_tracking_error"]}
            if te > benchmark["max_tracking_error"] + 1e-10:
                violations.append("原 CMA 下相对基准风险超过授权上限。")
            if excess < benchmark["target_excess_return"] - 1e-10:
                violations.append("原 CMA 下相对基准收益低于授权目标。")
        if policy_weights is not None:
            base = np.asarray([policy_weights[name] for name in names], dtype=np.float64)
            te = kernels.expected_active_risk_kernel(values, base, covariance)
            row["expected_tracking_error"] = float(te)
            if te > mandate["max_tracking_error"] + 1e-10 or (tracking_error_limit is not None and te > tracking_error_limit + 1e-10):
                violations.append("原 CMA 下相对 SAA 主动风险超过上限。")
        if paths is not None:
            candidate = {"id": "cross-model", "weights": weights, "metrics": row["metrics"]}
            diagnose_funding(mandate, [candidate], paths=paths, seed=seed)
            require_goal_checks(mandate, [candidate])
            row["goal_check"] = candidate.get("goal_check")
            if row["goal_check"] and not row["goal_check"]["within_limits"]:
                violations.append("原 CMA 下资金成功率区间下界未达到目标。")
        row.update(within_limits=not violations, violations=violations)
        rows.append(row)
    return rows
