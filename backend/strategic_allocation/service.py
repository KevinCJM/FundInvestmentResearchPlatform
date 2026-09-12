"""Goals -> economic classes -> explicit CMA -> immutable SAA policy.

Previews are pure reads/calculations. Confirmation repeats those calculations;
only their matching results may enter the existing managed artifact store.
"""
from __future__ import annotations

import copy
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.custom_indicators.errors import ConflictError, ValidationError
from backend.market_data import resolve_market_data_file
from backend.research_input_checks import return_quality
from backend.sensitivity.repository import ArtifactRepository, digest_json
from backend.strategy import equal_weights
from backend.tactical_allocation.contracts import AssetLimit
from backend.tactical_allocation.data import TacticalAllocationData, warm_tactical_data
from backend.tactical_allocation.repository import TacticalAllocationRepository
from . import kernels
from .contracts import (
    CmaRequest, MandateRequest, PolicyRequest, PublishCmaRequest,
    PublishPolicyRequest, RiskReferenceRequest,
)

METHODS = (
    ("minimum-risk", "候选中较低风险"),
    ("nominal-utility", "名义预期效用"),
    ("robust-utility", "区间稳健效用"),
    ("maximum-return", "约束内较高预期收益"),
)
METRICS = ("expected_return", "volatility", "conservative_return", "nominal_utility", "robust_utility")


def _hashed(payload: dict) -> dict:
    return {**payload, "preview_hash": digest_json(payload)}


def _finite_list(values: np.ndarray) -> list:
    return [float(value) if np.isfinite(value) else None for value in values]


class StrategicAllocationService:
    def __init__(self, root: Path, data_dir: Path, *, universe_dir: Path | None = None,
                 tactical_repository: TacticalAllocationRepository | None = None):
        self.artifacts = ArtifactRepository(Path(root) / "strategic_allocation" / "artifacts")
        self.data = TacticalAllocationData(data_dir, universe_dir=universe_dir)
        self.baselines = tactical_repository or TacticalAllocationRepository(root)

    def warm(self) -> dict:
        warm_tactical_data()
        return kernels.warm_strategic_kernels()

    def _get(self, identifier: str, artifact_type: str) -> dict:
        item = self.artifacts.get(identifier, "series")
        if item.get("artifact_type") != artifact_type:
            raise ValidationError("SAA_VERSION_TYPE", "所选目标或假设的版本类型不匹配。")
        return item

    def get_mandate(self, identifier: str) -> dict:
        return self._get(identifier, "investment_mandate")

    def get_cma(self, identifier: str) -> dict:
        return self._get(identifier, "capital_market_assumptions")

    def catalog(self) -> dict:
        mandates, assumptions = [], []
        for summary in self.artifacts.list("series"):
            item = self.artifacts.get(summary["id"], "series")
            common = {key: item[key] for key in ("id", "name", "created_at", "content_hash")}
            if item.get("artifact_type") == "investment_mandate":
                mandates.append({**common, "definition": item["definition"]})
            elif item.get("artifact_type") == "capital_market_assumptions":
                assumptions.append({**common, **{key: item["definition"][key] for key in
                    ("alloc_name", "as_of", "currency", "horizon_years")}})
        policies = [{key: item[key] for key in ("id", "name", "as_of", "created_at", "content_hash", "alloc_name")}
                    for item in self.baselines.list_baselines() if item.get("policy")]
        return {**self.data.catalog(), "mandates": mandates, "assumptions": assumptions, "policies": policies}

    def save_mandate(self, request: MandateRequest) -> dict:
        return self.artifacts.save("series", {"artifact_type": "investment_mandate", "name": request.name,
            "definition": request.model_dump(mode="json"), "research_only": True})

    def _source(self, alloc_name: str, as_of: str) -> dict:
        kernels.require_ready()
        frame, _, _ = self.data._configuration(alloc_name)
        names = frame["asset_name"].unique().tolist()
        # Equal weights here only let the existing source reader describe the
        # real class baskets. They are not a recommended or adopted policy.
        weights = equal_weights(len(names))
        source = self.data.create_baseline({"alloc_name": alloc_name, "name": "长期假设的数据来源",
            "as_of": as_of, "weights": dict(zip(names, weights, strict=True))})
        seen = set()
        for asset in source["assets"]:
            for product in asset["products"]:
                key = (product["kind"], product["product_id"].upper())
                if key in seen:
                    raise ValidationError("SAA_DUPLICATE_CLASS_PRODUCT", "同一产品跨大类出现，请先明确唯一预算归属。")
                seen.add(key)
        return source

    def _validate_daily_risk_axis(self, request: RiskReferenceRequest, data: dict) -> None:
        if request.periods_per_year != 252:
            raise ValidationError("SAA_RISK_FREQUENCY", "历史风险参考当前只支持 SSE 日频共同净值；年化周期必须为 252。需要周/月频时请先使用显式重采样能力。")
        path = resolve_market_data_file("trade_day_df.parquet", self.data.data_dir)
        if not path.is_file():
            raise ValidationError("SAA_RISK_CALENDAR_REQUIRED", "缺少 SSE 交易日日历，无法证明日频风险样本连续；请先补齐交易日数据。")
        try:
            calendar = pd.read_parquet(path, columns=["exchange", "cal_date", "is_open"])
        except (OSError, ValueError, KeyError) as exc:
            raise ValidationError("SAA_RISK_CALENDAR_INVALID", "SSE 交易日日历无法读取，不能继续年化历史风险。") from exc
        calendar = calendar.loc[(calendar["exchange"].astype(str).str.upper() == "SSE")
                                & (pd.to_numeric(calendar["is_open"], errors="coerce") == 1)].copy()
        raw_dates = calendar["cal_date"]
        if pd.api.types.is_datetime64_any_dtype(raw_dates):
            parsed = pd.to_datetime(raw_dates, errors="coerce").dt.normalize()
        else:
            compact = raw_dates.astype(str).str.replace("-", "", regex=False).str[:8]
            parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce").dt.normalize()
        expected = pd.DatetimeIndex(parsed.dropna().unique()).sort_values()
        expected = expected[(expected >= pd.Timestamp(request.start_date)) & (expected <= pd.Timestamp(request.end_date))]
        if expected.empty:
            raise ValidationError("SAA_RISK_CALENDAR_RANGE", "所选风险样本区间在 SSE 交易日日历中没有开放日。")
        observed = pd.DatetimeIndex(pd.to_datetime([data["period_starts"][0], *data["dates"]])).normalize().unique().sort_values()
        missing = expected.difference(observed)
        unexpected = observed.difference(expected)
        if len(missing) or len(unexpected):
            diagnostics = ([{"code": "missing_trading_day", "date": stamp.strftime("%Y-%m-%d")} for stamp in missing[:20]]
                           + [{"code": "unexpected_observation_day", "date": stamp.strftime("%Y-%m-%d")} for stamp in unexpected[:20]])
            raise ValidationError("SAA_RISK_GAPPED_DATES",
                f"历史风险样本与 SSE 交易日不连续：缺少 {len(missing)} 个开放日，含 {len(unexpected)} 个非开放日观察；不能按 252 日频年化。",
                diagnostics=diagnostics)

    def risk_reference(self, request: RiskReferenceRequest) -> dict:
        source = self._source(request.alloc_name, str(request.as_of))
        data = self.data.load_data(source, str(request.start_date), str(request.end_date), str(request.as_of))
        self._validate_daily_risk_axis(request, data)
        assets = [item["id"] for item in source["assets"]]
        quality = return_quality(data["returns"], data["dates"], assets)
        if quality["issues"]:
            raise ValidationError("SAA_NAV_QUALITY", quality["issues"][0]["message"], diagnostics=quality["issues"])
        if data["lineage"]["excluded_incomplete_dates"]:
            raise ValidationError("SAA_RISK_GAPPED_DATES", "共同净值日期存在缺口，不能将跨期收益当成连续日收益年化；请调整资产或样本。")
        _, vol, corr, means = kernels.historical_risk_kernel(data["returns"], request.shrinkage, request.periods_per_year)
        return _hashed({"request": request.model_dump(mode="json"), "assets": assets,
            "volatility": vol.tolist(), "correlation": corr.tolist(), "historical_mean": means.tolist(),
            "observations": len(data["dates"]), "lineage": data["lineage"], "source_hash": data["source_hash"],
            "method": "sample_covariance_with_fixed_diagonal_shrinkage", "execution": kernels.execution_audit(),
            "warnings": [*data["reasons"], "历史风险只是参考，不是未来风险承诺；历史均值不自动转为长期预期。",
                         "收缩强度由研究员指定，不是自动估计的 Ledoit–Wolf 系数。"]})

    def _cma_calculation(self, request: CmaRequest) -> tuple[dict, dict]:
        source = self._source(request.alloc_name, str(request.as_of))
        names = [asset["id"] for asset in source["assets"]]
        if [asset.id for asset in request.assets] != names:
            raise ValidationError("SAA_CMA_AXIS", "长期假设的资产及顺序须与所选大类一致；请重新加载分类。")
        vol = np.asarray([a.annual_volatility for a in request.assets], dtype=np.float64)
        corr = np.asarray(request.correlation, dtype=np.float64)
        covariance, min_eigenvalue = kernels.cma_covariance_kernel(vol, corr)
        reference = None
        if request.risk_origin == "historical_reference":
            reference = self.risk_reference(request.risk_reference)
            if reference["preview_hash"] != request.risk_reference_hash:
                raise ConflictError("SAA_RISK_REFERENCE_CHANGED", "历史风险来源已变化，请重新读取风险参考。")
            if reference["volatility"] != vol.tolist() or reference["correlation"] != request.correlation:
                raise ValidationError("SAA_RISK_REFERENCE_EDITED", "风险数值已被人工修改，请明确改为人工风险假设，不沿用原参考认证。")
        payload = {"definition": request.model_dump(mode="json"), "source_snapshot": source,
            "covariance": covariance.tolist(), "min_correlation_eigenvalue": float(min_eigenvalue),
            "risk_reference": reference, "execution": kernels.execution_audit(),
            "warnings": [*source["pit"]["reasons"], "经济角色、流动性与同币种总收益口径由研究员确认，不是系统校准的宏观暴露。",
                         "预期收益与均值不确定半宽是研究假设；不确定半宽不是波动率或统计置信区间。"]}
        return _hashed(payload), {"covariance": covariance}

    def preview_cma(self, request: CmaRequest) -> dict:
        return self._cma_calculation(request)[0]

    def publish_cma(self, body: PublishCmaRequest) -> dict:
        preview, arrays = self._cma_calculation(body.request)
        if preview["preview_hash"] != body.preview_hash:
            raise ConflictError("SAA_CMA_PREVIEW_CHANGED", "假设或来源已变化，请重新验证后确认保存。")
        return self.artifacts.save("series", {"artifact_type": "capital_market_assumptions", "name": body.request.name,
            **preview, "research_only": True}, arrays)

    @staticmethod
    def _constraints(request: PolicyRequest, definition: dict, mandate: dict) -> tuple[list, dict]:
        names = [asset["id"] for asset in definition["assets"]]
        if set(request.constraints) - set(names):
            raise ValidationError("SAA_CONSTRAINT_AXIS", "约束包含不属于当前假设的资产。")
        constraints = {name: (request.constraints.get(name) or AssetLimit()).model_dump() for name in names}
        groups, seen = [], set()
        for raw in request.group_limits:
            if raw.id.startswith("policy-") or raw.id in seen or len(set(raw.assets)) != len(raw.assets) or set(raw.assets) - set(names) or raw.lo > raw.hi:
                raise ValidationError("SAA_GROUP_INVALID", "分组名称、成员或边界无效；policy- 前缀保留给目标约束。")
            groups.append(raw.model_dump())
            seen.add(raw.id)
        liquid = [a["id"] for a in definition["assets"] if a["liquidity"] == "liquid"]
        illiquid = [a["id"] for a in definition["assets"] if a["liquidity"] == "illiquid"]
        if mandate["min_liquid_weight"] > 0 and not liquid:
            raise ValidationError("SAA_LIQUID_ASSETS_MISSING", "目标要求流动性储备，但没有标明可提供流动性的资产类别。")
        if liquid:
            groups.append({"id": "policy-liquid-reserve", "assets": liquid, "lo": mandate["min_liquid_weight"], "hi": 1.0})
        if illiquid:
            groups.append({"id": "policy-illiquid-cap", "assets": illiquid, "lo": 0.0, "hi": mandate["max_illiquid_weight"]})
        return groups, constraints

    def preview_policy(self, request: PolicyRequest) -> dict:
        kernels.require_ready()
        mandate_artifact, cma = self.get_mandate(request.mandate_id), self.get_cma(request.cma_id)
        mandate, definition = mandate_artifact["definition"], cma["definition"]
        if mandate["currency"] != definition["currency"] or mandate["horizon_years"] != definition["horizon_years"]:
            raise ValidationError("SAA_MANDATE_CMA_BASIS", "投资目标与长期假设的计价币种、投资期限须一致。")
        if not mandate["as_of"] <= definition["as_of"] < mandate["review_date"] or str(date.today()) > mandate["review_date"]:
            raise ValidationError("SAA_MANDATE_EXPIRED", "假设日期不在政策有效区间，或目标已到复核日；请先保存适用的新目标。")
        source = self._source(definition["alloc_name"], definition["as_of"])
        if any(source["lineage"].get(key) != cma["source_snapshot"]["lineage"].get(key) for key in ("config_hash", "nav_hash")):
            raise ConflictError("SAA_CMA_SOURCE_CHANGED", "大类或净值来源已变，请重新验证长期假设；旧版本仍可读取。")
        groups, limits = self._constraints(request, definition, mandate)
        names = [asset["id"] for asset in source["assets"]]
        means = np.asarray([a["annual_return"] for a in definition["assets"]], dtype=np.float64)
        uncertainty = np.asarray([a["mean_uncertainty"] for a in definition["assets"]], dtype=np.float64)
        covariance = self.artifacts.arrays(cma["id"], ("covariance",))["covariance"]
        bounds = np.asarray([[limits[name]["min_weight"], limits[name]["max_weight"]] for name in names], dtype=np.float64)
        membership = np.asarray([[int(name in g["assets"]) for name in names] for g in groups], dtype=np.uint8).reshape(len(groups), len(names))
        weights, metrics, contributions, accepted = kernels.policy_candidates_kernel(
            means, covariance, uncertainty, bounds, membership,
            np.asarray([g["lo"] for g in groups], dtype=np.float64), np.asarray([g["hi"] for g in groups], dtype=np.float64),
            mandate["risk_aversion"], request.uncertainty_penalty, mandate["target_return"], mandate["max_volatility"],
            request.candidate_count, request.seed)
        if not accepted:
            raise ValidationError("SAA_NO_FEASIBLE_CANDIDATE", "当前目标和约束下未找到可行候选。请检查收益目标、风险上限与权重边界；有限搜索失败不证明数学上无解。")
        candidates = [{"id": key, "name": label, "weights": dict(zip(names, weights[i].tolist(), strict=True)),
            "metrics": dict(zip(METRICS, _finite_list(metrics[i]), strict=True)),
            "risk_contributions": dict(zip(names, _finite_list(contributions[i]), strict=True))}
            for i, (key, label) in enumerate(METHODS)]
        return _hashed({"request": request.model_dump(mode="json"), "mandate_id": mandate_artifact["id"],
            "mandate_hash": mandate_artifact["content_hash"], "cma_id": cma["id"], "cma_hash": cma["content_hash"],
            "mandate": mandate, "assumptions": definition, "source_snapshot": source,
            "constraints": limits, "group_limits": groups, "covariance": covariance.tolist(),
            "candidates": candidates, "accepted_candidates": accepted,
            "method": "finite_long_only_moment_candidates", "execution": kernels.execution_audit(),
            "warnings": [*cma["warnings"], "有限候选比较不保证全局最优；预期收益不是历史业绩或收益承诺。",
                         "政策再平衡规则仅记录；TAA 历史比较仍按其明确的日频扣费口径重新计算。"]})

    def publish_policy(self, body: PublishPolicyRequest) -> dict:
        preview = self.preview_policy(body.request)
        if preview["preview_hash"] != body.preview_hash:
            raise ConflictError("SAA_POLICY_PREVIEW_CHANGED", "政策输入已变化，请重新比较再确认采用。")
        candidate = next(item for item in preview["candidates"] if item["id"] == body.candidate_id)
        baseline = copy.deepcopy(preview["source_snapshot"])
        baseline["name"] = body.name
        baseline["group_limits"] = preview["group_limits"]
        for asset in baseline["assets"]:
            asset.update(preview["constraints"][asset["id"]])
            asset["base_weight"] = candidate["weights"][asset["id"]]
        baseline["policy"] = {"schema_version": "1.0", "mandate_id": preview["mandate_id"],
            "mandate_hash": preview["mandate_hash"], "cma_id": preview["cma_id"], "cma_hash": preview["cma_hash"],
            "mandate": preview["mandate"], "assumptions": preview["assumptions"], "covariance": preview["covariance"],
            "expires_on": preview["mandate"]["review_date"], "selection": candidate,
            "selection_request": preview["request"], "selection_preview_hash": preview["preview_hash"],
            "reason": body.reason, "confirmation_type": "researcher_policy_adoption", "independent_approval": False,
            "execution": preview["execution"]}
        return self.baselines.save_baseline(baseline)
