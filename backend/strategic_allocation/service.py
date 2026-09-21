"""Goals -> economic classes -> explicit CMA -> immutable SAA policy.

Previews are pure reads/calculations. Confirmation repeats those calculations;
only their matching results may enter the existing managed artifact store.
"""
from __future__ import annotations

import copy
from datetime import date
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backend.custom_indicators.errors import ConflictError, ValidationError
from backend.market_data import resolve_market_data_file
from backend.research_input_checks import return_quality
from backend.sensitivity.repository import ArtifactRepository, digest_json
from backend.tactical_allocation.contracts import AssetLimit
from backend.tactical_allocation.data import TacticalAllocationData, warm_tactical_data
from backend.tactical_allocation.repository import TacticalAllocationRepository
from . import kernels, goal_kernels, institution_kernels, cma_model_kernels, mandate_kernels, multi_cma_kernels, compatibility_kernels
from .mandate_diagnosis import diagnose_reference, validate_fixed_candidate
from .cma_application import frozen_assumptions, frozen_model_lineage, frozen_numeric_inputs
from .cma_service import CmaResearchService
from .institution import diagnose_institution, review_blockers
from .sources import product_source, strategic_source
from .universes import StrategicScopes
from .planning import funding_inputs, diagnose_funding, require_goal_checks, LIMITATIONS
from .mandate_inputs import (cash_success_required, has_cash_budget, resolve_authorization,
                             require_resolved_authorization, effective_cash_floor, effective_return_floor,
                             _freeze_reference_benchmark)
from .contracts import (
    CmaRequest, PolicyRequest, PublishCmaRequest,
    PublishPolicyRequest, RiskReferenceRequest, MandateStudyRequest, ConfirmMandateRequest,
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


def _policy_expiry(mandate: dict) -> str:
    """Operational consumers need one date even when the user leaves review optional."""
    dates = [value for value in (mandate.get("review_date"), mandate.get("risk_reference_valid_until")) if value]
    return min(dates) if dates else "9999-12-31"


class StrategicAllocationService:
    def __init__(self, root: Path, data_dir: Path, *, universe_dir: Path | None = None,
                 tactical_repository: TacticalAllocationRepository | None = None):
        self.artifacts = ArtifactRepository(Path(root) / "strategic_allocation" / "artifacts")
        self.data = TacticalAllocationData(data_dir, universe_dir=universe_dir)
        self.baselines = tactical_repository or TacticalAllocationRepository(root)
        self.scopes = StrategicScopes(self.artifacts, self.data)
        from .risk_scale_service import RiskScaleService
        from .risk_scale_store import RiskScaleStore
        from .reference_inputs import ReferenceInputs
        from .reference_sources import ReferenceSources
        self.risk_scales = RiskScaleService(
            self.artifacts, RiskScaleStore(self.artifacts.root.parent),
            ReferenceInputs(self.artifacts, ReferenceSources(data_dir)))
        self.cma = CmaResearchService(self)

    def warm(self) -> dict:
        warm_tactical_data()
        institution_kernels.warm()
        cma_model_kernels.warm()
        self.risk_scales.warm()
        cma_statistics = self.cma.warm()
        multi_cma = multi_cma_kernels.warm()
        compatibility = compatibility_kernels.warm()
        from . import uncertainty_kernels
        uncertainty_radius = uncertainty_kernels.warm()
        audit = kernels.warm_strategic_kernels()
        mandate_kernels.warm()
        institution = institution_kernels.execution_audit()
        models = cma_model_kernels.execution_audit()
        if not audit["complete"] or not institution["complete"] or not models["complete"] or not cma_statistics["complete"] or not multi_cma["complete"] or not compatibility["complete"] or not uncertainty_radius["complete"]:
            raise RuntimeError("战略配置及机构诊断启动预热未完成。")
        return {**audit, "institution": institution, "cma_models": models, "cma_statistics": cma_statistics,
                "mandate_reference": mandate_kernels.execution_audit(), "multi_cma": multi_cma,
                "compatibility": compatibility, "uncertainty_radius": uncertainty_radius}

    def _get(self, identifier: str, artifact_type: str) -> dict:
        item = self.artifacts.get(identifier, "series")
        if item.get("artifact_type") != artifact_type:
            raise ValidationError("SAA_VERSION_TYPE", "所选目标或假设的版本类型不匹配。")
        return item

    def get_mandate(self, identifier: str) -> dict:
        return self._get(identifier, "investment_mandate")

    def get_cma(self, identifier: str) -> dict:
        return self.cma.get(identifier)

    def _retired_mandate_ids(self) -> set[str]:
        retired = set()
        for summary in self.artifacts.list("retirement"):
            item = self.artifacts.get(summary["id"], "retirement")
            if item.get("artifact_type") == "investment_mandate_retirement" and item.get("mandate_id"):
                retired.add(item["mandate_id"])
        return retired

    def _active_mandate_ids(self) -> set[str]:
        items = [self.artifacts.get(summary["id"], "series") for summary in self.artifacts.list("series")]
        mandates = [item for item in items if item.get("artifact_type") == "investment_mandate"]
        superseded = {item["supersedes_mandate_id"] for item in mandates if item.get("supersedes_mandate_id")}
        return {item["id"] for item in mandates} - superseded - self._retired_mandate_ids()

    def _require_active_mandate(self, identifier: str) -> dict:
        mandate = self.get_mandate(identifier)
        if identifier not in self._active_mandate_ids():
            raise ConflictError("MANDATE_INACTIVE", "投资目标已删除或被新版本替代，不能建立新政策；请刷新并选择当前版本。")
        return mandate

    def retire_mandate(self, identifier: str) -> dict:
        mandate = self.get_mandate(identifier)
        # Serialize lifecycle changes so concurrent edit/delete requests cannot
        # produce two active replacements or duplicate retirement records.
        with self.artifacts.governance_lock.locked():
            if identifier in self._retired_mandate_ids():
                return {"deleted": True, "id": identifier}
            if identifier not in self._active_mandate_ids():
                raise ConflictError("MANDATE_ALREADY_REPLACED", "该投资目标已被新版本替代，请刷新列表后操作当前版本。")
            self.artifacts.save("retirement", {
                "artifact_type": "investment_mandate_retirement",
                "name": mandate["name"],
                "mandate_id": identifier,
                "mandate_hash": mandate["content_hash"],
                "research_only": True,
            })
        return {"deleted": True, "id": identifier}

    def catalog(self) -> dict:
        mandates, assumptions, universes, mappings = [], [], [], []
        series = [self.artifacts.get(summary["id"], "series") for summary in self.artifacts.list("series")]
        superseded = {item["supersedes_mandate_id"] for item in series
                      if item.get("artifact_type") == "investment_mandate" and item.get("supersedes_mandate_id")}
        retired = self._retired_mandate_ids()
        retired_cma = self.cma.retired_ids()
        for item in series:
            common = {key: item[key] for key in ("id", "name", "created_at", "content_hash")}
            if item.get("artifact_type") == "investment_mandate":
                if item["id"] in superseded or item["id"] in retired:
                    continue
                mandates.append({**common, "definition": item["definition"],
                    "assessment_status": item.get("assessment", {}).get("status", "inputs_only")})
            elif item.get("artifact_type") == "capital_market_assumptions":
                if item["id"] in retired_cma:
                    continue
                assumptions.append({**common, **{key: item["definition"][key] for key in
                    ("alloc_name", "as_of", "currency", "horizon_years")},
                    "strategic_universe_id": item["definition"].get("strategic_universe_id"),
                    "implementation_mapping_id": item["definition"].get("implementation_mapping_id"),
                    "schema_version": item["definition"].get("schema_version", "1.0")})
            elif item.get("artifact_type") == "strategic_universe":
                universes.append({**common, "definition": item["definition"]})
            elif item.get("artifact_type") == "implementation_mapping":
                mappings.append({**common, "definition": item["definition"],
                    "implementation_status": item["implementation_status"], "implementation_gaps": item["implementation_gaps"]})
        policies = [{key: item[key] for key in ("id", "name", "as_of", "created_at", "content_hash", "alloc_name")}
                    for item in self.baselines.list_baselines() if item.get("policy")]
        return {**self.data.catalog(), "mandates": mandates, "assumptions": assumptions, "policies": policies,
                "strategic_universes": universes, "implementation_maps": mappings}

    def mandate_funding(self, request: MandateStudyRequest) -> dict:
        """填写页的现金流确定性回显：只跑资金算术，不碰CMA、不跑模拟、不解参考组合。"""
        goal_kernels.require_ready()
        definition = request.definition.model_dump(mode="json")
        prepared = funding_inputs(definition)
        required = prepared[0]["cashflow_required_return"] if prepared else None
        return {"funding": prepared[0] if prepared else None,
                "effective_target_return": effective_return_floor(definition, required),
                "execution": goal_kernels.execution_audit()}

    def preview_mandate(self, request: MandateStudyRequest) -> dict:
        kernels.require_ready()
        goal_kernels.require_ready()
        requested_benchmark = request.definition.model_dump(mode="json").get("benchmark")
        definition, risk_decision = resolve_authorization(request.definition.model_dump(mode="json"), self.risk_scales)
        if definition.get("strategic_universe_id"):
            universe = self.scopes.get_universe(definition["strategic_universe_id"])
            if universe["definition"]["currency"] != definition["currency"] or universe["definition"]["as_of"] > definition["as_of"]:
                raise ValidationError("MANDATE_UNIVERSE_BASIS", "目标须与战略范围采用同本位币及适用研究日。")
        prepared = funding_inputs(definition)
        if definition.get("schema_version", "1.0") == "2.0":
            definition["effective_cash_reserve_weight"] = effective_cash_floor(
                definition, prepared[0]["required_liquid_weight"] if prepared else 0.)
            definition["effective_target_return"] = effective_return_floor(
                definition, prepared[0]["cashflow_required_return"] if prepared else None)
            if risk_decision.get("reference_valid_until"):
                definition["risk_reference_valid_until"] = risk_decision["reference_valid_until"]
        payload = {"request": request.model_dump(mode="json"), "definition": definition,
            "funding": prepared[0] if prepared else None, "candidates": [], "cma": None,
            "status": "inputs_only", "blockers": [], "warnings": list(LIMITATIONS),
            "execution": goal_kernels.execution_audit(),
            "confirmation_type": "research_inputs_not_external_approval"}
        if risk_decision is not None:
            payload["risk_decision"] = risk_decision
        if definition.get("institutional_context") is not None:
            payload["institutional_diagnostics"] = diagnose_institution(definition)
        if prepared and prepared[0]["required_liquid_weight"] > 1:
            payload["blockers"].append("流动性窗口内的压力净支出超过可投资本金；先调整资金或支付计划。")
            payload["status"] = "needs_revision"
        if risk_decision and risk_decision.get("risk_scale_ref"):
            reference = diagnose_reference(self, request, definition, risk_decision)
            payload["reference_diagnosis"] = reference
            risk_decision["minimum_tested_feasible_level"] = reference["minimum_tested_feasible_level"]
            if reference["status"] == "validated":
                if risk_decision["selection_pending"]:
                    level = reference["minimum_tested_feasible_level"]
                    if level is not None:
                        cap = risk_decision["applied_boundaries"][level - 1]
                        proposed_definition = copy.deepcopy(definition)
                        proposed_decision = copy.deepcopy(risk_decision)
                        proposed_decision.update(selected_max_level=level, selected_volatility_cap=cap,
                                                 selection_pending=False)
                        proposed_definition["risk_authorization"]["selected_max_level"] = level
                        proposed_definition["max_volatility"] = cap
                        automatic_relative_benchmark = (requested_benchmark is None
                                                        and definition.get("objective_kind") == "benchmark_relative")
                        if automatic_relative_benchmark:
                            version = self.risk_scales.get_version(risk_decision["risk_scale_ref"]["id"])
                            proposed_definition["benchmark"] = None
                            _freeze_reference_benchmark(proposed_definition, version, level)
                            working_level = level
                            working_definition = proposed_definition
                            working_decision = proposed_decision
                            stable = working_level == risk_decision["authorized_max_level"]
                            for _ in range(len(risk_decision["applied_boundaries"])):
                                if stable:
                                    break
                                reference = diagnose_reference(self, request, working_definition, working_decision)
                                payload["reference_diagnosis"] = reference
                                working_decision["minimum_tested_feasible_level"] = reference["minimum_tested_feasible_level"]
                                risk_decision["minimum_tested_feasible_level"] = reference["minimum_tested_feasible_level"]
                                if reference["status"] != "validated":
                                    break
                                next_level = reference["minimum_tested_feasible_level"]
                                if (next_level is None or not 1 <= next_level <= len(risk_decision["applied_boundaries"])
                                        or next_level > working_level):
                                    break
                                if next_level == working_level:
                                    stable = True
                                    break
                                working_level = next_level
                                cap = risk_decision["applied_boundaries"][working_level - 1]
                                working_definition["benchmark"] = None
                                working_definition["risk_authorization"]["selected_max_level"] = working_level
                                working_definition["max_volatility"] = cap
                                working_decision.update(selected_max_level=working_level,
                                                        selected_volatility_cap=cap)
                                _freeze_reference_benchmark(working_definition, version, working_level)
                            if stable:
                                definition.clear()
                                definition.update(working_definition)
                                risk_decision.clear()
                                risk_decision.update(working_decision)
                                risk_decision["status"] = "recommendation_validated"
                            else:
                                payload["status"] = "needs_revision"
                                payload["blockers"].append("冻结参考基准后的推荐等级未稳定通过复核，请重新诊断。")
                                definition["max_volatility"] = None
                        else:
                            definition.clear()
                            definition.update(proposed_definition)
                            risk_decision.clear()
                            risk_decision.update(proposed_decision)
                            risk_decision["status"] = "recommendation_validated"
                if reference["status"] == "validated" and risk_decision["status"] == "recommendation_validated":
                    payload["status"] = "diagnosed"
                    payload["diagnosis_scope"] = "universal_reference"
                elif not payload["blockers"] and not risk_decision["selection_pending"]:
                    payload["status"] = "diagnosed"
                    payload["diagnosis_scope"] = "universal_reference"
            elif reference["status"] in {"validation_failed", "no_validated_candidate_in_search", "constraint_conflict", "solver_failed"}:
                payload["status"] = "needs_revision"
        if request.cma_id and definition.get("max_volatility") is not None:
            cma = self.get_cma(request.cma_id)
            assumed = cma["definition"]
            if (assumed["currency"] != definition["currency"] or assumed["horizon_years"] != definition["horizon_years"]
                    or assumed["as_of"] != definition["as_of"]):
                raise ValidationError("MANDATE_CMA_BASIS", "诊断须使用同研究日、同币种、同投资期限的CMA；不能自动移动现金流日期。")
            payload["cma"] = {"id": cma["id"], "name": cma["name"], "content_hash": cma["content_hash"],
                              "as_of": assumed["as_of"]}
            try:
                calculation = self._candidate_calculation(
                    PolicyRequest(mandate_id="unsaved-input", cma_id=request.cma_id,
                                  uncertainty_penalty=request.uncertainty_penalty, seed=request.seed),
                    definition, cma, paths=request.simulation_paths)
            except ValidationError as exc:
                if exc.code not in {"SAA_NO_FEASIBLE_CANDIDATE", "SAA_LIQUID_ASSETS_MISSING", "SAA_CASH_ASSETS_MISSING", "MANDATE_LIQUIDITY_CONFLICT", "MANDATE_LIMIT_CONFLICT"}:
                    raise
                payload["blockers"].append(str(exc))
                payload["status"] = "needs_revision"
            else:
                require_goal_checks(definition, calculation["candidates"])
                payload.update(calculation)
                payload["diagnosis_scope"] = "actual_cma"
                passing = any(item["goal_check"]["within_limits"] if cash_success_required(definition) else True
                              for item in calculation["candidates"])
                payload["status"] = "diagnosed" if passing else "needs_revision"
                if not passing:
                    payload["blockers"].append("当前代表组合均未达到目标成功概率门槛；不代表所有可能组合数学上都无解。")
        if risk_decision and risk_decision["selection_pending"]:
            definition["max_volatility"] = None
        return _hashed(payload)

    def confirm_mandate(self, body: ConfirmMandateRequest) -> dict:
        if body.request.definition.schema_version == "1.0" and len(body.request.definition.boundary_reason) < 5:
            raise ValidationError("MANDATE_BOUNDARY_REASON", "旧版目标须说明风险和流动性边界的依据，至少5个字符。")
        if body.replaces_mandate_id is not None:
            self.get_mandate(body.replaces_mandate_id)
        with self.risk_scales.store.document.locked():
            preview = self.preview_mandate(body.request)
            if preview["preview_hash"] != body.preview_hash:
                raise ConflictError("MANDATE_PREVIEW_CHANGED", "目标、模型或CMA已变化，请重新诊断后确认。")
            fields = {"artifact_type": "investment_mandate",
                "name": body.request.definition.name, "definition": preview["definition"],
                "assessment": preview, "supersedes_mandate_id": body.replaces_mandate_id,
                "planning_settings": {"simulation_paths": body.request.simulation_paths,
                    "seed": body.request.seed, "validation_seed": body.request.validation_seed,
                    "uncertainty_penalty": body.request.uncertainty_penalty},
                "research_only": True}
            if body.replaces_mandate_id is None:
                return self.artifacts.save("series", fields)
            with self.artifacts.governance_lock.locked():
                if body.replaces_mandate_id not in self._active_mandate_ids():
                    raise ConflictError("MANDATE_ALREADY_REPLACED", "原投资目标已被修改或删除，请刷新列表后重新选择。")
                return self.artifacts.save("series", fields)

    def _source(self, alloc_name: str | None, as_of: str, *, strategic_universe_id: str | None = None,
                implementation_mapping_id: str | None = None) -> dict:
        kernels.require_ready()
        if not strategic_universe_id:
            return product_source(self.data, alloc_name, as_of)
        universe = self.scopes.get_universe(strategic_universe_id)
        mapping = self.scopes.get_mapping(implementation_mapping_id) if implementation_mapping_id else None
        if mapping:
            if mapping["definition"]["strategic_universe_id"] != universe["id"] or mapping["strategic_universe_hash"] != universe["content_hash"]:
                raise ValidationError("SAA_MAPPING_UNIVERSE", "映射不属于当前不可变战略范围。")
            current = product_source(self.data, mapping["definition"]["alloc_name"], as_of)
            frozen = mapping["source_snapshot"]
            if any(current["lineage"].get(k) != frozen["lineage"].get(k) for k in ("config_hash", "nav_hash", "universe")):
                raise ConflictError("SAA_MAPPING_SOURCE_CHANGED", "映射的代理或产品域已变化，请确认新映射；历史版本保持只读。")
            self.data.validate_application(frozen)
        return strategic_source(universe, mapping, as_of)

    def _definition_source(self, definition: dict) -> dict:
        return self._source(definition.get("alloc_name"), definition["as_of"],
            strategic_universe_id=definition.get("strategic_universe_id"),
            implementation_mapping_id=definition.get("implementation_mapping_id"))

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

    def preview_cma(self, request: CmaRequest) -> dict:
        return self.cma.preview(request)

    def publish_cma(self, body: PublishCmaRequest) -> dict:
        return self.cma.publish(body)

    @staticmethod
    def _constraints(request: PolicyRequest, definition: dict, mandate: dict) -> tuple[list, dict]:
        require_resolved_authorization(mandate)
        names = [asset["id"] for asset in definition["assets"]]
        if set(request.constraints) - set(names):
            raise ValidationError("SAA_CONSTRAINT_AXIS", "约束包含不属于当前假设的资产。")
        if mandate.get("strategic_universe_id") and mandate["strategic_universe_id"] != definition.get("strategic_universe_id"):
            raise ValidationError("SAA_CONSTRAINT_AXIS", "投资目标的战略范围与CMA不同。")
        if mandate.get("allocation_scope") and (definition.get("strategic_universe_id") or mandate["allocation_scope"] != definition["alloc_name"]):
            raise ValidationError("SAA_CONSTRAINT_AXIS", "投资目标指定的大类方案与CMA不同；不能只因资产同名就转移授权。")
        authorised = mandate.get("asset_limits", {})
        if (authorised or mandate.get("group_limits")) and not (mandate.get("allocation_scope") or mandate.get("strategic_universe_id")):
            raise ValidationError("MANDATE_AUTHORIZATION_SCOPE", "资产或分组授权缺少所属大类方案；请复制目标并补齐范围，不能按资产同名转移授权。")
        if set(authorised) - set(names):
            raise ValidationError("SAA_CONSTRAINT_AXIS", "投资目标的资产边界不属于当前CMA大类轴。")
        constraints = {}
        for name in names:
            requested = (request.constraints.get(name) or AssetLimit()).model_dump()
            policy = authorised.get(name, {})
            requested["min_weight"] = max(requested["min_weight"], policy.get("min_weight", 0))
            requested["max_weight"] = min(requested["max_weight"], policy.get("max_weight", 1))
            requested["max_abs_tilt"] = min(requested["max_abs_tilt"], policy.get("max_abs_tilt", 1))
            if requested["min_weight"] > requested["max_weight"]:
                raise ValidationError("MANDATE_LIMIT_CONFLICT", "当前权重设置与投资授权边界冲突；下游不得放宽授权。")
            if mandate["max_tracking_error"] == 0:
                requested["max_abs_tilt"] = 0.0
            constraints[name] = requested
        groups, seen = [], set()
        for raw in request.group_limits:
            if raw.id.startswith("policy-") or raw.id in seen or len(set(raw.assets)) != len(raw.assets) or set(raw.assets) - set(names) or raw.lo > raw.hi:
                raise ValidationError("SAA_GROUP_INVALID", "分组名称、成员或边界无效；policy- 前缀保留给目标约束。")
            groups.append(raw.model_dump())
            seen.add(raw.id)
        for index, group in enumerate(mandate.get("group_limits", [])):
            if (set(group["assets"]) - set(names) or len(set(group["assets"])) != len(group["assets"])
                    or not group["assets"] or group["lo"] > group["hi"]):
                raise ValidationError("SAA_GROUP_INVALID", "投资授权的分组成员或边界无效。")
            groups.append({**group, "id": f"policy-mandate-{index}"})
        liquid = [a["id"] for a in definition["assets"] if a["liquidity"] == "liquid"]
        illiquid = [a["id"] for a in definition["assets"] if a["liquidity"] == "illiquid"]
        liquidity_floor = mandate["min_liquid_weight"]
        funding = funding_inputs(mandate)
        funding_floor = funding[0]["required_liquid_weight"] if funding else 0.
        if mandate.get("schema_version", "1.0") == "1.0":
            liquidity_floor = max(liquidity_floor, funding_floor)
        if max(liquidity_floor, funding_floor) > 1:
            raise ValidationError("MANDATE_LIQUIDITY_CONFLICT", "现金流要求的流动性储备超过可投资本金。")
        if liquidity_floor > 0 and not liquid:
            raise ValidationError("SAA_LIQUID_ASSETS_MISSING", "目标要求流动性储备，但没有标明可提供流动性的资产类别。")
        if liquid:
            groups.append({"id": "policy-liquid-reserve", "assets": liquid, "lo": liquidity_floor, "hi": 1.0})
        if illiquid:
            groups.append({"id": "policy-illiquid-cap", "assets": illiquid, "lo": 0.0, "hi": mandate["max_illiquid_weight"]})
        cash_floor = effective_cash_floor(mandate, funding_floor)
        if cash_floor > 0:
            cash = [a["id"] for a in definition["assets"] if a["role"] == "liquidity" and a["liquidity"] == "liquid"]
            if not cash:
                raise ValidationError("SAA_CASH_ASSETS_MISSING", "现金用途下限要求明确的可流动现金角色；可交易权益ETF不能代替现金储备。")
            groups.append({"id": "policy-cash-reserve", "assets": cash, "lo": cash_floor, "hi": 1.0})
        return groups, constraints

    def _candidate_calculation(self, request: PolicyRequest, mandate: dict, cma: dict, *,
                               paths: int = 2000, simulation_seed: int | None = None) -> dict:
        if request.mode == "compatible_all_models":
            from .compatibility import calculate
            return calculate(self, request, mandate, cma, paths=paths, simulation_seed=simulation_seed)
        budget = {}
        if "multi_cma" in cma:
            from .multi_cma import require_calculation_budget
            budget["multi_cma_budget"] = require_calculation_budget(request, mandate, paths)
        definition = frozen_assumptions(cma)
        if cma["definition"].get("model") is not None:
            cma_model_kernels.require_ready()
        groups, limits = self._constraints(request, definition, mandate)
        names = [asset["id"] for asset in definition["assets"]]
        means, covariance, uncertainty = frozen_numeric_inputs(cma, self.artifacts)
        from .uncertainty import resolve_uncertainty
        mean_covariance, uncertainty_model = resolve_uncertainty(request, cma, self.artifacts)
        bounds = np.asarray([[limits[name]["min_weight"], limits[name]["max_weight"]] for name in names], dtype=np.float64)
        membership = np.asarray([[int(name in g["assets"]) for name in names] for g in groups], dtype=np.uint8).reshape(len(groups), len(names))
        benchmark = mandate.get("benchmark")
        benchmark_weights = np.empty(0, dtype=np.float64)
        if benchmark:
            if (benchmark.get("source", "explicit") == "explicit" and benchmark["alloc_name"] != definition["alloc_name"]
                    or set(benchmark["weights"]) != set(names)):
                raise ValidationError("MANDATE_BENCHMARK_AXIS", "基准必须与当前CMA使用完整一致的资产轴；显式基准还须属于同一大类方案。")
            benchmark_weights = np.asarray([benchmark["weights"][name] for name in names], dtype=np.float64)
        floor = mandate.get("effective_target_return")
        if floor is None:
            floor = mandate["target_return"] if mandate.get("objective_kind", "absolute_return") == "absolute_return" else -np.inf
        if request.risk_budget is not None and set(request.risk_budget) != set(names):
            raise ValidationError("SAA_RISK_BUDGET_AXIS", "风险预算须完整覆盖当前资产轴，不能遗漏或包含未知资产。")
        risk_budget = np.asarray([request.risk_budget[name] for name in names] if request.risk_budget is not None else [], dtype=np.float64)
        search = (kernels.policy_candidates_ellipsoidal_kernel if uncertainty_model is not None
                  else kernels.policy_candidates_with_budget_kernel)
        weights, metrics, contributions, accepted = search(
            means, covariance, mean_covariance if uncertainty_model is not None else uncertainty, bounds, membership,
            np.asarray([g["lo"] for g in groups], dtype=np.float64), np.asarray([g["hi"] for g in groups], dtype=np.float64),
            mandate["risk_aversion"], uncertainty_model["kappa"] if uncertainty_model is not None else request.uncertainty_penalty,
            floor, mandate["max_volatility"], benchmark_weights, benchmark["max_tracking_error"] if benchmark else 1.,
            benchmark["target_excess_return"] if benchmark else 0., request.candidate_count, request.seed, risk_budget)
        if not accepted:
            raise ValidationError("SAA_NO_FEASIBLE_CANDIDATE", "当前目标与硬约束下未找到可行候选。检查收益、波动、基准主动风险、流动性和权重边界；有限搜索失败不证明数学无解。")
        candidates = [{"id": key, "name": label, "weights": dict(zip(names, weights[i].tolist(), strict=True)),
            "metrics": dict(zip(METRICS, _finite_list(metrics[i]), strict=True)),
            "risk_contributions": dict(zip(names, _finite_list(contributions[i]), strict=True))}
            for i, (key, label) in enumerate(METHODS)]
        if uncertainty_model is not None:
            candidates[2]["name"] = "椭球稳健效用（有限搜索）"
        unavailable = []
        if request.risk_budget is not None:
            if np.all(np.isfinite(metrics[4])):
                candidates.append({"id": "risk-budget", "name": "风险预算匹配（有限搜索）", "available": True,
                    "weights": dict(zip(names, weights[4].tolist(), strict=True)),
                    "metrics": dict(zip(METRICS, _finite_list(metrics[4]), strict=True)),
                    "risk_contributions": dict(zip(names, _finite_list(contributions[4]), strict=True)),
                    "risk_budget": request.risk_budget,
                    "risk_budget_distance": float(kernels.risk_budget_error_kernel(contributions[4], risk_budget)),
                    "distance_basis": "squared_distance_signed_euler_shares_finite_search"})
            else:
                unavailable.append({"id": "risk-budget", "name": "风险预算匹配（有限搜索）", "available": False,
                    "weights": {}, "metrics": {key: None for key in METRICS}, "risk_contributions": {},
                    "risk_budget": request.risk_budget, "risk_budget_distance": None,
                    "unavailable_reason": "可行候选的风险贡献未定义（零方差）；不能生成或采纳风险预算权重。"})
        if benchmark:
            for i, candidate in enumerate(candidates):
                candidate["benchmark_check"] = {"name": benchmark["name"],
                    "expected_excess_return": float(kernels.expected_excess_return_kernel(weights[i], benchmark_weights, means)),
                    "tracking_error": float(kernels.expected_active_risk_kernel(weights[i], benchmark_weights, covariance)),
                    "target_excess_return": benchmark["target_excess_return"], "max_tracking_error": benchmark["max_tracking_error"]}
        # Candidate exploration and the confirmed goal's scenario ensemble are
        # different controls. Changing the former must not silently resample the latter.
        diagnosis = diagnose_funding(mandate, candidates, paths=paths,
                                     seed=request.seed if simulation_seed is None else simulation_seed)
        require_goal_checks(mandate, candidates)
        if "multi_cma" in cma:
            from .multi_cma import cross_model_results
            for candidate in candidates:
                candidate["cross_model_results"] = cross_model_results(cma["multi_cma"], candidate["weights"], mandate,
                    penalty=request.uncertainty_penalty, paths=paths,
                    seed=request.seed if simulation_seed is None else simulation_seed)
        return {"constraints": limits, "group_limits": groups, "covariance": covariance.tolist(),
                "candidates": candidates + unavailable, "accepted_candidates": accepted, "funding": diagnosis.get("funding"),
                "funding_model": diagnosis.get("model"), "funding_execution": diagnosis.get("execution"),
                **({"uncertainty_model": uncertainty_model} if uncertainty_model is not None else {}), **budget}

    def preview_policy(self, request: PolicyRequest) -> dict:
        kernels.require_ready()
        from .multi_cma import resolve, request_payload
        mandate_artifact = self._require_active_mandate(request.mandate_id)
        if request.mode == "single":
            cma = self.cma.require_selectable(request.cma_id)
        elif request.mode in ("parameter_average", "compatible_all_models"):
            cma = resolve(self, request)
        else:
            raise ValidationError("SAA_MULTI_CMA_MODE", "未知 SAA 模式，未执行计算。")
        assessment = mandate_artifact.get("assessment", {})
        if not assessment.get("preview_hash") or not assessment.get("request"):
            raise ValidationError("MANDATE_CONFIRMATION_REQUIRED", "此目标缺少诊断与确认记录；请复制为新研究，诊断并确认后再建立政策。")
        mandate, definition = mandate_artifact["definition"], cma["definition"]
        require_resolved_authorization(mandate)
        if mandate["currency"] != definition["currency"] or mandate["horizon_years"] != definition["horizon_years"]:
            raise ValidationError("SAA_MANDATE_CMA_BASIS", "投资目标与长期假设的计价币种、投资期限须一致。")
        if definition["as_of"] < mandate["as_of"] or (mandate.get("review_date") and definition["as_of"] >= mandate["review_date"]):
            raise ValidationError("SAA_MANDATE_EXPIRED", "假设日期不在目标的研究有效区间；请保存适用的新目标。")
        if has_cash_budget(mandate) and mandate["as_of"] != definition["as_of"]:
            raise ValidationError("MANDATE_CMA_BASIS", "金额计划和CMA研究日须一致；更新时必须显式滚动现金流计划。")
        source = self._definition_source(definition)
        if any(source["lineage"].get(key) != cma["source_snapshot"]["lineage"].get(key) for key in ("config_hash", "nav_hash", "strategic_universe_hash", "implementation_mapping_hash")):
            raise ConflictError("SAA_CMA_SOURCE_CHANGED", "大类或净值来源已变，请重新验证长期假设；旧版本仍可读取。")
        if request.implementation_mapping_id:
            if definition.get("schema_version") != "2.0" or not definition.get("strategic_universe_id"):
                raise ValidationError("LTCMA_POLICY_MAPPING", "独立政策映射仅适用于 2.0 战略 LTCMA；旧版本保持原冻结映射。")
            source = self._source(None, definition["as_of"],
                strategic_universe_id=definition["strategic_universe_id"],
                implementation_mapping_id=request.implementation_mapping_id)
        planning_settings = mandate_artifact.get("planning_settings", {})
        calculation = self._candidate_calculation(request, mandate, cma,
            paths=planning_settings.get("simulation_paths", 2000),
            simulation_seed=planning_settings.get("seed"))
        application_day = str(date.today())
        application_blockers = [*source["apply_reasons"], *review_blockers(mandate, application_day)]
        if mandate.get("schema_version", "1.0") == "2.0":
            risk_ref = mandate["risk_authorization"].get("risk_scale_ref")
            if risk_ref:
                scale = self.risk_scales.get_version(risk_ref["id"])
                if scale["content_hash"] != risk_ref["content_hash"]:
                    raise ConflictError("MANDATE_RISK_SCALE_CHANGED", "目标引用标尺指纹已变化，不能用于新政策。")
                application_blockers.extend(x["message"] for x in scale["current_eligibility"]["blockers"])
        if mandate.get("review_date") and application_day >= mandate["review_date"]:
            application_blockers.append("政策已到复核日，历史研究可保留；请确认新政策后再用于当前产品应用。")
        mapping = source.get("implementation_mapping_snapshot")
        if mapping and not mapping["definition"]["as_of"] <= application_day < mapping["definition"]["valid_until"]:
            application_blockers.append("实施映射尚未生效或已到复核日，请确认新映射后再用于当前产品应用。")
        payload = {"request": request_payload(request), "mandate_id": mandate_artifact["id"],
            "mandate_hash": mandate_artifact["content_hash"], "cma_id": cma["id"], "cma_hash": cma["content_hash"],
            "mandate": mandate, "assumptions": frozen_assumptions(cma), **frozen_model_lineage(cma), "source_snapshot": source,
            **calculation, "current_application_eligible": source["apply_eligible"] and not application_blockers,
            "application_blockers": application_blockers,
            "method": "finite_long_only_moment_candidates", "execution": kernels.execution_audit(),
            "warnings": [*cma["warnings"], *([text.replace("四类代表组合", "代表组合（含显式风险预算）") for text in LIMITATIONS] if request.risk_budget is not None and cash_success_required(mandate) else LIMITATIONS if cash_success_required(mandate) else []),
                         "历史研究可以保存；当前应用需另行核对政策是否到期。",
                         "有限候选比较不保证全局最优；预期收益不是历史业绩或收益承诺。",
                         "政策再平衡约定不自动执行；TAA 按所选决策、执行频率及费用口径独立验证。"]}
        if calculation.get("uncertainty_model"):
            payload["warnings"].extend(calculation["uncertainty_model"]["warnings"])
        else:
            payload["warnings"].append("区间稳健化按逐资产半宽求最坏收益，不是组合联合95%置信下界；惩罚倍数是研究设置。")
        if "multi_cma" in payload:
            from .multi_cma import require_payload_budget
            if request.mode == "compatible_all_models":
                payload["method"] = calculation["method"]
                payload["execution"] = calculation["execution"]
                payload["warnings"] = [*cma["warnings"], *calculation["compatibility"]["limitations"],
                    "仅核验原始约束及目标界限后报告凸问题收敛；资金检查不保证全局成功概率最优。",
                    "研究保存不等于当前产品实施资格，也不代表未来收益保证。"]
            require_payload_budget(payload)
        return _hashed(payload)

    def publish_policy(self, body: PublishPolicyRequest) -> dict:
        preview = self.preview_policy(body.request)
        if preview["preview_hash"] != body.preview_hash:
            raise ConflictError("SAA_POLICY_PREVIEW_CHANGED", "政策输入已变化，请重新比较再确认采用。")
        candidate = next((item for item in preview["candidates"] if item["id"] == body.candidate_id), None)
        if candidate is None or candidate.get("available") is False:
            raise ValidationError("SAA_CANDIDATE_UNAVAILABLE", "所选候选不存在或风险贡献未定义，请选择可用候选。")
        require_goal_checks(preview["mandate"], [item for item in preview["candidates"] if item.get("available") is not False])
        if cash_success_required(preview["mandate"]) and not candidate["goal_check"]["within_limits"]:
            raise ValidationError("MANDATE_GOAL_NOT_MET", "所选组合的模拟成功概率区间下界未达到目标门槛，请调整目标或资金后重新研究。")
        funding_validation = None
        if (preview["mandate"].get("schema_version", "1.0") == "2.0" or body.request.mode == "compatible_all_models") and cash_success_required(preview["mandate"]):
            saved = self.get_mandate(preview["mandate_id"])
            settings = saved["planning_settings"]
            # Separate stream from both the mandate recommendation and SAA exploration.
            verification = SimpleNamespace(simulation_paths=settings["simulation_paths"],
                validation_seed=int(settings["validation_seed"]) ^ 0x9E3779B9)
            if verification.validation_seed == int(settings["seed"]):
                raise ValidationError("MANDATE_VALIDATION_SEED_COLLISION", "冻结目标的搜索与SAA验证样本重复，请重新确认使用不同随机种子的目标。")
            if body.request.mode == "compatible_all_models":
                validations = []
                prepared = funding_inputs(preview["mandate"])
                for row in candidate["cross_model_results"]:
                    result = validate_fixed_candidate(
                        np.asarray([row["metrics"][key] for key in METRICS], dtype=np.float64),
                        preview["mandate"], prepared, verification, {"method_code": 0, "periods_per_year": 1})
                    validations.append({"cma_id": row["cma_id"], **result})
                funding_validation = {"within_limits": all(row["within_limits"] for row in validations),
                    "models": validations, "distribution": "each_source_annual_moment_proxy", "validation_seed": verification.validation_seed}
            else:
                funding_validation = validate_fixed_candidate(
                    np.asarray([candidate["metrics"][key] for key in METRICS], dtype=np.float64),
                    preview["mandate"], funding_inputs(preview["mandate"]), verification,
                    {"method_code": 0, "periods_per_year": 1})
                funding_validation["distribution"] = "annual_moment_proxy_approximation"
            if not funding_validation["within_limits"]:
                raise ValidationError("MANDATE_INDEPENDENT_VALIDATION_FAILED", "所选SAA未通过独立资金验证，未采纳政策；请复核目标和模型。")
        baseline = copy.deepcopy(preview["source_snapshot"])
        baseline["name"] = body.name
        baseline["group_limits"] = preview["group_limits"]
        for asset in baseline["assets"]:
            asset.update(preview["constraints"][asset["id"]])
            asset["base_weight"] = candidate["weights"][asset["id"]]
        baseline["policy"] = {"schema_version": "1.0", "mandate_id": preview["mandate_id"],
            "mandate_hash": preview["mandate_hash"], "cma_id": preview["cma_id"], "cma_hash": preview["cma_hash"],
            "mandate": preview["mandate"], "assumptions": preview["assumptions"], "covariance": preview["covariance"],
            **{key: preview[key] for key in ("raw_assumptions", "model_result", "effective_returns", "effective_covariance", "multi_cma", "compatibility", "uncertainty_model") if key in preview},
            "expires_on": _policy_expiry(preview["mandate"]), "selection": candidate,
            "selection_request": preview["request"], "selection_preview_hash": preview["preview_hash"],
            "funding_model": preview.get("funding_model"), "funding_execution": preview.get("funding_execution"),
            **({"funding_validation": funding_validation} if funding_validation is not None else {}),
            "reason": body.reason, "confirmation_type": "researcher_policy_adoption", "independent_approval": False,
            "execution": preview["execution"]}
        if "multi_cma" in preview:
            baseline["policy"].update(schema_version="2.0", mode=body.request.mode)
            baseline["policy"]["multi_cma_budget"] = preview["multi_cma_budget"]
            from .multi_cma import require_payload_budget
            require_payload_budget(baseline)
        with self.artifacts.governance_lock.locked():
            self._require_active_mandate(preview["mandate_id"])
            identifiers = ([ref["cma_id"] for ref in preview["multi_cma"]["refs"]]
                           if "multi_cma" in preview else [preview["cma_id"]])
            for identifier in identifiers:
                self.cma.require_selectable(identifier)
            return self.baselines.save_baseline(baseline)
