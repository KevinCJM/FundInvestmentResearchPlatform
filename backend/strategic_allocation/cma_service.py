"""Independent CMA lifecycle; one calculation implementation serves old and new APIs."""
from __future__ import annotations

from datetime import date
import numpy as np

from backend.custom_indicators.errors import ConflictError, ValidationError
from backend.sensitivity.repository import digest_json
from . import kernels
from .cma_application import apply_model, frozen_assumptions, frozen_numeric_inputs
from .cma_evidence import CmaEvidence
from .cma_model_contracts import StatisticalCmaContext, BayesianCmaRequest, RegimeCmaRequest
from . import cma_statistical_kernels
from .reference_inputs import automatic_research_day
from .cma_store import CmaDraftStore
from .reference_inputs import freeze_hash


class CmaResearchService:
    def __init__(self, strategic):
        self.strategic = strategic
        self.artifacts = strategic.artifacts
        self.drafts = CmaDraftStore(self.artifacts.root.parent)
        self.evidence = CmaEvidence(strategic)

    def warm(self):
        self.evidence.prepare_regime_reader()
        return cma_statistical_kernels.warm()

    def get(self, identifier: str) -> dict:
        return self.strategic._get(identifier, "capital_market_assumptions")

    def retired_ids(self) -> set[str]:
        retired = set()
        for summary in self.artifacts.list("retirement"):
            if summary.get("artifact_type") == "cma_retirement":
                item = self.artifacts.get(summary["id"], "retirement")
                retired.add(item["cma_id"])
        return retired

    def require_selectable(self, identifier: str) -> dict:
        item = self.get(identifier)
        if identifier in self.retired_ids():
            raise ValidationError("CMA_RETIRED", "此 LTCMA 已停止新引用；历史政策仍可读取，请选择有效版本。")
        return item

    def list(self, query: str = "", method: str = "", include_retired: bool = False,
             offset: int = 0, limit: int = 50) -> dict:
        retired = self.retired_ids()
        items = []
        for summary in self.artifacts.list("series"):
            if summary.get("artifact_type") != "capital_market_assumptions":
                continue
            item = self.get(summary["id"])
            definition = item["definition"]
            model = definition.get("model") or {}
            kind = model.get("method", "manual")
            is_retired = item["id"] in retired
            if ((is_retired and not include_retired) or (method and method != kind)
                    or (query and query.casefold() not in item["name"].casefold())):
                continue
            items.append({key: item[key] for key in ("id", "name", "content_hash", "created_at")} | {
                "method": kind, "as_of": definition["as_of"], "currency": definition["currency"],
                "schema_version": definition.get("schema_version", "1.0"),
                "moment_semantics": definition.get("moment_semantics"),
                "horizon_years": definition["horizon_years"], "retired": is_retired,
                "alloc_name": definition.get("alloc_name"),
                "strategic_universe_id": definition.get("strategic_universe_id"),
                "implementation_mapping_id": definition.get("implementation_mapping_id"),
                "asset_ids": [a["id"] for a in definition["assets"]],
                "scope_name": definition.get("alloc_name") or item["source_snapshot"].get("name"),
            })
        return {"items": items[offset:offset + limit], "total": len(items), "offset": offset, "limit": limit}

    def view(self, identifier: str) -> dict:
        item = self.get(identifier)
        return {"version": item, "retired": identifier in self.retired_ids()}

    def retire(self, identifier: str, body) -> dict:
        item = self.get(identifier)
        if item["content_hash"] != body.content_hash:
            raise ConflictError("CMA_VERSION_CHANGED", "所选版本与校验值不一致，请重新读取。")
        with self.artifacts.governance_lock.locked():
            if identifier not in self.retired_ids():
                self.artifacts.save("retirement", {"artifact_type": "cma_retirement", "name": item["name"],
                    "cma_id": identifier, "cma_hash": item["content_hash"], "reason": body.reason,
                    "retired_on": str(date.today()), "research_only": True})
        return {"id": identifier, "retired": True}

    def capabilities(self) -> dict:
        available = {"manual", "black_litterman", "scenario_mixture"}
        if cma_statistical_kernels.execution_audit()["complete"]:
            available.update({"historical_statistics", "bayesian_niw", "historical_regime_occupancy"})
        labels = {"manual": "直接假设", "historical_statistics": "历史统计",
                  "black_litterman": "基准与观点", "bayesian_niw": "贝叶斯更新",
                  "scenario_mixture": "人工情景", "historical_regime_occupancy": "历史状态"}
        return {"methods": [{"id": key, "name": label, "available": key in available,
                            "reason": None if key in available else "本进程尚未完成统计模型预热。"}
                           for key, label in labels.items()],
                "maximum_assets": 30, "maximum_scenarios": 60, "historical_frequency": "daily",
                "historical_currency": "CNY", "research_only": True}

    def study_options(self) -> dict:
        catalog = self.strategic.catalog()
        return {"allocations": catalog["allocations"], "strategic_universes": catalog["strategic_universes"],
                "assumptions": self.list(limit=200)["items"],
                "regime_runs": self.evidence.regime_options()}

    def _evidence(self, request, source):
        model = request.model
        evidence = self.evidence.build(request, source)
        if isinstance(model, BayesianCmaRequest):
            prior = self.require_selectable(model.prior_ref.id)
            if prior["content_hash"] != model.prior_ref.content_hash:
                raise ConflictError("LTCMA_PRIOR_HASH", "先验版本指纹不一致，请重新选择。")
            definition = frozen_assumptions(prior)
            if ([a["id"] for a in definition["assets"]] != model.asset_ids
                    or definition["currency"] != request.currency or definition["horizon_years"] != request.horizon_years
                    or definition["as_of"] > str(request.as_of)
                    or definition.get("strategic_universe_id") != request.strategic_universe_id
                    or definition.get("alloc_name") != request.alloc_name):
                raise ValidationError("LTCMA_PRIOR_CONTEXT", "先验须属于同一资产定义、币种和期限，且在研究日可得。")
            if definition.get("schema_version") != "2.0" or definition.get("moment_semantics") != "annualized_periodic_arithmetic":
                raise ValidationError("LTCMA_PRIOR_MOMENTS", "NIW 先验须明确为基础期算术年化；旧假设请先复制并确认口径，不能直接猜测。")
            means, covariance, _ = frozen_numeric_inputs(prior, self.artifacts)
            audit = prior.get("model_result", {}).get("model_audit", {})
            evidence["prior"] = {"means": means, "covariance": covariance,
                "posterior": audit.get("niw_posterior"), "evidence_end": audit.get("evidence", {}).get("actual_end")}
        elif isinstance(model, RegimeCmaRequest):
            evidence["regime"] = self.evidence.regime(model, evidence)
        return evidence

    def calculation(self, request) -> tuple[dict, dict]:
        if request.schema_version == "2.0":
            from .contracts import CmaRequest
            request = CmaRequest.model_validate(request.model_dump(mode="json"))
        if request.schema_version == "2.0" and request.as_of > automatic_research_day(self.strategic.data.data_dir):
            raise ValidationError("LTCMA_KNOWLEDGE_CUTOFF", "LTCMA 研究日晚于平台知识截止日。")
        source = self.strategic._definition_source(request.model_dump(mode="json"))
        if request.strategic_universe_id:
            universe = source["strategic_universe_snapshot"]["definition"]
            if request.currency != universe["currency"]:
                raise ValidationError("SAA_UNIVERSE_CURRENCY", "长期假设与战略范围的本位币不同。")
            metadata = {a["id"]: a for a in universe["assets"]}
            if any(a.id not in metadata or a.role != metadata[a.id]["role"] or a.liquidity != metadata[a.id]["liquidity"] for a in request.assets):
                raise ValidationError("SAA_UNIVERSE_ROLE", "经济角色或流动性与不可变战略定义不同；请确认新的范围版本。")
        names = [asset["id"] for asset in source["assets"]]
        if [asset.id for asset in request.assets] != names:
            raise ValidationError("SAA_CMA_AXIS", "长期假设的资产及顺序须与所选大类一致；请重新加载分类。")
        model_payload = {}
        evidence = None
        if request.model is not None:
            if isinstance(request.model, StatisticalCmaContext):
                cma_statistical_kernels.require_ready()
                evidence = self._evidence(request, source)
                years, short, ratio = cma_statistical_kernels.sample_horizon_diagnostics_kernel(
                    evidence["returns"].shape[0], request.horizon_years)
                continuation = isinstance(request.model, BayesianCmaRequest) and request.model.prior_mode == "continue"
                evidence["metadata"]["sample_horizon"] = {"observation_years": float(years),
                    "forecast_years": request.horizon_years, "window_to_horizon_ratio": float(ratio),
                    "scope": "incremental_evidence_batch" if continuation else "selected_likelihood_window",
                    "short_window_review": bool(short), "review_rule": "below_three_observation_years_soft_guidance"}
                if short:
                    text = ("新增 NIW 证据批次不足三年；这不是全部历史信息量，需连同冻结先验审阅。" if continuation
                            else "历史窗口不足三个观察年，建议复核周期覆盖和均值稳定性；三年是研究提示，不是统计有效性硬门槛。")
                    evidence["metadata"]["warnings"] = [*evidence["metadata"].get("warnings", []), text]
            model_payload, arrays = apply_model(request, names, evidence=evidence)
            if evidence is not None:
                arrays["evidence_returns"] = evidence["returns"]
                arrays["evidence_dates"] = np.asarray(evidence["dates"], dtype="datetime64[D]").astype(np.int64)
            covariance = arrays["covariance"]
            min_eigenvalue = model_payload["model_result"]["model_audit"]["min_correlation_eigenvalue"]
        else:
            vol = np.asarray([a.annual_volatility for a in request.assets], dtype=np.float64)
            corr = np.asarray(request.correlation, dtype=np.float64)
            if request.schema_version == "2.0" and any(a.annual_volatility == 0 for a in request.assets):
                cma_statistical_kernels.require_ready()
                covariance, min_eigenvalue = cma_statistical_kernels.manual_covariance_with_cash(vol, corr)
            else:
                covariance, min_eigenvalue = kernels.cma_covariance_kernel(vol, corr)
            arrays = {"covariance": covariance}
        reference = None
        if request.risk_origin == "historical_reference":
            reference = self.strategic.risk_reference(request.risk_reference)
            if reference["preview_hash"] != request.risk_reference_hash:
                raise ConflictError("SAA_RISK_REFERENCE_CHANGED", "历史风险来源已变化，请重新读取风险参考。")
            if reference["volatility"] != vol.tolist() or reference["correlation"] != request.correlation:
                raise ValidationError("SAA_RISK_REFERENCE_EDITED", "风险数值已被人工修改，请明确改为人工风险假设，不沿用原参考认证。")
        payload = {"definition": request.model_dump(mode="json"), "source_snapshot": source,
            "covariance": covariance.tolist(), "min_correlation_eigenvalue": float(min_eigenvalue),
            "risk_reference": reference, "execution": kernels.execution_audit(),
            "warnings": [*source["pit"]["reasons"], "经济角色、流动性与同币种总收益口径由研究员确认，不是系统校准的宏观暴露。",
                         ("预期收益与均值不确定半宽是研究假设；不确定半宽不是波动率或统计置信区间。"
                          if evidence is None else "均值不确定性按模型标注其估计方法与限制，不等同于资产波动率。") ]}
        payload.update(model_payload)
        if request.schema_version == "2.0":
            payload["semantics"] = {"moment_semantics": request.moment_semantics, "fee_basis": request.fee_basis,
                "fx_hedging_basis": request.fx_hedging_basis, "covariance_role": "asset_return",
                "historical_pit_proven": False}
            payload["warnings"].append("LTCMA 保存的是研究假设；长期预测期限本身不证明预测有效。")
        if model_payload:
            payload["execution"] = {**payload["execution"], "cma_models": model_payload["model_result"]["execution"]}
            payload["warnings"].extend(model_payload["model_result"]["model_audit"]["limitations"])
        return freeze_hash(payload), arrays

    def preview(self, request) -> dict:
        return self.calculation(request)[0]

    def publish(self, body) -> dict:
        key = getattr(body, "idempotency_key", None)
        if body.request.schema_version == "2.0" and (getattr(body, "confirm", None) is not True or key is None):
            raise ValidationError("LTCMA_CONFIRM_REQUIRED", "新 LTCMA 发布需要明确确认和幂等操作键。")
        copied_from = getattr(body, "copied_from_id", None)
        request_hash = digest_json(body.model_dump(mode="json")) if key else None
        operation = "ltcma:" + key if key else None
        if operation:
            replay = self.artifacts.idempotent_result(operation, request_hash)
            if replay:
                return replay
        if copied_from:
            self.get(copied_from)
        preview, arrays = self.calculation(body.request)
        if preview["preview_hash"] != body.preview_hash:
            raise ConflictError("SAA_CMA_PREVIEW_CHANGED", "假设或来源已变化，请重新验证后确认保存。")
        fields = {"artifact_type": "capital_market_assumptions", "name": body.request.name,
                  **preview, "research_only": True}
        if copied_from:
            fields["copied_from_id"] = copied_from
        return self.artifacts.save("series", fields, arrays,
                                   idempotency_key=operation, request_hash=request_hash)
