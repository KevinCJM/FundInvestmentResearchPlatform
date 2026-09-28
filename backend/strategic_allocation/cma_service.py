"""Independent CMA lifecycle; one calculation implementation serves old and new APIs."""
from __future__ import annotations

from datetime import date
from hashlib import sha256
import numpy as np

from backend.custom_indicators.errors import ConflictError, IndicatorDomainError, ValidationError
from backend.product_pools.errors import ProductPoolError
from backend.product_pools.scope_lifecycle import normalize_scope_name
from backend.sensitivity.repository import digest_json
from . import kernels
from .cma_application import apply_model, frozen_assumptions, frozen_numeric_inputs
from .cma_evidence import CmaEvidence
from .cma_center_contracts import CmaSampleRequest, CmaSampleSummary
from .cma_model_contracts import StatisticalCmaContext, BayesianCmaRequest, RegimeCmaRequest
from . import cma_statistical_kernels
from .reference_inputs import automatic_research_day
from .cma_store import CmaDraftStore
from .reference_inputs import freeze_hash
from .versioning import BLOCKED, STALE, ResearchVersions
from .scope_facts import cma_scope_facts, cma_scope_difference, research_proxy_facts, research_proxy_difference, SCOPE_MESSAGES


def history_summary(item: dict) -> dict | None:
    """Describe saved evidence without resolving sources or recomputing history."""
    model = item["definition"].get("model") or {}
    if model.get("method") not in {"historical_statistics", "bayesian_niw", "historical_regime_occupancy", "long_term_scenario", "conditional_scenario"}:
        return None
    evidence = item.get("model_result", {}).get("model_audit", {}).get("evidence", {})
    sources = evidence.get("sources", [])
    if not item["definition"].get("strategic_universe_id"):
        sources = [product for asset in item["source_snapshot"].get("assets", [])
                   for product in asset.get("products", [])]
    names = [source.get("name") or source.get("code") or source.get("product_id") for source in sources]
    return {"window": model["window"], "start_date": evidence.get("actual_start"),
            "end_date": evidence.get("actual_end"), "observations": evidence.get("observations"),
            "source_names": list(dict.fromkeys(name for name in names if name))}


class CmaResearchService:
    def __init__(self, strategic):
        self.strategic = strategic
        self.artifacts = strategic.artifacts
        self.drafts = CmaDraftStore(self.artifacts.root.parent)
        self.evidence = CmaEvidence(strategic)

    def warm(self):
        self.evidence.prepare_regime_reader()
        from . import cma_scenario_kernels
        statistical = cma_statistical_kernels.warm()
        scenario = cma_scenario_kernels.warm()
        return {**statistical, "complete": statistical["complete"] and scenario["complete"],
                "scenario": scenario}

    def get(self, identifier: str) -> dict:
        return self.strategic._get(identifier, "capital_market_assumptions")

    def explicit_retired_ids(self) -> set[str]:
        retired = set()
        for summary in self.artifacts.list("retirement"):
            if summary.get("artifact_type") == "cma_retirement":
                item = self.artifacts.get(summary["id"], "retirement")
                retired.add(item["cma_id"])
        return retired

    def versions(self) -> ResearchVersions:
        from backend.product_pools.repository import ProductPoolRepository
        return ResearchVersions(self.artifacts, lambda: ProductPoolRepository(
            self.strategic.data.universe_dir / "product_pools.json"))

    def state(self, item: dict, versions: ResearchVersions | None = None) -> dict:
        """版本、上游与可用性；显式删除或上游删除都使此版本停止新引用。"""
        return (versions or self.versions()).cma_state(item)

    def retired_ids(self) -> set[str]:
        versions = self.versions()
        return versions.retired["cma"] | {item["id"] for item in self.current_versions()
                                          if versions.cma_state(item)["usable"]["status"] == BLOCKED}

    def require_selectable(self, identifier: str) -> dict:
        item = self.get(identifier)
        usable = self.state(item)["usable"]
        if identifier in self.explicit_retired_ids():
            raise ValidationError("CMA_RETIRED", "此 LTCMA 已停止新引用；历史政策仍可读取，请选择有效版本。")
        if usable["status"] == BLOCKED:
            raise ValidationError("CMA_UPSTREAM_DELETED", "此 LTCMA 对应的投资目标或研究范围已删除，已自动停止新引用。")
        if usable["status"] == STALE:
            raise ValidationError("CMA_NOT_CURRENT", "此 LTCMA 或其投资目标、研究范围已有新版本；旧版本仍可查看，请基于最新版本修改后再使用。")
        from .cma_selection import require_downstream_eligible
        require_downstream_eligible(item)
        return item

    def current_versions(self) -> list[dict]:
        versions = [self.get(summary["id"]) for summary in self.artifacts.list("series")
                    if summary.get("artifact_type") == "capital_market_assumptions"]
        superseded = {item["supersedes_cma_id"] for item in versions if item.get("supersedes_cma_id")}
        return [item for item in versions if item["id"] not in superseded]

    def active_names(self, excluding_id: str | None = None) -> list[str]:
        """名称是研究员在清单里区分版本的唯一线索，停止引用的版本释放其名称。"""
        retired = self.retired_ids()
        return [item["name"] for item in self.current_versions()
                if item["id"] not in retired and item["id"] != excluding_id]

    def _require_unique_name(self, name: str, excluding_id: str | None = None) -> None:
        normalized = normalize_scope_name(name)
        if any(normalize_scope_name(existing) == normalized for existing in self.active_names(excluding_id)):
            raise ConflictError("CMA_NAME_CONFLICT", "LTCMA 名称已存在，请修改名称。", "request.name")

    def list(self, query: str = "", method: str = "", include_retired: bool = False,
             offset: int = 0, limit: int | None = 50, selected_id: str | None = None) -> dict:
        from .cma_selection import downstream_eligibility
        lineage = self.versions()
        items = []
        versions = self.current_versions()
        if selected_id and not any(item["id"] == selected_id for item in versions):
            versions.append(self.get(selected_id))
        for item in versions:
            definition = item["definition"]
            model = definition.get("model") or {}
            kind = model.get("method", "manual")
            state = lineage.cma_state(item)
            is_retired = state["usable"]["status"] == BLOCKED
            if ((is_retired and not include_retired) or (method and method != kind)
                    or (query and query.casefold() not in item["name"].casefold())):
                continue
            facts = cma_scope_facts(item)
            items.append({key: item[key] for key in ("id", "name", "content_hash", "created_at")} | {
                "method": kind, "as_of": definition["as_of"], "currency": definition["currency"],
                "schema_version": definition.get("schema_version", "1.0"),
                "moment_semantics": definition.get("moment_semantics"),
                "retired": is_retired,
                "alloc_name": definition.get("alloc_name"),
                "strategic_universe_id": definition.get("strategic_universe_id"),
                "implementation_mapping_id": definition.get("implementation_mapping_id"),
                "asset_ids": [a["id"] for a in definition["assets"]],
                "scope_name": definition.get("alloc_name") or item["source_snapshot"].get("name"),
                "history": history_summary(item),
                "scope_facts": facts,
                "scope_fingerprint": digest_json({"contract": "strategic_scope_facts_v1", **facts}) if facts is not None else None,
                "research_proxy_facts": research_proxy_facts(model),
                **{key: definition.get(key) for key in ("return_basis", "fee_basis", "fx_hedging_basis")},
                **downstream_eligibility(item),
                **state,
            })
        # 同一目标版本、路径与范围版本排在一起；组按首次出现的顺序排列，组内保持原顺序。
        groups: dict = {}
        items.sort(key=lambda value: groups.setdefault(
            (value["research_path"], *(ref["id"] for ref in value["upstream"])), len(groups)))
        end = offset + limit if limit is not None else None
        return {"items": items[offset:end], "total": len(items), "offset": offset, "limit": limit}

    def view(self, identifier: str) -> dict:
        item = self.get(identifier)
        state = self.state(item)
        return {"version": item, "retired": state["usable"]["status"] == BLOCKED,
                **{key: state[key] for key in ("upstream", "usable")}, "version_info": state["version"]}

    def retire(self, identifier: str, body) -> dict:
        item = self.get(identifier)
        if item["content_hash"] != body.content_hash:
            raise ConflictError("CMA_VERSION_CHANGED", "所选版本与校验值不一致，请重新读取。")
        with self.artifacts.governance_lock.locked():
            if identifier not in self.explicit_retired_ids():
                if identifier not in {current["id"] for current in self.current_versions()}:
                    raise ConflictError("CMA_EDIT_CONFLICT", "此方案已被修改或删除，请返回列表重新打开后操作。")
                self.artifacts.save("retirement", {"artifact_type": "cma_retirement", "name": item["name"],
                    "cma_id": identifier, "cma_hash": item["content_hash"], "reason": body.reason,
                    "retired_on": str(date.today()), "research_only": True})
        return {"id": identifier, "retired": True}

    def capabilities(self) -> dict:
        available = {"manual", "black_litterman", "scenario_mixture"}
        if cma_statistical_kernels.execution_audit()["complete"]:
            available.update({"historical_statistics", "bayesian_niw", "historical_regime_occupancy"})
        from . import cma_scenario_kernels
        if cma_scenario_kernels.execution_audit()["complete"]:
            available.update({"long_term_scenario", "conditional_scenario"})
        labels = {"manual": "直接假设", "historical_statistics": "历史统计",
                  "black_litterman": "基准与观点", "bayesian_niw": "贝叶斯更新",
                  "scenario_mixture": "人工情景", "historical_regime_occupancy": "历史状态",
                  "long_term_scenario": "长期情景", "conditional_scenario": "条件情景"}
        return {"methods": [{"id": key, "name": label, "available": key in available,
                            "reason": None if key in available else "本进程尚未完成统计模型预热。"}
                           for key, label in labels.items()],
                "maximum_assets": 30, "maximum_scenarios": 60, "historical_frequency": "daily",
                "historical_currency": "CNY", "research_only": True}

    def study_options(self, as_of: date | None = None, section: str = "all",
                      selected_prior_id: str | None = None) -> dict:
        cutoff = automatic_research_day(self.strategic.data.data_dir)
        if as_of is not None and as_of > cutoff:
            raise ValidationError("LTCMA_KNOWLEDGE_CUTOFF", "LTCMA 研究日晚于平台知识截止日。")
        result = {}
        if section in {"all", "base"}:
            catalog = self.strategic.catalog()
            result.update(allocations=catalog["allocations"], strategic_universes=catalog["strategic_universes"],
                          assumptions=[], regime_runs=[], existing_names=[item["name"] for item in catalog["assumptions"]])
        if section in {"all", "priors"}:
            result["assumptions"] = self.list(include_retired=True, limit=None, selected_id=selected_prior_id)["items"]
        if section in {"all", "regimes", "scenarios"}:
            with self.evidence.runs.read_snapshot(self.evidence._regime_items()):
                if section in {"all", "regimes"}:
                    result["regime_runs"] = self.evidence.regime_options(as_of or cutoff)
                if section in {"all", "scenarios"}:
                    result["scenario_options"] = self.evidence.scenario_options(as_of or cutoff)
        return result

    def _evidence(self, request, source):
        model = request.model
        evidence = self.evidence.build(request, source)
        if isinstance(model, BayesianCmaRequest):
            prior = self.require_selectable(model.prior_ref.id)
            if prior["content_hash"] != model.prior_ref.content_hash:
                raise ConflictError("LTCMA_PRIOR_HASH", "先验版本指纹不一致，请重新选择。")
            definition = frozen_assumptions(prior)
            if definition["as_of"] > str(request.as_of):
                raise ValidationError("LTCMA_PRIOR_CONTEXT", "先验研究日晚于当前研究日。")
            if definition["currency"] != request.currency:
                raise ValidationError("LTCMA_PRIOR_CONTEXT", SCOPE_MESSAGES["scopeCurrency"])
            if [a["id"] for a in definition["assets"]] != model.asset_ids:
                raise ValidationError("LTCMA_PRIOR_CONTEXT", SCOPE_MESSAGES["scopeAssets"])
            issue = cma_scope_difference(prior, {"definition": request.model_dump(mode="json"), "source_snapshot": source})
            if issue:
                raise ValidationError("LTCMA_PRIOR_CONTEXT", SCOPE_MESSAGES[issue])
            if definition.get("schema_version") != "2.0" or definition.get("moment_semantics") != "annualized_periodic_arithmetic":
                raise ValidationError("LTCMA_PRIOR_MOMENTS", "NIW 先验须明确为基础期算术年化；旧假设请先复制并确认口径，不能直接猜测。")
            if any(definition.get(key) != getattr(request, key) for key in ("return_basis", "fee_basis", "fx_hedging_basis")):
                raise ValidationError("LTCMA_PRIOR_MOMENTS", "先验与当前研究的收益、费用或汇率口径不同。")
            previous_proxies = research_proxy_facts(prior["definition"].get("model"))
            current_proxies = research_proxy_facts(model.model_dump(mode="json"))
            proxy_issue = research_proxy_difference(previous_proxies, current_proxies)
            if proxy_issue:
                raise ValidationError("LTCMA_PRIOR_CONTEXT", SCOPE_MESSAGES[proxy_issue])
            means, covariance, _ = frozen_numeric_inputs(prior, self.artifacts)
            audit = prior.get("model_result", {}).get("model_audit", {})
            evidence["prior"] = {"means": means, "covariance": covariance,
                "posterior": audit.get("niw_posterior"), "evidence_end": audit.get("evidence", {}).get("actual_end")}
        elif isinstance(model, RegimeCmaRequest):
            evidence["regime"] = self.evidence.regime(model, evidence)
        elif model.method in {"long_term_scenario", "conditional_scenario"}:
            self.evidence.scenario(model, evidence)
        return evidence

    def sample(self, request: CmaSampleRequest) -> CmaSampleSummary:
        """Read evidence only: no prior, model estimate or saved artifact is needed."""
        model = request.model
        if model.as_of > automatic_research_day(self.strategic.data.data_dir):
            raise ValidationError("LTCMA_KNOWLEDGE_CUTOFF", "LTCMA 研究日晚于平台知识截止日。")
        source = self.strategic._source(request.alloc_name, model.as_of.isoformat(),
                                        strategic_universe_id=request.strategic_universe_id)
        if model.asset_ids != [asset["id"] for asset in source["assets"]]:
            raise ValidationError("SAA_CMA_AXIS", "样本资产及顺序须与所选范围一致，请重新加载范围。")
        if request.strategic_universe_id and model.currency != source["strategic_universe_snapshot"]["definition"]["currency"]:
            raise ValidationError("SAA_UNIVERSE_CURRENCY", "样本与战略范围的本位币不同。")
        evidence = self.evidence.build_sample(model, source, request.strategic_universe_id)
        return CmaSampleSummary(**{key: evidence["metadata"][key] for key in CmaSampleSummary.model_fields})

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
                years, short = cma_statistical_kernels.sample_window_diagnostics_kernel(evidence["returns"].shape[0])
                continuation = isinstance(request.model, BayesianCmaRequest) and request.model.prior_mode == "continue"
                evidence["metadata"]["sample_window"] = {"observation_years": float(years),
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
                arrays["evidence_period_contiguous"] = evidence["period_contiguous"]
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
            payload["warnings"].append("LTCMA 保存的是年化收益与风险研究假设，不代表已验证的多年预测。")
        if model_payload:
            payload["execution"] = {**payload["execution"], "cma_models": model_payload["model_result"]["execution"]}
            payload["warnings"].extend(model_payload["model_result"]["model_audit"]["limitations"])
        return freeze_hash(payload), arrays

    def preview(self, request) -> dict:
        return self.calculation(request)[0]

    def publish(self, body, replacing_id: str | None = None) -> dict:
        key = getattr(body, "idempotency_key", None)
        if body.request.schema_version == "2.0" and (getattr(body, "confirm", None) is not True or key is None):
            raise ValidationError("LTCMA_CONFIRM_REQUIRED", "新 LTCMA 发布需要明确确认和幂等操作键。")
        copied_from = getattr(body, "copied_from_id", None)
        request_payload = body.model_dump(mode="json")
        if replacing_id:
            request_payload["replacing_id"] = replacing_id
        request_hash = digest_json(request_payload) if key else None
        operation = "ltcma:" + key if key else None
        if operation:
            replay = self.artifacts.idempotent_result(operation, request_hash)
            if replay:
                return replay
        if copied_from:
            self.get(copied_from)
        original = self.get(replacing_id) if replacing_id else None
        if original and original["content_hash"] != body.expected_content_hash:
            raise ConflictError("CMA_VERSION_CHANGED", "方案与打开时的内容不一致，请重新读取后修改。")
        preview, arrays = self.calculation(body.request)
        if preview["preview_hash"] != body.preview_hash:
            raise ConflictError("SAA_CMA_PREVIEW_CHANGED", "假设或来源已变化，请重新验证后确认保存。")
        fields = {"artifact_type": "capital_market_assumptions", "name": body.request.name,
                  **preview, "research_only": True}
        if copied_from:
            fields["copied_from_id"] = copied_from
        if original:
            fields.update(study_id=original.get("study_id", original["id"]),
                          supersedes_cma_id=original["id"], supersedes_cma_hash=original["content_hash"])
        with self.artifacts.governance_lock.locked():
            # Replay is historical; only a new publication creates a prior reference.
            if operation:
                replay = self.artifacts.idempotent_result(operation, request_hash)
                if replay:
                    return replay
            if isinstance(body.request.model, BayesianCmaRequest):
                self.require_selectable(body.request.model.prior_ref.id)
            if original:
                current = self.current_versions()
                # Finish an interrupted index promotion through the existing store,
                # before treating this exact retry as a competing edit.
                recovered = next((item for item in current if item.get("supersedes_cma_id") == replacing_id
                                  and item.get("idempotency") == {"key": sha256(operation.encode()).hexdigest(),
                                                                 "request_hash": request_hash}), None)
                if recovered:
                    return self.artifacts.save("series", fields, arrays,
                                               idempotency_key=operation, request_hash=request_hash)
                if replacing_id in self.retired_ids() or replacing_id not in {item["id"] for item in current}:
                    raise ConflictError("CMA_EDIT_CONFLICT", "此方案已被修改或删除，请返回列表重新打开后操作。")
            self._require_unique_name(body.request.name, replacing_id)
            return self.artifacts.save("series", fields, arrays,
                                       idempotency_key=operation, request_hash=request_hash)
