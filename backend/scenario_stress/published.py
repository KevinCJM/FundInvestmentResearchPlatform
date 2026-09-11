"""Three scenario entry points, one immutable application path; no request-time fitting."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from backend.custom_indicators.errors import ValidationError
from backend.custom_indicators.portfolio_repository import PortfolioRunRepository
from backend.data_storage import guard_path
from backend.sensitivity.catalog import UNIT_CODES
from backend.sensitivity.kernels import (
    cashflow_shock_kernel, display_shocks_kernel, ending_weights_kernel,
    execution_audit, impact_summary_kernel, linked_factor_contributions_kernel,
    transmission_kernel, wealth_impact_kernel,
)
from backend.sensitivity.repository import ArtifactRepository, digest_json
from backend.sensitivity.service import ModelResearchService, ROOT_DATA, _timestamp, query_time, release_status
from .numba_kernels import factor_to_asset_kernel, portfolio_weight_summary_kernel
from .published_contracts import ImpactRequest, ScenarioFields, ScenarioPublish


def _match_columns(required, supplied):
    by_id = {item["id"]: (index, item) for index, item in enumerate(supplied)}
    indices = []
    for variable in required:
        match = by_id.get(variable["id"])
        if not match or match[1]["contract_hash"] != variable["contract_hash"]:
            raise ValidationError("TRANSMISSION_VARIABLE_MISMATCH", f'缺少「{variable["name"]}」或其单位、来源版本不一致；不能按名称猜测匹配。')
        indices.append(match[0])
    return indices


def _valid_path(path, variables):
    if path.ndim != 2 or path.shape[1] != len(variables) or not 1 <= path.shape[0] <= 1200:
        raise ValidationError("SCENARIO_PATH_SHAPE", "情景路径维度与所选变量不一致。")
    # Small public-boundary validation, not a numerical transform or fit.
    for column, variable in enumerate(variables):
        for number in path[:, column]:
            if not np.isfinite(number) or (variable["unit"] == "return" and number <= -1.0):
                raise ValidationError("SCENARIO_PATH_INVALID", f'「{variable["name"]}」出现无效变动；收益率不能低于或等于 -100%。')


class PublishedScenarioService:
    def __init__(self, data_dir: Path = ROOT_DATA):
        self.data_dir = Path(data_dir)
        self.models = ModelResearchService(self.data_dir, "transmission")
        self.risks = ModelResearchService(self.data_dir, "product")
        self.variables = self.risks.variables
        self.artifacts = ArtifactRepository(self.data_dir / "scenario_stress/published")
        self.impacts = ArtifactRepository(self.data_dir / "scenario_stress/impacts")
        self.portfolios = PortfolioRunRepository(self.data_dir / "portfolio_runs.json")

    def _project(self, release_id, path, supplied, frequency, expected_stage):
        release, run = self.models.resolve_release(release_id)
        if run["stage"] != expected_stage or run["frequency"] != frequency:
            raise ValidationError("TRANSMISSION_MODEL_MISMATCH", "传导模型的阶段或研究频率不匹配。")
        indices = _match_columns(run["inputs"], supplied)
        ordered = np.ascontiguousarray(path[:, indices])
        arrays = self.models.artifacts.arrays(run["id"], names=("coefficients",))
        output = transmission_kernel(ordered, arrays["coefficients"], np.int64(run["model"]["lags"]))
        _valid_path(output, run["outputs"])
        return output, run["outputs"], {"release_id": release["id"], "release_hash": release["content_hash"],
            "run_id": run["id"], "run_hash": run["content_hash"], "stage": expected_stage,
            "name": run["name"], "inputs": run["inputs"], "outputs": run["outputs"],
            "path": output.tolist(), "expires_at": release["expires_at"],
            "interpretation": "conditional_statistical_response_not_identified_causality"}

    def _compile_preview(self, raw):
        execution_audit()
        request = ScenarioFields.model_validate(raw)
        definition = request.model_dump(mode="json")
        if request.entry == "market":
            inputs = [self.variables.get(key) for key in request.input_ids]
            if any("market" not in variable["roles"] for variable in inputs):
                raise ValidationError("SCENARIO_MARKET_FACTOR_REQUIRED", "直接市场入口只能选择市场风险因子。")
        else:
            first_id = request.event_model_release_id if request.entry == "event" else request.macro_model_release_id
            _, first_run = self.models.resolve_release(first_id)
            inputs = first_run["inputs"]
        if any(len(row) != len(inputs) for row in request.rows):
            raise ValidationError("SCENARIO_INPUT_DIMENSION", "每期变动数量必须与模型输入变量一一对应。")
        if len(inputs) * len(request.rows) > 9600:
            raise ValidationError("SCENARIO_INPUT_LIMIT", "情景输入超过计算预算。")
        display = np.asarray(request.rows, dtype=np.float64)
        path = display_shocks_kernel(
            display,
            np.array([UNIT_CODES[item["unit"]] for item in inputs], dtype=np.int64),
        )
        _valid_path(path, inputs)
        source_path = path
        current_variables = inputs
        lineage = []
        if request.entry == "event":
            path, current_variables, step = self._project(
                request.event_model_release_id,
                path,
                current_variables,
                request.frequency,
                "event_macro",
            )
            lineage.append(step)
        if request.entry != "market":
            path, current_variables, step = self._project(
                request.macro_model_release_id,
                path,
                current_variables,
                request.frequency,
                "macro_market",
            )
            lineage.append(step)
        fields = {
            "name": request.name,
            "entry": request.entry,
            "definition": definition,
            "definition_hash": digest_json(definition),
            "frequency": request.frequency,
            "input_variables": inputs,
            "factors": current_variables,
            "path": path.tolist(),
            "lineage": lineage,
            "horizon": path.shape[0],
            "market": "CN",
            "currency": "CNY",
            "execution": execution_audit(),
            "limitations": [
                "这是每期变动的假设路径，不是情景发生概率或预测。",
                "宏观传导为已验证的条件统计关系，未认定经济因果；未加入额外金融反馈。",
                "传导截距不作为额外冲击，路径表示相对无冲击基线的变化。",
            ],
        }
        preview_hash = digest_json(fields)
        preview = {
            **fields,
            "id": f"transient-{preview_hash[:32]}",
            "preview_hash": preview_hash,
            "content_hash": preview_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "immutable": False,
            "transient": True,
        }
        return preview, {"source_path": source_path, "factor_path": path}

    def preview(self, raw):
        preview, _ = self._compile_preview(raw)
        return preview

    def publish(self, raw):
        request = ScenarioPublish.model_validate(raw)
        guard_path(self.artifacts.root, write=True)
        preview, arrays = self._compile_preview(request.definition.model_dump(mode="json"))
        if preview["preview_hash"] != request.preview_hash:
            raise ValidationError(
                "SCENARIO_PREVIEW_CHANGED",
                "当前输入、上游模型或数据与刚才确认的预览不一致，请重新预览后再发布；未写入任何情景成果。",
            )
        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=request.valid_days)
        for step in preview["lineage"]:
            release, _ = self.models.resolve_release(step["release_id"])
            if release["content_hash"] != step["release_hash"]:
                raise ValidationError("SCENARIO_LINEAGE_CHANGED", "传导成果引用发生变化，请重新预览。")
            expires = min(expires, _timestamp(release["expires_at"]))
        key = digest_json({"preview_hash": preview["preview_hash"], "valid_days": request.valid_days})
        with self.artifacts.governance_lock.locked():
            existing = self.artifacts.find("release", key)
            if existing:
                status = release_status(existing, self.artifacts.list("retirement"), now)
                if status != "active":
                    raise ValidationError("SCENARIO_ALREADY_INACTIVE", "该预览对应的情景已停用或到期，请重新研究并预览。")
                return existing
            preview_key = digest_json({"preview_hash": preview["preview_hash"], "kind": "confirmed_scenario_preview"})
            persisted_preview = self.artifacts.find("preview", preview_key)
            if persisted_preview is None:
                persisted = {
                    key: value for key, value in preview.items()
                    if key not in {"id", "created_at", "content_hash", "immutable", "transient"}
                }
                persisted["cache_key"] = preview_key
                persisted_preview = self.artifacts.save("preview", persisted, arrays)
            return self.artifacts.save("release", {
                "name": persisted_preview["name"],
                "entry": persisted_preview["entry"],
                "preview_id": persisted_preview["id"],
                "preview_hash": preview["preview_hash"],
                "preview_content_hash": persisted_preview["content_hash"],
                "frequency": persisted_preview["frequency"],
                "factors": persisted_preview["factors"],
                "horizon": persisted_preview["horizon"],
                "lineage": persisted_preview["lineage"],
                "effective_at": now.isoformat(),
                "expires_at": expires.isoformat(),
                "usage": "research_only",
                "note": request.note,
                "cache_key": key,
            })

    def releases(self, as_of=None):
        at = query_time(as_of)
        items = []
        retirements = self.artifacts.list("retirement")
        for summary in self.artifacts.list("release"):
            item = self.artifacts.get(summary["id"], "release")
            item["status"] = release_status(item, retirements, at)
            if item["status"] == "active":
                try:
                    for step in item["lineage"]:
                        self.models.resolve_release(step["release_id"], as_of)
                except ValidationError as exc:
                    item["status"], item["reason"] = "dependency_unavailable", exc.message
            items.append(item)
        return {"items": items}

    def resolve_release(self, identifier, as_of=None):
        release = self.artifacts.get(identifier, "release")
        status = release_status(release, self.artifacts.list("retirement"), query_time(as_of))
        if status != "active":
            raise ValidationError("SCENARIO_RELEASE_UNAVAILABLE", "情景在所选研究日期尚未发布、已停用或已到期。")
        for step in release["lineage"]:
            dependency, _ = self.models.resolve_release(step["release_id"], as_of)
            if dependency["content_hash"] != step["release_hash"]:
                raise ValidationError("SCENARIO_LINEAGE_CHANGED", "情景传导模型引用校验失败。")
        preview = self.artifacts.get(release["preview_id"], "preview")
        expected_hash = release.get("preview_content_hash") or release["preview_hash"]
        if preview["content_hash"] != expected_hash:
            raise ValidationError("SCENARIO_PREVIEW_CHANGED", "情景预览内容与发布引用不一致。")
        return release, preview

    def retire(self, identifier, note=""):
        guard_path(self.artifacts.root, write=True)
        with self.artifacts.governance_lock.locked():
            self.artifacts.get(identifier, "release")
            old = next((item for item in self.artifacts.list("retirement") if item["release_id"] == identifier), None)
            if old:
                return self.artifacts.get(old["id"], "retirement")
            return self.artifacts.save("retirement", {"release_id": identifier, "note": note})

    def portfolio_choices(self):
        guard_path(self.data_dir)
        if not self.portfolios.store.path.exists():
            return {"items": []}
        # Read the existing immutable repository without invoking diagnosis/backtest.
        items = self.portfolios.store.read_unlocked()["items"]
        return {"items": [{"id": item["id"], "name": item.get("target_name", item["id"]),
                           "as_of": item.get("effective_as_of"), "created_at": item.get("created_at")}
                          for item in reversed(items) if item.get("immutable")][:200]}

    def _target(self, request, exposure_run):
        if request.target.kind == "product":
            key = request.target.product_key
            target = next((item for item in exposure_run["targets"] if item["key"] == key), None)
            if not target:
                raise ValidationError("EXPOSURE_PRODUCT_NOT_COVERED", "所选已发布模型没有覆盖这只产品。")
            return [target], np.ones(1), {"kind": "product", "product_key": key, "name": target["name"]}
        guard_path(self.data_dir)
        if not self.portfolios.store.path.exists():
            raise ValidationError("PORTFOLIO_SNAPSHOT_MISSING", "没有可读取的组合研究快照。")
        snapshot = self.portfolios.get(request.target.portfolio_run_id)
        if not snapshot.get("immutable") or not snapshot.get("assets") or not snapshot.get("daily_weights"):
            raise ValidationError("PORTFOLIO_SNAPSHOT_INVALID", "组合运行不是包含真实权重路径的不可变快照。")
        at = query_time(request.as_of)
        if _timestamp(snapshot["created_at"]) > at or snapshot["effective_as_of"] > str(request.as_of):
            raise ValidationError("PORTFOLIO_SNAPSHOT_FUTURE", "组合快照在所选研究日期尚不可得。")
        assets = [{"key": item["key"], "kind": item["kind"], "product_id": item["product_id"],
                   "name": item.get("name", item["product_id"])} for item in snapshot["assets"]]
        previous = np.array(snapshot["daily_weights"][-1], dtype=np.float64)
        last_returns = np.array(snapshot["asset_returns"][-1], dtype=np.float64)
        if previous.shape != (len(assets),) or last_returns.shape != previous.shape:
            raise ValidationError("PORTFOLIO_SNAPSHOT_DIMENSION", "组合快照资产与期末权重维度不一致。")
        # Existing daily_weights are beginning-of-period, not the end holdings.
        weights, status = ending_weights_kernel(previous, last_returns)
        if int(status):
            raise ValidationError("PORTFOLIO_END_WEIGHTS_INVALID", "无法从锁定快照还原期末持仓权重。")
        return assets, weights, {"kind": "portfolio_run", "portfolio_run_id": snapshot["id"],
            "name": snapshot.get("target_name", snapshot["id"]), "holdings_date": snapshot["effective_as_of"],
            "snapshot_hash": digest_json(snapshot), "weight_basis": "after_last_observed_return"}

    def impact(self, raw):
        """Calculate one transient impact from published inputs; never persist it."""
        execution_audit()
        request = ImpactRequest.model_validate(raw)
        scenario_release, preview = self.resolve_release(request.scenario_release_id, request.as_of)
        exposure_release, run = self.risks.resolve_release(request.exposure_release_id, request.as_of)
        assets, weights, target = self._target(request, run)
        _, _, _, status = portfolio_weight_summary_kernel(
            weights, np.float64(1), np.float64(1e-8), np.float64(2), np.float64(3)
        )
        if int(status) or any(value < 0 for value in weights):
            raise ValidationError("STRESS_WEIGHTS_INVALID", "本研究应用仅支持权重合计 100% 的非负产品持仓。")
        input_indices = _match_columns(preview["factors"], run["inputs"])
        source = self.artifacts.arrays(preview["id"], names=("factor_path",))["factor_path"]
        required_arrays = ("times", "amounts") if run["method"] == "cashflow" else ("coefficients",)
        risk_arrays = self.risks.artifacts.arrays(run["id"], names=required_arrays)
        shocks = np.zeros((source.shape[0], len(run["inputs"])), dtype=np.float64)
        for source_index, destination in enumerate(input_indices):
            shocks[:, destination] = source[:, source_index]
        zero_factors = [item["id"] for index, item in enumerate(run["inputs"]) if index not in input_indices]
        if run["method"] == "ols" and run["frequency"] != preview["frequency"]:
            raise ValidationError("STRESS_FREQUENCY_MISMATCH", "情景周期与暴露模型周期不同，不能直接混用。")
        coefficient_rows = {item["key"]: index for index, item in enumerate(run["targets"])}
        betas = np.zeros((len(assets), len(run["inputs"])), dtype=np.float64)
        returns = np.zeros((shocks.shape[0], len(assets)), dtype=np.float64)
        missing = []
        for asset_index, asset in enumerate(assets):
            if weights[asset_index] == 0:
                continue
            row = coefficient_rows.get(asset["key"])
            if row is None:
                missing.append(asset["name"])
                continue
            if run["method"] == "ols":
                betas[asset_index] = risk_arrays["coefficients"][row, :len(run["inputs"])]
            else:
                if len(assets) != 1 or shocks.shape[0] != 1:
                    raise ValidationError("CASHFLOW_SINGLE_SHOCK_ONLY", "固定现金流估值当前仅支持单产品、单期冲击。")
                values, price_status = cashflow_shock_kernel(
                    risk_arrays["times"], risk_arrays["amounts"],
                    np.float64(run["model"]["yield_percent"]), np.int64(run["model"]["compounding"]),
                    np.float64(shocks[0, 0]),
                )
                if int(price_status):
                    raise ValidationError("CASHFLOW_STRESS_INVALID", "冲击后收益率无法形成有效现金流价格。")
                returns[0, 0] = values[0]
                betas[0, 0] = -float(run["metrics"]["modified_duration"]) / 10000.0
        if missing:
            raise ValidationError(
                "EXPOSURE_HOLDINGS_MISSING",
                "以下非零持仓没有已发布暴露：" + "、".join(missing) + "。请在风险模型中心补充研究；不会删除持仓或重分配权重。",
            )
        if run["method"] == "ols":
            returns, projection_status, _, _ = factor_to_asset_kernel(
                shocks, betas, np.zeros(len(assets)), np.ones(len(assets), dtype=np.uint8)
            )
            if int(projection_status):
                raise ValidationError("STRESS_LINEAR_RANGE", "线性模型在该冲击下产生无效收益，请降低冲击或改用适合的估值模型。")
        path, contributions, _daily, beginnings, wealth_status = wealth_impact_kernel(
            returns, weights, np.int64(0 if request.holding_policy == "buy_and_hold" else 1)
        )
        if int(wealth_status):
            raise ValidationError("STRESS_WEALTH_INVALID", "情景产生非有限或非正组合净值，已停止计算。")
        factor_contributions = linked_factor_contributions_kernel(shocks, betas, beginnings, returns)
        summary = impact_summary_kernel(path, contributions, factor_contributions, np.float64(request.notional))
        if abs(float(summary[4])) > 1e-9 or abs(float(summary[5])) > 1e-9:
            raise ValidationError("STRESS_RECONCILIATION_FAILED", "情景损益贡献未能与终值对账，已阻断结果返回。")
        fields = {
            "name": f'{target["name"]} · {scenario_release["name"]}',
            "request": request.model_dump(mode="json"), "target": target,
            "as_of": str(request.as_of), "scenario_release_id": scenario_release["id"],
            "scenario_hash": scenario_release["content_hash"], "exposure_release_id": exposure_release["id"],
            "exposure_hash": exposure_release["content_hash"], "exposure_run_id": run["id"],
            "frequency": preview["frequency"], "factors": run["inputs"], "assumed_unchanged_factors": zero_factors,
            "holding_policy": request.holding_policy, "notional": request.notional,
            "summary": {"terminal_return": float(summary[0]), "terminal_nav": float(summary[1]),
                "max_drawdown": float(summary[2]), "pnl_amount": float(summary[3]),
                "asset_reconciliation_error": float(summary[4]), "factor_reconciliation_error": float(summary[5])},
            "path": [{"step": index + 1, "return": float(row[0]), "nav": float(row[1]), "drawdown": float(row[2])}
                     for index, row in enumerate(path)],
            "by_asset": [{**asset, "weight": float(weights[index]), "contribution": float(contributions[index])}
                         for index, asset in enumerate(assets)],
            "by_factor": [{"id": item["id"], "name": item["name"], "contribution": float(factor_contributions[index])}
                          for index, item in enumerate(run["inputs"])] + [{"id": "nonlinear", "name": "非线性重新估值差额", "contribution": float(factor_contributions[-1])}],
            "lineage": preview["lineage"], "execution": execution_audit(), "cache_hit": False,
            "transient": True,
            "limitations": ["这是已发布模型解释的确定性情景影响，不是总风险预测，不给出 VaR、ES 或发生概率。",
                "未建模的产品残差、流动性、费用、违约风险不会被宣称已经覆盖。",
                "固定权重模式明确假设每期零成本再平衡；买入持有模式保留自然权重漂移。",
                "本次产品/组合压测结果只存在于当前页面，不写入磁盘。"]}
        content_hash = digest_json(fields)
        return {**fields, "id": f"transient-impact-{content_hash[:32]}", "content_hash": content_hash,
                "created_at": datetime.now(timezone.utc).isoformat()}
