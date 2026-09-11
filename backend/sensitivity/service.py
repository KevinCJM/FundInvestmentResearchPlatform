"""Research previews stay ephemeral; only confirmed publications are persisted."""
from __future__ import annotations

import threading
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.custom_indicators.errors import ValidationError
from backend.data_storage import guard_path
from backend.factor_research.numba_kernels import attribution_kernel
from .catalog import EVENT_TEMPLATES, FREQUENCY_LABELS, UNIT_CODES, VariableRegistry
from .contracts import CashflowPublishRequest, CashflowStudy, ModelFields, PublishRequest, RetireRequest
from .data import ResearchData
from .kernels import (
    cashflow_metrics_kernel, execution_audit, lag_features_kernel,
    reference_responses_kernel, restore_coefficients_kernel, sample_metadata_kernel,
    standardize_kernel, validation_status_kernel,
)
from .repository import ArtifactRepository, digest_json

ROOT_DATA = Path(__file__).resolve().parents[2] / "data"
_COMPUTE_SLOT = threading.BoundedSemaphore(1)
STATUS_TEXT = {0: "通过", 1: "训练或验证样本不足", 2: "常量、共线或无有效验证变异", 3: "样本外解释力低于门槛"}


def _timestamp(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def query_time(as_of=None):
    """Dates mean end of that UTC research day, capped at the real current time."""
    now = datetime.now(timezone.utc)
    if as_of is None:
        return now
    requested = datetime.combine(date.fromisoformat(str(as_of)), time.max, tzinfo=timezone.utc)
    if requested.date() > now.date():
        raise ValidationError("FUTURE_RESEARCH_DATE", "研究日期不能晚于今天。")
    return min(requested, now)


def release_status(release, retirements, at):
    created = _timestamp(release["created_at"])
    effective = _timestamp(release["effective_at"])
    if at < created or at < effective:
        return "not_yet_available"
    if any(item.get("release_id") == release["id"] and _timestamp(item["created_at"]) <= at for item in retirements):
        return "retired"
    if at >= _timestamp(release["expires_at"]):
        return "expired"
    return "active"


def _fit(x, y, split):
    standardized, means, scales, status = standardize_kernel(x, np.int64(split))
    if int(status):
        raise ValidationError("MODEL_INPUT_DEGENERATE", "训练输入样本不足或含常量因子；请检查数据与窗口，不会填充或编造系数。")
    coefficients, stats = attribution_kernel(y, standardized, np.zeros(y.shape[0]), np.int64(split), np.int64(1))
    restored = restore_coefficients_kernel(coefficients, means, scales)
    return restored, stats, means, scales


def _transient_result(fields: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic preview receipt without writing any file."""
    preview_hash = digest_json(fields)
    return {
        **fields,
        "id": f"transient-{preview_hash[:32]}",
        "preview_hash": preview_hash,
        "content_hash": preview_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "immutable": False,
        "transient": True,
    }


class ModelResearchService:
    def __init__(self, data_dir: Path = ROOT_DATA, domain="product"):
        if domain not in {"product", "transmission"}:
            raise ValueError("invalid research domain")
        self.data_dir = Path(data_dir)
        self.domain = domain
        self.root = self.data_dir / ("risk_models" if domain == "product" else "scenario_stress/transmission_models")
        self.artifacts = ArtifactRepository(self.root / "artifacts")
        self.variables = VariableRegistry(self.data_dir)
        self.data = ResearchData(self.data_dir, self.variables)

    def catalog(self):
        guard_path(self.root)
        return {
            "domain": self.domain,
            "variables": self.variables.list(),
            "methods": [{
                "id": "ols", "name": "多因子线性回归", "available": True,
                "description": "系数由历史数据联合估计，先验证再发布；不是经济因果证明。",
            }],
            "frequencies": [{"id": key, "name": value} for key, value in FREQUENCY_LABELS.items()],
            "event_templates": EVENT_TEMPLATES,
            "storage": {"logical_path": str(self.root), "managed_data_disk": True},
            "capabilities": {
                "cashflow_valuation": self.domain == "product",
                "coefficient_editing": False,
                "formal_backtest": False,
                "auto_training_on_read": False,
                "preview_persistence": False,
                "persist_only_on_publish": True,
            },
        }

    def _definition(self, raw):
        definition = ModelFields.model_validate(raw).model_dump(mode="json")
        if (definition["stage"] == "product") != (self.domain == "product"):
            raise ValidationError("MODEL_DOMAIN_MISMATCH", "产品暴露与宏观传导必须在各自研究模块中维护。")
        if date.fromisoformat(definition["as_of"]) > datetime.now(timezone.utc).date():
            raise ValidationError("FUTURE_MODEL_DATE", "模型研究截止日不能晚于今天。")
        roles = {"product": ("market", None), "event_macro": ("driver", "macro"), "macro_market": ("macro", "market")}
        input_role, output_role = roles[definition["stage"]]
        for key, role in (("inputs", input_role), ("outputs", output_role)):
            for identifier in definition[key]:
                variable = self.variables.get(identifier)
                if role not in variable["roles"]:
                    raise ValidationError(
                        "MODEL_VARIABLE_ROLE",
                        f'「{variable["name"]}」不能在此处作为{key == "inputs" and "输入" or "输出"}。',
                    )
        return definition

    def _calculate_model(self, raw):
        definition = self._definition(raw)
        panel = self.data.load(definition)
        input_count = len(panel["inputs"])
        features = lag_features_kernel(
            np.ascontiguousarray(panel["values"][:, :input_count]),
            np.int64(definition["lags"]),
        )
        start = panel["start_index"]
        x = features[start:]
        y = np.ascontiguousarray(panel["values"][start:, input_count:])
        split = panel["validation_index"] - start
        if split < 1 or split >= x.shape[0]:
            raise ValidationError("MODEL_VALIDATION_WINDOW", "所选区间缺少独立训练期或验证期。")
        sample_info = sample_metadata_kernel(x, y, panel["dates"][start:], np.int64(split))
        last_dates = [
            str(np.datetime64(int(max(item[2], item[3])), "D")) if max(item[2], item[3]) >= 0 else None
            for item in sample_info
        ]
        data_as_of = min(value for value in last_dates if value is not None) if any(last_dates) else None
        validation_coefficients, stats, means, scales = _fit(x, y, split)
        status = validation_status_kernel(
            validation_coefficients,
            stats,
            np.int64(definition["min_train"]),
            np.int64(definition["min_validation"]),
            np.float64(definition["minimum_validation_r2"]),
        )
        coefficients = validation_coefficients
        deployment_fit = "training_window"
        if definition["refit_after_validation"] and all(int(item) == 0 for item in status):
            coefficients, deployment_stats, _, _ = _fit(x, y, x.shape[0])
            if any(float(item) != 0.0 for item in deployment_stats[:, 7]):
                raise ValidationError("MODEL_REFIT_FAILED", "验证后全样本重估失败，请检查模型数据；未发布结果。")
            deployment_fit = "full_window_after_holdout_validation"
        references = reference_responses_kernel(
            coefficients,
            np.array([UNIT_CODES[item["unit"]] for item in panel["inputs"]], dtype=np.int64),
            np.array([UNIT_CODES[item["unit"]] for item in panel["outputs"]], dtype=np.int64),
        )
        rows = []
        for index, output in enumerate(panel["outputs"]):
            rows.append({
                "target_id": output["id"],
                "name": output["name"],
                "unit": output["unit"],
                "train_observations": int(sample_info[index, 0]),
                "validation_observations": int(sample_info[index, 1]),
                "data_as_of": last_dates[index],
                "training_r2": float(stats[index, 2]),
                "validation_r2": float(stats[index, 3]),
                "status": int(status[index]),
                "status_label": STATUS_TEXT[int(status[index])],
                "betas": coefficients[index, :-1].tolist(),
                "reference_responses": references[index].tolist(),
            })
        fields = {
            "name": definition["name"],
            "stage": definition["stage"],
            "method": "ols",
            "model": definition,
            "as_of": definition["as_of"],
            "data_as_of": data_as_of,
            "frequency": definition["frequency"],
            "inputs": panel["inputs"],
            "outputs": panel["outputs"],
            "targets": panel["targets"],
            "target_keys": [item["key"] for item in panel["targets"]],
            "rows": rows,
            "coefficients": coefficients.tolist(),
            "validation_coefficients": validation_coefficients.tolist(),
            "deployment_fit": deployment_fit,
            "publishable": all(int(item) == 0 for item in status),
            "blockers": [f'{row["name"]}：{row["status_label"]}' for row in rows if row["status"]],
            "provenance": panel["provenance"],
            "source_snapshot": panel["source_snapshot"],
            "calendar": panel["calendar"],
            "input_identity": panel["input_identity"],
            "execution": execution_audit(),
            "economic_causality": "not_identified",
            "limitations": [
                "仅解释模型覆盖的风险，不包含未建模残差或流动性反馈。",
                "验证区间未参与验证模型估计；选择重估时，发布系数另使用截至日期的全部样本。",
                "当前数据未认证完整历史修订PIT；发布后可用于研究，不提供过去已经可用的承诺。",
            ],
        }
        arrays = {
            "values": panel["values"],
            "available": panel["available"],
            "dates": panel["dates"],
            "features": x,
            "responses": y,
            "coefficients": coefficients,
            "validation_coefficients": validation_coefficients,
            "training_means": means,
            "training_scales": scales,
            "validation_stats": stats,
        }
        return _transient_result(fields), arrays

    def preview(self, raw):
        execution_audit()
        if not _COMPUTE_SLOT.acquire(blocking=False):
            raise ValidationError("MODEL_RESEARCH_BUSY", "已有敏感性研究正在计算，请稍后重试。")
        try:
            preview, _ = self._calculate_model(raw)
            return preview
        finally:
            _COMPUTE_SLOT.release()

    def _calculate_cashflow(self, raw):
        if self.domain != "product":
            raise ValidationError("CASHFLOW_DOMAIN", "现金流估值属于风险模型中心。")
        request = CashflowStudy.model_validate(raw)
        if request.as_of > datetime.now(timezone.utc).date():
            raise ValidationError("FUTURE_MODEL_DATE", "估值日期不能晚于今天。")
        factor = self.variables.get(request.yield_factor_id)
        if (
            "market" not in factor["roles"]
            or factor["unit"] != "bp"
            or factor["basis"] != "parallel_yield_change_cashflow_valuation"
        ):
            raise ValidationError(
                "CASHFLOW_YIELD_FACTOR",
                "固定现金流估值必须使用债券到期收益率平行变动；不能把 Shibor、政策利率或任意 bp 因子直接当作债券收益率。",
            )
        times = np.array([item.years for item in request.cashflows], dtype=np.float64)
        amounts = np.array([item.amount for item in request.cashflows], dtype=np.float64)
        metrics, status = cashflow_metrics_kernel(
            times, amounts, np.float64(request.yield_percent), np.int64(request.compounding)
        )
        if int(status):
            raise ValidationError("CASHFLOW_PRICING_FAILED", "现金流无法形成有限正价格，请检查期限、金额和收益率。")
        fields = {
            "name": request.name,
            "stage": "product",
            "method": "cashflow",
            "model": request.model_dump(mode="json"),
            "as_of": request.as_of.isoformat(),
            "data_as_of": request.as_of.isoformat(),
            "frequency": "single_shock",
            "inputs": [factor],
            "outputs": [],
            "target_keys": [f"bond:{request.product_id}"],
            "targets": [{
                "kind": "bond", "product_id": request.product_id, "name": request.name,
                "key": f"bond:{request.product_id}",
            }],
            "publishable": True,
            "blockers": [],
            "rows": [],
            "metrics": {
                "price": float(metrics[0]),
                "modified_duration": float(metrics[1]),
                "convexity": float(metrics[2]),
            },
            "execution": execution_audit(),
            "limitations": [
                "固定确定现金流，仅支持单次收益率平行冲击，不代表含权债或债券基金。",
                "未计入违约、利差分层、交易费用及持有期应计收益；原始现金流由研究者提供。",
            ],
        }
        return _transient_result(fields), {"times": times, "amounts": amounts}

    def cashflow_preview(self, raw):
        execution_audit()
        if not _COMPUTE_SLOT.acquire(blocking=False):
            raise ValidationError("MODEL_RESEARCH_BUSY", "已有敏感性研究正在计算，请稍后重试。")
        try:
            preview, _ = self._calculate_cashflow(raw)
            return preview
        finally:
            _COMPUTE_SLOT.release()

    def runs(self):
        """Only confirmed-publication runs exist on disk."""
        return {"items": self.artifacts.list("run")}

    def get_run(self, identifier):
        return self.artifacts.get(identifier, "run")

    def _persist_publication(self, preview, arrays, *, valid_days, note, effective_from=None):
        if not preview.get("publishable"):
            raise ValidationError("MODEL_NOT_PUBLISHABLE", "研究验证未通过，不能发布；请查看失败原因。")
        now = datetime.now(timezone.utc)
        requested = datetime.combine(effective_from or now.date(), time.min, tzinfo=timezone.utc)
        if requested.date() < now.date():
            raise ValidationError("PUBLICATION_BACKDATE", "发布不能回填过去的生效日期。")
        data_as_of = preview.get("data_as_of") or preview.get("model", {}).get("end_date") or preview["as_of"]
        if (now.date() - date.fromisoformat(data_as_of)).days > valid_days:
            raise ValidationError("MODEL_DATA_STALE", "研究数据已超出所选有效天数，请重新研究后发布。")
        effective = max(now, requested)
        data_expiry = datetime.combine(
            date.fromisoformat(data_as_of) + timedelta(days=valid_days + 1),
            time.min,
            tzinfo=timezone.utc,
        )
        if effective >= data_expiry:
            raise ValidationError("MODEL_EFFECTIVE_AFTER_EXPIRY", "生效日超出研究数据的有效期限。")
        release_key = digest_json({
            "preview_hash": preview["preview_hash"],
            "effective_date": effective.date().isoformat(),
            "valid_days": valid_days,
            "domain": self.domain,
        })
        with self.artifacts.governance_lock.locked():
            existing = self.artifacts.find("release", release_key)
            if existing:
                if release_status(existing, self.artifacts.list("retirement"), now) == "retired":
                    raise ValidationError("RELEASE_RETIRED", "此成果已停用，请重新研究产生新成果。")
                return existing
            run_key = digest_json({"preview_hash": preview["preview_hash"], "kind": "confirmed_publication_run"})
            run = self.artifacts.find("run", run_key)
            if run is None:
                persisted = {
                    key: value for key, value in preview.items()
                    if key not in {"id", "created_at", "content_hash", "immutable", "transient"}
                }
                persisted["cache_key"] = run_key
                run = self.artifacts.save("run", persisted, arrays)
            release = self.artifacts.save("release", {
                "name": run["name"],
                "stage": run["stage"],
                "run_id": run["id"],
                "run_hash": run["content_hash"],
                "method": run["method"],
                "as_of": run["as_of"],
                "data_as_of": data_as_of,
                "frequency": run["frequency"],
                "inputs": run["inputs"],
                "outputs": run["outputs"],
                "target_keys": run["target_keys"],
                "targets": run["targets"],
                "effective_at": effective.isoformat(),
                "expires_at": data_expiry.isoformat(),
                "note": note,
                "usage": "research_only",
                "cache_key": release_key,
                "preview_hash": preview["preview_hash"],
                "economic_causality": "not_identified" if self.domain == "transmission" else "not_applicable",
            })
            return release

    def publish(self, raw):
        request = PublishRequest.model_validate(raw)
        guard_path(self.root, write=True)
        execution_audit()
        if not _COMPUTE_SLOT.acquire(timeout=30):
            raise ValidationError("MODEL_RESEARCH_BUSY", "已有敏感性研究正在计算，请稍后重试。")
        try:
            preview, arrays = self._calculate_model(request.definition.model_dump(mode="json"))
        finally:
            _COMPUTE_SLOT.release()
        if preview["preview_hash"] != request.preview_hash:
            raise ValidationError(
                "MODEL_PREVIEW_CHANGED",
                "当前数据或研究参数与刚才确认的预览不一致，请重新计算并核对后再发布；未写入任何成果。",
            )
        return self._persist_publication(
            preview,
            arrays,
            valid_days=request.valid_days,
            note=request.note,
            effective_from=request.effective_from,
        )

    def publish_cashflow(self, raw):
        request = CashflowPublishRequest.model_validate(raw)
        guard_path(self.root, write=True)
        execution_audit()
        if not _COMPUTE_SLOT.acquire(timeout=30):
            raise ValidationError("MODEL_RESEARCH_BUSY", "已有敏感性研究正在计算，请稍后重试。")
        try:
            preview, arrays = self._calculate_cashflow(request.study.model_dump(mode="json"))
        finally:
            _COMPUTE_SLOT.release()
        if preview["preview_hash"] != request.preview_hash:
            raise ValidationError(
                "MODEL_PREVIEW_CHANGED",
                "当前现金流或估值参数与刚才确认的预览不一致，请重新计算后再发布；未写入任何成果。",
            )
        return self._persist_publication(
            preview,
            arrays,
            valid_days=request.valid_days,
            note=request.note,
            effective_from=request.effective_from,
        )

    def releases(self, as_of=None, product_key=None, include_unavailable=True):
        at = query_time(as_of)
        retirements = self.artifacts.list("retirement")
        items = []
        for summary in self.artifacts.list("release"):
            if product_key and product_key not in summary.get("target_keys", []):
                continue
            release = self.artifacts.get(summary["id"], "release")
            release["status"] = release_status(release, retirements, at)
            if include_unavailable or release["status"] == "active":
                items.append(release)
        return {"items": items, "as_of": at.isoformat()}

    def resolve_release(self, identifier, as_of=None):
        release = self.artifacts.get(identifier, "release")
        status = release_status(release, self.artifacts.list("retirement"), query_time(as_of))
        if status != "active":
            label = {"not_yet_available": "当时尚未发布或生效", "retired": "已停用", "expired": "已过期"}[status]
            raise ValidationError("MODEL_RELEASE_UNAVAILABLE", f"所选模型成果{label}，不能用于本次研究。")
        run = self.get_run(release["run_id"])
        if run["content_hash"] != release["run_hash"]:
            raise ValidationError("MODEL_RELEASE_CHANGED", "发布引用与研究运行不一致，已停止使用。")
        return release, run

    def retire(self, identifier, raw):
        request = RetireRequest.model_validate(raw)
        guard_path(self.root, write=True)
        with self.artifacts.governance_lock.locked():
            self.artifacts.get(identifier, "release")
            previous = next(
                (item for item in self.artifacts.list("retirement") if item["release_id"] == identifier),
                None,
            )
            if previous:
                return self.artifacts.get(previous["id"], "retirement")
            return self.artifacts.save("retirement", {"release_id": identifier, "note": request.note})
