"""Small orchestration layer over existing policy, numerical and artifact services."""

import io
import json
import threading
import zipfile
from contextlib import contextmanager
from datetime import date
import numpy as np
from backend.custom_indicators.errors import (
    ConflictError,
    IndicatorDomainError,
    ValidationError,
)
from backend.sensitivity.repository import digest_json
from backend.sensitivity.kernels import warm_sensitivity_kernels
from .contracts import ImplementationCandidate
from .repository import PackageRepository
from .sources import AllocationSources
from . import risk, risk_kernels, cost_kernels, funding, path_kernels
from .evaluation import evaluate, execution_manifest, combined_fee

_COMPUTE = threading.BoundedSemaphore(1)


@contextmanager
def compute_slot():
    if not _COMPUTE.acquire(blocking=False):
        raise ConflictError("IMPLEMENTATION_BUSY", "已有实施研究正在计算，请稍后重试。")
    try:
        yield
    finally:
        _COMPUTE.release()


def operation_key(key, stage):
    return digest_json({"operation": key, "stage": stage})


class ImplementationService:
    def __init__(self, strategic, *, scenarios=None):
        self.strategic = strategic
        self.sources = AllocationSources(strategic)
        self.repository = PackageRepository(
            strategic.artifacts.root.parents[1] / "pre_investment" / "artifacts"
        )
        self.scenarios = scenarios

    def warm(self):
        warm_sensitivity_kernels()
        risk_kernels.warm()
        cost_kernels.warm()
        funding.warm()
        path_kernels.warm()
        combined_fee(0.01, 0.01)
        return {
            "complete": True,
            "risk": risk_kernels.audit(),
            "costs": cost_kernels.audit(),
        }

    def candidate_identity(self, candidate):
        source = self.sources.resolve(candidate.source, candidate.as_of, current=False)
        raw = candidate.model_dump(mode="json")
        refs = source["refs"]
        return digest_json({"candidate": raw, "dependencies": refs}), refs

    def catalog(self):
        return {
            **self.sources.catalog(),
            "scenario_options": (
                self.scenarios.options()
                if self.scenarios
                else {"scenarios": [], "exposures": []}
            ),
        }

    def preview(self, candidate):
        candidate_hash, _ = self.candidate_identity(candidate)
        with compute_slot():
            return evaluate(self, candidate, candidate_hash)[0]

    def optimize(self, candidate):
        risk_kernels.require_ready()
        with compute_slot():
            return self._optimize(candidate)

    def _optimize(self, candidate):
        source = self.sources.resolve(candidate.source, candidate.as_of)
        factors, products, metadata = risk.load_panel(self.strategic, source, candidate)
        from .costs import fee_arrays

        buy, _, _ = fee_arrays(candidate)
        names = [a["id"] for a in source["baseline"]["assets"]]
        if any(p.asset_class_id not in names for p in candidate.products):
            raise ValidationError("IMPLEMENTATION_PRODUCT_CLASS", "产品引用未知大类。")
        weights, status, iterations, diagnostics = (
            risk_kernels.implementation_qp_kernel(
                products,
                factors,
                np.asarray([source["target"][x] for x in names], dtype=np.float64),
                np.asarray(
                    [names.index(p.asset_class_id) for p in candidate.products],
                    dtype=np.int64,
                ),
                np.asarray(
                    [p.max_weight for p in candidate.products], dtype=np.float64
                ),
                buy,
            )
        )
        if status:
            raise ValidationError(
                "IMPLEMENTATION_QP_FAILED",
                (
                    "类内产品容量不足。"
                    if status == 4
                    else "此候选求解未收敛，不能据此认定所有产品方案无解。"
                ),
            )
        raw = candidate.model_dump(mode="json")
        for item, value in zip(raw["products"], weights, strict=True):
            item["weight"] = float(value)
        updated = ImplementationCandidate.model_validate(raw)
        return {
            "candidate": updated.model_dump(mode="json"),
            "iterations": iterations,
            "diagnostics": diagnostics.tolist(),
            "objective": "historical_tracking_variance_plus_declared_buy_unit_cost",
            "objective_units": {
                "risk": "annualized_decimal_return_variance",
                "cost": "one_time_buy_fee_fraction",
                "cost_penalty": 1.0,
                "penalty_unit": "annual_return_variance_per_unit_cost_fraction",
                "scope": "entry_cost_proxy_not_current_holdings_transition_cost_optimum",
            },
            "data_hash": metadata["content_hash"],
            "validation": evaluate(self, updated, self.candidate_identity(updated)[0])[
                0
            ],
            "limitation": "历史 TE 候选仍按同一产品风险和单／A／B 门禁验收，不是多模型 QCQP 全局最优证明。",
        }

    def save(self, body, package_id=None):
        copied_from_id = body.copied_from_id
        if package_id:
            # Copy lineage is fixed at creation, independent of later UI state.
            copied_from_id = self.repository.current(package_id).get("copied_from_id")
        elif copied_from_id is not None:
            source = self.repository.artifacts.get(copied_from_id, "series")
            if source.get("artifact_type") != "implementation_package":
                raise ValidationError(
                    "PACKAGE_COPY_SOURCE_TYPE", "复制来源必须是已保存的研究包版本。"
                )
        candidate_hash, refs = self.candidate_identity(body.candidate)
        return self.repository.append(
            package_id,
            body.expected_revision,
            {
                "name": body.candidate.name,
                "candidate": body.candidate.model_dump(mode="json"),
                "candidate_hash": candidate_hash,
                "dependencies": refs,
                "stage": "draft",
                "copied_from_id": copied_from_id,
                "report_id": None,
            },
            key=body.idempotency_key,
        )

    def _checked(self, package_id, action):
        item = self.repository.current(package_id)
        if (
            item["revision"] != action.expected_revision
            or item["candidate_hash"] != action.candidate_hash
        ):
            raise ConflictError(
                "PACKAGE_STALE", "候选或研究包已更新，请刷新后重新验证。"
            )
        if item["stage"] == "finalized":
            raise ConflictError("PACKAGE_FINALIZED", "此研究包已定稿，请复制重研。")
        return item

    @staticmethod
    def _fields(item):
        return {
            key: item[key]
            for key in (
                "name",
                "candidate",
                "candidate_hash",
                "dependencies",
                "copied_from_id",
                "report_id",
            )
            if key in item
        }

    def validate(self, package_id, action):
        operation_hash = digest_json(
            {"package_id": package_id, "action": action.model_dump(mode="json")}
        )
        store = self.repository.artifacts
        saved = store.idempotent_result(
            operation_key(action.idempotency_key, "report"), operation_hash
        )
        if saved:
            current = self.repository.current(package_id)
            if current.get("report_id") == saved["id"]:
                return current
        versions = self.repository.history(package_id)
        original = next(
            (x for x in versions if x["revision"] == action.expected_revision), None
        )
        if (
            original is None
            or original["candidate_hash"] != action.candidate_hash
            or original["stage"] == "finalized"
        ):
            raise ConflictError(
                "PACKAGE_STALE", "候选或研究包已更新，请刷新后重新验证。"
            )
        with compute_slot():
            item = self.repository.append(
                package_id,
                original["revision"],
                {**self._fields(original), "stage": "candidate_frozen", "report_id": None},
                key=operation_key(action.idempotency_key, "freeze"),
            )
            if self.repository.current(package_id)["id"] != item["id"]:
                raise ConflictError(
                    "PACKAGE_STALE", "候选锁定后已被新版本替代，本报告不能放行当前方案。"
                )
            try:
                candidate = ImplementationCandidate.model_validate(item["candidate"])
                # Persist an explicit attempt before computation; failures remain visible.
                attempt = self.register_attempt_raw(
                    {
                        "logical_attempt_id": action.idempotency_key,
                        "hypothesis_family": "implementation_validation",
                        "candidate_hash": item["candidate_hash"],
                        "status": "started",
                        "reason": "用户明确登记候选验证运行",
                        "package_id": package_id,
                    },
                    operation_key(action.idempotency_key, "start"),
                )
                if saved:
                    report = saved
                    result = saved
                else:
                    result, arrays = evaluate(
                        self, candidate, item["candidate_hash"], validation=True
                    )
                    result.update(
                        artifact_type="implementation_validation",
                        package_id=package_id,
                        attempt_id=attempt["id"],
                        candidate=item["candidate"],
                        manifest_scope="exact_inputs_arrays_source_fingerprints",
                    )
                    report = store.save(
                        "run",
                        result,
                        arrays,
                        idempotency_key=operation_key(action.idempotency_key, "report"),
                        request_hash=operation_hash,
                    )
                self.register_attempt_raw(
                    {
                        "logical_attempt_id": action.idempotency_key,
                        "hypothesis_family": "implementation_validation",
                        "candidate_hash": item["candidate_hash"],
                        "status": "succeeded" if result["research_ready"] else "failed",
                        "reason": "验证报告已冻结",
                        "package_id": package_id,
                        "report_id": report["id"],
                    },
                    operation_key(action.idempotency_key, "finish"),
                )
                return self.repository.append(
                    package_id,
                    item["revision"],
                    {
                        **self._fields(item),
                        "stage": "validation_complete",
                        "report_id": report["id"],
                        "validation_report_hash": report["content_hash"],
                        "candidate_frozen": True,
                    },
                    key=operation_key(action.idempotency_key, "package"),
                )
            except Exception:
                self.register_attempt_raw(
                    {
                        "logical_attempt_id": action.idempotency_key,
                        "hypothesis_family": "implementation_validation",
                        "candidate_hash": item["candidate_hash"],
                        "status": "failed",
                        "reason": "计算或保存失败，原始候选仍保留",
                        "package_id": package_id,
                    },
                    operation_key(action.idempotency_key, "failure"),
                )
                raise

    def current_eligibility(self, item, report=None):
        reasons = []
        candidate = ImplementationCandidate.model_validate(item["candidate"])
        try:
            source = self.sources.resolve(candidate.source, candidate.as_of)
            if report:
                self.repository.require_report(item["candidate_hash"], report)
                current_execution = execution_manifest()
                if (
                    report["execution"]["source_hash"]
                    != current_execution["source_hash"]
                ):
                    reasons.append("计算代码已变化，请重新验证新版本。")
                if report["execution"].get("runtime") != current_execution["runtime"]:
                    reasons.append("计算运行环境已变化，请重新验证新版本。")
                if report.get("data"):
                    _, _, current = risk.load_panel(self.strategic, source, candidate)
                    if current["content_hash"] != report["data"]["content_hash"]:
                        reasons.append("历史输入快照已变化，请复制重研。")
                from .costs import fee_arrays

                if report.get("transition"):
                    fee_arrays(candidate)
                if self.scenarios and candidate.scenario_release_ids:
                    self.scenarios.current(candidate)
            if (
                item.get("review")
                and str(date.today()) >= item["review"]["review_due_at"]
            ):
                reasons.append("研究包已到复核日。")
        except (IndicatorDomainError, ValueError) as exc:
            reasons.append(str(exc))
        return {"status": "needs_review" if reasons else "current", "reasons": reasons}

    def finalize(self, package_id, action):
        # Shared with CMA/mandate lifecycle writes; package CAS covers concurrent edits.
        request_hash = digest_json(
            {"package_id": package_id, "finalize": action.model_dump(mode="json")}
        )
        prior = self.repository.current(package_id)
        if prior.get("finalization_request_hash") == request_hash:
            return prior
        with self.strategic.artifacts.governance_lock.locked():
            item = self._checked(package_id, action)
            if item["stage"] != "validation_complete" or not item.get("report_id"):
                raise ValidationError("PACKAGE_REPORT_REQUIRED", "请先完成候选验证。")
            report = self.repository.report(item["report_id"])
            self.repository.require_report(item["candidate_hash"], report)
            if (
                report["content_hash"] != action.validation_report_hash
                or report.get("validation_mode") != "frozen_candidate_validation"
            ):
                raise ConflictError(
                    "PACKAGE_REPORT_HASH", "报告指纹不匹配，或尚未完成锁定候选验证。"
                )
            if not report["research_ready"]:
                raise ValidationError(
                    "PACKAGE_HARD_BLOCKERS",
                    "硬约束失败或完整性缺失，不能通过备注放行。",
                )
            eligibility = self.current_eligibility(item, report)
            if eligibility["status"] != "current":
                raise ValidationError(
                    "PACKAGE_NEEDS_REVIEW", "；".join(eligibility["reasons"])
                )
            if action.review_due_at <= date.today():
                raise ValidationError("PACKAGE_REVIEW_DATE", "下次复核日期须晚于今天。")
            version = self.repository.append(
                package_id,
                item["revision"],
                {
                    **self._fields(item),
                    "stage": "finalized",
                    "validation_report_hash": report["content_hash"],
                    "finalization_request_hash": request_hash,
                    "research_scope": report["scope"],
                    "implementation_eligibility": report["implementation_eligibility"],
                    "historical_pit_eligibility": False,
                    "review": {
                        "reviewer": action.reviewer,
                        "reason": action.reason,
                        "review_due_at": str(action.review_due_at),
                        "candidate_hash": item["candidate_hash"],
                        "report_hash": report["content_hash"],
                        "independent_review": False,
                        "identity_basis": "self_declared_local_researcher",
                    },
                },
                key=operation_key(action.idempotency_key, "finalize"),
            )
            return version

    def view(self, package_id):
        item = self.repository.current(package_id)
        report = (
            self.repository.report(item["report_id"]) if item.get("report_id") else None
        )
        attempts = [
            self.repository.artifacts.get(row["id"], "run")
            for row in self.repository.artifacts.list("run")
            if row.get("artifact_type") == "implementation_attempt"
            and row.get("scheme_id") == package_id
        ]
        return {
            "package": item,
            "report": report,
            "history": self.repository.history(package_id),
            "attempts": attempts,
            "current_eligibility": self.current_eligibility(item, report),
        }

    def register_attempt_raw(self, fields, key):
        return self.repository.artifacts.save(
            "run",
            {
                "artifact_type": "implementation_attempt",
                **fields,
                "scheme_id": fields.get("package_id"),
            },
            idempotency_key=key,
            request_hash=digest_json(fields),
        )

    def register_attempt(self, body):
        fields = body.model_dump(mode="json", exclude={"idempotency_key"})
        return self.register_attempt_raw(fields, body.idempotency_key)

    def export(self, package_id):
        view = self.view(package_id)
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "research-package.json",
                json.dumps(view, ensure_ascii=False, indent=2, allow_nan=False),
            )
            if view["report"]:
                identifier = view["report"]["id"]
                arrays = self.repository.artifacts.arrays(identifier)
                for name, value in arrays.items():
                    buffer = io.BytesIO()
                    np.save(buffer, value, allow_pickle=False)
                    archive.writestr("arrays/" + name + ".npy", buffer.getvalue())
            archive.writestr(
                "README.txt",
                "冻结研究输入、结果及数组。当前应用资格单独记录；代码指纹含未提交工作树。\n不是交易指令、PIT 认证或机构独立审批。\n",
            )
        return content.getvalue()
