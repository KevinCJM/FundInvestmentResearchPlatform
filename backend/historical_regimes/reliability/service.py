"""Bounded preview and atomic immutable report persistence."""

import copy
import re
import threading
import time
from datetime import datetime, timezone
from custom_indicators.repository import AtomicJsonStore
from custom_indicators.errors import ConflictError, NotFoundError, ValidationError
from ..v2_service import _content_hash, _json_safe
from .contracts import PreviewRequest, ConfirmRequest
from .references import resolve_reference, reference_catalog
from .execution import load_definition, replay
from .report import build_report
from . import kernels


class ReliabilityService:
    request_model = PreviewRequest
    confirm_model = ConfirmRequest
    report_prefix = "reliability"
    storage_name = "historical_regime_reliability"

    def __init__(self, graph):
        self.graph = graph
        self.root = graph.workspace_data_dir / self.storage_name
        self.catalog_store = AtomicJsonStore(self.root / "catalog.json")
        self._admission = threading.Lock()
        self._previews = {}
        self.runtime = kernels.warm()

    def references(self):
        return reference_catalog(self.graph)

    def _compute(self, request):
        kernels.audit()
        definition = load_definition(self.graph, request)
        raw, _ = resolve_reference(self.graph, request.reference, hydrate=False)
        if int(raw.get("series_summary", {}).get("row_count", 20001)) > 20000:
            raise ValidationError("RELIABILITY_BUDGET", "参考最多20000条观测。")
        reference, publication = resolve_reference(self.graph, request.reference)
        started = time.monotonic()
        cache = {}
        predictions, lineage, cutoff = replay(
            self.graph, definition, request, reference, source_cache=cache
        )
        canonical = request.model_dump(mode="json", exclude={"compile_token"})
        canonical["policy"]["test_end"] = cutoff
        report = build_report(
            definition,
            reference,
            publication,
            predictions,
            lineage,
            request.policy,
            cutoff,
        )
        from .diagnostics import run_diagnostics

        report["stability"]["parameter_sensitivity"] = run_diagnostics(
            self.graph,
            definition,
            request.policy.stability,
            predictions,
            "realtime",
            cutoff,
            cache=cache,
            request=request,
            reference=reference,
            folds=lineage["folds"],
            started=started,
        )
        if time.monotonic() - started > 120:
            raise ConflictError("RELIABILITY_TIME_BUDGET", "验证超过时间预算。")
        result = _json_safe({"request": canonical, "report": report})
        return {"preview_hash": _content_hash(result), **result}

    def preview(self, payload):
        request = self.request_model.model_validate(payload)
        if not self._admission.acquire(blocking=False):
            raise ConflictError("RELIABILITY_BUSY", "已有验证正在运行，请稍后重试。")
        try:
            result = self._compute(request)
            now = time.monotonic()
            self._previews = {
                h: p for h, p in self._previews.items() if p["expires"] > now
            }
            while len(self._previews) >= 8:
                self._previews.pop(next(iter(self._previews)))
            self._previews[result["preview_hash"]] = {
                "expires": now + 900,
                "value": copy.deepcopy(result),
            }
            return result
        finally:
            self._admission.release()

    def _store(self, report_id):
        if not re.fullmatch(self.report_prefix + r"-[a-f0-9]{64}", report_id):
            raise NotFoundError("RELIABILITY_REPORT_NOT_FOUND", "报告ID无效。")
        return AtomicJsonStore(self.root / (report_id + ".json"))

    def get(self, report_id):
        store = self._store(report_id)
        with store.locked():
            items = store.read_unlocked()["items"]
        if not items:
            raise NotFoundError("RELIABILITY_REPORT_NOT_FOUND", "报告不存在。")
        item = items[0]
        if (
            len(items) != 1
            or item.get("id") != report_id
            or item.get("content_hash")
            != _content_hash({k: v for k, v in item.items() if k != "content_hash"})
        ):
            raise ConflictError("RELIABILITY_REPORT_INTEGRITY", "报告完整性校验失败。")
        return copy.deepcopy(item)

    def confirm(self, payload):
        request = self.confirm_model.model_validate(payload)
        if not self._admission.acquire(blocking=False):
            raise ConflictError("RELIABILITY_BUSY", "已有验证正在运行，请稍后重试。")
        try:
            report_id = self.report_prefix + "-" + request.preview_hash
            cached = self._previews.get(request.preview_hash)
            if not cached or cached["expires"] <= time.monotonic():
                raise ConflictError(
                    "RELIABILITY_PREVIEW_EXPIRED", "预览已过期，请重新验证。"
                )
            canonical = request.request.model_dump(
                mode="json", exclude={"compile_token"}
            )
            if canonical != cached["value"]["request"]:
                raise ConflictError(
                    "RELIABILITY_PREVIEW_MISMATCH", "确认请求与预览不一致。"
                )
            # Re-resolve immutable inputs and actual source bytes via sole executor.
            result = self._compute(request.request)
            if result["preview_hash"] != request.preview_hash:
                raise ConflictError(
                    "RELIABILITY_PROVENANCE_CHANGED",
                    "模型、参考或数据血缘已变化，请重新预览。",
                )
            store = self._store(report_id)
            with store.locked():
                stored = store.read_unlocked()
                if not stored["items"]:
                    item = {
                        "id": report_id,
                        **(
                            {"calibration_id": report_id}
                            if self.report_prefix == "reliability"
                            else {}
                        ),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "immutable": True,
                        **result,
                    }
                    item["content_hash"] = _content_hash(item)
                    store.write_unlocked({"schema_version": 1, "items": [item]})
            item = self.get(report_id)
            summary = self._summary(item)
            with self.catalog_store.locked():
                catalog = self.catalog_store.read_unlocked()
                if not any(p["id"] == report_id for p in catalog["items"]):
                    catalog["items"].append(summary)
                    self.catalog_store.write_unlocked(catalog)
            return item
        finally:
            self._admission.release()

    def catalog(self):
        with self.catalog_store.locked():
            catalog = self.catalog_store.read_unlocked()
        items = []
        for summary in catalog["items"]:
            item = self.get(summary["id"])
            items.append(self._summary(item))
        return {"items": items}

    def recognition_evidence(self, report_id):
        """Stable read-only state-recognition evidence; never an LTCMA authority."""
        item = self.get(report_id)
        report = item["report"]
        verification = report.get("verification")
        if not isinstance(verification, dict):
            raise ConflictError(
                "RECOGNITION_EVIDENCE_UNAVAILABLE",
                "该历史报告早于状态级验证协议，请重新运行识别有效性验证。",
            )
        ready = verification.get("recognition_ready")
        if ready is None:
            # Read-only compatibility for immutable reports created before the
            # Realtime -> LTCMA relationship was explicitly removed.
            ready = verification.get("cma_research_ready", False)
        labels = {state["id"]: state.get("label", state["id"]) for state in report["states"]}
        return {
            "schema_version": "1.0",
            "kind": "regime_recognition_evidence",
            "report_id": item["id"],
            "calibration_id": item.get("calibration_id"),
            "model": {
                "definition_id": item["request"]["definition_id"],
                "revision": item["request"]["revision"],
                "model_binding_hash": report.get("lineage", {}).get("model_binding_hash"),
            },
            "reference": copy.deepcopy(item["request"]["reference"]),
            "evaluation_scope": verification["scope"],
            "status": verification["status"],
            "research_ready": bool(ready),
            "production_eligible": bool(report.get("calibration", {}).get("deployment_eligible")),
            "confidence_floor": report.get("calibration", {}).get("confidence_floor"),
            "verified_states": list(verification["verified_states"]),
            "unverified_states": list(verification.get("fallback_states", [])),
            "unverified_state_policy": verification.get(
                "unverified_state_policy", "do_not_authorize_unverified_states"
            ),
            "states": [
                {**copy.deepcopy(row), "label": labels.get(row["state_id"], row["state_id"])}
                for row in verification["states"]
            ],
            "probability_improves_class_base": bool(verification["probability_improves_class_base"]),
            "limitations": [
                "Historical reference agreement is not latent market truth.",
                "Unverified states cannot authorize downstream realtime decisions.",
                "Retrospective recognition evidence is not prospective deployment qualification.",
                "Realtime recognition evidence is not an LTCMA input; LTCMA consumes historical-reference evidence instead.",
            ],
        }

    def cma_evidence(self, report_id):
        raise ConflictError(
            "REALTIME_CMA_EVIDENCE_REMOVED",
            "实时状态识别不再作为 LTCMA 输入；请使用 recognition-evidence 做识别验证，LTCMA 读取历史状态质量证据。",
        )

    @staticmethod
    def _verification_summary(verification):
        ready = verification.get("recognition_ready")
        if ready is None:
            ready = verification.get("cma_research_ready", False)
        return {
            "status": verification.get("status"),
            "recognition_ready": bool(ready),
            "verified_states": copy.deepcopy(verification.get("verified_states", [])),
            "unverified_states": copy.deepcopy(verification.get("fallback_states", [])),
        }

    def _summary(self, item):
        request = item["request"]
        return {
            "id": item["id"],
            "calibration_id": item["calibration_id"],
            "created_at": item["created_at"],
            "definition_id": request["definition_id"],
            "revision": request["revision"],
            "reference": request["reference"],
            "status": item["report"]["status"],
            "calibration": item["report"]["calibration"],
            "verification": (self._verification_summary(item["report"]["verification"])
                if isinstance(item["report"].get("verification"), dict) else None),
        }
