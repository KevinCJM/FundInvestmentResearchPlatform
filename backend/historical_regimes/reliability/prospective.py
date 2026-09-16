"""Explicit forward journal. Historical reports never acquire deployment flags.

Only this service writes signed append-only records. Date-only captures use the
previous UTC day; an injected aware clock is a test/deployment dependency, never
an HTTP input. See the dated API document for the local-store trust boundary.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import re
import secrets
import time
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Literal
from types import CodeType
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from custom_indicators.errors import ConflictError, NotFoundError, ValidationError, IndicatorDomainError
from custom_indicators.repository import AtomicJsonStore
from ..v2_contracts import definition_content_hash, parse_definition_v2
from ..v2_service import _content_hash, _json_safe, _required_node_ids, _definition_output_frequency
from ..temporal_audit import audit_execution
from .contracts import Reference, PreviewRequest
from .execution import LATENT, load_definition, model_binding_hash, target_identity, probability_provenance
from .references import resolve_reference
from .report import axis, classification, probability
from . import kernels as nk
from . import prospective_kernels as pk
from . import source_versions as sv


def _engine_hash(graph):
    """Bind loaded executor/Numba code, rather than trusting a mutable version label."""
    from .. import v2_numba, numba_kernels, segment_numba, condition_numba, trend_numba, granular_runtime
    functions = {name: getattr(type(graph), name) for name in
                 ("_execute_graph", "_execute_numeric_node", "_execute_latent_model")}
    functions["granular.execute"] = granular_runtime.execute_granular_node
    functions.update({"forward." + name: getattr(ProspectiveService, name)
                      for name in ("_execute", "capture", "_score", "verify_qualification")})
    functions.update({"source_version." + name: value for name, value in vars(sv).items()
                      if callable(value) and getattr(value, "__module__", None) == sv.__name__
                      and hasattr(value, "__code__")})
    for module in (v2_numba, numba_kernels, segment_numba, condition_numba, trend_numba, nk, pk):
        for name, value in vars(module).items():
            if hasattr(value, "py_func"):
                functions[module.__name__ + "." + name] = value.py_func
    def constant(value):
        if isinstance(value, CodeType):
            return code_record(value)
        if isinstance(value, (tuple, frozenset)):
            items = [constant(v) for v in value]
            return {type(value).__name__: sorted(items, key=_content_hash) if isinstance(value, frozenset) else items}
        return {type(value).__name__: repr(value)}

    def code_record(code):
        # marshal's reference flags depend on live object interning/refcounts.
        # Normalize immutable code fields instead of hashing marshal bytes.
        return {"bytecode": code.co_code.hex(), "constants": [constant(v) for v in code.co_consts],
                "names": code.co_names, "locals": code.co_varnames, "free": code.co_freevars,
                "cells": code.co_cellvars, "flags": code.co_flags,
                "args": [code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount],
                "exceptions": code.co_exceptiontable.hex()}

    return _content_hash({name: {"code": code_record(fn.__code__),
                                "defaults": repr(fn.__defaults__), "kwdefaults": repr(fn.__kwdefaults__)}
                          for name, fn in functions.items()})


class ForwardPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    observation_window: int = Field(default=252, ge=60, le=2000)
    minimum_observations: int = Field(default=120, ge=60, le=2000)
    minimum_class_observations: int = Field(default=10, ge=5, le=1000)
    minimum_class_complete_regimes: int = Field(default=2, ge=1, le=100)
    block_size: int = Field(default=20, ge=5, le=252)
    minimum_complete_blocks: int = Field(default=5, ge=3, le=100)
    minimum_coverage: float = Field(default=.95, ge=.8, le=1.)
    minimum_agreement: float = Field(default=.6, ge=0., le=1.)
    minimum_state_precision: float = Field(default=.65, ge=0., le=1.)
    minimum_brier_improvement: float = Field(default=.01, gt=0., le=2.)
    max_capture_lag_days: int = Field(default=3, ge=1, le=7)
    max_observation_gap_days: int = Field(default=7, ge=1, le=62)
    expires_after_days: int = Field(default=30, ge=1, le=730)
    maximum_protocol_days: int = Field(default=730, ge=90, le=3650)
    criterion: Literal["all_complete_blocks_improve_no_significance_claim"] = "all_complete_blocks_improve_no_significance_claim"

    @model_validator(mode="after")
    def feasible(self):
        if (self.minimum_observations > self.observation_window
                or self.minimum_complete_blocks * self.block_size > self.observation_window):
            raise ValueError("Forward sample/block gates exceed frozen window")
        return self


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    calibration_id: str = Field(pattern=r"^reliability-[a-f0-9]{64}$")
    policy: ForwardPolicy = Field(default_factory=ForwardPolicy)


def default_forward_policy(frequency):
    if frequency == "monthly":
        return ForwardPolicy(observation_window=60, minimum_observations=60,
                             block_size=5, max_observation_gap_days=45, max_capture_lag_days=7,
                             expires_after_days=365, maximum_protocol_days=3650)
    if frequency == "weekly":
        return ForwardPolicy(observation_window=120, minimum_observations=120,
                             block_size=10, max_observation_gap_days=14, max_capture_lag_days=7,
                             expires_after_days=90, maximum_protocol_days=1460)
    return ForwardPolicy()


class AssessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference: Reference


class CaptureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _utc(value: str | datetime) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValidationError("PROSPECTIVE_TIME", "需要带时区的服务端时间。")
    return result.astimezone(timezone.utc)


def _fail(code, message):
    raise ConflictError("PROSPECTIVE_" + code, message)


def _definition_identity(run):
    definition = parse_definition_v2(run["definition"])
    study = definition.study
    if study is None or study.purpose != "historical_reference":
        _fail("REFERENCE_PURPOSE", "需要历史参考 study 定义。")
    return {"definition_id": run["definition_id"], "revision": run["definition_revision"],
            "definition_hash": definition_content_hash(definition), "family": study.family,
            "states": run["states"], "frequency": run["frequency"],
            "target": target_identity(definition)}


def _state_qualification(states, support, cycles, y, accepted, policy):
    state_counts, state_values = nk.state_evidence_kernel(y, accepted, len(states))
    rows = []
    for state_index, state_id in enumerate(states):
        reasons = []
        paired_reference = int(support[0, state_index])
        accepted_predictions = int(state_counts[state_index, 1])
        complete_regimes = int(cycles[state_index])
        precision = float(state_values[state_index, 0]) if np.isfinite(state_values[state_index, 0]) else None
        recall = float(state_values[state_index, 1]) if np.isfinite(state_values[state_index, 1]) else None
        if paired_reference < policy["minimum_class_observations"]:
            reasons.append("insufficient_paired_state_observations")
        if accepted_predictions < policy["minimum_class_observations"]:
            reasons.append("insufficient_accepted_state_predictions")
        if complete_regimes < policy["minimum_class_complete_regimes"]:
            reasons.append("insufficient_complete_state_regimes")
        if reasons:
            status = "insufficient_evidence"
        elif precision is not None and precision >= policy["minimum_state_precision"]:
            status = "qualified"
        else:
            status = "failed"
            reasons.append("state_precision_below_policy")
        rows.append({"state_id": state_id, "status": status,
                     "paired_reference_observations": paired_reference,
                     "accepted_predictions": accepted_predictions,
                     "complete_regimes": complete_regimes, "precision": precision,
                     "recall": recall, "reasons": reasons})
    qualified = [row["state_id"] for row in rows if row["status"] == "qualified"]
    fallback = [row["state_id"] for row in rows if row["status"] != "qualified"]
    return rows, qualified, fallback


class ProspectiveService:
    def __init__(self, graph, reliability, *, clock: Callable[[], datetime] | None = None):
        self.graph, self.reliability = graph, reliability
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.root = reliability.root / "prospective"
        self.journal = AtomicJsonStore(self.root / "journal.json")
        self.keys = AtomicJsonStore(self.root / "signing-key.json")
        self._ready_pid = None
        self._source_previews = {}

    def warm(self) -> dict:
        """Call in each worker lifespan, never from a capture/request fallback."""
        import os
        self._ready_pid = None
        nk.audit()
        result = pk.warm()
        prepared, blocked = 0, []
        if self.journal.path.exists():
            with self.journal.locked():
                data = self._read()
                for item in data["items"]:
                    if item["kind"] == "protocol" and sv.versions(data, item["id"]):
                        try:
                            protocol, definition = self._protocol(data, item["id"])
                        except IndicatorDomainError as exc:
                            # Invalidated research is not a numerical-readiness
                            # failure. Its capture/qualification gates still reject it.
                            blocked.append({"protocol_id": item["id"], "code": exc.code})
                            continue
                        current, _, _ = sv.context(data, protocol, definition)
                        self.graph.prepare(current.model_dump(mode="json"))
                        prepared += 1
        self._ready_pid = os.getpid()
        self.prewarm_status = {**result, "source_versions": {"prepared": prepared, "blocked": blocked}}
        return self.prewarm_status

    def _audit(self):
        import os
        if self._ready_pid != os.getpid():
            raise RuntimeError("PROSPECTIVE_SERVICE_NOT_READY")
        nk.audit()
        return pk.audit()

    def _now(self):
        return _utc(self.clock())

    def _key(self, create=False):
        with self.keys.locked():
            data = self.keys.read_unlocked()
            if not data["items"] and create:
                data["items"] = [secrets.token_hex(32)]
                self.keys.write_unlocked(data)
                self.keys.path.chmod(0o600)
            if len(data["items"]) != 1:
                _fail("JOURNAL_KEY", "前向日志密钥缺失。")
            return bytes.fromhex(data["items"][0])

    def _read(self):
        data = self.journal.read_unlocked()
        if not data["items"]:
            return data
        key, previous, ids = self._key(), None, set()
        last_time = None
        for item in data["items"]:
            body = {k: v for k, v in item.items() if k not in {"content_hash", "signature"}}
            digest = _content_hash(body)
            stamp = _utc(item["recorded_at"])
            if (item.get("previous_hash") != previous or item.get("content_hash") != digest
                    or not hmac.compare_digest(item.get("signature", ""), hmac.new(key, digest.encode(), hashlib.sha256).hexdigest())
                    or item.get("id") in ids or (last_time and stamp < last_time)):
                _fail("JOURNAL_INTEGRITY", "前向日志签名、顺序或内容不一致。")
            previous, last_time = digest, stamp
            ids.add(item["id"])
        return data

    def _append(self, data, kind, payload, now):
        if len(data["items"]) >= 20000:
            _fail("JOURNAL_BUDGET", "前向日志达到容量上限，请归档工作区。")
        if data["items"] and now < _utc(data["items"][-1]["recorded_at"]):
            _fail("CLOCK_REVERSED", "服务端时钟回退，禁止写入。")
        item = _json_safe({"id": "forward-" + uuid4().hex, "kind": kind,
                          "recorded_at": now.isoformat(), "immutable": True,
                          "previous_hash": data["items"][-1]["content_hash"] if data["items"] else None,
                          **payload})
        item["content_hash"] = _content_hash(item)
        item["signature"] = hmac.new(self._key(create=not data["items"]), item["content_hash"].encode(), hashlib.sha256).hexdigest()
        data["items"].append(item)
        self.journal.write_unlocked(data)
        return copy.deepcopy(item)

    @staticmethod
    def _find(data, item_id, kind):
        if not re.fullmatch(r"forward-[a-f0-9]{32}", item_id):
            raise NotFoundError("PROSPECTIVE_NOT_FOUND", "前向记录不存在。")
        for item in data["items"]:
            if item["id"] == item_id and item["kind"] == kind:
                return item
        raise NotFoundError("PROSPECTIVE_NOT_FOUND", "前向记录不存在。")

    def _candidate(self, calibration_id):
        artifact = self.reliability.get(calibration_id)
        report = artifact["report"]
        digest = _content_hash({"request": artifact["request"], "report": report})
        if artifact["id"] != "reliability-" + digest or artifact.get("preview_hash") != digest:
            _fail("CALIBRATION_INTEGRITY", "校准制品不匹配原始预览。")
        request = PreviewRequest.model_validate(artifact["request"])
        definition = load_definition(self.graph, request)
        reference, publication = self._reference(request.reference)
        identity = _definition_identity(reference)
        mapping = definition.study.state_mapping or {s.id: s.id for s in definition.states}
        calibration = report["calibration"]
        if calibration.get("fitted") is not True:
            _fail("UNFITTED", "尚未拟合的校准器不能注册。")
        if (report["lineage"]["definition_hash"] != definition_content_hash(definition)
                or report["lineage"]["model_binding_hash"] != model_binding_hash(definition)
                or report["lineage"]["state_mapping"] != mapping
                or report["states"] != reference["states"]
                or target_identity(definition) != identity["target"]
                or definition.study.family != identity["family"]
                or _definition_output_frequency(definition) not in {"daily", "weekly", "monthly"}
                or identity["frequency"] != _definition_output_frequency(definition)):
            _fail("BINDING", "需要同定义、同对象、同频率和冻结状态映射；支持日、周、月频。")
        states = [s["id"] for s in report["states"]]
        if set(mapping) != {s.id for s in definition.states} or not set(mapping.values()).issubset(states):
            _fail("MAPPING", "冻结状态轴不合法。")
        method = calibration["method"]
        if method not in {"class_frequency", "temperature"} or (method == "temperature" and not probability_provenance(definition)["temperature_supported"]):
            _fail("CALIBRATION_MODE", "不支持该冻结概率来源。")
        params = calibration["parameters"]
        base = np.asarray(params["class_base"], np.float64)
        counts = np.asarray(params["counts"], np.float64)
        if (base.shape != (len(states),) or counts.shape != (len(states), len(states))
                or not np.isfinite(base).all() or not np.isfinite(counts).all()
                or (counts < 0).any() or (base < 0).any() or abs(float(base.sum()) - 1.) > 1e-8
                or (method == "temperature" and (params["temperature"] is None or not 0 < params["temperature"] < float("inf")))):
            _fail("PARAMETERS", "冻结校准参数无效。")
        return artifact, definition, reference, publication, identity

    def _reference(self, ref):
        raw, _ = resolve_reference(self.graph, ref, hydrate=False)
        if int(raw.get("series_summary", {}).get("row_count", 20001)) > 20000:
            _fail("INPUT_BUDGET", "参考最多20000条。")
        result, publication = resolve_reference(self.graph, ref)
        axis(result["series"])
        return result, publication

    def _protocol(self, data, protocol_id):
        protocol = self._find(data, protocol_id, "protocol")
        artifact, definition, reference, publication, identity = self._candidate(protocol["calibration_id"])
        if (artifact["content_hash"] != protocol["artifact_hash"]
                or _engine_hash(self.graph) != protocol["engine_hash"]
                or identity != protocol["reference_definition"]
                or definition_content_hash(definition) != protocol["definition_hash"]
                or _content_hash(artifact["report"]["calibration"]) != protocol["calibrator_hash"]
                or publication != protocol["original_publication"]):
            _fail("LINEAGE", "注册后的模型、参考或校准血缘已变化。")
        return protocol, definition

    @staticmethod
    def _prefix(port, count):
        if count > len(port.dates):
            _fail("SOURCE_REVISION", "冻结输入前缀已缺失。")
        # Hashing is serialization, not numerical calculation. Slices are views.
        payload = {}
        for name in ("dates", "available", "values"):
            array = getattr(port, name)[:count]
            payload[name] = {"dtype": str(array.dtype), "shape": array.shape,
                             "bytes": hashlib.sha256(array.tobytes(order="C")).hexdigest()}
        return _content_hash(payload)

    def _execute(self, definition, cutoff, expected=None):
        started = time.monotonic()
        required = _required_node_ids(definition)
        nodes = [n for n in definition.graph.nodes if n.id in required]
        sources = [n for n in nodes if n.type.startswith("source.") and n.type != "source.constant"]
        latent = [n for n in nodes if n.type in LATENT]
        if (len(nodes) > 32 or not 1 <= len(sources) <= 2 or len(latent) > 1
                or any(int(n.parameters.get("iterations", 60)) > 100 for n in latent)
                or any(n.type.startswith("model.") and n.type not in LATENT | {"model.threshold", "model.range_threshold", "model.drawdown_cycle_realtime"} for n in nodes)
                or any(n.type not in {"source.index", "source.upload"} for n in sources)):
            _fail("UNSUPPORTED_GRAPH", "前向捕获支持最多32节点、2个指数/冻结上传源及1个标准拟合模型。")
        if not sv.same_source_targets(definition):
            _fail("UNSUPPORTED_AXIS", "前向捕获的评价对象必须与唯一输入源一致，不能混入其他对象。")
        if _definition_output_frequency(definition) not in {"daily", "weekly", "monthly"}:
            _fail("UNSUPPORTED_AXIS", "前向捕获只支持日、周、月频。")
        token = self.graph._plans_by_graph_hash.get(self.graph._preparation_hash(definition))
        plan = self.graph._validate_plan(definition, token)
        cache = {}
        bundles, _ = self.graph._resolve_sources(definition, required, "realtime", cutoff, cache)
        if any(len(b.frame) > 20000 for b in bundles.values()) or any(len(b.frame) > 20000 for b in cache.values()):
            _fail("INPUT_BUDGET", "每个输入最多20000条。")
        result = self.graph._execute_graph(None, definition, "realtime", cutoff, plan=plan, source_cache=cache)
        axis(result["series"])
        if len(result["series"]) > 20000:
            _fail("INPUT_BUDGET", "输出最多20000条。")
        prefixes = {}
        for node in sources:
            port = result["node_outputs"][node.id]["value"]
            if expected:
                old = expected[node.id]
                if self._prefix(port, old["count"]) != old["hash"]:
                    _fail("SOURCE_REVISION", "历史输入发生修订；该协议不得继续累积证据。")
            prefixes[node.id] = {"count": len(port.dates), "hash": self._prefix(port, len(port.dates))}
        temporal = audit_execution(self.graph, definition, "realtime", cutoff, plan, result, max_seconds=15.)
        temporal.pop("elapsed_ms", None)
        if (temporal.get("verified") is not True or not temporal.get("realtime_supported")
                or temporal.get("may_repaint") or temporal.get("semantic_hindsight")):
            _fail("PIT", "当前执行未通过既有因果与时点门禁。")
        if time.monotonic() - started > 45:
            _fail("TIME_BUDGET", "前向捕获超过时间预算。")
        return result, prefixes, temporal

    def register(self, payload: RegisterRequest | dict) -> dict:
        request = RegisterRequest.model_validate(payload)
        self._audit()
        with self.journal.locked():
            data = self._read()
            artifact, definition, reference, publication, identity = self._candidate(request.calibration_id)
            if "policy" not in request.model_fields_set:
                request.policy = default_forward_policy(identity["frequency"])
            minimum_gap = {"daily": 1, "weekly": 7, "monthly": 31}[identity["frequency"]]
            if request.policy.max_observation_gap_days < minimum_gap:
                _fail("POLICY_FREQUENCY", "允许的观测间隔短于模型频率，请在登记前设置匹配的固定门槛。")
            for item in data["items"]:
                if item["kind"] == "protocol" and item["calibration_id"] == request.calibration_id:
                    self._protocol(data, item["id"])
                    if item["policy"] != request.policy.model_dump():
                        _fail("ALREADY_REGISTERED", "同一校准制品已注册，不允许更换门槛重试。")
                    return copy.deepcopy(item)
            now = self._now()
            cutoff = (now.date() - timedelta(days=1)).isoformat()
            if (_utc(artifact["created_at"]) > now or _utc(publication["published_at"]) > now
                    or reference["series"][-1]["observation_date"] > cutoff
                    or artifact["report"]["calibration"]["test_end"] > cutoff
                    or any(max(p.get("recognized_at") or "9999", p.get("data_available_at") or "9999") > cutoff for p in reference["series"])):
                _fail("FUTURE_REGISTRATION_INPUT", "注册输入包含服务端尚未经历的数据或发布时间。")
            execution, prefixes, temporal = self._execute(definition, cutoff)
            original_snapshots = artifact["report"]["lineage"]["data_snapshots"]
            if any(s.get("fingerprint") != original_snapshots.get(node, {}).get("fingerprint")
                   for node, s in execution["result"]["data_snapshots"].items()):
                _fail("CANDIDATE_SOURCE_CHANGED", "候选保存后输入快照已改变，请基于当前输入重新保存校准候选。")
            finished = self._now()
            if finished.date() != now.date():
                _fail("CLOCK_CHANGED", "注册跨日，请重新执行。")
            return self._append(data, "protocol", {
                "calibration_id": request.calibration_id, "artifact_hash": artifact["content_hash"],
                "definition_id": definition.id, "revision": definition.revision,
                "definition_hash": definition_content_hash(definition),
                "model_binding_hash": model_binding_hash(definition),
                "engine_hash": _engine_hash(self.graph),
                "forward_source_mode": "growing_legacy_index" if all(
                    n.type == "source.index" and not n.parameters.get("file_checksum")
                    for n in definition.graph.nodes if n.type.startswith("source.") and n.type != "source.constant"
                ) else "immutable_source_pending_new_observations",
                "reference": artifact["request"]["reference"], "reference_definition": identity,
                "original_publication": publication, "calibrator": artifact["report"]["calibration"],
                "calibrator_hash": _content_hash(artifact["report"]["calibration"]),
                "mapping": artifact["report"]["lineage"]["state_mapping"],
                "policy": request.policy.model_dump(), "registration_day": now.date().isoformat(),
                "deadline": (now + timedelta(days=request.policy.maximum_protocol_days)).isoformat(),
                "model_fit_origin": "declared_initial_training_interval_then_causal_inference" if any(n.type in LATENT for n in definition.graph.nodes) else "fixed_rule_causal_inference",
                "historical_assessment_recipe": artifact["report"]["lineage"]["prediction_method"],
                "source_prefixes": prefixes, "initial_data_snapshots": execution["result"]["data_snapshots"],
                "initial_model_audits": execution["result"]["diagnostics"].get("model_audits", {}),
                "temporal_audit": temporal, "status": "pending",
            }, finished)

    def capture(self, protocol_id: str) -> dict:
        self._audit()
        with self.journal.locked():
            data = self._read()
            protocol, definition = self._protocol(data, protocol_id)
            now = self._now()
            if now >= _utc(protocol["deadline"]):
                return {"status": "expired", "reason": "protocol_deadline"}
            if any(i["kind"] == "assessment" and i["protocol_id"] == protocol_id and i["status"] != "pending" for i in data["items"]):
                return {"status": "closed", "reason": "fixed_window_assessed"}
            cutoff = (now.date() - timedelta(days=1)).isoformat()
            if cutoff <= protocol["registration_day"]:
                return {"status": "pending", "reason": "no_post_registration_day"}
            observations = [i for i in data["items"] if i["kind"] == "observation" and i["protocol_id"] == protocol_id]
            definition, expected, checkpoint_id = sv.context(data, protocol, definition)
            source_history = sv.versions(data, protocol_id)
            try:
                execution, prefixes, temporal = self._execute(definition, cutoff, expected)
            except IndicatorDomainError as exc:
                if exc.code not in {"EMPTY_SERIES", "EMPTY_DATE_RANGE", "INDEX_DATA_NOT_FOUND",
                                    "INDEX_SERIES_NOT_FOUND", "INSUFFICIENT_ALIGNED_OBSERVATIONS"}:
                    raise
                return {"status": "unavailable", "reason": exc.code}
            eligible = [p for p in execution["series"] if p["observation_date"] <= cutoff
                        and p.get("data_available_at") and p.get("recognized_at")
                        and p["data_available_at"] <= cutoff and p["recognized_at"] <= cutoff]
            if not eligible:
                return {"status": "pending", "reason": "no_eligible_observation"}
            point = eligible[-1]
            day = point["observation_date"]
            if day <= protocol["registration_day"]:
                return {"status": "pending", "reason": "no_post_registration_observation"}
            policy = protocol["policy"]
            if (now.date() - date.fromisoformat(day)).days > policy["max_capture_lag_days"]:
                return {"status": "stale", "reason": "observation_too_old", "observation_date": day}
            old = next((i for i in observations if i["observation_date"] == day), None)
            if old:
                if old["raw_point"] != _json_safe(point):
                    _fail("OBSERVATION_CONFLICT", "同一观察日期出现不同预测，不允许覆盖。")
                return {"status": "duplicate", "observation": copy.deepcopy(old)}
            if observations and day <= observations[-1]["observation_date"]:
                _fail("BACKFILL", "不能补录已过去的预测。")
            # A later re-publication must not launder labels already published
            # before this capture. Inspect verified vintages of this definition.
            allowed = {(d["definition_id"], d["revision"]) for d in
                       sv.reference_definitions(protocol, sv.versions(data, protocol_id))}
            vintages = [r for r in self.graph.runs.list()
                        if (r.get("definition_id"), r.get("definition_revision")) in allowed]
            if len(vintages) > 128:
                _fail("REFERENCE_BUDGET", "参考版本超过128个，需整理工作区后捕获。")
            for run in vintages:
                for publication in run.get("publications", []):
                    if _utc(publication["published_at"]) <= now:
                        ref = {"run_id": run["id"], "publication_id": publication["id"], "content_hash": run["content_hash"]}
                        snapshot, _ = self._reference(ref)
                        if any(p["observation_date"] == day and p.get("state_id") not in {None, "unclassified"}
                               and p.get("recognized_at") and p["recognized_at"] <= cutoff for p in snapshot["series"]):
                            return {"status": "unavailable", "reason": "reference_label_already_published", "observation_date": day}
            if len(observations) >= 2000:
                _fail("OBSERVATION_BUDGET", "协议观测达到上限。")
            chosen, raw, q = self._probabilities(protocol, definition, point)
            if chosen < 0 or not np.isfinite(q).all():
                return {"status": "unavailable", "reason": "unknown_state_or_invalid_probability", "observation_date": day}
            finished = self._now()
            if finished.date() != now.date():
                _fail("CLOCK_CHANGED", "捕获跨越服务端日期，请重新执行。")
            if finished >= _utc(protocol["deadline"]):
                return {"status": "expired", "reason": "protocol_deadline"}
            item = self._append(data, "observation", {
                "protocol_id": protocol_id, "observation_date": day, "as_of": cutoff,
                "input_checkpoint_id": checkpoint_id,
                "source_version_id": source_history[-1]["id"] if source_history else None,
                "execution_model_binding_hash": model_binding_hash(definition),
                "observed_dates": [p["observation_date"] for p in execution["series"]
                                   if (observations[-1]["observation_date"] if observations else protocol["registration_day"])
                                   < p["observation_date"] <= day],
                "raw_point": point, "raw_point_hash": _content_hash(point),
                "chosen": chosen, "raw_probabilities": raw[0], "probabilities": q[0],
                "confidence": q[0, chosen], "calibrator_hash": protocol["calibrator_hash"],
                "model_binding_hash": protocol["model_binding_hash"], "source_prefixes": prefixes,
                "data_snapshots": execution["result"]["data_snapshots"], "temporal_audit": temporal,
                "model_audits": execution["result"]["diagnostics"].get("model_audits", {}),
            }, finished)
            return {"status": "captured", "observation": item}

    @staticmethod
    def _probabilities(protocol, definition, point):
        states = [s["id"] for s in protocol["reference_definition"]["states"]]
        chosen_state = protocol["mapping"].get(point.get("state_id"))
        chosen = states.index(chosen_state) if chosen_state in states else -1
        model_states = [s.id for s in definition.states]
        calibration = protocol["calibrator"]
        if calibration["method"] == "temperature":
            source = point.get("probabilities") or {}
            raw_source = np.asarray([[source.get(s, np.nan) for s in model_states]], np.float64)
            mapping = np.asarray([states.index(protocol["mapping"][s]) for s in model_states], np.int64)
            raw = nk.map_probability_kernel(raw_source, mapping, len(states))
        else:
            raw = np.asarray([[float(c == chosen) for c in range(len(states))]], np.float64)
        params = calibration["parameters"]
        q = nk.apply_kernel(np.array([chosen], np.int64), raw, np.asarray(params["counts"], np.float64),
                            float(params["temperature"] or 1.), int(calibration["method"] == "temperature"))
        return chosen, raw, q

    def _score(self, protocol, definition, observations, request, now, source_versions=()):
        reference, publication = self._reference(request.reference)
        if _definition_identity(reference) not in sv.reference_definitions(protocol, source_versions):
            _fail("REFERENCE_DEFINITION", "新参考必须保持历史定义修订、状态、族和对象完全一致。")
        if any(p["observation_date"] > now.date().isoformat() for p in reference["series"]):
            _fail("FUTURE_REFERENCE_OBSERVATIONS", "参考包含服务端尚未经历的未来观测。")
        published = _utc(publication["published_at"])
        original = request.reference.model_dump() == protocol["reference"]
        if published > now or (not original and published <= _utc(protocol["recorded_at"])):
            _fail("REFERENCE_TIME", "需要注册之后、当前已发布的新参考。")
        row_map = {p["observation_date"]: p for p in reference["series"]
                   if protocol["registration_day"] < p["observation_date"] < now.date().isoformat()}
        # Capture-time source-axis metadata also preserves dates later absent
        # from a reference vintage. Such dates remain unknown, not compressed.
        for observation in observations:
            for day in observation["observed_dates"]:
                if protocol["registration_day"] < day < now.date().isoformat():
                    row_map.setdefault(day, {"observation_date": day})
        rows = [row_map[day] for day in sorted(row_map)]
        policy = protocol["policy"]
        rows = rows[:policy["observation_window"]]
        dates, days = axis(rows)
        by_date = {o["observation_date"]: o for o in observations}
        states = [s["id"] for s in reference["states"]]
        codes = {s: i for i, s in enumerate(states)}
        y, pred = np.full(len(rows), -1, np.int64), np.full(len(rows), -1, np.int64)
        q = np.full((len(rows), len(states)), np.nan)
        used, immature = [], 0
        for i, row in enumerate(rows):
            recognized, available = row.get("recognized_at"), row.get("data_available_at")
            if not recognized or not available or max(recognized, available) >= now.date().isoformat():
                immature += 1
                continue
            if min(recognized, available) < row["observation_date"]:
                _fail("LABEL_TIME", "参考标签早于观察日期。")
            y[i] = codes.get(row.get("state_id"), -1)
            obs = by_date.get(dates[i])
            if obs is None:
                continue
            captured = _utc(obs["recorded_at"])
            if (captured <= _utc(protocol["recorded_at"]) or captured >= now
                    or captured >= _utc(protocol["deadline"])
                    or published <= captured or obs["observation_date"] <= protocol["registration_day"]
                    or obs["as_of"] >= captured.date().isoformat()
                    or (captured.date() - date.fromisoformat(dates[i])).days > policy["max_capture_lag_days"]
                    or obs["calibrator_hash"] != protocol["calibrator_hash"]
                    or obs["model_binding_hash"] != protocol["model_binding_hash"]):
                _fail("OBSERVATION_TIME", "预测必须先被捕获，之后才有成熟发布标签。")
            chosen, _, frozen = self._probabilities(protocol, definition, obs["raw_point"])
            if (obs["raw_point_hash"] != _content_hash(obs["raw_point"])
                    or chosen != obs["chosen"] or frozen[0].tolist() != obs["probabilities"]
                    or frozen[0, chosen] != obs["confidence"]):
                _fail("OBSERVATION_PARAMETERS", "预测概率不匹配冻结参数。")
            pred[i], q[i] = chosen, frozen[0]
            used.append({"id": obs["id"], "content_hash": obs["content_hash"]})
        base = np.asarray(protocol["calibrator"]["parameters"]["class_base"], np.float64)
        support, cycles, blocks, pairs, coverage, count, worst = pk.forward_blocks_kernel(
            days, y, pred, q, base, policy["block_size"], policy["max_observation_gap_days"])
        confidence_floor = float(protocol["calibrator"].get("confidence_floor", .6))
        accepted = nk.decision_kernel(pred, q, confidence_floor)
        state_evidence, qualified_states, fallback_states = _state_qualification(
            states, support, cycles, y, accepted, policy
        )
        metrics = {"classification": classification(y, pred, states),
                   "selective_classification": classification(y, accepted, states),
                   "probability": probability(y, pred, q, 10),
                   "class_base": probability(y, pred, np.broadcast_to(base, q.shape), 10),
                   "coverage": float(coverage) if len(rows) else None,
                   "paired_samples": int(pairs), "per_class": support.tolist(),
                   "complete_regimes_per_class": cycles.tolist(), "complete_blocks": int(count),
                   "state_evidence": state_evidence,
                   "qualified_states": qualified_states, "fallback_states": fallback_states,
                   "confidence_floor": confidence_floor,
                   "complete_time_blocks": int(count),
                   "block_brier_improvements": _json_safe(blocks), "axis_rows": len(rows),
                   "first_date": dates[0] if dates else None, "last_date": dates[-1] if dates else None}
        reasons = []
        if len(rows) < policy["observation_window"]:
            reasons.append("future_observation_window_incomplete")
        if immature:
            reasons.append("reference_labels_not_mature")
        if not observations:
            reasons.append("no_forward_captures")
        pending = bool(reasons)
        if pairs < policy["minimum_observations"]:
            reasons.append("insufficient_paired_observations")
        if not np.isfinite(coverage) or coverage < policy["minimum_coverage"]:
            reasons.append("insufficient_capture_coverage")
        if count < policy["minimum_complete_blocks"]:
            reasons.append("insufficient_complete_time_blocks")
        if not np.isfinite(worst) or worst < policy["minimum_brier_improvement"]:
            reasons.append("complete_blocks_do_not_all_improve_class_base")
        agreement = metrics["classification"]["accuracy"]
        metrics["global_agreement_meets_policy"] = bool(
            agreement is not None and agreement >= policy["minimum_agreement"]
        )
        # Overall agreement remains a diagnostic. State authorization is based on
        # state-specific accepted precision and independent episodes, so one rare
        # state cannot invalidate well-supported states.
        expiry = min(now + timedelta(days=policy["expires_after_days"]), _utc(protocol["deadline"]))
        if dates:
            expiry = min(expiry, datetime.combine(date.fromisoformat(dates[-1]) + timedelta(days=policy["expires_after_days"] + 1), datetime.min.time(), timezone.utc))
        if now >= expiry:
            reasons.append("forward_evidence_expired")
            pending = False
        failed_states = [row["state_id"] for row in state_evidence if row["status"] == "failed"]
        insufficient_states = [row["state_id"] for row in state_evidence if row["status"] == "insufficient_evidence"]
        global_failed = bool(reasons)
        if pending:
            status, outcome = "pending", "pending"
        elif global_failed:
            status, outcome = "rejected", "failed"
        elif qualified_states:
            status = "qualified"
            outcome = "qualified" if len(qualified_states) == len(states) else "partially_qualified"
        else:
            status = "rejected"
            outcome = "failed" if failed_states else "insufficient_evidence"
            reasons.append("no_state_qualified" if failed_states else "no_state_has_sufficient_forward_evidence")
        return _json_safe({"status": status, "outcome": outcome,
                          "qualified_states": qualified_states, "fallback_states": fallback_states,
                          "state_evidence": state_evidence,
                          "reasons": reasons, "metrics": metrics, "reference": request.reference.model_dump(),
                          "reference_publication": publication, "observations": used,
                          "available_from": now.isoformat(),
                          "available_from_date": (now.date() + timedelta(days=1)).isoformat(),
                          "expires_at": expiry.isoformat(), "criterion": policy["criterion"],
                          "statistical_significance_claim": False})

    def assess(self, protocol_id: str, payload: AssessRequest | dict) -> dict:
        request = AssessRequest.model_validate(payload)
        self._audit()
        with self.journal.locked():
            data = self._read()
            protocol, definition = self._protocol(data, protocol_id)
            terminal = next((i for i in data["items"] if i["kind"] == "assessment" and i["protocol_id"] == protocol_id and i["status"] != "pending"), None)
            if terminal:
                return copy.deepcopy(terminal)
            now = self._now()
            observations = [i for i in data["items"] if i["kind"] == "observation" and i["protocol_id"] == protocol_id]
            score = self._score(protocol, definition, observations, request, now,
                                sv.versions(data, protocol_id, before=now.isoformat()))
            finished = self._now()
            if finished.date() != now.date():
                _fail("CLOCK_CHANGED", "评估跨日，请重新执行。")
            if score["status"] == "qualified" and finished >= _utc(score["expires_at"]):
                score["status"] = "rejected"
                score["reasons"].append("forward_evidence_expired")
            score["available_from"] = finished.isoformat()
            return self._append(data, "assessment", {"protocol_id": protocol_id,
                                "calibration_id": protocol["calibration_id"],
                                "model_binding_hash": protocol["model_binding_hash"], **score}, finished)

    def get_protocol(self, protocol_id: str) -> dict:
        with self.journal.locked():
            data = self._read()
            protocol, _ = self._protocol(data, protocol_id)
            return copy.deepcopy(protocol)

    def catalog(self) -> dict:
        with self.journal.locked():
            data = self._read()
            items = [copy.deepcopy(self._protocol(data, i["id"])[0]) for i in data["items"] if i["kind"] == "protocol"]
            return {"items": items}

    def get_progress(self, protocol_id: str) -> dict:
        """Mutable read projection over immutable records; never re-captures data."""
        with self.journal.locked():
            data = self._read()
            protocol, _ = self._protocol(data, protocol_id)
            observations = [i for i in data["items"]
                            if i["kind"] == "observation" and i["protocol_id"] == protocol_id]
            assessments = [i for i in data["items"]
                           if i["kind"] == "assessment" and i["protocol_id"] == protocol_id]
            assessment_id = assessments[-1]["id"] if assessments else None
            source_versions = sv.versions(data, protocol_id)
            result = {"protocol": copy.deepcopy(protocol), "observations": len(observations),
                      "last_observation_date": observations[-1]["observation_date"] if observations else None,
                      "current_source_version": copy.deepcopy(source_versions[-1]) if source_versions else None,
                      "reference_definitions": sv.reference_definitions(protocol, source_versions)}
        result["latest_assessment"] = self.get_qualification(assessment_id) if assessment_id else None
        return result

    def get_qualification(self, qualification_id: str) -> dict:
        self._audit()
        with self.journal.locked():
            data = self._read()
            item = self._find(data, qualification_id, "assessment")
            protocol, definition = self._protocol(data, item["protocol_id"])
            observations = [i for i in data["items"] if i["kind"] == "observation" and i["protocol_id"] == protocol["id"] and _utc(i["recorded_at"]) < _utc(item["recorded_at"])]
            # Recompute gates from authenticated captures, never trust status alone.
            score = self._score(protocol, definition, observations, AssessRequest(reference=item["reference"]),
                                _utc(item["available_from"]),
                                sv.versions(data, protocol["id"], before=item["recorded_at"]))
            for key in ("status", "outcome", "qualified_states", "fallback_states", "state_evidence",
                        "reasons", "metrics", "observations", "reference_publication"):
                if score[key] != item[key]:
                    _fail("QUALIFICATION_INTEGRITY", "资格内容与实际证据不一致。")
            return copy.deepcopy(item)

    def verify_qualification(self, qualification_id: str, calibration_id: str,
                             model_binding_hash: str, reference_definition: dict | None = None,
                             *, decision_at: str | datetime | None = None) -> dict:
        """Validate current lineage and BOTH current and consumer decision times."""
        item = self.get_qualification(qualification_id)
        protocol = self.get_protocol(item["protocol_id"])
        with self.journal.locked():
            source_data = self._read()
            approved = sv.versions(source_data, protocol["id"])
        current_binding = approved[-1]["model_binding_hash"] if approved else protocol["model_binding_hash"]
        if (calibration_id != protocol["calibration_id"] or model_binding_hash != current_binding
                or (reference_definition is not None and reference_definition != protocol["reference_definition"])):
            _fail("CONSUMER_BINDING", "资格不属于该模型、校准器或参考定义。")
        now = self._now()
        decision = now if decision_at is None else (
            datetime.combine(date.fromisoformat(decision_at), datetime.min.time(), timezone.utc)
            if isinstance(decision_at, str) and len(decision_at) == 10 else _utc(decision_at))
        if item["status"] != "qualified":
            _fail("NOT_QUALIFIED", "前向证据尚未满足预注册门槛。")
        if not (_utc(item["available_from"]) < decision <= now < _utc(item["expires_at"])):
            _fail("QUALIFICATION_TIME", "资格尚未生效、已过期或不能授权过去/未来决策。")
        with self.journal.locked():
            data = self._read()
            _, definition = self._protocol(data, protocol["id"])
            definition, expected, _ = sv.context(data, protocol, definition)
            from .execution import model_binding_hash as binding_hash
            if model_binding_hash != binding_hash(definition):
                _fail("CONSUMER_BINDING", "数据版本在核验期间变化，请重新载入。")
            self._execute(definition, (now.date() - timedelta(days=1)).isoformat(), expected)
        if not now <= self._now() < _utc(item["expires_at"]):
            _fail("QUALIFICATION_TIME", "核验期间资格过期或时钟回退。")
        return item


def install(router, get_service, call):
    """Install on the existing router using its error adapter and service getter."""
    @router.post("/api/historical-regimes/prospective/register")
    def register(request: RegisterRequest):
        return call(get_service().register, request)

    @router.post("/api/historical-regimes/prospective/{protocol_id}/capture")
    def capture(protocol_id: str, request: CaptureRequest | None = None):
        return call(get_service().capture, protocol_id)

    @router.post("/api/historical-regimes/prospective/{protocol_id}/sources/preview")
    def source_preview(protocol_id: str, request: CaptureRequest | None = None):
        return call(sv.preview, get_service(), protocol_id)

    @router.post("/api/historical-regimes/prospective/{protocol_id}/sources/confirm")
    def source_confirm(protocol_id: str, request: sv.SourceConfirmRequest):
        return call(sv.confirm, get_service(), protocol_id, request)

    @router.post("/api/historical-regimes/prospective/{protocol_id}/assess")
    def assess(protocol_id: str, request: AssessRequest):
        return call(get_service().assess, protocol_id, request)

    @router.get("/api/historical-regimes/prospective/catalog")
    def catalog():
        return call(get_service().catalog)

    @router.get("/api/historical-regimes/prospective/protocols/{protocol_id}")
    def protocol(protocol_id: str):
        return call(get_service().get_protocol, protocol_id)

    @router.get("/api/historical-regimes/prospective/protocols/{protocol_id}/progress")
    def progress(protocol_id: str):
        return call(get_service().get_progress, protocol_id)

    @router.get("/api/historical-regimes/prospective/qualifications/{qualification_id}")
    def qualification(qualification_id: str):
        return call(get_service().get_qualification, qualification_id)
