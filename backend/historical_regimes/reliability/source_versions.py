"""Explicit, prefix-verified successors of frozen index data.

These functions orchestrate the existing graph/version services. They do not
replace source resolution, change model parameters, or publish reference labels.
"""
from __future__ import annotations

import copy
import time
from datetime import timedelta

from pydantic import BaseModel, ConfigDict, Field
from custom_indicators.errors import ConflictError
from ..v2_contracts import parse_definition_v2, definition_content_hash
from ..v2_service import _content_hash, _required_node_ids, _definition_output_frequency
from .execution import model_binding_hash

BINDING_FIELDS = ("snapshot_id", "snapshot_generation", "source_file", "file_checksum")


class SourceConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preview_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


def fail(code, message):
    raise ConflictError("PROSPECTIVE_SOURCE_" + code, message)


def sources(definition):
    required = _required_node_ids(definition)
    return [n for n in definition.graph.nodes if n.id in required
            and n.type.startswith("source.") and n.type != "source.constant"]


def same_source_targets(definition):
    if not definition.evaluation_targets:
        return True
    nodes = sources(definition)
    if len(nodes) != 1:
        return False
    def identity(spec):
        return {k: v for k, v in spec.items()
                if k not in {"name", "description", "frequency", "availability_mode", "start_date", "end_date"}}
    expected = identity({"kind": nodes[0].type.split(".", 1)[1], **nodes[0].parameters})
    return all(identity(target.source) == expected for target in definition.evaluation_targets)


def rebind(definition, bindings):
    if not same_source_targets(definition):
        fail("AXIS", "评价对象与唯一输入源不一致，不能续接。")
    payload = definition.model_dump(mode="json")
    nodes = {n["id"]: n for n in payload["graph"]["nodes"]}
    if set(bindings) != {n.id for n in sources(definition)}:
        fail("AXIS", "续接来源与冻结模型不一致。")
    for node_id, binding in bindings.items():
        if nodes[node_id]["type"] != "source.index" or set(binding) != set(BINDING_FIELDS):
            fail("BINDING", "仅允许替换指数的完整快照绑定，不允许修改模型参数。")
        nodes[node_id]["parameters"].update(binding)
    if payload["evaluation_targets"]:
        binding = next(iter(bindings.values()))
        for target in payload["evaluation_targets"]:
            target["source"].update(binding)
    return parse_definition_v2(payload)


def recipe_hash(definition):
    payload = definition.model_dump(mode="json", exclude={"id", "revision", "created_at", "updated_at"})
    for node in payload["graph"]["nodes"]:
        if node["type"] == "source.index":
            for key in BINDING_FIELDS:
                node["parameters"].pop(key, None)
    for target in payload["evaluation_targets"]:
        if target["source"].get("kind") == "index":
            for key in BINDING_FIELDS:
                target["source"].pop(key, None)
    return _content_hash(payload)


def versions(data, protocol_id, before=None):
    return [i for i in data["items"] if i["kind"] == "source_version"
            and i["protocol_id"] == protocol_id
            and (before is None or i["recorded_at"] <= before)]


def context(data, protocol, definition):
    records = [i for i in data["items"] if i.get("protocol_id") == protocol["id"]
               and i["kind"] in {"observation", "source_version"}]
    current = versions(data, protocol["id"])
    if current:
        definition = rebind(definition, current[-1]["model_bindings"])
        if model_binding_hash(definition) != current[-1]["model_binding_hash"]:
            fail("INTEGRITY", "已确认数据版本不匹配冻结模型。")
    expected = records[-1]["source_prefixes"] if records else protocol["source_prefixes"]
    return definition, expected, records[-1]["id"] if records else protocol["id"]


def reference_definitions(protocol, source_versions):
    return [protocol["reference_definition"], *[v["reference_definition"] for v in source_versions]]


def _build(service, data, protocol, definition):
    now = service._now()
    if now.isoformat() >= protocol["deadline"]:
        fail("EXPIRED", "前瞻协议已过期，不能续接数据。")
    if any(i["kind"] == "assessment" and i.get("protocol_id") == protocol["id"]
           and i["status"] == "rejected" for i in data["items"]):
        fail("CLOSED", "固定窗口已检验失败，不能通过续接数据重考。")
    current, expected, previous_id = context(data, protocol, definition)
    original = parse_definition_v2(service.graph.get_definition(
        protocol["reference_definition"]["definition_id"], protocol["reference_definition"]["revision"]))
    latest = parse_definition_v2(service.graph.get_definition(original.id))
    if recipe_hash(latest) != recipe_hash(original):
        fail("REFERENCE_CHANGED", "历史参考算法已修改，请恢复原定义或重新研究候选，不能覆盖现有编辑。")
    ms, rs = sources(current), sources(original)
    if (len(ms) != 1 or len(rs) != 1 or ms[0].type != "source.index"
            or rs[0].type != "source.index" or not same_source_targets(current) or not same_source_targets(original)):
        fail("UNSUPPORTED", "数据续接目前仅支持单一指数来源；上传、多源及宏观定义不能自动续接。")
    model_bindings = {ms[0].id: service.graph._active_snapshot_binding(ms[0].type, ms[0].parameters)}
    reference_bindings = {rs[0].id: service.graph._active_snapshot_binding(rs[0].type, rs[0].parameters)}
    if model_bindings[ms[0].id] != reference_bindings[rs[0].id]:
        fail("SNAPSHOT_CHANGED", "读取期间活跃快照发生变化，请重新检查。")
    candidate = rebind(definition, model_bindings)
    reference = rebind(latest, reference_bindings)
    cutoff = (now.date() - timedelta(days=1)).isoformat()
    # Preparing is an explicit user operation; the executor never compiles on fallback.
    service.graph.prepare(candidate.model_dump(mode="json"))
    execution, prefixes, _ = service._execute(candidate, cutoff, expected)
    old_count, new_count = expected[ms[0].id]["count"], prefixes[ms[0].id]["count"]
    if new_count <= old_count:
        fail("NO_NEW_OBSERVATIONS", "当前快照没有新增的已知观测，无需续接；不要覆盖旧文件。")
    # Both recipes read the same immutable index file/field. Reference bindings
    # are validated by the original service at save and subsequent execution.
    if service._now().date() != now.date():
        fail("CLOCK_CHANGED", "检查跨越服务端日期，请重试。")
    value = {"protocol_id": protocol["id"], "previous_id": previous_id,
             "as_of": cutoff, "model_bindings": model_bindings,
             "model_binding_hash": model_binding_hash(candidate),
             "reference_bindings": reference_bindings,
             "reference_id": latest.id, "reference_revision": latest.revision,
             "reference_candidate_hash": definition_content_hash(reference),
             "source_prefixes": prefixes, "data_snapshots": execution["result"]["data_snapshots"],
             "old_observations": old_count, "new_observations": new_count,
             "added_observations": new_count - old_count, "prefix_unchanged": True,
             "previous_bindings": {ms[0].id: {k: ms[0].parameters.get(k) for k in BINDING_FIELDS}}}
    return value, reference


def preview(service, protocol_id):
    service._audit()
    with service.journal.locked():
        data = service._read()
        protocol, definition = service._protocol(data, protocol_id)
        value, _ = _build(service, data, protocol, definition)
        digest = _content_hash(value)
        now = time.monotonic()
        service._source_previews = {h: v for h, v in service._source_previews.items() if v["expires"] > now}
        while len(service._source_previews) >= 4:
            service._source_previews.pop(next(iter(service._source_previews)))
        service._source_previews[digest] = {"value": copy.deepcopy(value), "expires": now + 900}
        return {"preview_hash": digest, **value}


def confirm(service, protocol_id, payload):
    from .prospective import _definition_identity
    request = SourceConfirmRequest.model_validate(payload)
    service._audit()
    with service.journal.locked():
        data = service._read()
        protocol, definition = service._protocol(data, protocol_id)
        existing = next((v for v in versions(data, protocol_id)
                         if v.get("preview_hash") == request.preview_hash), None)
        if existing:
            return copy.deepcopy(existing)
        cached = service._source_previews.get(request.preview_hash)
        if not cached or cached["expires"] <= time.monotonic() or cached["value"]["protocol_id"] != protocol_id:
            fail("PREVIEW_EXPIRED", "续接预览已过期或不属于当前协议，请重新检查。")
        value, candidate_ref = _build(service, data, protocol, definition)
        # A prior reference-version write followed by a journal write failure may
        # be reused only when that exact candidate is now the latest revision.
        previous = cached["value"]
        check = dict(value)
        if value["reference_candidate_hash"] == previous["reference_candidate_hash"]:
            check["reference_revision"] = previous["reference_revision"]
        if _content_hash(check) != request.preview_hash:
            fail("PREVIEW_CHANGED", "数据、参考或实际捕获已变化，请重新检查再确认。")
        latest = service.graph.get_definition(candidate_ref.id)
        if definition_content_hash(parse_definition_v2(latest)) == definition_content_hash(candidate_ref):
            saved = latest
        else:
            saved = service.graph.update_definition(candidate_ref.id, candidate_ref.revision,
                                                    candidate_ref.model_dump(mode="json"))
        # Construct only metadata, not a second historical result or publication.
        ref_definition = parse_definition_v2(saved)
        identity = _definition_identity({"definition": saved, "definition_id": saved["id"],
            "definition_revision": saved["revision"],
            "states": [s.model_dump(mode="json") for s in ref_definition.states],
            "frequency": _definition_output_frequency(ref_definition)})
        event = service._append(data, "source_version", {**value, "preview_hash": request.preview_hash,
            "reference_definition": identity, "status": "accepted"}, service._now())
        service._source_previews.pop(request.preview_hash, None)
        return event
