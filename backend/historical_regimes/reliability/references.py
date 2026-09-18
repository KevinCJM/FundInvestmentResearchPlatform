"""Verified projection of existing publications; no second reference store."""
from custom_indicators.errors import IndicatorDomainError, ValidationError
from ..v2_contracts import parse_definition_v2, definition_content_hash


def resolve_reference(graph, reference, *, hydrate=True):
    from ..v2_service import _stored_run_snapshot_hash
    ref = reference.model_dump() if hasattr(reference, "model_dump") else reference
    run = graph.runs.get(ref["run_id"])
    if (run.get("schema_version") != "2.0" or run.get("immutable") is not True
            or run.get("mode") != "retrospective"
            or run.get("content_hash") != ref["content_hash"]
            or _stored_run_snapshot_hash(run) != ref["content_hash"]):
        raise ValidationError("REFERENCE_SNAPSHOT_MISMATCH", "参考必须是完整的不可变事后运行。")
    definition = parse_definition_v2(graph.get_definition(run["definition_id"], run["definition_revision"]))
    if (definition_content_hash(definition) != run.get("definition_snapshot_hash")
            or run.get("states") != [s.model_dump(mode="json") for s in definition.states]):
        raise ValidationError("REFERENCE_DEFINITION_MISMATCH", "参考定义或状态轴已改变。")
    if any(n.type == "annotation.manual_events" for n in definition.graph.nodes):
        raise ValidationError("REFERENCE_MANUAL_EVENTS", "可重叠人工事件不能作为互斥参考标签。")
    publications = [p for p in run.get("publications", [])
                    if p.get("id") == ref["publication_id"] and p.get("run_id") == run["id"]
                    and p.get("run_content_hash") == run["content_hash"]
                    and p.get("definition_revision") == run["definition_revision"]
                    and p.get("usage") in {"research_display", "product_research", "formal_backtest", "taa"}
                    and p.get("published_at") and p.get("fit_mode") == "retrospective"]
    if len(publications) != 1:
        raise ValidationError("REFERENCE_PUBLICATION_MISMATCH", "参考发布记录与精确快照不符。")
    return (graph.hydrate_run_snapshot(run) if hydrate else run), publications[0]


def reference_catalog(graph):
    items = []
    for run in graph.runs.list():
        for publication in run.get("publications", []):
            ref = dict(run_id=run["id"], publication_id=publication["id"], content_hash=run.get("content_hash"))
            try:
                verified, verified_publication = resolve_reference(graph, ref, hydrate=False)
            except IndicatorDomainError:
                continue
            items.append({**ref, "publication_usage": verified_publication["usage"], **{key: verified.get(key) for key in (
                "definition_id", "definition_revision", "name", "frequency", "states", "as_of", "created_at", "series_summary")}})
    return {"items": items}


def require_research_reference(graph, definition):
    """Unbound realtime studies remain drafts; legacy untyped studies stay readable."""
    study = definition.study
    if study is None or study.purpose != "realtime_recognition":
        return
    if study.reference is None:
        raise ValidationError("REALTIME_REFERENCE_REQUIRED", "请先绑定已确认的历史参考；无参考模型可保存为探索草稿。", "study.reference")
    run, _ = resolve_reference(graph, study.reference, hydrate=False)
    targets = {s["id"] for s in run["states"]}
    sources = {s.id for s in definition.states}
    mapping = study.state_mapping
    if ((mapping is None and sources != targets)
            or (mapping is not None and (set(mapping) != sources or not set(mapping.values()).issubset(targets)))):
        raise ValidationError("REFERENCE_STATE_MAPPING_REQUIRED", "请为每个实时状态明确配置历史参考状态。", "study.state_mapping")
