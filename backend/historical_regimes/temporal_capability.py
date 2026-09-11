"""Pure per-output temporal obligations. Empirical evidence is a separate layer.

Legacy booleans describe a registry entry, not an authored graph. This resolver
owns graph eligibility and never executes a numeric plan or reads market data.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
from custom_indicators.errors import IndicatorDomainError

POLICY_VERSION = "regime-temporal/1"
STATUSES = {"conditional": "可按当时信息试算", "retrospective_required": "仅事后研究", "audit_unknown": "尚未验证"}


@lru_cache(maxsize=1)
def operator_baseline() -> dict:
    # The same reviewed baseline is used by backend/causality/audit.py and CI.
    path = Path(__file__).resolve().parents[1] / "causality" / "baseline.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("operators", {})


def temporal_contract(node_type: str, metadata: Mapping[str, Any]) -> dict:
    """Describe time semantics, not a claim that every configuration is safe."""
    if node_type == "annotation.manual_events":
        return {"rule": "manual_hindsight", "reason": "历史事件的选界与认定来自事后知识；日期匹配不等于当时可知。"}
    if node_type in {"model.hmm", "model.markov", "model.gmm"}:
        return {"rule": "fit_and_infer", "reason": "由本次训练范围、推断方式和所选输出决定。"}
    if metadata.get("knowledge_scope") == "full_input":
        return {"rule": "full_input", "reason": "需要完整输入或右侧数据确认；下游不能消除该知识依赖。"}
    if node_type == "post.merge_short_regimes":
        return {"rule": "future_confirmation", "reason": "短反向段需要后续主状态确认后才可事后合并。"}
    if node_type.startswith("source.") and node_type != "source.constant":
        return {"rule": "source_availability", "reason": "以真实公布时间与数据版本为准，待运行核对。"}
    if node_type == "feature.formula" or metadata.get("typed_operator_id") or metadata.get("indicator_reference"):
        return {"rule": "typed_expression", "reason": "按实际公式、参数、窗口和共享算子审计基线推导。"}
    if node_type == "model.external_optimized":
        return {"rule": "unknown", "reason": "外部模型尚无已核验的时点契约。"}
    if node_type.startswith("segment."):
        return {"rule": "boundary_window", "reason": "依赖完整区间边界及窗口内全部输入的可得时点。"}
    if node_type == "align.resample":
        return {"rule": "calendar_window", "reason": "周期完成且聚合输入已公布后才能使用。"}
    # Existing supported sequential kernels are audited in their real execution
    # context. A declaration alone never becomes realtime_verified.
    if metadata.get("causal") is True:
        return {"rule": "history", "reason": "单向历史依赖；本次结果仍需数据和数值探针验证。"}
    if metadata.get("causal") is False:
        return {"rule": "future_confirmation", "reason": "当前内核的确认/定界使用后续信息。"}
    return {"rule": "unknown", "reason": "缺少时点契约，不能判定为实时。"}


def _formula_obligations(node, registry) -> tuple[str, str]:
    from .indicator_nodes import typed_node_expressions
    from .formula import CAUSAL_OPERATOR_IDS
    from cal_indicators.typed_dsl import compose_typed_expression, TypedDslError
    from cal_indicators.typed_operators import TYPED_DSL_VERSION, TYPED_OPERATOR_REGISTRY_VERSION
    from computation_graph.causal_series import causal_violations, series_variable_types
    try:
        expressions = typed_node_expressions(node, registry)
        if not expressions:
            return "audit_unknown", "没有可审计的公式。"
        baseline = operator_baseline()
        definition = registry[node.type].get('_indicator_definition')
        inputs = set(node.inputs)
        aliases = node.parameters.get('variables')
        if isinstance(aliases, Mapping):
            inputs.update(name for name, port in aliases.items() if port in node.inputs)
        types = series_variable_types(inputs, definition)
        protocol = definition if definition and definition.get('result_kind') == 'time_series' else {}
        for expression in expressions.values():
            plan = compose_typed_expression(expression, variable_types=types, output_contract='series',
                dsl_version=protocol.get('dsl_version', TYPED_DSL_VERSION),
                operator_registry_version=protocol.get('operator_registry_version', TYPED_OPERATOR_REGISTRY_VERSION))
            violations = causal_violations(plan, CAUSAL_OPERATOR_IDS)
            if violations:
                return 'audit_unknown', f'{violations[0]} 未被合法的历史窗口约束。'
            for step in plan.nodes:
                operator = step.operator_id
                if operator is None:
                    continue
                verdict = baseline.get(operator, {}).get('verdict')
                if verdict == 'leak':
                    return 'retrospective_required', f'共享审计发现 {operator} 读取未来。'
                if verdict not in {'causal', 'window_consuming'}:
                    return 'audit_unknown', f'算子 {operator} 缺少有效审计基线。'
                # A window-consuming primitive is safe here only because the
                # actual DAG value uses were proved scoped above, not by name.
        return "conditional", "实际公式的共享算子基线通过；运行时还需检查数据和窗口。"
    except (SyntaxError, ValueError, KeyError, TypeError, IndicatorDomainError, TypedDslError):
        return "audit_unknown", "公式或参数未完成，无法推导时点能力。"


def analyze_temporal(definition, registry, mode: str = "realtime", snapshots: Mapping | None = None) -> dict:
    nodes = {n.id: n for n in definition.graph.nodes}
    memo, visiting = {}, set()
    obligations = {}

    def visit(node_id: str, port: str) -> dict:
        key = node_id + "." + port
        if key in memo:
            return memo[key]
        if key in visiting or node_id not in nodes or nodes[node_id].type not in registry:
            return {"status": "audit_unknown", "reasons": [{"node_id": node_id, "code": "UNKNOWN_DEPENDENCY", "message": "依赖缺失或循环。", "path": [key]}], "node_ids": []}
        visiting.add(key)
        node = nodes[node_id]
        metadata = registry[node.type]
        contract = temporal_contract(node.type, metadata)
        rule = contract["rule"]
        status, reasons = "conditional", []
        hindsight, repaint = False, False
        if rule == "manual_hindsight":
            status, hindsight = "retrospective_required", True
        elif rule in {"full_input", "future_confirmation"}:
            status, repaint = "retrospective_required", True
        elif rule == "fit_and_infer":
            if mode == "retrospective" and port != "score":
                status, repaint = "retrospective_required", True
                contract = {**contract, "reason": "本次采用全样本训练/平滑并回标历史；训练参数也含未来信息。"}
            else:
                contract = {**contract, "reason": f"初始训练窗口锁定（{node.parameters.get('initial_train_size', 60)}期），之后推断；需验证预热与训练边界。"}
        elif rule == "typed_expression":
            status, reason = _formula_obligations(node, registry)
            contract = {**contract, "reason": reason}
        elif rule == "source_availability":
            params = node.parameters
            snap = (snapshots or {}).get(node_id, {})
            if params.get("availability_mode") == "latest" or snap.get("revision_policy") == "latest_vintage":
                status = "retrospective_required"
                contract = {**contract, "reason": "使用最新修订值，不能还原历史当时版本。"}
            elif snap.get("release_dates_verified") is False or snap.get("availability_status") == "release_date_unknown" or (snap.get("pit") or {}).get("supported") is False:
                status = "retrospective_required"
                contract = {**contract, "reason": "该数据快照缺少已核实的历史公布/版本时点。"}
        elif rule == "unknown":
            status = "audit_unknown"
        if status != "conditional":
            reasons.append({"node_id": node_id, "node_type": node.type, "code": rule.upper(), "message": contract["reason"], "path": [key]})
        ids = {node_id}
        parents = [visit(ref.node_id, ref.port) for ref in node.inputs.values()]
        for parent in parents:
            ids.update(parent["node_ids"])
            hindsight |= parent.get("semantic_hindsight", False)
            repaint |= parent.get("may_repaint", False)
            if parent["status"] == "retrospective_required" or parent["status"] == "audit_unknown" and status == "conditional":
                status = parent["status"]
            for reason in parent["reasons"]:
                reasons.append({**reason, "path": [*reason["path"], key]})
        unique = {(r["node_id"], r["code"]): r for r in reasons}
        result = {"status": status, "reasons": list(unique.values()), "node_ids": sorted(ids),
                  "semantic_hindsight": hindsight, "may_repaint": repaint}
        obligations[node_id] = {"node_id": node_id, "label": node.label or metadata.get("label", node.type),
                                "rule": rule, "description": contract["reason"],
                                "warmup_sensitive": node.type in {"filter.ema", "filter.kama", "filter.super_smoother", "filter.kalman"} or metadata.get("typed_operator_id") == "recursive_smooth"}
        memo[key] = result
        visiting.remove(key)
        return result

    outputs = {name: visit(ref.node_id, ref.port) for name, ref in definition.graph.outputs.items()}
    statuses = [p["status"] for p in outputs.values()]
    status = "retrospective_required" if "retrospective_required" in statuses else "audit_unknown" if not statuses or "audit_unknown" in statuses else "conditional"
    reasons = {(r["node_id"], r["code"]): r for p in outputs.values() for r in p["reasons"]}
    return {"policy_version": POLICY_VERSION, "status": status, "label": STATUSES[status], "mode": mode,
            "realtime_supported": status == "conditional", "verified": False,
            "outputs": outputs, "ports": memo, "nodes": list(obligations.values()), "reasons": list(reasons.values()),
            "semantic_hindsight": any(p.get("semantic_hindsight") for p in outputs.values()),
            "may_repaint": any(p.get("may_repaint") for p in outputs.values()),
            "evidence_scope": "contract_and_operator_baseline", "runtime_audit": "not_run",
            "note": "条件通过不是数学证明或发布许可；数值审计、数据可得性与用途门禁独立校验。"}


def compatibility_projection(report: dict) -> dict:
    reasons = report["reasons"]
    ids = sorted({r["node_id"] for r in reasons})
    return {"causal": report["status"] == "conditional", "repaints": report["may_repaint"],
            "realtime_supported": report["realtime_supported"], "noncausal_node_ids": ids,
            "repaint_node_ids": sorted({n for p in report["outputs"].values() if p.get("may_repaint") for n in p["node_ids"]}),
            "non_realtime_node_ids": ids}


def executable_dependencies(definition, registry, mode):
    """Real outputs always retain their obligations; optional debug branches do not.

    An unsafe debug node must be explicitly previewed in retrospective mode.
    Safe debug branches retain their existing preview contract.
    """
    from .v2_contracts import GraphPortRefV2
    actual = analyze_temporal(definition, registry, mode)
    required = {n for output in actual["outputs"].values() for n in output["node_ids"]}
    node_map = {n.id: n for n in definition.graph.nodes}
    skipped = []
    for node_id in definition.graph.exposed_node_ids:
        if node_id in required or node_id not in node_map:
            continue
        metadata = registry.get(node_map[node_id].type, {})
        references = {"diagnostic_" + str(i): GraphPortRefV2(node_id=node_id, port=p["name"])
                      for i, p in enumerate(metadata.get("outputs", []))}
        projection = definition.model_copy(update={"graph": definition.graph.model_copy(update={"outputs": references})})
        report = analyze_temporal(projection, registry, mode)
        if mode == "realtime" and not report["realtime_supported"]:
            skipped.append(node_id)
            continue
        required.update(n for output in report["outputs"].values() for n in output["node_ids"])
    return required, skipped
