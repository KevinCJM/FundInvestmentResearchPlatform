"""Bounded empirical audit of the real warmed graph, never the Python evaluator."""
from __future__ import annotations

import time
import numpy as np
from causality.probes import Verdict, PERTURB_STYLES, WARMUP_RTOL
from causality.temporal_numba import (perturb_available_tail, compare_available_prefix, available_positions,
                                     latest_relative_difference)
from .temporal_capability import analyze_temporal, POLICY_VERSION


def _matrix(value):
    array = np.ascontiguousarray(value, dtype=np.float64)
    return array.reshape(-1, 1) if array.ndim == 1 else array


def source_time_evidence(source_ids, snapshots):
    checks = []
    for source_id in source_ids:
        snapshot = snapshots.get(source_id, {})
        pit = snapshot.get("pit") or {}
        invalid = snapshot.get("revision_policy") == "latest_vintage" or snapshot.get("release_dates_verified") is False or pit.get("supported") is False
        declared = snapshot.get("availability_evidence") == "provided" or snapshot.get("availability_contract") == "date_only_market_close" or snapshot.get("release_dates_verified") is True or pit.get("supported") is True
        passed = bool(snapshot.get("fingerprint")) and declared and not invalid
        checks.append({"node_id": source_id, "passed": passed,
                       "status": "passed" if passed else "unverified",
                       "evidence": snapshot.get("availability_contract") or snapshot.get("availability_evidence") or pit.get("availability_status") or "unknown",
                       "revision_policy": snapshot.get("revision_policy"),
                       "message": "按本次来源契约和冻结版本检查；日期级数据不证明日内可交易时点。" if passed else "缺少公布/版本时点证据，或使用最终修订数据，不能验证实时资格。"})
    return {"passed": bool(checks) and all(c["passed"] for c in checks), "sources": checks,
            "scope": "declared_source_contract_and_frozen_snapshot_not_independent_historical_provenance_proof"}


def audit_execution(service, definition, mode, as_of, plan, execution, cancel=None, max_seconds=30.0):
    from .v2_registry import NODE_REGISTRY
    from .v2_service import PortValue
    from .v2_contracts import inspect_definition_v2, definition_content_hash

    capability = analyze_temporal(definition, NODE_REGISTRY, mode, execution["result"].get("data_snapshots"))
    started = time.monotonic()
    base = execution["node_outputs"]
    inspection = inspect_definition_v2(definition)
    required = set(inspection["dependencies"]["required_for_outputs"])
    nodes = {n.id: n for n in definition.graph.nodes}
    order = [n for n in inspection["topological_order"] if n in required]
    source_ids = [n for n in order if nodes[n].type.startswith("source.") and nodes[n].type != "source.constant"]
    refs = {(r.node_id, r.port) for r in definition.graph.outputs.values()}
    refs.update((r.node_id, r.port) for n in order for r in nodes[n].inputs.values())
    refs = sorted(refs)
    roots = {name: (r.node_id, r.port) for name, r in definition.graph.outputs.items()}
    primary = definition.graph.outputs.get("state") or next(iter(definition.graph.outputs.values()))
    axis = base[primary.node_id][primary.port].dates
    sources = {n: base[n]["value"] for n in source_ids}
    reference = {(n, p): _matrix(base[n][p].values) for n, p in refs}
    findings = []
    runs, comparisons = 0, 0
    errors = []
    tested = set()

    def check():
        if cancel:
            cancel()
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("审计时间预算已用完；未覆盖部分不得判通过。")

    def evaluate(overrides):
        nonlocal runs
        check()
        outputs = {n: {"value": overrides.get(n, sources[n])} for n in source_ids}
        formula_audits, model_audits = {}, {}
        for node_id in order:
            check()
            if node_id not in outputs:
                outputs[node_id] = service._execute_numeric_node(nodes[node_id], outputs, len(definition.states), mode,
                    as_of, None, plan.get("formula_plans", {}), formula_audits, model_audits)
        runs += 1
        return outputs

    def compare(candidate, cutoff, probe, source_id=None, style=None):
        nonlocal comparisons
        for node_id, port in refs:
            check()
            a, b = base[node_id][port], candidate[node_id][port]
            verdict, count, mismatch = compare_available_prefix(reference[(node_id, port)], a.dates, a.available,
                _matrix(b.values), b.dates, b.available, np.int64(cutoff), np.int64(np.issubdtype(a.values.dtype, np.integer)))
            comparisons += 1
            if count:
                tested.add((node_id, port, probe))
            if verdict:
                findings.append({"probe": probe, "node_id": node_id, "port": port,
                    "verdict": str(Verdict.LEAK if verdict == 1 else Verdict.UNKNOWN), "source_id": source_id,
                    "style": style, "decision_at": str(np.datetime64(int(cutoff), "ns")),
                    "first_mismatch_date": str(np.datetime64(int(a.dates[mismatch]), "ns")) if mismatch >= 0 else None,
                    "message": "未来信息改变了已可得输出。" if verdict == 1 else "差异位于浮点灰区，需复核。"})

    can_probe = bool(source_ids) and len(source_ids) <= 4 and 8 <= len(axis) <= 30000 and len(required) <= 64
    if not can_probe:
        errors.append("样本须为8—30000条，最多4个数据源、64个实际依赖节点；当前图谱超出自动探针覆盖范围。")
    if capability["semantic_hindsight"]:
        can_probe = False
        errors.append("人工历史事件属于事后选界；数值扰动不能证明该标签当时已知。")
    cutoffs = sorted({int(axis[len(axis) // 3]), int(axis[2 * len(axis) // 3]), int(axis[-2])}) if len(axis) >= 8 else []
    warmup = {"status": "not_applicable", "sensitive": False, "findings": [], "affects_causality_verdict": False}
    if can_probe:
        try:
            for cutoff in cutoffs:
                for source_id, source in sources.items():
                    for style, (style_name, _, _) in enumerate(PERTURB_STYLES):
                        changed = perturb_available_tail(_matrix(source.values), source.available, np.int64(cutoff), np.int64(style))
                        # Skip an ineffective perturbation instead of manufacturing evidence.
                        if np.array_equal(changed, _matrix(source.values), equal_nan=True):
                            continue
                        values = changed[:, 0] if source.values.ndim == 1 else changed
                        candidate = evaluate({source_id: PortValue(values, source.dates, source.available)})
                        compare(candidate, cutoff, "tail_perturbation", source_id, style_name)
                truncated = {}
                for source_id, source in sources.items():
                    positions = available_positions(source.available, np.int64(cutoff))
                    if len(positions) and int(positions[-1]) == len(positions) - 1:
                        # Prefix slices share memory; no repeated copy of historical inputs.
                        k = len(positions)
                        truncated[source_id] = PortValue(source.values[:k], source.dates[:k], source.available[:k])
                    else:
                        truncated[source_id] = service._take_port(source, positions, source.dates[positions], source.available[positions])
                try:
                    compare(evaluate(truncated), cutoff, "prefix_replay")
                except (ValueError, KeyError) as exc:
                    errors.append(f"前缀样本不足：{type(exc).__name__}")
            recursive = any(n["warmup_sensitive"] for n in capability["nodes"])
            if recursive:
                warmup["status"] = "tested"
                for fraction in (0.25, 0.5):
                    clipped = {n: PortValue(v.values[int(len(v.dates) * fraction):], v.dates[int(len(v.dates) * fraction):], v.available[int(len(v.dates) * fraction):]) for n, v in sources.items()}
                    try:
                        candidate = evaluate(clipped)
                        for name, (n, p) in roots.items():
                            delta = latest_relative_difference(reference[(n, p)], _matrix(candidate[n][p].values))
                            if np.isfinite(delta):
                                warmup["findings"].append({"output": name, "history_fraction_removed": fraction, "relative_difference": float(delta)})
                                warmup["sensitive"] |= bool(delta > WARMUP_RTOL)
                            else:
                                warmup["status"] = "insufficient"
                    except Exception as exc:
                        # Cancellation/time budget must propagate out of this diagnostic.
                        check()
                        warmup["status"] = "insufficient"
                        warmup["findings"].append({"reason": type(exc).__name__})
        except Exception as exc:
            if cancel:
                cancel()
            errors.append(str(exc) if isinstance(exc, TimeoutError) else f"探针无法完成：{type(exc).__name__}")
    missing = [{"node_id": n, "port": p, "probe": probe} for n, p in refs for probe in ("tail_perturbation", "prefix_replay") if (n, p, probe) not in tested]
    leaked = any(f["verdict"] == "leak" for f in findings)
    numerical = "leak" if leaked else "unknown" if errors or missing or findings else "causal"
    # A report says 'tested', not a universal proof. Full release gates stay independent.
    data_checks = source_time_evidence(source_ids, execution["result"].get("data_snapshots", {}))
    verified = numerical == "causal" and capability["status"] == "conditional" and data_checks["passed"]
    status = "retrospective_required" if capability["status"] == "retrospective_required" or leaked else "realtime_verified" if verified else "audit_unknown"
    return {**capability, "status": status, "label": "本次时点审计通过" if verified else "仅事后研究" if status == "retrospective_required" else "尚未完成验证",
            "verified": verified, "runtime_audit": "completed" if not errors else "incomplete",
            "definition_hash": definition_content_hash(definition), "as_of": as_of,
            "data_snapshots": execution["result"].get("data_snapshots", {}),
            "numerical_verdict": numerical, "data_checks": data_checks, "findings": findings[:100], "errors": errors[:20],
            "coverage": {"executions": runs, "comparisons": comparisons, "cutoffs": len(cutoffs), "ports": len(refs), "untested": missing},
            "warmup": warmup, "elapsed_ms": (time.monotonic() - started) * 1000,
            "execution_backend": "numba_njit_fixed_signature", "python_fallback": 0, "request_time_compilation": 0,
            "note": "结论仅适用于本次冻结定义、模式、数据及已覆盖探针；不等于盈利保证，也不能覆盖事后语义或发布日期缺失。"}
