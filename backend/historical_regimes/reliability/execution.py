"""Bounded causal replay through the sole existing graph executor."""

import copy
import time
from datetime import date
import numpy as np
from custom_indicators.errors import (
    ValidationError,
    ConflictError,
    IndicatorDomainError,
)
from ..v2_contracts import parse_definition_v2, definition_content_hash
from ..v2_service import _required_node_ids, _definition_output_frequency, _source_spec
from ..numba_kernels import validation_windows_kernel
from ..temporal_audit import audit_execution

LATENT = {"model.hmm", "model.gmm", "model.markov"}


def model_binding_hash(definition):
    from ..v2_service import _content_hash

    payload = definition.model_dump(
        mode="json", exclude={"id", "revision", "created_at", "updated_at"}
    )
    if payload.get("study"):
        payload["study"].pop("calibration_id", None)
        payload["study"].pop("qualification_id", None)
    return _content_hash(payload)


def target_identity(definition):
    primary = next((t for t in definition.evaluation_targets if t.primary), None)
    if primary is None and definition.evaluation_targets:
        primary = definition.evaluation_targets[0]
    if primary:
        spec = copy.deepcopy(primary.source)
    else:
        required = _required_node_ids(definition)
        sources = [
            n
            for n in definition.graph.nodes
            if n.id in required
            and n.type.startswith("source.")
            and n.type != "source.constant"
        ]
        if len(sources) != 1:
            raise ValidationError(
                "RELIABILITY_TARGET_REQUIRED", "多数据源图必须指定明确的评价对象。"
            )
        spec = _source_spec(sources[0].type, sources[0].parameters)
    # Keep source version, field and checksum. Cosmetic/range choices do not
    # identify an economic target; availability mode is checked separately.
    return {
        key: value
        for key, value in spec.items()
        if key
        not in {
            "name",
            "description",
            "start_date",
            "end_date",
            "as_of",
            "availability_mode",
            "frequency",
        }
    }


def probability_provenance(definition):
    nodes = {n.id: n for n in definition.graph.nodes}
    current = definition.graph.outputs["state"]
    path = []
    while True:
        node = nodes[current.node_id]
        path.append(node.id)
        if (
            node.type in {"post.confirmation", "post.confidence_gate"}
            and "state" in node.inputs
        ):
            current = node.inputs["state"]
        else:
            break
    probability = definition.graph.outputs.get("probabilities")
    verified = bool(
        node.type in LATENT
        and probability is not None
        and probability.node_id == node.id
        and probability.port == "probabilities"
        and current.port == "state"
    )
    # component_map/ensembles and unrelated diagnostic probabilities are not
    # inferred to be posterior probabilities. Final-state class-frequency is safe.
    return {
        "type": "model_posterior" if verified else "deterministic_state",
        "probability_node_id": node.id if verified else None,
        "final_state_path": path,
        "final_state_node_id": path[0],
        "temperature_supported": verified,
        "reason": (
            None
            if verified
            else "No verified posterior on the final state's unchanged class axis"
        ),
    }


def load_definition(graph, request):
    definition = parse_definition_v2(
        graph.get_definition(request.definition_id, request.revision)
    )
    if definition.study is None or definition.study.purpose != "realtime_recognition":
        raise ValidationError(
            "RELIABILITY_STUDY_REQUIRED", "验证需要已保存的实时识别 study 定义。"
        )
    if (
        definition.study.reference is None
        or definition.study.reference != request.reference
    ):
        raise ValidationError(
            "RELIABILITY_REFERENCE_MISMATCH", "请求参考与已保存模型绑定不一致。"
        )
    graph._validate_realtime_graph(definition, "realtime")
    graph._formal_source_gate(definition)
    return definition


def replay(
    graph,
    definition,
    request,
    reference,
    *,
    source_cache=None,
    frozen_folds=None,
    cutoff_override=None
):
    started = time.monotonic()
    cutoff = str(
        cutoff_override
        or request.policy.test_end
        or reference.get("as_of")
        or reference["series_summary"]["last_observation_date"]
    )
    if date.fromisoformat(cutoff) <= (
        request.policy.validation_end or request.policy.calibration_end
    ):
        raise ValidationError(
            "RELIABILITY_TIME_BLOCKS", "参考截止必须晚于校准/验证截止。"
        )
    rd = parse_definition_v2(reference["definition"])
    if _definition_output_frequency(definition) != reference["frequency"]:
        raise ValidationError(
            "RELIABILITY_FREQUENCY_MISMATCH", "仅支持同频率比较，不自动填充或跨频映射。"
        )
    if target_identity(definition) != target_identity(rd):
        raise ValidationError(
            "RELIABILITY_TARGET_MISMATCH",
            "参考与识别模型必须使用同一精确对象及数据版本。",
        )
    from .diagnostics import budget_definition, check_inputs

    required = budget_definition(definition)
    nodes = [n for n in definition.graph.nodes if n.id in required]
    latent = [n for n in nodes if n.type in LATENT]
    token = request.compile_token
    if not token:
        token = graph._plans_by_graph_hash.get(graph._preparation_hash(definition))
    plan = graph._validate_plan(definition, token)
    cache = {} if source_cache is None else source_cache
    check_inputs(graph, definition, "realtime", cutoff, cache)
    execution = graph._execute_graph(
        None, definition, "realtime", cutoff, plan=plan, source_cache=cache
    )
    series = execution["series"]
    if len(series) > 20000:
        raise ValidationError("RELIABILITY_BUDGET", "输出最多20000个观测。")
    temporal = audit_execution(graph, definition, "realtime", cutoff, plan, execution)
    temporal.pop("elapsed_ms", None)
    if (
        temporal.get("verified") is not True
        or not temporal.get("realtime_supported")
        or temporal.get("may_repaint")
        or temporal.get("semantic_hindsight")
    ):
        raise ValidationError(
            "RELIABILITY_PIT_REQUIRED", "实际输入未通过严格因果时点检查。"
        )
    folds = []
    if latent:
        _, windows = validation_windows_kernel(
            np.int64(len(series)), np.int64(definition.validation.get("folds", 4))
        )
        predictions = {}
        fold_cuts = (
            frozen_folds
            if frozen_folds is not None
            else [
                {
                    "train_end": int(w[0]),
                    "test_end": int(w[1]),
                    "training_as_of": series[int(w[0]) - 1]["data_available_at"],
                    "test_as_of": series[int(w[1]) - 1]["data_available_at"],
                }
                for w in windows
                if int(w[0]) > 0 and int(w[1]) > int(w[0])
            ]
        )
        for cut in fold_cuts:
            train_end, test_end = cut["train_end"], cut["test_end"]
            if time.monotonic() - started > 120:
                raise ConflictError("RELIABILITY_TIME_BUDGET", "验证超过时间预算。")
            train_end, test_end = int(train_end), int(test_end)
            if train_end < 1 or test_end <= train_end:
                continue
            training_at = cut["training_as_of"]
            stop = min(cut["test_as_of"], cutoff)
            if training_at >= stop:
                continue
            fold = {
                "training_as_of": training_at,
                "test_as_of": stop,
                "train_end": train_end,
                "test_end": test_end,
            }
            try:
                result = graph._execute_graph(
                    None,
                    definition,
                    "realtime",
                    stop,
                    plan=plan,
                    source_cache=cache,
                    latent_training_as_of=training_at,
                )
                for point in result["series"]:
                    if training_at < point["data_available_at"] <= stop:
                        predictions[point["observation_date"]] = point
                fold.update(
                    status="completed",
                    model_audits=result["result"]["diagnostics"]["model_audits"],
                )
            except IndicatorDomainError as exc:
                if exc.code != "INSUFFICIENT_WALK_FORWARD_MODEL_DATA":
                    raise
                fold.update(status="insufficient", reason=exc.code)
            folds.append(fold)
        # Preserve full time axis; training and unavailable fold outputs abstain.
        series = [
            predictions.get(
                p["observation_date"],
                {**p, "state_id": "unclassified", "probabilities": None},
            )
            for p in series
        ]
    if time.monotonic() - started > 120:
        raise ConflictError("RELIABILITY_TIME_BUDGET", "验证超过时间预算。")
    lineage = {
        "definition_hash": definition_content_hash(definition),
        "model_binding_hash": model_binding_hash(definition),
        "target_identity": target_identity(definition),
        "data_snapshots": execution["result"]["data_snapshots"],
        "evaluation_snapshot": execution["result"].get("evaluation_snapshot"),
        "temporal_audit": temporal,
        "folds": folds,
        "prediction_method": (
            "expanding_walk_forward" if latent else "fixed_rule_causal_replay"
        ),
        "probability_provenance": probability_provenance(definition),
        "execution_audit": execution["result"]["diagnostics"],
        "selection_history": "unknown; user model/reference selection may have seen this period",
    }
    return series, lineage, cutoff
