"""Boundary alignment, report assembly; every numerical statistic uses NJIT."""

from bisect import bisect_right
from datetime import date, datetime, timedelta, timezone
import numpy as np
from custom_indicators.errors import ValidationError
from ..v2_numba import state_probabilities_kernel
from . import kernels as nk


def finite(value):
    return float(value) if np.isfinite(value) else None


def axis(points):
    dates = [str(p["observation_date"]) for p in points]
    for d in dates:
        if date.fromisoformat(d).isoformat() != d:
            raise ValidationError("RELIABILITY_DATE_AXIS", "日期必须为 ISO 日精度。")
    if dates != sorted(set(dates)):
        raise ValidationError("RELIABILITY_DATE_AXIS", "观察日期必须唯一且严格递增。")
    return dates, np.asarray(
        [date.fromisoformat(d).toordinal() for d in dates], dtype=np.int64
    )


def classification(y, p, states):
    cm, per, values = nk.classification_kernel(y, p, len(states))
    return {
        "confusion": cm.tolist(),
        "columns": [*states, "abstained"],
        "rows": states,
        "macro_class_set": "observed_reference_classes",
        "per_state": [
            {
                "state_id": state,
                **{
                    key: finite(per[i, j])
                    for j, key in enumerate(
                        ("support", "precision", "recall", "f1", "iou")
                    )
                },
            }
            for i, state in enumerate(states)
        ],
        **{
            key: finite(values[i])
            for i, key in enumerate(
                (
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "accepted_coverage",
                    "accepted_error",
                )
            )
        },
    }


def events(y, p, tolerance):
    counts, values = nk.events_kernel(y, p, tolerance)
    return (
        {
            **{
                key: int(counts[i])
                for i, key in enumerate(
                    (
                        "reference_segments",
                        "predicted_segments",
                        "matches",
                        "misses",
                        "false_events",
                        "complete_reference_segments",
                    )
                )
            },
            "equal_reference_segment_iou": finite(values[0]),
            "matching": "one-to-one greatest overlap first; unmatched reference segments score zero",
        },
        {
            **{
                key: int(counts[i + 6])
                for i, key in enumerate(
                    (
                        "reference_events",
                        "predicted_events",
                        "matches",
                        "misses",
                        "false_events",
                    )
                )
            },
            "delay_median": finite(values[1]),
            "delay_p90": finite(values[2]),
            "delay_unit": "observation_steps",
            "tolerance": tolerance,
            "matching": "same ordered state pair; nearest unused event; earlier date breaks ties",
        },
    )


def probability(y, p, q, bins):
    totals, buckets = nk.probability_kernel(y, p, q, bins)
    return {
        "samples": int(totals[0]),
        "brier": finite(totals[1]),
        "logloss": finite(totals[2]),
        "ece": finite(totals[3]),
        "log_epsilon": 1e-12,
        "reason": None if totals[0] > 0 else "no_valid_probability_samples",
        "bins": [
            {
                "index": i,
                "samples": int(b[0]),
                "mean_confidence": finite(b[1]),
                "match_rate": finite(b[2]),
            }
            for i, b in enumerate(buckets)
        ],
    }


def state_verification(y, accepted, states, policy, calibrated_brier, base_brier, scope):
    counts, metrics = nk.state_evidence_kernel(y, accepted, len(states))
    rows = []
    for index, state in enumerate(states):
        reference_observations = int(counts[index, 0])
        accepted_predictions = int(counts[index, 1])
        matches = int(counts[index, 2])
        complete_episodes = int(counts[index, 3])
        precision = finite(metrics[index, 0])
        recall = finite(metrics[index, 1])
        reasons = []
        if complete_episodes < policy.minimum_state_episodes:
            reasons.append("insufficient_independent_state_episodes")
        if accepted_predictions < policy.minimum_state_predictions:
            reasons.append("insufficient_accepted_state_predictions")
        if reasons:
            status = "insufficient_evidence"
        elif precision is not None and precision >= policy.minimum_state_precision:
            status = "verified"
        else:
            status = "failed"
            reasons.append("accepted_state_precision_below_policy")
        rows.append({
            "state_id": state,
            "status": status,
            "reference_observations": reference_observations,
            "independent_complete_episodes": complete_episodes,
            "accepted_predictions": accepted_predictions,
            "matches": matches,
            "precision": precision,
            "recall": recall,
            "reasons": reasons,
        })
    verified = [row["state_id"] for row in rows if row["status"] == "verified"]
    failed = [row["state_id"] for row in rows if row["status"] == "failed"]
    insufficient = [row["state_id"] for row in rows if row["status"] == "insufficient_evidence"]
    probability_improves_base = (
        calibrated_brier is not None
        and base_brier is not None
        and calibrated_brier < base_brier
    )
    if len(verified) == len(states):
        status = "verified"
    elif verified:
        status = "partially_verified"
    elif failed:
        status = "failed"
    else:
        status = "insufficient_evidence"
    reasons = []
    if not probability_improves_base:
        reasons.append("calibrated_probability_does_not_improve_class_base")
    if insufficient:
        reasons.append("some_states_have_insufficient_evidence")
    if failed:
        reasons.append("some_states_failed_precision_policy")
    return {
        "status": status,
        "scope": scope,
        "purpose": "recognition_state_evidence",
        "states": rows,
        "verified_states": verified,
        "fallback_states": [state for state in states if state not in verified],
        "probability_improves_class_base": probability_improves_base,
        "recognition_ready": bool(verified and probability_improves_base),
        "production_eligible": False,
        "reasons": reasons,
        "policy": {
            "minimum_state_episodes": policy.minimum_state_episodes,
            "minimum_state_predictions": policy.minimum_state_predictions,
            "minimum_state_precision": policy.minimum_state_precision,
            "confidence_floor": policy.confidence_floor,
        },
        "unverified_state_policy": "do_not_authorize_unverified_states",
    }


def build_report(
    definition, reference, publication, predictions, lineage, policy, cutoff
):
    nk.audit()
    reference_points = [
        p for p in reference["series"] if p["observation_date"] <= cutoff
    ]
    if len(reference_points) > 20000:
        raise ValidationError("RELIABILITY_BUDGET", "参考最多20000条观测。")
    dates, rd = axis(reference_points)
    prediction_dates, pd = axis(predictions)
    indices = nk.align_kernel(rd, pd)
    states = [s["id"] for s in reference["states"]]
    model_states = [s.id for s in definition.states]
    mapping = definition.study.state_mapping
    if mapping is None:
        if set(model_states) != set(states):
            raise ValidationError(
                "RELIABILITY_STATE_MAPPING_REQUIRED", "不同状态ID必须显式映射。"
            )
        mapping = {s: s for s in model_states}
    if set(mapping) != set(model_states) or not set(mapping.values()).issubset(states):
        raise ValidationError(
            "RELIABILITY_STATE_MAPPING", "状态映射不属于精确参考状态轴。"
        )
    codes = {s: i for i, s in enumerate(states)}
    y = np.asarray(
        [codes.get(p.get("state_id"), -1) for p in reference_points], np.int64
    )
    aligned = [predictions[i] if i >= 0 else {} for i in indices]
    pred = np.asarray(
        [codes.get(mapping.get(p.get("state_id")), -1) for p in aligned], np.int64
    )
    method = policy.calibration_method
    genuine = lineage["probability_provenance"]["temperature_supported"]
    if method == "temperature" and not genuine:
        raise ValidationError(
            "RELIABILITY_TEMPERATURE_SOURCE",
            "确定性状态或未验证的概率来源不能使用temperature。",
        )
    method = (
        ("temperature" if genuine else "class_frequency")
        if method == "auto"
        else method
    )
    if genuine:
        raw_source = np.asarray(
            [
                [
                    (
                        p.get("probabilities", {}).get(s, np.nan)
                        if isinstance(p.get("probabilities"), dict)
                        else np.nan
                    )
                    for s in model_states
                ]
                for p in aligned
            ],
            dtype=np.float64,
        ).reshape((len(y), len(model_states)))
        raw = nk.map_probability_kernel(
            raw_source,
            np.asarray([codes[mapping[s]] for s in model_states], np.int64),
            len(states),
        )
    else:
        raw = state_probabilities_kernel(pred, np.int64(len(states)))
    # A prediction with a later knowledge date cannot train an earlier block.
    # Keep the sample on the reference axis as abstained, never borrow it back.
    for i, p in enumerate(aligned):
        block_end = (
            policy.calibration_end.isoformat()
            if dates[i] <= policy.calibration_end.isoformat()
            else (
                policy.validation_end.isoformat()
                if policy.validation_end
                and dates[i] <= policy.validation_end.isoformat()
                else cutoff
            )
        )
        if (
            p
            and max(
                str(p.get("data_available_at") or cutoff),
                str(p.get("recognized_at") or cutoff),
            )
            > block_end
        ):
            pred[i] = -1
    train_end = bisect_right(dates, policy.calibration_end.isoformat())
    validation_end = (
        bisect_right(dates, policy.validation_end.isoformat())
        if policy.validation_end
        else train_end
    )
    blocks = [("calibration", 0, train_end)]
    if policy.validation_end:
        blocks.append(("validation", train_end, validation_end))
    blocks.append(
        ("test" if policy.validation_end else "holdout", validation_end, len(y))
    )
    samples = {}
    sufficient = True
    for name, start, end in blocks:
        c = classification(y[start:end], pred[start:end], states)
        intervals, _ = events(
            y[start:end], pred[start:end], policy.transition_tolerance
        )
        support = [int(row["support"]) for row in c["per_state"]]
        valid_count = sum(
            support
        )  # integer metadata aggregation, not numeric data path
        ok = (
            valid_count >= policy.minimum_samples
            and min(support) >= policy.minimum_class_samples
            and intervals["complete_reference_segments"] >= policy.minimum_segments
        )
        samples[name] = {
            "start": dates[start] if start < end else None,
            "end": dates[end - 1] if start < end else None,
            "samples": valid_count,
            "per_class": dict(zip(states, support)),
            "complete_segments": intervals["complete_reference_segments"],
            "sufficient": ok,
        }
        sufficient = sufficient and ok
    q, counts, base, temperature = nk.calibrate_kernel(
        y, pred, raw, train_end, 1 if method == "temperature" else 0
    )
    usable_pairs, usable_support, usable_sufficient = nk.calibration_support_kernel(
        y,
        pred,
        raw,
        train_end,
        1 if method == "temperature" else 0,
        policy.minimum_samples,
        policy.minimum_class_samples,
    )
    samples["calibration"].update(
        usable_prediction_pairs=int(usable_pairs),
        usable_reference_classes=dict(zip(states, usable_support[0].tolist())),
        usable_prediction_classes=dict(zip(states, usable_support[1].tolist())),
    )
    samples["calibration"]["sufficient"] = bool(
        samples["calibration"]["sufficient"] and usable_sufficient
    )
    sufficient = sufficient and bool(usable_sufficient)
    fitted = samples["calibration"]["sufficient"] and (
        method != "temperature" or np.isfinite(temperature)
    )
    if not fitted:
        q = np.full(raw.shape, np.nan)
    block_metrics = {}
    base_matrix = np.broadcast_to(base, raw.shape)  # readonly zero-copy stride view
    for name, start, end in blocks:
        block_metrics[name] = {
            "raw": probability(
                y[start:end], pred[start:end], raw[start:end], policy.bins
            ),
            "calibrated": probability(
                y[start:end], pred[start:end], q[start:end], policy.bins
            ),
            "class_base": probability(
                y[start:end], pred[start:end], base_matrix[start:end], policy.bins
            ),
            "classification": classification(y[start:end], pred[start:end], states),
        }
    # These immutable retrospective snapshots do not establish historical label
    # availability before their actual publication, even if observation dates precede it.
    label_known = max(
        [
            str(publication["published_at"])[:10],
            *[
                str(p.get("recognized_at") or p.get("data_available_at") or cutoff)[:10]
                for p in reference["series"]
            ],
        ]
    )
    retrospective = label_known > policy.calibration_end.isoformat()
    reasons = []
    if retrospective:
        reasons.append("reference_labels_unavailable_at_calibration_end")
    if not sufficient:
        reasons.append("insufficient_samples_classes_or_complete_segments")
    if not usable_sufficient:
        reasons.append("insufficient_usable_calibration_predictions")
    last_metrics = block_metrics[blocks[-1][0]]
    cal_brier = last_metrics["calibrated"]["brier"]
    base_brier = last_metrics["class_base"]["brier"]
    if cal_brier is None or base_brier is None or cal_brier >= base_brier:
        reasons.append("independent_holdout_does_not_improve_class_base_brier")
    # No selection-history ledger exists yet; never manufacture deployment evidence.
    reasons.append("model_and_reference_selection_history_unverified")
    now = datetime.now(timezone.utc).date()
    available = max((now + timedelta(days=1)).isoformat(), label_known, cutoff)
    calibration = {
        "method": method,
        "evidence_type": (
            "model_reference_probability"
            if method == "temperature"
            else "class_average"
        ),
        "parameters": {
            "temperature": finite(temperature),
            "counts": counts.tolist(),
            "class_base": [finite(v) for v in base],
        },
        "fitted": bool(fitted),
        "reason": None if fitted else "insufficient_calibration_evidence",
        "calibration_end": policy.calibration_end.isoformat(),
        "validation_end": (
            policy.validation_end.isoformat() if policy.validation_end else None
        ),
        "test_end": cutoff,
        "label_known_at": label_known,
        "deployment_eligible": False,
        "reasons": reasons,
        "available_from": available,
        "expires_on": (date.fromisoformat(available) + timedelta(days=90)).isoformat(),
        "confidence_floor": policy.confidence_floor,
        "split_kind": (
            "calibration_validation_final_test"
            if policy.validation_end
            else "calibration_holdout_no_tuning"
        ),
    }
    from .diagnostic_kernels import evidence_kernel

    evidence = evidence_kernel(pred, raw) if genuine else None
    from .bootstrap import build_intervals

    scope, holdout_start, holdout_end = blocks[-1]
    accepted = nk.decision_kernel(pred, q, policy.confidence_floor)
    verification = state_verification(
        y[holdout_start:holdout_end],
        accepted[holdout_start:holdout_end],
        states,
        policy,
        cal_brier,
        base_brier,
        scope,
    )
    confidence_interval = build_intervals(
        y[holdout_start:holdout_end],
        pred[holdout_start:holdout_end],
        q[holdout_start:holdout_end],
        base,
        policy.bootstrap,
        policy.confidence_floor,
        scope,
    )
    points = []
    for i, d in enumerate(dates):
        chosen = int(pred[i])
        confidence = finite(q[i, chosen]) if chosen >= 0 else None
        raw_dict = (
            {s: finite(raw[i, c]) for c, s in enumerate(states)}
            if chosen >= 0
            else None
        )
        calibrated = (
            {s: finite(q[i, c]) for c, s in enumerate(states)}
            if confidence is not None
            else None
        )
        points.append(
            {
                "observation_date": d,
                "reference_state": states[y[i]] if y[i] >= 0 else None,
                "predicted_state": states[chosen] if chosen >= 0 else None,
                "data_available_at": aligned[i].get("data_available_at"),
                "recognized_at": aligned[i].get("recognized_at"),
                "block": next(name for name, start, end in blocks if start <= i < end),
                "probability_evidence": (
                    None
                    if evidence is None or not np.isfinite(evidence[i, 0])
                    else {
                        **{
                            key: finite(evidence[i, j])
                            for j, key in enumerate(
                                (
                                    "selected_probability",
                                    "top_probability",
                                    "second_probability",
                                    "margin",
                                    "entropy",
                                )
                            )
                        },
                        "entropy_unit": "nats",
                    }
                ),
                "raw_probabilities": raw_dict,
                "calibrated_probabilities": calibrated,
                "calibrated_confidence": confidence,
                "decision_status": (
                    "abstained"
                    if chosen < 0
                    else (
                        "uncalibrated"
                        if confidence is None
                        else (
                            "below_floor"
                            if confidence < policy.confidence_floor
                            else "diagnostic_only"
                        )
                    )
                ),
            }
        )
    intervals, transitions = events(y, pred, policy.transition_tolerance)
    counts_summary = nk.sample_kernel(y, pred, indices)
    return {
        "schema_version": "1.0",
        "policy_version": "reference-confidence/1",
        "status": "retrospective_only" if retrospective else "insufficient_evidence",
        "states": reference["states"],
        "warnings": [
            "Reference agreement is not ground truth.",
            "Dependent daily observations are not independent regimes.",
            "Retrospective calibration is diagnostic, never a historically deployable probability.",
            *reasons,
        ],
        "sample": {
            "input": len(reference_points),
            "matched": int(counts_summary[0]),
            "unknown_reference": int(counts_summary[1]),
            "prediction_abstentions": int(counts_summary[2]),
            "missing_prediction": int(counts_summary[3]),
            "invalid_prediction_labels": sum(
                bool(p) and p.get("state_id") not in {*model_states, "unclassified"}
                for p in aligned
            ),
            "excluded_dates": {
                "reference_after_cutoff": len(reference["series"])
                - len(reference_points),
                "prediction_without_reference": len(set(prediction_dates) - set(dates)),
            },
            "blocks": samples,
        },
        "classification": classification(y, pred, states),
        "selective_classification": classification(y, accepted, states),
        "intervals": intervals,
        "transitions": transitions,
        "verification": verification,
        "probability": {
            "raw_type": lineage["probability_provenance"]["type"],
            "blocks": block_metrics,
        },
        "calibration": calibration,
        "lineage": {
            **lineage,
            "state_mapping": mapping,
            "reference_content_hash": reference["content_hash"],
            "reference_publication": publication,
        },
        "stability": {
            "status": "causal_probes_executed",
            "temporal_audit": lineage["temporal_audit"],
            "parameter_sensitivity": {"status": "not_executed"},
        },
        "confidence_interval": confidence_interval,
        "execution": nk.audit(),
        "points": points,
    }
