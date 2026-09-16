"""Bounded rolling-origin search for the CSI300 real-time market-trend model.

The historical reference stays fixed. Every fold fits only the class-frequency
calibrator on labels before that fold, then scores the next time block. This is
retrospective model research, not prospective deployment evidence.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from historical_regimes.numba_kernels import warm_historical_regime_numba_kernels
from historical_regimes.reliability import kernels as nk
from historical_regimes.reliability.references import resolve_reference
from historical_regimes.v2_contracts import parse_definition_v2
from historical_regimes.v2_numba import state_probabilities_kernel
from historical_regimes.v2_service import RegimeGraphV2Service, _json_safe
from historical_regimes.v2_templates import instantiate_template_v2

from backend.scripts.csi300_reference_study import PROJECT

MODEL_TEMPLATE = "csi300-maintrend-sma9-realtime-v2"
WINDOWS = (4, 5, 6, 7, 8, 9, 10, 11, 12)
BANDS = (.015, .02, .025, .03, .035, .04, .045, .05)
FOLDS = (
    ("2014-12-31", "2015-01-01", "2018-12-31"),
    ("2018-12-31", "2019-01-01", "2022-12-31"),
    ("2022-12-31", "2023-01-01", "2026-12-31"),
)
CONFIDENCE_FLOOR = .60
MIN_ACCEPTED_ACCURACY = .70
MIN_COVERAGE = .35


def candidates() -> list[tuple[str, dict]]:
    """Return only editable graphs composed from existing causal operators."""
    items: list[tuple[str, dict]] = []
    for window in WINDOWS:
        for band in BANDS:
            draft = instantiate_template_v2(MODEL_TEMPLATE)
            nodes = {node["id"]: node for node in draft["graph"]["nodes"]}
            nodes["average"]["parameters"]["window"] = window
            nodes["average"]["label"] = f"{window}月单边均线"
            nodes["upper"]["parameters"]["value"] = band
            nodes["lower"]["parameters"]["value"] = -band
            draft["name"] = f"沪深300主趋势 · 实时候选 SMA{window} / {band:.3f}"
            draft["description"] = "滚动时间折研究候选；只使用已闭合月份，历史参考不参与特征计算。"
            draft.pop("template_id", None)
            draft["graph"]["channel_metadata"]["average"]["label"] = nodes["average"]["label"]
            draft["graph"]["edges"] = []
            items.append((f"sma{window}-band{band:.3f}", draft))
    return items


def _block_metrics(y, pred, q, base, indices):
    confusion, _, classification = nk.classification_kernel(y[indices], pred[indices], 3)
    accepted = nk.decision_kernel(pred[indices], q[indices], CONFIDENCE_FLOOR)
    _, _, accepted_metrics = nk.classification_kernel(y[indices], accepted, 3)
    calibrated = nk.probability_kernel(y[indices], pred[indices], q[indices], 10)[0]
    class_base = nk.probability_kernel(
        y[indices], pred[indices], np.broadcast_to(base, (len(indices), 3)), 10
    )[0]
    return {
        "accuracy": float(classification[0]),
        "balanced_accuracy": float(classification[1]),
        "accepted_accuracy": (
            float(1.0 - accepted_metrics[4]) if np.isfinite(accepted_metrics[4]) else None
        ),
        "accepted_coverage": float(accepted_metrics[3]),
        "brier_improvement": float(class_base[1] - calibrated[1]),
        "confusion": confusion.tolist(),
    }


def _score_candidate(graph, draft, reference, cutoff, source_cache):
    states = reference["states"]
    codes = {state["id"]: index for index, state in enumerate(states)}
    dates = [point["observation_date"] for point in reference["series"]]
    y = np.asarray([codes.get(point.get("state_id"), -1) for point in reference["series"]], np.int64)
    model = parse_definition_v2(draft)
    plan = graph.prepare(draft)
    result = graph._execute_graph(
        None, model, "realtime", cutoff, plan=plan, source_cache=source_cache
    )
    by_date = {
        point["observation_date"]: codes.get(point.get("state_id"), -1)
        for point in result["series"]
    }
    pred = np.asarray([by_date.get(day, -1) for day in dates], np.int64)
    raw = state_probabilities_kernel(pred, np.int64(3))
    fold_results = []
    for calibration_end, validation_start, validation_end in FOLDS:
        train_end = max(index for index, day in enumerate(dates) if day <= calibration_end) + 1
        q, _, base, _ = nk.calibrate_kernel(y, pred, raw, train_end, 0)
        indices = np.asarray(
            [index for index, day in enumerate(dates) if validation_start <= day <= validation_end],
            np.int64,
        )
        fold_results.append(_block_metrics(y, pred, q, base, indices))
    accepted = [item["accepted_accuracy"] or 0.0 for item in fold_results]
    coverages = [item["accepted_coverage"] for item in fold_results]
    improvements = [item["brier_improvement"] for item in fold_results]
    minimum_accuracy = min(accepted)
    minimum_coverage = min(coverages)
    minimum_brier_improvement = min(improvements)
    eligible = (
        minimum_accuracy >= MIN_ACCEPTED_ACCURACY
        and minimum_coverage >= MIN_COVERAGE
        and minimum_brier_improvement > 0.0
    )
    utility = minimum_accuracy * float(np.sqrt(minimum_coverage)) if eligible else -1.0
    return {
        "folds": fold_results,
        "minimum_accepted_accuracy": minimum_accuracy,
        "minimum_accepted_coverage": minimum_coverage,
        "minimum_brier_improvement": minimum_brier_improvement,
        "cma_research_gate": eligible,
        "robust_utility": utility,
    }


def main() -> None:
    output = PROJECT / "frontend/test-results/csi300-study/isolated"
    summary = json.loads((output / "summary.json").read_text())
    reference_id = {key: summary["reference"][key] for key in ("run_id", "publication_id", "content_hash")}
    warm_historical_regime_numba_kernels()
    graph = RegimeGraphV2Service(output / "store", PROJECT / "data")
    reference, _ = resolve_reference(graph, reference_id)
    historic = reference["definition"]
    source = copy.deepcopy(next(node for node in historic["graph"]["nodes"] if node["id"] == "market"))
    ranking = []
    source_cache = {}
    for ordinal, (name, draft) in enumerate(candidates()):
        draft["graph"]["nodes"][0] = copy.deepcopy(source)
        draft["evaluation_targets"] = copy.deepcopy(historic["evaluation_targets"])
        draft["study"]["reference"] = reference_id
        score = _score_candidate(graph, draft, reference, summary["data_cutoff"], source_cache)
        ranking.append({"candidate": name, "ordinal": ordinal, **score, "definition": draft})
    eligible = [item for item in ranking if item["cma_research_gate"]]
    if not eligible:
        raise RuntimeError("No candidate passes the frozen CMA research gate")
    selected = max(
        eligible,
        key=lambda item: (
            item["robust_utility"],
            item["minimum_accepted_accuracy"],
            item["minimum_brier_improvement"],
            -item["ordinal"],
        ),
    )
    manifest = {
        "scope": "retrospective_rolling_origin_model_research_not_prospective_qualification",
        "reference": reference_id,
        "folds": FOLDS,
        "confidence_floor": CONFIDENCE_FLOOR,
        "gate": {
            "minimum_accepted_accuracy": MIN_ACCEPTED_ACCURACY,
            "minimum_accepted_coverage": MIN_COVERAGE,
            "all_folds_brier_improvement_positive": True,
        },
        "selection_criterion": "maximize min_accepted_accuracy * sqrt(min_accepted_coverage)",
        "selected": selected["candidate"],
        "selected_metrics": {key: value for key, value in selected.items() if key != "definition"},
        "ranking": [{key: value for key, value in item.items() if key != "definition"} for item in ranking],
    }
    (output / "rolling-search.json").write_text(
        json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2)
    )
    (output / "rolling-selected-draft.json").write_text(
        json.dumps(selected["definition"], ensure_ascii=False, indent=2)
    )
    print(json.dumps(_json_safe(manifest["selected_metrics"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
