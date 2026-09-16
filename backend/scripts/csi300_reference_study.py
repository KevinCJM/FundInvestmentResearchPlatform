"""Execute the fixed CSI300 reference study, without changing market data.

Default: isolated research stores under ignored frontend/test-results.
--save-to-system explicitly saves the same objects to the project's configured
research data directory. This never grants deployment/qualification authority.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from historical_regimes.v2_service import RegimeGraphV2Service, _json_safe
from historical_regimes.v2_templates import instantiate_template_v2
from historical_regimes.reliability.references import resolve_reference

REFERENCE_TEMPLATE = "market-trend-reference-csi300-v1"
MODEL_TEMPLATE = "csi300-maintrend-sma9-realtime-v3"
CALIBRATION_END = "2022-12-31"
PROJECT = Path(__file__).resolve().parents[2]


def _run(graph: RegimeGraphV2Service, draft: dict, mode: str, cutoff: str):
    saved = graph.create_definition(draft)
    prepared = graph.prepare(saved)
    run = graph.run_saved({"schema_version": "2.0", "id": saved["id"], "revision": saved["revision"]},
                          mode, cutoff, prepared["compile_token"])
    publication = graph.publish(run["id"], "research_display")["publication"]
    return saved, run, publication


def execute(
    workspace: Path,
    market: Path,
    output: Path,
    cutoff: str,
    model_draft: dict | None = None,
    *,
    adopt_calibration: bool = False,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    from historical_regimes.numba_kernels import warm_historical_regime_numba_kernels
    warm_historical_regime_numba_kernels()
    graph = RegimeGraphV2Service(workspace, market)
    draft = instantiate_template_v2(REFERENCE_TEMPLATE)
    draft["study"] = {"purpose": "historical_reference", "family": "market_trend"}
    historical, historical_run, publication = _run(graph, draft, "retrospective", cutoff)
    reference = {"run_id": historical_run["id"], "publication_id": publication["id"],
                 "content_hash": historical_run["content_hash"]}
    frozen, _ = resolve_reference(graph, reference)
    print(json.dumps({"stage": "reference", "name": historical["name"],
                      "count": len(frozen["series"]), "classes": dict(Counter(p["state_id"] for p in frozen["series"]))}, ensure_ascii=False), flush=True)
    quality_plan = graph.prepare(historical)
    quality_preview = graph.reference_quality.preview({
        "definition_id": historical["id"], "revision": historical["revision"],
        "mode": "retrospective", "as_of": cutoff,
        "compile_token": quality_plan["compile_token"],
        "policy": {"minimum_state_episodes_for_estimation": 3},
    })
    quality_artifact = graph.reference_quality.confirm({
        "request": quality_preview["request"], "preview_hash": quality_preview["preview_hash"]
    })
    realtime = copy.deepcopy(model_draft) if model_draft is not None else instantiate_template_v2(MODEL_TEMPLATE)
    realtime["study"]["reference"] = reference
    # Preserve the precise economic target/data release used by the reference.
    source = next(n for n in historical["graph"]["nodes"] if n["id"] == "market")
    realtime["graph"]["nodes"][0] = copy.deepcopy(source)
    realtime["evaluation_targets"] = copy.deepcopy(historical["evaluation_targets"])
    model, model_run, model_publication = _run(graph, realtime, "realtime", cutoff)
    print(json.dumps({"stage": "realtime", "name": model["name"], "id": model["id"]}, ensure_ascii=False), flush=True)
    request = {"definition_id": model["id"], "revision": model["revision"], "reference": reference,
               "policy": {"calibration_end": CALIBRATION_END, "test_end": cutoff,
                          "confidence_floor": .60, "minimum_state_episodes": 3,
                          "minimum_state_predictions": 5, "minimum_state_precision": .65}}
    preview = graph.reliability.preview(request)
    artifact = graph.reliability.confirm({"request": preview["request"], "preview_hash": preview["preview_hash"]})
    report = artifact["report"]
    recognition_evidence = graph.reliability.recognition_evidence(artifact["id"])
    calibrated_model = None
    if adopt_calibration:
        adopted_payload = copy.deepcopy(model)
        adopted_payload["study"]["calibration_id"] = artifact["id"]
        adopted = graph.update_definition(model["id"], model["revision"], adopted_payload)
        adopted_plan = graph.prepare(adopted)
        adopted_run = graph.run_saved(
            {"schema_version": "2.0", "id": adopted["id"], "revision": adopted["revision"]},
            "realtime",
            cutoff,
            adopted_plan["compile_token"],
        )
        adopted_publication = graph.publish(adopted_run["id"], "research_display")["publication"]
        calibrated_model = {
            "definition_id": adopted["id"],
            "revision": adopted["revision"],
            "run_id": adopted_run["id"],
            "publication_id": adopted_publication["id"],
            "calibration_id": artifact["id"],
            "deployment_eligible": False,
        }
        (output / "calibrated-model.json").write_text(json.dumps(adopted, ensure_ascii=False, indent=2))
    summary = {
        "reference": {**reference, "definition_id": historical["id"], "revision": historical["revision"], "name": historical["name"]},
        "model": {"definition_id": model["id"], "revision": model["revision"], "name": model["name"], "run_id": model_run["id"], "publication_id": model_publication["id"]},
        "report_id": artifact["id"], "calibrated_model": calibrated_model,
        "quality_report_id": quality_artifact["id"],
        "historical_conditional_estimation": quality_artifact["report"]["conditional_estimation"],
        "historical_state_evidence": quality_artifact["report"]["segments"]["per_state"],
        "verification": report.get("verification"), "recognition_evidence": recognition_evidence,
        "data_cutoff": cutoff, "status": report["status"],
        "sample": report["sample"], "classification": report["classification"],
        "intervals": report["intervals"], "transitions": report["transitions"],
        "calibration": report["calibration"],
        "holdout": report["probability"]["blocks"].get("holdout"),
        "confidence_interval": report["confidence_interval"], "stability": report["stability"],
        "latest_diagnostic": next((p for p in reversed(report["points"]) if p["predicted_state"] is not None), None),
    }
    (output / "summary.json").write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))
    (output / "report.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
    (output / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2))
    (output / "reference.json").write_text(json.dumps(historical, ensure_ascii=False, indent=2))
    print(json.dumps(_json_safe({k: v for k, v in summary.items() if k != "stability"}), ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-to-system", action="store_true")
    parser.add_argument("--selected", action="store_true", help="Use the frozen rolling-origin selection; never rerun search")
    parser.add_argument("--as-of", default="2026-09-03", help="Explicit experiment cutoff; not wall-clock freshness")
    args = parser.parse_args()
    output = PROJECT / "frontend/test-results/csi300-study" / ("system" if args.save_to_system else "isolated")
    market = PROJECT / "data"
    workspace = market if args.save_to_system else output / "store"
    selected_draft = None
    if args.selected:
        selection_path = PROJECT / "frontend/test-results/csi300-study/isolated/rolling-search.json"
        draft_path = PROJECT / "frontend/test-results/csi300-study/isolated/rolling-selected-draft.json"
        selection = json.loads(selection_path.read_text())
        if selection.get("selected") != "sma9-band0.040" or selection.get("confidence_floor") != .60:
            raise ValueError("Rolling-origin selection differs from the reviewed experiment")
        selected_draft = json.loads(draft_path.read_text())
        selected_draft["name"] = "沪深300主趋势 · 实时SMA9（4%缓冲）"
        selected_draft["description"] = (
            "闭合月末价格相对9月SMA高于4%为牛、低于-4%为熊，其余震荡。"
            "参数由三段扩展式时间验证筛选；每折校准只使用此前标签。"
            "只有通过状态级验证且达到校准置信门槛的判断才可授权实时Regime信号；"
            "实时识别不作为LTCMA输入，历史通过也不等于前瞻部署资格。"
        )
    execute(
        workspace,
        market,
        output,
        args.as_of,
        selected_draft,
        adopt_calibration=bool(args.save_to_system and args.selected),
    )
