"""Retain a saved TAA decision and enforce its class budgets downstream."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from backend.custom_indicators.errors import ValidationError
from backend.tactical_allocation.numeric import aggregate_class_weights
from backend.tactical_allocation.data import TacticalAllocationData
from backend.tactical_allocation.repository import TacticalAllocationRepository


def validate_decision_application(decision: dict, data: TacticalAllocationData, frozen_returns=None) -> None:
    """The same application gate protects export and direct portfolio writes."""
    preview = decision["preview"]
    baseline = preview["baseline"]
    request, recommendation = preview["request"], preview["recommendation"]
    try:
        expires = date.fromisoformat(recommendation["expires_on"])
        as_of = date.fromisoformat(request["as_of"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError("TAA_DECISION_DATES", "决策缺少有效研究日或复核日，请重新研究。") from exc
    if as_of > date.today() or expires < date.today() or expires < as_of:
        raise ValidationError("TAA_DECISION_EXPIRED", "该决策已到复核日或研究日无效，请以最新数据重新研究后应用。")
    if request.get("signal_mode") in {"momentum", "regime"} and recommendation.get("is_saa") is not True:
        try:
            signal_day = date.fromisoformat(recommendation["signal_date"][:10])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("TAA_SIGNAL_DATE_MISSING", "偏离决策缺少信号日期，请重新研究。") from exc
        if (date.today() - signal_day).days > request["max_signal_age_days"]:
            raise ValidationError("TAA_SIGNAL_EXPIRED", "该偏离使用的信号已经过期，请使用最新数据重新研究。")
    current_turnover = recommendation.get("turnover_from_current")
    if current_turnover is not None and (not np.isfinite(current_turnover)
            or current_turnover > request["max_turnover"] + 1e-8):
        raise ValidationError("TAA_CURRENT_TURNOVER_LIMIT", "从当前持仓调整至目标的单次换手超过限制，请重新研究。")
    selected = next((item for item in preview.get("candidates", []) if item["id"] == preview.get("selected_id")), None)
    if selected is None:
        raise ValidationError("TAA_CANDIDATE_MISSING", "决策缺少已核验的候选记录，请重新比较后保存。")
    if selected.get("strength") != 0 and selected.get("validation_feasible") is not True:
        raise ValidationError("TAA_VALIDATION_LIMIT", "所选偏离在留出区未满足风险或换手限制，请重新研究或明确选择 SAA 基线。")
    if frozen_returns is not None:
        from backend.research_input_checks import return_quality
        dates = [row["date"] for row in preview["weight_path"]]
        quality = return_quality(frozen_returns, dates, [item["id"] for item in baseline["assets"]])
        if quality["issues"]:
            raise ValidationError("TAA_NAV_SCALE_BREAK", quality["issues"][0]["message"], diagnostics=quality["issues"])
    elif preview.get("data", {}).get("quality", {}).get("status") == "blocked":
        raise ValidationError("TAA_NAV_SCALE_BREAK", "研究包含未解决的净值数量级断点，请检查产品后重新研究。")
    data.validate_application(baseline)


def validate_allocation_source(source: Any, components: list[dict], strategy: dict,
                               universe_id: str, workspace: Path) -> dict | None:
    if source is None:
        return None
    if not isinstance(source, dict) or source.get("kind") != "taa" or not source.get("decision_id"):
        raise ValidationError("TAA_SOURCE_INVALID", "大类预算来源无效，请从 TAA 重新应用。")
    repository = TacticalAllocationRepository(Path(os.getenv("TACTICAL_ALLOCATION_DATA_DIR", str(workspace))))
    decision = repository.get_decision(source["decision_id"])
    preview = decision["preview"]
    baseline = preview["baseline"]
    validate_decision_application(decision, TacticalAllocationData(workspace, universe_dir=workspace), repository.decision_arrays(decision["id"])["returns"])
    if universe_id != baseline.get("universe_snapshot_id"):
        raise ValidationError("TAA_UNIVERSE_MISMATCH", "产品组合须使用 TAA 锁定的同一可投资域。")
    if strategy["type"] != "manual":
        raise ValidationError("TAA_CLASS_BUDGET_METHOD", "已承接 TAA 大类预算，请在各大类内调整产品权重；切换全组合优化会改变大类预算。")
    assets = [item["id"] for item in baseline["assets"]]
    product_classes: dict[tuple[str, str], str] = {}
    for asset in baseline["assets"]:
        for product in asset.get("products", []):
            key = (str(product["kind"]).lower(), str(product["product_id"]).upper())
            if key in product_classes and product_classes[key] != asset["id"]:
                raise ValidationError("TAA_DUPLICATE_PRODUCT_CLASS", "基线同一产品跨大类出现，须先在 SAA 明确唯一归属。")
            product_classes[key] = asset["id"]
    indices = []
    for component in components:
        class_id = component.get("asset_class_id")
        if class_id not in assets:
            raise ValidationError("TAA_CLASS_UNKNOWN", "产品的大类归属不在 TAA 预算中，请重新指定。")
        key = (str(component.get("kind") or "").lower(), str(component.get("product_id") or "").upper())
        if product_classes.get(key) != class_id:
            raise ValidationError("TAA_PRODUCT_CLASS_MISMATCH", "产品与 SAA 冻结的大类归属不一致；新增或替换产品须先在 SAA 确认归属并重新应用。")
        indices.append(assets.index(class_id))
    totals = aggregate_class_weights(np.asarray(strategy["weights"], dtype=np.float64),
                                     np.asarray(indices, dtype=np.int64), len(assets))
    expected = preview["recommendation"]["weights"]
    for index, asset in enumerate(assets):
        if abs(float(totals[index]) - expected[asset]) > 1e-8:
            raise ValidationError("TAA_CLASS_BUDGET_CHANGED", f"{asset} 的产品权重合计与 TAA 预算不一致，请在类内调整，或回到 TAA 重定预算。")
    return {"kind": "taa", "decision_id": decision["id"], "decision_hash": decision["content_hash"],
            "baseline_id": baseline["id"], "baseline_hash": baseline["content_hash"],
            "class_weights": expected, "as_of": preview["request"]["as_of"],
            "expires_on": preview["recommendation"]["expires_on"], "research_only": True,
            "product_backtest_policy": "static_target_historical_replay_not_dynamic_taa"}
