"""SAA-linked tactical research orchestration with immutable decisions."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np

from backend.custom_indicators.errors import ConflictError, ValidationError
from backend.tactical_allocation.contracts import PreviewRequest, ScenarioRequest, SaveDecisionRequest
from backend.research_input_checks import return_quality, training_readiness_kernel
from backend.tactical_allocation.data import TacticalAllocationData, warm_tactical_data
from backend.tactical_allocation.repository import TacticalAllocationRepository
from backend.tactical_allocation import numeric


def _day(value: str | date) -> int:
    return (date.fromisoformat(str(value)[:10]) - date(1970, 1, 1)).days


def _vector(values: dict[str, float], assets: list[str], label: str) -> np.ndarray:
    if set(values) != set(assets):
        raise ValidationError("TAA_ASSET_AXIS_MISMATCH", f"{label}必须逐项覆盖相同资产，不能缺项或混入其他资产。")
    result = np.asarray([values[key] for key in assets], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValidationError("TAA_NON_FINITE", f"{label}必须是有限数值。")
    return result


class TacticalAllocationService:
    def __init__(self, root: Path, data_dir: Path, regime_resolver: Callable | None = None, universe_dir: Path | None = None):
        self.repository = TacticalAllocationRepository(root)
        self.data = TacticalAllocationData(data_dir, universe_dir=universe_dir or root)
        self.regime_resolver = regime_resolver

    def warm(self) -> dict[str, Any]:
        warm_tactical_data()
        audit = numeric.warm_tactical_allocation_kernels()
        from backend.tactical_allocation.walk_forward import warm_walk_forward_kernels
        audit["walk_forward"] = warm_walk_forward_kernels()
        if not audit["walk_forward"]["complete"]:
            raise RuntimeError("多段样本外验证预热未完成")
        return audit

    def catalog(self) -> dict[str, Any]:
        catalog = self.data.catalog()
        return {**catalog, "baselines": self.repository.list_baselines(),
                "decisions": self.repository.list_decisions()}

    def create_baseline(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.repository.save_baseline(self.data.create_baseline(request))

    def _preflight_data(self, request: PreviewRequest, baseline: dict, data: dict, signals: dict | None = None) -> dict:
        assets = [item["id"] for item in baseline["assets"]]
        days = np.asarray([_day(value) for value in data["dates"]], dtype=np.int64)
        future, unknown, earliest, suggested = training_readiness_kernel(data["available_at"], days, _day(request.train_end_date))
        split = next((i for i, value in enumerate(data["dates"]) if value > str(request.train_end_date)), len(days))
        enough = split >= 20 and len(days) - split >= 20
        quality = return_quality(data["returns"], data["dates"], assets)
        reasons, guidance = [], []
        signal_counts = None
        if request.signal_mode in ("momentum", "composite"):
            signals = signals if signals is not None else self._signals(request, data, assets)
            flags = signals["use_signal"]
            if request.decision_policy:
                from .clocks import clock_plan, decision_signal_flags_kernel
                flags = decision_signal_flags_kernel(flags, clock_plan(data["period_starts"], request.decision_policy)["decisions"])
            counts = numeric.signal_counts_kernel(flags, signals["knowledge_verified"], split)
            signal_counts = dict(zip(("train_signal_observations", "validation_signal_observations",
                                      "train_known_observations", "validation_known_observations"), map(int, counts)))
        strict_unknown = bool(unknown) and (request.decision_policy is not None or request.signal_mode == "composite")
        no_signals = signal_counts is not None and signal_counts["train_signal_observations"] == 0
        if no_signals:
            reasons.append("训练期没有有效趋势信号，各档强度无法形成有效比较，不能据此选出最优偏离。请检查观察窗口、信号有效期与数据公布日期。")
        if not enough:
            reasons.append(f"训练区 {split} 条、验证区 {len(days) - split} 条共同收益；各至少需要 20 条。")
        if future:
            reasons.append(f"{future} 个训练收益值在训练截止日尚不可得，不能用于选择偏离强度。")
        if unknown:
            reasons.append(f"{unknown} 个训练收益值缺少可得时间，仅可作研究，不能认证历史 PIT。")
        if (future or not enough) and suggested >= 0:
            suggested_date = str(date(1970, 1, 1) + timedelta(days=int(suggested)))
            guidance.append({"code": "TRAINING_BOUNDARY", "message": f"按数据可得时间将训练截止日调整为 {suggested_date}。",
                             "action": "adjust_dates", "patch": {"train_end_date": suggested_date}})
        if (future or no_signals) and enough:
            guidance.append({"code": "FIXED_HYPOTHESIS", "message": "改为固定假设比较：不搜索强度，保留历史回放限制。",
                             "action": "fixed_comparison", "patch": {"search": False}})
        if quality["issues"]:
            guidance.append({"code": "NAV_SCALE_BREAK", "message": quality["issues"][0]["message"], "action": "review_data"})
        return {"coverage": {"start_date": data["period_starts"][0], "end_date": data["dates"][-1]},
                "dates": {key: str(getattr(request, key)) for key in ("start_date", "end_date", "as_of", "train_end_date")},
                "quality": quality,
                "training": {"eligible": enough and not bool(future) and not strict_unknown and not no_signals, "train_observations": split,
                             "validation_observations": len(days) - split, "unavailable_count": int(future),
                             "unknown_count": int(unknown), **(signal_counts or {}),
                             "earliest_available_date": str(date(1970, 1, 1) + timedelta(days=int(earliest))) if earliest >= 0 else None,
                             "reasons": reasons}, "guidance": guidance, "pit": data["pit"],
                "can_calculate": enough and quality["status"] == "clear" and (not request.search or (not future and not strict_unknown and not no_signals))}

    def preflight(self, request: PreviewRequest) -> dict:
        numeric._require_ready()
        baseline = self.repository.get_baseline(request.baseline_id)
        data = self.data.load_data(baseline, str(request.start_date), str(request.end_date), str(request.as_of))
        return self._preflight_data(request, baseline, data)

    def _signals(self, request: PreviewRequest, data: dict, assets: list[str]) -> dict:
        returns = data["returns"]
        starts = np.asarray([_day(value) for value in data["period_starts"]], dtype=np.int64)
        if request.signal_mode == "composite":
            from .signals import build_composite
            return build_composite(request, data, assets)
        if request.signal_mode == "momentum":
            signals = numeric.build_momentum_signals(
                returns, request.lookback, request.max_abs_tilt,
                available_at=data["available_at"], period_starts=starts,
                as_of_day=_day(request.as_of),
                period_ends=np.asarray([_day(value) for value in data["dates"]], dtype=np.int64),
                max_signal_age_days=request.max_signal_age_days,
            )
            timing = []
            for index, row in enumerate(signals["windows"]):
                first, end, known_day, lag, status = map(int, row)
                timing.append({"period_start": data["period_starts"][index] if index < len(returns) else str(request.as_of),
                               "window_start": data["period_starts"][first] if first >= 0 else None,
                               "window_end": data["dates"][end - 1] if end >= 0 else None,
                               "available_at": str(date(1970, 1, 1) + timedelta(days=known_day)) if known_day >= 0 else None,
                               "lag_days": lag if lag >= 0 else None,
                               "status": {0: "warmup", 1: "not_available", 2: "expired", 3: "available"}[status],
                               "active": bool(signals["use_signal"][index] if index < len(returns) else signals["current_use_signal"])})
            signals["audit"] = {"method": "latest_available_contiguous_momentum", "algorithm_version": numeric.MOMENTUM_ALGORITHM_VERSION,
                                "lookback": request.lookback, "max_signal_age_days": request.max_signal_age_days,
                                "signal_rule": "latest contiguous complete window with every endpoint known by period start; unknown is unavailable",
                                "knowledge_verified": signals["knowledge_verified"].tolist(), "signal_timing": timing}
            signals["current_date"] = timing[-1]["window_end"]
            signals["current_timing"] = timing[-1]
            signals["confidence"] = None
            signals["fallback_reason"] = None if signals["current_use_signal"] else (
                "最近已知窗口超过信号有效期，当前建议回归 SAA。" if timing[-1]["status"] == "expired" else
                "已知窗口内各资产趋势相同，当前保持 SAA。" if timing[-1]["status"] == "available" else
                "没有完整且已可得的观察窗口，当前保持 SAA。")
            return signals
        if request.signal_mode == "manual":
            tilt = _vector(request.manual_tilts, assets, "人工偏离")
            if abs(float(tilt.sum())) > 1e-8:
                raise ValidationError("TAA_TILTS_NOT_ZERO_SUM", "偏离合计必须为 0；增配需要明确减配来源。")
            return {"probabilities": np.ones((len(returns), 1)), "use_signal": np.ones(len(returns), dtype=np.uint8),
                    "state_tilts": tilt.reshape(1, -1), "current_probabilities": np.ones(1),
                    "current_use_signal": 1, "current_date": str(request.as_of), "confidence": None,
                    "fallback_reason": None,
                    "audit": {"method": "manual_hypothesis_replay", "historically_deployed": False}}
        if self.regime_resolver is None:
            raise ValidationError("TAA_REGIME_UNAVAILABLE", "市场状态服务不可用，请稍后重试。")
        run, gate = self.regime_resolver(request.regime_run_id)
        if gate.get("passed") is False:
            raise ValidationError("TAA_RUN_NOT_PUBLISHED", "历史情景运行必须先发布到 TAA 或正式回测。", "run_id")
        # Resolve the immutable analytical run through the same server gate as the
        # compatibility backtest. Metadata alignment is an input-boundary operation.
        from historical_regimes.reliability.consumer import calibrated_output, consumer_states, allocation_probabilities
        states = consumer_states(run)
        if set(request.state_tilts) != set(states):
            raise ValidationError("TAA_STATE_AXIS_MISMATCH", "请为每个已发布状态设置一组偏离。")
        tilts = np.asarray([_vector(request.state_tilts[state], assets, "状态偏离") for state in states])
        if any(abs(float(row.sum())) > 1e-8 for row in tilts):
            raise ValidationError("TAA_TILTS_NOT_ZERO_SUM", "每个市场状态的偏离合计必须为 0。")
        qualification = (run.get("_reliability") or {}).get("qualification")
        qualified = qualification.get("qualified_states") if qualification else None
        if qualified is not None:
            for state_index, state in enumerate(states):
                if state not in qualified:
                    tilts[state_index] = 0.0
        from historical_regimes.taa import _regime_points
        points = _regime_points(run)
        axes = list(data["period_starts"]) + [str(request.as_of)]
        probabilities = np.zeros((len(axes), len(states)))
        use = np.zeros(len(axes), dtype=np.uint8)
        audit_rows = []
        point_index = 0
        latest = None
        for index, start in enumerate(axes):
            while point_index < len(points) and points[point_index]["effective_date"] <= start:
                latest = points[point_index]
                point_index += 1
            reason = "没有可用状态"
            confidence = None
            probs = None
            allocation = None
            if latest:
                probs, confidence, invalid_reason = calibrated_output(run, latest, start, states)
                if max(latest["observation_date"], latest["recognized_at"], latest["effective_date"]) > start:
                    reason = "状态在本期开始时尚不可得"
                elif _day(start) - _day(latest["effective_date"]) > request.max_signal_age_days:
                    reason = "市场状态已过期"
                elif invalid_reason and (run.get("definition") or {}).get("study"):
                    reason = invalid_reason
                elif confidence is None or not np.isfinite(confidence) or not request.confidence_floor <= confidence <= 1:
                    reason = "状态置信度不足"
                elif invalid_reason:
                    reason = invalid_reason
                else:
                    allocation = allocation_probabilities(run, probs)
                    # Keep the numerical kernel's normalized distribution contract;
                    # zero tilts leave unverified probability mass in the baseline.
                    probabilities[index] = [probs[state] for state in states]
                    use[index] = 1
                    reason = None
            audit_rows.append({"period_start": start, "signal_date": latest["effective_date"] if latest else None,
                               "recognized_at": latest["recognized_at"] if latest else None,
                               "fallback_reason": reason, "confidence": confidence,
                               "probabilities": probs, "allocation_probabilities": allocation})
        return {"probabilities": probabilities[:-1], "use_signal": use[:-1], "state_tilts": tilts,
                "current_probabilities": probabilities[-1], "current_use_signal": int(use[-1]),
                "current_date": audit_rows[-1]["signal_date"], "confidence": audit_rows[-1]["confidence"],
                "fallback_reason": audit_rows[-1]["fallback_reason"],
                "audit": {"method": "published_regime", "gate": gate, "states": states,
                          "signal_timing": audit_rows, "run_created_at": run.get("created_at"),
                          "publications": run.get("publications", [])}}

    def _calculate(self, request: PreviewRequest) -> tuple[dict, dict]:
        numeric._require_ready()
        baseline = self.repository.get_baseline(request.baseline_id)
        if baseline.get("policy") and request.max_tracking_error > baseline["policy"]["mandate"]["max_tracking_error"] + 1e-10:
            raise ValidationError("SAA_POLICY_TRACKING_ERROR", "战术主动风险上限不得超过已确认政策预算；需要扩大时请回长期配置重新研究。")
        data = self.data.load_data(baseline, str(request.start_date), str(request.end_date), str(request.as_of))
        signals = self._signals(request, data, [item["id"] for item in baseline["assets"]]) if request.signal_mode in ("momentum", "composite") else None
        preflight = self._preflight_data(request, baseline, data, signals)
        if preflight["quality"]["issues"]:
            raise ValidationError("TAA_NAV_SCALE_BREAK", preflight["quality"]["issues"][0]["message"], diagnostics=preflight["quality"]["issues"])
        assets = [item["id"] for item in baseline["assets"]]
        base = np.asarray([item["base_weight"] for item in baseline["assets"]], dtype=np.float64)
        lower = np.asarray([item["min_weight"] for item in baseline["assets"]], dtype=np.float64)
        upper = np.asarray([item["max_weight"] for item in baseline["assets"]], dtype=np.float64)
        limits = np.asarray([min(item["max_abs_tilt"], request.max_abs_tilt) for item in baseline["assets"]])
        split = next((i for i, value in enumerate(data["dates"]) if value > str(request.train_end_date)), len(data["dates"]))
        training_knowledge = numeric.knowledge_window_status(data["available_at"], _day(request.train_end_date), 0, split)
        if request.search and training_knowledge["future_cells"]:
            raise ValidationError("TAA_TRAINING_LABEL_NOT_MATURE", "部分训练收益在训练截止日尚不可得，不能用于选优；请先检查数据可得时间，调整训练边界或明确改为固定假设比较。", diagnostics=preflight["guidance"])
        if request.search and training_knowledge["unknown_cells"] and (request.decision_policy or request.signal_mode == "composite"):
            raise ValidationError("TAA_TRAINING_KNOWLEDGE_UNKNOWN", "新策略训练期存在未知可得日期；请补齐证据或显式改为固定假设比较。")
        signals = signals if signals is not None else self._signals(request, data, assets)
        if request.search and preflight["training"].get("train_signal_observations") == 0:
            raise ValidationError("TAA_NO_TRAINING_SIGNAL", "训练期没有有效趋势信号，不能选择最优偏离；请调整观察窗口、信号有效期或明确改为固定假设比较。", diagnostics=preflight["guidance"])
        groups = baseline.get("group_limits") or []
        group_args = {
            "group_membership": np.asarray([[int(asset in group["assets"]) for asset in assets] for group in groups], dtype=np.uint8).reshape(len(groups), len(assets)),
            "group_min": np.asarray([group["lo"] for group in groups], dtype=np.float64),
            "group_max": np.asarray([group["hi"] for group in groups], dtype=np.float64),
        }
        plan = None
        if request.decision_policy:
            from .clocks import simulation_clock
            plan = simulation_clock(request, data, signals, include_current=True)
        clock_args = {"decision_policy": request.decision_policy,
                      "clock": None if plan is None else {k: v[:-1] for k, v in plan.items()},
                      "direct_tilts": signals.get("direct_tilts")}
        strengths = np.asarray([0, .25, .5, .75, 1, 1.25, 1.5] if request.search else [0, 1], dtype=np.float64)
        result = numeric.evaluate_candidates(
            data["returns"], signals["probabilities"], signals["use_signal"], base, signals["state_tilts"],
            lower, upper, limits, split, strengths, request.transaction_cost_bps, 252, request.risk_penalty,
            request.max_tracking_error, request.max_turnover, request.objective,
            selected_candidate_id=request.selected_candidate_id or (None if request.search else "scale-1"),
            allow_infeasible_selected=request.decision_policy is not None and not request.search,
            **group_args, **clock_args,
        )
        candidate = next(item for item in result["candidates"] if item["id"] == result["selected_id"])
        current = _vector(request.current_weights, assets, "当前持仓") if request.current_weights is not None else None
        decision_index = len(data["dates"])
        rec_prob, rec_use, rec_tilts = signals["current_probabilities"], signals["current_use_signal"], signals["state_tilts"]
        if plan is not None:
            decision_index = next((i for i in range(len(plan["decisions"]) - 1 - request.decision_policy.execution_lag, -1, -1) if plan["decisions"][i]), -1)
            if decision_index < 0:
                rec_use = 0
            elif decision_index < len(data["dates"]):
                rec_prob, rec_use = signals["probabilities"][decision_index], signals["use_signal"][decision_index]
        if plan is not None and decision_index >= 0 and plan["valid_until"][decision_index] < _day(request.as_of):
            rec_use = 0
        if "direct_tilts" in signals:
            rec_tilts = (np.zeros(len(assets)) if decision_index < 0 else signals["current_direct_tilt"] if decision_index == len(data["dates"]) else signals["direct_tilts"][decision_index]).reshape(1, -1)
            rec_prob = np.ones(1)
        recommendation = numeric.recommend_weights(
            rec_prob, rec_use, base, rec_tilts,
            lower, upper, limits, candidate["strength"], current_weights=current,
            **group_args,
        )
        recommendation_signal_date = signals["current_date"]
        recommendation_window = signals.get("current_timing")
        if plan is not None and decision_index >= 0:
            timing = signals["audit"].get("signal_timing")
            if timing:
                recommendation_window = timing[decision_index]
                recommendation_signal_date = recommendation_window.get("window_end") or recommendation_window.get("signal_date")
            elif request.signal_mode == "composite":
                used = [item["timing"][decision_index] for item in signals["audit"]["components"] if item["weight"] > 0]
                recommendation_signal_date = min((row["observed_on"] for row in used if row["observed_on"]), default=None)
        elif plan is not None:
            recommendation_signal_date = None
        reasons = list(dict.fromkeys([*baseline.get("pit", {}).get("reasons", []), *data.get("pit", {}).get("reasons", []),
                    "本次规则与 SAA 在当前研究中确定；历史回放不等于当时已部署，不能认定为正式 PIT 业绩。"] ))
        warnings = [*data.get("reasons", []),
                    "按冻结决策与执行时钟推进；未交易时持仓漂移。SAA/TAA 使用同一执行规则与成本，训练和验证独立从 SAA 起步。" if request.decision_policy else
                    "日频目标再平衡；SAA/TAA 使用相同交易成本。训练和验证分别从 SAA 起步。",
                    "候选选择只看训练区；多次查看留出结果后调参会降低验证独立性。"]
        if data["lineage"].get("excluded_incomplete_dates"):
            warnings.append("数据存在不完整日期，已采用共同净值区间；年化按 252 个观察期估算，不代表连续日频实盘收益。")
        warnings.append("可得时间按日期校验；实际 ETF 开盘与基金申赎时点需在产品执行层另行验证。")
        if training_knowledge["unknown_cells"]:
            warnings.append("训练收益存在未知可得时间，排名仅为事后研究证据，留出结果不构成正式 PIT 验证。")
        if current is not None and recommendation["turnover"] > request.max_turnover + 1e-8:
            warnings.append("从当前实际持仓调整至目标的单次换手超过上限，请调整方案后再应用。")
        if request.signal_mode == "manual":
            warnings.append("人工观点在整段历史中作为固定假设重演，不代表观点在历史当时已经形成。")
        if request.signal_mode == "regime":
            audit = signals["audit"]
            published_dates = [str(item.get("published_at") or "")[:10] for item in audit["publications"]]
            created = str(audit.get("run_created_at") or "")[:10]
            if created > str(request.as_of) or any(value > str(request.as_of) for value in published_dates):
                warnings.append("该市场状态模型或发布在研究时点尚不存在，本次仅为事后规则回放。")
        if not candidate["feasible"]:
            warnings.append("固定假设的训练实际持仓或风险预算超限；仅保留诊断，不可应用。")
        if not candidate.get("validation_feasible", True):
            warnings.append("所选候选在留出区超出风险或换手约束；请复核，不自动改选其他候选。")
        if signals["fallback_reason"]:
            warnings.append(signals["fallback_reason"])
        chart, weights = [], []
        expires = request.as_of + timedelta(days=request.review_days)
        if request.signal_mode != "manual" and not recommendation["is_saa"]:
            # Expiry belongs to the adopted decision, which may precede the
            # latest observation while execution lag is still pending.
            if plan is not None and decision_index >= 0:
                expires = min(expires, date(1970, 1, 1) + timedelta(days=int(plan["valid_until"][decision_index])))
            elif request.signal_mode == "composite":
                expires = min(expires, date.fromisoformat(signals["current_expires_on"]))
            elif signals["current_date"]:
                expires = min(expires, date.fromisoformat(signals["current_date"][:10]) + timedelta(days=request.max_signal_age_days))
        offset = len(assets) * 2
        for segment, dates, path in [
            ("train", data["dates"][:split], result["selected_train_path"]),
            ("validation", data["dates"][split:], result["selected_path"]),
        ]:
            for index, value in enumerate(dates):
                chart.append({"date": value, "baseline": float(path[index, offset + 8]),
                              "taa": float(path[index, offset + 9]), "segment": segment})
                weights.append({"date": value, "weights": dict(zip(assets, path[index, :len(assets)].tolist())),
                                "turnover": float(path[index, offset + 1]),
                                "cost": float(path[index, offset + 3]), "segment": segment,
                                "two_way_turnover": float(path[index, offset + 1] * 2),
                                **({"decision": bool(path[index, offset + 14]), "execution_opportunity": bool(path[index, offset + 15]),
                                    "traded": bool(path[index, offset + 16]),
                                    "target_decision_date": data["period_starts"][(0 if segment == "train" else split) + int(path[index, offset + 18])] if path[index, offset + 18] >= 0 else None,
                                    "no_trade_reason": {0: "executed", 1: "waiting_decision_or_lag", 2: "not_execution_opportunity", 3: "minimum_holding", 4: "below_threshold"}[int(path[index, offset + 17])],
                                    "research_target": dict(zip(assets, path[index, offset + 19:].tolist()))}
                                   if request.decision_policy else {})})
        raw_tilts = numeric.current_signal_tilt_kernel(rec_prob, rec_tilts, int(rec_use), float(candidate["strength"]))
        momentum = signals.get("current_momentum")
        signal_details = []
        for i, asset in enumerate(assets):
            raw, applied = float(raw_tilts[i]), float(recommendation["tilts"][i])
            signal_details.append({"asset_id": asset,
                                   "value": float(momentum[i]) if momentum is not None and np.isfinite(momentum[i]) else None,
                                   "signal_date": recommendation_signal_date, "window": recommendation_window, "direction": "增配" if raw > 1e-10 else "减配" if raw < -1e-10 else "维持",
                                   "raw_tilt": raw, "applied_tilt": applied,
                                   "constraint_reason": signals["fallback_reason"] or ("资产或分组约束缩小了偏离。" if abs(raw - applied) > 1e-8 else None)})
        payload = {
            "request": request.model_dump(mode="json"), "baseline": baseline,
            "data": {"start_date": data["dates"][0], "end_date": data["dates"][-1],
                     "observations": len(data["dates"]), "train_observations": split,
                     "validation_observations": len(data["dates"]) - split,
                     "source_hash": data["source_hash"], "lineage": data["lineage"],
                     "pit": {"status": "research_only", "reasons": reasons}, "quality": preflight["quality"], "training": preflight["training"]},
            "candidates": result["candidates"], "selected_id": result["selected_id"],
            "recommendation": {
                "signal_details": signal_details,
                "weights": dict(zip(assets, recommendation["weights"].tolist())),
                "tilts": dict(zip(assets, recommendation["tilts"].tolist())),
                "trade_deltas": dict(zip(assets, recommendation["trade_deltas"].tolist())) if current is not None else None,
                "reason": "约束与当前信号下保留 SAA 基线。" if recommendation["is_saa"] else
                          (f"按{'研究员观点' if request.signal_mode == 'manual' else '趋势信号' if request.signal_mode == 'momentum' else '组合信号' if request.signal_mode == 'composite' else '已发布市场状态'}比较预设偏离强度；留出结果仅供验证。" if request.selected_candidate_id or not request.search else
                           "采用训练窗口内可行候选中得分最高的强度；按最新可得信号生成当前权重。"),
                "signal_date": recommendation_signal_date, "expires_on": str(expires), "is_saa": recommendation["is_saa"],
                "confidence": signals["confidence"], "fallback_reason": signals["fallback_reason"],
                "turnover_from_current": recommendation["turnover"] if current is not None else None,
            },
            "chart": chart, "weight_path": weights, "warnings": list(dict.fromkeys(warnings)),
            "execution": result["execution"],
            "audit": {"signal": signals["audit"], "latest_observation_signal": {"date": signals["current_date"], "active": bool(signals["current_use_signal"])}, "selection": result["selection_policy"],
                      "training_label_knowledge": training_knowledge,
                      "auto_selected_id": result.get("auto_selected_id", result["selected_id"]),
                      "baseline_hash": baseline["content_hash"], "formal_pit_eligible": False,
                      "decision_as_of": str(request.as_of), "cost_basis": request.decision_policy.cost_basis if request.decision_policy else "half_turnover", "rebalance": request.decision_policy.model_dump(mode="json") if request.decision_policy else "daily_target", "periods_per_year": 252},
        }
        if plan is not None:
            from .clocks import application_status
            payload["application"] = application_status(request, data, plan, recommendation, decision_index)
            if not signals["current_use_signal"] and not recommendation["is_saa"]:
                payload["application"]["eligible"] = False
                payload["application"]["state"] = "ineligible"
                payload["application"]["reasons"].append("当前信号已不可用，已有决策目标不能直接交接；等待下一次决策复核。")
            if not candidate["feasible"] or not candidate.get("validation_feasible", True):
                payload["application"]["eligible"] = False
                payload["application"]["state"] = "ineligible"
                payload["application"]["reasons"].append("所选路径存在实际持仓、风险或换手预算超限，不能交接。")
            latest_decision = next(i for i in range(len(plan["decisions"]) - 1, -1, -1) if plan["decisions"][i])
            payload["application"]["latest_decision_date"] = (data["period_starts"] + [str(request.as_of)])[latest_decision]
            payload["application"]["pending_decision"] = latest_decision > decision_index
            if latest_decision > decision_index:
                payload["warnings"].append("最新决策仍在等待滞后；拟议权重采用最近已满足滞后的决策，不能把新目标当作已成交。")
            payload["warnings"].extend(payload["application"]["reasons"])
        if request.walk_forward is not None:
            from backend.tactical_allocation.walk_forward import evaluate_walk_forward
            payload["walk_forward"] = evaluate_walk_forward(request, data, signals, base, lower, upper, limits, group_args)
        if baseline.get("policy"):
            from backend.strategic_allocation.policy_gate import check_policy
            payload["policy_check"] = check_policy(baseline, payload["recommendation"]["weights"], request.max_tracking_error, str(request.as_of))
            payload["warnings"].extend(payload["policy_check"]["violations"])
            payload["recommendation"]["expires_on"] = min(payload["recommendation"]["expires_on"], baseline["policy"]["expires_on"])
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        # Keep the eventual immutable manifest within its checked reader budget.
        # Reject before any save, rather than creating a version it cannot reopen.
        if len(encoded) > 7_000_000:
            raise ValidationError("TAA_PREVIEW_BUDGET", "当前区间与资产数量生成的结果过大，请缩短回测区间后再比较。")
        payload["preview_hash"] = hashlib.sha256(encoded).hexdigest()
        return payload, data

    def preview(self, request: PreviewRequest) -> dict:
        return self._calculate(request)[0]

    def scenario(self, request: ScenarioRequest) -> dict:
        preview_request = request.preview_request.model_copy(update={"selected_candidate_id": request.candidate_id}) if request.candidate_id else request.preview_request
        preview, data = self._calculate(preview_request)
        return self._scenario_calculation(preview, data, request.scenario)[0]

    def _scenario_calculation(self, preview: dict, data: dict, scenario) -> tuple[dict, dict]:
        preview_request = PreviewRequest.model_validate(preview["request"])
        assets = [item["id"] for item in preview["baseline"]["assets"]]
        base = _vector({item["id"]: item["base_weight"] for item in preview["baseline"]["assets"]}, assets, "SAA")
        target = _vector(preview["recommendation"]["weights"], assets, "TAA")
        snapshot = {}
        evidence = {}
        if scenario.kind == "shock":
            matrix = _vector(scenario.shocks, assets, "情景冲击").reshape(1, -1)
        else:
            if not scenario.start_date or not scenario.end_date or scenario.start_date >= scenario.end_date:
                raise ValidationError("TAA_SCENARIO_DATES", "历史重演需要有效的起止日期。")
            if scenario.end_date > preview_request.as_of:
                raise ValidationError("TAA_SCENARIO_FUTURE", "历史情景结束日不能晚于研究时点。")
            if str(scenario.start_date) >= data["period_starts"][0] and str(scenario.end_date) <= data["dates"][-1]:
                left = next((i for i, day in enumerate(data["period_starts"]) if day >= str(scenario.start_date)), len(data["dates"]))
                right = next((i for i, day in enumerate(data["dates"]) if day > str(scenario.end_date)), len(data["dates"]))
                if right - left < 2:
                    raise ValidationError("TAA_SCENARIO_SHORT", "历史情景至少需要 3 个共同净值日，请扩大区间。")
                historical = {**data, "returns": data["returns"][left:right], "available_at": data["available_at"][left:right],
                              "dates": data["dates"][left:right]}
            else:
                raise ValidationError("TAA_SCENARIO_COVERAGE", f"历史情景须位于本次预览范围 {data['period_starts'][0]} 至 {data['dates'][-1]}；请先调整研究区间并重新比较。")
            matrix = historical["returns"]
            quality = return_quality(matrix, historical["dates"], assets)
            if quality["issues"]:
                raise ValidationError("TAA_NAV_SCALE_BREAK", quality["issues"][0]["message"], diagnostics=quality["issues"])
            snapshot = {"returns": matrix, "available_days": historical["available_at"]}
            evidence = {"dates": historical["dates"], "source_hash": historical["source_hash"]}
        result = numeric.stress_compare(matrix, base, target, preview_request.transaction_cost_bps,
                                        cost_basis=preview_request.decision_policy.cost_basis if preview_request.decision_policy else "half_turnover")
        return {"name": scenario.name, "kind": scenario.kind, "simulation_scope": "current_target_fixed_stress", "cost_basis": result["cost_basis"],
                "preview_hash": preview["preview_hash"], "evidence": evidence,
                "baseline_return": result["baseline"]["total_return"], "taa_return": result["target"]["total_return"],
                "excess_return": result["total_return_difference"], "relative_excess_return": result["excess_return"],
                "contributions": [{"asset_id": key, "baseline": float(result["baseline_contributions"][i]),
                                   "taa": float(result["target_contributions"][i]),
                                   "excess": float(result["excess_contributions"][i])}
                                  for i, key in enumerate(assets)],
                "cost": {"baseline": result["baseline_cost"], "taa": result["target_cost"]},
                "warnings": ["这是指定冲击/历史条件下的比较，不是预测或发生概率。",
                             "使用当前建议权重、每日恢复目标、同一成本；资产收益贡献扣除显式费用后与净收益对账；改善表示两者收益百分点之差。"],
                "execution": result["execution"]}, snapshot

    def save_decision(self, body: SaveDecisionRequest) -> dict:
        preview, data = self._calculate(body.request)
        if preview["preview_hash"] != body.preview_hash:
            raise ConflictError("TAA_PREVIEW_CHANGED", "输入数据或方案已变化，请重新预览后保存。")
        scenarios, arrays = [], {"returns": data["returns"], "available_days": data["available_at"]}
        for index, scenario in enumerate(body.scenarios):
            result, frozen = self._scenario_calculation(preview, data, scenario)
            scenarios.append({"scenario": scenario.model_dump(mode="json"), "result": result})
            arrays.update({f"scenario_{index}_{key}": value for key, value in frozen.items()})
        fields = {"name": body.name, "note": body.note,
                    "baseline_id": body.request.baseline_id, "as_of": str(body.request.as_of),
                    "expires_on": preview["recommendation"]["expires_on"], "preview": preview,
                    "research_only": True, "scenarios": scenarios}
        if len(json.dumps(fields, ensure_ascii=False, allow_nan=False).encode()) > 7_500_000:
            raise ValidationError("TAA_DECISION_BUDGET", "研究版本与情景过大，请减少情景或缩短区间后保存。")
        return self.repository.save_decision(fields, arrays=arrays)

    def product_allocation(self, decision_id: str) -> dict:
        decision = self.repository.get_decision(decision_id)
        preview = decision["preview"]
        baseline = preview["baseline"]
        from backend.tactical_allocation.portfolio_bridge import validate_decision_application
        from .clocks import validate_clock_application
        validate_clock_application(preview)
        validate_decision_application(decision, self.data, self.repository.decision_arrays(decision_id)["returns"])
        assets = baseline["assets"]
        class_weights = _vector(preview["recommendation"]["weights"], [item["id"] for item in assets], "TAA")
        products, class_indices, within = [], [], []
        seen = set()
        for index, asset in enumerate(assets):
            if not asset.get("products"):
                raise ValidationError("TAA_PRODUCTS_REQUIRED", f"{asset['name']} 缺少明确的产品映射。")
            for product in asset["products"]:
                key = (product["kind"], product["product_id"])
                if key in seen:
                    raise ValidationError("TAA_DUPLICATE_PRODUCT_CLASS", "同一产品跨大类出现，请在 SAA 中明确唯一预算归属后应用。")
                seen.add(key)
                products.append({**product, "asset_class_id": asset["id"], "asset_class_name": asset["name"]})
                class_indices.append(index)
                within.append(product["weight"])
        weights = numeric.compose_product_weights(class_weights, np.asarray(class_indices, dtype=np.int64), np.asarray(within, dtype=np.float64))
        # Scaling to display percentages is a serialization unit conversion.
        constituents = [{**product, "weight": float(weights[i]) * 100} for i, product in enumerate(products)]
        return {"name": decision["name"][:80], "method": "manual", "constituents": constituents,
                "universe_snapshot_id": baseline["universe_snapshot_id"],
                "allocation_source": {"kind": "taa", "decision_id": decision_id, "baseline_id": baseline["id"],
                                      "class_weights": copy.deepcopy(preview["recommendation"]["weights"]),
                                      "expires_on": preview["recommendation"]["expires_on"]}}
