"""Real-data portfolio research, immutable runs, and diagnostics.

The service deliberately keeps portfolio research separate from product
ranking.  It uses adjusted NAV, strict date intersection and beginning-of-day
weights so a rebalance decision only affects the following common data day.
"""

from __future__ import annotations

import csv
import copy
import hashlib
import io
import json
import math
import os
import threading
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

try:
    from backend.market_data import resolve_market_data_file
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from market_data import resolve_market_data_file

from backtest_engine import gen_rebalance_dates
from strategy import compute_risk_budget_weights, compute_target_weights

from .errors import ConflictError, IndicatorDomainError, ValidationError
from .presentation import metric_presentation
from .portfolio_repository import PortfolioRunRepository, ResearchTargetRepository
from .series_provider import DEFAULT_DATA_DIR, load_adjusted_product_series


PORTFOLIO_SCHEMA_VERSION = "2.0.0"
MAX_ASSETS = 50
MAX_OBSERVATIONS = 5000
MAX_ARRAY_BYTES = 64 * 1024 * 1024
WEIGHT_TOLERANCE = 1e-8
STRATEGY_ALIASES = {
    "equal": "equal_weight",
    "equal_weight": "equal_weight",
    "manual": "manual",
    "fixed": "manual",
    "risk_budget": "risk_budget",
    "target": "target_optimization",
    "target_optimization": "target_optimization",
}

PORTFOLIO_SUMMARY_SPECS = {
    "cumulative_return": ("累计收益率", "percent", 2, "%", "return_decimal", "higher_better", "组合收益逐期复合后的累计收益。"),
    "annual_return": ("年化收益率", "percent", 2, "%", "return_decimal", "higher_better", "按共同收益观察数折算的年化复合收益。"),
    "annual_volatility": ("年化波动率", "percent", 2, "%", "return_decimal", "lower_better", "组合收益样本标准差乘年化因子平方根。"),
    "sharpe_ratio": ("夏普比率", "number", 3, "", "dimensionless", "higher_better", "年化超额收益与年化波动率之比。"),
    "max_drawdown": ("最大回撤", "percent", 2, "%", "return_decimal", "lower_better", "组合净值相对历史峰值的最大损失幅度。"),
    "var_99": ("历史 VaR 99%", "percent", 3, "%", "return_decimal", "lower_better", "组合收益经验分布 1% 分位损失幅度。"),
    "es_99": ("历史 CVaR 99%", "percent", 3, "%", "return_decimal", "lower_better", "不高于 1% 分位的平均损失幅度。"),
}


def _portfolio_metric_presentation(metric_id: str) -> dict[str, Any]:
    name, display_format, precision, unit, measure, direction, methodology = PORTFOLIO_SUMMARY_SPECS[metric_id]
    return metric_presentation(
        {
            "id": f"portfolio-summary-{metric_id}",
            "revision": 1,
            "name": name,
            "source": "built_in",
            "category_id": "portfolio_risk" if direction == "lower_better" else "portfolio_return",
            "category_label": "组合风险" if direction == "lower_better" else "组合收益",
            "context_kind": "portfolio",
            "dsl_version": "2.1.0",
            "display_format": display_format,
            "precision": precision,
            "unit": unit,
            "output_measure": measure,
            "direction": direction,
            "description": methodology,
            "methodology": methodology,
            "data_basis": "锁定运行快照、真实复权净值、严格共同日期",
            "minimum_observations": 2,
            "applicable_product_kinds": ["portfolio"],
        }
    )


def _summary_metric_rows(summary: dict[str, Optional[float]]) -> list[dict[str, Any]]:
    return [
        {
            "metric_id": metric_id,
            "name": PORTFOLIO_SUMMARY_SPECS[metric_id][0],
            "value": value,
            "status": "ok" if value is not None else "unavailable",
            "warnings": [] if value is not None else [{"code": "NON_FINITE_RESULT", "message": "该汇总指标不可计算。"}],
            "presentation": _portfolio_metric_presentation(metric_id),
        }
        for metric_id, value in summary.items()
        if metric_id in PORTFOLIO_SUMMARY_SPECS
    ]


class PortfolioSnapshotCache:
    def __init__(self, max_size: int = 128, ttl_seconds: int = 300) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, payload = item
            if expires_at <= now:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return copy.deepcopy(payload)

    def put(self, key: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._items[key] = (time.monotonic() + self.ttl_seconds, copy.deepcopy(payload))
            self._items.move_to_end(key)
            while len(self._items) > self.max_size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


def _finite_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _date_text(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _common_date_hash(index: pd.DatetimeIndex) -> str:
    payload = "\n".join(_date_text(item) for item in index)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_date(value: Optional[str], field: str) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("INVALID_DATE", f"{field} 必须是有效日期。", field) from exc
    if pd.isna(parsed):
        raise ValidationError("INVALID_DATE", f"{field} 必须是有效日期。", field)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_localize(None)
    return parsed.normalize()


def _deep_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


class PortfolioResearchService:
    def __init__(
        self,
        workspace_data_dir: Optional[Path] = None,
        market_data_dir: Optional[Path] = None,
        cache: Optional[PortfolioSnapshotCache] = None,
    ) -> None:
        configured = os.getenv("CUSTOM_INDICATOR_DATA_DIR")
        self.workspace_data_dir = workspace_data_dir or (Path(configured) if configured else DEFAULT_DATA_DIR)
        self.market_data_dir = market_data_dir or DEFAULT_DATA_DIR
        self.targets = ResearchTargetRepository(self.workspace_data_dir / "research_targets.json")
        self.runs = PortfolioRunRepository(self.workspace_data_dir / "portfolio_runs.json")
        self.cache = cache or PortfolioSnapshotCache()

    # ------------------------------------------------------------------
    # Research target lifecycle

    def list_targets(self) -> list[dict[str, Any]]:
        return self.targets.list()

    def get_target(self, target_id: str, revision: Optional[int] = None) -> dict[str, Any]:
        return self.targets.get(target_id, revision)

    def create_target(self, fields: dict[str, Any]) -> dict[str, Any]:
        return self.targets.create(self._normalize_target(fields))

    def update_target(self, target_id: str, revision: int, fields: dict[str, Any]) -> dict[str, Any]:
        updated = self.targets.update(target_id, revision, self._normalize_target(fields))
        self.cache.clear()
        return updated

    def delete_target(self, target_id: str, revision: int) -> None:
        if self.runs.references_target(target_id):
            raise ConflictError(
                "RESEARCH_TARGET_IN_USE",
                "该研究组合已被运行快照引用，不能直接删除。",
            )
        self.targets.delete(target_id, revision)
        self.cache.clear()

    @staticmethod
    def _definition(fields: dict[str, Any]) -> dict[str, Any]:
        nested = fields.get("definition")
        if isinstance(nested, dict):
            return {**nested}
        return {
            key: fields.get(key)
            for key in ("components", "strategy", "constraints", "rebalance", "benchmark", "alignment")
            if key in fields
        }

    def _normalize_target(self, fields: dict[str, Any]) -> dict[str, Any]:
        name = str(fields.get("name") or "").strip()
        if not name or len(name) > 80:
            raise ValidationError("INVALID_TARGET_NAME", "研究组合名称长度应为 1 至 80 个字符。", "name")
        description = str(fields.get("description") or "").strip()
        if len(description) > 500:
            raise ValidationError("INVALID_TARGET_DESCRIPTION", "研究组合说明不能超过 500 个字符。", "description")
        definition = self._definition(fields)
        components = definition.get("components")
        if not isinstance(components, list) or not components:
            raise ValidationError("EMPTY_COMPONENTS", "请至少选择一个真实产品。", "definition.components")
        if len(components) > MAX_ASSETS:
            raise ValidationError("TOO_MANY_ASSETS", f"研究组合最多包含 {MAX_ASSETS} 个产品。", "definition.components")

        normalized_components: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for index, component in enumerate(components):
            if not isinstance(component, dict):
                raise ValidationError("INVALID_COMPONENT", "产品定义格式无效。", f"definition.components.{index}")
            kind = str(component.get("kind") or "").lower()
            product_id = str(component.get("product_id") or component.get("ts_code") or component.get("code") or "").strip()
            if kind not in {"etf", "fund"}:
                raise ValidationError("INVALID_COMPONENT_KIND", "产品类型必须为 etf 或 fund。", f"definition.components.{index}.kind")
            if not product_id:
                raise ValidationError("INVALID_PRODUCT_ID", "产品代码不能为空。", f"definition.components.{index}.product_id")
            key = (kind, product_id.lower())
            if key in seen:
                raise ValidationError("DUPLICATE_COMPONENT", "同一产品不能重复加入组合。", f"definition.components.{index}")
            seen.add(key)
            normalized_components.append(
                {
                    "kind": kind,
                    "product_id": product_id,
                    "name": str(component.get("name") or product_id),
                }
            )

        alignment = str(definition.get("alignment") or "strict_intersection")
        if alignment != "strict_intersection":
            raise ValidationError(
                "UNSUPPORTED_ALIGNMENT",
                "当前仅支持 strict_intersection，禁止前值填充或缺失收益补零。",
                "definition.alignment",
            )
        strategy = self._normalize_strategy(definition.get("strategy") or {}, normalized_components)
        rebalance = self._normalize_rebalance(definition.get("rebalance") or strategy.pop("rebalance", None) or {})
        constraints = _deep_copy(definition.get("constraints") or {})
        if constraints:
            min_weight = float(constraints.get("min_weight", 0.0) or 0.0)
            max_weight = float(constraints.get("max_weight", 1.0) if constraints.get("max_weight") is not None else 1.0)
            if not (0 <= min_weight <= max_weight <= 1):
                raise ValidationError(
                    "INVALID_WEIGHT_CONSTRAINT",
                    "全局权重约束必须满足 0 ≤ min_weight ≤ max_weight ≤ 1。",
                    "definition.constraints",
                )
            count = len(normalized_components)
            if min_weight * count > 1 + WEIGHT_TOLERANCE or max_weight * count < 1 - WEIGHT_TOLERANCE:
                raise ValidationError(
                    "INFEASIBLE_WEIGHT_CONSTRAINT",
                    "当前最小/最大权重约束不存在合计为 1 的可行解。",
                    "definition.constraints",
                )
        benchmark = definition.get("benchmark")
        if benchmark is not None:
            if not isinstance(benchmark, dict) or benchmark.get("kind") not in {"etf", "fund"}:
                raise ValidationError("INVALID_BENCHMARK", "基准必须引用一个 ETF 或基金。", "definition.benchmark")
            benchmark = {
                "kind": str(benchmark["kind"]),
                "product_id": str(benchmark.get("product_id") or benchmark.get("ts_code") or "").strip(),
            }
            if not benchmark["product_id"]:
                raise ValidationError("INVALID_BENCHMARK", "基准产品代码不能为空。", "definition.benchmark.product_id")

        normalized_definition = {
            "components": normalized_components,
            "strategy": strategy,
            "constraints": constraints,
            "rebalance": rebalance,
            "benchmark": benchmark,
            "alignment": "strict_intersection",
        }
        return {
            "name": name,
            "description": description,
            "kind": "portfolio",
            "context_kind": "portfolio",
            "schema_version": PORTFOLIO_SCHEMA_VERSION,
            "source": str(fields.get("source") or "workspace"),
            "definition": normalized_definition,
        }

    @staticmethod
    def _normalize_strategy(strategy: dict[str, Any], components: list[dict[str, str]]) -> dict[str, Any]:
        if not isinstance(strategy, dict):
            raise ValidationError("INVALID_STRATEGY", "策略配置格式无效。", "definition.strategy")
        raw_type = str(strategy.get("type") or "equal_weight").lower()
        strategy_type = STRATEGY_ALIASES.get(raw_type)
        if strategy_type is None:
            raise ValidationError("UNSUPPORTED_STRATEGY", "不支持的组合策略。", "definition.strategy.type")
        normalized = {**_deep_copy(strategy), "type": strategy_type}
        normalized.pop("weights_by_asset", None)
        if strategy_type == "target_optimization":
            target_aliases = {
                "min_volatility": "min_risk",
                "target_return": "risk_min_given_return",
            }
            normalized["target"] = target_aliases.get(
                str(strategy.get("target") or "min_risk"),
                str(strategy.get("target") or "min_risk"),
            )
        asset_count = len(components)
        if strategy_type == "manual":
            raw_weights = strategy.get("weights")
            if isinstance(raw_weights, dict):
                raw_weights = [raw_weights.get(item["product_id"]) for item in components]
            if not isinstance(raw_weights, list) or len(raw_weights) != asset_count:
                raise ValidationError(
                    "WEIGHT_COUNT_MISMATCH",
                    "手工权重必须与有序产品一一对应。",
                    "definition.strategy.weights",
                )
            weights = np.asarray(raw_weights, dtype=float)
            if not np.isfinite(weights).all() or (weights < 0).any():
                raise ValidationError("INVALID_WEIGHTS", "手工权重必须为非负有限数值。", "definition.strategy.weights")
            if abs(float(weights.sum()) - 1.0) > WEIGHT_TOLERANCE:
                raise ValidationError(
                    "WEIGHTS_NOT_NORMALIZED",
                    "手工权重合计必须为 1（容差 1e-8），系统不会自动归一化。",
                    "definition.strategy.weights",
                )
            normalized["weights"] = [float(value) for value in weights]
        elif strategy_type == "risk_budget":
            raw_budgets = strategy.get("budgets") or [1.0 / asset_count] * asset_count
            if not isinstance(raw_budgets, list) or len(raw_budgets) != asset_count:
                raise ValidationError("BUDGET_COUNT_MISMATCH", "风险预算必须与产品一一对应。", "definition.strategy.budgets")
            budgets = np.asarray(raw_budgets, dtype=float)
            if not np.isfinite(budgets).all() or (budgets < 0).any() or float(budgets.sum()) <= 0:
                raise ValidationError("INVALID_RISK_BUDGET", "风险预算必须为非负有限数值且至少一项大于零。", "definition.strategy.budgets")
            normalized["budgets"] = [float(value / budgets.sum()) for value in budgets]
        lookback = int(strategy.get("lookback_observations") or (60 if strategy_type in {"risk_budget", "target_optimization"} else 2))
        if lookback < 2 or lookback > MAX_OBSERVATIONS:
            raise ValidationError("INVALID_LOOKBACK", "回看窗口必须在 2 至 5000 个观察值之间。", "definition.strategy.lookback_observations")
        normalized["lookback_observations"] = lookback
        return normalized

    @staticmethod
    def _normalize_rebalance(rebalance: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(rebalance, dict):
            raise ValidationError("INVALID_REBALANCE", "调仓配置格式无效。", "definition.rebalance")
        enabled = bool(rebalance.get("enabled", False))
        mode = str(rebalance.get("mode") or "monthly")
        if mode not in {"weekly", "monthly", "yearly", "fixed"}:
            raise ValidationError("INVALID_REBALANCE_MODE", "不支持的调仓周期。", "definition.rebalance.mode")
        fixed_interval = int(rebalance.get("fixed_interval") or rebalance.get("fixedInterval") or 20)
        if fixed_interval < 1 or fixed_interval > MAX_OBSERVATIONS:
            raise ValidationError("INVALID_REBALANCE_INTERVAL", "固定调仓间隔超出范围。", "definition.rebalance.fixed_interval")
        transaction_cost = float(rebalance.get("transaction_cost_bps") or 0.0)
        if not math.isfinite(transaction_cost) or transaction_cost < 0:
            raise ValidationError("INVALID_TRANSACTION_COST", "交易成本必须为非负有限数值。", "definition.rebalance.transaction_cost_bps")
        if transaction_cost != 0:
            raise ValidationError(
                "TRANSACTION_COST_OUT_OF_SCOPE",
                "当前阶段仅输出毛收益，请将交易成本设为 0。",
                "definition.rebalance.transaction_cost_bps",
            )
        return {
            "enabled": enabled,
            "mode": mode,
            "which": str(rebalance.get("which") or "first"),
            "N": int(rebalance.get("N") or 1),
            "unit": str(rebalance.get("unit") or "trading"),
            "fixed_interval": fixed_interval,
        }

    # ------------------------------------------------------------------
    # Real-data run

    def list_runs(self, target_id: Optional[str] = None) -> list[dict[str, Any]]:
        items = self.runs.list()
        if target_id:
            items = [item for item in items if item.get("target_id") == target_id]
        heavy = {
            "asset_returns",
            "daily_weights",
            "contribution_series",
            "portfolio_returns",
            "portfolio_nav",
            "drawdown",
            "dates",
        }
        return [{key: value for key, value in item.items() if key not in heavy} for item in items]

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self.runs.get(run_id)

    def run_target(
        self,
        target_id: str,
        *,
        as_of: Optional[str] = None,
        start_date: Optional[str] = None,
    ) -> dict[str, Any]:
        target = self.targets.get(target_id)
        cache_key = self._snapshot_cache_key(target, as_of=as_of, start_date=start_date, end_date=None)
        payload = self.cache.get(cache_key)
        if payload is None:
            payload = self._build_snapshot(target, as_of=as_of, start_date=start_date)
            self.cache.put(cache_key, payload)
            payload["cache"] = {"hit": False, "ttl_seconds": self.cache.ttl_seconds}
        else:
            payload["cache"] = {"hit": True, "ttl_seconds": self.cache.ttl_seconds}
        return self.runs.create(payload)

    def _snapshot_cache_key(
        self,
        target: dict[str, Any],
        *,
        as_of: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> str:
        kinds = {item["kind"] for item in target["definition"]["components"]}
        benchmark = target["definition"].get("benchmark")
        if benchmark:
            kinds.add(benchmark["kind"])
        file_fingerprints: dict[str, str] = {}
        for kind in sorted(kinds):
            filename = "etf_daily_df.parquet" if kind == "etf" else "fund_nav_df.parquet"
            path = resolve_market_data_file(filename, self.market_data_dir)
            if not path.exists():
                file_fingerprints[filename] = "missing"
                continue
            stat = path.stat()
            file_fingerprints[filename] = f"{stat.st_size}:{stat.st_mtime_ns}"
        raw = {
            "schema_version": PORTFOLIO_SCHEMA_VERSION,
            "target_id": target["id"],
            "target_revision": target["revision"],
            "definition": target["definition"],
            "as_of": as_of,
            "start_date": start_date,
            "end_date": end_date,
            "files": file_fingerprints,
        }
        return hashlib.sha256(
            json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _load_common_nav(
        self,
        target: dict[str, Any],
        *,
        as_of: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str] = None,
    ) -> tuple[
        pd.DataFrame,
        list[dict[str, Any]],
        dict[str, str],
        list[dict[str, str]],
        Optional[dict[str, Any]],
    ]:
        definition = target["definition"]
        frames: list[pd.Series] = []
        assets: list[dict[str, Any]] = []
        fingerprints: dict[str, str] = {}
        warnings: list[dict[str, str]] = []
        requested_end = end_date or as_of
        end_ts = _parse_date(requested_end, "as_of" if not end_date else "end_date")

        for position, component in enumerate(definition["components"]):
            product = load_adjusted_product_series(
                component["kind"],
                component["product_id"],
                self.market_data_dir,
            )
            if product is None:
                raise ValidationError(
                    "PRODUCT_DATA_NOT_FOUND",
                    f"{component['product_id']} 缺少真实复权净值数据。",
                    f"definition.components.{position}",
                )
            frame = product.frame[["date", "value"]].copy()
            if end_ts is not None:
                frame = frame[frame["date"] <= end_ts]
            if frame.empty:
                raise ValidationError(
                    "NO_DATA_AS_OF",
                    f"{component['product_id']} 在截止日前没有可用数据。",
                    "as_of",
                )
            asset_key = f"{component['kind']}:{product.identity.ts_code}"
            frames.append(frame.set_index("date")["value"].rename(asset_key))
            fingerprints[asset_key] = product.fingerprint
            assets.append(
                {
                    "position": position,
                    "key": asset_key,
                    "kind": component["kind"],
                    "product_id": product.identity.product_id,
                    "ts_code": product.identity.ts_code,
                    "name": product.identity.name,
                    "data_latest_date": product.data_latest_date,
                }
            )

        nav = pd.concat(frames, axis=1, join="inner").sort_index()
        benchmark_payload: Optional[dict[str, Any]] = None
        benchmark = definition.get("benchmark")
        if benchmark:
            product = load_adjusted_product_series(
                benchmark["kind"],
                benchmark["product_id"],
                self.market_data_dir,
            )
            if product is None:
                warnings.append(
                    {
                        "code": "BENCHMARK_DATA_NOT_FOUND",
                        "message": f"基准 {benchmark['product_id']} 缺少真实复权净值，基准上下文不可用。",
                    }
                )
            else:
                benchmark_frame = product.frame[["date", "value"]].copy()
                if end_ts is not None:
                    benchmark_frame = benchmark_frame[benchmark_frame["date"] <= end_ts]
                benchmark_series = benchmark_frame.set_index("date")["value"].rename("__benchmark__")
                nav = nav.join(benchmark_series, how="inner")
                benchmark_key = f"benchmark:{benchmark['kind']}:{product.identity.ts_code}"
                fingerprints[benchmark_key] = product.fingerprint
                benchmark_payload = {
                    "key": benchmark_key,
                    "kind": benchmark["kind"],
                    "product_id": product.identity.product_id,
                    "name": product.identity.name,
                }
        nav = nav[~nav.index.duplicated(keep="last")].dropna(how="any")
        if start_date:
            start_ts = _parse_date(start_date, "start_date")
            assert start_ts is not None
            strategy = definition["strategy"]
            lookback = int(strategy.get("lookback_observations") or 2)
            prior_count = int((nav.index < start_ts).sum())
            keep_prior = min(prior_count, max(lookback, 2))
            nav = nav.iloc[max(0, prior_count - keep_prior) :]
        if len(nav) < 3:
            raise ValidationError(
                "INSUFFICIENT_COMMON_SAMPLE",
                "严格日期交集后至少需要 3 个共同净值点。",
            )
        if len(nav) > MAX_OBSERVATIONS + 1:
            nav = nav.iloc[-(MAX_OBSERVATIONS + 1) :]
            warnings.append(
                {
                    "code": "PORTFOLIO_SERIES_TRUNCATED",
                    "message": f"共同样本已截取最近 {MAX_OBSERVATIONS} 个收益观察值。",
                }
            )
        estimated_bytes = int(nav.shape[0] * nav.shape[1] * 8 * 8)
        if estimated_bytes > MAX_ARRAY_BYTES:
            raise ValidationError(
                "COMPUTE_BUDGET_EXCEEDED",
                "预计矩阵中间结果超过 64MB 计算预算。",
            )
        if benchmark_payload is not None:
            benchmark_payload["values"] = nav.pop("__benchmark__").astype(float)
        return nav.astype(float), assets, fingerprints, warnings, benchmark_payload

    @staticmethod
    def _constraint_bounds(strategy: dict[str, Any], definition: dict[str, Any], asset_keys: list[str]) -> list[tuple[float, float]]:
        constraints = {**(definition.get("constraints") or {}), **(strategy.get("constraints") or {})}
        raw_limits = constraints.get("single_limits") or {}
        global_lo = float(constraints.get("min_weight", 0.0) or 0.0)
        global_hi = float(constraints.get("max_weight", 1.0) if constraints.get("max_weight") is not None else 1.0)
        bounds: list[tuple[float, float]] = []
        for key in asset_keys:
            plain_id = key.split(":", 1)[-1]
            value = raw_limits.get(key, raw_limits.get(plain_id, {})) if isinstance(raw_limits, dict) else {}
            lo = float(value.get("lo", global_lo)) if isinstance(value, dict) else global_lo
            hi = float(value.get("hi", global_hi)) if isinstance(value, dict) else global_hi
            if not (0 <= lo <= hi <= 1):
                raise ValidationError("INVALID_WEIGHT_CONSTRAINT", "单资产权重约束必须满足 0 ≤ lo ≤ hi ≤ 1。")
            bounds.append((lo, hi))
        return bounds

    def _strategy_weights(
        self,
        nav_history: pd.DataFrame,
        definition: dict[str, Any],
    ) -> np.ndarray:
        strategy = definition["strategy"]
        strategy_type = strategy["type"]
        asset_count = nav_history.shape[1]
        if strategy_type == "equal_weight":
            return np.full(asset_count, 1.0 / asset_count, dtype=float)
        if strategy_type == "manual":
            return np.asarray(strategy["weights"], dtype=float)
        lookback = int(strategy.get("lookback_observations") or 60)
        fit_nav = nav_history.tail(lookback + 1)
        if len(fit_nav) < min(lookback + 1, 3):
            raise ValidationError("INSUFFICIENT_STRATEGY_SAMPLE", "可用历史不足以计算策略权重。")
        try:
            if strategy_type == "risk_budget":
                weights = compute_risk_budget_weights(
                    fit_nav,
                    {"metric": str(strategy.get("risk_metric") or "vol")},
                    list(strategy["budgets"]),
                )
            else:
                bounds = self._constraint_bounds(strategy, definition, list(fit_nav.columns))
                weights = compute_target_weights(
                    fit_nav,
                    {
                        "metric": str(strategy.get("return_metric") or "annual"),
                        "days": int(strategy.get("annualization_factor") or 252),
                    },
                    {
                        "metric": str(strategy.get("risk_metric") or "annual_vol"),
                        "days": int(strategy.get("annualization_factor") or 252),
                    },
                    target=str(strategy.get("target") or "min_risk"),
                    single_limits=bounds,
                    risk_free_rate=float(strategy.get("risk_free_rate") or 0.0),
                    target_return=strategy.get("target_return"),
                    target_risk=strategy.get("target_risk"),
                    use_exploration=False,
                )
        except IndicatorDomainError:
            raise
        except Exception as exc:
            message = str(exc) or "策略权重计算失败。"
            code = "SINGULAR_MATRIX" if "singular" in message.lower() else "STRATEGY_COMPUTE_FAILED"
            raise ValidationError(code, message) from exc
        parsed = np.asarray(weights, dtype=float)
        if parsed.shape != (asset_count,) or not np.isfinite(parsed).all() or (parsed < -WEIGHT_TOLERANCE).any():
            raise ValidationError("INVALID_STRATEGY_WEIGHTS", "策略返回了无效权重。")
        parsed = np.maximum(parsed, 0.0)
        total = float(parsed.sum())
        if total <= 0:
            raise ValidationError("INVALID_STRATEGY_WEIGHTS", "策略权重合计必须大于零。")
        return parsed / total

    def _decision_dates(self, nav: pd.DataFrame, definition: dict[str, Any], start_date: Optional[str]) -> list[pd.Timestamp]:
        strategy = definition["strategy"]
        rebalance = definition["rebalance"]
        optimizer = strategy["type"] in {"risk_budget", "target_optimization"}
        lookback = int(strategy.get("lookback_observations") or 2)
        start_ts = _parse_date(start_date, "start_date")
        if optimizer:
            minimum_position = min(max(lookback, 2), len(nav) - 1) - 1
        else:
            minimum_position = 0
        if start_ts is not None:
            before_start = np.flatnonzero(nav.index < start_ts)
            if len(before_start):
                minimum_position = max(minimum_position, int(before_start[-1]))
        if minimum_position >= len(nav) - 1:
            raise ValidationError("INSUFFICIENT_STRATEGY_SAMPLE", "没有足够的样本用于策略拟合和下一日生效。")
        initial = nav.index[minimum_position]
        if not rebalance["enabled"]:
            return [initial]
        generated = gen_rebalance_dates(
            nav.index,
            rebalance["mode"],
            N=rebalance["N"],
            which=rebalance["which"],
            unit=rebalance["unit"],
            fixed_interval=rebalance["fixed_interval"] if rebalance["mode"] == "fixed" else None,
        )
        eligible = [pd.Timestamp(item) for item in generated if initial <= item < nav.index[-1]]
        return sorted(set([initial, *eligible]))

    def _backtest(
        self,
        nav: pd.DataFrame,
        definition: dict[str, Any],
        start_date: Optional[str],
    ) -> dict[str, Any]:
        returns = nav.pct_change(fill_method=None).iloc[1:].replace([np.inf, -np.inf], np.nan).dropna(how="any")
        if len(returns) < 2:
            raise ValidationError("INSUFFICIENT_COMMON_SAMPLE", "共同收益观察值不足。")
        decisions = self._decision_dates(nav, definition, start_date)
        schedules: dict[pd.Timestamp, tuple[pd.Timestamp, np.ndarray]] = {}
        weight_path: list[dict[str, Any]] = []
        for decision_date in decisions:
            location = int(nav.index.get_loc(decision_date))
            if location + 1 >= len(nav):
                continue
            effective_date = nav.index[location + 1]
            if effective_date not in returns.index:
                continue
            weights = self._strategy_weights(nav.loc[:decision_date], definition)
            schedules[effective_date] = (decision_date, weights)
            weight_path.append(
                {
                    "decision_date": _date_text(decision_date),
                    "effective_date": _date_text(effective_date),
                    "weights": [float(value) for value in weights],
                }
            )
        if not schedules:
            raise ValidationError("NO_EFFECTIVE_WEIGHTS", "没有可在下一共同数据日生效的权重。")

        first_effective = min(schedules)
        returns = returns.loc[first_effective:]
        current: Optional[np.ndarray] = None
        portfolio_returns: list[float] = []
        daily_weights: list[list[float]] = []
        contribution_rows: list[list[float]] = []
        for date, row in returns.iterrows():
            if date in schedules:
                current = schedules[date][1].copy()
            if current is None:
                continue
            values = row.to_numpy(dtype=float)
            daily_weights.append([float(value) for value in current])
            contributions = current * values
            portfolio_return = float(contributions.sum())
            if not math.isfinite(portfolio_return) or portfolio_return <= -1.0:
                raise ValidationError("NON_FINITE_RESULT", "组合收益出现非有限值或小于等于 -100%。")
            portfolio_returns.append(portfolio_return)
            contribution_rows.append([float(value) for value in contributions])
            denominator = 1.0 + portfolio_return
            current = current * (1.0 + values) / denominator
        result_dates = returns.index[: len(portfolio_returns)]
        return {
            "dates": result_dates,
            "asset_returns": returns.iloc[: len(portfolio_returns)].to_numpy(dtype=float),
            "portfolio_returns": np.asarray(portfolio_returns, dtype=float),
            "daily_weights": np.asarray(daily_weights, dtype=float),
            "contributions": np.asarray(contribution_rows, dtype=float),
            "weight_path": weight_path,
        }

    @staticmethod
    def _summary(portfolio_returns: np.ndarray, annual_risk_free_rate: float = 0.0) -> dict[str, Optional[float]]:
        count = int(portfolio_returns.size)
        nav = np.cumprod(1.0 + portfolio_returns)
        cumulative_return = float(nav[-1] - 1.0)
        annual_return = float((nav[-1] ** (252.0 / count)) - 1.0) if count else float("nan")
        annual_vol = float(np.std(portfolio_returns, ddof=1) * math.sqrt(252.0)) if count > 1 else float("nan")
        sharpe = (annual_return - annual_risk_free_rate) / annual_vol if annual_vol > 0 else float("nan")
        peaks = np.maximum.accumulate(nav)
        drawdown = nav / peaks - 1.0
        q01 = float(np.quantile(portfolio_returns, 0.01))
        tail = portfolio_returns[portfolio_returns <= q01]
        values = {
            "cumulative_return": cumulative_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": abs(float(drawdown.min())),
            "var_99": -q01,
            "es_99": -float(tail.mean()) if tail.size else float("nan"),
        }
        return {key: _finite_float(value) for key, value in values.items()}

    def _build_snapshot(
        self,
        target: dict[str, Any],
        *,
        as_of: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str] = None,
    ) -> dict[str, Any]:
        nav, assets, fingerprints, warnings, benchmark = self._load_common_nav(
            target,
            as_of=as_of,
            start_date=start_date,
            end_date=end_date,
        )
        backtest = self._backtest(nav, target["definition"], start_date)
        portfolio_returns = backtest["portfolio_returns"]
        portfolio_nav = np.cumprod(1.0 + portfolio_returns)
        drawdown = portfolio_nav / np.maximum.accumulate(portfolio_nav) - 1.0
        dates = pd.DatetimeIndex(backtest["dates"])
        benchmark_returns: Optional[list[float]] = None
        benchmark_info: Optional[dict[str, Any]] = None
        if benchmark is not None:
            benchmark_series = benchmark.pop("values")
            computed = benchmark_series.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
            benchmark_returns = [float(value) for value in computed.loc[dates].to_numpy(dtype=float)]
            benchmark_info = benchmark
        annual_rf = float(target["definition"]["strategy"].get("risk_free_rate") or 0.0)
        summary = self._summary(portfolio_returns, annual_rf)
        asset_order = [asset["key"] for asset in assets]
        weight_path = [
            {
                **entry,
                "date": entry["effective_date"],
                "weights": {
                    key: float(value)
                    for key, value in zip(asset_order, entry["weights"])
                },
            }
            for entry in backtest["weight_path"]
        ]
        return {
            "schema_version": PORTFOLIO_SCHEMA_VERSION,
            "context_schema": "portfolio-v2",
            "target_id": target["id"],
            "target_revision": int(target["revision"]),
            "target_name": target["name"],
            "target_definition": _deep_copy(target["definition"]),
            "asset_order": asset_order,
            "assets": assets,
            "data_fingerprints": fingerprints,
            "common_date_hash": _common_date_hash(dates),
            "requested_as_of": as_of,
            "effective_as_of": _date_text(dates[-1]),
            "actual_start_date": _date_text(dates[0]),
            "actual_end_date": _date_text(dates[-1]),
            "observation_count": int(len(dates)),
            "strategy_config": _deep_copy(target["definition"]["strategy"]),
            "rebalance_config": _deep_copy(target["definition"]["rebalance"]),
            "weight_path": weight_path,
            "dates": [_date_text(item) for item in dates],
            "portfolio_returns": [float(value) for value in portfolio_returns],
            "portfolio_nav": [float(value) for value in portfolio_nav],
            "drawdown": [float(value) for value in drawdown],
            "asset_returns": backtest["asset_returns"].tolist(),
            "benchmark": benchmark_info,
            "benchmark_returns": benchmark_returns,
            "daily_weights": backtest["daily_weights"].tolist(),
            "contribution_series": backtest["contributions"].tolist(),
            "summary": summary,
            "summary_metrics": _summary_metric_rows(summary),
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Diagnostics and historical scenarios

    def diagnose(self, run_id: str, indicator_ids: Optional[list[str]] = None) -> dict[str, Any]:
        run = self.runs.get(run_id)
        returns = np.asarray(run["asset_returns"], dtype=float)
        weights = np.asarray(run["daily_weights"], dtype=float)
        contributions = np.asarray(run["contribution_series"], dtype=float)
        if returns.ndim != 2 or weights.shape != returns.shape:
            raise ValidationError("SNAPSHOT_DATA_INVALID", "运行快照中的收益或权重矩阵无效。")
        covariance = np.cov(returns, rowvar=False, ddof=1)
        if covariance.ndim == 0:
            covariance = covariance.reshape(1, 1)
        correlation = np.corrcoef(returns, rowvar=False)
        if correlation.ndim == 0:
            correlation = correlation.reshape(1, 1)
        current_weights = weights[-1]
        portfolio_variance = float(current_weights @ covariance @ current_weights)
        portfolio_volatility = math.sqrt(max(portfolio_variance, 0.0))
        if portfolio_volatility > 0:
            marginal = covariance @ current_weights / portfolio_volatility
            component_risk = current_weights * marginal
            component_total = float(component_risk.sum())
            risk_shares = component_risk / component_total if abs(component_total) > 1e-15 else np.full_like(component_risk, np.nan)
        else:
            marginal = np.full_like(current_weights, np.nan)
            component_risk = np.full_like(current_weights, np.nan)
            risk_shares = np.full_like(current_weights, np.nan)
        hhi = float(np.square(current_weights).sum())
        ordered = np.sort(current_weights)[::-1]
        asset_keys = list(run["asset_order"])
        interval_contribution = contributions.sum(axis=0)
        component_rows = []
        risk_rows = []
        contribution_rows = []
        for index, asset in enumerate(run["assets"]):
            component_rows.append(
                {
                    **asset,
                    "weight": float(current_weights[index]),
                    "current_weight": float(current_weights[index]),
                    "period_return": float(np.prod(1.0 + returns[:, index]) - 1.0),
                    "simple_return_contribution": float(interval_contribution[index]),
                }
            )
            risk_rows.append(
                {
                    "product_id": asset["product_id"],
                    "name": asset["name"],
                    "asset_key": asset_keys[index],
                    "contribution": _finite_float(risk_shares[index]),
                    "risk_contribution": _finite_float(risk_shares[index]),
                    "marginal_risk": _finite_float(marginal[index]),
                    "component_risk": _finite_float(component_risk[index]),
                    "risk_contribution_ratio": _finite_float(risk_shares[index]),
                }
            )
            contribution_rows.append(
                {
                    "product_id": asset["product_id"],
                    "name": asset["name"],
                    "contribution": float(interval_contribution[index]),
                    "risk_contribution": _finite_float(risk_shares[index]),
                }
            )
        custom_indicators = self._evaluate_custom_indicators(run, indicator_ids or [])
        concentration_summary = {
            "max_weight": float(ordered[0]),
            "top3_weight": float(ordered[:3].sum()),
            "hhi": hhi,
            "effective_holdings": (1.0 / hhi) if hhi > 0 else None,
        }
        concentration = [
            {
                "name": name,
                "value": concentration_summary[key],
                "status": "ok" if concentration_summary[key] is not None else "unavailable",
                "warnings": [],
                "presentation": metric_presentation(
                    {
                        "id": f"portfolio-concentration-{key}",
                        "revision": 1,
                        "name": name,
                        "source": "built_in",
                        "category_id": "portfolio_risk",
                        "category_label": "集中度",
                        "context_kind": "portfolio",
                        "dsl_version": "2.1.0",
                        "display_format": display_format,
                        "precision": precision,
                        "unit": "%" if display_format == "percent" else "",
                        "output_measure": "dimensionless",
                        "direction": "lower_better",
                        "description": description,
                        "methodology": description,
                        "data_basis": "锁定运行快照中的当前权重",
                        "minimum_observations": 1,
                        "applicable_product_kinds": ["portfolio"],
                    }
                ),
            }
            for key, name, display_format, precision, description in (
                ("max_weight", "最大权重", "percent", 2, "当前权重中的最大单一资产占比。"),
                ("top3_weight", "Top3 权重", "percent", 2, "当前权重最高三项的合计占比。"),
                ("hhi", "HHI", "number", 3, "当前资产权重平方和。"),
                ("effective_holdings", "有效持仓数", "number", 2, "HHI 的倒数。"),
            )
        ]
        contribution_series = [
            {
                "date": date,
                "values": {key: float(value) for key, value in zip(asset_keys, row)},
            }
            for date, row in zip(run["dates"], run["contribution_series"])
        ]
        rebalances: list[dict[str, Any]] = []
        previous: Optional[np.ndarray] = None
        for item in run["weight_path"]:
            current = np.asarray([item["weights"][key] for key in asset_keys], dtype=float)
            turnover = None if previous is None else float(np.abs(current - previous).sum() / 2.0)
            rebalances.append(
                {
                    "date": item["effective_date"],
                    "decision_date": item["decision_date"],
                    "turnover": turnover,
                    "message": "使用决策日数据计算，并在下一共同数据日生效。",
                }
            )
            previous = current
        return {
            "run_id": run_id,
            "target_id": run["target_id"],
            "summary": run["summary"],
            "summary_metrics": _summary_metric_rows(run["summary"]),
            "custom_indicators": custom_indicators,
            "components": component_rows,
            "contributions": contribution_rows,
            "dates": run["dates"],
            "contribution_series": contribution_series,
            "covariance": {"labels": asset_keys, "values": covariance.tolist()},
            "correlation": {"labels": asset_keys, "values": correlation.tolist()},
            "risk_contributions": risk_rows,
            "concentration": concentration,
            "concentration_summary": concentration_summary,
            "weight_path": run["weight_path"],
            "daily_weights": run["daily_weights"],
            "rebalances": rebalances,
            "warnings": run.get("warnings", []),
        }

    def _evaluate_custom_indicators(self, run: dict[str, Any], indicator_ids: list[str]) -> list[dict[str, Any]]:
        if not indicator_ids:
            return []
        try:
            from .service import CustomIndicatorService

            service = CustomIndicatorService(
                workspace_data_dir=self.workspace_data_dir,
                market_data_dir=self.market_data_dir,
            )
            response = service.evaluate_portfolio_snapshot(indicator_ids, run)
            return [{**item, "name": item.get("indicator_name")} for item in response["results"]]
        except AttributeError:
            return [
                {
                    "indicator_id": indicator_id,
                    "value": None,
                    "status": "unsupported",
                    "warnings": [{"code": "TYPED_RUNTIME_UNAVAILABLE", "message": "组合指标运行时尚不可用。"}],
                }
                for indicator_id in indicator_ids
            ]

    def scenario(
        self,
        run_id: str,
        *,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        locked_run = self.runs.get(run_id)
        target = self.targets.get(locked_run["target_id"], int(locked_run["target_revision"]))
        start_ts = _parse_date(start_date, "start_date")
        end_ts = _parse_date(end_date, "end_date")
        assert start_ts is not None and end_ts is not None
        if start_ts >= end_ts:
            raise ValidationError("INVALID_SCENARIO_WINDOW", "历史情景开始日期必须早于结束日期。")
        scenario = self._build_snapshot(
            target,
            as_of=end_date,
            start_date=start_date,
            end_date=end_date,
        )
        current_fingerprints = scenario["data_fingerprints"]
        if current_fingerprints != locked_run.get("data_fingerprints"):
            scenario["warnings"].append(
                {
                    "code": "DATA_FINGERPRINT_CHANGED",
                    "message": "数据文件已变化；情景使用当前真实数据并保留锁定策略版本。",
                }
            )
        metric_rows = _summary_metric_rows(scenario["summary"])
        return {
            "source_run_id": run_id,
            "locked_target_revision": int(locked_run["target_revision"]),
            "metrics": metric_rows,
            **scenario,
        }

    def export(
        self,
        run_id: str,
        archive: bool,
        *,
        table: str = "summary",
        scenario_start: Optional[str] = None,
        scenario_end: Optional[str] = None,
    ) -> tuple[str, bytes, str]:
        diagnosis = self.diagnose(run_id)
        run = self.runs.get(run_id)
        files = self._csv_files(run, diagnosis)
        if bool(scenario_start) != bool(scenario_end):
            raise ValidationError(
                "INCOMPLETE_SCENARIO_WINDOW",
                "导出情景结果时必须同时提供 scenario_start 与 scenario_end。",
            )
        if scenario_start and scenario_end:
            scenario = self.scenario(
                run_id,
                start_date=scenario_start,
                end_date=scenario_end,
            )
            files["scenario-summary.csv"] = self._write_csv(
                ["metric", "value", "unit"],
                (
                    [
                        item["name"],
                        item["value"],
                        (item.get("presentation") or {}).get("unit", ""),
                    ]
                    for item in scenario["metrics"]
                ),
            )
            files["scenario-series.csv"] = self._write_csv(
                ["date", "portfolio_return", "portfolio_nav", "drawdown"],
                zip(
                    scenario["dates"],
                    scenario["portfolio_returns"],
                    scenario["portfolio_nav"],
                    scenario["drawdown"],
                ),
            )
        if not archive:
            filename = f"{table}.csv"
            if filename not in files:
                raise ValidationError("EXPORT_TABLE_NOT_FOUND", "不支持的 CSV 导出表。", "table")
            return f"{run_id}-{filename}", files[filename], "text/csv; charset=utf-8"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, content in files.items():
                bundle.writestr(name, content)
        return f"{run_id}-portfolio-research.zip", buffer.getvalue(), "application/zip"

    @staticmethod
    def _write_csv(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> bytes:
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(list(headers))
        writer.writerows(rows)
        return ("\ufeff" + buffer.getvalue()).encode("utf-8")

    def _csv_files(self, run: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, bytes]:
        asset_keys = list(run["asset_order"])
        files: dict[str, bytes] = {}
        files["summary.csv"] = self._write_csv(
            ["metric", "value"],
            diagnosis["summary"].items(),
        )
        files["components.csv"] = self._write_csv(
            ["asset_key", "name", "kind", "current_weight", "period_return", "simple_return_contribution"],
            (
                [row["key"], row["name"], row["kind"], row["current_weight"], row["period_return"], row["simple_return_contribution"]]
                for row in diagnosis["components"]
            ),
        )
        files["daily-contributions.csv"] = self._write_csv(
            ["date", *asset_keys],
            ([date, *values] for date, values in zip(run["dates"], run["contribution_series"])),
        )
        files["weight-path.csv"] = self._write_csv(
            ["date", *asset_keys],
            ([date, *values] for date, values in zip(run["dates"], run["daily_weights"])),
        )
        for field in ("covariance", "correlation"):
            matrix = diagnosis[field]
            files[f"{field}.csv"] = self._write_csv(
                ["asset_key", *matrix["labels"]],
                ([label, *values] for label, values in zip(matrix["labels"], matrix["values"])),
            )
        files["risk-contributions.csv"] = self._write_csv(
            ["asset_key", "marginal_risk", "component_risk", "risk_contribution_ratio"],
            (
                [row["asset_key"], row["marginal_risk"], row["component_risk"], row["risk_contribution_ratio"]]
                for row in diagnosis["risk_contributions"]
            ),
        )
        return files
