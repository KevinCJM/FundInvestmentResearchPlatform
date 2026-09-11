"""Published historical-regime conditioning for portfolio backtests.

This module is an integration boundary: it reads one explicit immutable v2
publication, verifies its complete lineage, and then delegates all temporal
alignment and performance statistics to already-warmed fixed-signature NJIT
kernels.  It never resolves a mutable "latest" definition or publication.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

try:
    from compute_policy import NJIT_BACKEND, validate_execution_audit
    from custom_indicators.errors import ConflictError, ValidationError
    from custom_indicators.portfolio_numba import (
        finite_series_close_kernel,
        portfolio_summary_kernel,
    )
    from historical_regimes.numba_kernels import conditional_statistics_kernel
    from historical_regimes.repository import RegimeDefinitionRepository, RegimeRunRepository
    from historical_regimes.v2_numba import (
        pit_asof_positions_kernel,
        stable_time_order_kernel,
        state_count_kernel,
        take_int64_kernel,
    )
    from instrument_analytics_numba import coverage_ratio_kernel
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from backend.compute_policy import NJIT_BACKEND, validate_execution_audit
    from backend.custom_indicators.errors import ConflictError, ValidationError
    from backend.custom_indicators.portfolio_numba import (
        finite_series_close_kernel,
        portfolio_summary_kernel,
    )
    from backend.historical_regimes.numba_kernels import conditional_statistics_kernel
    from backend.historical_regimes.repository import (
        RegimeDefinitionRepository,
        RegimeRunRepository,
    )
    from backend.historical_regimes.v2_numba import (
        pit_asof_positions_kernel,
        stable_time_order_kernel,
        state_count_kernel,
        take_int64_kernel,
    )
    from backend.instrument_analytics_numba import coverage_ratio_kernel


CONDITIONING_SCHEMA_VERSION = "portfolio-regime-conditioning/1.0"
CONDITIONING_ENGINE_VERSION = "portfolio-regime-conditioning-njit/1.0.0"
UNCLASSIFIED_STATE_ID = "unclassified"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class PublishedRegimeBacktestReference(BaseModel):
    """Exact immutable publication selected by a backtest request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1, max_length=128)
    publication_id: str = Field(min_length=1, max_length=128)


@dataclass(frozen=True)
class ResolvedPublishedRegime:
    run: dict[str, Any]
    binding: dict[str, Any]
    states: tuple[dict[str, Any], ...]


def _configured_workspace_dir(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    configured = os.getenv("HISTORICAL_REGIME_DATA_DIR") or os.getenv(
        "CUSTOM_INDICATOR_DATA_DIR"
    )
    return Path(configured).expanduser().resolve() if configured else DEFAULT_DATA_DIR.resolve()


def _stored_run_snapshot_hash(run: Mapping[str, Any]) -> str:
    analytical = copy.deepcopy(dict(run))
    for key in ("id", "created_at", "immutable", "publications", "content_hash"):
        analytical.pop(key, None)
    analytical["application_bindings"] = []
    encoded = json.dumps(
        analytical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PublishedRegimeBacktestResolver:
    """Resolve and verify one formal-backtest publication without warming a service."""

    def __init__(self, workspace_data_dir: Path | None = None) -> None:
        self.workspace_data_dir = _configured_workspace_dir(workspace_data_dir)
        self.runs = RegimeRunRepository(
            self.workspace_data_dir / "historical_regime_runs.json"
        )
        self.definitions = RegimeDefinitionRepository(
            self.workspace_data_dir / "historical_regime_v2_definitions.json"
        )
        self.artifact_dir = self.workspace_data_dir / "historical_regime_v2_artifacts"

    def resolve(
        self,
        reference: PublishedRegimeBacktestReference | Mapping[str, Any],
    ) -> ResolvedPublishedRegime:
        locked = (
            reference
            if isinstance(reference, PublishedRegimeBacktestReference)
            else PublishedRegimeBacktestReference.model_validate(reference)
        )
        raw = self.runs.get(locked.run_id)
        if str(raw.get("schema_version") or "") != "2.0":
            raise ValidationError(
                "FORMAL_BACKTEST_REQUIRES_V2_REGIME",
                "组合回测只接受已发布的历史情景 v2 运行。",
                "historical_regime.run_id",
            )
        if raw.get("immutable") is not True:
            raise ConflictError(
                "REGIME_RUN_NOT_IMMUTABLE",
                "历史情景运行不是不可变快照，已阻断组合回测引用。",
                field="historical_regime.run_id",
            )
        if _stored_run_snapshot_hash(raw) != raw.get("content_hash"):
            raise ConflictError(
                "REGIME_RUN_SNAPSHOT_MISMATCH",
                "历史情景运行快照校验失败，已阻断组合回测引用。",
                field="historical_regime.run_id",
            )
        if raw.get("mode") != "realtime":
            raise ValidationError(
                "FORMAL_BACKTEST_REQUIRES_REALTIME_REGIME",
                "正式组合回测只能引用 realtime 历史情景运行。",
                "historical_regime.run_id",
            )

        definition_id = str(raw.get("definition_id") or "")
        definition_revision = int(raw.get("definition_revision") or 0)
        if not definition_id or definition_revision < 1:
            raise ConflictError(
                "REGIME_DEFINITION_LINEAGE_MISMATCH",
                "历史情景运行缺少锁定的定义版本。",
                field="historical_regime.run_id",
            )
        stored_definition = self.definitions.get(definition_id, definition_revision)
        try:
            from historical_regimes.v2_contracts import (
                definition_content_hash,
                parse_definition_v2,
            )
        except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
            from backend.historical_regimes.v2_contracts import (
                definition_content_hash,
                parse_definition_v2,
            )

        locked_definition_hash = definition_content_hash(
            parse_definition_v2(stored_definition)
        )
        embedded_definition = raw.get("definition")
        if not isinstance(embedded_definition, Mapping):
            raise ConflictError(
                "REGIME_DEFINITION_LINEAGE_MISMATCH",
                "历史情景运行缺少定义快照。",
                field="historical_regime.run_id",
            )
        embedded_definition_hash = definition_content_hash(
            parse_definition_v2(dict(embedded_definition))
        )
        if (
            locked_definition_hash != raw.get("definition_snapshot_hash")
            or embedded_definition_hash != locked_definition_hash
        ):
            raise ConflictError(
                "REGIME_DEFINITION_LINEAGE_MISMATCH",
                "历史情景运行与锁定定义版本不一致。",
                field="historical_regime.run_id",
            )

        governance = raw.get("governance")
        governance = governance if isinstance(governance, Mapping) else {}
        if (
            governance.get("formal_gate_passed") is not True
            or "formal_backtest" not in set(
                governance.get("publish_eligible_usages") or []
            )
        ):
            raise ValidationError(
                "REGIME_FORMAL_BACKTEST_GATE_FAILED",
                "该历史情景运行未通过正式回测门禁。",
                "historical_regime.run_id",
            )

        publication = next(
            (
                item
                for item in raw.get("publications") or []
                if isinstance(item, Mapping)
                and item.get("id") == locked.publication_id
            ),
            None,
        )
        if not isinstance(publication, Mapping):
            raise ValidationError(
                "REGIME_PUBLICATION_NOT_FOUND",
                "历史情景运行没有匹配的发布记录。",
                "historical_regime.publication_id",
            )
        if publication.get("usage") != "formal_backtest":
            raise ValidationError(
                "FORMAL_BACKTEST_PUBLICATION_REQUIRED",
                "该历史情景版本未发布到正式回测。",
                "historical_regime.publication_id",
            )
        if (
            publication.get("run_id") != raw.get("id")
            or publication.get("run_content_hash") != raw.get("content_hash")
            or int(publication.get("definition_revision") or 0)
            != definition_revision
            or publication.get("gate") != "comprehensive_formal_gate_passed"
        ):
            raise ConflictError(
                "REGIME_PUBLICATION_LINEAGE_MISMATCH",
                "历史情景发布记录与运行版本不一致。",
                field="historical_regime.publication_id",
            )

        try:
            from historical_regimes.v2_service import hydrate_v2_run_snapshot
        except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
            from backend.historical_regimes.v2_service import hydrate_v2_run_snapshot

        hydrated = hydrate_v2_run_snapshot(raw, artifact_dir=self.artifact_dir)
        raw_states = hydrated.get("states")
        series = hydrated.get("series")
        if not isinstance(raw_states, list) or not raw_states or not isinstance(series, list):
            raise ValidationError(
                "REGIME_SERIES_REQUIRED",
                "历史情景运行缺少状态定义或状态序列。",
                "historical_regime.run_id",
            )
        states: list[dict[str, Any]] = []
        state_ids: set[str] = set()
        for raw_state in raw_states:
            if not isinstance(raw_state, Mapping):
                raise ValidationError(
                    "REGIME_STATE_SCHEMA_INVALID",
                    "历史情景状态定义无效。",
                    "historical_regime.run_id",
                )
            state_id = str(raw_state.get("id") or "")
            if not state_id or state_id in state_ids:
                raise ValidationError(
                    "REGIME_STATE_SCHEMA_INVALID",
                    "历史情景状态 id 为空或重复。",
                    "historical_regime.run_id",
                )
            state_ids.add(state_id)
            states.append(
                {
                    "id": state_id,
                    "label": str(raw_state.get("label") or state_id),
                    "color": str(raw_state.get("color") or "#64748b"),
                }
            )

        binding = {
            "schema_version": "historical-regime-publication-binding/1.0",
            "run_id": str(raw["id"]),
            "publication_id": str(publication["id"]),
            "publication_usage": "formal_backtest",
            "published_at": publication.get("published_at"),
            "definition_id": definition_id,
            "definition_revision": definition_revision,
            "definition_snapshot_hash": locked_definition_hash,
            "run_content_hash": str(raw["content_hash"]),
            "mode": "realtime",
            "usage_intent": stored_definition.get("usage_intent"),
        }
        return ResolvedPublishedRegime(
            run=hydrated,
            binding=binding,
            states=tuple(states),
        )


def _normalized_dates(values: Sequence[Any], field: str) -> pd.DatetimeIndex:
    try:
        index = pd.DatetimeIndex(pd.to_datetime(list(values), errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "REGIME_BACKTEST_DATE_INVALID",
            f"{field} 包含无效日期。",
            field,
        ) from exc
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValidationError(
            "REGIME_BACKTEST_DATE_INVALID",
            f"{field} 必须是严格有效且递增的日期序列。",
            field,
        )
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _conditioning_audit() -> dict[str, Any]:
    kernels = {
        "pit_asof_positions": pit_asof_positions_kernel,
        "stable_time_order": stable_time_order_kernel,
        "take_int64": take_int64_kernel,
        "state_count": state_count_kernel,
        "conditional_statistics": conditional_statistics_kernel,
        "finite_series_close": finite_series_close_kernel,
        "portfolio_summary": portfolio_summary_kernel,
        "coverage_ratio": coverage_ratio_kernel,
    }
    signatures = {
        name: [str(signature) for signature in kernel.signatures]
        for name, kernel in kernels.items()
    }
    material = "|".join(
        [CONDITIONING_ENGINE_VERSION]
        + [f"{name}:{','.join(values)}" for name, values in sorted(signatures.items())]
    )
    return validate_execution_audit(
        {
            "engine": CONDITIONING_ENGINE_VERSION,
            "backend": NJIT_BACKEND,
            "kernel_signatures": signatures,
            "kernel_fingerprint": hashlib.sha256(material.encode("utf-8")).hexdigest(),
            "nopython": all(bool(kernel.nopython_signatures) for kernel in kernels.values()),
            "object_mode": 0,
            "python_fallback": 0,
            "python_callback": False,
            "request_time_compilation": 0,
            "fully_warmed": all(bool(kernel.signatures) for kernel in kernels.values()),
        }
    )


def _align_regime_states(
    resolved: ResolvedPublishedRegime,
    period_starts: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    anchors = _normalized_dates(period_starts, "period_start_dates")
    state_index = {state["id"]: index for index, state in enumerate(resolved.states)}
    effective_points: list[dict[str, Any]] = []
    for item in resolved.run.get("series") or []:
        if not isinstance(item, Mapping) or not item.get("effective_date"):
            continue
        state_id = str(item.get("state_id") or UNCLASSIFIED_STATE_ID)
        if state_id == UNCLASSIFIED_STATE_ID:
            continue
        if state_id not in state_index:
            raise ConflictError(
                "REGIME_STATE_LINEAGE_MISMATCH",
                "历史情景序列引用了未登记的状态。",
                field="historical_regime.run_id",
            )
        try:
            effective_at = pd.Timestamp(item["effective_date"])
        except (TypeError, ValueError) as exc:
            raise ConflictError(
                "REGIME_STATE_DATE_INVALID",
                "历史情景序列包含无效生效日。",
                field="historical_regime.run_id",
            ) from exc
        if effective_at.tzinfo is not None:
            effective_at = effective_at.tz_localize(None)
        effective_points.append(
            {
                "effective_ns": int(effective_at.normalize().value),
                "state_code": int(state_index[state_id]),
                "state_id": state_id,
                "state_label": str(
                    item.get("state_label")
                    or resolved.states[state_index[state_id]]["label"]
                ),
                "observation_date": item.get("observation_date") or item.get("date"),
                "recognized_at": item.get("recognized_at"),
                "effective_date": pd.Timestamp(effective_at).date().isoformat(),
                "confidence": _number(item.get("confidence")),
            }
        )
    if not effective_points:
        raise ValidationError(
            "REGIME_EFFECTIVE_SERIES_EMPTY",
            "已发布历史情景没有可用于回测的生效状态。",
            "historical_regime.run_id",
        )

    effective_ns = np.ascontiguousarray(
        np.asarray([item["effective_ns"] for item in effective_points], dtype=np.int64)
    )
    source_codes = np.ascontiguousarray(
        np.asarray([item["state_code"] for item in effective_points], dtype=np.int64)
    )
    order = stable_time_order_kernel(effective_ns)
    sorted_effective_ns = take_int64_kernel(effective_ns, order)
    sorted_source_codes = take_int64_kernel(source_codes, order)
    sorted_points = [effective_points[int(position)] for position in order]
    anchor_ns = np.ascontiguousarray(anchors.asi8, dtype=np.int64)
    positions = pit_asof_positions_kernel(anchor_ns, sorted_effective_ns, -1)
    source_codes_with_unknown = np.empty(sorted_source_codes.size + 1, dtype=np.int64)
    source_codes_with_unknown[:-1] = sorted_source_codes
    source_codes_with_unknown[-1] = -1
    aligned_codes = take_int64_kernel(
        np.ascontiguousarray(source_codes_with_unknown),
        np.ascontiguousarray(positions),
    )
    counts = state_count_kernel(
        np.ascontiguousarray(aligned_codes),
        len(resolved.states),
    )
    classified = int(counts[len(resolved.states)])
    coverage = {
        "periods": int(len(anchors)),
        "classified_periods": classified,
        "classified_ratio": _number(
            coverage_ratio_kernel(classified, int(len(anchors)))
        ),
        "state_counts": {
            state["id"]: int(counts[index])
            for index, state in enumerate(resolved.states)
        },
        "state_transitions": int(counts[len(resolved.states) + 1]),
    }
    return aligned_codes, positions, sorted_points, coverage


def _conditional_rows(
    values: np.ndarray,
    aligned_codes: np.ndarray,
    states: Sequence[Mapping[str, Any]],
    periods_per_year: int,
) -> list[dict[str, Any]]:
    labels = np.full(values.size, -1, dtype=np.int64)
    labels[: aligned_codes.size] = aligned_codes
    metrics = conditional_statistics_kernel(
        np.ascontiguousarray(values, dtype=np.float64),
        np.ascontiguousarray(labels),
        len(states),
        int(periods_per_year),
    )
    rows: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        row = metrics[index]
        rows.append(
            {
                "state_id": str(state["id"]),
                "state_label": str(state["label"]),
                "observations": int(row[0]),
                "return_observations": int(row[1]),
                "mean_period_return": _number(row[2]),
                "annualized_return": _number(row[3]),
                "volatility": _number(row[4]),
                "max_drawdown": _number(row[5]),
                "sharpe": _number(row[6]),
                "positive_rate": _number(row[7]),
                "return_alignment": "period_start_effective_state",
            }
        )
    return rows


def _period_state_rows(
    period_ends: pd.DatetimeIndex,
    period_starts: pd.DatetimeIndex,
    positions: np.ndarray,
    sorted_points: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw_position in enumerate(positions):
        position = int(raw_position)
        point = sorted_points[position] if position >= 0 else None
        rows.append(
            {
                "date": period_ends[index].date().isoformat(),
                "period_start": period_starts[index].date().isoformat(),
                "state_id": point["state_id"] if point is not None else UNCLASSIFIED_STATE_ID,
                "state_label": point["state_label"] if point is not None else "未分类",
                "regime_observation_date": point.get("observation_date") if point else None,
                "regime_recognized_at": point.get("recognized_at") if point else None,
                "regime_effective_date": point.get("effective_date") if point else None,
                "confidence": point.get("confidence") if point else None,
            }
        )
    return rows


def condition_nav_backtest(
    resolved: ResolvedPublishedRegime,
    dates: Sequence[Any],
    nav_paths: Mapping[str, Sequence[float | None]],
    *,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Condition legacy multi-strategy NAV paths on prior available regimes."""

    nav_dates = _normalized_dates(dates, "backtest.dates")
    if len(nav_dates) < 2:
        raise ValidationError(
            "REGIME_BACKTEST_SAMPLE_INSUFFICIENT",
            "组合回测至少需要两个净值观察值才能按状态统计。",
            "backtest.dates",
        )
    period_starts = nav_dates[:-1]
    period_ends = nav_dates[1:]
    aligned_codes, positions, points, coverage = _align_regime_states(
        resolved,
        period_starts,
    )
    conditional: dict[str, list[dict[str, Any]]] = {}
    for name, raw_values in nav_paths.items():
        if len(raw_values) != len(nav_dates):
            raise ValidationError(
                "REGIME_BACKTEST_SHAPE_MISMATCH",
                "策略净值长度与回测日期不一致。",
                "backtest.series",
            )
        values = np.ascontiguousarray(
            np.asarray(
                [np.nan if value is None else value for value in raw_values],
                dtype=np.float64,
            )
        )
        conditional[str(name)] = _conditional_rows(
            values,
            aligned_codes,
            resolved.states,
            periods_per_year,
        )
    return {
        "schema_version": CONDITIONING_SCHEMA_VERSION,
        "binding": copy.deepcopy(resolved.binding),
        "alignment": {
            "rule": "regime.effective_date <= return.period_start",
            "same_period_end_signal_allowed": False,
            "implicit_latest_version": False,
        },
        "states": [copy.deepcopy(state) for state in resolved.states],
        "period_states": _period_state_rows(
            period_ends,
            period_starts,
            positions,
            points,
        ),
        "coverage": coverage,
        "conditional_performance": conditional,
        "execution": _conditioning_audit(),
    }


def condition_return_backtest(
    resolved: ResolvedPublishedRegime,
    period_end_dates: Sequence[Any],
    period_start_dates: Sequence[Any],
    return_paths: Mapping[str, Sequence[float]],
    *,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Condition direct return paths from immutable portfolio-run snapshots."""

    period_ends = _normalized_dates(period_end_dates, "backtest.dates")
    period_starts = _normalized_dates(period_start_dates, "period_start_dates")
    if len(period_ends) == 0 or len(period_starts) != len(period_ends):
        raise ValidationError(
            "REGIME_BACKTEST_SHAPE_MISMATCH",
            "组合收益日期与收益区间起点不一致。",
            "backtest.dates",
        )
    aligned_codes, positions, points, coverage = _align_regime_states(
        resolved,
        period_starts,
    )
    conditional: dict[str, list[dict[str, Any]]] = {}
    for name, raw_returns in return_paths.items():
        returns = np.ascontiguousarray(np.asarray(raw_returns, dtype=np.float64))
        if returns.size != len(period_ends) or finite_series_close_kernel(
            returns,
            returns,
            0.0,
            0.0,
        ) != 1:
            raise ValidationError(
                "REGIME_BACKTEST_SHAPE_MISMATCH",
                "组合收益长度或数值与回测日期不一致。",
                "backtest.portfolio_returns",
            )
        nav, _, _ = portfolio_summary_kernel(
            returns,
            float(periods_per_year),
            0.0,
        )
        wealth = np.empty(nav.size + 1, dtype=np.float64)
        wealth[0] = 1.0
        wealth[1:] = nav
        conditional[str(name)] = _conditional_rows(
            np.ascontiguousarray(wealth),
            aligned_codes,
            resolved.states,
            periods_per_year,
        )
    return {
        "schema_version": CONDITIONING_SCHEMA_VERSION,
        "binding": copy.deepcopy(resolved.binding),
        "alignment": {
            "rule": "regime.effective_date <= return.period_start",
            "same_period_end_signal_allowed": False,
            "implicit_latest_version": False,
        },
        "states": [copy.deepcopy(state) for state in resolved.states],
        "period_states": _period_state_rows(
            period_ends,
            period_starts,
            positions,
            points,
        ),
        "coverage": coverage,
        "conditional_performance": conditional,
        "execution": _conditioning_audit(),
    }


__all__ = [
    "PublishedRegimeBacktestReference",
    "PublishedRegimeBacktestResolver",
    "ResolvedPublishedRegime",
    "condition_nav_backtest",
    "condition_return_backtest",
]
