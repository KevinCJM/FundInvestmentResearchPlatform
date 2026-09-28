"""Read-only, explicit-date evidence for LTCMA, not an implementation-product map."""
from __future__ import annotations

from datetime import date
import os
from pathlib import Path

import numpy as np
import pandas as pd

from backend.custom_indicators.errors import ConflictError, NotFoundError, ValidationError
from backend.market_data import resolve_market_data_file
from backend.research_input_checks import return_quality
from backend.sensitivity.repository import digest_json
from backend.data_storage import guard_path
from backend.historical_regimes.repository import RegimeRunRepository
from custom_indicators.errors import IndicatorDomainError as RegimeDomainError
from .contracts import RiskReferenceRequest
from backend.tactical_allocation.data import common_daily_periods
from .reference_inputs import _rebalance_reset_flags, array_digest, automatic_research_day, common_return_periods
from . import reference_evidence_kernels as proxy_kernels
from . import cma_statistical_kernels as statistical_kernels


def _year_before(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(year=day.year - years, day=28)


def _period_contiguity(starts: list[str], ends: list[str]) -> np.ndarray:
    """Decode interval adjacency before the filtered sample loses its date gaps."""
    if len(starts) != len(ends):
        raise ValidationError("LTCMA_PERIOD_AXIS", "收益区间起止日期不一致。")
    contiguous = np.zeros(len(ends), dtype=np.int64)
    contiguous[1:] = np.asarray(starts[1:]) == np.asarray(ends[:-1])
    contiguous.flags.writeable = False
    return contiguous


def _calendar(data_dir: Path, *, through: date | None = None) -> np.ndarray:
    path = resolve_market_data_file("trade_day_df.parquet", data_dir)
    if not path.is_file():
        raise ValidationError("LTCMA_CALENDAR_REQUIRED", "缺少 SSE 日历，不能证明日收益连续。")
    frame = pd.read_parquet(path, columns=["exchange", "cal_date", "is_open"])
    sse = frame.loc[frame["exchange"].astype(str).str.upper() == "SSE"]
    values = sse["cal_date"]
    open_flags = pd.to_numeric(sse["is_open"], errors="coerce")
    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = pd.to_datetime(values, errors="coerce")
    else:
        parsed = pd.to_datetime(values.astype(str).str.replace("-", "", regex=False).str[:8],
                                format="%Y%m%d", errors="coerce")
    if parsed.isna().any() or parsed.empty or not open_flags.isin([0, 1]).all() or not (open_flags == 1).any():
        raise ValidationError("LTCMA_CALENDAR_INVALID", "SSE 交易日历包含无效日期、开放标记或没有开放日。")
    # A missing calendar tail is not evidence that those days were closed.
    # Keep closed dates for coverage checks, but only open dates enter returns.
    if through is not None and parsed.max().date() < through:
        raise ValidationError("LTCMA_CALENDAR_COVERAGE", "日历未覆盖所选样本结束日；请更新日历或明确调整窗口，不能静默缩短。")
    return np.unique(parsed.loc[open_flags == 1].to_numpy(dtype="datetime64[D]").astype(np.int64))


def _window(model, first_day: str, calendar: np.ndarray) -> tuple[str, str, np.ndarray]:
    window = model.window
    end = window.end_date if window.kind == "custom" else model.as_of
    start = (window.start_date if window.kind == "custom" else date.fromisoformat(first_day)
             if window.kind == "common_since_inception" else _year_before(model.as_of, int(window.kind[:-1])))
    if end > model.as_of or start >= end:
        raise ValidationError("LTCMA_WINDOW_INVALID", "历史样本须开始于研究日前，且结束不晚于研究日。")
    start_int, end_int = (int(np.datetime64(x, "D").astype(np.int64)) for x in (start, end))
    if start_int < calendar[0]:
        raise ValidationError("LTCMA_CALENDAR_COVERAGE", "日历未覆盖所选历史窗口；不会静默缩短窗口。")
    expected = calendar[(calendar >= start_int) & (calendar <= end_int)]
    if expected.size < 21:
        raise ValidationError("LTCMA_SAMPLE_TOO_SHORT", "至少需要 21 个连续共同净值日（20 个收益期）。")
    if expected.size > 10000:
        raise ValidationError("LTCMA_SAMPLE_CAPACITY", "单次最多 10000 个共同净值日，请缩短历史窗口。")
    return str(start), str(end), expected


class CmaEvidence:
    def __init__(self, strategic):
        self.strategic = strategic
        self.sources = strategic.risk_scales.references.sources
        configured = os.getenv("HISTORICAL_REGIME_DATA_DIR")
        self.regime_root = Path(configured) if configured else strategic.artifacts.root.parent.parent
        self.runs = RegimeRunRepository(self.regime_root / "historical_regime_runs.json")
        self._hydrate_regime = None
        self._regime_hash = None
        from .cma_scenario_evidence import CmaScenarioEvidence
        self.scenarios = CmaScenarioEvidence(self)

    def bind_scenario_graph(self, graph):
        self.scenarios.bind(graph)

    def scenario_options(self, as_of):
        return self.scenarios.options(as_of)

    def scenario(self, model, evidence):
        from custom_indicators.errors import IndicatorDomainError as RegimeDomainError
        try:
            evidence["regime"] = self.scenarios.historical(model, evidence)
            if model.method == "conditional_scenario":
                evidence["forecast_evidence"] = self.scenarios.conditional(model, evidence)
        except RegimeDomainError as exc:
            raise ValidationError(exc.code, exc.message) from exc

    def prepare_regime_reader(self):
        from backend.historical_regimes.v2_service import _stored_run_snapshot_hash, hydrate_v2_run_snapshot
        self._hydrate_regime, self._regime_hash = hydrate_v2_run_snapshot, _stored_run_snapshot_hash

    def build(self, request, source: dict) -> dict:
        return self.build_sample(request.model, source, request.strategic_universe_id)

    def build_sample(self, model, source: dict, strategic_universe_id: str | None) -> dict:
        """The same validated daily panel serves sample inspection and estimation."""
        if model.as_of > automatic_research_day(self.strategic.data.data_dir):
            raise ValidationError("LTCMA_KNOWLEDGE_CUTOFF", "LTCMA 研究日晚于平台知识截止日。")
        if strategic_universe_id:
            if model.proxy_inputs is None:
                raise ValidationError("LTCMA_RESEARCH_PROXY_REQUIRED", "统计方法需要每类资产的研究代理；这不是实施产品映射。")
            result = self._proxies(model)
        else:
            if model.proxy_inputs is not None:
                raise ValidationError("LTCMA_DUPLICATE_PROXY_SOURCE", "产品大类已有收益来源，不能同时传入另一套代理。")
            result = self._product(model, source)
        result["metadata"].update({"historical_pit_proven": False,
            "observation_frequency": "daily", "periods_per_year": 252,
            "forecast_semantics": "historical_evidence_not_a_guarantee",
            "moment_semantics": "annualized_periodic_arithmetic", "currency": model.currency,
            "currency_basis": "user_confirmed_same_currency_no_automatic_conversion"})
        if isinstance(result["returns"], np.ndarray):
            result["returns"].flags.writeable = False
        result["metadata"]["return_panel_hash"] = array_digest(result["returns"])
        result["metadata"]["period_contiguous_hash"] = array_digest(result["period_contiguous"])
        result["metadata"]["transition_gap_count"] = int(np.count_nonzero(result["period_contiguous"][1:] == 0))
        return result

    def _product(self, model, source):
        frame, _, _ = self.strategic.data._nav_version(source["alloc_name"], source["lineage"].get("as_of") or "")
        dates = pd.to_datetime(frame["date"], errors="coerce")
        first = []
        for asset in model.asset_ids:
            rows = dates[(frame["asset_name"] == asset) & (dates <= pd.Timestamp(model.as_of))]
            if rows.empty or rows.isna().any():
                raise ValidationError("LTCMA_ASSET_COVERAGE", "所选大类在研究日前缺少有效净值日期。")
            first.append(rows.min().date().isoformat())
        requested_end = model.window.end_date if model.window.kind == "custom" else model.as_of
        start, end, _ = _window(model, max(first), _calendar(self.strategic.data.data_dir, through=requested_end))
        loaded = self.strategic.data.load_data(source, start, end, str(model.as_of))
        reference = RiskReferenceRequest(alloc_name=source["alloc_name"], as_of=model.as_of,
                                         start_date=start, end_date=end, periods_per_year=252)
        excluded_nav_dates = self.strategic._validate_daily_risk_axis(reference, loaded)
        lineage = loaded["lineage"]
        if lineage["missing_availability_rows"]:
            raise ValidationError("LTCMA_INCOMPLETE_EVIDENCE", "净值可得时间缺失，不能证明历史信息在研究日已知。")
        returns, dates, excluded_periods = common_daily_periods(loaded)
        quality = return_quality(returns, dates, model.asset_ids)
        if quality["issues"]:
            raise ValidationError("LTCMA_RETURN_QUALITY", quality["issues"][0]["message"], diagnostics=quality["issues"])
        if len(dates) < 20:
            raise ValidationError("LTCMA_SAMPLE_TOO_SHORT", "共同可得的单日收益不足 20 个，请调整大类或历史窗口。")
        starts = [day for day, keep in zip(loaded["period_starts"], loaded["period_complete"], strict=True) if keep]
        return {"returns": returns, "dates": dates, "period_contiguous": _period_contiguity(starts, dates),
            "metadata": {"requested_start": start, "requested_end": end,
                "actual_start": loaded["period_starts"][0], "actual_end": dates[-1],
                "observations": len(dates), "excluded_return_periods": excluded_periods,
                "excluded_nav_dates": excluded_nav_dates, "source_hash": loaded["source_hash"],
                "lineage": lineage, "warnings": [*loaded["reasons"],
                    *([f"按共同可得的单日收益取样：{excluded_nav_dates} 个日期有资产缺净值，"
                       f"排除跨过它们的 {excluded_periods} 个收益期。"]
                      if excluded_periods else [])]}}

    def _proxies(self, model):
        request = model.proxy_inputs
        proxy_kernels.require_ready()
        loaded, first = {}, []
        for asset in request.assets:
            for component in asset.components:
                key = digest_json(component.model_dump(exclude={"weight"}))
                if key in loaded:
                    continue
                raw = self.sources.load(component, request)
                dates = np.asarray(raw["dates"], dtype="datetime64[D]").astype(np.int64)
                if dates.size < 21 or np.any(dates[1:] <= dates[:-1]):
                    raise ValidationError("LTCMA_PROXY_DATES", "代理日期须完整、有序且不重复。")
                loaded[key] = (raw, dates)
                first.append(raw["dates"][0])
        if not first:
            raise ValidationError("LTCMA_PROXY_REQUIRED", "历史统计至少需要一个非现金研究代理。")
        requested_end = model.window.end_date if model.window.kind == "custom" else model.as_of
        start, end, trading_days = _window(model, max(first), _calendar(self.strategic.data.data_dir, through=requested_end))
        low, high = (int(np.datetime64(bound, "D").astype(np.int64)) for bound in (start, end))
        # Intersection cannot prove the requested window was covered. Check each
        # source's support first; known SSE closures do not require an observation.
        for raw, dates in loaded.values():
            if dates[0] > trading_days[0] or dates[-1] < trading_days[-1]:
                name = raw["identity"].get("name") or raw["identity"]["series_id"]
                raise ValidationError("LTCMA_PROXY_WINDOW_COVERAGE",
                    f"代理 {name} 的可读区间 {raw['dates'][0]} 至 {raw['dates'][-1]} 未覆盖所选窗口 {start} 至 {end} 的边界；"
                    "尚不能确认是休市还是缺失，请补齐数据或明确调整取样窗口，不会自动截短。")
        # 代理各有交易日历：按共同可得的单日收益对齐，不要求逐日覆盖 SSE 开放日。
        window_dates = [dates[(dates >= low) & (dates <= high)] for _, dates in loaded.values()]
        expected, adjacent = common_return_periods(window_dates)
        if expected.size:
            # 所有代理都没有数据的 SSE 开放日不是日历差异，是缺数据：相邻性看不出来，仍然阻断。
            span = trading_days[(trading_days >= expected[0]) & (trading_days <= expected[-1])]
            blind = np.setdiff1d(span, np.unique(np.concatenate(window_dates)), assume_unique=False)
            if blind.size and any(bool(np.isin(dates, trading_days).all()) for dates in window_dates):
                raise ValidationError("LTCMA_PROXY_GAPS",
                    f"共同样本区间内有 {blind.size} 个 SSE 开放日所有代理都没有数据；不会把跨日收益当成单日收益。")
        if expected.size < 21 or int(adjacent.sum()) < 20:
            raise ValidationError("LTCMA_SAMPLE_TOO_SHORT", "各代理共同可得的单日收益不足 20 个，请调整代理或历史窗口。")
        if expected.size > 10000:
            raise ValidationError("LTCMA_SAMPLE_CAPACITY", "单次最多 10000 个共同净值日，请缩短历史窗口。")
        days = expected.astype("datetime64[D]").astype(str).tolist()
        positions, sources = {}, []
        for key, (raw, dates) in loaded.items():
            index = np.searchsorted(dates, expected)
            if any(not raw["available_at"][i] or raw["available_at"][i] > str(model.as_of) for i in index):
                raise ValidationError("LTCMA_PROXY_AVAILABILITY", "代理信息在研究日未知或尚不可得。")
            positions[key] = index
            sources.append(raw["identity"])
        panel = np.empty((expected.size - 1, len(request.assets)), dtype=np.float64)
        for j, asset in enumerate(request.assets):
            if asset.asset_type == "cash":
                panel[:, j] = float(asset.cash_return) / 252.0
                continue
            levels = np.empty((expected.size, len(asset.components)), dtype=np.float64)
            for k, component in enumerate(asset.components):
                key = digest_json(component.model_dump(exclude={"weight"}))
                levels[:, k] = np.asarray(loaded[key][0]["values"], dtype=np.float64)[positions[key]]
            levels.flags.writeable = False
            component_returns = proxy_kernels.adjacent_returns(levels)
            panel[:, j] = proxy_kernels.proxy_returns(component_returns,
                np.asarray([c.weight for c in asset.components], dtype=np.float64),
                _rebalance_reset_flags(days, asset.rebalance))
        # 持有路径按全部共同日推进；只有跨日的收益期不进样本，这一次边界复制之后不再复制。
        excluded_periods = int(adjacent.size - adjacent.sum())
        if excluded_periods:
            panel = np.ascontiguousarray(panel[adjacent])
        sample_days = [day for day, keep in zip(days[1:], adjacent.tolist(), strict=True) if keep]
        sample_starts = [day for day, keep in zip(days[:-1], adjacent.tolist(), strict=True) if keep]
        quality = return_quality(panel, sample_days, model.asset_ids)
        if not np.isfinite(panel).all() or quality["issues"]:
            raise ValidationError("LTCMA_PROXY_VALUES", "研究代理收益存在缺失、断点或无效值，未生成假设。")
        missing_trading_days = int(np.setdiff1d(trading_days, expected).size)
        return {"returns": panel, "dates": sample_days, "period_contiguous": _period_contiguity(sample_starts, sample_days), "metadata": {
            "requested_start": start, "requested_end": end, "actual_start": days[0], "actual_end": days[-1],
            "observations": panel.shape[0], "common_days": int(expected.size),
            "excluded_return_periods": excluded_periods, "missing_trading_days": missing_trading_days,
            "proxy_definition": request.model_dump(mode="json"),
            "source_hash": digest_json(sources), "sources": sources,
            "warnings": ["研究代理按明确的再平衡规则构造；不是最终实施产品。",
                "指数值的价格／全收益含义由所选来源决定；本次不自动补分红或转换币种。",
                *([f"按各代理共同可得的交易日对齐：共同日 {expected.size} 个，排除 {excluded_periods} 个跨日收益期，"
                   f"{missing_trading_days} 个 SSE 开放日不在共同日内。"]
                  if excluded_periods or missing_trading_days else [])]}}

    def _regime_items(self):
        path = self.regime_root / "historical_regime_runs.json"
        guard_path(path)
        if not path.exists():
            return []
        if path.is_symlink() or path.stat().st_size > 64_000_000:
            raise ValidationError("LTCMA_REGIME_STORE", "历史状态存储不可用或超过读取预算。")
        try:
            items = self.runs.read_items()
        except (ValueError, KeyError, OSError, RegimeDomainError) as exc:
            raise ValidationError("LTCMA_REGIME_STORE", "历史状态存储无法解析。") from exc
        if not isinstance(items, list):
            raise ValidationError("LTCMA_REGIME_STORE", "历史状态目录格式无效。")
        return items

    def regime_options(self, as_of=None):
        cutoff = as_of or automatic_research_day(self.strategic.data.data_dir)
        options = []
        for raw in self._regime_items():
            if raw.get("schema_version") != "2.0" or raw.get("mode") != "retrospective":
                continue
            reasons = self.scenarios.historical_reasons(raw, cutoff)
            options.append({**{k: raw.get(k) for k in ("id", "name", "content_hash", "as_of", "frequency", "states")},
                            "available": not reasons, "reasons": reasons})
        return options

    def regime(self, model, evidence, *, include_available_dates=False):
        if self._hydrate_regime is None or self._regime_hash is None:
            raise RuntimeError("LTCMA_REGIME_NOT_READY: 状态读取器尚未完成启动预热。")
        raw = next((x for x in self._regime_items() if x.get("id") == model.run_ref.id), None)
        if raw is None:
            raise NotFoundError("LTCMA_REGIME_NOT_FOUND", "未找到所选历史状态运行版本。")
        if raw.get("content_hash") != model.run_ref.content_hash or self._regime_hash(raw) != raw.get("content_hash"):
            raise ConflictError("LTCMA_REGIME_HASH", "历史状态版本校验不一致，不能生成 CMA。")
        if (raw.get("immutable") is not True or raw.get("schema_version") != "2.0"
                or raw.get("mode") != "retrospective"
                or not raw.get("as_of") or raw["as_of"] > str(model.as_of)):
            raise ValidationError("LTCMA_REGIME_SCOPE", "请选择数据及情景划分均截至研究日的事后研究，不能截短使用未来数据生成的区间。")
        hydrated = self._hydrate_regime(raw, workspace_data_dir=self.regime_root)
        state_ids = [s["id"] for s in raw.get("states", [])]
        if not state_ids or len(state_ids) > 60 or len(set(state_ids)) != len(state_ids):
            raise ValidationError("LTCMA_REGIME_STATES", "状态定义为空、重复或超过数量上限。")
        mapping = {s: i for i, s in enumerate(state_ids)}
        rows, known_at = {}, {}
        for row in hydrated["series"]:
            day = row["observation_date"]
            if day in rows:
                raise ValidationError("LTCMA_REGIME_DATES", "状态日期重复，不能按名称猜测归属。")
            # Full sample information is relevant even when the requested window is shorter.
            for key in ("available_at", "recognized_at"):
                value = row.get(key)
                if value and value[:10] > str(model.as_of):
                    raise ValidationError("LTCMA_REGIME_FUTURE", "状态识别依赖研究日之后的信息。")
            if (not row.get("available_at") or row["available_at"][:10] < day
                    or day > raw["as_of"]):
                raise ValidationError("LTCMA_REGIME_KNOWLEDGE", "状态数据的观察日与可得时间无效。")
            state = row.get("state_id")
            if state in mapping and not row.get("recognized_at"):
                raise ValidationError("LTCMA_REGIME_KNOWLEDGE", "已分类状态缺少识别可得时间。")
            if "state_code" in row and row["state_code"] != mapping.get(state, -1):
                raise ValidationError("LTCMA_REGIME_STATE_CODE", "状态整数编码与冻结标识映射不一致。")
            if state not in mapping and state not in ("unclassified", "unknown", None):
                raise ValidationError("LTCMA_REGIME_STATE_UNKNOWN", "状态序列包含未声明的编码。")
            rows[day] = mapping.get(state, -1)
            known_at[day] = max(row.get("available_at") or "", row.get("recognized_at") or "")[:10] or None
        # Metadata decoding is the only allocation boundary; interval projection
        # uses one warmed readonly-array kernel, without resampling asset returns.
        days = list(rows)
        statistical_kernels.require_ready()
        try:
            observation_days = np.asarray(days, dtype="datetime64[D]").astype(np.int64)
            codes = np.asarray(list(rows.values()), dtype=np.int64)
            availability = np.asarray([known_at[day] for day in days], dtype="datetime64[D]").astype(np.int64)
            target_days = np.asarray(evidence["dates"], dtype="datetime64[D]").astype(np.int64)
            for array in (observation_days, codes, availability, target_days):
                array.flags.writeable = False
            states, aligned_known_at = statistical_kernels.align_regime_intervals(
                observation_days, codes, availability, target_days)
        except ValueError as exc:
            raise ValidationError("LTCMA_REGIME_DATES", "情景日期须按时间排列且不能重复，请重新保存有效的情景区间。") from exc
        states.flags.writeable = False
        audit = {"run_id": raw["id"], "run_hash": raw["content_hash"],
            "as_of": raw["as_of"], "return_assignment": "return_end_state",
            "source_frequency": raw.get("frequency"), "alignment": "closed_observed_state_intervals",
            "state_labels": {s["id"]: s.get("label") or s["id"] for s in raw["states"]},
            "unknown_observations": int(np.count_nonzero(states == -1)), "historical_pit_proven": False}
        if include_available_dates:
            audit["label_available_dates"] = [str(np.datetime64(int(day), "D")) if day >= 0 else None
                                              for day in aligned_known_at]
        return states, state_ids, audit
