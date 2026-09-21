"""Read-only, explicit-date evidence for LTCMA, not an implementation-product map."""
from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from backend.custom_indicators.errors import ConflictError, NotFoundError, ValidationError
from backend.market_data import resolve_market_data_file
from backend.research_input_checks import return_quality
from backend.sensitivity.repository import digest_json
from backend.data_storage import guard_path
from .contracts import RiskReferenceRequest
from .reference_inputs import _rebalance_reset_flags, array_digest, automatic_research_day
from . import reference_evidence_kernels as proxy_kernels


def _year_before(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(year=day.year - years, day=28)


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
        self._hydrate_regime = None
        self._regime_hash = None

    def prepare_regime_reader(self):
        from backend.historical_regimes.v2_service import _stored_run_snapshot_hash, hydrate_v2_run_snapshot
        self._hydrate_regime, self._regime_hash = hydrate_v2_run_snapshot, _stored_run_snapshot_hash

    def build(self, request, source: dict) -> dict:
        model = request.model
        if model.as_of > automatic_research_day(self.strategic.data.data_dir):
            raise ValidationError("LTCMA_KNOWLEDGE_CUTOFF", "LTCMA 研究日晚于平台知识截止日。")
        if request.strategic_universe_id:
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
            "moment_semantics": "annualized_periodic_arithmetic", "currency": request.currency,
            "currency_basis": "user_confirmed_same_currency_no_automatic_conversion"})
        if isinstance(result["returns"], np.ndarray):
            result["returns"].flags.writeable = False
        result["metadata"]["return_panel_hash"] = array_digest(result["returns"])
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
        self.strategic._validate_daily_risk_axis(reference, loaded)
        lineage = loaded["lineage"]
        if lineage["excluded_incomplete_dates"] or lineage["missing_availability_rows"]:
            raise ValidationError("LTCMA_INCOMPLETE_EVIDENCE", "历史日期或可得时间缺失，不能静默删日后计算收益。")
        quality = return_quality(loaded["returns"], loaded["dates"], model.asset_ids)
        if quality["issues"]:
            raise ValidationError("LTCMA_RETURN_QUALITY", quality["issues"][0]["message"], diagnostics=quality["issues"])
        return {"returns": loaded["returns"], "dates": loaded["dates"],
            "metadata": {"requested_start": start, "requested_end": end,
                "actual_start": loaded["period_starts"][0], "actual_end": loaded["dates"][-1],
                "observations": len(loaded["dates"]), "source_hash": loaded["source_hash"],
                "lineage": lineage, "warnings": loaded["reasons"]}}

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
        start, end, expected = _window(model, max(first), _calendar(self.strategic.data.data_dir, through=requested_end))
        days = expected.astype("datetime64[D]").astype(str).tolist()
        positions, sources = {}, []
        for key, (raw, dates) in loaded.items():
            index = np.searchsorted(dates, expected)
            if np.any(index >= dates.size) or not np.array_equal(dates[index], expected):
                raise ValidationError("LTCMA_PROXY_GAPS", "代理未完整覆盖所选 SSE 交易日；不会缩短窗口或拼接跨日收益。")
            # Each requested return must be an actual adjacent source observation.
            if np.any(np.diff(index) != 1):
                raise ValidationError("LTCMA_PROXY_CALENDAR", "代理含不同交易日历的间隔，当前 SSE 日频适配不支持。")
            if any(not raw["available_at"][i] or raw["available_at"][i] > str(model.as_of) for i in index):
                raise ValidationError("LTCMA_PROXY_AVAILABILITY", "代理信息在研究日未知或尚不可得。")
            positions[key] = (int(index[0]), int(index[-1]) + 1)
            sources.append(raw["identity"])
        panel = np.empty((expected.size - 1, len(request.assets)), dtype=np.float64)
        for j, asset in enumerate(request.assets):
            if asset.asset_type == "cash":
                panel[:, j] = float(asset.cash_return) / 252.0
                continue
            levels = np.empty((expected.size, len(asset.components)), dtype=np.float64)
            for k, component in enumerate(asset.components):
                key = digest_json(component.model_dump(exclude={"weight"}))
                left, right = positions[key]
                levels[:, k] = loaded[key][0]["values"][left:right]
            levels.flags.writeable = False
            component_returns = proxy_kernels.adjacent_returns(levels)
            panel[:, j] = proxy_kernels.proxy_returns(component_returns,
                np.asarray([c.weight for c in asset.components], dtype=np.float64),
                _rebalance_reset_flags(days, asset.rebalance))
        quality = return_quality(panel, days[1:], model.asset_ids)
        if not np.isfinite(panel).all() or quality["issues"]:
            raise ValidationError("LTCMA_PROXY_VALUES", "研究代理收益存在缺失、断点或无效值，未生成假设。")
        return {"returns": panel, "dates": days[1:], "metadata": {
            "requested_start": start, "requested_end": end, "actual_start": days[0], "actual_end": days[-1],
            "observations": panel.shape[0], "proxy_definition": request.model_dump(mode="json"),
            "source_hash": digest_json(sources), "sources": sources,
            "warnings": ["研究代理按明确的再平衡规则构造；不是最终实施产品。",
                "指数值的价格／全收益含义由所选来源决定；本次不自动补分红或转换币种。"]}}

    def _regime_items(self):
        path = self.regime_root / "historical_regime_runs.json"
        guard_path(path)
        if not path.exists():
            return []
        if path.is_symlink() or path.stat().st_size > 64_000_000:
            raise ValidationError("LTCMA_REGIME_STORE", "历史状态存储不可用或超过读取预算。")
        try:
            items = json.loads(path.read_text(encoding="utf-8"))["items"]
        except (ValueError, KeyError, OSError) as exc:
            raise ValidationError("LTCMA_REGIME_STORE", "历史状态存储无法解析。") from exc
        if not isinstance(items, list):
            raise ValidationError("LTCMA_REGIME_STORE", "历史状态目录格式无效。")
        return items

    def regime_options(self):
        return [{k: x.get(k) for k in ("id", "name", "content_hash", "as_of", "frequency", "states")}
                for x in self._regime_items() if x.get("schema_version") == "2.0"
                and x.get("mode") == "retrospective" and x.get("immutable") is True]

    def regime(self, model, evidence):
        if self._hydrate_regime is None or self._regime_hash is None:
            raise RuntimeError("LTCMA_REGIME_NOT_READY: 状态读取器尚未完成启动预热。")
        raw = next((x for x in self._regime_items() if x.get("id") == model.run_ref.id), None)
        if raw is None:
            raise NotFoundError("LTCMA_REGIME_NOT_FOUND", "未找到所选历史状态运行版本。")
        if raw.get("content_hash") != model.run_ref.content_hash or self._regime_hash(raw) != raw.get("content_hash"):
            raise ConflictError("LTCMA_REGIME_HASH", "历史状态版本校验不一致，不能生成 CMA。")
        if (raw.get("immutable") is not True or raw.get("schema_version") != "2.0"
                or raw.get("mode") != "retrospective" or raw.get("frequency") != "daily"
                or not raw.get("as_of") or raw["as_of"] > str(model.as_of)):
            raise ValidationError("LTCMA_REGIME_SCOPE", "请选择研究日前可得的日频事后状态，不把未来划分截短后当历史知识。")
        hydrated = self._hydrate_regime(raw, workspace_data_dir=self.regime_root)
        state_ids = [s["id"] for s in raw.get("states", [])]
        if not state_ids or len(state_ids) > 60 or len(set(state_ids)) != len(state_ids):
            raise ValidationError("LTCMA_REGIME_STATES", "状态定义为空、重复或超过数量上限。")
        mapping = {s: i for i, s in enumerate(state_ids)}
        rows = {}
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
        states = np.asarray([rows.get(day, -1) for day in evidence["dates"]], dtype=np.int64)
        states.flags.writeable = False
        return states, state_ids, {"run_id": raw["id"], "run_hash": raw["content_hash"],
            "as_of": raw["as_of"], "return_assignment": "return_end_state",
            "state_labels": {s["id"]: s.get("label") or s["id"] for s in raw["states"]},
            "unknown_observations": int(np.count_nonzero(states == -1)), "historical_pit_proven": False}
