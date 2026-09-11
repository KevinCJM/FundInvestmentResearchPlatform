"""Read existing data adapters once, align on a complete calendar, freeze inputs."""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

from backend.custom_indicators.errors import ValidationError
from backend.custom_indicators.series_provider import resolve_identity
from backend.data_storage import guard_path
from historical_regimes.data import resolve_target
from backend.market_data import resolve_tushare_data_dir
from backend.trading_calendar import _load_calendar_file
from .catalog import TRANSFORM_CODES, VariableRegistry, decorate
from .kernels import resample_transform_kernel
from .repository import digest_json

PERIOD_CODES = {"daily": "D", "weekly": "W-FRI", "monthly": "M", "quarterly": "Q-DEC"}


def _days(dates):
    return np.ascontiguousarray(pd.DatetimeIndex(dates).to_numpy(dtype="datetime64[D]").astype(np.int64))


def _periods(dates, frequency):
    return np.ascontiguousarray(pd.DatetimeIndex(dates).to_period(PERIOD_CODES[frequency]).asi8)


@lru_cache(maxsize=8)
def _calendar_last_known_day(path_text: str, mtime_ns: int, size: int):
    """Calendar coverage includes known closed days, not only trading sessions."""
    del mtime_ns, size
    frame = pd.read_parquet(path_text, columns=["exchange", "cal_date"])
    selected = frame.loc[frame["exchange"].str.upper() == "SSE", "cal_date"]
    parsed = pd.to_datetime(selected.astype(str), format="%Y%m%d", errors="coerce")
    if parsed.empty or parsed.isna().any():
        raise ValidationError("RISK_CALENDAR_COVERAGE", "交易日历包含无效日期或没有上交所记录。")
    return parsed.max()


def canonical_product(raw, root):
    identity = resolve_identity(raw["kind"], raw["product_id"], root)
    return {"kind": identity.kind, "product_id": identity.ts_code, "name": identity.name,
            "key": f"{identity.kind}:{identity.ts_code}"}


def product_variable(product):
    return decorate({"id": product["key"], "name": product["name"], "roles": [],
        "unit": "return", "frequency": "daily", "transform": "price_return",
        "source": {"kind": product["kind"], "ts_code": product["product_id"], "field": "adj_nav"},
        "basis": "adjusted_nav", "market": "CN", "currency": "CNY", "revision": 1})


class ResearchData:
    def __init__(self, data_dir: Path, variables: VariableRegistry):
        self.data_dir = Path(data_dir)
        self.variables = variables

    def _observations(self, variable, root, as_of):
        source = variable["source"]
        if not source:
            raise ValidationError("RISK_SERIES_NOT_AVAILABLE", f'「{variable["name"]}」没有历史时序，不能训练回归模型。')
        if source["kind"] == "import":
            item = self.variables.imports.get(source["artifact_id"], "series")
            arrays = self.variables.imports.arrays(item["id"])
            return arrays["days"], arrays["available"], arrays["values"], {
                "source_id": item["id"], "content_hash": item["content_hash"], "pit_verified": False,
                "availability_basis": item["availability_basis"], "source_label": item["source_label"],
            }
        spec = {key: value for key, value in source.items() if key != "filename"}
        spec["availability_mode"] = "latest"
        # Existing canonical readers own revision selection and announcement handling.
        if source["kind"] == "macro":
            from historical_regimes.v2_service import _macro_bundle
            bundle = _macro_bundle(spec, "retrospective", as_of, root)
        else:
            bundle = resolve_target(spec, "retrospective", as_of, root)
        frame = bundle.frame
        if len(frame) > 30_000:
            raise ValidationError("RISK_SERIES_TOO_LARGE", "单条研究序列超过 30,000 个观察值，请缩小数据源范围。")
        days = _days(frame["observation_date"])
        available = _days(frame["available_at"])
        values = np.ascontiguousarray(frame["value"].to_numpy(dtype=np.float64))
        return days, available, values, {**bundle.snapshot, "pit_verified": False,
            "historical_pit_note": "按截止日过滤公告，但当前修订快照未认证完整历史 vintage；仅供当时发布后的研究使用。"}

    def load(self, definition):
        guard_path(self.data_dir)
        root = resolve_tushare_data_dir(self.data_dir, strict=True)
        inputs = [self.variables.get(key) for key in definition["inputs"]]
        targets = [canonical_product(item, root) for item in definition["targets"]]
        if len({item["key"] for item in targets}) != len(targets):
            raise ValidationError("RISK_DUPLICATE_PRODUCT", "研究对象解析后重复：同一产品的简称与完整代码只能选择一次。")
        outputs = ([product_variable(item) for item in targets] if targets else
                   [self.variables.get(key) for key in definition["outputs"]])
        variables = inputs + outputs
        frequency = definition["frequency"]
        for variable in variables:
            native = variable["frequency"]
            allowed = ({"daily", "weekly", "monthly", "quarterly"} if native == "daily" else
                       {"monthly", "quarterly"} if native == "monthly" else {native})
            if frequency not in allowed or (variable["transform"] == "identity" and frequency != native):
                raise ValidationError("RISK_FREQUENCY_MISMATCH", f'「{variable["name"]}」不能用于所选频率；不插值或重复累计低频数据。')
        start = pd.Timestamp(definition["start_date"])
        end = pd.Timestamp(definition["end_date"])
        first_period = start.to_period(PERIOD_CODES[frequency]) - 1
        final_period = end.to_period(PERIOD_CODES[frequency])
        if frequency != "daily" and final_period.end_time.normalize() > end:
            final_period -= 1
        if final_period < first_period:
            raise ValidationError("RISK_EMPTY_WINDOW", "所选区间没有完整的研究周期。")
        period_grid = pd.period_range(first_period, final_period, freq=PERIOD_CODES[frequency])
        grid = np.ascontiguousarray(period_grid.asi8)
        period_ends = period_grid.end_time.normalize()
        expected = _days(period_ends)
        native_daily = any(item["frequency"] == "daily" for item in variables)
        if native_daily:
            path = root / "trade_day_df.parquet"
            if not path.is_file():
                raise ValidationError("RISK_CALENDAR_MISSING", "缺少上交所交易日历，无法确认收益期末；请先同步交易日历。")
            stat = path.stat()
            calendar = _load_calendar_file(str(path), stat.st_mtime_ns, stat.st_size, "SSE")
            selected = calendar[(calendar >= first_period.start_time) & (calendar <= end)]
            required_end = end if frequency == "daily" else final_period.end_time.normalize()
            known_through = _calendar_last_known_day(str(path), stat.st_mtime_ns, stat.st_size)
            if selected.empty or known_through < required_end:
                raise ValidationError("RISK_CALENDAR_COVERAGE", "交易日历未覆盖完整研究区间。")
            if frequency == "daily":
                grid = _periods(selected, frequency)
                period_ends = selected
                expected = _days(selected)
            else:
                by_period = {}
                for item in selected:
                    by_period[item.to_period(PERIOD_CODES[frequency]).ordinal] = item
                expected = np.array([np.datetime64(by_period.get(int(key), pd.NaT), "D").astype(np.int64)
                                     for key in grid], dtype=np.int64)
        if len(grid) < 3 or len(grid) > 20_000:
            raise ValidationError("RISK_WINDOW_SIZE", "研究区间须包含 3 至 20,000 个完整周期。")
        cutoff = np.int64(np.datetime64(definition["as_of"], "D").astype(np.int64))
        columns, knowledge, raw_columns, provenance = [], [], [], []
        for variable in variables:
            days, available, values, source_meta = self._observations(variable, root, definition["as_of"])
            dates = pd.DatetimeIndex(days.astype("datetime64[D]"))
            periods = _periods(dates, frequency)
            if variable["frequency"] == "monthly" and frequency == "quarterly":
                # Only the terminal month's observation may represent a quarter.
                periods = periods.copy()
                periods[dates.month % 3 != 0] = np.iinfo(np.int64).min
            transformed, known, raw = resample_transform_kernel(
                days, periods, available, values, grid, expected, cutoff,
                np.int64(TRANSFORM_CODES[variable["transform"]]),
                np.int64(variable["frequency"] == "daily"),
            )
            columns.append(transformed)
            knowledge.append(known)
            raw_columns.append(raw)
            provenance.append({"variable_id": variable["id"], **source_meta})
        # Single matrix allocation at the data-to-compute boundary, not per node.
        values = np.ascontiguousarray(np.column_stack(columns))
        available_matrix = np.ascontiguousarray(np.column_stack(knowledge), dtype=np.int64)
        raw_values = np.ascontiguousarray(np.column_stack(raw_columns))
        dates = _days(period_ends)
        start_index = int(np.searchsorted(dates, np.datetime64(definition["start_date"], "D").astype(np.int64)))
        validation_index = int(np.searchsorted(dates, np.datetime64(definition["validation_start"], "D").astype(np.int64)))
        return {"values": values, "available": available_matrix, "raw_values": raw_values,
            "dates": dates, "start_index": start_index, "validation_index": validation_index,
            "inputs": inputs, "outputs": outputs, "targets": targets, "provenance": provenance,
            "source_snapshot": str(root), "input_identity": digest_json(provenance),
            "calendar": {"frequency": frequency, "complete_periods_only": True,
                         "alignment": "full_calendar_no_fill", "market_endpoint": "SSE" if native_daily else None}}
