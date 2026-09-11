"""Isolated canonical-to-analytics boundary. Never reads the active market directory."""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import pandas as pd

from .models import CenterError


def build_snapshot(directory: Path, inputs: dict[str, Path], config_dir: Path) -> dict:
    """Project explicit resolved inputs into a private compatibility workspace."""
    # The legacy numerical modules are rooted at backend/, as in app.py.
    backend_dir = str(Path(__file__).resolve().parents[1])
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from backend.services.instrument_analytics import rebuild_analytics_snapshot

    master = pd.read_parquet(inputs["master.instrument"])
    nav = pd.read_parquet(inputs["market.nav_daily"])
    master = master.sort_values("valid_from").drop_duplicates("instrument_id", keep="last")
    kinds = master.set_index("instrument_id")["instrument_type"].map({"ETF": "etf", "FUND_SHARE": "fund"})
    names = master.set_index("instrument_id")["canonical_name"]
    nav["instrument_type"] = nav["instrument_id"].map(kinds)
    if nav.empty or nav["instrument_type"].isna().any():
        raise CenterError("SNAPSHOT_IDENTITY_REQUIRED", "净值标的缺少已合并的 ETF/基金份额主数据，无法计算快照。")
    if nav.duplicated(["instrument_id", "valuation_date"]).any():
        raise CenterError("SNAPSHOT_BASIS_CONFLICT", "同一标的同一天存在多个币种或复权口径，请先缩小取值范围。")
    directory.mkdir(parents=True, exist_ok=True)
    # Internal IDs are the exact join key throughout this isolated workspace.
    rename = {"instrument_id": "ts_code", "valuation_date": "date", "adjusted_nav": "adj_nav", "accumulated_nav": "accum_nav"}
    for kind, filename in (("etf", "etf_daily_df.parquet"), ("fund", "fund_nav_df.parquet")):
        part = nav.loc[nav.instrument_type.eq(kind)].rename(columns=rename)
        if part.empty:
            continue
        part["nav_date"] = part["date"]
        # Missing publication times remain missing; no fake historical dates.
        part["ann_date"] = pd.to_datetime(part["announced_at"], utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        part["date"] = pd.to_datetime(part["date"])
        part = part.sort_values(["ts_code", "date"])
        part.to_parquet(directory / filename, index=False)
        info = part[["ts_code"]].drop_duplicates()
        info["code"] = info["ts_code"]
        info["name"] = info["ts_code"].map(names)
        info["instrument_type"] = kind
        info.to_parquet(directory / ("etf_info_df.parquet" if kind == "etf" else "fund_info_df.parquet"), index=False)
    if "market.quote_daily" in inputs:
        quotes = pd.read_parquet(inputs["market.quote_daily"])
        quotes = quotes.loc[quotes.instrument_id.map(kinds).eq("etf")]
        if not quotes.empty:
            if not quotes.adjustment_basis.eq("RAW").all() or quotes.duplicated(["instrument_id", "trade_date"]).any():
                raise CenterError("SNAPSHOT_QUOTE_BASIS", "分析快照要求每个 ETF 每日唯一的未复权行情。")
            quotes = quotes.rename(columns={"instrument_id": "ts_code", "trade_date": "date", "return_decimal": "pct_chg"})
            quotes["date"] = pd.to_datetime(quotes["date"])
            # Convert canonical base units back to the existing reader contract.
            quotes["vol"] = quotes["volume"] / 100
            quotes["amount"] = pd.to_numeric(quotes["turnover_amount"], errors="raise") / 1000
            quotes["pct_chg"] = quotes["pct_chg"] * 100
            quotes.sort_values(["ts_code", "date"]).to_parquet(directory / "etf_daily_candle_df.parquet", index=False)
    if "master.trading_calendar" in inputs:
        calendar = pd.read_parquet(inputs["master.trading_calendar"]).rename(columns={"calendar_code": "exchange", "calendar_date": "cal_date"})
        calendar["cal_date"] = pd.to_datetime(calendar["cal_date"])
        calendar.to_parquet(directory / "trade_day_df.parquet", index=False)
    result = rebuild_analytics_snapshot(directory, workspace_data_dir=config_dir)
    if not result.get("rows"):
        raise CenterError("SNAPSHOT_NO_DATA", "没有可构建快照的净值记录；单位净值不会冒充复权净值。")
    result.update(published=False, input_tables=list(inputs), identity_basis="internal_instrument_id")
    return result


def main():
    payload = json.load(sys.stdin)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = build_snapshot(Path(payload["directory"]), {k: Path(v) for k, v in payload["inputs"].items()}, Path(payload["config_dir"]))
        answer = {"ok": True, "result": result}
    except CenterError as exc:
        answer = {"ok": False, "code": exc.code, "message": exc.message}
    except Exception as exc:
        # Data files may contain sensitive values; only expose the exception type.
        answer = {"ok": False, "code": "SNAPSHOT_FAILED", "message": f"快照计算未完成（{type(exc).__name__}），已保留前置步骤，可修复后继续。"}
    print(json.dumps(answer, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
