import importlib.util
import json
import sys
import threading
import time
import types
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

try:
    import tushare  # noqa: F401
except ModuleNotFoundError:
    sys.modules["tushare"] = types.SimpleNamespace(
        set_token=lambda *_args, **_kwargs: None,
        pro_api=lambda *_args, **_kwargs: None,
    )


def _load_data_script():
    spec = importlib.util.spec_from_file_location("tushare_data_script", ROOT / "T01_get_data.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_config_reads_frontend_credential_file_and_ignores_environment(monkeypatch, tmp_path: Path) -> None:
    credential = tmp_path / ".tushare_token"
    monkeypatch.setenv("TUSHARE_TOKEN", "legacy-environment-token")
    monkeypatch.setattr(config, "TUSHARE_CREDENTIAL_PATH", credential)

    with pytest.raises(RuntimeError, match="主界面"):
        config.require_tushare_token()

    credential.write_text("frontend-token-1234567890\n", encoding="utf-8")
    assert config.require_tushare_token() == "frontend-token-1234567890"


def test_run_actions_passes_frontend_token_directly_without_writing_tushare_home_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_data_script()
    calls: list[str] = []

    def reject_persistent_token(*_args, **_kwargs):
        raise AssertionError("ts.set_token 会写入用户主目录 tk.csv")

    monkeypatch.setattr(module, "require_tushare_token", lambda: "frontend-token")
    monkeypatch.setattr(
        module,
        "ts",
        types.SimpleNamespace(
            set_token=reject_persistent_token,
            pro_api=lambda token: calls.append(token) or object(),
        ),
    )
    args = types.SimpleNamespace(
        latest=False,
        output_dir=tmp_path,
        max_calls_per_minute=100,
        min_call_interval_sec=0,
        start_date="20260831",
        end_date="20260901",
    )

    module._run_actions(args, [])

    assert calls == ["frontend-token"]


def test_build_etf_info_df_uses_tushare_schema_and_project_units() -> None:
    module = _load_data_script()
    fund_df = pd.DataFrame(
        [
            {
                "ts_code": "510050.SH",
                "name": "上证50ETF",
                "management": "华夏基金",
                "custodian": "中国银行",
                "fund_type": "股票型",
                "found_date": "20040205",
                "list_date": "20040223",
                "issue_date": "20040210",
                "issue_amount": 45.0,
                "m_fee": 0.5,
                "c_fee": 0.1,
                "status": "L",
                "market": "E",
                "benchmark": None,
                "invest_type": "被动指数型",
                "type": "契约型开放式",
            }
        ]
    )
    etf_df = pd.DataFrame(
        [
            {
                "ts_code": "510050.SH",
                "index_code": "000016.SH",
                "index_name": "上证50指数",
                "exchange": "SH",
                "list_status": "L",
                "mgr_name": "华夏基金管理有限公司",
                "custod_name": "中国工商银行",
                "etf_type": "宽基ETF",
            }
        ]
    )

    out = module.build_etf_info_df(fund_df, etf_df)

    assert list(out["ts_code"]) == ["510050.SH"]
    row = out.iloc[0]
    assert row["code"] == "510050"
    assert row["market"] == "上交所"
    assert row["market_code"] == "SH"
    assert row["status"] == "上市交易"
    assert row["status_code"] == "L"
    assert row["benchmark"] == "上证50指数"
    assert row["index_code"] == "000016.SH"
    assert row["issue_amount"] == 450000.0
    assert row["m_fee"] == 0.5
    assert row["c_fee"] == 0.1
    assert str(row["list_date"].date()) == "2004-02-23"
    assert row["management"] == "华夏基金"
    assert row["custodian"] == "中国银行"


def test_build_etf_info_df_falls_back_to_code_suffix_without_etf_basic() -> None:
    module = _load_data_script()
    fund_df = pd.DataFrame(
        [
            {
                "ts_code": "159915.SZ",
                "name": "创业板ETF",
                "management": "易方达基金",
                "custodian": "中国建设银行",
                "fund_type": "股票型",
                "found_date": "20110520",
                "list_date": "20110613",
                "issue_amount": 32.0,
                "status": "L",
                "market": "E",
            }
        ]
    )

    out = module.build_etf_info_df(fund_df)

    row = out.iloc[0]
    assert row["code"] == "159915"
    assert row["market"] == "深交所"
    assert row["status"] == "上市交易"
    assert row["issue_amount"] == 320000.0


def test_build_public_fund_info_keeps_off_exchange_domain_separate() -> None:
    module = _load_data_script()
    fund_df = pd.DataFrame(
        [
            {
                "ts_code": "000001.OF",
                "name": "华夏成长",
                "management": "华夏基金",
                "fund_type": "混合型",
                "status": "L",
                "market": "O",
                "found_date": "20011218",
            }
        ]
    )

    out = module.build_public_fund_info_df(fund_df)

    row = out.iloc[0]
    assert row["instrument_type"] == "fund"
    assert row["market_code"] == "O"
    assert row["market"] == "场外"
    assert row["status"] == "存续"


def test_adjusted_nav_normalization_preserves_rows_and_nulls_unusable_values() -> None:
    module = _load_data_script()

    values = module.normalise_adj_nav(pd.Series([1.0, "2.0", 0.0, -1.0, float("inf"), None]))

    assert values.iloc[:2].tolist() == [1.0, 2.0]
    assert values.iloc[2:].isna().all()


def test_filter_fund_basic_to_etfs_uses_etf_basic_as_authority() -> None:
    module = _load_data_script()
    fund_df = pd.DataFrame(
        [
            {"ts_code": "500001.SH", "name": "国泰金泰封闭"},
            {"ts_code": "510050.SH", "name": "上证50ETF"},
        ]
    )
    etf_df = pd.DataFrame([{"ts_code": "510050.SH", "index_name": "上证50指数"}])

    out = module.filter_fund_basic_to_etfs(fund_df, etf_df)

    assert list(out["ts_code"]) == ["510050.SH"]


def test_missing_only_helpers_filter_and_merge_existing_rows(tmp_path: Path) -> None:
    module = _load_data_script()
    existing_path = tmp_path / "etf_daily_df.parquet"
    existing = pd.DataFrame(
        [
            {"ts_code": "510050.SH", "date": pd.Timestamp("2024-01-01"), "adj_nav": 1.0},
        ]
    )
    existing.to_parquet(existing_path, index=False)
    universe = pd.DataFrame(
        [
            {"ts_code": "510050.SH", "name": "上证50ETF"},
            {"ts_code": "159915.SZ", "name": "创业板ETF"},
        ]
    )

    missing = module.filter_missing_universe(universe, existing_path, label="fund_nav")
    assert list(missing["ts_code"]) == ["159915.SZ"]

    new_rows = pd.DataFrame(
        [
            {"ts_code": "159915.SZ", "date": pd.Timestamp("2024-01-02"), "adj_nav": 2.0},
        ]
    )
    merged = module.merge_existing_rows(
        new_rows,
        existing_path,
        subset=["ts_code", "date"],
        sort_cols=["ts_code", "date"],
    )
    assert set(merged["ts_code"]) == {"510050.SH", "159915.SZ"}


def test_filter_missing_universe_streams_code_column_without_pandas_full_read(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_data_script()
    existing_path = tmp_path / "fund_nav_df.parquet"
    pd.DataFrame(
        {
            "ts_code": ["000001.OF", "000001.OF", "000002.OF"],
            "date": pd.to_datetime(["2026-08-27", "2026-08-28", "2026-08-28"]),
            "adj_nav": [1.0, 1.01, 2.0],
        }
    ).to_parquet(existing_path, index=False)
    universe = pd.DataFrame(
        {
            "ts_code": ["000001.OF", "000002.OF", "000003.OF"],
            "name": ["基金一", "基金二", "基金三"],
        }
    )

    monkeypatch.setattr(
        module.pd,
        "read_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("不应整表读取")),
    )

    missing = module.filter_missing_universe(universe, existing_path, label="fund_nav")

    assert missing["ts_code"].tolist() == ["000003.OF"]


def test_save_dataframe_is_atomic_when_parquet_write_fails(monkeypatch, tmp_path: Path) -> None:
    module = _load_data_script()
    path = tmp_path / "data.parquet"
    original = pd.DataFrame([{"value": 1}])
    original.to_parquet(path, index=False)

    def fail_write(*_args, **_kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)
    with pytest.raises(RuntimeError, match="simulated"):
        module.save_dataframe(pd.DataFrame([{"value": 2}]), path)

    assert pd.read_parquet(path).to_dict("records") == [{"value": 1}]
    assert list(tmp_path.glob(".data.parquet.*.tmp")) == []


def test_fetch_latest_dates_uses_one_request_per_date_and_filters_universe() -> None:
    module = _load_data_script()
    calls = []

    def fake_api(**kwargs):
        calls.append(kwargs)
        date = kwargs["trade_date"]
        return pd.DataFrame(
            [
                {"ts_code": "510050.SH", "trade_date": date, "close": 2.5},
                {"ts_code": "NOT_ETF.SH", "trade_date": date, "close": 9.9},
            ]
        )

    args = types.SimpleNamespace(max_retries=1, backoff_sec=0.0, wait_on_rate_limit_sec=0.0)
    universe = pd.DataFrame([{"ts_code": "510050.SH", "name": "上证50ETF"}])
    frames = module.fetch_latest_dates(
        api_func=fake_api,
        api_name="fund_daily",
        date_param="trade_date",
        dates=["20260827", "20260828"],
        fields=module.FUND_DAILY_FIELDS,
        universe=universe,
        limiter=module.RateLimiter(1000),
        args=args,
    )

    assert [call["trade_date"] for call in calls] == ["20260827", "20260828"]
    assert [frame["ts_code"].tolist() for frame in frames] == [["510050.SH"], ["510050.SH"]]
    assert frames[0].iloc[0]["name"] == "上证50ETF"


def test_fetch_latest_dates_uses_bounded_thread_pool_and_preserves_date_order() -> None:
    module = _load_data_script()
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_api(**kwargs):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with state_lock:
            active -= 1
        date = kwargs["trade_date"]
        return pd.DataFrame([{"ts_code": "510050.SH", "trade_date": date, "close": 2.5}])

    args = types.SimpleNamespace(
        max_retries=1,
        backoff_sec=0.0,
        wait_on_rate_limit_sec=0.0,
        max_workers=3,
    )
    universe = pd.DataFrame([{"ts_code": "510050.SH", "name": "上证50ETF"}])
    dates = ["20260825", "20260826", "20260827", "20260828"]

    frames = module.fetch_latest_dates(
        api_func=fake_api,
        api_name="fund_daily",
        date_param="trade_date",
        dates=dates,
        fields=module.FUND_DAILY_FIELDS,
        universe=universe,
        limiter=module.RateLimiter(1000),
        args=args,
    )

    assert 2 <= max_active <= 3
    assert [frame.iloc[0]["trade_date"] for frame in frames] == dates


def test_save_latest_candles_merges_missing_trade_days(tmp_path: Path) -> None:
    module = _load_data_script()
    pd.DataFrame(
        [{"ts_code": "510050.SH", "name": "上证50ETF", "date": pd.Timestamp("2026-08-26"), "trade_date": "20260826", "close": 2.4}]
    ).to_parquet(tmp_path / "etf_daily_candle_df.parquet", index=False)
    pd.DataFrame([{"ts_code": "510050.SH", "name": "上证50ETF"}]).to_parquet(
        tmp_path / "etf_info_df.parquet", index=False
    )
    pd.DataFrame(
        [
            {"exchange": "SSE", "cal_date": "20260827", "is_open": 1},
            {"exchange": "SSE", "cal_date": "20260828", "is_open": 1},
        ]
    ).to_parquet(tmp_path / "trade_day_df.parquet", index=False)

    class Pro:
        @staticmethod
        def fund_daily(**kwargs):
            date = kwargs["trade_date"]
            return pd.DataFrame([{"ts_code": "510050.SH", "trade_date": date, "close": 2.5}])

    args = types.SimpleNamespace(
        end_date="20260828",
        max_latest_days=10,
        max_retries=1,
        backoff_sec=0.0,
        wait_on_rate_limit_sec=0.0,
        limit=None,
    )
    module.save_latest_candles(Pro(), tmp_path, module.RateLimiter(1000), args)

    out = pd.read_parquet(tmp_path / "etf_daily_candle_df.parquet")
    assert out["trade_date"].tolist() == ["20260826", "20260827", "20260828"]
    assert out["date"].max() == pd.Timestamp("2026-08-28")


def test_incremental_start_date_defaults_to_five_recent_trading_days(tmp_path: Path) -> None:
    module = _load_data_script()
    pd.DataFrame(
        [
            {"exchange": "SSE", "cal_date": date, "is_open": 1}
            for date in ["20260820", "20260821", "20260824", "20260825", "20260826", "20260827", "20260828"]
        ]
        + [
            {"exchange": "SSE", "cal_date": "20260823", "is_open": 0},
            {"exchange": "SZSE", "cal_date": "20260819", "is_open": 1},
        ]
    ).to_parquet(tmp_path / "trade_day_df.parquet", index=False)

    assert module.incremental_start_date(tmp_path, pd.Timestamp("2026-08-28")) == "20260824"


def test_save_latest_candles_refetches_default_overlap_and_applies_revision(tmp_path: Path) -> None:
    module = _load_data_script()
    pd.DataFrame(
        [
            {
                "ts_code": "510050.SH",
                "name": "上证50ETF",
                "date": pd.Timestamp("2026-08-28"),
                "trade_date": "20260828",
                "close": 1.0,
            }
        ]
    ).to_parquet(tmp_path / "etf_daily_candle_df.parquet", index=False)
    pd.DataFrame([{"ts_code": "510050.SH", "name": "上证50ETF"}]).to_parquet(
        tmp_path / "etf_info_df.parquet", index=False
    )
    expected_dates = ["20260824", "20260825", "20260826", "20260827", "20260828", "20260831"]
    pd.DataFrame(
        [{"exchange": "SSE", "cal_date": date, "is_open": 1} for date in expected_dates]
    ).to_parquet(tmp_path / "trade_day_df.parquet", index=False)
    calls = []

    class Pro:
        @staticmethod
        def fund_daily(**kwargs):
            calls.append(kwargs["trade_date"])
            return pd.DataFrame(
                [{"ts_code": "510050.SH", "trade_date": kwargs["trade_date"], "close": 2.5}]
            )

    args = types.SimpleNamespace(
        end_date="20260831",
        max_latest_days=10,
        max_retries=1,
        backoff_sec=0.0,
        wait_on_rate_limit_sec=0.0,
        incremental_batch_days=2,
        limit=None,
    )
    module.save_latest_candles(Pro(), tmp_path, module.RateLimiter(1000), args)

    out = pd.read_parquet(tmp_path / "etf_daily_candle_df.parquet")
    assert calls == expected_dates
    assert out["trade_date"].tolist() == expected_dates
    assert out.loc[out["trade_date"] == "20260828", "close"].item() == 2.5


def test_latest_etf_share_bootstraps_five_days_when_dataset_is_new(tmp_path: Path) -> None:
    module = _load_data_script()
    pd.DataFrame([{"ts_code": "510050.SH", "name": "上证50ETF"}]).to_parquet(
        tmp_path / "etf_info_df.parquet", index=False
    )
    dates = ["20260824", "20260825", "20260826", "20260827", "20260828", "20260831"]
    pd.DataFrame(
        [{"exchange": "SSE", "cal_date": date, "is_open": 1} for date in dates]
    ).to_parquet(tmp_path / "trade_day_df.parquet", index=False)
    calls: list[str] = []

    class Pro:
        @staticmethod
        def etf_share_size(**kwargs):
            calls.append(kwargs["trade_date"])
            if kwargs.get("offset", 0) > 0:
                return pd.DataFrame()
            return pd.DataFrame([{
                "ts_code": "510050.SH",
                "trade_date": kwargs["trade_date"],
                "total_share": 100.0,
                "nav": 1.25,
            }])

    args = types.SimpleNamespace(
        end_date="20260831",
        max_latest_days=120,
        max_retries=1,
        backoff_sec=0.0,
        wait_on_rate_limit_sec=0.0,
        retry_jitter_sec=0.0,
        empty_response_retries=0,
        incremental_lookback_days=5,
        incremental_batch_days=20,
        max_workers=2,
        limit=None,
    )
    module.save_latest_etf_share_size(Pro(), tmp_path, module.RateLimiter(10_000), args)

    assert sorted(set(calls)) == dates[-5:]
    saved = pd.read_parquet(tmp_path / "etf_share_size_df.parquet")
    assert saved["date"].astype(str).tolist() == [
        pd.to_datetime(value, format="%Y%m%d").date().isoformat() for value in dates[-5:]
    ]
    output = pd.read_parquet(tmp_path / "etf_share_size_df.parquet")
    assert output["trade_date"].tolist() == dates[-5:]
    assert output["total_share"].tolist() == [100.0] * 5
    assert output["nav"].tolist() == [1.25] * 5


def test_call_tushare_api_waits_and_retries_rate_limit(monkeypatch) -> None:
    module = _load_data_script()
    attempts = 0
    sleeps = []

    def fake_api():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("访问频次超限")
        return pd.DataFrame([{"ts_code": "510050.SH"}])

    monkeypatch.setattr(module.random, "uniform", lambda *_args: 0.5)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    out = module.call_tushare_api(
        fake_api,
        module.RateLimiter(1000),
        max_retries=3,
        backoff_sec=1.0,
        wait_on_rate_limit_sec=3.0,
        retry_jitter_sec=1.0,
        context="fund_nav test",
        api_name="fund_nav",
    )

    assert len(out) == 1
    assert attempts == 2
    assert sleeps == [3.5]


def test_rate_limiter_enforces_minimum_spacing(monkeypatch) -> None:
    module = _load_data_script()
    clock = [100.0]
    sleeps = []

    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def advance(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(module.time, "sleep", advance)
    limiter = module.RateLimiter(100, min_interval_sec=1.5)
    limiter.acquire()
    limiter.acquire()

    assert sleeps == [1.5]


def test_documented_row_limit_fails_instead_of_saving_truncated_data() -> None:
    module = _load_data_script()
    capped = pd.DataFrame({"ts_code": ["510050.SH"] * 5000})

    with pytest.raises(module.ResponseTruncatedError, match="可能被截断"):
        module.ensure_response_not_truncated(capped, "fund_daily", "fund_daily 20260828")


def test_fetch_fund_daily_splits_long_history_into_date_chunks() -> None:
    module = _load_data_script()
    calls = []

    class Pro:
        @staticmethod
        def fund_daily(**kwargs):
            calls.append((kwargs["start_date"], kwargs["end_date"]))
            return pd.DataFrame(
                [{"ts_code": kwargs["ts_code"], "trade_date": kwargs["start_date"], "close": 1.0}]
            )

    args = types.SimpleNamespace(
        start_date="20100101",
        end_date="20260828",
        history_chunk_days=3650,
        max_retries=1,
        backoff_sec=0.0,
        wait_on_rate_limit_sec=0.0,
        retry_jitter_sec=0.0,
    )
    out = module.fetch_fund_daily(Pro(), "510050.SH", "上证50ETF", module.RateLimiter(1000), args)

    assert out is not None
    assert len(calls) == 2
    assert calls[0] == ("20100101", "20191229")
    assert calls[1][0] == "20191230"
    assert calls[1][1] == "20260828"


def test_full_history_retries_once_when_every_chunk_is_transiently_empty(monkeypatch) -> None:
    module = _load_data_script()
    calls = []
    sleeps = []

    class Pro:
        @staticmethod
        def fund_nav(**kwargs):
            calls.append(kwargs["ts_code"])
            if len(calls) == 1:
                return pd.DataFrame()
            return pd.DataFrame(
                [{"ts_code": kwargs["ts_code"], "nav_date": "20260828", "adj_nav": 1.01}]
            )

    args = types.SimpleNamespace(
        start_date="20260801",
        end_date="20260828",
        history_chunk_days=3650,
        max_retries=1,
        empty_response_retries=1,
        backoff_sec=2.0,
        wait_on_rate_limit_sec=0.0,
        retry_jitter_sec=0.0,
    )
    monkeypatch.setattr(module.time, "sleep", sleeps.append)

    out = module.fetch_fund_nav(
        Pro(), "000001.OF", "测试基金", module.RateLimiter(1000), args, market="O"
    )

    assert out is not None
    assert calls == ["000001.OF", "000001.OF"]
    assert sleeps == [2.0]


def test_full_history_confirms_each_empty_chunk_even_when_neighbour_has_data(monkeypatch) -> None:
    module = _load_data_script()
    calls: list[tuple[str, str]] = []
    sleeps: list[float] = []

    class Pro:
        @staticmethod
        def fund_nav(**kwargs):
            key = (kwargs["start_date"], kwargs["end_date"])
            calls.append(key)
            if key[0] == "20100101" and calls.count(key) == 1:
                return pd.DataFrame()
            return pd.DataFrame(
                [{"ts_code": kwargs["ts_code"], "nav_date": key[0], "adj_nav": 1.01}]
            )

    args = types.SimpleNamespace(
        start_date="20100101",
        end_date="20260828",
        history_chunk_days=3650,
        max_retries=1,
        empty_response_retries=1,
        backoff_sec=2.0,
        wait_on_rate_limit_sec=0.0,
        retry_jitter_sec=0.0,
    )
    monkeypatch.setattr(module.time, "sleep", sleeps.append)

    out = module.fetch_fund_nav(
        Pro(), "000001.OF", "测试基金", module.RateLimiter(1000), args, market="O"
    )

    assert out is not None
    assert calls == [
        ("20100101", "20191229"),
        ("20100101", "20191229"),
        ("20191230", "20260828"),
    ]
    assert sleeps == [2.0]


def test_history_requests_are_clipped_to_instrument_lifecycle() -> None:
    module = _load_data_script()
    calls: list[tuple[str, str]] = []

    class Pro:
        @staticmethod
        def fund_nav(**kwargs):
            calls.append((kwargs["start_date"], kwargs["end_date"]))
            return pd.DataFrame(
                [{"ts_code": kwargs["ts_code"], "nav_date": kwargs["start_date"], "adj_nav": 1.0}]
            )

    args = types.SimpleNamespace(
        start_date="20100101",
        end_date="20260828",
        history_chunk_days=3650,
        max_retries=1,
        empty_response_retries=1,
        backoff_sec=0.0,
        wait_on_rate_limit_sec=0.0,
        retry_jitter_sec=0.0,
    )

    out = module.fetch_fund_nav(
        Pro(),
        "000001.OF",
        "测试基金",
        module.RateLimiter(1000),
        args,
        market="O",
        history_start_date="20200115",
        history_end_date="20231231",
    )

    assert out is not None
    assert calls == [("20200115", "20231231")]


def test_product_universe_fetches_are_partitioned() -> None:
    module = _load_data_script()
    fund_statuses = []
    etf_partitions = []

    class Pro:
        @staticmethod
        def fund_basic(**kwargs):
            fund_statuses.append(kwargs["status"])
            if kwargs["offset"] > 0:
                return pd.DataFrame()
            return pd.DataFrame([{"ts_code": f"{kwargs['status']}00001.SH"}])

        @staticmethod
        def etf_basic(**kwargs):
            etf_partitions.append((kwargs["exchange"], kwargs["list_status"]))
            return pd.DataFrame([{"ts_code": f"{kwargs['list_status']}00001.{kwargs['exchange']}"}])

    args = types.SimpleNamespace(
        max_retries=1,
        backoff_sec=0.0,
        wait_on_rate_limit_sec=0.0,
        retry_jitter_sec=0.0,
    )
    limiter = module.RateLimiter(1000)
    module.fetch_fund_basic(Pro(), limiter, args, market="O")
    module.fetch_etf_basic(Pro(), limiter, args)

    assert fund_statuses == ["L", "L", "L", "I", "I", "I", "D", "D", "D"]
    assert etf_partitions == [
        ("SH", "L"), ("SH", "P"), ("SH", "D"),
        ("SZ", "L"), ("SZ", "P"), ("SZ", "D"),
    ]


def test_fund_basic_paginates_when_status_reaches_single_call_cap() -> None:
    module = _load_data_script()
    calls = []

    class Pro:
        @staticmethod
        def fund_basic(**kwargs):
            calls.append((kwargs["status"], kwargs["offset"], kwargs["limit"]))
            if kwargs["status"] == "L" and kwargs["offset"] == 0:
                return pd.DataFrame({"ts_code": [f"{index:06d}.OF" for index in range(15000)]})
            if kwargs["status"] == "L":
                if kwargs["offset"] == 15000:
                    return pd.DataFrame({"ts_code": ["015000.OF", "015001.OF"]})
                return pd.DataFrame()
            if kwargs["offset"] > 0:
                return pd.DataFrame()
            return pd.DataFrame({"ts_code": [f"{kwargs['status']}00001.OF"]})

    args = types.SimpleNamespace(
        max_retries=1,
        backoff_sec=0.0,
        wait_on_rate_limit_sec=0.0,
        retry_jitter_sec=0.0,
        max_fund_basic_pages=5,
    )
    out = module.fetch_fund_basic(Pro(), module.RateLimiter(1000), args, market="O")

    assert ("L", 0, 15000) in calls
    assert ("L", 15000, 15000) in calls
    assert len(out) == 15004


def test_latest_fund_nav_paginates_until_empty_page() -> None:
    module = _load_data_script()
    calls = []

    def fake_api(**kwargs):
        calls.append(kwargs["offset"])
        pages = {
            0: ["000001.OF", "000002.OF"],
            2: ["000003.OF"],
            3: [],
        }
        return pd.DataFrame(
            [{"ts_code": code, "nav_date": kwargs["nav_date"], "adj_nav": 1.0} for code in pages[kwargs["offset"]]]
        )

    universe = pd.DataFrame(
        [
            {"ts_code": "000001.OF", "name": "基金一"},
            {"ts_code": "000002.OF", "name": "基金二"},
            {"ts_code": "000003.OF", "name": "基金三"},
        ]
    )
    args = types.SimpleNamespace(
        max_retries=1,
        backoff_sec=0.0,
        wait_on_rate_limit_sec=0.0,
        retry_jitter_sec=0.0,
        max_fund_nav_pages=5,
    )

    frames = module.fetch_latest_dates(
        api_func=fake_api,
        api_name="fund_nav",
        date_param="nav_date",
        dates=["20260828"],
        fields=module.FUND_NAV_FIELDS,
        universe=universe,
        limiter=module.RateLimiter(1000),
        args=args,
        market="O",
        universe_label="场外公募基金",
        page_size=2,
    )

    assert calls == [0, 2, 3, 3]
    assert len(frames) == 1
    assert set(frames[0]["ts_code"]) == {"000001.OF", "000002.OF", "000003.OF"}


def test_full_history_checkpoints_resume_only_failed_codes(tmp_path: Path) -> None:
    module = _load_data_script()
    universe = pd.DataFrame(
        [
            {"ts_code": "000001.OF", "name": "基金一"},
            {"ts_code": "000002.OF", "name": "基金二"},
            {"ts_code": "000003.OF", "name": "基金三"},
        ]
    )
    out_path = tmp_path / "fund_nav_df.parquet"
    args = types.SimpleNamespace(
        start_date="20260101",
        end_date="20260831",
        max_workers=2,
        missing_only=False,
    )
    calls = []
    fail_code_three = True

    def fetcher(code, name):
        nonlocal fail_code_three
        calls.append(code)
        if code == "000002.OF":
            return None
        if code == "000003.OF" and fail_code_three:
            raise RuntimeError("simulated transient failure")
        return pd.DataFrame(
            [
                {
                    "ts_code": code,
                    "name": name,
                    "date": pd.Timestamp("2026-08-28"),
                    "adj_nav": 1.0,
                }
            ]
        )

    with pytest.raises(RuntimeError, match="已保留检查点"):
        module.save_full_history_with_checkpoints(
            universe=universe,
            out_path=out_path,
            label="test fund_nav",
            fetcher=fetcher,
            duplicate_subset=["ts_code", "date"],
            sort_cols=["ts_code", "date"],
            args=args,
        )

    checkpoint_dir = module.history_checkpoint_dir(out_path, args)
    assert (checkpoint_dir / "000001.OF.parquet").exists()
    assert (checkpoint_dir / "000002.OF.empty").exists()
    assert not out_path.exists()

    calls.clear()
    fail_code_three = False
    module.save_full_history_with_checkpoints(
        universe=universe,
        out_path=out_path,
        label="test fund_nav",
        fetcher=fetcher,
        duplicate_subset=["ts_code", "date"],
        sort_cols=["ts_code", "date"],
        args=args,
    )

    assert calls == ["000003.OF"]
    out = pd.read_parquet(out_path)
    assert out["ts_code"].tolist() == ["000001.OF", "000003.OF"]
    assert out.duplicated(["ts_code", "date"]).sum() == 0


def test_full_history_checkpoints_use_bounded_thread_pool(tmp_path: Path) -> None:
    module = _load_data_script()
    universe = pd.DataFrame(
        [{"ts_code": f"00000{index}.OF", "name": f"基金{index}"} for index in range(1, 5)]
    )
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def fetcher(code, name):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with state_lock:
            active -= 1
        return pd.DataFrame(
            [{"ts_code": code, "name": name, "date": pd.Timestamp("2026-08-28"), "adj_nav": 1.0}]
        )

    out_path = tmp_path / "fund_nav_df.parquet"
    args = types.SimpleNamespace(
        start_date="20260101",
        end_date="20260831",
        max_workers=2,
        missing_only=False,
    )

    module.save_full_history_with_checkpoints(
        universe=universe,
        out_path=out_path,
        label="test fund_nav",
        fetcher=fetcher,
        duplicate_subset=["ts_code", "date"],
        sort_cols=["ts_code", "date"],
        args=args,
    )

    assert max_active == 2
    assert pd.read_parquet(out_path)["ts_code"].tolist() == universe["ts_code"].tolist()


def test_action_resume_marker_skips_only_completed_candidate_action(
    tmp_path: Path,
    capsys,
) -> None:
    module = _load_data_script()
    args = types.SimpleNamespace(
        output_dir=tmp_path,
        start_date="20100101",
        end_date="20260831",
        history_chunk_days=3650,
        missing_only=False,
        latest=False,
        resume=False,
    )
    calls: list[str] = []

    module._run_action_once(
        args,
        "fund_info",
        lambda: calls.append("first"),
        step_index=1,
        step_total=2,
    )
    args.resume = True
    module._run_action_once(
        args,
        "fund_info",
        lambda: calls.append("duplicate"),
        step_index=1,
        step_total=2,
    )

    assert calls == ["first"]
    output = capsys.readouterr().out
    assert "[STAGE] 正在处理场外公募基金基础信息（节点 1/2）" in output
    assert "[DONE] 场外公募基金基础信息（节点 1/2）已完成" in output
    assert "续跑直接复用" in output
    marker = module.read_json_object(tmp_path / ".tushare_action_fund_info.json")
    assert marker is not None and marker["action"] == "fund_info"


def test_latest_parquet_date_uses_statistics_without_pandas_read(monkeypatch, tmp_path: Path) -> None:
    module = _load_data_script()
    path = tmp_path / "dates.parquet"
    pd.DataFrame(
        {
            "ts_code": ["000001.OF", "000001.OF", "000002.OF"],
            "date": pd.to_datetime(["2026-08-27", "2026-08-28", "2026-08-29"]),
        }
    ).to_parquet(path, index=False, row_group_size=1)

    monkeypatch.setattr(module.pd, "read_parquet", lambda *_args, **_kwargs: pytest.fail("pandas read not allowed"))

    assert module.latest_parquet_date(path, "date") == pd.Timestamp("2026-08-29")


def test_latest_parquet_date_scans_only_date_batches_when_statistics_missing(monkeypatch, tmp_path: Path) -> None:
    module = _load_data_script()
    path = tmp_path / "dates-no-statistics.parquet"
    table = module.pa.Table.from_pandas(
        pd.DataFrame(
            {
                "ts_code": ["000001.OF", "000002.OF"],
                "date": ["20260827", "20260830"],
                "payload": ["x" * 1000, "y" * 1000],
            }
        ),
        preserve_index=False,
    )
    module.parquet.write_table(table, path, row_group_size=1, write_statistics=False)
    monkeypatch.setattr(module.pd, "read_parquet", lambda *_args, **_kwargs: pytest.fail("pandas read not allowed"))

    assert module.latest_parquet_date(path, "date") == pd.Timestamp("2026-08-30")


def test_incremental_rows_stream_merge_preserves_code_contiguity(monkeypatch, tmp_path: Path) -> None:
    module = _load_data_script()
    path = tmp_path / "fund_nav_df.parquet"
    existing = pd.DataFrame(
        [
            {"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-27"), "adj_nav": 1.0},
            {"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-28"), "adj_nav": 1.1},
            {"ts_code": "000003.OF", "date": pd.Timestamp("2026-08-27"), "adj_nav": 3.0},
            {"ts_code": "000003.OF", "date": pd.Timestamp("2026-08-28"), "adj_nav": 3.1},
        ]
    )
    existing.to_parquet(path, index=False, row_group_size=2)
    incoming = pd.DataFrame(
        [
            {"ts_code": "000003.OF", "date": pd.Timestamp("2026-08-29"), "adj_nav": 3.2},
            {"ts_code": "000002.OF", "date": pd.Timestamp("2026-08-29"), "adj_nav": 2.0},
            {"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-29"), "adj_nav": 1.2},
        ]
    )

    with monkeypatch.context() as context:
        context.setattr(module.pd, "read_parquet", lambda *_args, **_kwargs: pytest.fail("pandas read not allowed"))
        merged_rows = module.append_incremental_rows(
            incoming,
            path,
            subset=["ts_code", "date"],
            sort_cols=["ts_code", "date"],
            date_column="date",
        )

    out = pd.read_parquet(path)
    assert merged_rows == 7
    assert out["ts_code"].tolist() == [
        "000001.OF",
        "000001.OF",
        "000001.OF",
        "000002.OF",
        "000003.OF",
        "000003.OF",
        "000003.OF",
    ]
    assert out.equals(out.sort_values(["ts_code", "date"]).reset_index(drop=True))
    assert out.duplicated(["ts_code", "date"]).sum() == 0


def test_incremental_rows_copy_untouched_instruments_without_pandas_merge(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_data_script()
    path = tmp_path / "fund_nav_df.parquet"
    pd.DataFrame(
        [
            {"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-27"), "adj_nav": 1.0},
            {"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-28"), "adj_nav": 1.1},
            {"ts_code": "000002.OF", "date": pd.Timestamp("2026-08-28"), "adj_nav": 2.0},
        ]
    ).to_parquet(path, index=False, row_group_size=2)

    with monkeypatch.context() as context:
        context.setattr(
            module.pd,
            "concat",
            lambda *_args, **_kwargs: pytest.fail("untouched instruments must stay in Arrow"),
        )
        merged_rows = module.append_incremental_rows(
            pd.DataFrame(
                [{"ts_code": "000003.OF", "date": pd.Timestamp("2026-08-29"), "adj_nav": 3.0}]
            ),
            path,
            subset=["ts_code", "date"],
            sort_cols=["ts_code", "date"],
            date_column="date",
        )

    out = pd.read_parquet(path)
    assert merged_rows == 4
    assert out["ts_code"].tolist() == ["000001.OF", "000001.OF", "000002.OF", "000003.OF"]


def test_incremental_overlap_revision_overwrites_matching_key(tmp_path: Path) -> None:
    module = _load_data_script()
    path = tmp_path / "fund_nav_df.parquet"
    pd.DataFrame(
        [
            {"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-27"), "adj_nav": 1.0},
            {"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-28"), "adj_nav": 1.1},
            {"ts_code": "000002.OF", "date": pd.Timestamp("2026-08-28"), "adj_nav": 2.0},
        ]
    ).to_parquet(path, index=False)

    merged_rows = module.append_incremental_rows(
        pd.DataFrame(
            [{"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-28"), "adj_nav": 9.9}]
        ),
        path,
        subset=["ts_code", "date"],
        sort_cols=["ts_code", "date"],
        date_column="date",
    )

    out = pd.read_parquet(path)
    assert merged_rows == 3
    assert out.loc[
        (out["ts_code"] == "000001.OF") & (out["date"] == pd.Timestamp("2026-08-28")),
        "adj_nav",
    ].item() == 9.9
    assert out.loc[out["ts_code"] == "000002.OF", "adj_nav"].item() == 2.0
    assert out.duplicated(["ts_code", "date"]).sum() == 0


def test_incremental_identical_overlap_preserves_existing_parquet(monkeypatch, tmp_path: Path) -> None:
    module = _load_data_script()
    path = tmp_path / "fund_nav_df.parquet"
    existing = pd.DataFrame(
        [
            {"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-27"), "adj_nav": 1.0},
            {"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-28"), "adj_nav": 1.1},
            {"ts_code": "000002.OF", "date": pd.Timestamp("2026-08-28"), "adj_nav": 2.0},
        ]
    )
    existing.to_parquet(path, index=False, row_group_size=2)
    original = path.read_bytes()

    with monkeypatch.context() as context:
        context.setattr(
            module.os,
            "replace",
            lambda *_args, **_kwargs: pytest.fail("identical overlap must not replace baseline"),
        )
        context.setattr(
            module.parquet,
            "ParquetWriter",
            lambda *_args, **_kwargs: pytest.fail("identical overlap must not rewrite baseline"),
        )
        merged_rows = module.append_incremental_rows(
            existing.iloc[[1]].copy(),
            path,
            subset=["ts_code", "date"],
            sort_cols=["ts_code", "date"],
            date_column="date",
        )

    assert merged_rows == 3
    assert path.read_bytes() == original
    assert list(tmp_path.glob(".fund_nav_df.parquet.*.tmp")) == []


def test_incremental_calendar_merge_normalises_string_dates_before_arrow_write(tmp_path: Path) -> None:
    module = _load_data_script()
    path = tmp_path / "trade_day_df.parquet"
    pd.DataFrame(
        [
            {"exchange": "SSE", "cal_date": "20260830", "is_open": 0},
            {"exchange": "SSE", "cal_date": "20260831", "is_open": 0},
        ]
    ).to_parquet(path, index=False)

    merged_rows = module.append_incremental_rows(
        pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20260831", "is_open": 1},
                {"exchange": "SSE", "cal_date": "20260901", "is_open": 1},
            ]
        ),
        path,
        subset=["exchange", "cal_date"],
        sort_cols=["exchange", "cal_date"],
        date_column="cal_date",
    )

    out = pd.read_parquet(path)
    assert merged_rows == 3
    assert out.to_dict("records") == [
        {"exchange": "SSE", "cal_date": "20260830", "is_open": 0},
        {"exchange": "SSE", "cal_date": "20260831", "is_open": 1},
        {"exchange": "SSE", "cal_date": "20260901", "is_open": 1},
    ]
    assert module.parquet.ParquetFile(path).schema_arrow.field("cal_date").type == module.pa.string()


def test_incremental_late_report_inserts_missing_historical_key(tmp_path: Path) -> None:
    module = _load_data_script()
    path = tmp_path / "fund_nav_df.parquet"
    pd.DataFrame(
        [
            {"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-27"), "adj_nav": 1.0},
            {"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-29"), "adj_nav": 1.2},
            {"ts_code": "000002.OF", "date": pd.Timestamp("2026-08-28"), "adj_nav": 2.0},
        ]
    ).to_parquet(path, index=False, row_group_size=1)

    module.append_incremental_rows(
        pd.DataFrame(
            [{"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-28"), "adj_nav": 1.1}]
        ),
        path,
        subset=["ts_code", "date"],
        sort_cols=["ts_code", "date"],
        date_column="date",
    )

    out = pd.read_parquet(path)
    assert out[["ts_code", "date"]].to_dict("records") == [
        {"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-27")},
        {"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-28")},
        {"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-29")},
        {"ts_code": "000002.OF", "date": pd.Timestamp("2026-08-28")},
    ]


def test_incremental_atomic_replace_failure_preserves_baseline(monkeypatch, tmp_path: Path) -> None:
    module = _load_data_script()
    path = tmp_path / "fund_nav_df.parquet"
    pd.DataFrame(
        [{"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-28"), "adj_nav": 1.0}]
    ).to_parquet(path, index=False)
    original = path.read_bytes()
    monkeypatch.setattr(module.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("replace failed")))

    with pytest.raises(OSError, match="replace failed"):
        module.append_incremental_rows(
            pd.DataFrame(
                [{"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-28"), "adj_nav": 1.1}]
            ),
            path,
            subset=["ts_code", "date"],
            sort_cols=["ts_code", "date"],
            date_column="date",
        )

    assert path.read_bytes() == original
    assert list(tmp_path.glob(".fund_nav_df.parquet.*.tmp")) == []


def test_incremental_unsorted_code_baseline_fails_atomically(tmp_path: Path) -> None:
    module = _load_data_script()
    path = tmp_path / "fund_nav_df.parquet"
    pd.DataFrame(
        [
            {"ts_code": "000002.OF", "date": pd.Timestamp("2026-08-28"), "adj_nav": 2.0},
            {"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-28"), "adj_nav": 1.0},
        ]
    ).to_parquet(path, index=False, row_group_size=1)
    original = path.read_bytes()

    with pytest.raises(ValueError, match="连续排序"):
        module.append_incremental_rows(
            pd.DataFrame(
                [{"ts_code": "000003.OF", "date": pd.Timestamp("2026-08-29"), "adj_nav": 3.0}]
            ),
            path,
            subset=["ts_code", "date"],
            sort_cols=["ts_code", "date"],
            date_column="date",
        )

    assert path.read_bytes() == original
    assert list(tmp_path.glob(".fund_nav_df.parquet.*.tmp")) == []


def test_consolidation_streams_without_parquet_read_table(monkeypatch, tmp_path: Path) -> None:
    module = _load_data_script()
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    output = tmp_path / "combined.parquet"
    pd.DataFrame([{"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-28")}]).to_parquet(
        first, index=False
    )
    pd.DataFrame([{"ts_code": "000002.OF", "date": pd.Timestamp("2026-08-28")}]).to_parquet(
        second, index=False
    )

    with monkeypatch.context() as context:
        context.setattr(module.parquet, "read_table", lambda *_args, **_kwargs: pytest.fail("whole-file read"))
        assert module.consolidate_parquet_parts([first, second], output) == 2

    assert pd.read_parquet(output)["ts_code"].tolist() == ["000001.OF", "000002.OF"]


def test_missing_only_consolidation_merges_new_codes_into_sorted_baseline(tmp_path: Path) -> None:
    module = _load_data_script()
    baseline = tmp_path / "baseline.parquet"
    middle = tmp_path / "000002.OF.parquet"
    tail = tmp_path / "000004.OF.parquet"
    pd.DataFrame(
        [
            {"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-27"), "adj_nav": 1.0},
            {"ts_code": "000003.OF", "date": pd.Timestamp("2026-08-27"), "adj_nav": 3.0},
        ]
    ).to_parquet(baseline, index=False, row_group_size=1)
    pd.DataFrame(
        [{"ts_code": "000002.OF", "date": pd.Timestamp("2026-08-27"), "adj_nav": 2.0}]
    ).to_parquet(middle, index=False)
    pd.DataFrame(
        [{"ts_code": "000004.OF", "date": pd.Timestamp("2026-08-27"), "adj_nav": 4.0}]
    ).to_parquet(tail, index=False)

    assert module.consolidate_parquet_parts([tail, middle], baseline, base_path=baseline) == 4

    out = pd.read_parquet(baseline)
    assert out["ts_code"].tolist() == ["000001.OF", "000002.OF", "000003.OF", "000004.OF"]


def test_missing_only_consolidation_rejects_duplicate_code_without_replacing_baseline(tmp_path: Path) -> None:
    module = _load_data_script()
    baseline = tmp_path / "baseline.parquet"
    duplicate = tmp_path / "duplicate.parquet"
    pd.DataFrame(
        [{"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-27"), "adj_nav": 1.0}]
    ).to_parquet(baseline, index=False)
    pd.DataFrame(
        [{"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-28"), "adj_nav": 1.1}]
    ).to_parquet(duplicate, index=False)
    original = baseline.read_bytes()

    with pytest.raises(ValueError, match="重复包含"):
        module.consolidate_parquet_parts([duplicate], baseline, base_path=baseline)

    assert baseline.read_bytes() == original


def test_manual_cli_lock_persists_redacted_failure_and_releases(monkeypatch, tmp_path: Path) -> None:
    module = _load_data_script()
    lock_path = tmp_path / ".refresh.lock"
    state_path = tmp_path / ".refresh.json"
    args = types.SimpleNamespace(latest=True, output_dir=tmp_path)
    monkeypatch.setattr(module, "GLOBAL_REFRESH_LOCK_PATH", lock_path)
    monkeypatch.setattr(module, "GLOBAL_REFRESH_STATE_PATH", state_path)
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "selected_actions", lambda _args: ["fund_nav"])
    monkeypatch.setattr(module, "read_tushare_token", lambda: "secret-token")
    monkeypatch.delenv(module.PARENT_LOCK_ENV, raising=False)

    def fail_run(_args, _actions):
        raise RuntimeError("TUSHARE_TOKEN=secret-token request failed")

    monkeypatch.setattr(module, "_run_actions", fail_run)
    with pytest.raises(RuntimeError, match="secret-token"):
        module.main()

    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["job"]["status"] == "failed"
    assert "secret-token" not in persisted["job"]["message"]
    assert "[REDACTED]" in persisted["job"]["message"]
    probe = module.InterProcessFileLock(lock_path)
    assert probe.acquire(owner="probe") is True
    probe.release()


def test_manual_cli_rejects_duplicate_before_running(monkeypatch, tmp_path: Path) -> None:
    module = _load_data_script()
    lock_path = tmp_path / ".refresh.lock"
    args = types.SimpleNamespace(latest=False, output_dir=tmp_path)
    monkeypatch.setattr(module, "GLOBAL_REFRESH_LOCK_PATH", lock_path)
    monkeypatch.setattr(module, "GLOBAL_REFRESH_STATE_PATH", tmp_path / ".refresh.json")
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "selected_actions", lambda _args: ["fund_nav"])
    monkeypatch.delenv(module.PARENT_LOCK_ENV, raising=False)
    ran = []
    monkeypatch.setattr(module, "_run_actions", lambda *_args: ran.append(True))
    held = module.InterProcessFileLock(lock_path)
    assert held.acquire(owner="existing") is True
    try:
        with pytest.raises(RuntimeError, match="避免重复抓取"):
            module.main()
    finally:
        held.release()

    assert ran == []


def test_manual_cli_persists_heartbeat_and_local_snapshot(monkeypatch, tmp_path: Path) -> None:
    module = _load_data_script()
    lock_path = tmp_path / ".refresh.lock"
    state_path = tmp_path / ".refresh.json"
    args = types.SimpleNamespace(latest=False, output_dir=tmp_path)
    monkeypatch.setattr(module, "GLOBAL_REFRESH_LOCK_PATH", lock_path)
    monkeypatch.setattr(module, "GLOBAL_REFRESH_STATE_PATH", state_path)
    monkeypatch.setattr(module, "CLI_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "selected_actions", lambda _args: ["fund_nav"])
    monkeypatch.delenv(module.PARENT_LOCK_ENV, raising=False)
    monkeypatch.setattr(module, "_run_actions", lambda *_args: time.sleep(0.04))
    monkeypatch.setattr(
        module,
        "_rebuild_cli_analytics",
        lambda _path: {"rows": 12, "by_kind": {"etf": 2, "fund": 10}},
    )

    module.main()

    persisted = json.loads(state_path.read_text(encoding="utf-8"))["job"]
    assert persisted["status"] == "succeeded"
    assert persisted["heartbeat_at"] > persisted["started_at"]
    assert persisted["analytics_snapshot"]["status"] == "succeeded"
    assert persisted["analytics_snapshot"]["rows"] == 12


def test_manual_cli_snapshot_failure_does_not_mark_fetch_failed(monkeypatch, tmp_path: Path) -> None:
    module = _load_data_script()
    state_path = tmp_path / ".refresh.json"
    args = types.SimpleNamespace(latest=False, output_dir=tmp_path)
    monkeypatch.setattr(module, "GLOBAL_REFRESH_LOCK_PATH", tmp_path / ".refresh.lock")
    monkeypatch.setattr(module, "GLOBAL_REFRESH_STATE_PATH", state_path)
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "selected_actions", lambda _args: ["fund_nav"])
    monkeypatch.delenv(module.PARENT_LOCK_ENV, raising=False)
    monkeypatch.setattr(module, "_run_actions", lambda *_args: None)
    monkeypatch.setattr(
        module,
        "_rebuild_cli_analytics",
        lambda _path: (_ for _ in ()).throw(RuntimeError("snapshot failed")),
    )

    module.main()

    persisted = json.loads(state_path.read_text(encoding="utf-8"))["job"]
    assert persisted["status"] == "succeeded"
    assert persisted["analytics_snapshot"]["status"] == "failed"
    assert persisted["warnings"][0]["code"] == "ANALYTICS_REBUILD_FAILED"
    assert "无需重新拉取数据" in persisted["message"]


def test_index_capped_range_is_bisected_before_results_are_accepted() -> None:
    module = _load_data_script()
    calls: list[tuple[str, str]] = []

    class Pro:
        @staticmethod
        def index_daily(**kwargs):
            calls.append((kwargs["start_date"], kwargs["end_date"]))
            if kwargs["start_date"] != kwargs["end_date"]:
                return pd.DataFrame(
                    {"ts_code": ["000300.SH"] * module.API_ROW_LIMITS["index_daily"],
                     "trade_date": [kwargs["start_date"]] * module.API_ROW_LIMITS["index_daily"]}
                )
            return pd.DataFrame(
                [{"ts_code": "000300.SH", "trade_date": kwargs["start_date"], "close": 1.0}]
            )

    args = types.SimpleNamespace(
        max_retries=1, backoff_sec=0.0, wait_on_rate_limit_sec=0.0, retry_jitter_sec=0.0
    )
    result = module.fetch_index_date_window(
        pro=Pro(), api_name="index_daily", limiter=module.RateLimiter(10_000), args=args,
        code="000300.SH", start_date="20260830", end_date="20260831",
    )

    assert calls == [("20260830", "20260831"), ("20260830", "20260830"), ("20260831", "20260831")]
    assert len(result) == 2
    assert result["trade_date"].dtype.kind == "M"


def test_etf_share_size_capped_range_is_bisected_and_keeps_formula_inputs() -> None:
    module = _load_data_script()
    calls: list[tuple[str, str]] = []

    class Pro:
        @staticmethod
        def etf_share_size(**kwargs):
            calls.append((kwargs["start_date"], kwargs["end_date"]))
            if kwargs["start_date"] != kwargs["end_date"]:
                return pd.DataFrame(
                    {
                        "ts_code": [kwargs["ts_code"]] * module.API_ROW_LIMITS["etf_share_size"],
                        "trade_date": [kwargs["start_date"]] * module.API_ROW_LIMITS["etf_share_size"],
                    }
                )
            return pd.DataFrame(
                [{
                    "ts_code": kwargs["ts_code"],
                    "trade_date": kwargs["start_date"],
                    "total_share": 100.0,
                    "nav": 1.25,
                }]
            )

    args = types.SimpleNamespace(
        max_retries=1, backoff_sec=0.0, wait_on_rate_limit_sec=0.0, retry_jitter_sec=0.0
    )
    result = module.fetch_etf_share_size_window(
        pro=Pro(), limiter=module.RateLimiter(10_000), args=args,
        ts_code="510300.SH", name="沪深300ETF",
        start_date="20260830", end_date="20260831",
    )

    assert calls == [("20260830", "20260831"), ("20260830", "20260830"), ("20260831", "20260831")]
    assert len(result) == 2
    assert list(result["total_share"]) == [100.0, 100.0]
    assert list(result["nav"]) == [1.25, 1.25]
    assert result["date"].dtype.kind == "M"


def test_index_scope_selection_auto_includes_catalog_and_coverage() -> None:
    module = _load_data_script()
    args = types.SimpleNamespace(
        latest=False, all=False, etf_info=False, nav=False, etf_share=False, candle=False, calendar=False,
        stock_basic=False, index_info=False, etf_index=False, fund_info=False, fund_nav=False,
        fund_company=False, index_catalog=False, index_domestic=False, index_industry=False,
        index_concept=False, index_global=True, index_futures=False, index_valuation=False,
        index_constituents=False,
    )

    actions = module.selected_actions(args)

    assert actions == ["index_global", "index_catalog", "index_coverage"]
    assert module._modules_for_actions(actions) == ["index"]
    assert module._module_scopes_for_actions(actions) == {
        "index": ["catalog", "global"]
    }


def test_index_incremental_start_uses_exactly_five_sse_trade_days(tmp_path: Path) -> None:
    module = _load_data_script()
    history = tmp_path / "index_daily_df.parquet"
    pd.DataFrame(
        [{"trade_date": pd.Timestamp("2026-08-31"), "ts_code": "000300.SH"}]
    ).to_parquet(history, index=False)
    pd.DataFrame(
        {
            "exchange": ["SSE"] * 7,
            "cal_date": pd.to_datetime(
                [
                    "2026-08-21", "2026-08-24", "2026-08-25", "2026-08-26",
                    "2026-08-27", "2026-08-28", "2026-08-31",
                ]
            ),
            "is_open": [1] * 7,
        }
    ).to_parquet(tmp_path / "trade_day_df.parquet", index=False)
    args = types.SimpleNamespace(start_date="20200101", incremental_lookback_days=5)

    assert module._index_incremental_start(tmp_path, history, args) == "20260825"


def test_latest_index_weight_reads_only_recent_monthly_window(tmp_path: Path) -> None:
    module = _load_data_script()
    pd.DataFrame(
        [
            {
                "source_api": "index_basic",
                "quote_source_api": "index_daily",
                "ts_code": "000300.SH",
                "name": "沪深300",
                "status": "active",
            }
        ]
    ).to_parquet(tmp_path / "index_catalog_df.parquet", index=False)
    weight_calls: list[tuple[str, str, str]] = []

    class Pro:
        @staticmethod
        def index_member_all(**_kwargs):
            return pd.DataFrame(
                [{"index_code": "000300.SH", "con_code": "600000.SH", "is_new": "Y"}]
            )

        @staticmethod
        def ci_index_member(**_kwargs):
            return pd.DataFrame()

        @staticmethod
        def index_weight(**kwargs):
            weight_calls.append(
                (kwargs["index_code"], kwargs["start_date"], kwargs["end_date"])
            )
            return pd.DataFrame(
                [
                    {
                        "index_code": "000300.SH",
                        "con_code": "600000.SH",
                        "trade_date": "20260801",
                        "weight": 1.5,
                    }
                ]
            )

    args = types.SimpleNamespace(
        end_date="20260831",
        start_date="20200101",
        limit=None,
        max_retries=1,
        backoff_sec=0.0,
        wait_on_rate_limit_sec=0.0,
        retry_jitter_sec=0.0,
    )
    module.save_index_constituents(
        Pro(), tmp_path, module.RateLimiter(10_000), args
    )

    weights = pd.read_parquet(tmp_path / "index_weights_df.parquet")
    assert len(weight_calls) == 1
    assert weight_calls[0][0] == "000300.SH"
    assert weight_calls[0][1] == "20260504"
    assert weight_calls[0][2] == "20260831"
    assert weights.iloc[0]["trade_date"] == pd.Timestamp("2026-08-01")


def test_index_weight_resume_reuses_member_and_per_code_checkpoints(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_data_script()
    pd.DataFrame(
        [
            {
                "source_api": "index_basic", "quote_source_api": "index_daily",
                "ts_code": code, "name": code, "status": "active",
            }
            for code in ("000300.SH", "000905.SH")
        ]
    ).to_parquet(tmp_path / "index_catalog_df.parquet", index=False)
    calls: list[tuple[str, str | None]] = []
    fail_second = True

    def fake_call(_pro, api_name, _limiter, _args, **kwargs):
        nonlocal fail_second
        code = kwargs.get("index_code") or kwargs.get("ts_code")
        calls.append((api_name, code))
        if api_name == "index_member_all":
            return pd.DataFrame(
                [{"index_code": "000300.SH", "con_code": "600000.SH", "is_new": "Y"}]
            )
        if api_name == "ci_index_member":
            return pd.DataFrame()
        if api_name == "index_weight":
            if code == "000905.SH" and fail_second:
                raise RuntimeError("temporary failure")
            return pd.DataFrame(
                [{
                    "index_code": code, "con_code": "600000.SH",
                    "trade_date": "20260831", "weight": 1.0,
                }]
            )
        return pd.DataFrame()

    monkeypatch.setattr(module, "_call_index_api", fake_call)
    args = types.SimpleNamespace(
        end_date="20260831", start_date="20200101", limit=None,
        max_workers=2, resume=True,
    )

    with pytest.raises(RuntimeError, match="成功检查点"):
        module.save_index_constituents(object(), tmp_path, module.RateLimiter(10_000), args)

    checkpoint_dir = module.history_checkpoint_dir(tmp_path / "index_weights_df.parquet", args)
    assert (checkpoint_dir / "000300.SH.parquet").exists()
    assert not (checkpoint_dir / "000905.SH.parquet").exists()
    assert (tmp_path / ".tushare_stage_index_constituents_members.json").exists()

    calls.clear()
    fail_second = False
    module.save_index_constituents(object(), tmp_path, module.RateLimiter(10_000), args)

    assert calls == [("index_weight", "000905.SH")]
    weights = pd.read_parquet(tmp_path / "index_weights_df.parquet")
    assert set(weights["index_code"]) == {"000300.SH", "000905.SH"}


def test_index_constituents_and_weights_fetch_code_tasks_concurrently(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_data_script()
    pd.DataFrame(
        [
            {
                "source_api": "ths_index",
                "quote_source_api": "ths_daily",
                "ts_code": "885001.TI",
                "name": "概念一",
                "status": "active",
            },
            {
                "source_api": "ths_index",
                "quote_source_api": "ths_daily",
                "ts_code": "885002.TI",
                "name": "概念二",
                "status": "active",
            },
            {
                "source_api": "index_basic",
                "quote_source_api": "index_daily",
                "ts_code": "000300.SH",
                "name": "沪深300",
                "status": "active",
            },
            {
                "source_api": "index_basic",
                "quote_source_api": "index_daily",
                "ts_code": "000905.SH",
                "name": "中证500",
                "status": "active",
            },
        ]
    ).to_parquet(tmp_path / "index_catalog_df.parquet", index=False)
    concurrency_lock = threading.Lock()
    active_calls = 0
    max_active_calls = 0

    def fake_call(_pro, api_name, _limiter, _args, **kwargs):
        nonlocal active_calls, max_active_calls
        with concurrency_lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
        try:
            time.sleep(0.02)
            if api_name == "index_member_all":
                return pd.DataFrame(
                    [{"index_code": "000300.SH", "con_code": "600000.SH", "is_new": "Y"}]
                )
            if api_name == "ci_index_member":
                return pd.DataFrame()
            if api_name == "index_weight":
                code = kwargs["index_code"]
                return pd.DataFrame(
                    [{
                        "index_code": code,
                        "con_code": "600000.SH",
                        "trade_date": "20260831",
                        "weight": 1.0,
                    }]
                )
            code = kwargs["ts_code"]
            return pd.DataFrame([{"con_code": "600000.SH", "con_name": code}])
        finally:
            with concurrency_lock:
                active_calls -= 1

    monkeypatch.setattr(module, "_call_index_api", fake_call)
    args = types.SimpleNamespace(
        end_date="20260831",
        start_date="20200101",
        limit=None,
        max_workers=4,
    )

    module.save_index_constituents(
        object(), tmp_path, module.RateLimiter(10_000), args
    )

    members = pd.read_parquet(tmp_path / "index_members_df.parquet")
    weights = pd.read_parquet(tmp_path / "index_weights_df.parquet")
    assert max_active_calls >= 2
    assert {"885001.TI", "885002.TI"}.issubset(set(members["index_code"]))
    assert set(weights["index_code"]) == {"000300.SH", "000905.SH"}


def test_futures_index_universe_uses_documented_codes_without_discovery(tmp_path: Path) -> None:
    module = _load_data_script()
    args = types.SimpleNamespace(limit=None)

    universe = module._index_universe(tmp_path, "fut_index_daily", args)

    assert len(universe) == len(module.FUTURES_INDEX_UNIVERSE)
    assert {"NHCI.NH", "SC.NH", "AU.NH"}.issubset(set(universe["ts_code"]))


def test_concept_catalog_preserves_real_idx_type_field() -> None:
    module = _load_data_script()
    result = module._normalise_catalog_frame(
        pd.DataFrame(
            [{"ts_code": "BK1753.DC", "name": "光刻胶", "idx_type": "概念板块"}]
        ),
        "dc_index",
    )

    assert result.iloc[0]["category"] == "概念板块"
    assert result.iloc[0]["quote_source_api"] == "dc_daily"


@pytest.mark.parametrize("api_name", ["dc_index", "tdx_index"])
def test_daily_index_catalog_uses_latest_available_date_instead_of_capped_history(
    monkeypatch, api_name: str
) -> None:
    module = _load_data_script()
    calls: list[dict[str, str]] = []

    def fake_call(_pro, observed_api, _limiter, _args, **kwargs):
        assert observed_api == api_name
        calls.append(kwargs)
        if kwargs["trade_date"] == "20260902":
            return pd.DataFrame()
        return pd.DataFrame(
            [{"ts_code": f"sample.{api_name}", "trade_date": kwargs["trade_date"]}]
        )

    monkeypatch.setattr(module, "_call_index_api", fake_call)
    result = module.fetch_index_catalog_request(
        object(),
        api_name,
        {"idx_type": "概念板块"},
        module.RateLimiter(10_000),
        types.SimpleNamespace(end_date="20260902"),
    )

    assert [item["trade_date"] for item in calls] == ["20260902", "20260901"]
    assert all(item["idx_type"] == "概念板块" for item in calls)
    assert result.iloc[0]["trade_date"] == "20260901"


def test_index_full_history_resumes_only_missing_date_segments(tmp_path: Path) -> None:
    module = _load_data_script()
    universe = pd.DataFrame([{"ts_code": "000300.SH", "name": "沪深300"}])
    out_path = tmp_path / "index_daily_df.parquet"
    args = types.SimpleNamespace(
        start_date="20260830",
        end_date="20260831",
        history_chunk_days=1,
        max_workers=1,
        max_retries=1,
        backoff_sec=0.0,
        wait_on_rate_limit_sec=0.0,
        retry_jitter_sec=0.0,
    )
    first_calls: list[str] = []

    class FirstPro:
        @staticmethod
        def index_daily(**kwargs):
            first_calls.append(kwargs["start_date"])
            if kwargs["start_date"] == "20260831":
                raise RuntimeError("temporary")
            return pd.DataFrame(
                [{"ts_code": "000300.SH", "trade_date": kwargs["start_date"], "close": 1.0}]
            )

    with pytest.raises(RuntimeError, match="日期段失败"):
        module.save_index_full_history_with_segment_checkpoints(
            universe=universe,
            out_path=out_path,
            api_name="index_daily",
            start_date=args.start_date,
            pro=FirstPro(),
            limiter=module.RateLimiter(10_000),
            args=args,
        )
    assert first_calls == ["20260830", "20260831"]

    resumed_calls: list[str] = []

    class ResumedPro:
        @staticmethod
        def index_daily(**kwargs):
            resumed_calls.append(kwargs["start_date"])
            return pd.DataFrame(
                [{"ts_code": "000300.SH", "trade_date": kwargs["start_date"], "close": 2.0}]
            )

    module.save_index_full_history_with_segment_checkpoints(
        universe=universe,
        out_path=out_path,
        api_name="index_daily",
        start_date=args.start_date,
        pro=ResumedPro(),
        limiter=module.RateLimiter(10_000),
        args=args,
    )

    assert resumed_calls == ["20260831"]
    result = pd.read_parquet(out_path)
    assert result["trade_date"].tolist() == [
        pd.Timestamp("2026-08-30"),
        pd.Timestamp("2026-08-31"),
    ]


def test_index_history_writes_typed_empty_file_when_catalog_has_no_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_data_script()
    monkeypatch.setattr(
        module,
        "_index_universe",
        lambda *_args, **_kwargs: pd.DataFrame(columns=["ts_code", "name"]),
    )

    module.save_index_history_api(
        object(),
        tmp_path,
        module.RateLimiter(10_000),
        types.SimpleNamespace(),
        "ci_daily",
    )

    result = pd.read_parquet(tmp_path / "index_ci_daily_df.parquet")
    assert result.empty
    assert result.columns.tolist() == ["source_api", "ts_code", "trade_date"]
