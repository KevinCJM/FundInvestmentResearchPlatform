from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
for path in (ROOT, BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.services import index_data, index_routes


def _write_catalog(root: Path) -> None:
    pd.DataFrame(
        [
            {
                "source_api": "index_basic",
                "quote_source_api": "index_daily",
                "ts_code": "000300.SH",
                "name": "沪深300",
                "category": "规模指数",
                "market": "CSI",
                "publisher": "中证指数公司",
                "list_date": pd.Timestamp("2005-04-08"),
                "exp_date": pd.NaT,
                "status": "active",
            },
            {
                "source_api": "ths_index",
                "quote_source_api": "ths_daily",
                "ts_code": "885001.TI",
                "name": "示例概念",
                "category": "概念",
                "market": "A",
                "publisher": None,
                "list_date": pd.NaT,
                "exp_date": pd.NaT,
                "status": "active",
            },
        ]
    ).to_parquet(root / "index_catalog_df.parquet", index=False)


def test_coverage_snapshot_streams_history_and_query_uses_only_small_tables(
    monkeypatch, tmp_path: Path
) -> None:
    _write_catalog(tmp_path)
    pd.DataFrame(
        {
            "exchange": ["SSE", "SSE", "SSE"],
            "cal_date": pd.to_datetime(["2026-08-27", "2026-08-28", "2026-08-31"]),
            "is_open": [1, 1, 1],
        }
    ).to_parquet(tmp_path / "trade_day_df.parquet", index=False)
    pd.DataFrame(
        {
            "source_api": ["index_daily", "index_daily", "index_daily"],
            "ts_code": ["000300.SH"] * 3,
            "trade_date": pd.to_datetime(["2026-08-27", "2026-08-28", "2026-08-31"]),
            "close": [4000.0, 4010.0, 4020.0],
        }
    ).to_parquet(tmp_path / "index_daily_df.parquet", index=False)

    result = index_data.build_index_coverage_snapshot(tmp_path)
    assert result["rows"] == 1
    coverage = pd.read_parquet(tmp_path / "index_coverage_snapshot.parquet")
    assert coverage.iloc[0]["rows"] == 3
    assert coverage.iloc[0]["domestic_trade_day_coverage"] == 1.0

    original = index_data.pd.read_parquet
    index_data._read_small_tables_cached.cache_clear()

    def guarded_read(path, *args, **kwargs):
        if Path(path).name == "index_daily_df.parquet":
            raise AssertionError("请求路径不应读取完整行情")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(index_data.pd, "read_parquet", guarded_read)
    response = index_data.list_indices(query="沪深", coverage="ready", data_dir=tmp_path)
    summary = index_data.index_summary(tmp_path)

    assert response["total"] == 1
    assert response["items"][0]["latest_date"] == "2026-08-31"
    assert summary["catalog_count"] == 2
    assert summary["covered_count"] == 1
    assert summary["coverage_rate"] == pytest.approx(0.5)
    assert summary["execution"]["backend"] == "numba_njit_fixed_signature"
    assert summary["execution"]["nopython"] is True
    assert summary["execution"]["object_mode"] == 0
    assert summary["execution"]["python_fallback"] == 0
    assert summary["execution"]["request_time_compilation"] == 0
    domestic = next(item for item in summary["datasets"] if item["key"] == "index_domestic")
    assert domestic["earliest_date"] == "2026-08-27"
    assert domestic["latest_date"] == "2026-08-31"


def test_index_list_filters_and_preserves_nulls(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    pd.DataFrame(
        columns=[
            "source_api", "ts_code", "first_date", "latest_date", "rows", "stale_days",
            "domestic_trade_day_coverage", "source_file", "source_fingerprint",
        ]
    ).to_parquet(tmp_path / "index_coverage_snapshot.parquet", index=False)
    index_data._read_small_tables_cached.cache_clear()

    response = index_data.list_indices(source="ths_index", market="A", page_size=100, data_dir=tmp_path)

    assert response["total"] == 1
    assert response["items"][0]["publisher"] is None
    assert response["items"][0]["coverage_status"] == "missing"


def test_scope_normalisation_adds_catalog_and_rejects_unknown() -> None:
    assert index_data.normalise_index_scopes(["global"]) == ["catalog", "global"]
    try:
        index_data.normalise_index_scopes(["unknown"])
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown scope should fail")


def test_index_routes_forward_filters_and_declare_page_limit(monkeypatch) -> None:
    captured = {}

    def fake_list(**kwargs):
        captured.update(kwargs)
        return {
            "schema_version": 1,
            "status": "complete",
            "page": kwargs["page"],
            "page_size": kwargs["page_size"],
            "total": 0,
            "items": [],
            "filters": {},
        }

    monkeypatch.setattr(index_routes, "list_indices", fake_list)
    response = index_routes.get_indices(
        q="沪深",
        source="index_basic",
        market="CSI",
        category="规模指数",
        active="active",
        coverage="ready",
        page=2,
        page_size=50,
    )

    assert response["status"] == "complete"
    assert captured == {
        "query": "沪深",
        "source": "index_basic",
        "market": "CSI",
        "category": "规模指数",
        "active": "active",
        "coverage": "ready",
        "page": 2,
        "page_size": 50,
    }
    page_size_query = inspect.signature(index_routes.get_indices).parameters["page_size"].default
    assert any(getattr(item, "le", None) == 100 for item in page_size_query.metadata)


def test_selected_index_snapshot_validation_requires_complete_scope(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    catalog = pd.read_parquet(tmp_path / "index_catalog_df.parquet")
    catalog = pd.concat(
        [
            catalog,
            pd.DataFrame(
                [
                    {
                        "source_api": "index_global",
                        "quote_source_api": "index_global",
                        "ts_code": "SPX",
                        "name": "标普500",
                        "category": "国际指数",
                        "market": "GLOBAL",
                        "publisher": None,
                        "list_date": pd.NaT,
                        "exp_date": pd.NaT,
                        "status": "active",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    catalog.to_parquet(tmp_path / "index_catalog_df.parquet", index=False)
    pd.DataFrame([{"ts_code": "000300.SH"}]).to_parquet(
        tmp_path / "index_info.parquet", index=False
    )
    pd.DataFrame([{"ts_code": "000300.SH"}]).to_parquet(
        tmp_path / "etf_index.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "source_api": "index_global",
                "ts_code": "SPX",
                "trade_date": pd.Timestamp("2026-08-31"),
                "close": 1.0,
            }
        ]
    ).to_parquet(tmp_path / "index_global_daily_df.parquet", index=False)
    index_data.build_index_coverage_snapshot(tmp_path)

    report = index_data.validate_index_snapshot(tmp_path, ["global"])

    assert report["status"] == "passed"
    assert report["scopes"] == ["catalog", "global"]
    (tmp_path / "index_global_daily_df.parquet").unlink()
    with pytest.raises(index_data.IndexDataValidationError, match="缺少文件"):
        index_data.validate_index_snapshot(tmp_path, ["global"])


def test_index_validation_accepts_typed_empty_source_without_catalog_universe(
    tmp_path: Path,
) -> None:
    _write_catalog(tmp_path)
    pd.DataFrame([{"ts_code": "000300.SH"}]).to_parquet(
        tmp_path / "index_info.parquet", index=False
    )
    pd.DataFrame([{"ts_code": "000300.SH"}]).to_parquet(
        tmp_path / "etf_index.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "source_api": "sw_daily",
                "ts_code": "801010.SI",
                "trade_date": pd.Timestamp("2026-08-31"),
            }
        ]
    ).to_parquet(tmp_path / "index_sw_daily_df.parquet", index=False)
    pd.DataFrame(
        {
            "source_api": pd.Series(dtype="string"),
            "ts_code": pd.Series(dtype="string"),
            "trade_date": pd.Series(dtype="datetime64[ns]"),
        }
    ).to_parquet(tmp_path / "index_ci_daily_df.parquet", index=False)
    catalog = pd.read_parquet(tmp_path / "index_catalog_df.parquet")
    catalog = pd.concat(
        [
            catalog,
            pd.DataFrame(
                [
                    {
                        "source_api": "index_classify",
                        "quote_source_api": "sw_daily",
                        "ts_code": "801010.SI",
                        "name": "农林牧渔",
                        "category": "L1",
                        "market": "SW",
                        "publisher": "申万宏源研究",
                        "list_date": pd.Timestamp("2021-12-13"),
                        "exp_date": pd.Timestamp("2099-12-31"),
                        "status": "active",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    catalog.to_parquet(tmp_path / "index_catalog_df.parquet", index=False)
    index_data.build_index_coverage_snapshot(tmp_path)

    report = index_data.validate_index_snapshot(tmp_path, ["industry"])

    assert report["datasets"]["index_ci_daily_df.parquet"] == {
        "rows": 0,
        "status": "unavailable",
    }


def test_index_validation_accepts_valuation_source_for_catalog_code(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    pd.DataFrame([{"ts_code": "000300.SH"}]).to_parquet(
        tmp_path / "index_info.parquet", index=False
    )
    pd.DataFrame([{"ts_code": "000300.SH"}]).to_parquet(
        tmp_path / "etf_index.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "source_api": "index_dailybasic",
                "ts_code": "000300.SH",
                "trade_date": pd.Timestamp("2026-08-31"),
                "pe": 12.0,
            }
        ]
    ).to_parquet(tmp_path / "index_daily_basic_df.parquet", index=False)
    index_data.build_index_coverage_snapshot(tmp_path)

    report = index_data.validate_index_snapshot(tmp_path, ["valuation"])

    assert report["status"] == "passed"
