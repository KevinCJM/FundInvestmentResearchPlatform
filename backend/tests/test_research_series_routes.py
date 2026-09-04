from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.research_series.service import (  # noqa: E402
    ResearchSeriesError,
    ResearchSeriesService,
    read_upload_artifact,
)
from backend.services import research_series_routes as routes  # noqa: E402


_REGISTRY_SPEC = importlib.util.spec_from_file_location(
    "_research_series_v2_registry_contract",
    BACKEND_DIR / "historical_regimes" / "v2_registry.py",
)
assert _REGISTRY_SPEC is not None and _REGISTRY_SPEC.loader is not None
_REGISTRY_MODULE = importlib.util.module_from_spec(_REGISTRY_SPEC)
_REGISTRY_SPEC.loader.exec_module(_REGISTRY_MODULE)
NODE_REGISTRY = _REGISTRY_MODULE.NODE_REGISTRY


def _write_fixture(tmp_path: Path) -> ResearchSeriesService:
    data_dir = tmp_path / "data"
    snapshot = data_dir / "snapshot-a"
    snapshot.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "snapshot_dir": "snapshot-a",
        "activated_at": "2025-03-20T00:00:00+00:00",
        "files": {},
        "validation": {"status": "passed", "datasets": {}},
    }
    (data_dir / "tushare_active.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    pd.DataFrame(
        [
            {
                "source_api": "index_basic",
                "ts_code": "000300.SH",
                "name": "沪深300",
                "category": "规模指数",
                "market": "SSE",
                "publisher": "中证指数",
                "quote_source_api": "index_daily",
                "status": "active",
            },
            {
                "source_api": "etf_index",
                "ts_code": "000300.SH",
                "name": "沪深300",
                "category": "规模指数",
                "market": "SSE",
                "publisher": "中证指数",
                "quote_source_api": "index_daily",
                "status": "active",
            },
            {
                "source_api": "index_global",
                "ts_code": "SPX",
                "name": "标普500",
                "category": "全球指数",
                "market": "US",
                "publisher": "S&P",
                "quote_source_api": "index_global",
                "status": "active",
            },
        ]
    ).to_parquet(snapshot / "index_catalog_df.parquet", index=False)
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    pd.DataFrame(
        {
            "ts_code": ["000300.SH"] * 5,
            "trade_date": dates,
            "close": [100.0, 110.0, np.nan, 121.0, 133.1],
            "pct_chg": [np.nan, 10.0, np.nan, np.nan, 10.0],
            "source_api": ["index_daily"] * 5,
        }
    ).to_parquet(snapshot / "index_daily_df.parquet", index=False)
    pd.DataFrame(
        [
            {
                "source_api": "index_daily",
                "ts_code": "000300.SH",
                "first_date": dates[0],
                "latest_date": dates[-1],
                "rows": 5,
                "stale_days": 0,
                "domestic_trade_day_coverage": 0.8,
                "source_file": "index_daily_df.parquet",
                "source_fingerprint": "fixture",
            }
        ]
    ).to_parquet(snapshot / "index_coverage_snapshot.parquet", index=False)

    macro_dates = pd.date_range("2024-01-31", periods=14, freq="ME")
    macro_values = [100.0 + index for index in range(14)]
    macro_values[5] = np.nan
    macro = pd.DataFrame(
        {
            "observation_date": macro_dates,
            "available_at": macro_dates + pd.Timedelta(days=10),
            "availability_status": ["announced_date"] * 14,
            "source_api": ["cn_cpi"] * 14,
            "ingested_at": ["2025-03-01T00:00:00+00:00"] * 14,
            "revision": [1] * 14,
            "vintage": ["v1"] * 14,
            "nt_val": macro_values,
        }
    )
    revised = macro.iloc[[-1]].copy()
    revised["available_at"] = pd.Timestamp("2025-04-01")
    revised["ingested_at"] = "2025-04-01T00:00:00+00:00"
    revised["revision"] = 2
    revised["vintage"] = "v2"
    revised["nt_val"] = 999.0
    pd.concat([macro, revised], ignore_index=True).to_parquet(
        snapshot / "macro_cn_cpi_df.parquet", index=False
    )

    indicator_payload = {
        "schema_version": 1,
        "items": [
            {
                "current": {
                    "id": "indicator-demo",
                    "revision": 2,
                    "name": "示例指标",
                    "expression": "mean(returns)",
                    "periods": ["1M", "1Y"],
                    "unit": "%",
                    "dsl_version": "2.1.0",
                    "operator_registry_version": "typed-2.1",
                },
                "history": [
                    {
                        "id": "indicator-demo",
                        "revision": 1,
                        "name": "示例指标",
                        "expression": "mean(returns)",
                        "periods": ["1M"],
                        "unit": "%",
                        "dsl_version": "2.0.0",
                    }
                ],
            }
        ],
    }
    (data_dir / "custom_indicators.json").write_text(
        json.dumps(indicator_payload, ensure_ascii=False), encoding="utf-8"
    )
    return ResearchSeriesService(data_dir=data_dir, workspace_data_dir=data_dir)


def _client(monkeypatch, service: ResearchSeriesService) -> TestClient:
    monkeypatch.setattr(routes, "research_series_service", service)
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_catalog_discovers_active_index_macro_indicator_and_upload(monkeypatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, _write_fixture(tmp_path))

    index_response = client.get(
        "/api/research-series/catalog",
        params={"kind": "index", "q": "000300.SH"},
    )
    assert index_response.status_code == 200
    index_payload = index_response.json()
    assert index_payload["total"] == 1
    index = index_payload["items"][0]
    assert index["id"] == "index:index_daily:000300.SH"
    assert index["status"] == "available"
    assert index["default_field"] == "close"
    index_fields = {item["name"]: item["label"] for item in index["fields"]}
    assert index_fields["close"] == "收盘点位"
    assert index_fields["pct_chg"] == "涨跌幅"
    assert index["coverage"]["observations"] == 5
    assert np.isclose(index["missing"]["rate"], 0.2)
    assert index["pit"]["supported"] is True
    assert index["regime_node_type"] == "source.index"
    assert index["binding_parameters"]["ts_code"] == "000300.SH"
    assert index["binding_parameters"]["source_api"] == "index_daily"
    assert index["binding_parameters"]["snapshot_id"] == "snapshot-a"
    assert index["binding_parameters"]["snapshot_generation"] == "snapshot-a"
    assert index["binding_parameters"]["source_file"] == "index_daily_df.parquet"
    expected_checksum = "sha256:" + hashlib.sha256(
        (tmp_path / "data" / "snapshot-a" / "index_daily_df.parquet").read_bytes()
    ).hexdigest()
    assert index["binding_parameters"]["file_checksum"] == expected_checksum
    assert index_payload["execution"]["request_time_compilation"] == 0
    assert index_payload["snapshot"]["legacy_fallback"] is False

    macro_payload = client.get(
        "/api/research-series/catalog", params={"kind": "macro"}
    ).json()
    macro_by_id = {item["id"]: item for item in macro_payload["items"]}
    assert macro_by_id["macro:macro_cn_cpi_df"]["status"] == "available"
    assert macro_by_id["macro:macro_cn_cpi_df"]["pit"]["supported"] is True
    assert (
        macro_by_id["macro:macro_cn_cpi_df"]["pit"]["availability_status_field"]
        == "availability_status"
    )
    assert macro_by_id["macro:macro_cn_cpi_df"]["vintage"]["supported"] is True
    assert macro_by_id["macro:macro_cn_cpi_df"]["regime_node_type"] == "source.macro"
    macro_fields = {
        item["name"]: item["label"]
        for item in macro_by_id["macro:macro_cn_cpi_df"]["fields"]
    }
    assert macro_fields["nt_val"] == "本期值"
    assert macro_by_id["macro:macro_cn_gdp_df"]["status"] == "not_downloaded"

    indicator_payload = client.get(
        "/api/research-series/catalog", params={"kind": "indicator"}
    ).json()
    assert indicator_payload["total"] == 2
    assert {item["indicator_version"]["revision"] for item in indicator_payload["items"]} == {1, 2}
    assert all(item["regime_node_type"] == "source.indicator" for item in indicator_payload["items"])

    upload_payload = client.get(
        "/api/research-series/catalog", params={"kind": "upload"}
    ).json()
    assert upload_payload["items"][0]["status"] == "available"
    assert upload_payload["items"][0]["regime_node_type"] == "source.upload"
    assert upload_payload["capabilities"]["upload"] == {
        "available": True,
        "accepted_formats": ["csv", "json"],
        "parsing_location": "frontend",
        "transport": "profile_registration",
        "persisted": True,
        "persistence": "content_addressed_immutable_parquet",
        "max_rows": 20_000,
        "required_fields": ["date", "value"],
        "optional_fields": ["available_at", "vintage", "revision"],
    }

    for item in (index, indicator_payload["items"][0], upload_payload["items"][0]):
        node_schema = NODE_REGISTRY[item["regime_node_type"]]["parameter_schema"]
        assert set(node_schema.get("required", [])).issubset(item["binding_parameters"])


def test_index_profile_computes_full_sample_then_samples_and_preserves_null(monkeypatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, _write_fixture(tmp_path))

    response = client.post(
        "/api/research-series/profile",
        json={
            "series_id": "index:index_daily:000300.SH",
            "field": "close",
            "rolling_window": 2,
            "sample_limit": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["coverage"]["observations"] == 5
    assert payload["sampling"] == {
        "method": "deterministic_even_spacing",
        "computed_observations": 5,
        "displayed_observations": 3,
        "sample_limit": 3,
        "computed_before_sampling": True,
    }
    assert payload["values"]["raw"] == [100.0, None, 133.1]
    assert payload["values"]["normalized"][1] is None
    assert payload["values"]["return"][1] is None
    assert payload["distribution"]["raw"]["valid_count"] == 4
    assert payload["distribution"]["raw"]["missing_count"] == 1
    assert payload["distribution"]["return"]["valid_count"] == 2
    assert payload["missing"]["null_preserved"] is True
    assert payload["execution"]["execution_backend"] == "numba_njit_fixed_signature"
    assert payload["execution"]["object_mode"] == 0
    assert payload["execution"]["python_fallback"] == 0
    assert payload["execution"]["request_time_compilation"] == 0


def test_macro_profile_honors_as_of_vintage_and_full_sample_statistics(monkeypatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, _write_fixture(tmp_path))

    response = client.post(
        "/api/research-series/profile",
        json={
            "series_id": "macro:macro_cn_cpi_df",
            "field": "nt_val",
            "as_of": "2025-03-15",
            "sample_limit": 5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["coverage"]["observations"] == 14
    assert payload["sampling"]["computed_observations"] == 14
    assert payload["sampling"]["displayed_observations"] == 5
    assert payload["values"]["raw"][-1] == 113.0
    assert payload["values"]["raw"][-1] != 999.0
    assert payload["distribution"]["raw"]["valid_count"] == 13
    assert payload["vintage"]["values"][-1] == "v1"
    assert payload["vintage"]["revisions"][-1] == 1
    assert payload["pit"]["as_of"] == "2025-03-15"
    assert payload["values"]["release_lag_days"][-1] == 10.0
    assert payload["execution"]["request_time_compilation"] == 0


def test_compare_uses_full_strict_intersection_and_pairwise_njit_statistics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    client = _client(monkeypatch, _write_fixture(tmp_path))
    dates = pd.date_range("2024-01-02", periods=5, freq="B")

    response = client.post(
        "/api/research-series/compare",
        json={
            "sample_limit": 3,
            "sources": [
                {
                    "id": "benchmark",
                    "label": "沪深300",
                    "series_id": "index:index_daily:000300.SH",
                    "field": "close",
                },
                {
                    "id": "signal",
                    "label": "上传信号",
                    "frequency": "daily",
                    "inline_rows": [
                        {"date": day.strftime("%Y-%m-%d"), "value": value}
                        for day, value in zip(dates, [1.0, 2.0, 3.0, None, 5.0])
                    ],
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["alignment"] == {
        "method": "strict_date_intersection",
        "intersected_observations": 5,
        "source_observations": [5, 5],
        "computed_before_sampling": True,
    }
    assert payload["dates"] == ["2024-01-02", "2024-01-04", "2024-01-08"]
    assert payload["series"][0]["values"] == [100.0, None, 133.1]
    assert payload["series"][1]["values"] == [1.0, 3.0, 5.0]
    assert payload["series"][0]["binding_parameters"]["snapshot_id"] == "snapshot-a"
    assert payload["series"][1]["binding"] is None
    assert payload["correlation"]["observation_counts"] == [[4, 3], [3, 4]]
    assert payload["scatter_pairs"][0]["observation_count"] == 3
    assert payload["scatter_pairs"][0]["dates"] == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-08",
    ]
    assert payload["common_valid"] == {
        "definition": "all selected series finite on a strictly intersected date",
        "observation_count": 3,
        "start_date": "2024-01-02",
        "end_date": "2024-01-08",
    }
    assert payload["execution"]["execution_backend"] == "numba_njit_fixed_signature"
    assert payload["execution"]["request_time_compilation"] == 0


def test_compare_fails_closed_without_common_dates(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "empty-data"
    data_dir.mkdir()
    client = _client(
        monkeypatch,
        ResearchSeriesService(data_dir=data_dir, workspace_data_dir=data_dir),
    )

    response = client.post(
        "/api/research-series/compare",
        json={
            "sources": [
                {"inline_rows": [{"date": "2024-01-01", "value": 1.0}]},
                {"inline_rows": [{"date": "2024-02-01", "value": 2.0}]},
            ]
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "COMPARE_NO_DATE_OVERLAP"


def test_inline_profile_requires_no_snapshot_and_returns_executable_binding(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "empty-data"
    data_dir.mkdir()
    client = _client(
        monkeypatch,
        ResearchSeriesService(data_dir=data_dir, workspace_data_dir=data_dir),
    )

    response = client.post(
        "/api/research-series/profile",
        json={
            "name": "上传景气序列",
            "frequency": "monthly",
            "as_of": "2024-04-15",
            "rolling_window": 2,
            "sample_limit": 3,
            "inline_rows": [
                {
                    "date": "2024-01-31",
                    "value": "100.0",
                    "available_at": "2024-02-10",
                    "vintage": "v1",
                },
                {
                    "date": "2024-02-29",
                    "value": None,
                    "available_at": "2024-03-10",
                    "vintage": "v1",
                },
                {
                    "date": "2024-03-31",
                    "value": 110.0,
                    "available_at": "2024-04-10",
                    "vintage": "v1",
                    "revision": 1,
                },
                {
                    "date": "2024-03-31",
                    "value": 999.0,
                    "available_at": "2024-05-01",
                    "vintage": "v2",
                    "revision": 2,
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["coverage"] == {
        "start_date": "2024-01-31",
        "end_date": "2024-03-31",
        "input_observations": 4,
        "observations": 3,
        "valid_observations": 2,
    }
    assert payload["values"]["raw"] == [100.0, None, 110.0]
    assert payload["missing"] == {
        "count": 1,
        "rate": pytest.approx(1.0 / 3.0),
        "null_preserved": True,
    }
    assert payload["sampling"]["computed_before_sampling"] is True
    assert payload["regime_node_type"] == "source.upload"
    assert payload["binding"]["node_type"] == "source.upload"
    assert "rows" not in payload["binding_parameters"]
    assert set(payload["binding_parameters"]) == {
        "artifact_id",
        "checksum",
        "format",
        "value_field",
        "date_field",
        "available_at_field",
        "vintage_field",
        "revision_field",
        "frequency",
        "availability_mode",
        "name",
    }
    assert payload["artifact"]["artifact_id"] == payload["binding_parameters"]["artifact_id"]
    assert payload["artifact"]["checksum"] == payload["binding_parameters"]["checksum"]
    assert payload["artifact"]["uri"].startswith("research-series-upload://")
    assert payload["snapshot"]["source"] == "content_addressed_upload_artifact"
    assert payload["snapshot"]["persisted"] is True
    stored = read_upload_artifact(
        data_dir,
        payload["artifact"]["artifact_id"],
        payload["artifact"]["checksum"],
    )
    assert len(stored) == 4
    assert pd.isna(stored.loc[1, "value"])
    replay = client.post(
        "/api/research-series/profile",
        json={
            "artifact_id": payload["artifact"]["artifact_id"],
            "checksum": payload["artifact"]["checksum"],
            "frequency": "monthly",
            "as_of": "2024-04-15",
            "sample_limit": 3,
        },
    )
    assert replay.status_code == 200
    assert replay.json()["artifact"]["artifact_id"] == payload["artifact"]["artifact_id"]
    with pytest.raises(ResearchSeriesError) as checksum_error:
        read_upload_artifact(
            data_dir,
            payload["artifact"]["artifact_id"],
            "sha256:" + "0" * 64,
        )
    assert checksum_error.value.code == "UPLOAD_ARTIFACT_CHECKSUM_MISMATCH"
    with pytest.raises(ResearchSeriesError) as unsafe_id:
        read_upload_artifact(data_dir, "../snapshot-a")
    assert unsafe_id.value.code == "UPLOAD_ARTIFACT_ID_INVALID"
    assert payload["execution"]["execution_backend"] == "numba_njit_fixed_signature"
    assert payload["execution"]["request_time_compilation"] == 0


@pytest.mark.parametrize(
    ("row", "expected_code"),
    [
        (
            {"date": "2024-02-01", "value": 1.0, "available_at": "2024-01-31"},
            "INLINE_AVAILABLE_BEFORE_OBSERVATION",
        ),
        ({"date": "2024-02-01", "value": "NaN"}, "INLINE_VALUE_INVALID"),
        ({"date": "not-a-date", "value": 1.0}, "INVALID_DATE"),
        ({"date": "2024-02-01", "value": 1.0, "vintage": ""}, "INLINE_VINTAGE_INVALID"),
    ],
)
def test_inline_profile_validates_dates_values_and_vintage(
    monkeypatch,
    tmp_path: Path,
    row: dict[str, object],
    expected_code: str,
) -> None:
    data_dir = tmp_path / "empty-data"
    data_dir.mkdir()
    client = _client(
        monkeypatch,
        ResearchSeriesService(data_dir=data_dir, workspace_data_dir=data_dir),
    )

    response = client.post(
        "/api/research-series/profile",
        json={"rows": [row]},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == expected_code


def test_inline_profile_enforces_service_row_limit_without_compiling_new_signatures(
    tmp_path: Path,
) -> None:
    service = ResearchSeriesService(data_dir=tmp_path, workspace_data_dir=tmp_path)
    row = {"date": "2024-01-01", "value": 1.0}

    with pytest.raises(ResearchSeriesError) as caught:
        service.profile(inline_rows=[row] * 20_001)

    assert getattr(caught.value, "code", None) == "INLINE_ROWS_LIMIT_EXCEEDED"
    assert getattr(caught.value, "status_code", None) == 413


def test_profile_fails_closed_for_not_downloaded_or_non_profile_kind(monkeypatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, _write_fixture(tmp_path))

    missing = client.post(
        "/api/research-series/profile",
        json={"series_id": "macro:macro_cn_gdp_df", "field": "gdp"},
    )
    unsupported = client.post(
        "/api/research-series/profile",
        json={"series_id": "indicator:indicator-demo@2"},
    )

    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "SERIES_NOT_DOWNLOADED"
    assert unsupported.status_code == 400
    assert unsupported.json()["detail"]["code"] == "PROFILE_KIND_UNSUPPORTED"


def test_catalog_requires_active_manifest_and_never_uses_legacy_files(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {"ts_code": ["000300.SH"], "trade_date": [pd.Timestamp("2024-01-02")], "close": [100.0]}
    ).to_parquet(data_dir / "index_daily_df.parquet", index=False)
    client = _client(
        monkeypatch,
        ResearchSeriesService(data_dir=data_dir, workspace_data_dir=data_dir),
    )

    response = client.get("/api/research-series/catalog")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "ACTIVE_SNAPSHOT_REQUIRED"


def test_main_app_mounts_research_series_and_warms_njit_against_index_only_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    service = _write_fixture(tmp_path)
    snapshot = service.data_dir / "snapshot-a"
    (snapshot / "macro_cn_cpi_df.parquet").unlink()
    # A legacy-root macro file is deliberately present. The catalog must still
    # report macro data as unavailable because only the active snapshot counts.
    pd.DataFrame(
        {
            "observation_date": [pd.Timestamp("2020-01-31")],
            "nt_val": [123.0],
        }
    ).to_parquet(service.data_dir / "macro_cn_cpi_df.parquet", index=False)

    main_app = importlib.import_module("app")
    app_routes = importlib.import_module("services.research_series_routes")
    monkeypatch.setattr(app_routes, "research_series_service", service)

    with TestClient(main_app.app) as client:
        index_payload = client.get(
            "/api/research-series/catalog",
            params={"kind": "index", "q": "000300.SH"},
        ).json()
        macro_payload = client.get(
            "/api/research-series/catalog",
            params={"kind": "macro"},
        ).json()
        indicator_payload = client.get(
            "/api/research-series/catalog",
            params={"kind": "indicator"},
        ).json()
        compare_response = client.post(
            "/api/research-series/compare",
            json={
                "sources": [
                    {
                        "inline_rows": [
                            {"date": "2024-01-01", "value": 1.0},
                            {"date": "2024-01-02", "value": 2.0},
                        ]
                    },
                    {
                        "inline_rows": [
                            {"date": "2024-01-01", "value": 3.0},
                            {"date": "2024-01-02", "value": 4.0},
                        ]
                    },
                ]
            },
        )

        assert index_payload["items"][0]["status"] == "available"
        assert all(item["status"] == "not_downloaded" for item in macro_payload["items"])
        assert all(item["coverage"]["observations"] == 0 for item in macro_payload["items"])
        current = next(
            item
            for item in indicator_payload["items"]
            if item["id"] == "indicator:indicator-demo@2"
        )
        assert current["regime_node_type"] == "source.indicator"
        assert current["binding_parameters"] == {
            "indicator_id": "indicator-demo",
            "indicator_revision": 2,
            "product_kind": "",
            "product_id": "",
            "period": "1M",
            "name": "示例指标",
        }
        assert current["binding_required_inputs"] == ["product_kind", "product_id"]
        assert compare_response.status_code == 200
        assert compare_response.json()["alignment"]["intersected_observations"] == 2
        assert compare_response.json()["execution"]["request_time_compilation"] == 0

        startup = main_app.app.state.numba_warmup
        assert startup["complete"] is True
        assert startup["research_series"]["fully_warmed"] is True
        assert startup["research_series"]["execution_backend"] == "numba_njit_fixed_signature"
        assert startup["research_series"]["request_time_compilation"] == 0
