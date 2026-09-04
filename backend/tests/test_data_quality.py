from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
for path in (ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.services import data_quality, data_routes


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _write_quality_fixture(tmp_path: Path) -> Path:
    snapshot = tmp_path / "snapshot-v1"
    snapshot.mkdir()
    for kind in ("etf", "fund"):
        pd.DataFrame(
            [
                {"ts_code": f"{kind}-normal", "name": f"{kind} 正常", "status_code": "L"},
                {"ts_code": f"{kind}-issue", "name": f"{kind} 异常", "status_code": "L"},
            ]
        ).to_parquet(snapshot / f"{kind}_info_df.parquet", index=False)
        pd.DataFrame([{"placeholder": 1}]).to_parquet(
            snapshot / ("etf_daily_df.parquet" if kind == "etf" else "fund_nav_df.parquet"),
            index=False,
        )

    etf_fingerprint = _fingerprint(snapshot / "etf_daily_df.parquet")
    fund_fingerprint = _fingerprint(snapshot / "fund_nav_df.parquet")
    pd.DataFrame(
        [
            {
                "instrument_type": "etf", "ts_code": "etf-normal", "as_of": "2026-09-01",
                "latest_date": "2026-09-01", "stale_days": 0, "latest_adj_nav": 1.02,
                "adj_nav_anomaly_count": 0, "quality_reason_1y": None,
                "nav_source_fingerprint": etf_fingerprint,
            },
            {
                "instrument_type": "etf", "ts_code": "etf-issue", "as_of": "2026-09-01",
                "latest_date": "2026-08-20", "stale_days": 12, "latest_adj_nav": 0.98,
                "adj_nav_anomaly_count": 0, "quality_reason_1y": "internal_gap",
                "nav_source_fingerprint": etf_fingerprint,
            },
            {
                "instrument_type": "fund", "ts_code": "fund-normal", "as_of": "2026-09-01",
                "latest_date": "2026-09-01", "stale_days": 0, "latest_adj_nav": 1.12,
                "adj_nav_anomaly_count": 0, "quality_reason_1y": None,
                "nav_source_fingerprint": fund_fingerprint,
            },
            {
                "instrument_type": "fund", "ts_code": "fund-issue", "as_of": "2026-09-01",
                "latest_date": "2026-09-01", "stale_days": 0, "latest_adj_nav": 1.22,
                "adj_nav_anomaly_count": 2, "quality_reason_1y": "adjusted_nav_anomaly",
                "nav_source_fingerprint": fund_fingerprint,
            },
        ]
    ).to_parquet(snapshot / "instrument_metrics_snapshot.parquet", index=False)

    (tmp_path / "tushare_active.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot_dir": "snapshot-v1",
                "activated_at": "2026-09-02T07:19:06Z",
                "validation": {
                    "status": "passed",
                    "datasets": {
                        "etf_info": {"rows": 2, "unique_codes": 2},
                        "fund_info": {"rows": 2, "unique_codes": 2},
                        "etf_nav": {"rows": 20, "unique_codes": 2, "missing_values": 0, "code_coverage": 1.0},
                        "fund_nav": {"rows": 18, "unique_codes": 2, "missing_values": 3, "code_coverage": 1.0},
                        "analytics_snapshot": {"rows": 4, "nav_code_coverage": {"etf": 1.0, "fund": 1.0}},
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return snapshot


def test_quality_report_surfaces_nav_jumps_gaps_staleness_and_validation_evidence(tmp_path: Path) -> None:
    _write_quality_fixture(tmp_path)

    report = data_quality.build_data_quality_report(tmp_path)

    assert report["status"] == "attention"
    assert report["summary"]["total_products"] == 4
    assert report["summary"]["affected_products"] == 2
    assert report["summary"]["nav_anomaly_products"] == 1
    assert report["summary"]["nav_anomaly_events"] == 2
    assert report["summary"]["affected_rate"] == pytest.approx(0.5)
    assert report["execution"]["backend"] == "numba_njit_fixed_signature"
    assert report["execution"]["nopython"] is True
    assert report["execution"]["object_mode"] == 0
    assert report["execution"]["python_fallback"] == 0
    assert report["execution"]["request_time_compilation"] == 0
    issue_by_code = {issue["code"]: issue for issue in report["issues"]}
    assert issue_by_code["NAV_DISCONTINUITY"]["samples"][0]["ts_code"] == "fund-issue"
    assert issue_by_code["SERIES_INTERNAL_GAP"]["affected_count"] == 1
    assert issue_by_code["STALE_ACTIVE_SERIES"]["affected_count"] == 1
    assert issue_by_code["NAV_VALUE_MISSING"]["record_count"] == 3
    check_by_key = {check["key"]: check for check in report["checks"]}
    assert check_by_key["primary_key"]["status"] == "passed"
    assert check_by_key["nav_discontinuity"]["status"] == "warning"
    assert check_by_key["source_consistency"]["status"] == "passed"


def test_quality_report_blocks_duplicate_metric_keys(tmp_path: Path) -> None:
    snapshot = _write_quality_fixture(tmp_path)
    metrics_path = snapshot / "instrument_metrics_snapshot.parquet"
    metrics = pd.read_parquet(metrics_path)
    pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True).to_parquet(metrics_path, index=False)

    report = data_quality.build_data_quality_report(tmp_path)

    assert report["status"] == "blocked"
    duplicate = next(issue for issue in report["issues"] if issue["code"] == "DUPLICATE_METRIC_KEY")
    assert duplicate["severity"] == "critical"
    assert duplicate["affected_count"] == 1


def test_quality_report_fails_closed_on_missing_metric_schema(tmp_path: Path) -> None:
    snapshot = _write_quality_fixture(tmp_path)
    metrics_path = snapshot / "instrument_metrics_snapshot.parquet"
    pd.DataFrame([{"unexpected": "value"}]).to_parquet(metrics_path, index=False)

    report = data_quality.build_data_quality_report(tmp_path)

    assert report["status"] == "blocked"
    schema_issue = next(issue for issue in report["issues"] if issue["code"] == "METRICS_SCHEMA_MISMATCH")
    assert schema_issue["severity"] == "critical"
    assert "instrument_type" in schema_issue["evidence"]


def test_quality_report_flags_missing_latest_nav_and_partial_info_mapping(tmp_path: Path) -> None:
    snapshot = _write_quality_fixture(tmp_path)
    metrics_path = snapshot / "instrument_metrics_snapshot.parquet"
    metrics = pd.read_parquet(metrics_path)
    metrics.loc[metrics["ts_code"].eq("etf-normal"), "latest_adj_nav"] = None
    metrics.to_parquet(metrics_path, index=False)
    (snapshot / "fund_info_df.parquet").unlink()

    report = data_quality.build_data_quality_report(tmp_path)

    issue_by_code = {issue["code"]: issue for issue in report["issues"]}
    assert issue_by_code["LATEST_NAV_MISSING"]["affected_count"] == 1
    assert issue_by_code["INFO_MAPPING_UNAVAILABLE"]["evidence"] == "不可用产品类型：FUND。"
    check_by_key = {check["key"]: check for check in report["checks"]}
    assert check_by_key["field_validity"]["status"] == "warning"
    assert check_by_key["freshness"]["status"] == "warning"


def test_quality_route_is_read_only_and_uses_current_manager_root(monkeypatch, tmp_path: Path) -> None:
    _write_quality_fixture(tmp_path)

    class Manager:
        data_dir = tmp_path

    monkeypatch.setattr(data_routes, "refresh_manager", Manager())

    response = type("Response", (), {"headers": {}})()
    report = data_routes.data_quality(response)

    assert response.headers["Cache-Control"] == "no-store"
    assert report["summary"]["total_products"] == 4
