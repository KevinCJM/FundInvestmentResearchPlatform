import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.services import analytics_routes as routes


def _set_data_dir(tmp_path: Path) -> None:
    routes.DATA_DIR = tmp_path


def _json(resp: JSONResponse) -> Dict[str, Any]:
    return json.loads(resp.body.decode("utf-8"))


def test_fit_classes_returns_sanitised_payload(monkeypatch, tmp_path: Path) -> None:
    _set_data_dir(tmp_path)

    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    nav = pd.DataFrame({"ClassA": [1.0, 1.1, 1.2]}, index=idx)
    corr = pd.DataFrame([[1.0]], index=["ClassA"], columns=["ClassA"])
    metrics = pd.DataFrame(
        {
            "年化收益率": [0.12],
            "年化波动率": [0.2],
            "夏普比率": [0.6],
            "99%VaR(日)": [0.03],
            "99%ES(日)": [0.04],
            "最大回撤": [-0.1],
            "卡玛比率": [1.2],
        },
        index=["ClassA"],
    )
    consistency_rows: List[Dict[str, Any]] = [
        {"name": "ClassA", "mean_corr": 0.9, "pca_evr1": 0.8, "max_te": 0.05},
    ]

    monkeypatch.setattr(routes, "compute_classes_nav", lambda *_: (nav, corr, metrics))
    monkeypatch.setattr(routes, "compute_class_consistency", lambda *_: consistency_rows)

    payload = routes.FitRequest(
        startDate="2024-01-01",
        classes=[
            routes.FitClassIn(
                id="c1",
                name="ClassA",
                etfs=[routes.FitETFIn(code="ETF1", name="ETF One", weight=1.0)],
            )
        ],
    )

    resp = routes.fit_classes(payload)
    assert resp.dates == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert resp.navs["ClassA"][-1] == pytest.approx(1.2)
    assert resp.metrics[0]["annual_return"] == pytest.approx(0.12)
    assert resp.consistency[0]["mean_corr"] == pytest.approx(0.9)


def test_rolling_corr_rejects_non_finite(monkeypatch, tmp_path: Path) -> None:
    _set_data_dir(tmp_path)

    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    series = {"target": [float("nan"), float("inf"), 0.5]}
    metrics = [
        {"name": "target", "annual_vol": float("nan"), "sharpe": float("inf")}
    ]

    monkeypatch.setattr(
        routes,
        "compute_rolling_corr",
        lambda *_: (idx, series, metrics),
    )

    payload = routes.RollingRequest(
        startDate="2024-01-01",
        window=30,
        targetCode="T",
        targetName="Target",
        etfs=[routes.FitETFIn(code="ETF1", name="ETF One", weight=1.0)],
    )

    resp = routes.rolling_corr(payload)
    assert resp.series["target"] == [0.0, 0.0, 0.5]
    assert resp.metrics[0]["annual_vol"] == 0.0
    assert resp.metrics[0]["sharpe"] == 0.0


def test_efficient_frontier_handles_missing_file(tmp_path: Path) -> None:
    _set_data_dir(tmp_path)

    payload = routes.FrontierRequest(
        alloc_name="demo",
        start_date="2024-01-01",
        end_date="2024-01-31",
        return_metric={"type": "simple"},
        risk_metric={"type": "std"},
    )

    resp = routes.post_efficient_frontier(payload)
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 404
    assert "不存在" in _json(resp)["detail"]


def _write_asset_nv(tmp_path: Path, alloc: str = "demo") -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    records: List[Dict[str, Any]] = []
    for d in dates:
        for i, name in enumerate(["AssetA", "AssetB"]):
            records.append(
                {
                    "date": d,
                    "asset_name": name,
                    "asset_alloc_name": alloc,
                    "nv": float(100 + (i + 1) * (d - dates[0]).days),
                }
            )
    df = pd.DataFrame(records)
    df.to_parquet(tmp_path / "asset_nv.parquet", index=False)


def test_efficient_frontier_filters_invalid_points(monkeypatch, tmp_path: Path) -> None:
    _set_data_dir(tmp_path)
    _write_asset_nv(tmp_path)

    def fake_frontier(**_: Any) -> Dict[str, Any]:
        return {
            "asset_names": ["AssetA", "AssetB"],
            "scatter": [
                {"value": (0.1, 0.2)},
                {"value": (float("nan"), 0.3)},
            ],
            "frontier": [
                {"value": (0.05, 0.15)},
                {"value": (0.04, float("inf"))},
            ],
            "max_sharpe": {"value": (0.12, 0.25)},
            "min_variance": {"value": (float("nan"), 0.1)},
        }

    monkeypatch.setattr(routes, "calculate_efficient_frontier_exploration", fake_frontier)

    payload = routes.FrontierRequest(
        alloc_name="demo",
        start_date="2024-01-01",
        end_date="2024-01-31",
        return_metric={"type": "simple"},
        risk_metric={"type": "std"},
    )

    resp = routes.post_efficient_frontier(payload)
    assert resp["scatter"] == [{"value": (0.1, 0.2)}]
    assert resp["frontier"] == [{"value": (0.05, 0.15)}]
    assert resp["max_sharpe"] == {"value": (0.12, 0.25)}
    assert resp["min_variance"] is None
