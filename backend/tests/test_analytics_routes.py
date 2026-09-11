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
from fit import ClassNavResult


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

    # **_k absorbs the research-context kwargs the route now threads through.
    monkeypatch.setattr(routes, "compute_classes_nav", lambda *_a, **_k: ClassNavResult(nav, corr, metrics, {}, pd.Series(dtype="datetime64[ns]")))
    monkeypatch.setattr(routes, "compute_class_consistency", lambda *_a, **_k: consistency_rows)

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
    assert resp.metrics[0]["cumulative_return"] == pytest.approx(0.2)
    assert resp.execution["fit_analytics"]["python_fallback"] == 0
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
        lambda *_a, **_k: (idx, series, metrics),
    )

    payload = routes.RollingRequest(
        startDate="2024-01-01",
        window=30,
        targetCode="T",
        targetName="Target",
        etfs=[routes.FitETFIn(code="ETF1", name="ETF One", weight=1.0)],
    )

    resp = routes.rolling_corr(payload)
    assert resp.series["target"] == [None, None, 0.5]
    assert resp.metrics[0]["annual_vol"] is None
    assert resp.metrics[0]["sharpe"] is None
    assert resp.execution["execution_backend"] == "numba_njit_fixed_signature"
    assert resp.execution["nopython"] is True
    assert resp.execution["object_mode"] == 0
    assert resp.execution["python_fallback"] == 0
    assert resp.execution["request_time_compilation"] == 0


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


def test_efficient_frontier_refuses_what_the_backtest_refuses(
    monkeypatch, tmp_path: Path
) -> None:
    """The frontier is where the weights are picked, so it must ask the same question.

    Both endpoints read the same allocation through the same loader, but only
    the backtest ever judged the candidate set: a series that cannot be replayed
    was refused there and quietly plotted here, under the same strict口径.
    """

    from pit.context import build_context

    _set_data_dir(tmp_path)
    _write_asset_nv(tmp_path)  # no as_of column: a full-hindsight series
    monkeypatch.setattr(
        routes, "resolve_request_context", lambda *_a, **_k: build_context("2024-01-31", "STRICT_PIT")
    )

    resp = routes.post_efficient_frontier(
        routes.FrontierRequest(
            alloc_name="demo",
            start_date="2024-01-01",
            end_date="2024-01-31",
            return_metric={"type": "simple"},
            risk_metric={"type": "std"},
        )
    )

    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 400
    assert "幸存者偏差" in _json(resp)["detail"]


def test_efficient_frontier_carries_the_universe_verdict(monkeypatch, tmp_path: Path) -> None:
    """Research mode plots the cloud and prints why it should not be trusted."""

    from pit.context import build_context

    _set_data_dir(tmp_path)
    _write_asset_nv(tmp_path)
    monkeypatch.setattr(
        routes, "resolve_request_context", lambda *_a, **_k: build_context("2024-01-31")
    )
    monkeypatch.setattr(
        routes,
        "calculate_efficient_frontier_exploration",
        lambda **_: {"asset_names": ["AssetA", "AssetB"], "scatter": [], "frontier": []},
    )

    resp = routes.post_efficient_frontier(
        routes.FrontierRequest(
            alloc_name="demo",
            start_date="2024-01-01",
            end_date="2024-01-31",
            return_metric={"type": "simple"},
            risk_metric={"type": "std"},
        )
    )

    assert resp["pit"]["universe"]["clean"] is False
    assert [item["code"] for item in resp["pit"]["universe"]["findings"]] == [
        "UNIVERSE_NOT_REPLAYABLE"
    ]


def test_fit_classes_judges_the_locked_universe_against_the_research_day(
    monkeypatch, tmp_path: Path
) -> None:
    """The manual workspace showed no口径 at all and never named its pool.

    The classes are built out of a locked universe that carries the day it was
    screened; judged against the day being decided, a pool screened later is
    future knowledge wearing a historical date. Research mode records it on the
    result — which is also how the page finally gets a口径 to print — and strict
    mode refuses the run.
    """

    from product_pools.constants import UNIVERSE_SNAPSHOT_STORE
    from product_pools.repository import InvestableUniverseRepository
    from pit.context import build_context

    _set_data_dir(tmp_path)
    universe = InvestableUniverseRepository(tmp_path / UNIVERSE_SNAPSHOT_STORE).create(
        {
            "name": "手动大类测试域",
            "research_date": "2026-09-04",
            "version_refs": [{"pool_id": "pool-1", "version_id": "version-1"}],
            "members": [
                {"kind": "etf", "product_id": "ETF1", "name": "ETF One", "eligible": True}
            ],
            "summary": {"pool_count": 1, "member_count": 1, "eligible_count": 1},
            "content_hash": "manual-hash",
        }
    )

    idx = pd.date_range("2021-01-04", periods=3, freq="D")
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
    monkeypatch.setattr(routes, "compute_classes_nav", lambda *_a, **_k: ClassNavResult(nav, corr, metrics, {}, pd.Series(dtype="datetime64[ns]")))
    monkeypatch.setattr(routes, "compute_class_consistency", lambda *_a, **_k: [])

    payload = routes.FitRequest(
        startDate="2021-01-01",
        universe_snapshot_id=universe["id"],
        classes=[
            routes.FitClassIn(
                id="c1",
                name="ClassA",
                etfs=[routes.FitETFIn(code="ETF1", name="ETF One", weight=1.0)],
            )
        ],
    )

    monkeypatch.setattr(routes, "resolve_request_context", lambda *_a, **_k: build_context("2021-09-01"))
    resp = routes.fit_classes(payload)
    finding = resp.pit["universe"]
    assert finding["clean"] is False
    assert finding["established_at"] == "2026-09-04"
    assert [item["code"] for item in finding["findings"]] == ["UNIVERSE_LOOKAHEAD"]

    monkeypatch.setattr(
        routes, "resolve_request_context", lambda *_a, **_k: build_context("2021-09-01", "STRICT_PIT")
    )
    refused = routes.fit_classes(payload)
    assert isinstance(refused, JSONResponse)
    assert refused.status_code == 400
    assert "未来信息" in _json(refused)["detail"]


def test_fit_classes_without_a_universe_reports_that_rather_than_clean(
    monkeypatch, tmp_path: Path
) -> None:
    """Saying nothing about the pool must not read as "the pool was checked"."""

    _set_data_dir(tmp_path)
    idx = pd.date_range("2021-01-04", periods=2, freq="D")
    nav = pd.DataFrame({"ClassA": [1.0, 1.1]}, index=idx)
    corr = pd.DataFrame([[1.0]], index=["ClassA"], columns=["ClassA"])
    metrics = pd.DataFrame(
        {
            "年化收益率": [0.1],
            "年化波动率": [0.2],
            "夏普比率": [0.5],
            "99%VaR(日)": [0.03],
            "99%ES(日)": [0.04],
            "最大回撤": [-0.1],
            "卡玛比率": [1.0],
        },
        index=["ClassA"],
    )
    monkeypatch.setattr(routes, "compute_classes_nav", lambda *_a, **_k: ClassNavResult(nav, corr, metrics, {}, pd.Series(dtype="datetime64[ns]")))
    monkeypatch.setattr(routes, "compute_class_consistency", lambda *_a, **_k: [])
    resp = routes.fit_classes(
        routes.FitRequest(
            startDate="2021-01-01",
            classes=[
                routes.FitClassIn(
                    id="c1",
                    name="ClassA",
                    etfs=[routes.FitETFIn(code="ETF1", name="ETF One", weight=1.0)],
                )
            ],
        )
    )
    assert resp.pit["universe"]["source"] is None
    assert resp.pit["universe"]["replayable"] is False
