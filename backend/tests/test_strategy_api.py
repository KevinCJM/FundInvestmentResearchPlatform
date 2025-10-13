from pathlib import Path
from typing import Dict, Any

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND_DIR = ROOT / 'backend'
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import backend.services.strategy_routes as routes
import backend.backtest_engine as engine


def _set_data_dir(tmp_path: Path) -> None:
    routes.DATA_DIR = tmp_path


def _write_asset_nv(tmp_path: Path, alloc: str = "demo", days: int = 10) -> None:
    dates = pd.date_range('2024-01-01', periods=days, freq='D')
    records = []
    for d in dates:
        records.append({
            'date': d,
            'asset_name': 'ClassA',
            'asset_alloc_name': alloc,
            'nv': float(100 + (d - dates[0]).days)
        })
        records.append({
            'date': d,
            'asset_name': 'ClassB',
            'asset_alloc_name': alloc,
            'nv': float(90 + 0.5 * (d - dates[0]).days)
        })
    df = pd.DataFrame(records)
    (tmp_path / 'asset_nv.parquet').parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp_path / 'asset_nv.parquet', index=False)


def _json_content(resp: JSONResponse) -> Dict[str, Any]:
    return json.loads(resp.body.decode('utf-8'))


def test_compute_weights_respects_window_and_errors(monkeypatch, tmp_path):
    _write_asset_nv(tmp_path, days=6)
    _set_data_dir(tmp_path)

    captured: Dict[str, Any] = {}

    def stub_target(nav, *args, **kwargs):
        captured['length'] = len(nav)
        return [0.5, 0.5]

    monkeypatch.setattr(routes, 'compute_target_weights', stub_target)

    payload = routes.ComputeWeightsRequest(
        alloc_name="demo",
        data_len=3,
        window_mode="rollingN",
        strategy=routes.StrategySpec(
            type="target",
            classes=[routes.StrategyClassItem(name="ClassA"), routes.StrategyClassItem(name="ClassB")],
            return_metric="cumulative",
            risk_metric="vol",
            target="min_risk",
        ),
    )
    resp = routes.api_compute_weights(payload)
    assert isinstance(resp, dict)
    assert captured['length'] == 3

    payload_too_long = payload.copy(update={"data_len": 10})
    resp_err = routes.api_compute_weights(payload_too_long)
    assert isinstance(resp_err, JSONResponse)
    assert resp_err.status_code == 400
    assert "样本不足" in _json_content(resp_err)['detail']


def test_compute_schedule_weights_trims_first_valid(monkeypatch, tmp_path):
    _write_asset_nv(tmp_path, days=10)
    _set_data_dir(tmp_path)

    def stub_target(nav, *args, **kwargs):
        cols = nav.columns
        return [1.0 / len(cols) for _ in cols]

    monkeypatch.setattr(routes, 'compute_target_weights', stub_target)

    payload = routes.ComputeScheduleRequest(
        alloc_name="demo",
        strategy=routes.StrategySpec(
            type="target",
            classes=[routes.StrategyClassItem(name="ClassA"), routes.StrategyClassItem(name="ClassB")],
            rebalance={"enabled": True, "mode": "fixed", "fixedInterval": 1, "recalc": True},
            model={"window_mode": "rollingN", "data_len": 5, "return_metric": "cumulative", "risk_metric": "vol", "target": "min_risk"},
        ),
    )

    data = routes.api_compute_schedule_weights(payload)
    assert isinstance(data, dict)
    assert data['dates'][0] == '2024-01-05'
    assert data['cache_key']


def test_backtest_reuses_cached_weights(monkeypatch, tmp_path):
    _write_asset_nv(tmp_path, days=12)
    _set_data_dir(tmp_path)

    def stub_target(nav, *args, **kwargs):
        cols = nav.columns
        return [1.0 / len(cols) for _ in cols]

    monkeypatch.setattr(routes, 'compute_target_weights', stub_target)

    payload = routes.ComputeScheduleRequest(
        alloc_name="demo",
        strategy=routes.StrategySpec(
            type="target",
            classes=[routes.StrategyClassItem(name="ClassA"), routes.StrategyClassItem(name="ClassB")],
            rebalance={"enabled": True, "mode": "fixed", "fixedInterval": 2, "recalc": True},
            model={"window_mode": "rollingN", "data_len": 4, "return_metric": "cumulative", "risk_metric": "vol", "target": "min_risk"},
        ),
    )
    resp = routes.api_compute_schedule_weights(payload)
    assert isinstance(resp, dict)
    cache_key = resp.get('cache_key')
    assert cache_key

    def fail_compute(*args, **kwargs):
        raise AssertionError('unexpected recompute')

    monkeypatch.setattr(engine, 'compute_target_weights', fail_compute)

    backtest_req = routes.BacktestRequest(
        alloc_name="demo",
        strategies=[
            routes.StrategySpec(
                type="target",
                name="demo-target",
                classes=[
                    routes.StrategyClassItem(name="ClassA", weight=0.5),
                    routes.StrategyClassItem(name="ClassB", weight=0.5),
                ],
                rebalance={"enabled": True, "mode": "fixed", "fixedInterval": 2, "recalc": True},
                model=payload.strategy.model,
                precomputed=cache_key,
            )
        ]
    )

    result = routes.api_backtest(backtest_req)
    assert 'dates' in result
    assert 'demo-target' in (result.get('markers') or {})


def test_backtest_cached_and_uncached_results_match(monkeypatch, tmp_path):
    _write_asset_nv(tmp_path, days=14)
    _set_data_dir(tmp_path)

    def stub_target(nav, *args, **kwargs):
        cols = nav.columns
        base = [float(i + 1) for i in range(len(cols))]
        total = sum(base)
        return [x / total for x in base]

    monkeypatch.setattr(routes, 'compute_target_weights', stub_target)
    monkeypatch.setattr(engine, 'compute_target_weights', stub_target)

    schedule_payload = routes.ComputeScheduleRequest(
        alloc_name="demo",
        strategy=routes.StrategySpec(
            type="target",
            classes=[routes.StrategyClassItem(name="ClassA"), routes.StrategyClassItem(name="ClassB")],
            rebalance={"enabled": True, "mode": "fixed", "fixedInterval": 3, "recalc": True},
            model={"window_mode": "rollingN", "data_len": 5, "return_metric": "cumulative", "risk_metric": "vol", "target": "min_risk"},
        ),
    )

    schedule_data = routes.api_compute_schedule_weights(schedule_payload)
    assert isinstance(schedule_data, dict)
    cache_key = schedule_data.get('cache_key')
    assert cache_key

    precomputed_req = routes.BacktestRequest(
        alloc_name="demo",
        strategies=[
            routes.StrategySpec(
                type="target",
                name="cached",
                classes=[
                    routes.StrategyClassItem(name="ClassA", weight=0.5),
                    routes.StrategyClassItem(name="ClassB", weight=0.5),
                ],
                rebalance={"enabled": True, "mode": "fixed", "fixedInterval": 3, "recalc": True},
                model=schedule_payload.strategy.model,
                precomputed=cache_key,
            )
        ]
    )

    data_cached = routes.api_backtest(precomputed_req)

    with routes._schedule_cache_lock:
        routes._schedule_cache.clear()

    uncached_req = routes.BacktestRequest(
        alloc_name="demo",
        strategies=[
            routes.StrategySpec(
                type="target",
                name="uncached",
                classes=[
                    routes.StrategyClassItem(name="ClassA", weight=0.5),
                    routes.StrategyClassItem(name="ClassB", weight=0.5),
                ],
                rebalance={"enabled": True, "mode": "fixed", "fixedInterval": 3, "recalc": True},
                model=schedule_payload.strategy.model,
            )
        ]
    )

    data_uncached = routes.api_backtest(uncached_req)

    assert data_cached['dates'] == data_uncached['dates']
    assert data_cached['series'].keys() != data_uncached['series'].keys()
    cached_series = list(data_cached['series'].values())[0]
    uncached_series = list(data_uncached['series'].values())[0]
    for a, b in zip(cached_series, uncached_series):
        if a is None or b is None:
            assert a is None and b is None
            continue
        assert a == pytest.approx(b, rel=1e-6, abs=1e-7)
