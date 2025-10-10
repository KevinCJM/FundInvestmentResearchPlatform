import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.services.strategy_routes as routes


def _build_app(tmp_path: Path) -> TestClient:
    routes.DATA_DIR = tmp_path
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


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


def test_compute_weights_respects_window_and_errors(monkeypatch, tmp_path):
    _write_asset_nv(tmp_path, days=6)
    client = _build_app(tmp_path)

    captured: Dict[str, Any] = {}

    def stub_target(nav, *args, **kwargs):
        captured['length'] = len(nav)
        return [0.5, 0.5]

    monkeypatch.setattr(routes, 'compute_target_weights', stub_target)

    payload = {
        "alloc_name": "demo",
        "data_len": 3,
        "window_mode": "rollingN",
        "strategy": {
            "type": "target",
            "classes": [{"name": "ClassA"}, {"name": "ClassB"}],
            "return_metric": "cumulative",
            "risk_metric": "vol",
            "target": "min_risk"
        }
    }
    resp = client.post("/api/strategy/compute-weights", json=payload)
    assert resp.status_code == 200
    assert captured['length'] == 3

    payload['data_len'] = 10
    resp = client.post("/api/strategy/compute-weights", json=payload)
    assert resp.status_code == 400
    assert "样本不足" in resp.json()['detail']


def test_compute_schedule_weights_trims_first_valid(monkeypatch, tmp_path):
    _write_asset_nv(tmp_path, days=10)
    client = _build_app(tmp_path)

    def stub_target(nav, *args, **kwargs):
        cols = nav.columns
        return [1.0 / len(cols) for _ in cols]

    monkeypatch.setattr(routes, 'compute_target_weights', stub_target)

    payload = {
        "alloc_name": "demo",
        "strategy": {
            "type": "target",
            "classes": [{"name": "ClassA"}, {"name": "ClassB"}],
            "rebalance": {"enabled": True, "mode": "fixed", "fixedInterval": 1, "recalc": True},
            "model": {"window_mode": "rollingN", "data_len": 5, "return_metric": "cumulative", "risk_metric": "vol", "target": "min_risk"}
        }
    }

    resp = client.post("/api/strategy/compute-schedule-weights", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data['dates'][0] == '2024-01-05'
