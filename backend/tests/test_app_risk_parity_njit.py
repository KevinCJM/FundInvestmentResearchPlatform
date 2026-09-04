from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend import app as app_module


def _market_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=160)
    index = np.arange(dates.size, dtype=np.float64)
    first = 100.0 * np.exp(0.0004 * index + 0.004 * np.sin(index / 9.0))
    second = 100.0 * np.exp(0.0002 * index + 0.002 * np.cos(index / 13.0))
    return pd.concat(
        [
            pd.DataFrame({"ts_code": "A", "name": "产品A", "date": dates, "adj_nav": first}),
            pd.DataFrame({"ts_code": "B", "name": "产品B", "date": dates, "adj_nav": second}),
        ],
        ignore_index=True,
    )


def _request() -> app_module.SolveRequest:
    return app_module.SolveRequest(
        assetClassId="equity",
        riskMetric="vol",
        maxLeverage=0.0,
        etfs=[
            app_module.ETFIn(code="A", name="产品A", riskContribution=50.0),
            app_module.ETFIn(code="B", name="产品B", riskContribution=50.0),
        ],
    )


def test_risk_budget_route_returns_only_audited_njit_weights(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "_load_adj_nav", lambda *_args, **_kwargs: _market_frame())

    response = app_module.solve(_request())

    assert isinstance(response, app_module.SolveResponse)
    assert sum(response.weights) == 100.0
    assert response.execution["backend"] == "numba_njit_fixed_signature"
    assert response.execution["nopython"] is True
    assert response.execution["python_fallback"] == 0


def test_risk_budget_route_never_falls_back_for_missing_product(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "_load_adj_nav",
        lambda *_args, **_kwargs: _market_frame().loc[lambda frame: frame["ts_code"] == "A"],
    )

    response = app_module.solve(_request())

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    detail = json.loads(response.body.decode("utf-8"))["detail"]
    assert "禁止用预算权重降级替代" in detail
