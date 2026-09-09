"""Disposable fixture app using real indicator routes and deterministic prices."""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from fastapi import FastAPI

_workspace = TemporaryDirectory(prefix="indicator-primitives-e2e-")
os.environ["CUSTOM_INDICATOR_DATA_DIR"] = _workspace.name
from custom_indicators.service import CustomIndicatorService
from services import custom_indicator_routes as routes

_root = Path(_workspace.name)
pd.DataFrame([{"ts_code": "510050.SH", "code": "510050", "name": "上证50ETF"}]).to_parquet(_root / "etf_info_df.parquet", index=False)
pd.DataFrame({"ts_code": "510050.SH", "name": "上证50ETF", "date": pd.bdate_range("2026-01-02", periods=8), "adj_nav": [1., 1.25, 1., 1.25, 1.5, 1.2, 1.2, 1.5]}).to_parquet(_root / "etf_daily_df.parquet", index=False)
routes.indicator_service = CustomIndicatorService(_root, _root)


@asynccontextmanager
async def lifespan(app):
    from cal_indicators.typed_numba_kernels import warm_numba_kernel_registry
    warm_numba_kernel_registry()
    # This fixture executes only definitions created/validated by each test;
    # those explicit preparation calls warm every plan before formal execution.
    try:
        yield
    finally:
        routes.indicator_service.close_compute_engine()
        _workspace.cleanup()


app = FastAPI(lifespan=lifespan)
app.include_router(routes.router)


@app.get("/ready")
def ready():
    return {"ready": True, "fixture": True}
