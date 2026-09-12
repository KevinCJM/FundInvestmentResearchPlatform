"""Isolated offline browser API with real NJIT/routes and temporary Parquet.

Never import backend.app or the production service singletons here. The fixture
is launched only by the dedicated Playwright configuration, not by production.
"""
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from fastapi import FastAPI

from backend.strategic_allocation.routes import build_router, _call
from backend.strategic_allocation.service import StrategicAllocationService
from backend.tactical_allocation.service import TacticalAllocationService
from backend.tactical_allocation.contracts import PreviewRequest, SaveDecisionRequest
from backend.services import analytics_routes, strategy_routes

storage = tempfile.TemporaryDirectory(prefix='strategic-browser-')
root = Path(storage.name)
days = pd.bdate_range(end=date.today(), periods=301)
returns = np.column_stack((.0003 + .007 * np.sin(np.arange(300) * .3), .0001 + .0015 * np.cos(np.arange(300) * .4)))
nav = np.vstack((np.ones(2), np.cumprod(1 + returns, axis=0)))
pd.DataFrame([{'asset_alloc_name': '浏览器离线股债', 'asset_name': name, 'etf_code': code, 'etf_name': name + '代理',
               'etf_weight': 100., 'creat_time': days[0], 'as_of': None, 'universe_snapshot_id': None, 'data_release_id': None}
              for name, code in [('股票', '510300.SH'), ('债券', '511010.SH')]]).to_parquet(root / 'asset_alloc_info.parquet', index=False)
pd.DataFrame([{'asset_alloc_name': '浏览器离线股债', 'asset_name': name, 'date': day, 'nv': nav[t, i], 'available_at': day, 'as_of': None}
              for i, name in enumerate(('股票', '债券')) for t, day in enumerate(days)]).to_parquet(root / 'asset_nv.parquet', index=False)
pd.DataFrame({'exchange': ['SSE'] * len(days),
              'cal_date': [int(day.strftime('%Y%m%d')) for day in days],
              'is_open': [1] * len(days)}).to_parquet(root / 'trade_day_df.parquet', index=False)
analytics_routes.DATA_DIR = root
strategy_routes.DATA_DIR = root
strategic = StrategicAllocationService(root / 'research', root)
tactical = TacticalAllocationService(root / 'research', root)


@asynccontextmanager
async def lifespan(_app):
    strategic.warm()
    tactical.warm()
    yield
    storage.cleanup()


app = FastAPI(lifespan=lifespan)
app.include_router(build_router(strategic))
app.include_router(analytics_routes.router)
app.include_router(strategy_routes.router)


@app.get('/api/list-allocations')
def list_allocations():
    return ['浏览器离线股债']


@app.get('/api/load-allocation')
def load_allocation(name: str):
    if name != '浏览器离线股债':
        return []
    return [
        {'id': 'equity', 'name': '股票', 'etfs': [{'code': '510300.SH', 'name': '股票代理', 'weight': 100.0}]},
        {'id': 'bond', 'name': '债券', 'etfs': [{'code': '511010.SH', 'name': '债券代理', 'weight': 100.0}]},
    ]


@app.get('/api/historical-regimes/runs')
def historical_runs():
    return {'items': []}


@app.get('/ready')
def ready():
    return {'ready': True, 'fixture': 'offline-real-njit', 'today': str(date.today())}


@app.get('/api/tactical-allocation/catalog')
def tactical_catalog():
    return _call(tactical.catalog)


@app.get('/api/tactical-allocation/baselines/{identifier}')
def baseline(identifier: str):
    return _call(tactical.repository.get_baseline, identifier)


@app.post('/api/tactical-allocation/preflight')
def preflight(body: PreviewRequest):
    return _call(tactical.preflight, body)


@app.post('/api/tactical-allocation/preview')
def preview(body: PreviewRequest):
    return _call(tactical.preview, body)


@app.post('/api/tactical-allocation/decisions')
def save_decision(body: SaveDecisionRequest):
    return _call(tactical.save_decision, body)


@app.get('/api/tactical-allocation/decisions/{identifier}')
def get_decision(identifier: str):
    return _call(tactical.repository.get_decision, identifier)
