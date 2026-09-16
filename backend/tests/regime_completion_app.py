"""Isolated real regime API for browser integration; never uses production stores."""
from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path

# Set boundaries before importing routes, whose service instances are module-level.
_SANDBOX = tempfile.TemporaryDirectory(prefix="regime-completion-browser-")
_ROOT = Path(_SANDBOX.name)
os.environ["CUSTOM_INDICATOR_DATA_DIR"] = str(_ROOT)
os.environ["HISTORICAL_REGIME_DATA_DIR"] = str(_ROOT)

import numpy as np
from fastapi import FastAPI
from historical_regimes.v2_service import RegimeGraphV2Service
from services import historical_regime_routes as routes
from test_historical_regime_v2 import _definition
from test_historical_regime_v2_p1 import _upload_source


@asynccontextmanager
async def lifespan(app: FastAPI):
    graph = RegimeGraphV2Service(_ROOT, _ROOT)
    rows = [
        {
            "observation_date": (date(2020, 1, 1) + timedelta(days=i)).isoformat(),
            "available_at": (date(2020, 1, 1) + timedelta(days=i)).isoformat(),
            "value": float(100 + 12 * np.sin(i / 12)),
        }
        for i in range(900)
    ]
    historical = _definition(rows)
    historical["name"] = "浏览器验收·历史定义（合成数据）"
    historical["graph"]["nodes"][0] = _upload_source(graph, rows)
    historical["study"] = {"purpose": "historical_reference", "family": "market_trend"}
    saved = graph.create_definition(historical)
    plan = graph.prepare(saved)
    cutoff = rows[-1]["observation_date"]
    run = graph.run_saved(
        {"schema_version": "2.0", "id": saved["id"], "revision": saved["revision"]},
        "retrospective", cutoff, plan["compile_token"],
    )
    publication = graph.publish(run["id"], "research_display")["publication"]
    reference = {
        "run_id": run["id"], "publication_id": publication["id"],
        "content_hash": run["content_hash"],
    }
    realtime = _definition(rows)
    realtime["name"] = "浏览器验收·实时模型（合成数据）"
    realtime["graph"]["nodes"][0] = historical["graph"]["nodes"][0]
    realtime["study"] = {
        "purpose": "realtime_recognition", "family": "market_trend", "reference": reference,
    }
    model = graph.create_definition(realtime)
    graph.prepare(model)
    routes.regime_graph_v2_service = graph
    app.state.identity = {
        "historical_id": saved["id"], "realtime_id": model["id"],
        "reference": reference, "cutoff": cutoff,
    }
    yield
    _SANDBOX.cleanup()


app = FastAPI(lifespan=lifespan)
app.include_router(routes.router)


@app.get("/ready")
def ready():
    return {"ready": True, **app.state.identity}
