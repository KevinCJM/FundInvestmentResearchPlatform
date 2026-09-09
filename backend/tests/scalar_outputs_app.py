"""Disposable production-route app for scalar-output browser acceptance only."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI

_workspace = TemporaryDirectory(prefix="scalar-outputs-e2e-")
os.environ["CUSTOM_INDICATOR_DATA_DIR"] = _workspace.name

from custom_indicators.service import CustomIndicatorService  # noqa: E402
from services import custom_indicator_routes as routes  # noqa: E402
from test_custom_indicator_service import _write_market_data  # noqa: E402

_root = Path(_workspace.name)
_write_market_data(_root)
routes.indicator_service = CustomIndicatorService(_root, _root)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        yield
    finally:
        routes.indicator_service.close_compute_engine()
        _workspace.cleanup()


app = FastAPI(lifespan=lifespan)
app.include_router(routes.router)


@app.get("/ready")
def ready():
    return {"ready": True, "test_fixture": True}
