"""Isolated real formula API for the browser round-trip regression.

Only used by Playwright; never touches the user's saved indicators or market
files. No numerical evaluation endpoints are exercised by this UI test.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from tempfile import TemporaryDirectory

from fastapi import FastAPI

_workspace = TemporaryDirectory(prefix="indicator-formula-e2e-")
os.environ["CUSTOM_INDICATOR_DATA_DIR"] = _workspace.name

from services.custom_indicator_routes import router  # noqa: E402


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        yield
    finally:
        _workspace.cleanup()


app = FastAPI(lifespan=lifespan)
app.include_router(router)


@app.get("/ready")
def ready():
    return {"ready": True}
