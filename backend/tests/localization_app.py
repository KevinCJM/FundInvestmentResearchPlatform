"""Disposable real API fixture for localization and indicator browser acceptance."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from tempfile import TemporaryDirectory

from fastapi import FastAPI

_workspace = TemporaryDirectory(prefix="localization-e2e-")
os.environ["I18N_DATA_DIR"] = _workspace.name
os.environ["CUSTOM_INDICATOR_DATA_DIR"] = _workspace.name

from services.localization_routes import router as localization_router  # noqa: E402
from services.custom_indicator_routes import router as indicator_router  # noqa: E402


@asynccontextmanager
async def lifespan(_app):
    try:
        yield
    finally:
        _workspace.cleanup()


app = FastAPI(lifespan=lifespan)
app.include_router(localization_router)
app.include_router(indicator_router)


@app.get("/ready")
def ready():
    return {"ready": True}
