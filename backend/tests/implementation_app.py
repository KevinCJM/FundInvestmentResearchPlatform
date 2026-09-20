"""Isolated, offline browser fixture. Never points at production data."""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from backend.tests.test_strategic_allocation import workspace
from backend.tests.test_implementation_service import implementation
from backend.pre_investment.routes import build_router

root = Path(os.environ["IMPLEMENTATION_TEST_DIR"]).resolve()
if not str(root).startswith(("/private/tmp/", "/tmp/")):
    raise RuntimeError("Browser fixture requires a temporary directory")
root.mkdir(parents=True, exist_ok=True)
strategic, days = workspace.__wrapped__(root)
service, candidate = implementation.__wrapped__((strategic, days))


@asynccontextmanager
async def lifespan(app):
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(build_router(service))


@app.get("/api/test/implementation-candidate")
def fixture():
    return candidate.model_dump(mode="json")
