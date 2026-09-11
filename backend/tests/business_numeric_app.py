"""Isolated, data-free API used only for process-navigation browser regression."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from business_numeric_numba import warm_business_numeric_kernels
from services.business_numeric_routes import router


@asynccontextmanager
async def lifespan(_app):
    warm_business_numeric_kernels()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router)


@app.get("/ready")
def ready():
    return {"ready": True}
