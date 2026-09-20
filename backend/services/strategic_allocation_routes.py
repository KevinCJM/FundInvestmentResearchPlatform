"""Application wiring for forward-looking strategic allocation."""
import os
from pathlib import Path

from services.custom_indicator_routes import indicator_service
from services.tactical_allocation_routes import tactical_service
from backend.strategic_allocation.routes import build_router
from backend.strategic_allocation.service import StrategicAllocationService

strategic_service = StrategicAllocationService(
    Path(os.getenv("STRATEGIC_ALLOCATION_DATA_DIR", str(indicator_service.workspace_data_dir))),
    indicator_service.market_data_dir,
    universe_dir=indicator_service.workspace_data_dir,
    tactical_repository=tactical_service.repository,
)
router = build_router(strategic_service)

# One reference service is shared by settings and mandate diagnosis.
from backend.strategic_allocation.risk_scale_routes import build_router as build_risk_scale_router

risk_scale_service = strategic_service.risk_scales
from fastapi import APIRouter
_saa_router = router
router = APIRouter()
router.include_router(_saa_router)
router.include_router(build_risk_scale_router(risk_scale_service))

from backend.pre_investment.service import ImplementationService
from backend.pre_investment.routes import build_router as build_implementation_router

from .published_scenario_routes import published_scenario_service
from backend.pre_investment.scenarios import ScenarioAdapter

implementation_service = ImplementationService(strategic_service, scenarios=ScenarioAdapter(published_scenario_service))
router.include_router(build_implementation_router(implementation_service))
