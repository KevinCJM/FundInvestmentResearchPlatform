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
