"""Offline full-stack mandate fixture: real routes, frozen scales, NJIT and temporary storage."""
from contextlib import asynccontextmanager
from datetime import date

from backend.tests import strategic_allocation_app as base
from backend.tests.risk_scale_app import make_service
from backend.tests.test_risk_scale_service import freeze_reference, definition as scale_definition, publish
from backend.strategic_allocation.reference_inputs import ReferenceInputs
from backend.strategic_allocation.risk_scale_service import RiskScaleService
from backend.strategic_allocation.risk_scale_store import RiskScaleStore
from backend.strategic_allocation.risk_scale_routes import build_router
from backend.strategic_allocation.contracts import CmaRequest, PublishCmaRequest

fixture_risk, reference_input = make_service(base.root / 'reference-fixture')
research_day = date.today()
base.strategic.risk_scales = RiskScaleService(base.strategic.artifacts,
    RiskScaleStore(base.strategic.artifacts.root.parent),
    ReferenceInputs(base.strategic.artifacts, fixture_risk.references.sources))
app = base.app
app.include_router(build_router(base.strategic.risk_scales))
fixture = {}


@asynccontextmanager
async def lifespan(application):
    async with base.lifespan(application):
        reference, _ = freeze_reference(base.strategic.risk_scales, reference_input)
        scale, _, _ = publish(base.strategic.risk_scales, {**scale_definition(reference), 'research_as_of': str(research_day)})
        actual = CmaRequest(name='浏览器实际CMA', alloc_name='浏览器离线股债', as_of=research_day, currency='CNY',
            horizon_years=10, source='离线测试明确假设，绝非生产预测', basis_confirmed=True,
            assets=[{'id': '股票', 'role': 'growth', 'liquidity': 'liquid', 'rationale': '离线权益测试资产',
                     'annual_return': .06, 'annual_volatility': .15, 'mean_uncertainty': .01},
                    {'id': '债券', 'role': 'rates', 'liquidity': 'liquid', 'rationale': '离线利率测试资产',
                     'annual_return': .025, 'annual_volatility': .01, 'mean_uncertainty': .002}], correlation=[[1., 0.], [0., 1.]])
        preview = base.strategic.preview_cma(actual)
        actual_cma = base.strategic.publish_cma(PublishCmaRequest(request=actual, preview_hash=preview['preview_hash']))
        fixture.update(scale=scale, cma=actual_cma, today=str(research_day))
        yield


app.router.lifespan_context = lifespan


@app.get('/fixture')
def get_fixture():
    return fixture
