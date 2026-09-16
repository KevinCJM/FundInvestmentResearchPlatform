"""Isolated real API for audit regression browser tests; no production data."""
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
import regime_completion_app as regime_fixture
from backend.sensitivity.kernels import warm_sensitivity_kernels
from backend.services.risk_model_routes import build_router as risk_router
from backend.services.published_scenario_routes import build_router as published_router
from services import scenario_stress_routes as stress_routes
from scenario_stress.service import ScenarioStressService
from scenario_stress.numba_kernels import warm_scenario_numba_kernels
from test_published_risk_models import environment
from test_scenario_stress import _template

warm_sensitivity_kernels()
warm_scenario_numba_kernels()
env = environment.__wrapped__(regime_fixture._ROOT)
stress_routes.scenario_stress_service = ScenarioStressService(regime_fixture._ROOT, regime_fixture._ROOT)
first = stress_routes.scenario_stress_service.create_definition({**_template('factor_path'), 'name': '并发载入甲'})
second = stress_routes.scenario_stress_service.create_definition({**_template('factor_path'), 'name': '并发载入乙'})
fields = {'name': '旧版无 Horizon 证据', 'entry': 'market', 'frequency': 'monthly',
          'input_ids': [env.factor['id']], 'rows': [[-10.], [0.]]}
preview, arrays = env.scenarios._compile_preview(fields)
legacy_fields = {k: v for k, v in preview.items() if k not in {
    'id', 'created_at', 'content_hash', 'immutable', 'transient', 'horizon_evidence', 'publication_request_id'}}
legacy = env.scenarios.artifacts.save('preview', legacy_fields, arrays)
now = datetime.now(timezone.utc)
release = env.scenarios.artifacts.save('release', {
    'name': fields['name'], 'entry': 'market', 'frequency': 'monthly',
    'preview_id': legacy['id'], 'preview_hash': preview['preview_hash'], 'preview_content_hash': legacy['content_hash'],
    'factors': preview['factors'], 'horizon': 2, 'lineage': [], 'note': 'offline legacy fixture',
    'effective_at': (now - timedelta(days=1)).isoformat(), 'expires_at': (now + timedelta(days=90)).isoformat(),
})


@asynccontextmanager
async def lifespan(app):
    async with regime_fixture.lifespan(app):
        app.state.identity.update(factor_id=env.factor['id'], factor_name=env.factor['name'],
            legacy_release_id=release['id'], legacy_preview_id=legacy['id'],
            definition_a=first['id'], definition_b=second['id'])
        yield


app = FastAPI(lifespan=lifespan)
app.include_router(regime_fixture.routes.router)
app.include_router(stress_routes.router)
app.include_router(risk_router(env.model))
app.include_router(risk_router(env.transmission))
app.include_router(published_router(env.scenarios))


@app.get('/ready')
def ready():
    return {'ready': True, **app.state.identity}
