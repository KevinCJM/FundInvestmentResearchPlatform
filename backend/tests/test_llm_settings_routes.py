"""Contract tests for the LLM API settings endpoint and its storage."""

from __future__ import annotations

import json
import stat
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setenv("CUSTOM_INDICATOR_DATA_DIR", str(tmp_path))
    from services.llm_settings_routes import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_unconfigured_get_is_200_and_never_requires_a_key(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    response = client.get("/api/settings/llm")
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "configured": False,
        "provider": "openai-compatible",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "reasoning_effort": "default",
        "context_window_tokens": 0,
        "api_key_masked": None,
        "timeout_seconds": 60.0, "profiles": [], "active_profile_id": None,
    }


def test_put_masks_key_and_never_echoes_it(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    secret = "sk-test-1234567890abcdef"
    response = client.put(
        "/api/settings/llm",
        json={"model": "gpt-4o", "base_url": "https://api.example.com/v1", "api_key": secret},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["configured"] is True
    assert payload["api_key_masked"] == "••••cdef"
    assert secret not in json.dumps(payload)

    stored = json.loads((tmp_path / "llm_settings.json").read_text(encoding="utf-8"))
    assert stored["profiles"][0]["api_key"] == secret
    mode = stat.S_IMODE((tmp_path / "llm_settings.json").stat().st_mode)
    assert mode == 0o600

    fetched = client.get("/api/settings/llm").json()
    assert fetched["configured"] is True
    assert secret not in json.dumps(fetched)


def test_update_is_idempotent_and_keeps_key_when_omitted(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    first = client.put("/api/settings/llm", json={"model": "m1", "api_key": "long-secret-key-0001"})
    second = client.put("/api/settings/llm", json={"model": "m1"})
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert second.json()["api_key_masked"] == "••••0001"

    cleared = client.put("/api/settings/llm", json={"api_key": ""})
    assert cleared.status_code == 200
    assert cleared.json()["configured"] is False
    assert cleared.json()["api_key_masked"] is None


def test_validation_and_unknown_fields_are_422(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    missing_scheme = client.put("/api/settings/llm", json={"base_url": "api.example.com"})
    assert missing_scheme.status_code == 422
    assert missing_scheme.json()["detail"]["code"] == "REQUEST_VALIDATION_ERROR"
    extra = client.put("/api/settings/llm", json={"model": "m", "temperature": 0})
    assert extra.status_code == 422


def test_multiple_profiles_require_explicit_activation_and_preserve_keys(tmp_path, monkeypatch):
    from agent.llm_settings import LlmSettingsStore
    from agent.llm import build_client
    client = _client(tmp_path, monkeypatch)
    # Existing installations retain their current service without rewriting secrets on GET.
    old = {'schema_version': 1, 'settings': {'model': 'old-model', 'base_url': 'https://old.example/v1', 'api_key': 'secret-one-1234'}}
    path = tmp_path / 'llm_settings.json'
    path.write_text(json.dumps(old))
    state = client.get('/api/settings/llm').json()
    assert state['active_profile_id'] == 'default'
    assert json.loads(path.read_text()) == old
    new = client.post('/api/settings/llm/profiles', json={'name': '备用', 'model': 'new-model', 'base_url': 'https://new.example/v1', 'api_key': 'secret-two-5678'})
    assert new.status_code == 201
    state = new.json(); second = state['profiles'][1]['id']
    assert state['active_profile_id'] == 'default'
    assert LlmSettingsStore().read()['api_key'] == 'secret-one-1234'
    assert 'secret-one' not in new.text and 'secret-two' not in new.text
    client.put(f'/api/settings/llm/profiles/{second}', json={'name': '备用改名', 'model': 'new-model-v2'})
    assert client.get('/api/settings/llm').json()['profiles'][1]['api_key_masked'] == '••••5678'
    client.put('/api/settings/llm/active', json={'profile_id': second})
    active = LlmSettingsStore().read()
    assert active['model'] == 'new-model-v2' and active['api_key'] == 'secret-two-5678'
    assert build_client(active).model == 'new-model-v2'
    # Invalid selections never change the active service; clearing it never falls back.
    assert client.put('/api/settings/llm/active', json={'profile_id': 'missing'}).status_code == 404
    assert client.get('/api/settings/llm').json()['active_profile_id'] == second
    client.put('/api/settings/llm/active', json={'profile_id': None})
    assert build_client(LlmSettingsStore().read()) is None
    assert len(client.get('/api/settings/llm').json()['profiles']) == 2
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    incomplete = client.post('/api/settings/llm/profiles', json={'name': '未完成', 'model': 'm', 'base_url': 'https://empty.example/v1'}).json()['profiles'][-1]['id']
    assert client.put('/api/settings/llm/active', json={'profile_id': incomplete}).status_code == 400
    assert client.post('/api/settings/llm/profiles', json={'name': ' ', 'model': ' ', 'base_url': 'bad'}).status_code == 422


def test_reasoning_settings_are_independent_persisted_and_resettable(tmp_path, monkeypatch):
    from agent.llm import build_client
    from agent.llm_settings import LlmSettingsStore
    client = _client(tmp_path, monkeypatch)
    client.put('/api/settings/llm', json={'api_key': 'company-secret'})
    created = client.post('/api/settings/llm/profiles', json={
        'name': 'Go', 'model': 'deepseek-v4.1-flash', 'base_url': 'https://opencode.ai/zen/go/v1',
        'api_key': 'go-secret', 'reasoning_effort': 'max', 'timeout_seconds': 600})
    assert created.status_code == 201
    pid = created.json()['profiles'][-1]['id']
    assert LlmSettingsStore().read()['reasoning_effort'] == 'default'
    client.put('/api/settings/llm/active', json={'profile_id': pid})
    llm = build_client(LlmSettingsStore().read(), session_id='session-1')
    assert (llm.reasoning_effort, llm.timeout_seconds, llm.session_id) == ('max', 600, 'session-1')
    client.put(f'/api/settings/llm/profiles/{pid}', json={'name': 'Go 改名'})
    assert client.get('/api/settings/llm').json()['reasoning_effort'] == 'max'
    for bad in [{'reasoning_effort': 'super'}, {'timeout_seconds': 0}, {'timeout_seconds': 1801}, {'timeout_seconds': 1.5}]:
        assert client.put(f'/api/settings/llm/profiles/{pid}', json={'name': 'Go', **bad}).status_code == 422
    assert LlmSettingsStore().read()['reasoning_effort'] == 'max'
    reset = client.put('/api/settings/llm', json={'reasoning_effort': 'default'})
    assert reset.status_code == 200 and reset.json()['reasoning_effort'] == 'default'
    assert LlmSettingsStore().read()['timeout_seconds'] == 600
    assert LlmSettingsStore().read()['api_key'] == 'go-secret'


def test_context_window_override_is_independent_and_zero_restores_auto(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    first = client.post('/api/settings/llm/profiles', json={'name':'a','model':'m','base_url':'https://a.test/v1','api_key':'long-secret-key','context_window_tokens':65536}).json()
    first_id = first['profiles'][0]['id']
    second = client.post('/api/settings/llm/profiles', json={'name':'b','model':'n','base_url':'https://b.test/v1'}).json()
    assert second['profiles'][0]['context_window_tokens'] == 65536
    assert second['profiles'][1]['context_window_tokens'] == 0
    updated = client.put('/api/settings/llm/profiles/'+first_id, json={'name':'a','model':'m2'}).json()
    assert updated['profiles'][0]['context_window_tokens'] == 65536
    updated = client.put('/api/settings/llm/profiles/'+first_id, json={'name':'a','context_window_tokens':0}).json()
    assert updated['profiles'][0]['context_window_tokens'] == 0 and updated['profiles'][0]['configured']
    for invalid in (1,8191,-1,2000001,8192.5,True):
        assert client.put('/api/settings/llm/profiles/'+first_id, json={'name':'a','context_window_tokens':invalid}).status_code == 422
