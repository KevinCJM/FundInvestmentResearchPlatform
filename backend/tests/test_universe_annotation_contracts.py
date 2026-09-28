"""Optional annotations remove manual burden; quantitative identity stays gated."""
from datetime import date, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.strategic_allocation.routes import build_router
from backend.strategic_allocation.service import StrategicAllocationService

PREVIEW = "/api/strategic-allocation/universes/preview"
CONFIRM = "/api/strategic-allocation/universes/confirm"


def _client(tmp_path) -> TestClient:
    service = StrategicAllocationService(tmp_path / "research", tmp_path / "market")
    app = FastAPI()
    app.include_router(build_router(service))
    return TestClient(app)


def _asset(**overrides) -> dict:
    asset = {
        "id": "growth",
        "name": "增长资产",
        "role": "growth",
        "liquidity": "liquid",
        "currency": "CNY",
    }
    asset.update(overrides)
    return asset


def _body(**overrides) -> dict:
    body = {
        "name": "简化录入战略范围",
        "as_of": str(date.today()),
        "currency": "CNY",
        "assets": [_asset(), _asset(id="cash", name="现金储备", role="liquidity")],
    }
    body.update(overrides)
    return body


def test_blank_annotations_preview_confirm_and_read_back_unchanged(tmp_path) -> None:
    client = _client(tmp_path)
    body = _body()

    preview = client.post(PREVIEW, json=body)
    assert preview.status_code == 200, preview.text
    definition = preview.json()["definition"]
    assert definition["source"] == ""
    assert all(asset["rationale"] == "" and asset["source"] == "" for asset in definition["assets"])

    confirmed = client.post(
        CONFIRM,
        json={"request": body, "preview_hash": preview.json()["preview_hash"]},
    )
    assert confirmed.status_code == 201, confirmed.text
    saved = confirmed.json()
    assert saved["content_hash"]
    assert saved["definition"]["assets"][0]["rationale"] == ""
    read_back = client.get(f"/api/strategic-allocation/universes/{saved['id']}")
    assert read_back.status_code == 200
    assert read_back.json() == saved


def test_filled_annotations_are_frozen_with_their_hash(tmp_path) -> None:
    client = _client(tmp_path)
    body = _body(
        source="研究员显式来源",
        assets=[_asset(rationale="长期增长风险", source="研究员定义")],
    )

    preview = client.post(PREVIEW, json=body)
    assert preview.status_code == 200, preview.text
    confirmed = client.post(
        CONFIRM,
        json={"request": body, "preview_hash": preview.json()["preview_hash"]},
    )
    assert confirmed.status_code == 201, confirmed.text
    assert preview.json()["preview_hash"] == confirmed.json()["preview_hash"]
    assert confirmed.json()["definition"]["assets"][0]["rationale"] == "长期增长风险"
    assert confirmed.json()["definition"]["source"] == "研究员显式来源"


def test_quantitative_bounds_stay_enforced(tmp_path) -> None:
    client = _client(tmp_path)
    cases = {
        "bad identifier": _body(assets=[_asset(id="Growth")]),
        "currency mismatch": _body(assets=[_asset(currency="USD")]),
        "future research date": _body(as_of=str(date.today() + timedelta(days=1))),
        "no assets": _body(assets=[]),
        "blank name": _body(name=""),
        "overlong annotation": _body(assets=[_asset(rationale="x" * 1001)]),
    }
    for label, body in cases.items():
        response = client.post(PREVIEW, json=body)
        assert response.status_code == 422, f"{label}: {response.status_code} {response.text}"


def test_mapping_proxy_rationale_is_optional_but_identity_is_not() -> None:
    from pydantic import ValidationError

    from backend.strategic_allocation.universe_contracts import ProxyAssignment

    assignment = ProxyAssignment(strategic_asset_id="growth", proxy_asset_id="equity")
    assert assignment.rationale == ""
    with pytest.raises(ValidationError):
        ProxyAssignment(strategic_asset_id="", proxy_asset_id="equity")
    with pytest.raises(ValidationError):
        ProxyAssignment(strategic_asset_id="growth", proxy_asset_id="equity", rationale="x" * 2001)


def _proxy(**overrides):
    proxy = {"asset_type": "market", "cash_return": None, "rebalance": "monthly",
             "components": [{"kind": "index", "series_id": "index:index_daily:000300.SH", "field": "close", "weight": 1}],
             "source_labels": {"index:index_daily:000300.SH": "沪深300"}}
    return {**proxy, **overrides}


def test_proxy_and_cash_inputs_survive_confirm_read_and_hash_gate(tmp_path):
    client = _client(tmp_path)
    body = _body(assets=[_asset(research_proxy=_proxy()), _asset(id="cash", name="现金", role="liquidity",
        research_proxy=_proxy(asset_type="cash", cash_return=.02, components=[], rebalance=None, source_labels={}))])
    preview = client.post(PREVIEW, json=body)
    assert preview.status_code == 200, preview.text
    changed = _body(**{**body, "assets": [dict(body["assets"][0], research_proxy=_proxy(rebalance="daily")), body["assets"][1]]})
    assert client.post(CONFIRM, json={"request": changed, "preview_hash": preview.json()["preview_hash"]}).status_code == 409
    saved = client.post(CONFIRM, json={"request": body, "preview_hash": preview.json()["preview_hash"]})
    assert saved.status_code == 201, saved.text
    record = client.get(f"/api/strategic-allocation/universes/{saved.json()['id']}").json()
    assert record["definition"]["assets"][0]["research_proxy"] == body["assets"][0]["research_proxy"]
    assert record["definition"]["assets"][1]["research_proxy"]["cash_return"] == .02
    assert record["implementation_status"] == "unmapped"
    assert record["implementation_gaps"] == ["growth", "cash"]


@pytest.mark.parametrize("proxy", [
    _proxy(components=[{"kind": "etf", "series_id": "etf:fund_daily:510300.SH", "field": "close", "weight": 1}], source_labels={}),
    _proxy(components=[{"kind": "index", "series_id": "index:index_daily:000300.SH", "field": "close", "weight": .4}]),
    _proxy(asset_type="cash", cash_return=None, components=[], rebalance=None, source_labels={}),
])
def test_invalid_proxy_cannot_be_saved(tmp_path, proxy):
    response = _client(tmp_path).post(PREVIEW, json=_body(assets=[_asset(research_proxy=proxy)]))
    assert response.status_code == 422, response.text


def test_proxy_optional_and_old_preview_hash_preserved(tmp_path):
    from backend.sensitivity.repository import digest_json
    client = _client(tmp_path)
    response = client.post(PREVIEW, json=_body())
    result = response.json()
    assert all("research_proxy" not in asset for asset in result["definition"]["assets"])
    assert result["preview_hash"] == digest_json({k: v for k, v in result.items() if k != "preview_hash"})
    incomplete = _proxy(components=[], source_labels={})
    assert client.post(PREVIEW, json=_body(assets=[_asset(research_proxy=incomplete)])).status_code == 200
    assert client.post(PREVIEW, json=_body(currency="USD", assets=[_asset(currency="USD", research_proxy=_proxy())])).status_code == 422


def test_asset_type_drives_cash_membership_and_preserves_existing_liquidity_limits(tmp_path):
    from copy import deepcopy
    from backend.strategic_allocation.contracts import PolicyRequest

    client = _client(tmp_path)
    body = _body(assets=[
        _asset(research_proxy=_proxy(), role="liquidity"),
        _asset(id="restricted", role="credit", liquidity="illiquid", research_proxy=_proxy()),
        _asset(id="cash", name="现金", role="growth", liquidity="illiquid",
               research_proxy=_proxy(asset_type="cash", cash_return=.02, components=[], rebalance=None, source_labels={})),
    ])
    original = deepcopy(body)
    preview = client.post(PREVIEW, json=body)
    assert preview.status_code == 200, preview.text
    saved = client.post(CONFIRM, json={"request": body, "preview_hash": preview.json()["preview_hash"]})
    assert saved.status_code == 201, saved.text
    definition = saved.json()["definition"]
    assert [(a["role"], a["liquidity"]) for a in definition["assets"]] == [
        ("growth", "liquid"), ("credit", "illiquid"), ("liquidity", "liquid")]
    assert body == original
    assert client.get(f"/api/strategic-allocation/universes/{saved.json()['id']}").json()["definition"] == definition

    mandate = {"min_liquid_weight": .1, "max_illiquid_weight": .2, "max_tracking_error": 1.,
               "max_volatility": .3, "institutional_context": {"cash_reserve_weight": .1}}
    groups, _ = StrategicAllocationService(tmp_path / "research", tmp_path / "market")._constraints(PolicyRequest(mandate_id="m", cma_id="c"), definition, mandate)
    by_id = {group["id"]: group for group in groups}
    assert by_id["policy-cash-reserve"]["assets"] == ["cash"]
    assert by_id["policy-cash-reserve"]["lo"] == .1
    assert by_id["policy-liquid-reserve"]["assets"] == ["growth", "cash"]
    assert by_id["policy-illiquid-cap"]["assets"] == ["restricted"]
    assert by_id["policy-illiquid-cap"]["hi"] == .2


def test_typed_assets_need_no_manual_role_or_liquidity_and_untyped_contract_stays_explicit(tmp_path):
    client = _client(tmp_path)
    market = _asset(research_proxy=_proxy())
    cash = _asset(id="cash", research_proxy=_proxy(asset_type="cash", cash_return=.02, components=[], rebalance=None, source_labels={}))
    for asset in (market, cash):
        del asset["role"]
        del asset["liquidity"]
    response = client.post(PREVIEW, json=_body(assets=[market, cash]))
    assert response.status_code == 200, response.text
    assert [(a["role"], a["liquidity"]) for a in response.json()["definition"]["assets"]] == [
        ("growth", "liquid"), ("liquidity", "liquid")]
    del market["research_proxy"]
    assert client.post(PREVIEW, json=_body(assets=[market])).status_code == 422
