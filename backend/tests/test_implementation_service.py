import copy
import io
import json
import zipfile
from datetime import date, timedelta
import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.tests.test_strategic_allocation import workspace, saved_inputs, warm
from backend.strategic_allocation.contracts import PublishPolicyRequest
from backend.pre_investment.service import ImplementationService
from backend.pre_investment.routes import build_router
from backend.pre_investment.contracts import (
    ImplementationCandidate,
    PackageWrite,
    PackageAction,
    FinalizePackage,
)
from backend.custom_indicators.errors import ConflictError, ValidationError


@pytest.fixture
def implementation(workspace):
    strategic, days = workspace
    root = strategic.data.data_dir
    info = pd.read_parquet(root / "asset_alloc_info.parquet")
    info["universe_snapshot_id"] = "universe-implementation"
    info.to_parquet(root / "asset_alloc_info.parquet", index=False)
    codes = ["510300.SH", "511010.SH"]
    snapshot = {
        "id": "universe-implementation",
        "name": "实施测试域",
        "research_date": str(days[0].date()),
        "immutable": True,
        "members": [
            {"kind": "etf", "product_id": c, "name": c, "eligible": True} for c in codes
        ],
    }
    (root / "product_pools.json").write_text(
        json.dumps({"pools": [], "versions": [], "universe_snapshots": [snapshot]})
    )
    nav = pd.read_parquet(root / "asset_nv.parquet")
    rows = []
    for name, code in zip(["股票", "债券"], codes, strict=True):
        for item in nav.loc[nav.asset_name == name].itertuples():
            rows.append(
                {
                    "ts_code": code,
                    "date": item.date,
                    "trade_date": item.date.strftime("%Y%m%d"),
                    "adj_nav": item.nv,
                }
            )
    pd.DataFrame(rows).to_parquet(root / "etf_daily_df.parquet", index=False)
    pd.DataFrame(
        [{"ts_code": c, "name": c, "found_date": "20000101"} for c in codes]
    ).to_parquet(root / "etf_basic_df.parquet", index=False)
    strategic.warm()
    _, _, request = saved_inputs(strategic)
    preview = strategic.preview_policy(request)
    candidate = next(
        x for x in preview["candidates"] if x.get("available") is not False
    )
    baseline = strategic.publish_policy(
        PublishPolicyRequest(
            request=request,
            preview_hash=preview["preview_hash"],
            candidate_id=candidate["id"],
            name="实施政策",
            reason="离线研究确认采用此政策",
        )
    )
    service = ImplementationService(strategic)
    service.warm()
    body = ImplementationCandidate(
        name="股债实施方案",
        source={
            "kind": "saa_policy",
            "id": baseline["id"],
            "content_hash": baseline["content_hash"],
        },
        start_date=days[0].date(),
        as_of=date.today(),
        state={
            "valuation_at": str(date.today()),
            "knowledge_cutoff": str(date.today()),
            "confirmed_investable_value": 100000.0,
            "settled_cash": 100000.0,
            "evidence": "离线已确认账户余额",
        },
        return_basis_confirmed=True,
        horizon_stationarity_acknowledged=True,
        products=[
            {
                "kind": "etf",
                "product_id": code,
                "asset_class_id": name,
                "weight": candidate["weights"][name],
                "buy_rate": 0.0005,
                "sell_rate": 0.0005,
                "fee_basis": "each_side_notional",
                "fee_source": "离线声明每边费率",
                "fee_valid_until": str(date.today() + timedelta(days=90)),
                "nav_includes_management_fee": True,
            }
            for name, code in zip(["股票", "债券"], codes, strict=True)
        ],
    )
    return service, body


def test_oversized_package_is_rejected_without_breaking_saved_history(implementation):
    service, body = implementation
    first = service.save(PackageWrite(candidate=body, idempotency_key="size-valid-package"))
    raw = body.model_dump(mode="json")
    raw["state"]["reconciliation"] = [
        {"occurrence_id": f"{index:064x}", "status": "paid", "paid_amount": 1.,
         "evidence": "证" * 2000}
        for index in range(1400)
    ]
    oversized = ImplementationCandidate.model_validate(raw)
    for _ in range(2):
        with pytest.raises(ValidationError, match="容量上限"):
            service.save(PackageWrite(candidate=oversized, idempotency_key="size-large-package"))
    assert service.repository.list() == [first]
    assert service.repository.current(first["scheme_id"]) == first
    assert not list(service.repository.artifacts.root.glob(".writing-*"))
    assert len(list(service.repository.artifacts.root.glob("*/manifest.json"))) == 1
    assert service.save(PackageWrite(candidate=body, idempotency_key="size-valid-package")) == first


def test_direct_saa_risk_cost_preview_is_pure(implementation):
    service, body = implementation
    result = service.preview(body)
    assert result["research_ready"], result["checks"]
    assert result["models"] and all(x["status"] == "passed" for x in result["models"])
    assert result["transition"]["cost"] > 0
    assert (
        result["historical_replay"]["net_terminal"]
        < result["historical_replay"]["gross_terminal"]
    )
    assert result["implementation_eligibility"] == "conditions_incomplete"
    assert not service.repository.artifacts.root.exists()


def saved_report(service, body):
    item = service.save(
        PackageWrite(candidate=body, idempotency_key="package-create-001")
    )
    action = PackageAction(
        expected_revision=item["revision"],
        candidate_hash=item["candidate_hash"],
        idempotency_key="package-validate-001",
    )
    validated = service.validate(item["scheme_id"], action)
    assert service.validate(item["scheme_id"], action) == validated
    return validated


def test_busy_validation_leaves_no_revision_or_attempt_and_can_retry(implementation):
    from backend.pre_investment.service import _COMPUTE

    service, body = implementation
    item = service.save(PackageWrite(candidate=body, idempotency_key="busy-create"))
    action = PackageAction(expected_revision=item["revision"],
        candidate_hash=item["candidate_hash"], idempotency_key="busy-validation")
    before = service.view(item["scheme_id"])
    assert _COMPUTE.acquire(blocking=False)
    try:
        with pytest.raises(ConflictError, match="稍后重试"):
            service.validate(item["scheme_id"], action)
    finally:
        _COMPUTE.release()
    assert service.view(item["scheme_id"]) == before
    validated = service.validate(item["scheme_id"], action)
    assert validated["revision"] == item["revision"] + 2
    assert validated["stage"] == "validation_complete"
    assert service.validate(item["scheme_id"], action) == validated


def test_validation_releases_compute_slot_when_freeze_fails(implementation, monkeypatch):
    from backend.pre_investment.service import _COMPUTE

    service, body = implementation
    item = service.save(PackageWrite(candidate=body, idempotency_key="freeze-failure-create"))
    action = PackageAction(expected_revision=item["revision"],
        candidate_hash=item["candidate_hash"], idempotency_key="freeze-failure-validation")
    def fail(*args, **kwargs):
        raise OSError("controlled freeze failure")
    monkeypatch.setattr(service.repository, "append", fail)
    with pytest.raises(OSError, match="controlled freeze failure"):
        service.validate(item["scheme_id"], action)
    assert _COMPUTE.acquire(blocking=False)
    _COMPUTE.release()
    assert service.repository.history(item["scheme_id"]) == [item]


def test_copy_lineage_survives_reopen_edits_validation_and_export(implementation):
    service, body = implementation
    app = FastAPI()
    app.include_router(build_router(service))
    client = TestClient(app)
    original = service.save(PackageWrite(candidate=body, idempotency_key="copy-source-001"))
    create = {"candidate": body.model_dump(mode="json"), "copied_from_id": original["id"],
              "idempotency_key": "copy-create-001"}
    response = client.post("/api/pre-investment/packages", json=create)
    assert response.status_code == 201, response.text
    copied = response.json()
    assert client.post("/api/pre-investment/packages", json=create).json() == copied
    path = f"/api/pre-investment/packages/{copied['scheme_id']}"
    for index, supplied_source in enumerate((None, copied["id"])):
        reopened = client.get(path).json()["package"]
        edit = {"candidate": {**reopened["candidate"], "name": f"复制后编辑{index}"},
                "expected_revision": reopened["revision"], "copied_from_id": supplied_source,
                "idempotency_key": f"copy-edit-{index:03d}"}
        response = client.put(path, json=edit)
        assert response.status_code == 200, response.text
        current = response.json()
        assert current["copied_from_id"] == original["id"]
        assert client.put(path, json=edit).json() == current
    stale = {**edit, "idempotency_key": "copy-stale-edit"}
    assert client.put(path, json=stale).status_code == 409
    validated = service.validate(copied["scheme_id"], PackageAction(
        expected_revision=current["revision"], candidate_hash=current["candidate_hash"],
        idempotency_key="copy-validate-001"))
    report = service.repository.report(validated["report_id"])
    service.finalize(copied["scheme_id"], FinalizePackage(
        expected_revision=validated["revision"], candidate_hash=validated["candidate_hash"],
        validation_report_hash=report["content_hash"], idempotency_key="copy-finalize-001",
        reviewer="本地研究员", reason="复核复制来源与研究证据", accept_research_limits=True,
        review_due_at=date.today() + timedelta(days=30)))
    with zipfile.ZipFile(io.BytesIO(client.get(path + "/export").content)) as archive:
        exported = json.loads(archive.read("research-package.json"))
    assert exported["package"]["copied_from_id"] == original["id"]
    assert all(row["copied_from_id"] == original["id"] for row in exported["history"])
    assert service.repository.current(original["scheme_id"]) == original


def test_copy_requires_an_existing_package_revision(implementation):
    service, body = implementation
    app = FastAPI()
    app.include_router(build_router(service))
    client = TestClient(app)
    original = service.save(PackageWrite(candidate=body, idempotency_key="copy-source-001"))
    other = service.repository.artifacts.save("series", {"artifact_type": "other", "name": "其他成果"})
    for identifier, status in (("series-missing", 404), (original["scheme_id"], 404), (other["id"], 422)):
        response = client.post("/api/pre-investment/packages", json={
            "candidate": body.model_dump(mode="json"), "copied_from_id": identifier,
            "idempotency_key": "copy-invalid-source"})
        assert response.status_code == status, response.text
        assert service.repository.list() == [original]


def test_report_finalize_export_and_current_retirement(implementation):
    service, body = implementation
    item = saved_report(service, body)
    report = service.repository.report(item["report_id"])
    action = FinalizePackage(
        expected_revision=item["revision"],
        candidate_hash=item["candidate_hash"],
        validation_report_hash=report["content_hash"],
        idempotency_key="package-finalize-001",
        reviewer="本地研究员",
        reason="核对研究结果，并接受明确列出的实施证据限制",
        review_due_at=date.today() + timedelta(days=30),
        accept_research_limits=True,
    )
    final = service.finalize(item["scheme_id"], action)
    assert service.finalize(item["scheme_id"], action) == final
    assert final["stage"] == "finalized" and not final["review"]["independent_review"]
    with zipfile.ZipFile(io.BytesIO(service.export(item["scheme_id"]))) as archive:
        assert "arrays/beta.npy" in archive.namelist()
        exported = json.loads(archive.read("research-package.json"))
        assert exported["report"]["candidate_hash"] == final["candidate_hash"]
    baseline = service.strategic.baselines.get_baseline(body.source.id)
    service.strategic.retire_mandate(baseline["policy"]["mandate_id"])
    assert (
        service.view(item["scheme_id"])["current_eligibility"]["status"]
        == "needs_review"
    )
    assert service.repository.current(item["scheme_id"]) == final


def test_changed_weight_invalidates_old_report_and_failed_gate_blocks_finalize(
    implementation,
):
    service, body = implementation
    item = saved_report(service, body)
    changed = body.model_dump(mode="json")
    changed["products"][0]["weight"] += 0.1
    changed = ImplementationCandidate.model_validate(changed)
    updated = service.save(
        PackageWrite(
            candidate=changed,
            expected_revision=item["revision"],
            idempotency_key="changed-weight-001",
        ),
        item["scheme_id"],
    )
    assert (
        updated["candidate_hash"] != item["candidate_hash"]
        and updated["report_id"] is None
    )
    assert not service.preview(changed)["research_ready"]
    with pytest.raises(ConflictError):
        service.validate(
            item["scheme_id"],
            PackageAction(
                expected_revision=item["revision"],
                candidate_hash=item["candidate_hash"],
                idempotency_key="stale-report-001",
            ),
        )


def test_unknown_fees_no_zero_and_api_contract(implementation):
    service, body = implementation
    unknown = body.model_dump(mode="json")
    unknown["products"][0]["buy_rate"] = None
    result = service.preview(ImplementationCandidate.model_validate(unknown))
    assert (
        next(x for x in result["checks"] if x["check_id"] == "costs")["status"]
        == "unavailable"
    )
    app = FastAPI()
    app.include_router(build_router(service))
    client = TestClient(app)
    assert client.get("/api/pre-investment/catalog").status_code == 200
    response = client.post(
        "/api/pre-investment/preview", json=body.model_dump(mode="json")
    )
    assert response.status_code == 200, response.text
    assert response.json()["candidate_hash"]
    invalid = body.model_dump(mode="json")
    invalid["validation_seed"] = invalid["search_seed"]
    assert client.post("/api/pre-investment/preview", json=invalid).status_code == 422


def test_already_paid_transition_cannot_hide_pending_trades(implementation):
    service, body = implementation
    raw = body.model_dump(mode="json")
    raw["state"]["transition_cost_in_balance"] = True
    result = service.preview(ImplementationCandidate.model_validate(raw))
    assert "transition" not in result
    assert (
        next(x for x in result["checks"] if x["check_id"] == "costs")["error_code"]
        == "IMPLEMENTATION_COST_ALREADY_INCLUDED"
    )
    for p in raw["products"]:
        p["current_value"] = p["weight"] * 100000.0
    raw["state"]["settled_cash"] = 0.0
    result = service.preview(ImplementationCandidate.model_validate(raw))
    assert result["transition"]["cost"] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("mode", ["parameter_average", "compatible_all_models"])
def test_original_model_gates_and_frozen_sources(implementation, mode):
    from backend.tests.test_multi_cma import setup
    from backend.strategic_allocation.contracts import PolicyRequest

    service, body = implementation
    _, originals, request = setup(service.strategic)
    raw = request.model_dump(mode="json")
    raw["mode"] = mode
    if mode == "compatible_all_models":
        for ref in raw["cma_refs"]:
            ref.pop("weight", None)
    request = PolicyRequest.model_validate(raw)
    preview = service.strategic.preview_policy(request)
    selected = next(
        x
        for x in preview["candidates"]
        if x.get("available") is not False and x.get("within_limits", True)
    )
    baseline = service.strategic.publish_policy(
        PublishPolicyRequest(
            request=request,
            preview_hash=preview["preview_hash"],
            candidate_id=selected["id"],
            name=mode,
            reason="多模型产品桥接离线验收",
        )
    )
    raw = body.model_dump(mode="json")
    raw["source"] = {
        "kind": "saa_policy",
        "id": baseline["id"],
        "content_hash": baseline["content_hash"],
    }
    for p in raw["products"]:
        p["weight"] = selected["weights"][p["asset_class_id"]]
    result = service.preview(ImplementationCandidate.model_validate(raw))
    assert result["research_ready"], result["checks"]
    rows = result["models"]
    assert len(rows) == (3 if mode == "parameter_average" else 2)
    assert all(
        x["enforced"] == (mode == "compatible_all_models")
        for x in rows
        if x["model_id"] in {s["id"] for s in originals}
    )
    assert (
        len(result["frozen_source"]["baseline"]["policy"]["multi_cma"]["sources"]) == 2
    )


def test_saved_taa_target_flows_into_same_implementation_path(implementation):
    from backend.tactical_allocation.service import TacticalAllocationService
    from backend.tactical_allocation.contracts import (
        PreviewRequest,
        SaveDecisionRequest,
    )

    service, body = implementation
    taa = TacticalAllocationService(
        service.strategic.artifacts.root.parents[1], service.strategic.data.data_dir
    )
    dates = (
        pd.read_parquet(service.strategic.data.data_dir / "asset_nv.parquet")["date"]
        .drop_duplicates()
        .sort_values()
    )
    request = PreviewRequest(
        baseline_id=body.source.id,
        start_date=dates.iloc[0].date(),
        train_end_date=dates.iloc[100].date(),
        end_date=dates.iloc[-1].date(),
        as_of=date.today(),
        signal_mode="manual",
        manual_tilts={"股票": 0.001, "债券": -0.001},
        max_tracking_error=0.04,
        search=False,
    )
    preview = taa.preview(request)
    saved = taa.save_decision(
        SaveDecisionRequest(
            request=request, preview_hash=preview["preview_hash"], name="承接 TAA 研究"
        )
    )
    raw = body.model_dump(mode="json")
    raw["source"] = {
        "kind": "taa_decision",
        "id": saved["id"],
        "content_hash": saved["content_hash"],
    }
    for p in raw["products"]:
        p["weight"] = saved["preview"]["recommendation"]["weights"][p["asset_class_id"]]
    result = service.preview(ImplementationCandidate.model_validate(raw))
    assert result["research_ready"], result["checks"]
    assert result["dependencies"]["source"]["id"] == saved["id"]
    assert result["historical_replay"]["scope"].endswith("not_dynamic_taa")


def test_unmapped_product_is_hard_failure_and_validation_failure_remains_visible(
    implementation, monkeypatch
):
    service, body = implementation
    raw = body.model_dump(mode="json")
    raw["products"][0]["product_id"] = "999999.SH"
    result = service.preview(ImplementationCandidate.model_validate(raw))
    assert not result["research_ready"]
    assert (
        next(x for x in result["checks"] if x["check_id"] == "exposure")["status"]
        == "failed"
    )
    item = service.save(PackageWrite(candidate=body, idempotency_key="failure-created"))
    action = PackageAction(
        expected_revision=item["revision"],
        candidate_hash=item["candidate_hash"],
        idempotency_key="failure-retry-" + "x" * 145,
    )
    import backend.pre_investment.service as module

    original = module.evaluate

    def fail(*args, **kwargs):
        raise ValueError("controlled failure")

    monkeypatch.setattr(module, "evaluate", fail)
    with pytest.raises(ValueError, match="controlled"):
        service.validate(item["scheme_id"], action)
    assert service.repository.current(item["scheme_id"])["stage"] == "candidate_frozen"
    assert any(
        x["status"] == "failed" for x in service.view(item["scheme_id"])["attempts"]
    )
    monkeypatch.setattr(module, "evaluate", original)
    done = service.validate(item["scheme_id"], action)
    assert done["stage"] == "validation_complete"
    assert [x["stage"] for x in service.repository.history(item["scheme_id"])] == [
        "draft",
        "candidate_frozen",
        "validation_complete",
    ]
