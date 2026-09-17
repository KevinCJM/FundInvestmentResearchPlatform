"""Synthetic only: real graph execution, transient previews, immutable confirmation."""

import copy
from datetime import date, timedelta
import numpy as np
import pytest
from historical_regimes.v2_service import RegimeGraphV2Service
from historical_regimes.v2_contracts import parse_definition_v2, validate_definition_v2
from historical_regimes.v2_templates import get_template_v2, list_templates_v2
from historical_regimes.reliability.diagnostic_kernels import (
    quality_kernel,
    compare_kernel,
)
from historical_regimes.reliability import kernels
from test_historical_regime_v2 import _definition

IDS = (
    "historical_hmm_risk_v1",
    "historical_gmm_volatility_v1",
    "historical_window_mean_change_v1",
    "historical_trend_ensemble_v1",
)


@pytest.mark.parametrize("identity", (
    "historical_gmm_volatility_v1",
    "historical_window_mean_change_v1",
    "historical_trend_ensemble_v1",
    "size-rotation-v2",
    "growth-value-rotation-v2",
    "peak-trough-daily-v2",
    "peak-trough-daily-legacy-v1",
))
def test_rejected_market_state_templates_remain_exactly_loadable_but_hidden_from_authoring(identity):
    item = get_template_v2(identity)
    assert item is not None
    assert item.get("authoring_hidden") is True
    assert any(template["id"] == identity for template in list_templates_v2())


@pytest.fixture
def graph(tmp_path):
    return RegimeGraphV2Service(tmp_path, tmp_path)


def rows(n=240):
    return [
        {
            "observation_date": (date(2020, 1, 1) + timedelta(days=i)).isoformat(),
            "available_at": (date(2020, 1, 1) + timedelta(days=i)).isoformat(),
            "value": 100 + i * 0.05 + (1 + (i // 60) % 3) * np.sin(i / 4),
        }
        for i in range(n)
    ]


@pytest.mark.parametrize("identity", IDS)
def test_new_templates_execute_editable_and_default_historical(graph, identity):
    item = graph.instantiate_template(identity)
    payload = item["definition"]
    assert payload["study"]["purpose"] == "historical_reference"
    assert get_template_v2(identity)["default_mode"] == "retrospective"
    assert get_template_v2(identity)["supported_modes"] == ["retrospective"]
    nodes = payload["graph"]["nodes"]
    nodes[0].update(
        type="source.inline", type_id="source.inline", parameters={"rows": rows()}
    )
    parsed = parse_definition_v2(payload)
    validate_definition_v2(parsed)
    plan = graph.prepare(payload, persist_manifest=False)
    execution = graph._execute_graph(None, parsed, "retrospective", None, plan=plan)
    assert len(execution["series"]) == 240
    assert any(p["state_id"] != "unclassified" for p in execution["series"])
    assert set(execution["node_outputs"]) == {n["id"] for n in nodes}
    assert execution["result"]["diagnostics"]["python_fallback"] == 0
    if "risk" in identity or "volatility" in identity:
        assert [s.id for s in parsed.states] == ["high_risk", "normal_risk", "low_risk"]
    with pytest.raises(Exception, match="研究用途"):
        graph._execute_graph(None, parsed, "realtime", None, plan=plan)


def test_csi300_volatility_realtime_companion_is_causal_fixed_rule(graph):
    identity = "csi300-volatility-hmm-reference-recognition-v1"
    item = graph.instantiate_template(identity)
    payload = item["definition"]
    assert payload["study"]["purpose"] == "realtime_recognition"
    assert payload["study"]["family"] == "risk"
    assert payload["usage_intent"] == "taa"
    assert get_template_v2(identity)["default_mode"] == "realtime"
    assert get_template_v2(identity)["supported_modes"] == ["realtime"]
    nodes = payload["graph"]["nodes"]
    nodes[0].update(type="source.inline", type_id="source.inline", parameters={"rows": rows()})
    payload["evaluation_targets"] = []
    by_id = {node["id"]: node for node in nodes}
    assert by_id["volatility"]["parameters"] == {"window": 20}
    assert by_id["high_condition"]["parameters"] == {"operator": "gt", "threshold": pytest.approx(0.014)}
    assert by_id["low_condition"]["parameters"] == {"operator": "lt", "threshold": pytest.approx(0.0094)}
    assert by_id["classifier"]["type"] == "state.select"
    parsed = parse_definition_v2(payload)
    validate_definition_v2(parsed)
    plan = graph.prepare(payload, persist_manifest=False)
    execution = graph._execute_graph(None, parsed, "realtime", None, plan=plan)
    assert len(execution["series"]) == 240
    assert any(point["state_id"] != "unclassified" for point in execution["series"])
    assert set(point["state_id"] for point in execution["series"]) <= {
        "unclassified", "high_risk", "normal_risk", "low_risk"
    }
    assert execution["result"]["temporal_capability"]["realtime_supported"] is True
    assert execution["result"]["diagnostics"]["python_fallback"] == 0


def test_quality_preview_confirm_actual_routes_and_no_artifacts(graph, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from services import historical_regime_routes as routes

    payload = _definition(rows())
    payload["study"] = {"purpose": "historical_reference", "family": "market_trend"}
    saved = graph.create_definition(payload)
    request = {"definition_id": saved["id"], "revision": 1, "policy": {}}
    before = {
        p.relative_to(graph.workspace_data_dir)
        for p in graph.workspace_data_dir.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(routes, "regime_graph_v2_service", graph)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        prefix = "/api/historical-regimes/reference-quality"
        response = client.post(prefix + "/preview", json=request)
        assert response.status_code == 200, response.text
        preview = response.json()
        assert preview["report"]["sample"]["input"] == 240
        assert (
            preview["report"]["price_returns"]["reason"]
            == "no_verified_price_semantics"
        )
        assert preview["report"]["stability"]["status"] == "completed"
        assert preview["report"]["stability"]["seed_status"] == "not_applicable"
        assert preview["report"]["conditional_estimation"]["status"] in {"ready", "partially_ready", "insufficient_evidence"}
        assert preview["report"]["conditional_estimation"]["minimum_state_episodes"] == 3
        assert all("independent_complete_episodes" in row and "conditional_estimation_status" in row
                   for row in preview["report"]["segments"]["per_state"])
        after = {
            p.relative_to(graph.workspace_data_dir)
            for p in graph.workspace_data_dir.rglob("*")
            if p.is_file()
        }
        assert after == before
        body = {"request": preview["request"], "preview_hash": preview["preview_hash"]}
        confirmed = client.post(prefix + "/confirm", json=body)
        assert confirmed.status_code == 200, confirmed.text
        value = confirmed.json()
        assert value["immutable"] and value["id"].startswith("reference-quality-")
        assert client.get(prefix + "/reports/" + value["id"]).json() == value
        assert client.post(prefix + "/confirm", json=body).json() == value
        assert len(client.get(prefix + "/catalog").json()["items"]) == 1
        assert not graph.runs.list()
        assert (
            client.post(
                prefix + "/preview", json={**request, "mode": "realtime"}
            ).status_code
            == 422
        )
        assert (
            client.post(prefix + "/confirm", json={**body, "report": {}}).status_code
            == 422
        )
        changed = copy.deepcopy(body)
        changed["request"]["policy"]["include_price_returns"] = False
        assert client.post(prefix + "/confirm", json=changed).status_code == 409


def test_quality_numerical_oracle_readonly_stride_and_unknowns():
    kernels.warm()
    labels = np.array([-1, -9, 0, -9, 0, -9, 1, -9, 1, -9, 1, -9, -1, -9], np.int64)[
        ::2
    ]
    prices = np.array(
        [
            100.0,
            0.0,
            110.0,
            0.0,
            121.0,
            0.0,
            100.0,
            0.0,
            90.0,
            0.0,
            81.0,
            0.0,
            80.0,
            0.0,
        ]
    )[::2]
    labels.flags.writeable = prices.flags.writeable = False
    counts, stats = quality_kernel(labels, prices, 2)
    assert counts.tolist() == [5, 2, 1, 1, 2, 1]
    np.testing.assert_allclose(stats[0, :8], [2, 1, 2, 2, 2, 2, 1, 0.1])
    np.testing.assert_allclose(stats[1, :8], [3, 1, 3, 3, 3, 3, 1, -0.19])
    assert np.shares_memory(prices, prices.base)
    unknown = np.full(5, -1, np.int64)
    c, v = quality_kernel(unknown, np.ones(5), 2)
    assert c[:4].tolist() == [0, 5, 5, 5]
    assert np.isnan(v[:, 2:6]).all()
    assert np.isnan(compare_kernel(unknown, unknown, 2)[1])
    with pytest.raises(ValueError):
        quality_kernel(labels, prices[:2], 2)
    assert kernels.audit()["request_time_compilation"] == 0


@pytest.mark.parametrize("identity", IDS)
def test_template_formula_roundtrip_and_step_preview(graph, identity):
    from historical_regimes.authoring import graph_source, source_graph

    payload = graph.instantiate_template(identity)["definition"]
    payload["graph"]["nodes"][0].update(
        type="source.inline", type_id="source.inline", parameters={"rows": rows(120)}
    )
    parsed = parse_definition_v2(payload)
    source = graph_source(parsed)
    restored = {**payload, "graph": source_graph(source, payload)}
    restored_parsed = parse_definition_v2(restored)
    assert restored_parsed.graph.outputs == parsed.graph.outputs
    assert [
        (n.id, n.type, n.parameters, n.inputs) for n in restored_parsed.graph.nodes
    ] == [(n.id, n.type, n.parameters, n.inputs) for n in parsed.graph.nodes]
    plan = graph.prepare(restored, persist_manifest=False)
    out = graph._execute_graph(None, restored_parsed, "retrospective", None, plan=plan)
    assert len(out["series"]) == 120
    from historical_regimes.node_preview import node_preview_definition
    from historical_regimes.v2_registry import NODE_REGISTRY

    for node in restored_parsed.graph.nodes:
        assert node.id in out["node_outputs"]
        for metadata in NODE_REGISTRY[node.type]["outputs"]:
            target = {"node_id": node.id, "port": metadata["name"]}
            projected = node_preview_definition(restored, target)
            preview_plan = graph.prepare(
                restored, preview_target=target, persist_manifest=False
            )
            preview = graph._execute_graph(
                None, projected, "retrospective", None, plan=preview_plan
            )
            assert preview["result"]["result_kind"] == "node_preview"
            np.testing.assert_array_equal(
                preview["node_outputs"][node.id][metadata["name"]].values,
                out["node_outputs"][node.id][metadata["name"]].values,
            )


def test_quality_saved_revision_source_integrity_and_plan_cleanup(graph):
    payload = _definition(rows())
    saved = graph.create_definition(payload)
    request = {"definition_id": saved["id"], "revision": 1, "policy": {}}
    existing = set(graph._plans)
    preview = graph.reference_quality.preview(request)
    assert set(graph._plans) == existing
    # Simulate corrupted immutable definition bytes without advancing revision.
    with graph.definitions.store.locked():
        stored = graph.definitions.store.read_unlocked()
        stored["items"][0]["current"]["graph"]["nodes"][0]["parameters"]["rows"][20][
            "value"
        ] = 10000
        graph.definitions.store.write_unlocked(stored)
    with pytest.raises(Exception):
        graph.reference_quality.confirm(
            {"request": preview["request"], "preview_hash": preview["preview_hash"]}
        )
    assert not graph.reference_quality.root.exists()


def test_quality_verified_price_source_uses_display_source(graph):
    import pandas as pd

    source_rows = rows()
    frame = pd.DataFrame(
        [
            {
                "ts_code": "000300.SH",
                "trade_date": p["observation_date"].replace("-", ""),
                "close": p["value"],
            }
            for p in source_rows
        ]
    )
    frame.to_parquet(graph.market_data_dir / "index_daily_df.parquet")
    payload = _definition(source_rows)
    payload["graph"]["nodes"][0] = {
        "id": "source",
        "type": "source.index",
        "parameters": {
            "ts_code": "000300.SH",
            "source_api": "index_daily",
            "field": "close",
        },
    }
    saved = graph.create_definition(payload)
    request = {
        "definition_id": saved["id"],
        "revision": 1,
        "policy": {"stability": {"enabled": False}},
    }
    report = graph.reference_quality.preview(request)["report"]
    assert report["price_returns"]["status"] == "available"
    assert any(r["price_return_samples"] > 0 for r in report["segments"]["per_state"])
    assert all(
        r["mean_price_return"] is not None
        for r in report["segments"]["per_state"]
        if r["price_return_samples"]
    )


def test_low_quality_legacy_templates_are_hidden_but_drawdown_research_is_authorable():
    for identity in (
        "historical_gmm_volatility_v1",
        "historical_window_mean_change_v1",
        "historical_trend_ensemble_v1",
        "size-rotation-v2",
        "growth-value-rotation-v2",
    ):
        assert get_template_v2(identity)["authoring_hidden"] is True
    for identity in (
        "historical_hmm_risk_v1",
        "csi300-drawdown-cycle-reference-v1",
        "csi300-drawdown-cycle-realtime-v1",
    ):
        assert not get_template_v2(identity).get("authoring_hidden", False)


def test_drawdown_cycle_kernels_preserve_trough_and_use_only_past_prices():
    from historical_regimes.segment_numba import (
        drawdown_cycle_reference_kernel,
        drawdown_cycle_realtime_kernel,
    )

    # Two complete pivot segments share index 2: peak->trough is -15%, then
    # trough->peak. The shared trough remains Stress; Recovery starts after it.
    phases = np.array([1, 1, 0, 0, 0, -1], np.int64)
    changes = np.array([-.15, -.15, .12, .12, .12, np.nan], np.float64)
    starts = np.array([0, 0, 2, 2, 2, -1], np.int64)
    ends = np.array([2, 2, 5, 5, 5, -1], np.int64)
    reference = drawdown_cycle_reference_kernel(phases, changes, starts, ends, .12)
    assert reference.tolist() == [0, 2, 2, 1, 1, 1]

    prices = np.array([100., 95., 88., 80., 84., 88., 94., 98.], np.float64)
    realtime = drawdown_cycle_realtime_kernel(prices, 3, .10, .05, .08)
    assert realtime.tolist() == [0, 0, 2, 2, 1, 0, 0, 0]
    # Prefix causality: future observations cannot rewrite an earlier state.
    short = drawdown_cycle_realtime_kernel(prices[:5], 3, .10, .05, .08)
    np.testing.assert_array_equal(short, realtime[:5])
    assert len(drawdown_cycle_reference_kernel.nopython_signatures) == 1
    assert len(drawdown_cycle_realtime_kernel.nopython_signatures) == 1
    assert not drawdown_cycle_reference_kernel._can_compile
    assert not drawdown_cycle_realtime_kernel._can_compile


def test_transient_plan_overlapping_consumers_and_explicit_prepare_promotion(graph):
    from historical_regimes.reliability.diagnostics import prepared_plan

    payload = _definition(rows(80))
    definition = parse_definition_v2(payload)
    with prepared_plan(graph, definition) as first:
        token = first["compile_token"]
        with prepared_plan(graph, definition) as second:
            assert second["compile_token"] == token
            assert graph._plans[token]["diagnostic_leases"] == 2
        assert token in graph._plans
    assert token not in graph._plans
    with prepared_plan(graph, definition) as transient:
        explicit = graph.prepare(payload)
        assert explicit["compile_token"] == transient["compile_token"]
    assert graph._validate_plan(definition, explicit["compile_token"])


def test_recovery_peak_keeps_preceding_label_when_next_decline_completes():
    from historical_regimes.segment_numba import drawdown_cycle_reference_kernel
    phases = np.array([1, 1, 0, 0, 1, 1, -1], np.int64)
    changes = np.array([-.2, -.2, .15, .15, -.18, -.18, np.nan])
    starts = np.array([0, 0, 2, 2, 4, 4, -1], np.int64)
    ends = np.array([2, 2, 4, 4, 6, 6, -1], np.int64)
    arrays = (phases, changes, starts, ends)
    before = [a.copy() for a in arrays]
    result = drawdown_cycle_reference_kernel(*arrays, .12)
    assert result.tolist() == [0, 2, 2, 1, 1, 2, 2]
    prefix = drawdown_cycle_reference_kernel(*(a[:5] for a in arrays), .12)
    np.testing.assert_array_equal(prefix, result[:5])
    for a, b in zip(arrays, before): np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('missing', [np.nan, np.inf, 0., -1.])
def test_gap_reentry_is_unknown_until_an_entry_or_exit_boundary(missing):
    from historical_regimes.segment_numba import drawdown_cycle_realtime_kernel
    values = np.array([100., 80., 90., missing, 90., 79.5, 70., 75., 80.])
    actual = drawdown_cycle_realtime_kernel(values, 2, .12, .05, .08)
    assert actual.tolist() == [0, 2, 1, -1, -1, -1, -1, 0, 0]
    stressed = values.copy(); stressed[6] = 65.
    assert drawdown_cycle_realtime_kernel(stressed, 2, .12, .05, .08)[6] == 2
    for n in range(1, len(values) + 1):
        np.testing.assert_array_equal(drawdown_cycle_realtime_kernel(values[:n], 2, .12, .05, .08), actual[:n])


@pytest.mark.parametrize('input_name,source_type,output_port,code', [
    ('phase', 'segment.phase_direction', 'phase', 'SEGMENT_PHASE_MISMATCH'),
    ('change', 'segment.change', 'value', 'SEGMENT_CHANGE_MISMATCH'),
])
def test_drawdown_reference_rejects_mixed_segment_chains(input_name, source_type, output_port, code):
    from custom_indicators.errors import ValidationError
    payload = copy.deepcopy(get_template_v2('csi300-drawdown-cycle-reference-v1')['definition'])
    nodes = payload['graph']['nodes']
    boundaries = copy.deepcopy(next(n for n in nodes if n['id'] == 'segments'))
    boundaries['id'] = 'other_segments'; nodes.append(boundaries)
    source = copy.deepcopy(next(n for n in nodes if n['type'] == source_type))
    source['id'] = 'other_' + input_name
    source['inputs']['start'] = {'node_id': 'other_segments', 'port': 'start'}
    source['inputs']['end'] = {'node_id': 'other_segments', 'port': 'end'}
    nodes.append(source)
    next(n for n in nodes if n['id'] == 'cycle')['inputs'][input_name] = {'node_id': source['id'], 'port': output_port}
    with pytest.raises(ValidationError) as caught: validate_definition_v2(parse_definition_v2(payload))
    assert code in str(caught.value.diagnostics)


def test_failed_manifest_promotion_remains_transient_and_is_reclaimed(graph, monkeypatch):
    from historical_regimes.reliability.diagnostics import prepared_plan
    from custom_indicators.errors import ValidationError
    payload = _definition(rows(80)); definition = parse_definition_v2(payload)
    with prepared_plan(graph, definition) as transient:
        token = transient['compile_token']
        def fail(*args, **kwargs): raise OSError('disk full')
        with monkeypatch.context() as patch:
            patch.setattr(graph, '_persist_plan_manifest', fail)
            with pytest.raises(OSError, match='disk full'): graph.prepare(payload)
        assert graph._plans[token]['persistent'] is False
        assert graph._plans[token]['diagnostic_leases'] == 1
    assert token not in graph._plans
    with pytest.raises(ValidationError): graph._validate_plan(definition, token)
    prepared = graph.prepare(payload)
    assert graph._plans[prepared['compile_token']]['persistent'] is True


def test_realtime_study_templates_advertise_only_executable_mode():
    for item in list_templates_v2():
        purpose = (item['definition'].get('study') or {}).get('purpose')
        if purpose == 'realtime_recognition':
            assert item['supported_modes'] == ['realtime']
            assert item['default_mode'] == 'realtime'


@pytest.mark.parametrize('study', [None, {'purpose': 'realtime_recognition', 'family': 'market_trend'}])
def test_v2_unpublished_runs_are_rejected_before_taa_calibration(graph, study):
    from custom_indicators.errors import ValidationError
    import pandas as pd
    from research_series.service import write_upload_artifact
    payload = _definition(rows(80))
    frame = pd.DataFrame(rows(80))
    frame['vintage'] = None
    frame['revision'] = 1
    upload = write_upload_artifact(graph.market_data_dir, frame)
    payload['graph']['nodes'][0] = {
        'id': 'source', 'type': 'source.upload', 'parameters': {
            'artifact_id': upload['artifact_id'], 'checksum': upload['checksum'],
            'format': 'parquet', 'frequency': 'daily', 'availability_mode': 'point_in_time',
        },
    }
    if study: payload['study'] = study
    saved = graph.create_definition(payload)
    prepared = graph.prepare(saved)
    run = graph.run_saved({'schema_version': '2.0', 'id': saved['id'], 'revision': saved['revision']}, 'realtime', None, prepared['compile_token'])
    with pytest.raises(ValidationError) as caught: graph.resolve_taa_run(run['id'])
    assert caught.value.code == 'TAA_RUN_NOT_PUBLISHED'


def test_drawdown_kernels_accept_readonly_strided_views_without_input_copies():
    from historical_regimes.segment_numba import drawdown_cycle_reference_kernel as historical, drawdown_cycle_realtime_kernel as realtime
    owners = [np.repeat(a, 2) for a in (
        np.array([1, 1, 0, 0, 1, 1, -1], np.int64),
        np.array([-.2, -.2, .15, .15, -.18, -.18, np.nan]),
        np.array([0, 0, 2, 2, 4, 4, -1], np.int64),
        np.array([2, 2, 4, 4, 6, 6, -1], np.int64),
    )]
    views = [owner[::2] for owner in owners]
    for view in views: view.flags.writeable = False
    before = [owner.copy() for owner in owners]
    assert historical(*views, .12).tolist() == [0, 2, 2, 1, 1, 2, 2]
    for view, owner, original in zip(views, owners, before):
        assert np.shares_memory(view, owner)
        np.testing.assert_array_equal(owner, original)
    assert historical(*(view[:0] for view in views), .12).size == 0
    prices = np.repeat([100., 80., 90., np.nan, 90., 79.5], 2)
    view = prices[::2]; view.flags.writeable = False
    assert np.shares_memory(view, prices)
    assert realtime(view, 2, .12, .05, .08).tolist() == [0, 2, 1, -1, -1, -1]
    assert realtime(view[:0], 2, .12, .05, .08).size == 0
    for kernel in (historical, realtime):
        assert len(kernel.nopython_signatures) == 1
        assert not kernel._can_compile
