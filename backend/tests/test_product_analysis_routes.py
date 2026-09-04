from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
for path in (ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.services import instrument_routes  # noqa: E402
from backend.custom_indicators import series_provider  # noqa: E402
from backend.custom_indicators.series_provider import (  # noqa: E402
    InstrumentIdentity,
    ProductSeries,
)


def _points(count: int = 64) -> list[dict[str, object]]:
    dates = pd.bdate_range("2025-01-02", periods=count)
    return [
        {
            "date": date.strftime("%Y-%m-%d"),
            "open": 1.0 + index * 0.001,
            "high": 1.02 + index * 0.001,
            "low": 0.98 + index * 0.001,
            "close": 1.0 + index * 0.001 + np.sin(index / 4.0) * 0.005,
            "volume": 1_000.0 + index,
        }
        for index, date in enumerate(dates)
    ]


def test_product_analysis_route_runs_complete_njit_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        instrument_routes,
        "_load_instruments",
        lambda _kind: pd.DataFrame([{"ts_code": "510300.SH", "name": "沪深300ETF"}]),
    )
    monkeypatch.setattr(instrument_routes, "_load_timeseries", lambda _kind, _code: _points())
    request = instrument_routes.ProductAnalysisRequest(
        statistics_period="ALL",
        simulation_horizon=21,
        simulation_path_count=200,
        bootstrap_block_length=10,
    )

    response = instrument_routes.instrument_product_analysis("510300.SH", request, "etf")

    assert response["product_id"] == "510300.SH"
    assert response["execution"]["execution_backend"] == "numba_njit_fixed_signature"
    assert response["execution"]["kernel_coverage"] == "21/21"
    assert response["execution"]["nopython"] is True
    assert response["execution"]["object_mode"] == 0
    assert response["execution"]["python_fallback"] == 0
    assert response["simulation"]["parametric"]["method"] == "parametric"
    assert response["simulation"]["blockBootstrap"]["method"] == "block_bootstrap"


def test_product_analysis_price_points_preserve_missing_physical_fields(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "value": [1.0, 1.01],
        }
    )
    product_series = ProductSeries(
        identity=InstrumentIdentity("fund", "000001.OF", "000001.OF", "测试基金"),
        frame=frame,
        fingerprint="fixture",
        data_latest_date="2025-01-03",
    )
    monkeypatch.setattr(
        series_provider,
        "load_product_series",
        lambda _kind, _product_id, _data_dir: product_series,
    )

    points = series_provider.load_price_points(
        "fund",
        "000001.OF",
        preserve_missing=True,
    )

    assert points == [
        {
            "date": "2025-01-02",
            "open": None,
            "high": None,
            "low": None,
            "close": 1.0,
            "volume": None,
        },
        {
            "date": "2025-01-03",
            "open": None,
            "high": None,
            "low": None,
            "close": 1.01,
            "volume": None,
        },
    ]


def test_instrument_timeseries_always_requests_missing_value_preservation(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_load_price_points(
        kind: str,
        product_id: str,
        data_dir: Path,
        *,
        preserve_missing: bool = False,
    ) -> list[dict[str, object]]:
        captured.update(
            kind=kind,
            product_id=product_id,
            data_dir=data_dir,
            preserve_missing=preserve_missing,
        )
        return []

    monkeypatch.setattr(instrument_routes, "load_price_points", fake_load_price_points)

    assert instrument_routes._load_timeseries("fund", "000001.OF") == []
    assert captured["preserve_missing"] is True


def test_product_regime_reference_is_resolved_from_immutable_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HISTORICAL_REGIME_DATA_DIR", str(tmp_path))
    repository = instrument_routes.RegimeRunRepository(
        tmp_path / "historical_regime_runs.json"
    )
    analytical = {
        "schema_version": "2.0",
        "definition_id": "regime-v2",
        "definition_revision": 3,
        "states": [
            {"id": "bull", "label": "牛市", "color": "#16a34a"},
            {"id": "bear", "label": "熊市", "color": "#dc2626"},
        ],
        "segments": [
            {
                "state_id": "bull",
                "start_date": "2024-01-02",
                "end_date": "2024-01-31",
            }
        ],
        "application_bindings": [],
    }
    analytical["content_hash"] = instrument_routes._historical_regime_snapshot_hash(
        analytical
    )
    run = repository.create(analytical)
    publication = {
        "id": "publication-product-research",
        "usage": "product_research",
        "run_id": run["id"],
        "definition_revision": 3,
        "run_content_hash": run["content_hash"],
    }
    repository.add_publications(run["id"], [publication])

    regime, lineage = instrument_routes._resolve_product_regime_reference(
        instrument_routes.ProductAnalysisRegime(
            run_id=run["id"],
            publication_id=publication["id"],
        )
    )

    assert regime["states"][0]["id"] == "bull"
    assert regime["segments"][0]["state_id"] == "bull"
    assert lineage == {
        "run_id": run["id"],
        "publication_id": publication["id"],
        "definition_id": "regime-v2",
        "definition_revision": 3,
        "run_content_hash": run["content_hash"],
        "usage": "product_research",
    }


def test_product_regime_reference_rejects_wrong_publication_usage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HISTORICAL_REGIME_DATA_DIR", str(tmp_path))
    repository = instrument_routes.RegimeRunRepository(
        tmp_path / "historical_regime_runs.json"
    )
    analytical = {
        "schema_version": "2.0",
        "definition_id": "regime-v2",
        "definition_revision": 1,
        "states": [{"id": "bull", "label": "牛市"}],
        "segments": [],
        "application_bindings": [],
    }
    analytical["content_hash"] = instrument_routes._historical_regime_snapshot_hash(
        analytical
    )
    run = repository.create(analytical)
    publication = {
        "id": "publication-taa",
        "usage": "taa",
        "run_id": run["id"],
        "definition_revision": 1,
        "run_content_hash": run["content_hash"],
    }
    repository.add_publications(run["id"], [publication])

    with pytest.raises(instrument_routes.HTTPException) as error:
        instrument_routes._resolve_product_regime_reference(
            instrument_routes.ProductAnalysisRegime(
                run_id=run["id"],
                publication_id=publication["id"],
            )
        )
    assert error.value.status_code == 422
    assert error.value.detail == "该历史情景版本未发布到产品研究。"
