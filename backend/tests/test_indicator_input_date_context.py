"""Date diagnostics must explain unavailable inputs without changing PIT filters."""

import pandas as pd
import pytest

from custom_indicators.series_provider import (
    input_date_context,
    load_adjusted_product_series,
    load_product_chart_series,
    load_product_series,
    load_product_variable_series,
    load_product_variable_series_batch,
)


@pytest.mark.parametrize("date_format", ["iso", "compact", "unpadded"])
@pytest.mark.parametrize(
    "found_date,cutoff,announcement,expected_code,expected_rows",
    [
        ("20260102", "2026-01-01", "2026-01-02", "PRODUCT_NOT_ESTABLISHED_AS_OF", 0),
        ("20200101", "2026-01-01", "2026-01-02", "NO_DATA_BEFORE_CUTOFF", 0),
        ("20200101", "2026-01-02", "2026-01-05", "NO_DISCLOSURES_AS_OF", 0),
        ("20200101", "2026-01-02", None, "NO_DISCLOSURES_AS_OF", 0),
        # Predecessor history remains usable even before the current inception.
        ("20260105", "2026-01-02", "2026-01-02", None, 1),
    ],
)
def test_single_and_batch_preserve_date_filters_and_explain_missing_inputs(
    tmp_path, monkeypatch, found_date, cutoff, announcement, expected_code, expected_rows, date_format,
):
    pd.DataFrame([{
        "ts_code": "000001.OF", "code": "000001", "name": "Fixture fund",
        "found_date": found_date, "list_date": "20260201",
    }]).to_parquet(tmp_path / "fund_info_df.parquet", index=False)
    pd.DataFrame([{
        "ts_code": "000001.OF", "date": pd.Timestamp("2026-01-02"),
        "ann_date": pd.Timestamp(announcement) if announcement else pd.NaT,
        "adj_nav": 1.0,
    }]).to_parquet(tmp_path / "fund_nav_df.parquet", index=False)

    requested = (
        cutoff if date_format == "iso" else cutoff.replace("-", "")
        if date_format == "compact" else "-".join(str(int(part)) for part in cutoff.split("-"))
    )
    single = load_product_variable_series(
        "fund", "000001.OF", ["adjusted_nav"], tmp_path, requested,
    )
    batch = load_product_variable_series_batch(
        "fund", ["000001.OF"], ["adjusted_nav"], tmp_path, requested,
    )["000001.OF"]
    chart = load_product_chart_series(
        "fund", "000001.OF", ["adjusted_nav"], "adjusted_nav", tmp_path, requested,
    )
    assert single is not None and batch is not None
    for result in (single, batch, chart):
        assert len(result.frame) == expected_rows
        diagnostic = result.unavailable_variables.get("adjusted_nav")
        assert (diagnostic["code"] if diagnostic else None) == expected_code

    # Context is assembled from existing lineage; displaying it must not read data.
    def unexpected_read(*args, **kwargs):
        pytest.fail("date context unexpectedly read parquet")

    monkeypatch.setattr(pd, "read_parquet", unexpected_read)
    context = input_date_context(single, requested)
    assert context == input_date_context(batch, requested) == input_date_context(chart, requested)
    assert context["found_date"] == pd.Timestamp(found_date).strftime("%Y-%m-%d")
    assert context["list_date"] == "2026-02-01"
    assert context["as_of"] == cutoff
    source = context["sources"][0]
    assert source["first_date"] == source["latest_date"] == "2026-01-02"
    assert source["rows_before_as_of"] == 1
    assert source["rows_after_date_filter"] == int(cutoff >= "2026-01-02")
    assert source["rows_after_as_of"] == expected_rows
    assert source["uses_disclosure_date"] is True


@pytest.mark.parametrize("kind,loader,source_name", [
    ("etf", load_product_series, "etf_daily_candle_df.parquet"),
    ("fund", load_product_series, "fund_nav_df.parquet"),
    ("etf", load_adjusted_product_series, "etf_daily_df.parquet"),
])
@pytest.mark.parametrize("as_of,expected_count", [
    (None, 3), ("2026-01-01", 0), ("2026-01-03", 2), ("20260103", 2),
])
def test_scalar_loader_preserves_identity_and_normalized_source_coverage(
    tmp_path, monkeypatch, kind, loader, source_name, as_of, expected_count,
):
    code = "510300.SH" if kind == "etf" else "000001.OF"
    pd.DataFrame([{
        "ts_code": code, "name": "Fixture", "found_date": "20200101", "list_date": "20200201",
    }]).to_parquet(tmp_path / f"{kind}_info_df.parquet", index=False)
    for name,field in [("etf_daily_candle_df.parquet", "close"), ("etf_daily_df.parquet", "adj_nav"), ("fund_nav_df.parquet", "adj_nav")]:
        pd.DataFrame({
            "ts_code": [code] * 4, "date": pd.date_range("2026-01-02", periods=4),
            field: [10., 11., 12., float("nan")] if field == "close" else [1., 1.1, 1.2, float("nan")],
        }).to_parquet(tmp_path / name, index=False)
    result = loader(kind, code, tmp_path)
    assert result is not None
    original = result.frame.copy()

    def unexpected_read(*args, **kwargs):
        pytest.fail("scalar date context unexpectedly read parquet")

    monkeypatch.setattr(pd, "read_parquet", unexpected_read)
    context = input_date_context(result, as_of)
    assert context["found_date"] == "2020-01-01"
    assert context["list_date"] == "2020-02-01"
    assert context["as_of"] == (pd.Timestamp(as_of).strftime("%Y-%m-%d") if as_of else None)
    assert result.lineage[0]["dataset"] == source_name
    assert context["sources"] == [{
        "label": "ETF 行情" if source_name == "etf_daily_candle_df.parquet" else "ETF 净值" if kind == "etf" else "基金净值",
        "first_date": "2026-01-02", "latest_date": "2026-01-04",
        "rows_before_as_of": 3, "rows_after_date_filter": expected_count,
        "rows_after_as_of": expected_count, "uses_disclosure_date": False,
    }]
    pd.testing.assert_frame_equal(result.frame, original)
