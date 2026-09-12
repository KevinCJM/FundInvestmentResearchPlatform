"""Date diagnostics must explain unavailable inputs without changing PIT filters."""

import pandas as pd
import pytest

from custom_indicators.series_provider import (
    input_date_context,
    load_product_variable_series,
    load_product_variable_series_batch,
)


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
    tmp_path, monkeypatch, found_date, cutoff, announcement, expected_code, expected_rows,
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

    single = load_product_variable_series(
        "fund", "000001.OF", ["adjusted_nav"], tmp_path, cutoff,
    )
    batch = load_product_variable_series_batch(
        "fund", ["000001.OF"], ["adjusted_nav"], tmp_path, cutoff,
    )["000001.OF"]
    assert single is not None and batch is not None
    for result in (single, batch):
        assert len(result.frame) == expected_rows
        diagnostic = result.unavailable_variables.get("adjusted_nav")
        assert (diagnostic["code"] if diagnostic else None) == expected_code

    # Context is assembled from existing lineage; displaying it must not read data.
    def unexpected_read(*args, **kwargs):
        pytest.fail("date context unexpectedly read parquet")

    monkeypatch.setattr(pd, "read_parquet", unexpected_read)
    context = input_date_context(single, cutoff)
    assert context == input_date_context(batch, cutoff)
    assert context["found_date"] == pd.Timestamp(found_date).strftime("%Y-%m-%d")
    assert context["list_date"] == "2026-02-01"
    assert context["as_of"] == cutoff
    source = context["sources"][0]
    assert source["first_date"] == source["latest_date"] == "2026-01-02"
    assert source["rows_before_as_of"] == 1
    assert source["rows_after_date_filter"] == int(cutoff >= "2026-01-02")
    assert source["rows_after_as_of"] == expected_rows
    assert source["uses_disclosure_date"] is True
