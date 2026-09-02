from __future__ import annotations

from pathlib import Path

import pytest

from custom_indicators.errors import IndicatorDomainError
from custom_indicators.run_result_repository import EvaluationRunResultRepository


def _result(row_count: int) -> dict:
    return {
        "plan_id": "plan-1",
        "plan_revision": 3,
        "run_at": "2026-09-01T00:00:00+00:00",
        "as_of": None,
        "ranked_count": row_count,
        "excluded_count": 0,
        "normalization": {"method": "min_max_0_100"},
        "execution": {"engine_version": "test"},
        "rows": [
            {
                "rank": index + 1,
                "score": float(row_count - index),
                "status": "ranked",
                "target": {"kind": "etf", "product_id": f"{index:06d}.SH"},
                "values": [],
            }
            for index in range(row_count)
        ],
    }


def test_file_backed_result_pages_without_loading_all_rows(tmp_path: Path) -> None:
    repository = EvaluationRunResultRepository(
        tmp_path, ttl_seconds=60, row_group_size=10
    )
    result_id = repository.store(_result(25))

    page = repository.page(result_id, page=2, page_size=10)

    assert [row["rank"] for row in page["rows"]] == list(range(11, 21))
    assert page["pagination"]["page"] == 2
    assert page["pagination"]["page_size"] == 10
    assert page["pagination"]["total"] == 25
    assert page["pagination"]["page_count"] == 3
    assert page["pagination"]["has_next"] is True
    assert page["plan_revision"] == 3


def test_expired_result_is_removed_and_returns_410(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = EvaluationRunResultRepository(tmp_path, ttl_seconds=1)
    result_id = repository.store(_result(1))
    from custom_indicators import run_result_repository as repository_module

    original_time = repository_module.time.time()
    monkeypatch.setattr(repository_module.time, "time", lambda: original_time + 2)

    with pytest.raises(IndicatorDomainError) as error:
        repository.page(result_id)

    assert error.value.code == "RESULT_EXPIRED"
    assert error.value.status_code == 410
    assert not list(tmp_path.iterdir())
