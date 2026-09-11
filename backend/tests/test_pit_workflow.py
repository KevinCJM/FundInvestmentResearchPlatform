"""Point-in-time discipline: declarations, measurement, releases and the cut.

Every fixture is written into `tmp_path`; nothing here touches the network or the
real data directory.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend import fit
from backend.pit import audit as pit_audit
from backend.pit import catalog, context
from backend.pit.release import DataReleaseError, DataReleaseRepository
from backend.pit.settings import PitSettingsRepository


@pytest.fixture(autouse=True)
def _isolate_audit_cache() -> None:
    pit_audit.clear_cache()


def _nav_fixture(
    tmp_path: Path,
    *,
    announcement_lag: int = 1,
    drop_announcements: int = 0,
    include_ann_date: bool = True,
) -> Path:
    """Two products over 12 business days, announced `announcement_lag` days late."""

    dates = pd.bdate_range("2024-01-01", periods=12)
    rows = []
    for code, base in (("510300.SH", 3.0), ("511010.SH", 1.0)):
        for index, value_date in enumerate(dates):
            rows.append(
                {
                    "ts_code": code,
                    "name": f"fixture-{code}",
                    "nav_date": value_date,
                    "date": value_date,
                    "ann_date": value_date + pd.Timedelta(days=announcement_lag),
                    "adj_nav": base * (1.0 + 0.001 * index),
                }
            )
    frame = pd.DataFrame(rows)
    if drop_announcements:
        frame.loc[frame.index[:drop_announcements], "ann_date"] = pd.NaT
    if not include_ann_date:
        frame = frame.drop(columns=["ann_date"])
    frame.to_parquet(tmp_path / "etf_daily_df.parquet", index=False)
    return tmp_path / "etf_daily_df.parquet"


# --------------------------------------------------------------------------- #
# catalog: a claim alone never earns an A
# --------------------------------------------------------------------------- #


def test_grade_requires_the_declared_column_to_actually_be_measured() -> None:
    declaration = catalog.DATASETS_BY_ID["etf_nav"]
    assert catalog.grade(declaration, 1.0) == catalog.GRADE_STRICT
    assert catalog.grade(declaration, 0.9995) == catalog.GRADE_STRICT
    # Just under the floor: the column exists but cannot carry a strict claim.
    assert catalog.grade(declaration, 0.99) == catalog.GRADE_APPROXIMATE
    # Declared but unverifiable (file or column absent) must not stay an A.
    assert catalog.grade(declaration, None) == catalog.GRADE_APPROXIMATE


def test_revisable_dimension_tables_can_never_reach_a() -> None:
    revisable = catalog.DATASETS_BY_ID["etf_info"]
    assert revisable.revisable is True
    assert catalog.grade(revisable, None) == catalog.GRADE_NONE
    # Even a perfectly covered availability column cannot fix in-place rewrites:
    # yesterday's value is simply gone.
    with_column = catalog.DatasetPitDeclaration(
        **{**revisable.__dict__, "availability_field": "updated_at"}
    )
    assert catalog.grade(with_column, 1.0) == catalog.GRADE_NONE


def test_strict_mode_refuses_grade_c_and_research_mode_does_not() -> None:
    assert catalog.is_usable_under(catalog.RUN_MODE_STRICT, catalog.GRADE_STRICT) is True
    assert catalog.is_usable_under(catalog.RUN_MODE_STRICT, catalog.GRADE_APPROXIMATE) is True
    assert catalog.is_usable_under(catalog.RUN_MODE_STRICT, catalog.GRADE_NONE) is False
    assert catalog.is_usable_under(catalog.RUN_MODE_RESEARCH, catalog.GRADE_NONE) is True


def test_every_declaration_is_internally_consistent() -> None:
    seen: set[str] = set()
    for declaration in catalog.DATASETS:
        assert declaration.dataset_id not in seen, declaration.dataset_id
        seen.add(declaration.dataset_id)
        assert declaration.file.endswith(".parquet")
        assert declaration.label and declaration.note
        assert declaration.declared_lag_days >= 0
        assert catalog.grade(declaration, 1.0 if declaration.availability_field else None) in catalog.GRADE_LABELS


# --------------------------------------------------------------------------- #
# audit: measure the file, do not trust the declaration
# --------------------------------------------------------------------------- #


def test_audit_measures_the_announcement_lag_it_finds(tmp_path: Path) -> None:
    _nav_fixture(tmp_path, announcement_lag=3)
    result = pit_audit.audit_dataset(tmp_path, catalog.DATASETS_BY_ID["etf_nav"])
    assert result["present"] is True
    assert result["rows"] == 24
    assert result["availability_coverage"] == pytest.approx(1.0)
    assert result["lag"]["p50"] == pytest.approx(3.0)
    assert result["lag"]["p95"] == pytest.approx(3.0)
    assert result["lag"]["negative_rows"] == 0
    assert result["grade"] == catalog.GRADE_STRICT
    # 24 rows all land in the 3-5 day bucket and nowhere else.
    by_bucket = {item["bucket"]: item["rows"] for item in result["lag"]["histogram"]}
    assert by_bucket["3-5"] == 24
    assert by_bucket["1"] == 0


def test_audit_drops_to_b_when_the_declared_column_is_incomplete(tmp_path: Path) -> None:
    # 6 of 24 rows unannounced is far below the 99.5% floor.
    _nav_fixture(tmp_path, drop_announcements=6)
    result = pit_audit.audit_dataset(tmp_path, catalog.DATASETS_BY_ID["etf_nav"])
    assert result["availability_coverage"] == pytest.approx(18 / 24)
    assert result["grade"] == catalog.GRADE_APPROXIMATE


def test_audit_reports_a_missing_file_without_blowing_up(tmp_path: Path) -> None:
    result = pit_audit.audit_dataset(tmp_path, catalog.DATASETS_BY_ID["etf_nav"])
    assert result["present"] is False
    assert result["rows"] == 0
    assert result["available_through"] if "available_through" in result else True
    assert result["grade"] == catalog.GRADE_APPROXIMATE


def test_audit_flags_announcements_that_predate_the_event(tmp_path: Path) -> None:
    _nav_fixture(tmp_path, announcement_lag=-2)
    result = pit_audit.audit_dataset(tmp_path, catalog.DATASETS_BY_ID["etf_nav"])
    assert result["lag"]["negative_rows"] == 24


def test_audit_summary_floor_ignores_grade_c_dimension_tables(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    # A grade-C table whose publication date is years stale must not drag the
    # platform-wide "data available through" backwards.
    pd.DataFrame(
        {"ts_code": ["510300.SH"], "indx_name": ["x"], "pub_date": [pd.Timestamp("2019-01-01")]}
    ).to_parquet(tmp_path / "etf_index.parquet", index=False)
    payload = pit_audit.audit_all(tmp_path)
    assert payload["summary"]["available_through_basis"] == "grade_a_b"
    assert payload["summary"]["available_through"] == "2024-01-17"
    assert payload["summary"]["grade_c"] >= 1


def test_audit_cache_follows_the_file_not_the_clock(tmp_path: Path) -> None:
    _nav_fixture(tmp_path, announcement_lag=1)
    first = pit_audit.audit_dataset(tmp_path, catalog.DATASETS_BY_ID["etf_nav"])
    assert first["lag"]["p50"] == pytest.approx(1.0)
    _nav_fixture(tmp_path, announcement_lag=7)
    second = pit_audit.audit_dataset(tmp_path, catalog.DATASETS_BY_ID["etf_nav"])
    assert second["lag"]["p50"] == pytest.approx(7.0), "rewriting the file must invalidate the memo"


# --------------------------------------------------------------------------- #
# context: strict mode fails closed
# --------------------------------------------------------------------------- #


def test_strict_mode_without_a_research_day_is_rejected() -> None:
    with pytest.raises(context.PitContextError, match="必须指定研究日"):
        context.build_context(None, catalog.RUN_MODE_STRICT)
    ok = context.build_context("2024-01-10", catalog.RUN_MODE_STRICT)
    assert ok.as_of == "2024-01-10"
    assert ok.strict is True


def test_build_context_rejects_junk_dates_and_modes() -> None:
    with pytest.raises(context.PitContextError, match="研究日格式错误"):
        context.build_context("not-a-date")
    with pytest.raises(context.PitContextError, match="不支持的运行模式"):
        context.build_context("2024-01-10", "YOLO")
    # No as_of is legal in research mode and says so rather than pretending.
    loose = context.build_context(None, None)
    assert loose.as_of is None and loose.run_mode == catalog.RUN_MODE_RESEARCH


def test_resolve_blocks_grade_c_only_in_strict_mode(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    pd.DataFrame({"ts_code": ["510300.SH"], "name": ["x"], "fund_type": ["ETF"]}).to_parquet(
        tmp_path / "etf_info_df.parquet", index=False
    )
    strict = context.resolve(tmp_path, context.build_context("2024-01-10", catalog.RUN_MODE_STRICT))
    assert strict["usable"] is False
    assert "etf_info" in {item["dataset_id"] for item in strict["blocked_datasets"]}

    research = context.resolve(tmp_path, context.build_context("2024-01-10"))
    assert research["usable"] is True
    assert research["blocked_datasets"] == []


def test_require_usable_is_a_no_op_in_research_mode_and_raises_in_strict(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    pd.DataFrame({"ts_code": ["510300.SH"], "name": ["x"], "fund_type": ["ETF"]}).to_parquet(
        tmp_path / "etf_info_df.parquet", index=False
    )
    context.require_usable(tmp_path, context.build_context("2024-01-10"), ["etf_info"])
    with pytest.raises(context.PitContextError, match="严格 PIT 模式禁止"):
        context.require_usable(
            tmp_path, context.build_context("2024-01-10", catalog.RUN_MODE_STRICT), ["etf_info"]
        )


# --------------------------------------------------------------------------- #
# the actual look-ahead fix
# --------------------------------------------------------------------------- #


def test_as_of_cuts_on_the_announcement_date_not_the_value_date(tmp_path: Path) -> None:
    _nav_fixture(tmp_path, announcement_lag=1)
    dates = pd.bdate_range("2024-01-01", periods=12)
    cut = dates[5]  # a NAV date that is only announced the following day

    loaded = fit.load_adj_nav_pit(tmp_path, ["510300.SH"], [], as_of=cut)
    assert loaded.lineage["as_of_applied"] is True
    # The value dated `cut` was published on cut+1, so it must not be visible.
    assert loaded.frame["date"].max() == dates[4]
    assert loaded.lineage["rows_dropped_by_as_of"] > 0

    # Filtering on the value date instead would have kept one extra row; that
    # difference is exactly the look-ahead this fix removes.
    naive = loaded.lineage["rows_after_cut"]
    all_rows = fit.load_adj_nav_pit(tmp_path, ["510300.SH"], []).lineage["rows_after_cut"]
    assert naive < all_rows


def test_no_as_of_keeps_every_row_but_says_no_cut_happened(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    loaded = fit.load_adj_nav_pit(tmp_path, ["510300.SH", "511010.SH"], [])
    assert loaded.lineage["as_of_applied"] is False
    assert loaded.lineage["rows_dropped_by_as_of"] == 0
    assert loaded.lineage["rows_after_cut"] == 24


def test_research_mode_falls_back_to_the_value_date_and_warns(tmp_path: Path) -> None:
    _nav_fixture(tmp_path, drop_announcements=4)
    loaded = fit.load_adj_nav_pit(tmp_path, ["510300.SH"], [], as_of="2024-01-31")
    assert loaded.lineage["announcement_fallback"] is True
    assert loaded.lineage["rows_without_announcement"] == 4
    assert any("缺少 ann_date" in warning for warning in loaded.lineage["warnings"])
    # Falling back keeps the rows; it just refuses to call them strictly PIT.
    assert loaded.lineage["rows_after_cut"] == 12


def test_strict_mode_drops_unannounced_rows_instead_of_guessing(tmp_path: Path) -> None:
    _nav_fixture(tmp_path, drop_announcements=4)
    loaded = fit.load_adj_nav_pit(
        tmp_path, ["510300.SH"], [], as_of="2024-01-31", run_mode=catalog.RUN_MODE_STRICT
    )
    assert loaded.lineage["announcement_fallback"] is False
    assert loaded.lineage["rows_after_cut"] == 8
    assert any("丢弃 4 行" in warning for warning in loaded.lineage["warnings"])


def test_strict_mode_refuses_a_nav_file_without_an_announcement_column(tmp_path: Path) -> None:
    _nav_fixture(tmp_path, include_ann_date=False)
    with pytest.raises(context.PitContextError, match="要求净值数据带公告日"):
        fit.load_adj_nav_pit(
            tmp_path, ["510300.SH"], [], as_of="2024-01-31", run_mode=catalog.RUN_MODE_STRICT
        )
    # Research mode still works on the same file, with the fallback flagged.
    loaded = fit.load_adj_nav_pit(tmp_path, ["510300.SH"], [], as_of="2024-01-31")
    assert loaded.lineage["announcement_fallback"] is True


def test_strict_mode_without_as_of_is_refused_at_the_loader_too(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    with pytest.raises(context.PitContextError, match="必须指定研究日"):
        fit.load_adj_nav_pit(tmp_path, ["510300.SH"], [], run_mode=catalog.RUN_MODE_STRICT)


def test_the_latest_revision_knowable_on_as_of_wins(tmp_path: Path) -> None:
    """A restated NAV must not leak backwards through the de-duplication."""

    value_date = pd.Timestamp("2024-01-05")
    frame = pd.DataFrame(
        [
            # original print, published next day
            {
                "ts_code": "510300.SH",
                "name": "fixture",
                "nav_date": value_date,
                "date": value_date,
                "ann_date": value_date + pd.Timedelta(days=1),
                "adj_nav": 3.0,
            },
            # restatement, published a week later
            {
                "ts_code": "510300.SH",
                "name": "fixture",
                "nav_date": value_date,
                "date": value_date,
                "ann_date": value_date + pd.Timedelta(days=8),
                "adj_nav": 9.9,
            },
        ]
    )
    frame.to_parquet(tmp_path / "etf_daily_df.parquet", index=False)

    early = fit.load_adj_nav_pit(tmp_path, ["510300.SH"], [], as_of=value_date + pd.Timedelta(days=2))
    assert early.frame["adj_nav"].tolist() == [3.0], "the restatement was not public yet"

    later = fit.load_adj_nav_pit(tmp_path, ["510300.SH"], [], as_of=value_date + pd.Timedelta(days=30))
    assert later.frame["adj_nav"].tolist() == [9.9], "the latest knowable revision must win"


def test_load_adj_nav_wrapper_keeps_the_old_frame_only_contract(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    frame = fit._load_adj_nav(tmp_path, ["510300.SH"], [])
    assert isinstance(frame, pd.DataFrame)
    for column in ("ts_code", "name", "date", "adj_nav"):
        assert column in frame.columns


# --------------------------------------------------------------------------- #
# releases
# --------------------------------------------------------------------------- #


def test_release_pins_the_tables_and_chains_to_its_parent(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    repository = DataReleaseRepository(tmp_path / "data_releases.json")

    first = repository.create(tmp_path, "基线")
    assert first["parent_release_id"] is None
    assert first["immutable"] is True
    assert {table["dataset_id"] for table in first["tables"]} == {"etf_nav"}
    assert first["tables"][0]["rows"] == 24
    assert len(first["release_fingerprint"]) == 64

    second = repository.create(tmp_path, "同样的数据")
    assert second["parent_release_id"] == first["id"]
    # Same files, same fingerprint: a release is about content, not about time.
    assert second["release_fingerprint"] == first["release_fingerprint"]

    _nav_fixture(tmp_path, announcement_lag=4)
    pit_audit.clear_cache()
    third = repository.create(tmp_path, "改过数据")
    assert third["release_fingerprint"] != first["release_fingerprint"]

    assert [item["id"] for item in repository.list_releases()][0] == third["id"]
    assert repository.get(first["id"])["name"] == "基线"


def test_release_refuses_to_seal_nothing_or_an_unnamed_version(tmp_path: Path) -> None:
    repository = DataReleaseRepository(tmp_path / "data_releases.json")
    with pytest.raises(DataReleaseError, match="名称不能为空"):
        repository.create(tmp_path, "   ")
    with pytest.raises(DataReleaseError, match="没有任何可封版的数据集"):
        repository.create(tmp_path, "空版本")
    with pytest.raises(DataReleaseError, match="未找到数据版本"):
        repository.get("release-nope")
    assert repository.latest() is None


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.services import pit_routes

    monkeypatch.setattr(pit_routes, "DATA_DIR", tmp_path)
    app = FastAPI()
    app.include_router(pit_routes.router)
    return TestClient(app)


def _await_audit(client, attempts: int = 100) -> dict:
    """Poll the audit until the background scan has measured everything."""

    for _ in range(attempts):
        payload = client.get("/api/pit/audit").json()
        if not payload["summary"]["pending"]:
            return payload
        time.sleep(0.05)
    raise AssertionError("后台 PIT 扫描没有在预期时间内完成")


def test_audit_route_reports_the_sheet_and_the_latest_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _nav_fixture(tmp_path, announcement_lag=2)
    client = _client(tmp_path, monkeypatch)

    # The first call answers from an empty memo and starts the scan behind the
    # request: reading the clock columns of every declared file is minutes on
    # cold storage, and a settings page that hangs for minutes is a broken page.
    first = client.get("/api/pit/audit").json()
    assert first["latest_release"] is None
    assert first["summary"]["pending"] >= 1
    assert first["scan"]["state"] == "running"
    assert next(item for item in first["datasets"] if item["dataset_id"] == "etf_nav")["grade"] is None

    payload = _await_audit(client)
    nav = next(item for item in payload["datasets"] if item["dataset_id"] == "etf_nav")
    assert nav["grade"] == "A"
    assert nav["lag"]["p50"] == pytest.approx(2.0)
    assert payload["summary"]["grade_a"] >= 1
    assert payload["summary"]["pending"] == 0

    sealed = client.post("/api/pit/releases", json={"name": "基线", "note": "自审"})
    assert sealed.status_code == 200
    release_id = sealed.json()["id"]

    refreshed = client.get("/api/pit/audit").json()
    assert refreshed["latest_release"]["id"] == release_id
    assert client.get(f"/api/pit/releases/{release_id}").json()["name"] == "基线"
    assert client.get("/api/pit/releases/release-nope").status_code == 404
    assert len(client.get("/api/pit/releases").json()["releases"]) == 1


def test_release_route_rejects_an_empty_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    response = client.post("/api/pit/releases", json={"name": "空"})
    assert response.status_code == 400
    assert "可封版" in response.json()["detail"]


def test_context_route_returns_a_readable_reason_not_a_422_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _nav_fixture(tmp_path)
    client = _client(tmp_path, monkeypatch)

    bad = client.post("/api/pit/context/resolve", json={"runMode": "STRICT_PIT"})
    assert bad.status_code == 400
    # A plain string detail is what the UI can actually render.
    assert isinstance(bad.json()["detail"], str)
    assert "研究日" in bad.json()["detail"]

    ok = client.post(
        "/api/pit/context/resolve", json={"asOf": "2024-01-10", "runMode": "STRICT_PIT"}
    ).json()
    assert ok["context"]["run_mode_label"] == "严格 PIT"
    assert ok["usable"] is True

    missing_release = client.post(
        "/api/pit/context/resolve",
        json={"asOf": "2024-01-10", "runMode": "RESEARCH", "dataReleaseId": "release-nope"},
    ).json()
    assert missing_release["usable"] is False
    assert "未找到数据版本" in missing_release["release_error"]


def test_meta_route_exposes_every_grade_and_run_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _client(tmp_path, monkeypatch).get("/api/pit/meta").json()
    assert {item["id"] for item in payload["grades"]} == {"A", "B", "C"}
    assert {item["id"] for item in payload["run_modes"]} == set(catalog.RUN_MODES)
    assert payload["strict_coverage_floor"] == catalog.STRICT_COVERAGE_FLOOR


# --------------------------------------------------------------------------- #
# system-level PIT setting
# --------------------------------------------------------------------------- #


def test_no_release_means_no_pit_and_that_is_stated_not_implied(tmp_path: Path) -> None:
    payload = PitSettingsRepository(tmp_path).describe()
    assert payload["effective"]["no_pit"] is True
    assert payload["effective"]["as_of"] is None
    assert payload["effective"]["run_mode"] == catalog.RUN_MODE_RESEARCH
    assert payload["effective"]["label"] == "无 PIT 口径 · 使用全部磁盘数据"
    assert payload["can_apply"] is False
    assert payload["available_releases"] == []


def test_applying_a_release_fixes_the_research_day_from_the_release(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    releases = DataReleaseRepository(tmp_path / "data_releases.json")
    release = releases.create(tmp_path, "基线")
    expected_as_of = release["summary"]["available_through"]
    assert expected_as_of == "2024-01-17"

    repository = PitSettingsRepository(tmp_path)
    payload = repository.update(release["id"], catalog.RUN_MODE_STRICT)
    effective = payload["effective"]
    assert effective["no_pit"] is False
    # A release still implies a research day when none is stated, so existing
    # installs keep the口径 they had — but the source is now reported, because a
    # day the user chose and a day the vintage happened to end on are different
    # claims and used to be indistinguishable.
    assert effective["as_of"] == expected_as_of
    assert effective["as_of_source"] == "release"
    assert effective["run_mode"] == catalog.RUN_MODE_STRICT
    assert effective["data_release_id"] == release["id"]
    assert effective["label"] == f"站在 {expected_as_of} · 基线 · 严格 PIT"

    # The setting is on disk, not in a browser: a fresh repository sees it.
    assert PitSettingsRepository(tmp_path).effective_context().as_of == expected_as_of


def test_research_day_stands_alone_without_any_release(tmp_path: Path) -> None:
    repository = PitSettingsRepository(tmp_path)

    effective = repository.update(None, catalog.RUN_MODE_RESEARCH, "2010 年前研究", "2009-12-31")["effective"]

    assert effective["as_of"] == "2009-12-31"
    assert effective["as_of_source"] == "explicit"
    assert effective["no_pit"] is False
    assert effective["label"] == "站在 2009-12-31 · 最新数据（未封版） · 研究模式"
    assert PitSettingsRepository(tmp_path).effective_context().as_of == "2009-12-31"


def test_research_day_may_not_run_past_the_pinned_vintage(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    releases = DataReleaseRepository(tmp_path / "data_releases.json")
    release = releases.create(tmp_path, "基线")
    repository = PitSettingsRepository(tmp_path)

    # The one constraint between the two knobs, and it only runs one way.
    with pytest.raises(context.PitContextError, match="晚于数据版本"):
        repository.update(release["id"], catalog.RUN_MODE_RESEARCH, "", "2030-01-01")

    effective = repository.update(release["id"], catalog.RUN_MODE_RESEARCH, "", "2024-01-10")["effective"]
    assert effective["as_of"] == "2024-01-10"
    assert effective["as_of_source"] == "explicit"


def test_strict_pit_cannot_be_enabled_without_a_research_day(tmp_path: Path) -> None:
    repository = PitSettingsRepository(tmp_path)
    # Strict needs a day to enforce, not a release: requiring a sealed vintage
    # here was what left "stand on 2009-12-31" unreachable for anyone who had
    # never封版.
    with pytest.raises(context.PitContextError, match="严格 PIT 需要一个研究日"):
        repository.update(None, catalog.RUN_MODE_STRICT)
    with pytest.raises(context.PitContextError, match="不支持的运行模式"):
        repository.update(None, "YOLO")
    with pytest.raises(DataReleaseError, match="未找到数据版本"):
        repository.update("release-nope", catalog.RUN_MODE_RESEARCH)
    # None of the rejected attempts may have been persisted.
    assert repository.describe()["effective"]["no_pit"] is True


def test_clearing_the_release_returns_to_no_pit(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    release = DataReleaseRepository(tmp_path / "data_releases.json").create(tmp_path, "基线")
    repository = PitSettingsRepository(tmp_path)
    repository.update(release["id"], catalog.RUN_MODE_STRICT)
    payload = repository.update(None, catalog.RUN_MODE_RESEARCH)
    assert payload["effective"]["no_pit"] is True
    assert payload["effective"]["as_of"] is None


def test_a_release_that_disappears_degrades_to_no_pit_loudly(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    store = tmp_path / "data_releases.json"
    release = DataReleaseRepository(store).create(tmp_path, "基线")
    repository = PitSettingsRepository(tmp_path)
    repository.update(release["id"], catalog.RUN_MODE_STRICT)

    store.write_text('{"schema_version": 1, "releases": []}', encoding="utf-8")
    payload = repository.describe()
    # Never keep a stale as_of for a vintage that no longer exists.
    assert payload["effective"]["no_pit"] is True
    assert payload["effective"]["as_of"] is None
    assert payload["effective"]["run_mode"] == catalog.RUN_MODE_RESEARCH
    assert "未找到数据版本" in payload["release_error"]
    # The stored intent is preserved so the user can see what was configured.
    assert payload["settings"]["active_release_id"] == release["id"]


# --------------------------------------------------------------------------- #
# request context inheritance
# --------------------------------------------------------------------------- #


def test_a_request_that_states_nothing_inherits_the_system_setting(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    release = DataReleaseRepository(tmp_path / "data_releases.json").create(tmp_path, "基线")
    PitSettingsRepository(tmp_path).update(release["id"], catalog.RUN_MODE_STRICT)

    inherited = context.resolve_request_context(tmp_path)
    assert inherited.as_of == "2024-01-17"
    assert inherited.run_mode == catalog.RUN_MODE_STRICT
    assert inherited.data_release_id == release["id"]


def test_an_explicit_as_of_overrides_the_system_setting(tmp_path: Path) -> None:
    """A backtest sweeps as_of by nature, so a stated value has to win."""

    _nav_fixture(tmp_path)
    release = DataReleaseRepository(tmp_path / "data_releases.json").create(tmp_path, "基线")
    PitSettingsRepository(tmp_path).update(release["id"], catalog.RUN_MODE_STRICT)

    overridden = context.resolve_request_context(tmp_path, as_of="2024-01-10")
    assert overridden.as_of == "2024-01-10"
    # Unstated fields still come from the system setting.
    assert overridden.run_mode == catalog.RUN_MODE_STRICT
    assert overridden.data_release_id == release["id"]


def test_request_context_is_no_pit_until_something_is_applied(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    plain = context.resolve_request_context(tmp_path)
    assert plain.as_of is None
    assert plain.run_mode == catalog.RUN_MODE_RESEARCH
    # Which is exactly the pre-PIT behaviour: every row on disk is in play.
    loaded = fit.load_adj_nav_pit(tmp_path, ["510300.SH"], [], as_of=plain.as_of, run_mode=plain.run_mode)
    assert loaded.lineage["as_of_applied"] is False
    assert loaded.lineage["rows_after_cut"] == 12


def test_applied_setting_actually_truncates_a_nav_load(tmp_path: Path) -> None:
    """The end-to-end point of the whole feature, in one assertion."""

    _nav_fixture(tmp_path, announcement_lag=1)
    releases = DataReleaseRepository(tmp_path / "data_releases.json")
    release = releases.create(tmp_path, "基线")
    PitSettingsRepository(tmp_path).update(release["id"], catalog.RUN_MODE_RESEARCH)

    resolved = context.resolve_request_context(tmp_path)
    loaded = fit.load_adj_nav_pit(
        tmp_path, ["510300.SH"], [], as_of=resolved.as_of, run_mode=resolved.run_mode
    )
    assert loaded.lineage["as_of_applied"] is True
    assert loaded.lineage["as_of"] == "2024-01-17"
    # The last value date is 2024-01-16, announced 01-17, so it is visible.
    assert loaded.frame["date"].max() == pd.Timestamp("2024-01-16")


# --------------------------------------------------------------------------- #
# settings routes
# --------------------------------------------------------------------------- #


def test_settings_routes_apply_and_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _nav_fixture(tmp_path)
    client = _client(tmp_path, monkeypatch)

    assert client.get("/api/pit/settings").json()["effective"]["no_pit"] is True

    release_id = client.post("/api/pit/releases", json={"name": "基线"}).json()["id"]
    applied = client.put(
        "/api/pit/settings", json={"activeReleaseId": release_id, "runMode": "STRICT_PIT"}
    )
    assert applied.status_code == 200
    assert applied.json()["effective"]["as_of"] == "2024-01-17"
    # A later GET must report the same thing; the setting is server-held.
    assert client.get("/api/pit/settings").json()["effective"]["run_mode"] == "STRICT_PIT"

    rejected = client.put("/api/pit/settings", json={"activeReleaseId": None, "runMode": "STRICT_PIT"})
    assert rejected.status_code == 400
    assert isinstance(rejected.json()["detail"], str)
    assert "严格 PIT" in rejected.json()["detail"]
    # The rejection must not have changed anything.
    assert client.get("/api/pit/settings").json()["effective"]["run_mode"] == "STRICT_PIT"

    cleared = client.put("/api/pit/settings", json={"activeReleaseId": None, "runMode": "RESEARCH"})
    assert cleared.json()["effective"]["no_pit"] is True


# --------------------------------------------------------------------------- #
# per-tab viewing override
# --------------------------------------------------------------------------- #


def _view_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A client whose probe route runs the middleware and a *sync* endpoint.

    Sync on purpose: nearly every real endpoint is `def`, so it runs in the
    threadpool, and this is what proves the viewing口径 survives the hop.
    """

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.services import pit_routes

    monkeypatch.setattr(pit_routes, "DATA_DIR", tmp_path)
    app = FastAPI()
    app.add_middleware(pit_routes.PitViewOverrideMiddleware)
    app.include_router(pit_routes.router)

    @app.get("/probe")
    def probe() -> dict:
        resolved = context.resolve_request_context(tmp_path)
        return {
            "as_of": resolved.as_of,
            "run_mode": resolved.run_mode,
            "data_release_id": resolved.data_release_id,
        }

    return TestClient(app)


def test_headers_that_say_nothing_leave_the_system_setting_alone(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    assert context.parse_view_override(tmp_path, {}) is None
    assert context.parse_view_override(tmp_path, {"x-pit-off": ""}) is None


def test_pit_off_header_is_distinguishable_from_saying_nothing(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    release = DataReleaseRepository(tmp_path / "data_releases.json").create(tmp_path, "基线")
    PitSettingsRepository(tmp_path).update(release["id"], catalog.RUN_MODE_STRICT)

    override = context.parse_view_override(tmp_path, {"x-pit-off": "1"})
    assert override == context.ResearchContext()
    assert override.as_of is None


def test_viewing_a_version_takes_that_version_whole(tmp_path: Path) -> None:
    """A version is one口径: its day and its mode travel together.

    The tab carries only the version id; repeating the day in a header is how
    the two would drift apart.
    """

    _nav_fixture(tmp_path)
    releases = DataReleaseRepository(tmp_path / "data_releases.json")
    PitSettingsRepository(tmp_path).update(None, catalog.RUN_MODE_RESEARCH)
    strict = releases.create(
        tmp_path, "站在 01-10 的严格版", as_of="2024-01-10", run_mode=catalog.RUN_MODE_STRICT
    )

    override = context.parse_view_override(tmp_path, {"x-pit-release": strict["id"]})
    assert override.data_release_id == strict["id"]
    assert override.as_of == "2024-01-10"
    assert override.run_mode == catalog.RUN_MODE_STRICT


def test_a_version_that_states_no_mode_cannot_relax_a_strict_system(tmp_path: Path) -> None:
    """Versions sealed before the mode moved into them say nothing about it."""

    _nav_fixture(tmp_path)
    releases = DataReleaseRepository(tmp_path / "data_releases.json")
    first = releases.create(tmp_path, "基线")
    PitSettingsRepository(tmp_path).update(first["id"], catalog.RUN_MODE_STRICT)
    legacy = releases.create(tmp_path, "第二版")
    legacy.pop("run_mode")
    store = tmp_path / "data_releases.json"
    payload = json.loads(store.read_text(encoding="utf-8"))
    for item in payload["releases"]:
        if item["id"] == legacy["id"]:
            item.pop("run_mode", None)
    store.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    override = context.parse_view_override(tmp_path, {"x-pit-release": legacy["id"]})
    assert override.as_of == "2024-01-17"
    assert override.run_mode == catalog.RUN_MODE_STRICT


def test_an_unknown_release_header_is_rejected_not_ignored(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    with pytest.raises(DataReleaseError, match="未找到数据版本"):
        context.parse_view_override(tmp_path, {"x-pit-release": "release-nope"})


def test_view_override_layers_under_an_explicitly_stated_value(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    release = DataReleaseRepository(tmp_path / "data_releases.json").create(tmp_path, "基线")
    PitSettingsRepository(tmp_path).update(release["id"], catalog.RUN_MODE_RESEARCH)

    token = context.set_view_override(context.ResearchContext())
    try:
        # The tab turned PIT off, so an endpoint that states nothing sees no cut.
        assert context.resolve_request_context(tmp_path).as_of is None
        # A backtest that sweeps as_of still wins over the viewing choice.
        assert context.resolve_request_context(tmp_path, as_of="2024-01-10").as_of == "2024-01-10"
    finally:
        context.reset_view_override(token)
    # And the override never leaks past the request that set it.
    assert context.resolve_request_context(tmp_path).as_of == "2024-01-17"


def test_view_headers_reach_a_sync_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _nav_fixture(tmp_path)
    client = _view_client(tmp_path, monkeypatch)
    release_id = client.post("/api/pit/releases", json={"name": "基线"}).json()["id"]
    client.put("/api/pit/settings", json={"activeReleaseId": release_id, "runMode": "STRICT_PIT"})

    inherited = client.get("/probe").json()
    assert inherited["as_of"] == "2024-01-17"
    assert inherited["run_mode"] == "STRICT_PIT"

    turned_off = client.get("/probe", headers={"X-Pit-Off": "1"}).json()
    assert turned_off["as_of"] is None
    assert turned_off["run_mode"] == "RESEARCH"
    assert turned_off["data_release_id"] is None

    # The system setting is untouched by anyone's viewing choice.
    assert client.get("/api/pit/settings").json()["effective"]["run_mode"] == "STRICT_PIT"
    assert client.get("/probe").json()["run_mode"] == "STRICT_PIT"


def test_a_bad_view_header_fails_the_request_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _nav_fixture(tmp_path)
    client = _view_client(tmp_path, monkeypatch)

    rejected = client.get("/probe", headers={"X-Pit-Release": "release-nope"})
    assert rejected.status_code == 400
    assert "临时 PIT 口径无效" in rejected.json()["detail"]

    bad_mode = client.get("/probe", headers={"X-Pit-Run-Mode": "YOLO"})
    assert bad_mode.status_code == 400


def test_a_version_carries_the_research_day_it_was_defined_with(tmp_path: Path) -> None:
    """The one-concept model, end to end.

    Seal a version that stands on 2024-01-10, apply it, and every request that
    states nothing computes on that day — no second setting to remember.
    """

    _nav_fixture(tmp_path)
    releases = DataReleaseRepository(tmp_path / "data_releases.json")
    version = releases.create(tmp_path, "站在 01-10", as_of="2024-01-10")
    described = PitSettingsRepository(tmp_path).update(version["id"], None)

    assert described["effective"]["as_of"] == "2024-01-10"
    assert described["effective"]["as_of_source"] == "release"
    assert described["release"]["as_of"] == "2024-01-10"
    # Nothing is stored beside the version; the version is the setting.
    assert described["settings"]["as_of"] is None
    assert described["settings"]["run_mode"] is None
    assert context.resolve_request_context(tmp_path).as_of == "2024-01-10"


def test_a_version_may_not_stand_later_than_its_data_reaches(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    releases = DataReleaseRepository(tmp_path / "data_releases.json")
    with pytest.raises(DataReleaseError, match="晚于这批数据的可得截止日"):
        releases.create(tmp_path, "站在未来", as_of="2030-01-01")


def test_a_strict_version_needs_a_day_at_seal_time(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    releases = DataReleaseRepository(tmp_path / "data_releases.json")
    with pytest.raises(DataReleaseError, match="严格 PIT 需要一个研究日"):
        releases.create(tmp_path, "严格但没有日子", run_mode=catalog.RUN_MODE_STRICT)


def test_a_version_can_be_edited_without_re_sealing_its_vintage(tmp_path: Path) -> None:
    """Getting the research day wrong must not cost a re-seal.

    The口径 half (name, note, day, mode) is the user's own choice; the evidence
    half (tables and fingerprints) is what makes the version worth anything, so
    it is left exactly as sealed.
    """

    _nav_fixture(tmp_path)
    releases = DataReleaseRepository(tmp_path / "data_releases.json")
    version = releases.create(tmp_path, "名字起错了")

    edited = releases.update(
        version["id"], name="站在 01-10", note="改过", as_of="2024-01-10", run_mode=catalog.RUN_MODE_STRICT
    )
    assert edited["name"] == "站在 01-10"
    assert edited["as_of"] == "2024-01-10"
    assert edited["run_mode"] == catalog.RUN_MODE_STRICT
    assert edited["updated_at"]
    # The vintage is untouched.
    assert edited["release_fingerprint"] == version["release_fingerprint"]
    assert edited["tables"] == version["tables"]

    # Applying it now stands the platform on the edited day, with no second setting.
    described = PitSettingsRepository(tmp_path).update(version["id"], None)
    assert described["effective"]["as_of"] == "2024-01-10"
    assert described["effective"]["run_mode"] == catalog.RUN_MODE_STRICT


def test_an_edit_may_not_push_the_day_past_the_sealed_vintage(tmp_path: Path) -> None:
    _nav_fixture(tmp_path)
    releases = DataReleaseRepository(tmp_path / "data_releases.json")
    version = releases.create(tmp_path, "基线")
    with pytest.raises(DataReleaseError, match="晚于这个版本的可得截止日"):
        releases.update(version["id"], name="基线", as_of="2030-01-01")


def test_deleting_the_applied_version_is_refused_not_silently_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise every page quietly falls back to no PIT and nobody is told."""

    _nav_fixture(tmp_path)
    client = _client(tmp_path, monkeypatch)
    version = client.post("/api/pit/releases", json={"name": "基线"}).json()
    client.put("/api/pit/settings", json={"activeReleaseId": version["id"]})

    refused = client.delete(f"/api/pit/releases/{version['id']}")
    assert refused.status_code == 409
    assert "正在被全平台使用" in refused.json()["detail"]
    assert len(client.get("/api/pit/releases").json()["releases"]) == 1

    # Switch off first, then it deletes.
    client.put("/api/pit/settings", json={"activeReleaseId": None})
    assert client.delete(f"/api/pit/releases/{version['id']}").status_code == 200
    assert client.get("/api/pit/releases").json()["releases"] == []
    assert client.delete(f"/api/pit/releases/{version['id']}").status_code == 404
