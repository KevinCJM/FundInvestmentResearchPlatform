"""Point-in-time at the decision layer: universe replay and the decision clock.

The three earlier PIT rounds answered "which rows were visible on T". These
tests pin the two questions they could not: which *products* were selectable on
T, and what *decision* a desk standing on T would actually have reached.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import backend.backtest_engine as engine  # noqa: E402
from backend.pit.catalog import SNAPSHOT_FIELD  # noqa: E402
from backend.pit.clock import DecisionClock, visible_at  # noqa: E402
from backend.pit.context import PitContextError, ResearchContext, build_context  # noqa: E402
from backend.pit.frame import LATEST_ONLY, REPLAYED, read_pit  # noqa: E402
from backend.pit.guard import (  # noqa: E402
    UNIVERSE_LOOKAHEAD,
    UNIVERSE_NOT_REPLAYABLE,
    assert_no_universe_lookahead,
    check_universe,
)
from backend.pit.universe import INTERVAL, universe_as_of  # noqa: E402


# --------------------------------------------------------------------------- #
# L0 — the append-only dimension log
# --------------------------------------------------------------------------- #


def _load_t01():
    import importlib

    return importlib.import_module("T01_get_data")


def test_dimension_snapshot_keeps_the_state_of_each_day(tmp_path) -> None:
    t01 = _load_t01()
    path = tmp_path / "etf_info_df.parquet"

    day_one = pd.DataFrame({"ts_code": ["A.SH", "B.SH"], "name": ["甲", "乙"]})
    t01.append_dimension_snapshot(day_one, path, snapshot_date="2024-01-02")
    # B delisted, C launched: the overwrite-only file would lose B entirely.
    day_two = pd.DataFrame({"ts_code": ["A.SH", "C.SH"], "name": ["甲", "丙"]})
    log_path = t01.append_dimension_snapshot(day_two, path, snapshot_date="2024-02-01")

    log = pd.read_parquet(log_path)
    assert sorted(log[SNAPSHOT_FIELD].dt.strftime("%Y-%m-%d").unique()) == [
        "2024-01-02",
        "2024-02-01",
    ]
    on_day_one = log[log[SNAPSHOT_FIELD] == pd.Timestamp("2024-01-02")]
    assert set(on_day_one["ts_code"]) == {"A.SH", "B.SH"}


def test_identical_state_does_not_grow_the_log(tmp_path) -> None:
    t01 = _load_t01()
    path = tmp_path / "etf_info_df.parquet"
    frame = pd.DataFrame({"ts_code": ["A.SH"], "name": ["甲"]})

    t01.append_dimension_snapshot(frame, path, snapshot_date="2024-01-02")
    log_path = t01.append_dimension_snapshot(frame, path, snapshot_date="2024-02-01")

    log = pd.read_parquet(log_path)
    assert log[SNAPSHOT_FIELD].nunique() == 1


def test_same_day_rerun_replaces_rather_than_doubles(tmp_path) -> None:
    t01 = _load_t01()
    path = tmp_path / "etf_info_df.parquet"

    t01.append_dimension_snapshot(
        pd.DataFrame({"ts_code": ["A.SH"], "name": ["甲"]}), path, snapshot_date="2024-01-02"
    )
    log_path = t01.append_dimension_snapshot(
        pd.DataFrame({"ts_code": ["A.SH", "B.SH"], "name": ["甲", "乙"]}),
        path,
        snapshot_date="2024-01-02",
    )

    log = pd.read_parquet(log_path)
    assert len(log) == 2


# --------------------------------------------------------------------------- #
# L1/L2 — replaying the universe
# --------------------------------------------------------------------------- #


def _write_universe(tmp_path: Path, *, with_history: bool) -> None:
    latest = pd.DataFrame(
        {
            "ts_code": ["OLD.SH", "NEW.SH"],
            "name": ["老基金", "新基金"],
            "list_date": ["20150101", "20230101"],
            # OLD delisted in 2020; NEW had not launched in 2018.
            "delist_date": ["20200101", None],
        }
    )
    latest.to_parquet(tmp_path / "etf_info_df.parquet", index=False)
    if with_history:
        history = latest.copy()
        history[SNAPSHOT_FIELD] = pd.Timestamp("2018-06-30")
        history = history[history["ts_code"] == "OLD.SH"]
        (tmp_path / "pit_dim").mkdir(exist_ok=True)
        history.to_parquet(tmp_path / "pit_dim" / "etf_info_df_history.parquet", index=False)


def test_universe_excludes_products_that_did_not_exist_yet(tmp_path) -> None:
    _write_universe(tmp_path, with_history=False)

    view = universe_as_of(tmp_path, build_context("2018-06-30"), kind="fund")

    assert view.coverage == INTERVAL
    # The survivorship trap in one assertion: today's file has both, 2018 had one.
    assert set(view.codes) == {"OLD.SH"}


def test_universe_without_a_cut_off_is_todays_table(tmp_path) -> None:
    _write_universe(tmp_path, with_history=False)

    view = universe_as_of(tmp_path, ResearchContext(), kind="fund")

    assert set(view.codes) == {"OLD.SH", "NEW.SH"}
    assert view.coverage == LATEST_ONLY


def test_snapshot_log_upgrades_coverage_to_replayed(tmp_path) -> None:
    _write_universe(tmp_path, with_history=True)

    view = universe_as_of(tmp_path, build_context("2018-06-30"), kind="fund")

    assert view.coverage == REPLAYED
    assert view.history_begins_at == "2018-06-30"
    assert set(view.codes) == {"OLD.SH"}


def test_read_pit_before_history_begins_warns_instead_of_lying(tmp_path) -> None:
    _write_universe(tmp_path, with_history=True)

    loaded = read_pit("etf_info", tmp_path, build_context("2016-01-01"))

    assert loaded.lineage["coverage"] == LATEST_ONLY
    assert any("早于维表历史起点" in text for text in loaded.lineage["warnings"])


def test_strict_mode_refuses_a_universe_it_cannot_replay(tmp_path) -> None:
    _write_universe(tmp_path, with_history=False)

    with pytest.raises(PitContextError):
        read_pit("etf_info", tmp_path, build_context("2018-06-30", "STRICT_PIT"))


# --------------------------------------------------------------------------- #
# L3 — the decision clock
# --------------------------------------------------------------------------- #


def test_visible_at_drops_rows_published_after_the_decision() -> None:
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=index)
    # The 01-03 print is only announced on 01-08.
    available = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-08"]), index=index)

    assert len(visible_at(frame, "2024-01-03")) == 3
    assert len(visible_at(frame, "2024-01-03", available)) == 2


def test_decision_clock_never_looks_past_its_own_context() -> None:
    clock = DecisionClock.from_dates(
        ["2020-01-31", "2020-02-28", "2021-01-29"], build_context("2020-12-31")
    )

    assert clock.dates == ("2020-01-31", "2020-02-28")
    assert clock.at("2020-02-28").as_of == "2020-02-28"
    assert clock.lineage()["swept"] is True


def _write_alloc(tmp_path: Path, *, lag_days: int, as_of: str | None) -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    rows = []
    for offset, day in enumerate(dates):
        for name, base, step in (("ClassA", 100.0, 1.0), ("ClassB", 100.0, -0.4)):
            rows.append(
                {
                    "asset_alloc_name": "demo",
                    "asset_name": name,
                    "date": day,
                    "nv": base + step * offset,
                    "creat_time": pd.Timestamp("2026-01-01"),
                    "as_of": as_of,
                    "run_mode": "RESEARCH",
                    "available_at": day + pd.Timedelta(days=lag_days),
                }
            )
    pd.DataFrame(rows).to_parquet(tmp_path / "asset_nv.parquet", index=False)


def test_publication_lag_changes_the_backtest_result(tmp_path) -> None:
    _write_alloc(tmp_path, lag_days=7, as_of=None)
    loaded = engine.load_allocation_nav(tmp_path, "demo")
    strategies = [
        {
            "name": "min_risk",
            "type": "target",
            "weights": [0.5, 0.5],
            "rebalance": {"enabled": True, "mode": "fixed", "fixedInterval": 5, "recalc": True},
            "model": {"window_mode": "rollingn", "data_len": 10, "target": "min_risk"},
            "classes": [{"name": "ClassA"}, {"name": "ClassB"}],
        }
    ]

    naive = engine.backtest_portfolio(loaded.nav_wide, strategies)
    honest = engine.backtest_portfolio(loaded.nav_wide, strategies, available_at=loaded.available_at)

    # Same result would mean the clock never bit. The refits must see less.
    assert naive["markers"] != honest["markers"]


def test_allocation_read_prefers_the_series_built_for_the_research_day(tmp_path) -> None:
    _write_alloc(tmp_path, lag_days=0, as_of="2024-01-20")
    extra = pd.read_parquet(tmp_path / "asset_nv.parquet")
    hindsight = extra.copy()
    hindsight["as_of"] = None
    hindsight["nv"] = hindsight["nv"] * 2
    pd.concat([extra, hindsight], ignore_index=True).to_parquet(
        tmp_path / "asset_nv.parquet", index=False
    )

    picked = engine.load_allocation_nav(tmp_path, "demo", build_context("2024-02-05"))

    assert picked.lineage["series_as_of"] == "2024-01-20"
    assert picked.lineage["hindsight_series"] is False
    assert float(picked.nav_wide["ClassA"].iloc[0]) == pytest.approx(100.0)


def test_hindsight_series_is_labelled_not_hidden(tmp_path) -> None:
    _write_alloc(tmp_path, lag_days=0, as_of=None)

    picked = engine.load_allocation_nav(tmp_path, "demo", build_context("2024-02-05"))

    assert picked.lineage["hindsight_series"] is True
    assert picked.lineage["warnings"]


def test_rows_not_yet_published_are_cut_from_the_allocation(tmp_path) -> None:
    _write_alloc(tmp_path, lag_days=10, as_of=None)

    picked = engine.load_allocation_nav(tmp_path, "demo", build_context("2024-01-15"))

    assert picked.lineage["rows_dropped_by_as_of"] > 0
    assert picked.nav_wide.index.max() <= pd.Timestamp("2024-01-05")


def _lock_universe(tmp_path: Path, research_date: str, name: str) -> None:
    """A saved allocation plus the pool its products were screened from."""

    from backend.product_pools.repository import InvestableUniverseRepository

    InvestableUniverseRepository(tmp_path / "product_pools.json").create(
        {"id": "universe-1", "name": name, "research_date": research_date, "members": []}
    )
    pd.DataFrame(
        [
            {
                "asset_alloc_name": "demo",
                "asset_name": "ClassA",
                "etf_code": "A",
                "etf_name": "a",
                "etf_weight": 100.0,
                "universe_snapshot_id": "universe-1",
            }
        ]
    ).to_parquet(tmp_path / "asset_alloc_info.parquet", index=False)


def test_the_pool_behind_an_allocation_is_judged_not_only_its_nav(tmp_path) -> None:
    # The series was rebuilt for 2023-12-01, so on its own it is clean. The pool
    # its products were screened from was cut two years later, and until the
    # loader read that column back nothing downstream could see the difference.
    _write_alloc(tmp_path, lag_days=0, as_of="2023-12-01")
    _lock_universe(tmp_path, "2026-01-01", "2026筛出来的池子")

    loaded = engine.load_allocation_nav(tmp_path, "demo", build_context("2024-02-20"))
    universe = loaded.lineage["universe"]

    assert universe["established_at"] == "2023-12-01"
    assert universe["snapshot_established_at"] == "2026-01-01"
    assert [item["code"] for item in universe["findings"]] == [UNIVERSE_LOOKAHEAD]
    assert "2026筛出来的池子" in universe["findings"][0]["message"]
    with pytest.raises(PitContextError):
        engine.load_allocation_nav(tmp_path, "demo", build_context("2024-02-20", "STRICT_PIT"))


def test_a_pool_cut_before_the_decision_leaves_the_allocation_clean(tmp_path) -> None:
    _write_alloc(tmp_path, lag_days=0, as_of="2023-12-01")
    _lock_universe(tmp_path, "2023-11-01", "当时就定好的池子")

    loaded = engine.load_allocation_nav(tmp_path, "demo", build_context("2024-02-20"))

    assert loaded.lineage["universe"]["clean"] is True


def test_an_allocation_naming_no_pool_says_unknown_rather_than_clean(tmp_path) -> None:
    _write_alloc(tmp_path, lag_days=0, as_of="2023-12-01")

    loaded = engine.load_allocation_nav(tmp_path, "demo", build_context("2024-02-20"))

    assert loaded.lineage["universe_snapshot_id"] is None
    assert loaded.lineage["universe"]["snapshot_established_at"] is None


# --------------------------------------------------------------------------- #
# L4 — the guard
# --------------------------------------------------------------------------- #


def test_pool_screened_after_the_decision_is_flagged() -> None:
    findings = check_universe(
        build_context("2018-06-30"), established_at="2026-01-01", label="产品池"
    )

    assert [item["code"] for item in findings] == [UNIVERSE_LOOKAHEAD]


def test_pool_screened_before_the_decision_is_clean() -> None:
    assert check_universe(build_context("2026-01-01"), established_at="2018-06-30") == []


def test_non_replayable_universe_is_reported() -> None:
    findings = check_universe(build_context("2018-06-30"), coverage=LATEST_ONLY)

    assert [item["code"] for item in findings] == [UNIVERSE_NOT_REPLAYABLE]


def test_strict_mode_raises_where_research_mode_records() -> None:
    research = build_context("2018-06-30")
    strict = build_context("2018-06-30", "STRICT_PIT")

    assert assert_no_universe_lookahead(research, established_at="2026-01-01")
    with pytest.raises(PitContextError):
        assert_no_universe_lookahead(strict, established_at="2026-01-01")


# --------------------------------------------------------------------------- #
# replay: an entity absent from a later snapshot must stay absent
# --------------------------------------------------------------------------- #


def test_replay_uses_the_last_snapshot_not_the_union(tmp_path) -> None:
    (tmp_path / "pit_dim").mkdir()
    early = pd.DataFrame({"ts_code": ["A.SH", "GONE.SH"], "name": ["甲", "已退市"]})
    early[SNAPSHOT_FIELD] = pd.Timestamp("2024-01-02")
    late = pd.DataFrame({"ts_code": ["A.SH"], "name": ["甲"]})
    late[SNAPSHOT_FIELD] = pd.Timestamp("2024-06-01")
    pd.concat([early, late], ignore_index=True).to_parquet(
        tmp_path / "pit_dim" / "etf_info_df_history.parquet", index=False
    )
    pd.DataFrame({"ts_code": ["A.SH"], "name": ["甲"]}).to_parquet(
        tmp_path / "etf_info_df.parquet", index=False
    )

    on_january = read_pit("etf_info", tmp_path, build_context("2024-03-01"))
    on_july = read_pit("etf_info", tmp_path, build_context("2024-07-01"))

    assert set(on_january.frame["ts_code"]) == {"A.SH", "GONE.SH"}
    # Resurrecting GONE.SH from the January snapshot would be the survivorship
    # error in reverse — the July table simply does not contain it.
    assert set(on_july.frame["ts_code"]) == {"A.SH"}


# --------------------------------------------------------------------------- #
# 产品池回放
# --------------------------------------------------------------------------- #


def test_pool_version_replays_to_another_research_day(tmp_path) -> None:
    from tests.test_product_pools import FakeEvaluationGateway, _approve_all
    from product_pools.repository import ProductPoolRepository
    from product_pools.service import ProductPoolService

    gateway = FakeEvaluationGateway()
    service = ProductPoolService(ProductPoolRepository(tmp_path / "pools.json"), gateway)
    pool = service.create_pool({"name": "核心池", "description": "", "purpose": "配置", "owner": "Kevin"})
    pool = service.attach_evaluation_plan(
        pool["id"], pool["revision"], {"plan_id": "plan-equity", "selection_mode": "all_ranked"}
    )
    pool = _approve_all(service, pool)
    version = service.publish_pool(
        pool["id"], pool["revision"], {"effective_from": "2026-09-01", "publication_note": "首版"}
    )["version"]

    # Standing on 2018, the screen would only have ranked one of the two.
    gateway.runs["plan-equity"]["rows"] = [
        {"kind": "etf", "product_id": "510300.SH", "code": "510300.SH", "name": "沪深300ETF", "rank": 1, "score": 90.0},
        {"kind": "etf", "product_id": "OLD.SH", "code": "OLD.SH", "name": "老基金", "rank": 2, "score": 80.0},
    ]
    replayed = service.replay_version(version["id"], "2018-06-30")

    assert replayed["lookahead"] is True
    assert [item["code"] for item in replayed["added"]] == ["OLD.SH"]
    assert {item["product_id"] for item in replayed["removed"]} == {"510500.SH"}
    assert replayed["summary"]["kept"] == 1
