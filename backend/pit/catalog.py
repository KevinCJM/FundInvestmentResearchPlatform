"""Point-in-time declarations for every research dataset.

A dataset carries up to three clocks: when the thing happened (event time), when
we could first *see* it (availability time) and which vintage we are looking at
(release time).  Research that filters on event time when availability time is
what it can actually observe is look-ahead, silently.

This module is the declaration layer only: which column is which, and how a
declaration plus a measurement turns into a grade.  Nothing here touches the
filesystem, so it can be reasoned about and tested on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


GRADE_STRICT = "A"
GRADE_APPROXIMATE = "B"
GRADE_NONE = "C"

GRADE_LABELS = {
    GRADE_STRICT: "A · 严格 PIT",
    GRADE_APPROXIMATE: "B · 近似 PIT",
    GRADE_NONE: "C · 无 PIT",
}

GRADE_DESCRIPTIONS = {
    GRADE_STRICT: "有真实公告/发布时间且覆盖完整，可用于正式回测与实盘信号。",
    GRADE_APPROXIMATE: "没有公告时间，但数据不会被回溯修订；按事件日加保守滞后估计可得时间。",
    GRADE_NONE: "会被回溯修订且没有可得时间列，历史无法还原，严格 PIT 模式下禁止使用。",
}

# A declared availability column has to actually be there. Below this share of
# rows the column cannot carry a strict claim, so the dataset drops to B.
STRICT_COVERAGE_FLOOR = 0.995

# Column carrying the vintage clock in an append-only dimension snapshot log.
SNAPSHOT_FIELD = "pit_snapshot_date"

RUN_MODE_RESEARCH = "RESEARCH"
RUN_MODE_STRICT = "STRICT_PIT"
RUN_MODES = {
    RUN_MODE_RESEARCH: "研究模式",
    RUN_MODE_STRICT: "严格 PIT",
}


@dataclass(frozen=True)
class DatasetPitDeclaration:
    """What we claim about one dataset's clocks, before measuring anything."""

    dataset_id: str
    label: str
    file: str
    event_field: str
    # None means the source gives us no publication timestamp at all.
    availability_field: Optional[str]
    # Conservative publication lag in calendar days, used when there is no
    # availability column. 0 means "observable at the close of the event day".
    declared_lag_days: int
    # True when the provider rewrites history in place (restated financials,
    # revised macro prints). Such a dataset can never be strictly PIT without a
    # vintage column, because yesterday's value is simply gone.
    revisable: bool
    note: str
    # Append-only snapshot log for a revisable table: every refresh writes the
    # whole table again under a `snapshot_date`, so the state as known on any
    # past day can be replayed. None means the table is overwrite-only and its
    # history is simply gone.
    history_file: Optional[str] = None


DATASETS: tuple[DatasetPitDeclaration, ...] = (
    DatasetPitDeclaration(
        dataset_id="etf_nav",
        label="ETF / 场内基金净值",
        file="etf_daily_df.parquet",
        event_field="nav_date",
        availability_field="ann_date",
        declared_lag_days=1,
        revisable=False,
        note="Tushare 提供公告日；公募净值通常 T+1 公告，节假日与新发基金尾部可达 20 天以上。",
    ),
    DatasetPitDeclaration(
        dataset_id="fund_nav",
        label="场外公募基金净值",
        file="fund_nav_df.parquet",
        event_field="nav_date",
        availability_field="ann_date",
        declared_lag_days=1,
        revisable=False,
        note="与场内净值同源同口径；该文件缺失时本行不计入体检。",
    ),
    DatasetPitDeclaration(
        dataset_id="etf_candle",
        label="ETF 二级市场行情",
        file="etf_daily_candle_df.parquet",
        event_field="trade_date",
        availability_field=None,
        declared_lag_days=0,
        revisable=False,
        note="收盘行情当日收盘后即可得，且不回溯修订；按事件日当天可得处理。",
    ),
    DatasetPitDeclaration(
        dataset_id="etf_share_size",
        label="ETF 份额与规模",
        file="etf_share_size_df.parquet",
        event_field="trade_date",
        availability_field=None,
        declared_lag_days=0,
        revisable=False,
        note="份额变动随日行情披露，不回溯修订；按事件日当天可得处理。规模＝份额 × 同排单位净值。",
    ),
    DatasetPitDeclaration(
        dataset_id="trade_calendar",
        label="交易日历",
        file="trade_day_df.parquet",
        event_field="cal_date",
        availability_field=None,
        declared_lag_days=0,
        revisable=False,
        note="交易所提前发布，研究期内视为始终已知。",
    ),
    DatasetPitDeclaration(
        dataset_id="etf_info",
        label="ETF 合同与分类信息",
        file="etf_info_df.parquet",
        # `list_date` is when the fund became investable, which is also the
        # earliest day this row could have been observed at all.
        event_field="list_date",
        availability_field=None,
        declared_lag_days=0,
        revisable=True,
        note="维表按最新状态整体覆盖写入，没有生效时间也没有历史版本；用它做历史分类会带入今天的信息。",
        history_file="pit_dim/etf_info_df_history.parquet",
    ),
    DatasetPitDeclaration(
        dataset_id="fund_info",
        label="场外基金合同与分类信息",
        file="fund_info_df.parquet",
        event_field="list_date",
        availability_field=None,
        declared_lag_days=0,
        revisable=True,
        note="与场内合同信息同源同口径的最新态维表；分类改口、清盘转型都会就地覆盖。",
        history_file="pit_dim/fund_info_df_history.parquet",
    ),
    DatasetPitDeclaration(
        dataset_id="index_info",
        label="指数基础信息",
        file="index_info.parquet",
        event_field="list_date",
        availability_field=None,
        declared_lag_days=0,
        revisable=True,
        note="同为最新态维表；指数更名与口径调整会就地覆盖。",
        history_file="pit_dim/index_info_history.parquet",
    ),
    DatasetPitDeclaration(
        dataset_id="etf_index",
        label="ETF 跟踪指数映射",
        file="etf_index.parquet",
        event_field="pub_date",
        availability_field=None,
        declared_lag_days=0,
        revisable=True,
        note="映射关系就地更新，换标的后查不到历史对应关系。",
        history_file="pit_dim/etf_index_history.parquet",
    ),
    DatasetPitDeclaration(
        dataset_id="stock_basic",
        label="股票基础信息",
        file="stock_basic.parquet",
        event_field="list_date",
        availability_field=None,
        declared_lag_days=0,
        revisable=True,
        note="含退市状态的最新态维表；缺历史版本时无法还原当时的可选股票域。",
        history_file="pit_dim/stock_basic_history.parquet",
    ),
    DatasetPitDeclaration(
        dataset_id="asset_nv",
        label="大类配置净值",
        file="asset_nv.parquet",
        event_field="date",
        availability_field="creat_time",
        declared_lag_days=0,
        revisable=False,
        note="平台自算序列，创建时间即可得时间。滞后天数按构造就很大——2026 年算出的配置净值，2018 年确实不可得。",
    ),
)

DATASETS_BY_ID = {declaration.dataset_id: declaration for declaration in DATASETS}


def grade(
    declaration: DatasetPitDeclaration,
    availability_coverage: Optional[float],
    *,
    history_snapshots: int = 0,
) -> str:
    """Combine what the dataset claims with what the file actually contains.

    `availability_coverage` is the measured share of rows carrying a usable
    availability timestamp, or None when the dataset declares no such column.
    A declaration alone never earns an A: the column has to be there.

    `history_snapshots` is how many dated versions of a revisable table are on
    disk. Two or more means the table's past state can be replayed instead of
    guessed, which is what lifts an overwrite-only dimension out of grade C —
    but only from the first snapshot onwards, so it never earns an A either.
    """

    if declaration.availability_field:
        if availability_coverage is None:
            # Declared but not measurable (file absent, column missing): the
            # claim is unverified, so it does not get to keep the A.
            return GRADE_NONE if declaration.revisable else GRADE_APPROXIMATE
        if availability_coverage >= STRICT_COVERAGE_FLOOR and not declaration.revisable:
            return GRADE_STRICT
        return GRADE_NONE if declaration.revisable else GRADE_APPROXIMATE
    if declaration.revisable:
        return GRADE_APPROXIMATE if history_snapshots > 0 else GRADE_NONE
    return GRADE_APPROXIMATE


def is_usable_under(run_mode: str, dataset_grade: str) -> bool:
    """Strict mode refuses grade C outright; research mode warns instead."""

    if run_mode == RUN_MODE_STRICT:
        return dataset_grade in {GRADE_STRICT, GRADE_APPROXIMATE}
    return True


__all__ = [
    "DATASETS",
    "DATASETS_BY_ID",
    "GRADE_APPROXIMATE",
    "GRADE_DESCRIPTIONS",
    "GRADE_LABELS",
    "GRADE_NONE",
    "GRADE_STRICT",
    "RUN_MODES",
    "RUN_MODE_RESEARCH",
    "RUN_MODE_STRICT",
    "SNAPSHOT_FIELD",
    "STRICT_COVERAGE_FLOOR",
    "DatasetPitDeclaration",
    "grade",
    "is_usable_under",
]
