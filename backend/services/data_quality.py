"""Fast, read-only quality report for the currently active market-data snapshot.

The expensive history scan happens while the analytics snapshot is built and while
a full data version is promoted.  This module only aggregates those persisted facts
so opening the quality page never rescans multi-gigabyte parquet histories.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet

try:
    from backend.compute_policy import validate_execution_audit
    from backend.instrument_analytics_numba import (
        count_true_kernel,
        coverage_ratio_kernel,
        data_quality_masks_kernel,
        encoded_category_counts_kernel,
        encoded_unique_count_kernel,
        instrument_analytics_numba_execution_audit,
        numeric_comparison_mask_kernel,
        positive_finite_sum_kernel,
    )
    from backend.market_data import (
        ACTIVE_MANIFEST_NAME,
        MarketDataManifestError,
        read_active_manifest,
        resolve_tushare_data_dir,
    )
    from backend.series_quality import (
        ADJ_NAV_DISLOCATION_THRESHOLD,
        EXTREME_ADJ_NAV_RETURN_THRESHOLD,
        MAX_CONSECUTIVE_MISSING_OPEN_DAYS,
        MIN_OPEN_DAY_COVERAGE,
        REFERENCE_NAV_STABLE_THRESHOLD,
    )
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from compute_policy import validate_execution_audit
    from instrument_analytics_numba import (
        count_true_kernel,
        coverage_ratio_kernel,
        data_quality_masks_kernel,
        encoded_category_counts_kernel,
        encoded_unique_count_kernel,
        instrument_analytics_numba_execution_audit,
        numeric_comparison_mask_kernel,
        positive_finite_sum_kernel,
    )
    from market_data import (
        ACTIVE_MANIFEST_NAME,
        MarketDataManifestError,
        read_active_manifest,
        resolve_tushare_data_dir,
    )
    from series_quality import (
        ADJ_NAV_DISLOCATION_THRESHOLD,
        EXTREME_ADJ_NAV_RETURN_THRESHOLD,
        MAX_CONSECUTIVE_MISSING_OPEN_DAYS,
        MIN_OPEN_DAY_COVERAGE,
        REFERENCE_NAV_STABLE_THRESHOLD,
    )


METRICS_FILENAME = "instrument_metrics_snapshot.parquet"
STALE_ACTIVE_DAYS = 7
MIN_NAV_CODE_COVERAGE = 0.99
SUPPORTED_KINDS = {"etf", "fund"}
REQUIRED_METRIC_COLUMNS = {
    "instrument_type",
    "ts_code",
    "latest_date",
    "stale_days",
    "latest_adj_nav",
    "adj_nav_anomaly_count",
    "quality_reason_1y",
    "nav_source_fingerprint",
}
QUALITY_REASON_LABELS = {
    "internal_gap": "连续交易日缺口",
    "insufficient_density": "有效观测密度不足",
    "insufficient_span": "历史跨度不足",
    "start_anchor_too_old": "区间起点附近缺少观测",
    "insufficient_observations": "有效观测不足",
    "adjusted_nav_anomaly": "净值突变",
}


def _execution_audit() -> dict[str, object]:
    return validate_execution_audit(instrument_analytics_numba_execution_audit())


def _ratio(numerator: int, denominator: int) -> float | None:
    value = float(coverage_ratio_kernel(int(numerator), int(denominator)))
    return value if np.isfinite(value) else None


def _encoded_counts(values: Iterable[str], categories: tuple[str, ...]) -> dict[str, int]:
    category_by_name = {name: index for index, name in enumerate(categories)}
    codes = np.ascontiguousarray(
        np.array([category_by_name.get(str(value), -1) for value in values], dtype=np.int64)
    )
    counts = encoded_category_counts_kernel(codes, len(categories))
    return {name: int(counts[index]) for index, name in enumerate(categories)}


def _unique_text_count(values: Iterable[object]) -> int:
    code_by_value: dict[str, int] = {}
    codes: list[int] = []
    for raw_value in values:
        value = str(raw_value)
        code = code_by_value.get(value)
        if code is None:
            code = len(code_by_value)
            code_by_value[value] = code
        codes.append(code)
    return int(
        encoded_unique_count_kernel(
            np.ascontiguousarray(np.array(codes, dtype=np.int64))
        )
    )


def _unique_product_count(rows: pd.DataFrame) -> int:
    return _unique_text_count(
        f"{instrument_type}\0{ts_code}"
        for instrument_type, ts_code in rows[["instrument_type", "ts_code"]].itertuples(
            index=False, name=None
        )
    )


def _source_fingerprint(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _column(frame: pd.DataFrame, name: str, default: object = None) -> pd.Series:
    if name in frame:
        return frame[name]
    return pd.Series(default, index=frame.index, dtype="object")


def _numeric_column(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(_column(frame, name, default), errors="coerce")


def _active_manifest(base_dir: Path, active_dir: Path) -> Mapping[str, Any]:
    try:
        payload = read_active_manifest(base_dir)
    except MarketDataManifestError:
        return {}
    if not payload:
        return {}
    candidate = (base_dir / str(payload.get("snapshot_dir", ""))).resolve()
    return payload if candidate == active_dir else {}


def _read_info(active_dir: Path) -> tuple[pd.DataFrame, set[str]]:
    frames: list[pd.DataFrame] = []
    unavailable_kinds: set[str] = set()
    for kind in sorted(SUPPORTED_KINDS):
        path = active_dir / f"{kind}_info_df.parquet"
        if not path.exists():
            unavailable_kinds.add(kind)
            continue
        available = set(parquet.ParquetFile(path).schema.names)
        columns = [column for column in ("ts_code", "name", "status_code") if column in available]
        if "ts_code" not in columns:
            unavailable_kinds.add(kind)
            continue
        frame = pd.read_parquet(path, columns=columns)
        frame["instrument_type"] = kind
        frame["ts_code"] = frame["ts_code"].fillna("").astype(str).str.strip()
        for column in ("name", "status_code"):
            if column not in frame:
                frame[column] = None
        frames.append(frame[["instrument_type", "ts_code", "name", "status_code"]])
    if not frames:
        return (
            pd.DataFrame(columns=["instrument_type", "ts_code", "name", "status_code"]),
            unavailable_kinds,
        )
    return (
        pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["instrument_type", "ts_code"], keep="last"
        ),
        unavailable_kinds,
    )


def _sample_products(
    rows: pd.DataFrame,
    *,
    value_column: str | None = None,
    value_suffix: str = "",
    limit: int = 5,
) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for row in rows.head(limit).itertuples(index=False):
        raw_value = getattr(row, value_column, None) if value_column else None
        if value_column == "quality_reason_1y":
            observed = QUALITY_REASON_LABELS.get(str(raw_value), str(raw_value))
        elif raw_value is not None and not pd.isna(raw_value):
            observed = f"{int(raw_value)}{value_suffix}"
        else:
            observed = ""
        samples.append(
            {
                "kind": str(getattr(row, "instrument_type", "")),
                "ts_code": str(getattr(row, "ts_code", "")),
                "name": str(getattr(row, "name", "") or ""),
                "latest_date": (
                    pd.Timestamp(getattr(row, "latest_date")).strftime("%Y-%m-%d")
                    if pd.notna(getattr(row, "latest_date", None))
                    else ""
                ),
                "observed": observed,
            }
        )
    return samples


def _empty_report(*, generated_at: str | None, activated_at: str | None) -> dict[str, Any]:
    checks = [
        ("schema", "结构契约", "validity"),
        ("primary_key", "主键唯一性", "integrity"),
        ("field_validity", "字段合法性", "validity"),
        ("source_consistency", "源快照一致性", "consistency"),
        ("nav_discontinuity", "净值突变", "continuity"),
        ("series_density", "序列连续性", "continuity"),
        ("window_coverage", "研究窗口完整性", "completeness"),
        ("freshness", "存续产品及时性", "timeliness"),
        ("catalog_coverage", "产品覆盖", "completeness"),
    ]
    return {
        "schema_version": 1,
        "status": "unavailable",
        "generated_at": generated_at,
        "activated_at": activated_at,
        "as_of": None,
        "summary": {
            "checks_total": len(checks), "checks_passed": 0, "checks_warning": 0,
            "checks_failed": 0, "checks_unavailable": len(checks), "total_products": 0,
            "affected_products": 0, "affected_rate": None, "issue_count": 1,
            "critical_issue_count": 0, "high_issue_count": 1, "medium_issue_count": 0,
            "nav_anomaly_products": 0, "nav_anomaly_events": 0, "stale_active_products": 0,
        },
        "checks": [
            {
                "key": key, "label": label, "dimension": dimension, "status": "unavailable",
                "summary": "缺少深度质量快照", "detail": "需要先生成研究指标快照。", "threshold": None,
            }
            for key, label, dimension in checks
        ],
        "issues": [{
            "id": "metrics-snapshot-missing", "code": "METRICS_SNAPSHOT_MISSING",
            "severity": "high", "dimension": "consistency", "scope": "研究指标快照",
            "title": "深度质量快照不可用", "description": "未找到可读取的研究指标快照，净值突变、缺口和陈旧产品尚未完成检查。",
            "evidence": "instrument_metrics_snapshot.parquet 缺失或无法读取。",
            "impact": "不能对净值序列质量给出通过结论。", "affected_count": 0,
            "affected_rate": None, "record_count": 0, "samples": [], "action": "rebuild_analytics",
        }],
        "validation": {"status": "unavailable"},
        "execution": _execution_audit(),
    }


def build_data_quality_report(base_dir: Path) -> dict[str, Any]:
    """Aggregate persisted validation and series-quality facts for the active version."""

    base = Path(base_dir).expanduser().resolve()
    active_dir = resolve_tushare_data_dir(base)
    manifest = _active_manifest(base, active_dir)
    activated_at = str(manifest.get("activated_at")) if manifest.get("activated_at") else None
    metrics_path = active_dir / METRICS_FILENAME
    if not metrics_path.exists():
        return _empty_report(generated_at=None, activated_at=activated_at)
    generated_at = datetime.fromtimestamp(
        metrics_path.stat().st_mtime, timezone.utc
    ).isoformat()
    try:
        available_columns = set(parquet.ParquetFile(metrics_path).schema.names)
        frame = pd.read_parquet(metrics_path)
    except Exception:  # noqa: BLE001 - public report must fail closed without leaking internals
        return _empty_report(generated_at=generated_at, activated_at=activated_at)
    if frame.empty:
        return _empty_report(generated_at=generated_at, activated_at=activated_at)

    unavailable_info_kinds: set[str] = set()
    try:
        info, unavailable_info_kinds = _read_info(active_dir)
    except Exception:  # noqa: BLE001 - keep other quality dimensions inspectable
        info = pd.DataFrame(columns=["instrument_type", "ts_code", "name", "status_code"])
        unavailable_info_kinds = set(SUPPORTED_KINDS)
    for column in ("instrument_type", "ts_code"):
        frame[column] = _column(frame, column, "").fillna("").astype(str).str.strip()
    frame["latest_date"] = pd.to_datetime(_column(frame, "latest_date"), errors="coerce")
    frame = frame.merge(info, on=["instrument_type", "ts_code"], how="left")
    total_products = int(len(frame))
    issues: list[dict[str, Any]] = []
    product_code_by_key: dict[tuple[str, str], int] = {}
    for instrument_type, ts_code in frame[["instrument_type", "ts_code"]].itertuples(
        index=False, name=None
    ):
        key = (str(instrument_type), str(ts_code))
        if key not in product_code_by_key:
            product_code_by_key[key] = len(product_code_by_key)
    affected_mask = np.zeros(len(product_code_by_key), dtype=np.uint8)

    def add_issue(
        *,
        code: str,
        severity: str,
        dimension: str,
        scope: str,
        title: str,
        description: str,
        evidence: str,
        impact: str,
        rows: pd.DataFrame | None = None,
        samples: list[dict[str, str]] | None = None,
        record_count: int = 0,
        action: str = "inspect",
    ) -> None:
        affected_count = 0
        if rows is not None and not rows.empty:
            issue_mask = np.zeros(len(product_code_by_key), dtype=np.uint8)
            for instrument_type, ts_code in rows[["instrument_type", "ts_code"]].itertuples(
                index=False, name=None
            ):
                code_value = product_code_by_key.get(
                    (str(instrument_type), str(ts_code))
                )
                if code_value is not None:
                    issue_mask[code_value] = 1
                    affected_mask[code_value] = 1
            affected_count = int(count_true_kernel(np.ascontiguousarray(issue_mask)))
        issues.append({
            "id": code.lower().replace("_", "-"), "code": code, "severity": severity,
            "dimension": dimension, "scope": scope, "title": title, "description": description,
            "evidence": evidence, "impact": impact, "affected_count": affected_count,
            "affected_rate": _ratio(affected_count, total_products),
            "record_count": int(record_count), "samples": samples or [], "action": action,
        })

    validation = _as_mapping(manifest.get("validation"))
    validation_passed = validation.get("status") == "passed"
    validation_datasets = _as_mapping(validation.get("datasets"))
    if not validation_passed:
        add_issue(
            code="SNAPSHOT_VALIDATION_EVIDENCE_MISSING", severity="high", dimension="integrity",
            scope="已生效数据版本", title="缺少完整验收记录",
            description="当前数据目录没有可核验的全量验收记录，不能证明结构、主键和非法值检查已经通过。",
            evidence=f"未找到 {ACTIVE_MANIFEST_NAME} 中的 passed validation。",
            impact="研究结果可能来自未经过原子激活门槛的数据版本。", action="refresh",
        )
    if unavailable_info_kinds:
        unavailable_label = "、".join(kind.upper() for kind in sorted(unavailable_info_kinds))
        add_issue(
            code="INFO_MAPPING_UNAVAILABLE", severity="high", dimension="availability",
            scope="产品基础信息", title="产品状态与名称映射不可用",
            description="无法读取产品基础信息，存续状态及时性检查已停止。",
            evidence=f"不可用产品类型：{unavailable_label}。",
            impact="不能判断陈旧净值是否属于仍在存续的产品。", action="refresh",
        )

    missing_columns = REQUIRED_METRIC_COLUMNS - available_columns
    if missing_columns:
        add_issue(
            code="METRICS_SCHEMA_MISMATCH", severity="critical", dimension="validity",
            scope="研究指标快照", title="研究指标快照结构不完整",
            description="深度质量检查所需字段缺失。",
            evidence=f"缺少字段：{', '.join(sorted(missing_columns))}。",
            impact="部分质量结论无法计算，使用该快照可能产生错误解释。", action="rebuild_analytics",
        )

    duplicate_mask = frame.duplicated(subset=["instrument_type", "ts_code"], keep=False)
    duplicate_rows = frame.loc[duplicate_mask].sort_values(["instrument_type", "ts_code"])
    if not duplicate_rows.empty:
        duplicate_product_count = _unique_product_count(duplicate_rows)
        add_issue(
            code="DUPLICATE_METRIC_KEY", severity="critical", dimension="integrity",
            scope="研究指标快照", title="产品指标主键重复",
            description="同一产品在指标快照中出现多条记录。",
            evidence=f"发现 {duplicate_product_count} 个重复产品键。",
            impact="筛选、排行和指标展示可能随机读取不同记录。", rows=duplicate_rows,
            samples=_sample_products(duplicate_rows.drop_duplicates(["instrument_type", "ts_code"])),
            record_count=int(
                count_true_kernel(
                    np.ascontiguousarray(duplicate_mask.to_numpy(dtype=np.uint8))
                )
            ), action="rebuild_analytics",
        )

    unsupported = frame[~frame["instrument_type"].isin(SUPPORTED_KINDS)]
    if not unsupported.empty:
        unsupported_type_count = _unique_text_count(unsupported["instrument_type"])
        add_issue(
            code="UNSUPPORTED_INSTRUMENT_TYPE", severity="critical", dimension="validity",
            scope="研究指标快照", title="出现未知产品类型",
            description="指标快照包含系统无法识别的产品类型。",
            evidence=f"涉及 {unsupported_type_count} 个未知类型。",
            impact="产品分组和质量统计可能失真。", rows=unsupported,
            samples=_sample_products(unsupported), action="rebuild_analytics",
        )

    latest_nav = _numeric_column(frame, "latest_adj_nav")
    anomaly_values = _numeric_column(frame, "adj_nav_anomaly_count", 0).fillna(0)
    stale_values = _numeric_column(frame, "stale_days", 0).fillna(0)
    active_mask = _column(frame, "status_code", "").fillna("").astype(str).str.upper().eq("L")
    future_limit = pd.Timestamp(datetime.now(timezone.utc).date() + timedelta(days=1))
    latest_date_ns = np.ascontiguousarray(
        frame["latest_date"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    )
    (
        missing_nav_mask,
        invalid_nav_mask,
        future_date_mask,
        anomaly_mask,
        stale_active_mask,
    ) = data_quality_masks_kernel(
        np.ascontiguousarray(latest_nav.to_numpy(dtype=np.float64)),
        latest_date_ns,
        np.ascontiguousarray(anomaly_values.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(active_mask.to_numpy(dtype=np.uint8)),
        np.ascontiguousarray(stale_values.to_numpy(dtype=np.float64)),
        np.int64(future_limit.value),
        np.float64(STALE_ACTIVE_DAYS),
    )
    missing_latest_nav_count = int(count_true_kernel(missing_nav_mask))
    invalid_nav_count = int(count_true_kernel(invalid_nav_mask))
    future_date_count = int(count_true_kernel(future_date_mask))
    anomaly_product_count = int(count_true_kernel(anomaly_mask))
    stale_active_count = int(count_true_kernel(stale_active_mask))

    missing_latest_nav = frame.loc[missing_nav_mask.astype(bool)]
    if not missing_latest_nav.empty:
        add_issue(
            code="LATEST_NAV_MISSING", severity="high", dimension="validity",
            scope="产品最新净值", title="部分产品缺少可用最新净值",
            description="产品已进入研究指标快照，但最新复权净值为空或无法解析。",
            evidence=f"发现 {missing_latest_nav_count} 个产品缺少最新复权净值。",
            impact="这些产品不能形成当前收益、风险和回撤结论。", rows=missing_latest_nav,
            samples=_sample_products(missing_latest_nav), action="refresh",
        )
    invalid_nav = frame.loc[invalid_nav_mask.astype(bool)]
    if not invalid_nav.empty:
        add_issue(
            code="INVALID_NAV_VALUE", severity="critical", dimension="validity",
            scope="产品净值", title="存在非法净值",
            description="最新复权净值包含非有限值、零或负数。",
            evidence=f"发现 {invalid_nav_count} 条非法最新净值。",
            impact="收益率、波动率和回撤计算不再可信。", rows=invalid_nav,
            samples=_sample_products(invalid_nav), record_count=invalid_nav_count, action="refresh",
        )

    future_rows = frame.loc[future_date_mask.astype(bool)]
    if not future_rows.empty:
        add_issue(
            code="FUTURE_DATED_SERIES", severity="critical", dimension="validity",
            scope="产品净值", title="出现未来日期记录",
            description="产品净值日期晚于允许的时区容差。",
            evidence=f"发现 {future_date_count} 个产品的最新日期晚于 {future_limit.date().isoformat()}。",
            impact="PIT 研究可能发生时间穿越。", rows=future_rows,
            samples=_sample_products(future_rows), action="refresh",
        )

    anomaly_rows = frame.loc[anomaly_mask.astype(bool)].assign(
        adj_nav_anomaly_count=anomaly_values[anomaly_mask.astype(bool)].astype(int)
    ).sort_values("adj_nav_anomaly_count", ascending=False)
    anomaly_events = int(
        positive_finite_sum_kernel(
            np.ascontiguousarray(anomaly_values.to_numpy(dtype=np.float64))
        )
    )
    if not anomaly_rows.empty:
        add_issue(
            code="NAV_DISCONTINUITY", severity="high", dimension="continuity",
            scope="ETF 与场外基金净值", title="发现净值突变或复权断点",
            description="复权净值发生大幅跳变，但单位净值或累计净值没有同步变化，或单日变化超过极端阈值。",
            evidence=f"{anomaly_product_count} 个产品共发现 {anomaly_events} 个异常点。",
            impact="异常点所在区间的收益、波动率、回撤和风险调整指标会被判为不可用。",
            rows=anomaly_rows,
            samples=_sample_products(anomaly_rows, value_column="adj_nav_anomaly_count", value_suffix=" 个异常点"),
            record_count=anomaly_events, action="inspect_source",
        )

    reason = _column(frame, "quality_reason_1y", "").fillna("")
    gap_rows = frame[reason.isin({"internal_gap", "insufficient_density"})].assign(
        quality_reason_1y=reason[reason.isin({"internal_gap", "insufficient_density"})]
    ).sort_values(["instrument_type", "ts_code"])
    gap_count = int(
        count_true_kernel(
            np.ascontiguousarray(
                reason.isin({"internal_gap", "insufficient_density"}).to_numpy(dtype=np.uint8)
            )
        )
    )
    if not gap_rows.empty:
        add_issue(
            code="SERIES_INTERNAL_GAP", severity="high", dimension="continuity",
            scope="近一年产品净值", title="净值序列存在密度不足或连续缺口",
            description="近一年窗口的有效观测覆盖不足，或连续缺失交易日超过规则上限。",
            evidence=f"{gap_count} 个产品未通过近一年连续性检查。",
            impact="区间收益与风险指标会被排除，避免把缺数误当成零收益。", rows=gap_rows,
            samples=_sample_products(gap_rows, value_column="quality_reason_1y"), action="refresh",
        )

    short_reasons = {"insufficient_span", "start_anchor_too_old", "insufficient_observations"}
    short_rows = frame[reason.isin(short_reasons)].assign(
        quality_reason_1y=reason[reason.isin(short_reasons)]
    ).sort_values(["instrument_type", "ts_code"])
    short_count = int(
        count_true_kernel(
            np.ascontiguousarray(reason.isin(short_reasons).to_numpy(dtype=np.uint8))
        )
    )
    if not short_rows.empty:
        add_issue(
            code="RESEARCH_WINDOW_INCOMPLETE", severity="medium", dimension="completeness",
            scope="近一年研究窗口", title="部分产品没有完整近一年研究窗口",
            description="这可能是新成立产品的正常现象，也可能是历史数据没有补齐，需要结合成立日期判断。",
            evidence=f"{short_count} 个产品无法形成完整近一年窗口。",
            impact="这些产品不会进入依赖完整近一年数据的排行和正式比较。", rows=short_rows,
            samples=_sample_products(short_rows, value_column="quality_reason_1y"), action="inspect",
        )

    stale_rows = frame.loc[stale_active_mask.astype(bool)].assign(
        stale_days=stale_values[stale_active_mask.astype(bool)].astype(int)
    ).sort_values("stale_days", ascending=False)
    if not stale_rows.empty:
        add_issue(
            code="STALE_ACTIVE_SERIES", severity="high", dimension="timeliness",
            scope="存续产品净值", title="存续产品净值长时间未更新",
            description="只检查基础信息标记为存续的产品，已到期或终止产品不会被误报。",
            evidence=f"{stale_active_count} 个存续产品相对同类最新日期滞后超过 {STALE_ACTIVE_DAYS} 天。",
            impact="最新收益和风险指标不能代表当前时点。", rows=stale_rows,
            samples=_sample_products(stale_rows, value_column="stale_days", value_suffix=" 天"), action="refresh",
        )

    missing_value_count = int(
        positive_finite_sum_kernel(
            np.ascontiguousarray(
                np.array(
                    [
                        float(_as_mapping(validation_datasets.get(key)).get("missing_values") or 0)
                        for key in ("etf_nav", "fund_nav")
                    ],
                    dtype=np.float64,
                )
            )
        )
    )
    if missing_value_count:
        add_issue(
            code="NAV_VALUE_MISSING", severity="medium", dimension="validity",
            scope="ETF 与场外基金净值", title="源数据存在空净值记录",
            description="空值记录不会参与指标计算，但需要确认是否为供应商允许的稀疏字段。",
            evidence=f"全量验收记录了 {missing_value_count} 条空复权净值。",
            impact="若空值集中出现，可能缩短研究窗口或形成内部缺口。",
            record_count=missing_value_count, action="inspect_source",
        )

    coverage_by_kind: dict[str, float] = {}
    analytics_validation = _as_mapping(validation_datasets.get("analytics_snapshot"))
    for kind, raw in _as_mapping(analytics_validation.get("nav_code_coverage")).items():
        try:
            coverage_by_kind[str(kind)] = float(raw)
        except (TypeError, ValueError):
            continue
    if not coverage_by_kind:
        for kind in SUPPORTED_KINDS:
            raw = _as_mapping(validation_datasets.get(f"{kind}_nav")).get("code_coverage")
            try:
                coverage_by_kind[kind] = float(raw)
            except (TypeError, ValueError):
                continue
    coverage_kinds = sorted(coverage_by_kind)
    coverage_values = np.ascontiguousarray(
        np.array([coverage_by_kind[kind] for kind in coverage_kinds], dtype=np.float64)
    )
    low_coverage_mask = numeric_comparison_mask_kernel(
        coverage_values,
        np.float64(MIN_NAV_CODE_COVERAGE),
        np.float64(1.0),
        np.int64(3),
    )
    low_coverage = {
        kind: float(coverage_values[index])
        for index, kind in enumerate(coverage_kinds)
        if int(low_coverage_mask[index]) == 1
    }
    if low_coverage:
        evidence = "、".join(f"{kind.upper()} {value:.1%}" for kind, value in sorted(low_coverage.items()))
        add_issue(
            code="NAV_CODE_COVERAGE_LOW", severity="medium", dimension="completeness",
            scope="产品与净值映射", title="产品净值代码覆盖不足",
            description="部分基础信息中的产品代码没有对应净值或没有进入指标快照。",
            evidence=f"代码覆盖：{evidence}；目标不低于 {MIN_NAV_CODE_COVERAGE:.0%}。",
            impact="缺失产品不能参与筛选、比较、组合构建和回测。", action="refresh",
        )

    fingerprint_mismatch_parts: list[pd.DataFrame] = []
    if "nav_source_fingerprint" in frame:
        for kind, filename in (("etf", "etf_daily_df.parquet"), ("fund", "fund_nav_df.parquet")):
            source_path = active_dir / filename
            if not source_path.exists():
                continue
            expected = _source_fingerprint(source_path)
            mask = frame["instrument_type"].eq(kind) & frame["nav_source_fingerprint"].fillna("").astype(str).ne(expected)
            if int(
                count_true_kernel(
                    np.ascontiguousarray(mask.to_numpy(dtype=np.uint8))
                )
            ):
                fingerprint_mismatch_parts.append(frame.loc[mask])
    fingerprint_mismatch = (
        pd.concat(fingerprint_mismatch_parts, ignore_index=True)
        if fingerprint_mismatch_parts else frame.iloc[0:0]
    )
    fingerprint_mismatch_count = _unique_product_count(fingerprint_mismatch)
    if not fingerprint_mismatch.empty:
        add_issue(
            code="METRICS_SOURCE_STALE", severity="critical", dimension="consistency",
            scope="研究指标快照", title="指标快照与净值源版本不一致",
            description="净值文件已变化，但指标快照仍保留旧的源指纹。",
            evidence=f"{fingerprint_mismatch_count} 个产品的指标来源指纹已过期。",
            impact="页面展示的收益和风险指标可能不对应当前已生效净值。", rows=fingerprint_mismatch,
            samples=_sample_products(fingerprint_mismatch), action="rebuild_analytics",
        )

    issue_codes = {str(issue["code"]) for issue in issues}

    def check(
        key: str,
        label: str,
        dimension: str,
        related_codes: Iterable[str],
        summary: str,
        detail: str,
        threshold: str | None,
        *,
        unavailable: bool = False,
    ) -> dict[str, Any]:
        related = [issue for issue in issues if issue["code"] in set(related_codes)]
        status = "unavailable" if unavailable else (
            "failed" if any(issue["severity"] == "critical" for issue in related)
            else "warning" if related else "passed"
        )
        return {
            "key": key, "label": label, "dimension": dimension, "status": status,
            "summary": summary, "detail": detail, "threshold": threshold,
        }

    checks = [
        check("schema", "结构契约", "validity", {"METRICS_SCHEMA_MISMATCH"},
              "必要字段齐全" if not missing_columns else f"缺少 {len(missing_columns)} 个字段",
              "检查产品、日期、净值及质量快照字段是否满足固定契约。", "必要字段必须 100% 存在",
              unavailable=not validation_passed and not missing_columns),
        check("primary_key", "主键唯一性", "integrity", {"DUPLICATE_METRIC_KEY", "SNAPSHOT_VALIDATION_EVIDENCE_MISSING"},
              "产品键无重复" if "DUPLICATE_METRIC_KEY" not in issue_codes else "发现重复产品键",
              "基础信息按产品代码唯一；时间序列按产品代码和日期唯一。", "重复键 = 0",
              unavailable=not validation_passed and "DUPLICATE_METRIC_KEY" not in issue_codes),
        check("field_validity", "字段合法性", "validity", {"LATEST_NAV_MISSING", "INVALID_NAV_VALUE", "FUTURE_DATED_SERIES", "NAV_VALUE_MISSING", "UNSUPPORTED_INSTRUMENT_TYPE"},
              f"{missing_latest_nav_count} 个产品缺最新净值" if missing_latest_nav_count
              else f"{missing_value_count} 条源记录空净值" if missing_value_count else "关键值域正常",
              "检查代码、日期、正净值、有限数值和成交活动值域。", "净值 > 0；日期不晚于明日"),
        check("source_consistency", "源快照一致性", "consistency", {"METRICS_SOURCE_STALE", "SNAPSHOT_VALIDATION_EVIDENCE_MISSING"},
              "指标与净值源指纹一致" if not fingerprint_mismatch_count else f"{fingerprint_mismatch_count} 个指纹过期",
              "防止净值更新后继续展示旧指标。", "源文件大小与修改时间指纹一致"),
        check("nav_discontinuity", "净值突变", "continuity", {"NAV_DISCONTINUITY"},
              f"{anomaly_product_count} 个产品 / {anomaly_events} 个异常点",
              "用单位净值、累计净值交叉确认复权净值跳变。",
              f"复权变动 > {ADJ_NAV_DISLOCATION_THRESHOLD:.0%} 且参考净值变动 ≤ {REFERENCE_NAV_STABLE_THRESHOLD:.0%}；或单日变动 > {EXTREME_ADJ_NAV_RETURN_THRESHOLD:.0%}"),
        check("series_density", "序列连续性", "continuity", {"SERIES_INTERNAL_GAP"},
              f"{gap_count} 个产品需补数" if gap_count else "近一年序列连续",
              "检查交易日覆盖率与连续缺失区间。",
              f"交易日覆盖 ≥ {MIN_OPEN_DAY_COVERAGE:.0%}；连续缺失 ≤ {MAX_CONSECUTIVE_MISSING_OPEN_DAYS} 个交易日"),
        check("window_coverage", "研究窗口完整性", "completeness", {"RESEARCH_WINDOW_INCOMPLETE"},
              f"{short_count} 个产品近一年窗口不足" if short_count else "近一年窗口完整",
              "区分新产品历史不足与已有产品缺数，不将缺失值补零。", "区间起点容差 ≤ 10 天且至少 2 个观测"),
        check("freshness", "存续产品及时性", "timeliness", {"STALE_ACTIVE_SERIES", "INFO_MAPPING_UNAVAILABLE"},
              f"{stale_active_count} 个存续产品陈旧" if stale_active_count else "存续产品更新及时",
              "仅对存续产品比较同类数据最新日期。", f"相对同类最新日期滞后 ≤ {STALE_ACTIVE_DAYS} 天"),
        check("catalog_coverage", "产品覆盖", "completeness", {"NAV_CODE_COVERAGE_LOW"},
              "、".join(f"{kind.upper()} {value:.1%}" for kind, value in sorted(coverage_by_kind.items())) or "暂无覆盖记录",
              "检查基础产品代码是否拥有净值并进入研究指标快照。", f"代码覆盖率 ≥ {MIN_NAV_CODE_COVERAGE:.0%}",
              unavailable=not coverage_by_kind),
    ]

    severity_counts = _encoded_counts(
        (str(issue["severity"]) for issue in issues),
        ("critical", "high", "medium"),
    )
    check_counts = _encoded_counts(
        (str(item["status"]) for item in checks),
        ("passed", "warning", "failed", "unavailable"),
    )
    status = "blocked" if severity_counts["critical"] else "attention" if issues else "healthy"
    as_of = frame["latest_date"].max()
    affected_product_count = int(count_true_kernel(np.ascontiguousarray(affected_mask)))
    return {
        "schema_version": 1,
        "status": status,
        "generated_at": generated_at,
        "activated_at": activated_at,
        "as_of": None if pd.isna(as_of) else pd.Timestamp(as_of).strftime("%Y-%m-%d"),
        "summary": {
            "checks_total": len(checks), "checks_passed": check_counts["passed"],
            "checks_warning": check_counts["warning"], "checks_failed": check_counts["failed"],
            "checks_unavailable": check_counts["unavailable"], "total_products": total_products,
            "affected_products": affected_product_count,
            "affected_rate": _ratio(affected_product_count, total_products),
            "issue_count": len(issues), "critical_issue_count": severity_counts["critical"],
            "high_issue_count": severity_counts["high"], "medium_issue_count": severity_counts["medium"],
            "nav_anomaly_products": anomaly_product_count, "nav_anomaly_events": anomaly_events,
            "stale_active_products": stale_active_count,
        },
        "checks": checks,
        "issues": issues,
        "validation": {
            "status": "passed" if validation_passed else "unavailable",
            "manifest": ACTIVE_MANIFEST_NAME if manifest else None,
        },
        "execution": _execution_audit(),
    }
