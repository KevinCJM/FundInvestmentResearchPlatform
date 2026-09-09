"""Application service and in-process preview runner for Regime Graph v2."""

from __future__ import annotations

from .indicator_nodes import (is_typed_formula_node, typed_node_expression, typed_node_expressions, formula_plan_key, register_indicator_nodes)
from computation_graph.series_contracts import regime_series_outputs
from computation_graph.series_numba import causal_available_kernel

import copy
from research_series.product_sources import PRODUCT_SOURCES, ETF_ADJUSTED_FIELDS, adjustment_path, product_source_spec
import functools
import hashlib
import itertools
import json
import os
import secrets
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as arrow_dataset
import pyarrow.parquet as pq

try:
    from backend.market_data import read_active_manifest, resolve_tushare_data_dir
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from market_data import read_active_manifest, resolve_tushare_data_dir

try:
    from backend.research_series.service import ResearchSeriesError, read_upload_artifact
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from research_series.service import ResearchSeriesError, read_upload_artifact

try:
    from backend.config import DATA_DIR as DEFAULT_DATA_DIR
except (ImportError, ModuleNotFoundError):  # pragma: no cover - backend/ direct execution
    try:
        from config import DATA_DIR as DEFAULT_DATA_DIR
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - isolated tests
        DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

from compute_policy import ComputePolicyError, validate_execution_audit
from custom_indicators.errors import ConflictError, IndicatorDomainError, NotFoundError, ValidationError

from .analytics import (
    analytics_execution_audit,
    build_segments,
    compare_runs as compare_run_snapshots,
    conditional_statistics,
    transition_matrix,
)
from .data import (
    INDEX_HISTORY_FILES,
    DataBundle,
    _hash_frame,
    _normalise_observations,
    _parse_date,
    resolve_target,
)
from .formula import (
    _compose_formula,
    evaluate_formula,
    prepare_formula as prepare_formula_plan,
)
from .numba_kernels import (
    apply_standardization_kernel,
    change_point_kernel,
    comparison_pair_kernel,
    component_order_kernel,
    execution_audit,
    gmm_fit_kernel,
    gmm_posterior_kernel,
    historical_regime_numba_status,
    hmm_filtered_posterior_kernel,
    hmm_fit_kernel,
    hmm_smoothed_posterior_kernel,
    initialize_gaussian_means_kernel,
    integer_sum_kernel,
    label_summary_kernel,
    markov_filtered_posterior_kernel,
    markov_fit_kernel,
    markov_smoothed_posterior_kernel,
    perturb_scalar_kernel,
    prefix_stability_kernel,
    posterior_assignment_kernel,
    standardize_fit_kernel,
    state_counts_kernel,
    turning_point_kernel,
    validation_windows_kernel,
    warm_historical_regime_numba_kernels,
)
from .result_overview import build_result_overview, warm_result_overview_kernel
from .repository import (
    RegimeDefinitionRepository,
    RegimeExperimentRepository,
    RegimeGraphAssetRepository,
    RegimePlanManifestRepository,
    RegimeRunRepository,
)
from .v2_contracts import (
    RegimeDefinitionV2,
    definition_content_hash,
    inspect_definition_v2,
    parse_definition_v2,
    validate_definition_v2,
    validate_graph_fragment_v2,
)
from .v2_numba import (
    aligned_available_kernel,
    binary_math_kernel,
    calendar_bucket_kernel,
    calendar_resample_kernel,
    clip_kernel,
    combine_states_kernel,
    component_map_kernel,
    confirmation_state_kernel,
    confidence_gate_kernel,
    constant_like_kernel,
    drawdown_series_kernel,
    disagreement_spans_kernel,
    ema_kernel,
    ensemble_state_kernel,
    effective_from_recognition_kernel,
    feature_matrix_kernel,
    final_output_contract_kernel,
    formula_feature_kernel,
    finite_row_positions_kernel,
    hysteresis_state_kernel,
    kalman_filter_kernel,
    left_align_float_kernel,
    matrix_column_kernel,
    maximum_int64_kernel,
    pit_asof_positions_kernel,
    probability_contract_kernel,
    quadrant_state_kernel,
    recognition_delay_summary_kernel,
    regime_graph_numba_status,
    resample_positions_kernel,
    rolling_kernel,
    scatter_component_model_kernel,
    split_positions_at_available_kernel,
    state_confidence_kernel,
    state_count_kernel,
    state_probabilities_kernel,
    stable_time_order_kernel,
    strict_intersection_indices_kernel,
    take_float_kernel,
    take_float_or_nan_kernel,
    take_int64_kernel,
    take_matrix_rows_kernel,
    temporal_output_kernel,
    threshold_state_kernel,
    unary_transform_kernel,
    warm_regime_graph_numba_kernels,
)
from .v2_registry import NODE_REGISTRY, REGISTRY_VERSION, node_catalog
from .trend_numba import (super_smoother_kernel, kama_kernel, trend_features_kernel,
                          trend_regime_kernel, merge_short_regimes_kernel)
from .segment_numba import local_extrema_kernel, between_pivots_kernel, interval_statistic_kernel, range_threshold_kernel
from .v2_registry import STATISTIC_IDS
from .peak_trough_numba import peak_trough_asymmetric_kernel, peak_trough_sideways_kernel, retrospective_dating_timing_kernel
from .v2_migration import migrate_v1_definition
from .v2_templates import get_template_v2, instantiate_template_v2, list_templates_v2
from .node_preview import node_preview_definition, preview_output_context
from .taa import run_taa_backtest as execute_taa_backtest


TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "cancelled"})
JOB_STATUSES = frozenset({"queued", "preparing", "running", *TERMINAL_JOB_STATUSES})
DEFAULT_PREVIEW_TTL_SECONDS = 1800
MAX_PREVIEW_JOBS = 64
MIN_OBSERVATIONS = 5

MACRO_DATASETS: dict[str, str] = {
    "cn_gdp": "macro_cn_gdp_df.parquet",
    "macro_cn_gdp": "macro_cn_gdp_df.parquet",
    "cn_cpi": "macro_cn_cpi_df.parquet",
    "macro_cn_cpi": "macro_cn_cpi_df.parquet",
    "cn_ppi": "macro_cn_ppi_df.parquet",
    "macro_cn_ppi": "macro_cn_ppi_df.parquet",
    "cn_pmi": "macro_cn_pmi_df.parquet",
    "macro_cn_pmi": "macro_cn_pmi_df.parquet",
    "cn_m": "macro_cn_money_df.parquet",
    "macro_cn_money": "macro_cn_money_df.parquet",
    "sf_month": "macro_cn_social_financing_df.parquet",
    "macro_cn_social_financing": "macro_cn_social_financing_df.parquet",
    "shibor": "macro_shibor_df.parquet",
    "macro_shibor": "macro_shibor_df.parquet",
    "shibor_lpr": "macro_lpr_df.parquet",
    "macro_lpr": "macro_lpr_df.parquet",
    "repo_daily": "macro_repo_daily_df.parquet",
    "macro_repo_daily": "macro_repo_daily_df.parquet",
}

FORMAL_USAGE_INTENTS = frozenset({"formal_backtest", "taa"})
PUBLICATION_USAGES = frozenset({"research_display", "product_research", "formal_backtest", "taa"})


@dataclass(frozen=True)
class PortValue:
    values: np.ndarray
    dates: np.ndarray
    available: np.ndarray


def _macro_bundle(
    spec: Mapping[str, Any],
    mode: str,
    as_of: str | None,
    market_data_dir: Path,
) -> DataBundle:
    dataset_id = str(spec.get("dataset") or spec.get("series_id") or "").strip()
    filename = MACRO_DATASETS.get(dataset_id)
    if filename is None:
        raise ValidationError("UNSUPPORTED_MACRO_DATASET", "不支持的宏观数据集。", "parameters.dataset")
    field = str(spec.get("field") or "").strip()
    if not field:
        raise ValidationError("MISSING_MACRO_FIELD", "宏观数据源必须指定 field。", "parameters.field")
    root = resolve_tushare_data_dir(market_data_dir)
    path = root / filename
    if not path.exists():
        raise NotFoundError("MACRO_DATA_NOT_FOUND", f"宏观数据文件 {filename} 不存在。")
    dataset = arrow_dataset.dataset(path, format="parquet")
    names = set(dataset.schema.names)
    date_field = str(spec.get("date_field") or "observation_date")
    if date_field not in names:
        date_field = next(
            (candidate for candidate in ("observation_date", "trade_date", "date", "month", "quarter") if candidate in names),
            "",
        )
    available_field = str(spec.get("available_at_field") or "available_at")
    if not date_field or field not in names:
        raise ValidationError("MACRO_SCHEMA_MISMATCH", "宏观数据缺少日期列或所选字段。", "parameters")
    if mode == "realtime" and available_field not in names:
        raise ValidationError(
            "MACRO_RELEASE_DATE_REQUIRED",
            "实时识别必须使用带 available_at 的宏观数据版本。",
            "parameters.available_at_field",
        )
    columns = [date_field, field]
    for optional in (available_field, "availability_status", "revision", "vintage", "ts_code", "code"):
        if optional in names and optional not in columns:
            columns.append(optional)
    code = str(spec.get("code") or "").strip()
    code_field = next((candidate for candidate in ("ts_code", "code") if candidate in names), None)
    filter_expression = None
    if code:
        if code_field is None:
            raise ValidationError("MACRO_CODE_FIELD_MISSING", "宏观数据集不支持 code 筛选。", "parameters.code")
        filter_expression = arrow_dataset.field(code_field) == code
    table = dataset.to_table(columns=columns, filter=filter_expression)
    raw = table.to_pandas()
    if raw.empty:
        raise NotFoundError("MACRO_SERIES_NOT_FOUND", "未找到所选宏观时序。")
    if mode == "realtime" and "availability_status" in raw.columns:
        statuses = set(raw["availability_status"].dropna().astype(str))
        if "release_date_unknown" in statuses:
            raise ValidationError(
                "MACRO_RELEASE_DATE_UNKNOWN",
                "宏观数据发布日期未知，只能用于事后研究。",
                "parameters.dataset",
            )
    raw = raw.rename(columns={date_field: "observation_date", field: "value"})
    if available_field in raw.columns and available_field != "available_at":
        raw = raw.rename(columns={available_field: "available_at"})
    frame, revision_meta = _normalise_observations(
        raw,
        mode,
        as_of,
        availability_mode="point_in_time" if mode == "realtime" else str(spec.get("availability_mode") or "latest"),
    )
    if spec.get("start_date"):
        frame = frame.loc[frame["observation_date"] >= _parse_date(spec["start_date"], "parameters.start_date")].copy()
    if spec.get("end_date"):
        frame = frame.loc[frame["observation_date"] <= _parse_date(spec["end_date"], "parameters.end_date")].copy()
    if frame.empty:
        raise ValidationError("EMPTY_DATE_RANGE", "所选日期区间没有宏观数据。", "parameters")
    stat = path.stat()
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "dataset": dataset_id,
                "field": field,
                "code": code,
                "frame": _hash_frame(frame),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return DataBundle(
        frame=frame,
        snapshot={
            "kind": "macro",
            "dataset": dataset_id,
            "field": field,
            "code": code or None,
            "file": filename,
            "fingerprint": fingerprint,
            **revision_meta,
        },
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _as_dates(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column].to_numpy(dtype="datetime64[ns]").view(np.int64)
    return np.ascontiguousarray(values, dtype=np.int64)


def _as_values(frame: pd.DataFrame) -> np.ndarray:
    return np.ascontiguousarray(frame["value"].to_numpy(dtype=np.float64), dtype=np.float64)


def _safe_number(value: Any) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _content_hash(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stored_run_snapshot_hash(run: Mapping[str, Any]) -> str:
    analytical = copy.deepcopy(dict(run))
    for key in ("id", "created_at", "immutable", "publications", "content_hash"):
        analytical.pop(key, None)
    analytical["application_bindings"] = []
    return _content_hash(analytical)


@functools.lru_cache(maxsize=128)
def _file_checksum_cached(path_text: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _file_checksum(path: Path) -> str:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return _file_checksum_cached(str(resolved), int(stat.st_size), int(stat.st_mtime_ns))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def hydrate_v2_run_snapshot(
    run: Mapping[str, Any],
    *,
    workspace_data_dir: Path | None = None,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Hydrate a raw v2 run without constructing or warming a graph service."""

    hydrated = copy.deepcopy(dict(run))
    if str(hydrated.get("schema_version")) != "2.0" or "series" in hydrated:
        return hydrated
    manifest = hydrated.get("series_artifact")
    if not isinstance(manifest, Mapping):
        raise ValidationError(
            "REGIME_SERIES_ARTIFACT_REQUIRED",
            "v2 正式运行缺少主序列制品。",
            "series_artifact",
        )
    selected_artifact_dir = (
        Path(artifact_dir)
        if artifact_dir is not None
        else Path(workspace_data_dir or DEFAULT_DATA_DIR) / "historical_regime_v2_artifacts"
    ).expanduser().resolve()
    artifact_id = str(manifest.get("artifact_id") or "")
    prefix = "regime-series-sha256-"
    digest = artifact_id.removeprefix(prefix)
    if (
        not artifact_id.startswith(prefix)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or str(manifest.get("checksum") or "") != f"sha256:{digest}"
        or str(manifest.get("format") or "") != "parquet"
    ):
        raise ValidationError(
            "REGIME_SERIES_ARTIFACT_INVALID",
            "历史情景主序列制品标识无效。",
            "series_artifact",
        )
    path = (selected_artifact_dir / f"{digest}.parquet").resolve()
    if path.parent != selected_artifact_dir or not path.is_file():
        raise NotFoundError("REGIME_SERIES_ARTIFACT_NOT_FOUND", "历史情景主序列制品不存在。")
    if _sha256_file(path) != digest:
        raise ValidationError(
            "REGIME_SERIES_ARTIFACT_CHECKSUM_MISMATCH",
            "历史情景主序列制品校验失败。",
            "series_artifact",
        )
    frame = pd.read_parquet(path)
    if len(frame) != int(manifest.get("row_count") or -1):
        raise ValidationError(
            "REGIME_SERIES_ARTIFACT_ROW_COUNT_MISMATCH",
            "历史情景主序列行数与清单不一致。",
            "series_artifact",
        )
    series: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        item = {
            key: (None if pd.isna(value) else value)
            for key, value in row.items()
            if key not in {"probabilities_json", "features_json", "reasons_json"}
        }
        item["probabilities"] = json.loads(str(row.get("probabilities_json") or "{}"))
        item["features"] = json.loads(str(row.get("features_json") or "{}"))
        item["reasons"] = json.loads(str(row.get("reasons_json") or "[]"))
        series.append(_json_safe(item))
    hydrated["series"] = series
    return hydrated


def _snapshot_identity(root: Path, manifest: Mapping[str, Any] | None) -> tuple[str, str]:
    payload = manifest or {}
    snapshot_id = str(payload.get("snapshot_id") or root.name)
    generation = str(payload.get("generation") or root.name)
    return snapshot_id, generation


def _expected_source_filename(node_type: str, parameters: Mapping[str, Any]) -> str | None:
    if node_type.removeprefix("source.") in PRODUCT_SOURCES:
        return product_source_spec(node_type.removeprefix("source."), parameters.get("field"))["filename"]
    if node_type == "source.index":
        return INDEX_HISTORY_FILES.get(str(parameters.get("source_api") or "index_daily"))
    if node_type == "source.macro":
        dataset_id = str(parameters.get("dataset") or parameters.get("series_id") or "").strip()
        return MACRO_DATASETS.get(dataset_id) or (
            dataset_id if dataset_id in set(MACRO_DATASETS.values()) else None
        )
    return None


def _source_spec(node_type: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    kind = node_type.split(".", 1)[1]
    payload = {"kind": kind, **copy.deepcopy(dict(parameters))}
    if kind == "inline" and "rows" not in payload and "inline_rows" in payload:
        payload["rows"] = payload.pop("inline_rows")
    return payload


def _canonical_source_cache_key(
    source: Mapping[str, Any],
    mode: str,
    resolver_as_of: str | None,
) -> str:
    """Identify one resolved source view across graph and evaluation consumers.

    Only the display name is non-semantic. Field selection, date bounds,
    availability/vintage policy, transforms, immutable artifact checksums and
    snapshot bindings deliberately remain in the key so distinct source views
    can never share a resolved bundle accidentally.
    """

    canonical = copy.deepcopy(dict(source))
    canonical.pop("name", None)
    return _content_hash(
        {
            "source": canonical,
            "mode": mode,
            "resolver_as_of": resolver_as_of,
        }
    )


def _required_node_ids(definition: RegimeDefinitionV2) -> set[str]:
    node_map = {node.id: node for node in definition.graph.nodes}
    pending = [reference.node_id for reference in definition.graph.outputs.values()]
    pending.extend(definition.graph.exposed_node_ids)
    required: set[str] = set()
    while pending:
        node_id = pending.pop()
        if node_id in required or node_id not in node_map:
            continue
        required.add(node_id)
        pending.extend(reference.node_id for reference in node_map[node_id].inputs.values())
    return required


def _kernel_ids_for_definition(definition: RegimeDefinitionV2) -> list[str]:
    required = _required_node_ids(definition)
    kernel_ids = {
        "final_output_contract",
        "left_align_float",
        "probability_contract",
        "recognition_delay_summary",
        "state_count",
        "temporal_output",
    }
    if "probabilities" not in definition.graph.outputs:
        kernel_ids.add("state_probabilities")
    if "confidence" not in definition.graph.outputs:
        kernel_ids.add("state_confidence")
    if "recognition_index" in definition.graph.outputs and "effective_index" not in definition.graph.outputs:
        kernel_ids.add("effective_from_recognition")
    for node in definition.graph.nodes:
        if node.id not in required:
            continue
        node_type = node.type
        if is_typed_formula_node(node, NODE_REGISTRY):
            kernel_ids.update({"maximum_int64", "causal_available", "valid_series_output"})
        elif node_type == "source.constant":
            kernel_ids.add("constant_like")
        elif node_type == "align.strict_intersection":
            kernel_ids.update({"strict_intersection_indices", "take_float", "take_int64", "maximum_int64"})
        elif node_type == "align.pit_asof":
            kernel_ids.update({"stable_time_order", "take_float", "take_int64", "pit_asof_positions", "take_float_or_nan", "aligned_available"})
        elif node_type == "align.resample":
            if "frequency" in node.parameters or "every" not in node.parameters:
                kernel_ids.update({"calendar_bucket", "calendar_resample"})
            else:
                kernel_ids.update({"resample_positions", "take_float", "take_int64"})
        elif node_type == "align.cross_section":
            kernel_ids.update({"strict_intersection_indices", "take_float", "take_int64", "maximum_int64", "feature_matrix"})
        elif node_type == "feature.matrix":
            kernel_ids.add("feature_matrix")
        elif node_type in {"transform.identity", "transform.log", "transform.lag", "transform.diff", "transform.return", "transform.yoy", "transform.mom"}:
            kernel_ids.add("unary_transform")
        elif node_type == "transform.drawdown":
            kernel_ids.add("drawdown_series")
        elif node_type == "transform.standardize" or node_type == "filter.sma" or node_type.startswith("rolling."):
            kernel_ids.add("rolling")
        elif node_type == "transform.clip":
            kernel_ids.add("clip")
        elif node_type.startswith("math."):
            kernel_ids.add("binary_math")
        elif node_type == "filter.ema":
            kernel_ids.add("ema")
        elif node_type in {"pivot.local_extrema", "segment.between_pivots", "model.range_threshold", *STATISTIC_IDS}:
            kernel_ids.add(NODE_REGISTRY[node_type]["kernel_id"])
            if node_type == "pivot.local_extrema":
                kernel_ids.add("retrospective_dating_timing")
            if node_type == "model.range_threshold":
                kernel_ids.update({"constant_like", "maximum_int64"})
        elif node_type == "model.peak_trough":
            kernel_ids.update({"peak_trough", "peak_trough_asymmetric", "peak_trough_remove", "peak_trough_alternate", "peak_trough_sideways", "retrospective_dating_timing"})
        elif node_type in {"filter.super_smoother", "filter.kama", "feature.trend_metrics",
                           "model.trend_regime", "post.merge_short_regimes"}:
            kernel_ids.add(NODE_REGISTRY[node_type]["kernel_id"])
        elif node_type == "filter.kalman":
            kernel_ids.add("kalman_filter")
        elif node_type == "model.threshold":
            kernel_ids.update({"threshold_state", "unary_transform", "state_confidence", "state_probabilities", "temporal_output"})
        elif node_type == "model.hysteresis":
            kernel_ids.update({"hysteresis_state", "unary_transform", "state_confidence", "state_probabilities"})
        elif node_type == "post.hysteresis":
            kernel_ids.update({"hysteresis_state", "unary_transform", "state_confidence", "state_probabilities"})
        elif node_type == "model.quadrant":
            kernel_ids.update({"quadrant_state", "binary_math", "state_confidence", "state_probabilities"})
        elif node_type in {"model.turning_point", "model.change_point"}:
            kernel_ids.update({"state_probabilities", "temporal_output"})
        elif node_type in {"model.hmm", "model.markov", "model.gmm"}:
            kernel_ids.update({"finite_row_positions", "split_positions_at_available", "take_matrix_rows", "matrix_column", "scatter_component_model"})
        elif node_type == "model.ensemble":
            kernel_ids.add("ensemble_state")
        elif node_type in {"post.confirmation", "post.min_duration"}:
            kernel_ids.add("confirmation_state")
        elif node_type == "post.component_map":
            kernel_ids.add("component_map")
        elif node_type in {"post.priority", "post.conflict_reject"}:
            kernel_ids.add("combine_states")
        elif node_type == "post.confidence_gate":
            kernel_ids.add("confidence_gate")
        elif node_type == "output.temporal":
            kernel_ids.add("temporal_output")
    return sorted(kernel_ids)


def _shared_kernel_ids_for_definition(definition: RegimeDefinitionV2) -> list[str]:
    required = _required_node_ids(definition)
    kernel_ids: set[str] = set()
    for node in definition.graph.nodes:
        if node.id not in required:
            continue
        if node.type == "model.turning_point":
            kernel_ids.add("turning_point")
        elif node.type == "model.change_point":
            kernel_ids.add("change_point")
        elif node.type in {"model.hmm", "model.markov", "model.gmm"}:
            kernel_ids.update(
                {
                    "standardize_fit",
                    "apply_standardization",
                    "initialize_gaussian_means",
                    "component_order",
                    "posterior_assignment",
                }
            )
            if node.type == "model.gmm":
                kernel_ids.update({"gmm_fit", "gmm_posterior"})
            elif node.type == "model.markov":
                kernel_ids.update(
                    {
                        "markov_fit",
                        "markov_filtered_posterior",
                        "markov_smoothed_posterior",
                    }
                )
            else:
                kernel_ids.update({"hmm_fit", "hmm_filtered_posterior", "hmm_smoothed_posterior"})
    return sorted(kernel_ids)


class PreviewCancelled(RuntimeError):
    pass


class RegimeGraphV2Service:
    def __init__(
        self,
        workspace_data_dir: Path | None = None,
        market_data_dir: Path | None = None,
        indicator_service: Any = None,
    ) -> None:
        configured = os.getenv("HISTORICAL_REGIME_DATA_DIR") or os.getenv("CUSTOM_INDICATOR_DATA_DIR")
        self.workspace_data_dir = workspace_data_dir or (Path(configured) if configured else DEFAULT_DATA_DIR)
        self.market_data_dir = market_data_dir or DEFAULT_DATA_DIR
        self.indicator_service = indicator_service
        register_indicator_nodes(indicator_service, NODE_REGISTRY)
        self.definitions = RegimeDefinitionRepository(
            self.workspace_data_dir / "historical_regime_v2_definitions.json"
        )
        self.runs = RegimeRunRepository(
            self.workspace_data_dir / "historical_regime_runs.json"
        )
        self.graph_assets = RegimeGraphAssetRepository(
            self.workspace_data_dir / "historical_regime_v2_graph_assets.json"
        )
        self.experiments = RegimeExperimentRepository(
            self.workspace_data_dir / "historical_regime_v2_experiments.json"
        )
        self.plan_manifests = RegimePlanManifestRepository(
            self.workspace_data_dir / "historical_regime_v2_plan_manifests.json"
        )
        self.artifact_dir = self.workspace_data_dir / "historical_regime_v2_artifacts"
        self._runtime_audit = warm_regime_graph_numba_kernels()
        self._overview_runtime = warm_result_overview_kernel()
        self._plans: dict[str, dict[str, Any]] = {}
        self._plans_by_graph_hash: dict[str, str] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.startup_prewarm = self.prewarm_saved_definitions()

    def catalog(self) -> dict[str, Any]:
        register_indicator_nodes(self.indicator_service, NODE_REGISTRY)
        payload = node_catalog()
        payload["runtime"] = regime_graph_numba_status()
        payload["runtime"]["shared_historical_runtime"] = historical_regime_numba_status()
        payload["runtime"]["overview_runtime"] = copy.deepcopy(self._overview_runtime)
        payload["startup_prewarm"] = copy.deepcopy(self.startup_prewarm)
        payload["plan_persistence"] = {
            "manifest_count": len(self.plan_manifests.list()),
            "manifest_authorizes_execution": False,
            "token_storage": "process_local_memory_only",
            "startup_rewarm_required": True,
        }
        return payload

    def templates(self) -> dict[str, Any]:
        return {"schema_version": "2.0", "items": list_templates_v2()}

    def instantiate_template(self, template_id: str) -> dict[str, Any]:
        template = get_template_v2(template_id)
        payload = instantiate_template_v2(template_id)
        if payload is None or template is None:
            raise NotFoundError("REGIME_GRAPH_TEMPLATE_NOT_FOUND", "未找到指定的历史情景图谱模板。")
        payload["template_id"] = f"{template_id}@{template['version']}"
        definition = parse_definition_v2(payload)
        inspection = validate_definition_v2(definition)
        return {
            "template_id": template_id,
            "template_version": template["version"],
            "template_content_hash": template["content_hash"],
            "source": {
                "template_id": template_id,
                "template_version": template["version"],
                "content_hash": template["content_hash"],
            },
            "definition": definition.model_dump(mode="json"),
            "inference": inspection,
        }

    @staticmethod
    def _validate_available_asset_nodes(nodes: Any) -> list[dict[str, Any]]:
        if not isinstance(nodes, list) or not nodes:
            raise ValidationError(
                "EMPTY_REGIME_GRAPH_ASSET",
                "用户模板或子图至少需要一个节点。",
                "graph.nodes",
            )
        if len(nodes) > 128 or not all(isinstance(node, Mapping) for node in nodes):
            raise ValidationError(
                "INVALID_REGIME_GRAPH_ASSET_NODES",
                "用户模板或子图节点必须是对象且不能超过 128 个。",
                "graph.nodes",
            )
        node_ids: set[str] = set()
        normalized: list[dict[str, Any]] = []
        for index, raw_node in enumerate(nodes):
            node = copy.deepcopy(dict(raw_node))
            node_id = str(node.get("id") or "")
            node_type = str(node.get("type") or node.get("type_id") or "")
            metadata = NODE_REGISTRY.get(node_type)
            if not node_id or node_id in node_ids:
                raise ValidationError(
                    "INVALID_REGIME_GRAPH_ASSET_NODE_ID",
                    "用户模板或子图节点 id 不能为空或重复。",
                    f"graph.nodes.{index}.id",
                )
            if metadata is None or metadata.get("available") is not True:
                raise ValidationError(
                    "REGIME_GRAPH_ASSET_NODE_NOT_AVAILABLE",
                    f"节点 {node_type or '<empty>'} 不在当前可用白名单中。",
                    f"graph.nodes.{index}.type",
                )
            version = int(node.get("type_version", 1))
            if version != int(metadata["version"]):
                raise ValidationError(
                    "UNKNOWN_NODE_TYPE_VERSION",
                    f"节点 {node_type} 不支持版本 {version}。",
                    f"graph.nodes.{index}.type_version",
                )
            node["type"] = node_type
            node["type_id"] = node_type
            node["type_version"] = version
            node_ids.add(node_id)
            normalized.append(node)
        for index, node in enumerate(normalized):
            inputs = node.get("inputs") or {}
            if not isinstance(inputs, Mapping):
                raise ValidationError(
                    "INVALID_REGIME_GRAPH_ASSET_INPUTS",
                    "节点 inputs 必须是端口引用对象。",
                    f"graph.nodes.{index}.inputs",
                )
            for input_name, reference in inputs.items():
                if (
                    not isinstance(reference, Mapping)
                    or str(reference.get("node_id") or "") not in node_ids
                ):
                    raise ValidationError(
                        "REGIME_GRAPH_ASSET_DANGLING_INPUT",
                        "用户模板或子图不能保存悬空输入。",
                        f"graph.nodes.{index}.inputs.{input_name}",
                    )
        return normalized

    def _normalize_graph_asset(
        self,
        kind: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if kind not in {"template", "subgraph"}:
            raise ValidationError(
                "INVALID_REGIME_GRAPH_ASSET_KIND",
                "kind 只能是 template 或 subgraph。",
                "kind",
            )
        name = str(payload.get("name") or "").strip()
        if not name or len(name) > 100:
            raise ValidationError(
                "INVALID_REGIME_GRAPH_ASSET_NAME",
                "用户模板或子图名称长度必须在 1 到 100 个字符之间。",
                "name",
            )
        description = str(payload.get("description") or "")
        if len(description) > 1000:
            raise ValidationError(
                "INVALID_REGIME_GRAPH_ASSET_DESCRIPTION",
                "用户模板或子图说明不能超过 1000 个字符。",
                "description",
            )
        if kind == "template":
            raw_definition = payload.get("definition")
            if not isinstance(raw_definition, Mapping):
                raise ValidationError(
                    "REGIME_TEMPLATE_DEFINITION_REQUIRED",
                    "用户模板必须包含完整的 v2 definition。",
                    "definition",
                )
            definition = parse_definition_v2(raw_definition)
            inspection = validate_definition_v2(definition)
            self._validate_available_asset_nodes(
                [node.model_dump(mode="json") for node in definition.graph.nodes]
            )
            content = definition.model_dump(
                mode="json",
                exclude={"id", "revision", "created_at", "updated_at"},
            )
            return {
                "name": name,
                "description": description,
                "definition": content,
                "registry_version": REGISTRY_VERSION,
                "graph_hash": inspection["graph_hash"],
                "content_hash": _content_hash(content),
            }
        raw_graph = payload.get("graph")
        if not isinstance(raw_graph, Mapping):
            raise ValidationError(
                "REGIME_SUBGRAPH_REQUIRED",
                "用户子图必须包含 graph。",
                "graph",
            )
        graph = copy.deepcopy(dict(raw_graph))
        graph["nodes"] = self._validate_available_asset_nodes(graph.get("nodes"))
        edges = graph.get("edges") or []
        if not isinstance(edges, list) or len(edges) > 256:
            raise ValidationError(
                "INVALID_REGIME_SUBGRAPH_EDGES",
                "用户子图 edges 必须是数组且不能超过 256 条。",
                "graph.edges",
            )
        graph["edges"] = copy.deepcopy(edges)
        graph = validate_graph_fragment_v2(graph)
        return {
            "name": name,
            "description": description,
            "graph": graph,
            "registry_version": REGISTRY_VERSION,
            "content_hash": _content_hash(graph),
        }

    def list_graph_assets(self, kind: str | None = None) -> list[dict[str, Any]]:
        if kind is not None and kind not in {"template", "subgraph"}:
            raise ValidationError(
                "INVALID_REGIME_GRAPH_ASSET_KIND",
                "kind 只能是 template 或 subgraph。",
                "kind",
            )
        return self.graph_assets.list(kind)

    def get_graph_asset(self, asset_id: str, revision: int | None = None) -> dict[str, Any]:
        return self.graph_assets.get(asset_id, revision)

    def create_graph_asset(self, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.graph_assets.create(kind, self._normalize_graph_asset(kind, payload))

    def update_graph_asset(
        self,
        asset_id: str,
        expected_revision: int,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        current = self.graph_assets.get(asset_id)
        normalized = self._normalize_graph_asset(str(current["kind"]), payload)
        return self.graph_assets.update(asset_id, expected_revision, normalized)

    def instantiate_graph_asset(
        self,
        asset_id: str,
        revision: int | None = None,
    ) -> dict[str, Any]:
        asset = self.graph_assets.get(asset_id, revision)
        lineage = {
            "asset_id": asset["id"],
            "asset_revision": asset["revision"],
            "content_hash": asset["content_hash"],
            "registry_version": asset["registry_version"],
        }
        if asset["kind"] == "template":
            draft = copy.deepcopy(asset["definition"])
            draft.pop("id", None)
            draft.pop("revision", None)
            draft.pop("created_at", None)
            draft.pop("updated_at", None)
            draft["template_id"] = f"user:{asset['id']}@{asset['revision']}"
            return {
                "kind": "template",
                "source": lineage,
                "definition": draft,
                "inference": inspect_definition_v2(parse_definition_v2(draft)),
            }
        return {
            "kind": "subgraph",
            "source": lineage,
            "graph": copy.deepcopy(asset["graph"]),
        }

    def infer(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            definition = parse_definition_v2(payload)
        except IndicatorDomainError as exc:
            detail = exc.detail()
            return {
                "valid": False,
                "schema_version": "2.0",
                "registry_version": REGISTRY_VERSION,
                "graph_hash": None,
                "definition_hash": None,
                "topological_order": [],
                "inferred": {"ports": {}, "symbolic_shapes": {}, "state_count": 0},
                "dependencies": {"direct": {}, "transitive": {}, "required_for_outputs": []},
                "causality": {
                    "causal": False,
                    "repaints": False,
                    "realtime_supported": False,
                    "noncausal_node_ids": [],
                    "repaint_node_ids": [],
                    "non_realtime_node_ids": [],
                },
                "cost_estimate": {
                    "model": "relative_linearized_cost_v1",
                    "total_relative_units_per_observation": 0.0,
                    "nodes": [],
                },
                "execution_lanes": [],
                "errors": detail.get("diagnostics") or [
                    {
                        "code": detail["code"],
                        "path": detail.get("field") or "definition",
                        "message": detail["message"],
                        "severity": "error",
                    }
                ],
                "warnings": [],
            }
        return {**inspect_definition_v2(definition), "result_kind": "time_series", "series_outputs": regime_series_outputs(definition)}

    def _active_snapshot_binding(
        self,
        node_type: str,
        parameters: Mapping[str, Any],
    ) -> dict[str, Any]:
        filename = _expected_source_filename(node_type, parameters)
        if not filename:
            raise ValidationError(
                "UNSUPPORTED_SOURCE_DATASET",
                "数据源没有可锁定的文件版本。",
                "graph.nodes.parameters",
            )
        root = resolve_tushare_data_dir(self.market_data_dir, strict=True)
        path = root / filename
        if not path.is_file():
            raise NotFoundError("SOURCE_DATA_NOT_FOUND", f"数据文件 {filename} 不存在。")
        manifest = read_active_manifest(self.market_data_dir)
        if node_type in {"source.etf", "source.fund"} and (not manifest or filename not in manifest.get("files", {})):
            raise ValidationError("PRODUCT_FILE_NOT_PUBLISHED", "产品行情文件不在活跃快照清单中。", "graph.nodes.parameters")
        snapshot_id, generation = _snapshot_identity(root, manifest)
        factor = adjustment_path(root) if node_type == 'source.etf' and parameters.get('field') in ETF_ADJUSTED_FIELDS else None
        return {
            "snapshot_id": snapshot_id,
            "snapshot_generation": generation,
            "source_file": filename,
            "file_checksum": _file_checksum(path),
            **({'adjustment_checksum': _file_checksum(factor)} if factor else {}),
        }

    def _bound_source_root(
        self,
        node_type: str,
        parameters: Mapping[str, Any],
    ) -> Path:
        expected_file = _expected_source_filename(node_type, parameters)
        binding_names = (
            "snapshot_id",
            "snapshot_generation",
            "source_file",
            "file_checksum",
        )
        present = [bool(parameters.get(name)) for name in binding_names]
        if not any(present):
            if node_type in {"source.etf", "source.fund"}:
                self._active_snapshot_binding(node_type, parameters)
            return self.market_data_dir
        if not all(present):
            raise ValidationError(
                "SOURCE_SNAPSHOT_BINDING_INCOMPLETE",
                "数据源版本绑定不完整，必须同时提供快照、文件和指纹。",
                "graph.nodes.parameters",
            )
        generation = str(parameters["snapshot_generation"])
        if Path(generation).name != generation or generation in {".", ".."}:
            raise ValidationError(
                "SOURCE_SNAPSHOT_GENERATION_INVALID",
                "数据源快照代次无效。",
                "graph.nodes.parameters.snapshot_generation",
            )
        base = Path(self.market_data_dir).expanduser().resolve()
        candidate = (base / generation).resolve()
        active = resolve_tushare_data_dir(base)
        if active.name == generation:
            candidate = active
        if candidate != base and base not in candidate.parents:
            raise ValidationError(
                "SOURCE_SNAPSHOT_OUTSIDE_DATA_DIR",
                "数据源快照必须位于受控数据目录。",
                "graph.nodes.parameters.snapshot_generation",
            )
        if not candidate.is_dir():
            raise NotFoundError("SOURCE_SNAPSHOT_NOT_FOUND", "已锁定的数据快照不存在。")
        active_manifest = read_active_manifest(base) if candidate == active else None
        actual_snapshot_id, actual_generation = _snapshot_identity(candidate, active_manifest)
        if (
            str(parameters["snapshot_id"]) != actual_snapshot_id
            or generation != actual_generation
        ):
            raise ValidationError(
                "SOURCE_SNAPSHOT_IDENTITY_MISMATCH",
                "数据源快照身份与已保存定义不一致。",
                "graph.nodes.parameters.snapshot_id",
            )
        source_file = str(parameters["source_file"])
        if expected_file is None or source_file != expected_file:
            raise ValidationError(
                "SOURCE_FILE_MISMATCH",
                "数值字段与绑定的数据文件不一致，请重新选择产品后预览。",
                "graph.nodes.parameters.source_file",
            )
        if node_type in {"source.etf", "source.fund"} and active_manifest is not None and source_file not in active_manifest.get("files", {}):
            raise ValidationError("PRODUCT_FILE_NOT_PUBLISHED", "产品行情文件不在活跃快照清单中。", "graph.nodes.parameters.source_file")
        source_path = candidate / source_file
        if not source_path.is_file():
            raise NotFoundError("SOURCE_DATA_NOT_FOUND", "已锁定的数据文件不存在。")
        if _file_checksum(source_path) != str(parameters["file_checksum"]):
            raise ValidationError(
                "SOURCE_FILE_CHECKSUM_MISMATCH",
                "已锁定的数据文件内容校验值发生变化，拒绝复用。",
                "graph.nodes.parameters.file_checksum",
            )
        if node_type == 'source.etf' and parameters.get('field') in ETF_ADJUSTED_FIELDS:
            factor = adjustment_path(candidate)
            if not factor:
                raise ValidationError('ADJUSTMENT_DATA_NOT_FOUND', '当前快照缺少 ETF 复权因子，请补齐后重新选择 ETF。')
            if not parameters.get('adjustment_checksum') or _file_checksum(factor) != parameters['adjustment_checksum']:
                raise ValidationError('ADJUSTMENT_BINDING_REQUIRED', '复权因子未绑定或版本已变化，请重新选择 ETF。')
        return candidate

    def _upload_bundle(
        self,
        parameters: Mapping[str, Any],
        mode: str,
        as_of: str | None,
    ) -> DataBundle:
        try:
            raw = read_upload_artifact(
                self.market_data_dir,
                str(parameters.get("artifact_id") or ""),
                str(parameters.get("checksum") or "") or None,
            )
        except ResearchSeriesError as exc:
            raise ValidationError(exc.code, exc.message, exc.field or "artifact_id") from exc
        frame, revision_meta = _normalise_observations(
            raw,
            mode,
            as_of,
            availability_mode=str(parameters.get("availability_mode") or "point_in_time"),
        )
        if frame.empty:
            raise ValidationError("EMPTY_UPLOAD_SERIES", "上传数据版本没有可用观测。", "artifact_id")
        return DataBundle(
            frame=frame,
            snapshot={
                "kind": "upload",
                "artifact_id": str(parameters.get("artifact_id")),
                "checksum": str(parameters.get("checksum")),
                "checksum_scope": "canonical_rows_v1",
                "format": "parquet",
                "fingerprint": _hash_frame(frame),
                **revision_meta,
            },
        )

    def _freeze_source_versions(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        frozen = copy.deepcopy(dict(payload))
        graph = frozen.get("graph") if isinstance(frozen.get("graph"), dict) else {}
        for node in graph.get("nodes", []) if isinstance(graph.get("nodes"), list) else []:
            if not isinstance(node, dict):
                continue
            node_type = str(node.get("type") or node.get("type_id") or "")
            parameters = node.setdefault("parameters", {})
            if not isinstance(parameters, dict):
                continue
            if node_type in {"source.index", "source.macro", "source.etf", "source.fund"}:
                expected = _expected_source_filename(node_type, parameters)
                locked = all(
                    parameters.get(name)
                    for name in (
                        "snapshot_id",
                        "snapshot_generation",
                        "source_file",
                        "file_checksum",
                    )
                )
                if locked:
                    if expected and str(parameters.get("source_file")) != expected:
                        raise ValidationError(
                            "SOURCE_FILE_MISMATCH",
                            "数据源文件与节点类型或数据集不匹配。",
                            f"graph.nodes.{node.get('id')}.parameters.source_file",
                        )
                else:
                    parameters.update(self._active_snapshot_binding(node_type, parameters))
            if node_type == "source.upload":
                if str(parameters.get("format") or "parquet") != "parquet":
                    raise ValidationError(
                        "UPLOAD_FORMAT_UNSUPPORTED",
                        "上传数据版本只支持 Parquet。",
                        f"graph.nodes.{node.get('id')}.parameters.format",
                    )
                try:
                    read_upload_artifact(
                        self.market_data_dir,
                        str(parameters.get("artifact_id") or ""),
                        str(parameters.get("checksum") or "") or None,
                    )
                except ResearchSeriesError as exc:
                    raise ValidationError(exc.code, exc.message, exc.field or "artifact_id") from exc
            if node_type == "source.indicator" and not parameters.get("data_fingerprint"):
                bundle = resolve_target(
                    _source_spec(node_type, parameters),
                    "realtime",
                    None,
                    self.market_data_dir,
                    self.indicator_service,
                )
                fingerprint = str(bundle.snapshot.get("fingerprint") or "")
                if not fingerprint:
                    raise ValidationError(
                        "INDICATOR_DATA_FINGERPRINT_REQUIRED",
                        "指标时序没有返回可锁定的数据指纹。",
                        f"graph.nodes.{node.get('id')}.parameters.data_fingerprint",
                    )
                parameters["data_fingerprint"] = fingerprint
                parameters["indicator_data_snapshot"] = _json_safe(bundle.snapshot)
        targets = frozen.get("evaluation_targets")
        for target_index, target in enumerate(targets if isinstance(targets, list) else []):
            if not isinstance(target, dict) or not isinstance(target.get("source"), dict):
                continue
            source = target["source"]
            source_kind = str(source.get("kind") or "")
            if source_kind in {"index", "macro", "etf", "fund"}:
                node_type = f"source.{source_kind}"
                locked = all(
                    source.get(name)
                    for name in (
                        "snapshot_id",
                        "snapshot_generation",
                        "source_file",
                        "file_checksum",
                    )
                )
                if not locked:
                    source.update(self._active_snapshot_binding(node_type, source))
            elif source_kind == "upload":
                try:
                    read_upload_artifact(
                        self.market_data_dir,
                        str(source.get("artifact_id") or ""),
                        str(source.get("checksum") or "") or None,
                    )
                except ResearchSeriesError as exc:
                    raise ValidationError(
                        exc.code,
                        exc.message,
                        exc.field or f"evaluation_targets.{target_index}.source",
                    ) from exc
            elif source_kind == "indicator" and not source.get("data_fingerprint"):
                bundle = resolve_target(
                    source,
                    "realtime",
                    None,
                    self.market_data_dir,
                    self.indicator_service,
                )
                fingerprint = str(bundle.snapshot.get("fingerprint") or "")
                if not fingerprint:
                    raise ValidationError(
                        "INDICATOR_DATA_FINGERPRINT_REQUIRED",
                        "评价指标时序没有返回可锁定的数据指纹。",
                        f"evaluation_targets.{target_index}.source.data_fingerprint",
                    )
                source["data_fingerprint"] = fingerprint
                source["indicator_data_snapshot"] = _json_safe(bundle.snapshot)
            elif source_kind in {"inline", "relative"}:
                continue
        return frozen

    @staticmethod
    def _formula_frame(node: Any) -> pd.DataFrame:
        columns: dict[str, np.ndarray] = {
            name: np.linspace(1.0, 8.0, 8, dtype=np.float64)
            for name in node.inputs
        }
        aliases = node.parameters.get("variables")
        if isinstance(aliases, Mapping):
            for alias, port_name in aliases.items():
                if str(alias).isidentifier() and str(port_name) in columns:
                    columns[str(alias)] = columns[str(port_name)]
        return pd.DataFrame(columns)

    def _prepare_formula_nodes(self, definition: RegimeDefinitionV2) -> dict[str, Any]:
        prepared: dict[str, Any] = {}
        for node in definition.graph.nodes:
            if not is_typed_formula_node(node, NODE_REGISTRY):
                continue
            frame = self._formula_frame(node)
            for port, expression in typed_node_expressions(node, NODE_REGISTRY).items():
                entry = prepare_formula_plan(expression, frame)
                plan, _ = _compose_formula(expression, frame)
                entry["typed_expression"] = plan.to_dict()
                entry["node_id"] = node.id
                entry["output_port"] = port
                prepared[formula_plan_key(node.id, port)] = entry
        return prepared

    def _preparation_hash(
        self,
        definition: RegimeDefinitionV2,
        formula_plans: Mapping[str, Any] | None = None,
    ) -> str:
        del formula_plans
        payload = {
            "graph_hash": inspect_definition_v2(definition)["graph_hash"],
            "formulas": {
                node.id: typed_node_expressions(node, NODE_REGISTRY)
                for node in definition.graph.nodes
                if is_typed_formula_node(node, NODE_REGISTRY)
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _formula_manifest(formula_plans: Mapping[str, Any]) -> dict[str, Any]:
        """Keep the rebuild/audit facts while excluding every formula token."""

        allowed = (
            "compiled_plan_id",
            "compile_status",
            "expression_hash",
            "referenced_columns",
            "execution_backend",
            "njit_required",
            "nopython",
            "object_mode",
            "request_time_compilation",
            "python_fallback",
        )
        return {
            str(node_id): {
                key: copy.deepcopy(plan[key])
                for key in allowed
                if key in plan
            }
            for node_id, plan in formula_plans.items()
        }

    def _build_plan_manifest(
        self,
        definition: RegimeDefinitionV2,
        public_plan: Mapping[str, Any],
        runtime: Mapping[str, Any],
        formula_plans: Mapping[str, Any],
    ) -> dict[str, Any]:
        kernel_ids = list(public_plan.get("kernel_ids") or [])
        shared_kernel_ids = list(public_plan.get("shared_kernel_ids") or [])
        graph_signatures = runtime.get("kernel_signatures") or {}
        shared_runtime = historical_regime_numba_status()
        shared_signatures = shared_runtime.get("kernel_signatures") or {}
        runtime_contract = {
            "execution_backend": runtime.get("execution_backend"),
            "nopython": bool(runtime.get("nopython")),
            "object_mode": int(runtime.get("object_mode") or 0),
            "python_fallback": int(runtime.get("python_fallback") or 0),
            "request_time_compilation": int(runtime.get("request_time_compilation") or 0),
            "regime_graph_kernel_version": runtime.get("kernel_version"),
            "regime_graph_signature_fingerprint": _content_hash(
                {kernel_id: graph_signatures.get(kernel_id, []) for kernel_id in kernel_ids}
            ),
            "historical_engine_version": shared_runtime.get("engine_version"),
            "historical_kernel_version": shared_runtime.get("kernel_version"),
            "historical_signature_fingerprint": _content_hash(
                {
                    kernel_id: shared_signatures.get(kernel_id, [])
                    for kernel_id in shared_kernel_ids
                }
            ),
        }
        manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "manifest_kind": "regime_graph_v2_execution_plan",
            "plan_id": public_plan["plan_id"],
            "graph_hash": public_plan["graph_hash"],
            "preparation_hash": public_plan["preparation_hash"],
            "registry_version": public_plan["registry_version"],
            "topological_order": copy.deepcopy(public_plan["topological_order"]),
            "kernel_ids": kernel_ids,
            "shared_kernel_ids": shared_kernel_ids,
            "all_kernel_ids": copy.deepcopy(public_plan["all_kernel_ids"]),
            "formula_nodes": self._formula_manifest(formula_plans),
            "runtime_contract": runtime_contract,
            "token_policy": {
                "storage": "process_local_memory_only",
                "authorizes_execution": False,
                "ttl_hours": 8,
            },
            "manifest_hash_scope": "execution_recipe_without_definition_bindings_v1",
            "definition_bindings": [],
        }
        if definition.id:
            manifest["definition_bindings"].append(
                {
                    "definition_id": definition.id,
                    "revision": int(definition.revision or 1),
                }
            )
        hash_payload = copy.deepcopy(manifest)
        hash_payload.pop("definition_bindings", None)
        manifest["manifest_content_hash"] = _content_hash(hash_payload)
        return manifest

    def _persist_plan_manifest(
        self,
        definition: RegimeDefinitionV2,
        public_plan: Mapping[str, Any],
        runtime: Mapping[str, Any],
        formula_plans: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not definition.id:
            return None
        return self.plan_manifests.upsert(
            self._build_plan_manifest(
                definition,
                public_plan,
                runtime,
                formula_plans,
            )
        )

    def create_definition(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        definition = parse_definition_v2(self._freeze_source_versions(payload))
        validate_definition_v2(definition)
        fields = definition.model_dump(
            mode="json",
            exclude={"id", "revision", "created_at", "updated_at"},
        )
        return self.definitions.create(fields)

    def list_definitions(self) -> list[dict[str, Any]]:
        return self.definitions.list()

    def get_definition(self, definition_id: str, revision: int | None = None) -> dict[str, Any]:
        return self.definitions.get(definition_id, revision)

    def update_definition(
        self,
        definition_id: str,
        expected_revision: int,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        definition = parse_definition_v2(self._freeze_source_versions(payload))
        validate_definition_v2(definition)
        fields = definition.model_dump(
            mode="json",
            exclude={"id", "revision", "created_at", "updated_at"},
        )
        return self.definitions.update(definition_id, expected_revision, fields)

    def copy_v1_definition(self, source: Mapping[str, Any]) -> dict[str, Any]:
        draft, warnings = migrate_v1_definition(source)
        created = self.create_definition(draft)
        definition = parse_definition_v2(created)
        return {
            "source_v1": copy.deepcopy(definition.source_v1),
            "definition": created,
            "inference": validate_definition_v2(definition),
            "warnings": warnings,
        }

    def prewarm_saved_definitions(self) -> dict[str, Any]:
        prepared: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for item in self.definitions.list():
            try:
                definition = parse_definition_v2(item)
                expected_preparation_hash = self._preparation_hash(definition)
                try:
                    prior_manifest = self.plan_manifests.get(expected_preparation_hash)
                except NotFoundError:
                    prior_manifest = None
                plan = self.prepare(item)
                manifest = self.plan_manifests.get(plan["preparation_hash"])
                prepared.append(
                    {
                        "definition_id": item.get("id"),
                        "revision": item.get("revision"),
                        "plan_id": plan["plan_id"],
                        "graph_hash": plan["graph_hash"],
                        "preparation_hash": plan["preparation_hash"],
                        "manifest_content_hash": manifest["manifest_content_hash"],
                        "manifest_status": (
                            "verified_and_rewarmed"
                            if prior_manifest is not None
                            and prior_manifest.get("manifest_content_hash")
                            == manifest.get("manifest_content_hash")
                            else "created_or_refreshed"
                        ),
                    }
                )
            except IndicatorDomainError as exc:
                errors.append(
                    {
                        "definition_id": item.get("id"),
                        "revision": item.get("revision"),
                        "error": exc.detail(),
                    }
                )
        result = {
            "complete": not errors,
            "definition_count": len(prepared) + len(errors),
            "prepared_count": len(prepared),
            "prepared": prepared,
            "errors": errors,
            "request_time_compilation": 0,
            "manifest_authorizes_execution": False,
            "token_storage": "process_local_memory_only",
        }
        self.startup_prewarm = copy.deepcopy(result)
        return result

    def prepare(self, payload: Mapping[str, Any], *, preview_target: Mapping[str, Any] | None = None) -> dict[str, Any]:
        definition = node_preview_definition(payload, preview_target) if preview_target is not None else parse_definition_v2(payload)
        inspection = validate_definition_v2(definition)
        unavailable_nodes = [
            node
            for node in definition.graph.nodes
            if NODE_REGISTRY[node.type].get("available") is False
        ]
        if unavailable_nodes:
            first = unavailable_nodes[0]
            metadata = NODE_REGISTRY[first.type]
            raise ValidationError(
                (
                    "THIRD_PARTY_MODEL_ADAPTER_UNAVAILABLE"
                    if first.type == "model.external_optimized"
                    else "REGIME_GRAPH_NODE_UNAVAILABLE"
                ),
                str(metadata.get("unavailable_reason") or "该节点当前不可执行。"),
                f"graph.nodes.{first.id}",
            )
        runtime = regime_graph_numba_status()
        if runtime.get("complete") is not True:
            raise ConflictError("REGIME_GRAPH_RUNTIME_NOT_READY", "历史情景图谱 NJIT 内核尚未完成预热。")
        graph_hash = inspection["graph_hash"]
        formula_plans = self._prepare_formula_nodes(definition)
        preparation_hash = self._preparation_hash(definition, formula_plans)
        shared_kernel_ids = _shared_kernel_ids_for_definition(definition)
        if shared_kernel_ids:
            shared_runtime = historical_regime_numba_status()
            if shared_runtime.get("complete") is not True:
                shared_runtime = warm_historical_regime_numba_kernels()
            missing_shared_kernel_ids = sorted(
                set(shared_kernel_ids)
                - set(shared_runtime.get("kernel_signatures") or {})
            )
            if shared_runtime.get("complete") is not True or missing_shared_kernel_ids:
                raise ConflictError(
                    "REGIME_GRAPH_SHARED_RUNTIME_NOT_READY",
                    "历史情景模型 NJIT 内核尚未完成预热或覆盖不完整。",
                )
        with self._lock:
            existing_token = self._plans_by_graph_hash.get(preparation_hash)
            existing = self._plans.get(existing_token or "")
            if existing is not None and existing["expires_at"] > _utc_now():
                self._persist_plan_manifest(
                    definition,
                    existing["public"],
                    runtime,
                    formula_plans,
                )
                return copy.deepcopy(existing["public"])
            if existing_token:
                self._plans.pop(existing_token, None)
                self._plans_by_graph_hash.pop(preparation_hash, None)
            now = _utc_now()
            token = f"rg2-{secrets.token_urlsafe(32)}"
            plan_id = f"regime-plan-{graph_hash[:24]}"
            kernel_ids = _kernel_ids_for_definition(definition)
            missing_kernel_ids = sorted(
                set(kernel_ids) - set(runtime.get("kernel_signatures") or {})
            )
            if missing_kernel_ids:
                raise ConflictError(
                    "REGIME_GRAPH_KERNEL_COVERAGE_INCOMPLETE",
                    f"图谱执行计划缺少已预热内核: {', '.join(missing_kernel_ids)}。",
                )
            public = {
                "schema_version": "2.0",
                "plan_id": plan_id,
                "compile_token": token,
                "graph_hash": graph_hash,
                "preparation_hash": preparation_hash,
                "registry_version": REGISTRY_VERSION,
                "prepared_at": _iso(now),
                "expires_at": _iso(now + timedelta(hours=8)),
                "topological_order": inspection["topological_order"],
                "kernel_ids": kernel_ids,
                "shared_kernel_ids": shared_kernel_ids,
                "all_kernel_ids": [
                    *kernel_ids,
                    *(f"historical:{kernel_id}" for kernel_id in shared_kernel_ids),
                ],
                "formula_plans": copy.deepcopy(formula_plans),
                "runtime_audit": copy.deepcopy(runtime),
                "request_time_compilation": 0,
                "python_fallback": 0,
            }
            self._persist_plan_manifest(definition, public, runtime, formula_plans)
            self._plans[token] = {
                "public": public,
                "graph_hash": graph_hash,
                "preparation_hash": preparation_hash,
                "expires_at": now + timedelta(hours=8),
            }
            self._plans_by_graph_hash[preparation_hash] = token
            return copy.deepcopy(public)

    def _validate_plan(self, definition: RegimeDefinitionV2, compile_token: str | None) -> dict[str, Any]:
        if not compile_token:
            raise ValidationError(
                "REGIME_GRAPH_PREPARE_REQUIRED",
                "运行前必须先准备图谱；正式请求不会临时编译。",
                "compile_token",
            )
        inspection = validate_definition_v2(definition)
        preparation_hash = self._preparation_hash(definition)
        with self._lock:
            plan = self._plans.get(compile_token)
            if plan is None or plan["expires_at"] <= _utc_now():
                raise ValidationError(
                    "REGIME_GRAPH_PLAN_NOT_WARM",
                    "准备令牌不存在或已过期，请重新准备图谱。",
                    "compile_token",
                )
            if plan["graph_hash"] != inspection["graph_hash"]:
                raise ValidationError(
                    "REGIME_GRAPH_PLAN_MISMATCH",
                    "准备令牌与当前图谱结构不匹配。",
                    "compile_token",
                )
            if plan.get("preparation_hash") != preparation_hash:
                raise ValidationError(
                    "REGIME_GRAPH_PLAN_MISMATCH",
                    "准备令牌与当前公式或图谱结构不匹配。",
                    "compile_token",
                )
            return copy.deepcopy(plan["public"])

    def _prune_jobs(self) -> None:
        now = _utc_now()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job["status"] in TERMINAL_JOB_STATUSES and job["expires_at"] <= now
        ]
        for job_id in expired:
            del self._jobs[job_id]

    def create_preview(
        self,
        payload: Mapping[str, Any],
        *,
        compile_token: str | None,
        mode: str = "realtime",
        as_of: str | None = None,
        ttl_seconds: int = DEFAULT_PREVIEW_TTL_SECONDS,
        preview_target: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if mode not in {"realtime", "retrospective"}:
            raise ValidationError("INVALID_RUN_MODE", "mode 必须是 realtime 或 retrospective。", "mode")
        if ttl_seconds < 1 or ttl_seconds > 86400:
            raise ValidationError("INVALID_PREVIEW_TTL", "ttl_seconds 必须在 1 到 86400 之间。", "ttl_seconds")
        # Nested parameter rows must not share objects with the caller while queued.
        definition = (node_preview_definition(payload, preview_target) if preview_target is not None else parse_definition_v2(payload)).model_copy(deep=True)
        self._validate_realtime_graph(definition, mode)
        plan = self._validate_plan(definition, compile_token)
        with self._lock:
            self._prune_jobs()
            if len(self._jobs) >= MAX_PREVIEW_JOBS:
                raise ConflictError("PREVIEW_QUEUE_FULL", "试算队列已满，请稍后重试。")
            now = _utc_now()
            job_id = f"regime-preview-{uuid.uuid4().hex}"
            job = {
                "id": job_id,
                "schema_version": "2.0",
                "status": "queued",
                "stage": "queued",
                "progress": 0.0,
                "message": "试算已进入队列。",
                "created_at": _iso(now),
                "updated_at": _iso(now),
                "expires_at": now + timedelta(seconds=ttl_seconds),
                "ttl_seconds": ttl_seconds,
                "mode": mode,
                "as_of": as_of,
                "plan_id": plan["plan_id"],
                "graph_hash": plan["graph_hash"],
                "definition_hash": definition_content_hash(definition),
                "definition": definition,
                "cancel_event": threading.Event(),
                "result": None,
                "error": None,
                "_series": None,
                "_node_outputs": None,
                "_node_types": None,
                "_plan": plan,
                "preview_target": copy.deepcopy(preview_target),
            }
            self._jobs[job_id] = job
            thread = threading.Thread(
                target=self._execute_job,
                args=(job_id,),
                daemon=True,
                name=f"regime-preview-{job_id[-8:]}",
            )
            job["thread"] = thread
            thread.start()
            return self._public_job(job)

    def _public_job(self, job: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: copy.deepcopy(value)
            for key, value in job.items()
            if not key.startswith("_")
            and key not in {"definition", "cancel_event", "thread", "expires_at"}
        } | {"expires_at": _iso(job["expires_at"])}

    def _update_job(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job["status"] == "cancelled" and fields.get("status") != "cancelled":
                return
            job.update(fields)
            job["updated_at"] = _iso(_utc_now())

    def _check_cancelled(self, job: Mapping[str, Any]) -> None:
        if job["cancel_event"].is_set():
            raise PreviewCancelled()

    def _execute_job(self, job_id: str) -> None:
        try:
            with self._lock:
                job = self._jobs[job_id]
                definition = job["definition"]
                mode = job["mode"]
                as_of = job["as_of"]
            self._update_job(job_id, status="preparing", stage="resolving_sources", progress=0.05, message="正在解析分类输入。")
            execution = self._execute_graph(
                job_id,
                definition,
                mode,
                as_of,
                plan=job["_plan"],
            )
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job["status"] == "cancelled":
                    return
                now = _utc_now()
                job["status"] = "completed"
                job["stage"] = "completed"
                job["progress"] = 1.0
                job["message"] = "试算完成。"
                job["result"] = execution["result"]
                job["_series"] = execution["series"]
                job["_node_outputs"] = execution["node_outputs"]
                job["_node_types"] = execution["node_types"]
                job["expires_at"] = now + timedelta(seconds=int(job["ttl_seconds"]))
                job["updated_at"] = _iso(now)
        except PreviewCancelled:
            self._update_job(job_id, status="cancelled", stage="cancelled", progress=1.0, message="试算已取消。")
        except IndicatorDomainError as exc:
            self._update_job(
                job_id,
                status="failed",
                stage="failed",
                progress=1.0,
                message=exc.message,
                error=exc.detail(),
            )
        except Exception:
            self._update_job(
                job_id,
                status="failed",
                stage="failed",
                progress=1.0,
                message="历史情景图谱试算失败。",
                error={"code": "REGIME_GRAPH_EXECUTION_FAILED", "message": "历史情景图谱试算失败。"},
            )

    def _resolve_sources(
        self,
        definition: RegimeDefinitionV2,
        required: set[str],
        mode: str,
        as_of: str | None,
        source_cache: dict[str, DataBundle] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        bundles: dict[str, Any] = {}
        snapshots: dict[str, Any] = {}
        run_cache = source_cache if source_cache is not None else {}
        # Retrospective/latest selection must happen after the requested
        # cutoff; selecting the final vintage first can erase the vintage that
        # was actually available at that historical date.
        resolver_as_of = as_of if mode == "retrospective" else None
        for node in definition.graph.nodes:
            if (
                node.id not in required
                or not node.type.startswith("source.")
                or node.type == "source.constant"
            ):
                continue
            cache_key = _canonical_source_cache_key(
                _source_spec(node.type, node.parameters),
                mode,
                resolver_as_of,
            )
            full_bundle = run_cache.get(cache_key)
            if full_bundle is None:
                if node.type == "source.upload":
                    full_bundle = self._upload_bundle(node.parameters, mode, resolver_as_of)
                elif node.type == "source.macro":
                    data_root = self._bound_source_root(node.type, node.parameters)
                    full_bundle = _macro_bundle(node.parameters, mode, resolver_as_of, data_root)
                else:
                    data_root = (
                        self._bound_source_root(node.type, node.parameters)
                        if node.type in {"source.index", "source.etf", "source.fund"}
                        else self.market_data_dir
                    )
                    full_bundle = resolve_target(
                        _source_spec(node.type, node.parameters),
                        mode,
                        resolver_as_of,
                        data_root,
                        self.indicator_service,
                    )
                run_cache[cache_key] = full_bundle
            if resolver_as_of is not None:
                resolver_snapshot = copy.deepcopy(full_bundle.snapshot)
                resolver_snapshot["as_of"] = resolver_as_of
                bundle = DataBundle(
                    frame=full_bundle.frame,
                    snapshot=resolver_snapshot,
                )
            else:
                bundle = self._slice_cached_bundle(full_bundle, as_of)
            if node.type == "source.indicator" and node.parameters.get("data_fingerprint"):
                actual_fingerprint = str(bundle.snapshot.get("fingerprint") or "")
                if actual_fingerprint != str(node.parameters["data_fingerprint"]):
                    raise ValidationError(
                        "INDICATOR_DATA_FINGERPRINT_MISMATCH",
                        "指标底层数据已变化，与已保存定义不一致。",
                        f"graph.nodes.{node.id}.parameters.data_fingerprint",
                    )
            bundles[node.id] = bundle
            snapshots[node.id] = copy.deepcopy(bundle.snapshot)
        if not bundles:
            raise ValidationError("MISSING_SOURCE_NODE", "图谱没有可执行的数据源。", "graph.nodes")
        return bundles, snapshots

    @staticmethod
    def _slice_cached_bundle(bundle: DataBundle, as_of: str | None) -> DataBundle:
        if not as_of:
            return bundle
        cutoff = _parse_date(as_of, "as_of")
        frame = bundle.frame.loc[bundle.frame["available_at"] <= cutoff].copy()
        if frame.empty:
            raise ValidationError("NO_DATA_AS_OF", "截至所选日期尚无可得数据。", "as_of")
        snapshot = copy.deepcopy(bundle.snapshot)
        snapshot.update(
            {
                "selected_observations": int(len(frame)),
                "first_observation_date": frame["observation_date"].iloc[0].date().isoformat(),
                "last_observation_date": frame["observation_date"].iloc[-1].date().isoformat(),
                "latest_available_at": frame["available_at"].max().date().isoformat(),
                "as_of": cutoff.date().isoformat(),
            }
        )
        return DataBundle(frame=frame, snapshot=snapshot)

    def _input(
        self,
        node_outputs: Mapping[str, Mapping[str, PortValue]],
        reference: Any,
    ) -> PortValue:
        return node_outputs[reference.node_id][reference.port]

    @staticmethod
    def _port(values: np.ndarray, source: PortValue) -> PortValue:
        return PortValue(np.ascontiguousarray(values), source.dates, source.available)

    @staticmethod
    def _same_axis(*ports: PortValue) -> None:
        if not ports:
            return
        first = ports[0]
        if any(port.dates is not first.dates for port in ports[1:]):
            raise ValidationError(
                "EXPLICIT_ALIGNMENT_REQUIRED",
                "来自不同时间轴的输入必须先经过显式对齐节点。",
                "graph.nodes.inputs",
            )

    @staticmethod
    def _take_port(port: PortValue, positions: np.ndarray, dates: np.ndarray, available: np.ndarray) -> PortValue:
        if port.values.ndim == 2:
            values = take_matrix_rows_kernel(
                np.ascontiguousarray(port.values, dtype=np.float64),
                positions,
            )
        elif np.issubdtype(port.values.dtype, np.integer):
            values = take_int64_kernel(
                np.ascontiguousarray(port.values, dtype=np.int64),
                positions,
            )
        else:
            values = take_float_kernel(
                np.ascontiguousarray(port.values, dtype=np.float64),
                positions,
            )
        return PortValue(values, dates, available)

    def _strict_align_ports(self, ports: list[PortValue]) -> list[PortValue]:
        if len(ports) < 2:
            return ports
        aligned = [ports[0]]
        common_dates = ports[0].dates
        common_available = ports[0].available
        for next_port in ports[1:]:
            next_dates, left_positions, right_positions = strict_intersection_indices_kernel(
                common_dates,
                next_port.dates,
            )
            left_available = take_int64_kernel(common_available, left_positions)
            right_available = take_int64_kernel(next_port.available, right_positions)
            combined_available = maximum_int64_kernel(left_available, right_available)
            aligned = [
                self._take_port(port, left_positions, next_dates, combined_available)
                for port in aligned
            ]
            aligned.append(
                self._take_port(next_port, right_positions, next_dates, combined_available)
            )
            common_dates = next_dates
            common_available = combined_available
        if common_dates.shape[0] < MIN_OBSERVATIONS:
            raise ValidationError(
                "INSUFFICIENT_ALIGNED_OBSERVATIONS",
                "显式对齐后至少需要 5 条共同观测。",
                "graph.nodes",
            )
        return aligned

    def _execute_latent_model(
        self,
        node: Any,
        feature_port: PortValue,
        state_count: int,
        mode: str,
        training_as_of: str | None,
        model_audits: dict[str, Any],
    ) -> dict[str, PortValue]:
        matrix = np.ascontiguousarray(feature_port.values, dtype=np.float64)
        positions = finite_row_positions_kernel(matrix)
        components = min(int(node.parameters.get("components", 3)), state_count)
        initial_train_size = int(node.parameters.get("initial_train_size", 60))
        iterations = int(node.parameters.get("iterations", 60))
        if training_as_of is not None:
            cutoff = np.int64(_parse_date(training_as_of, "training_as_of").value)
            training_positions, classify_positions = split_positions_at_available_kernel(
                positions,
                np.ascontiguousarray(feature_port.available, dtype=np.int64),
                cutoff,
            )
            training_count = int(training_positions.shape[0])
            minimum = components * 5
            if training_count < minimum or classify_positions.shape[0] < 1:
                raise ValidationError(
                    "INSUFFICIENT_WALK_FORWARD_MODEL_DATA",
                    "走步折的训练样本或样本外区间不足。",
                    f"graph.nodes.{node.id}",
                )
        else:
            minimum = max(
                components * 5,
                initial_train_size + (1 if mode == "realtime" else 0),
            )
            if positions.shape[0] < minimum:
                raise ValidationError(
                    "INSUFFICIENT_MODEL_DATA",
                    "隐状态模型有效样本不足。",
                    f"graph.nodes.{node.id}",
                )
            training_count = (
                initial_train_size if mode == "realtime" else int(positions.shape[0])
            )
            training_positions = np.ascontiguousarray(
                positions[:training_count],
                dtype=np.int64,
            )
            classify_positions = np.ascontiguousarray(
                positions[training_count:] if mode == "realtime" else positions,
                dtype=np.int64,
            )
        training_raw = take_matrix_rows_kernel(matrix, training_positions)
        training, mean, scale = standardize_fit_kernel(training_raw)
        standardized = apply_standardization_kernel(matrix, mean, scale)
        classify = take_matrix_rows_kernel(standardized, classify_positions)
        initialization_strategy = str(
            node.parameters.get("initialization_strategy", "quantile")
        )
        strategy_codes = {"quantile": 0, "random": 1, "explicit": 2}
        if initialization_strategy not in strategy_codes:
            raise ValidationError(
                "INVALID_MODEL_INITIALIZATION_STRATEGY",
                "隐状态模型初始化策略不受支持。",
                f"graph.nodes.{node.id}.parameters.initialization_strategy",
            )
        random_seed = int(node.parameters.get("random_seed", 0))
        explicit_means = np.empty((0, 0), dtype=np.float64)
        if initialization_strategy == "explicit":
            try:
                explicit_means = np.ascontiguousarray(
                    node.parameters.get("initial_means", []),
                    dtype=np.float64,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValidationError(
                    "INVALID_MODEL_INITIAL_MEANS",
                    "显式初始中心必须是有限数值矩阵。",
                    f"graph.nodes.{node.id}.parameters.initial_means",
                ) from exc
            if (
                explicit_means.ndim != 2
                or explicit_means.shape != (components, training.shape[1])
                or not np.isfinite(explicit_means).all()
            ):
                raise ValidationError(
                    "INVALID_MODEL_INITIAL_MEANS",
                    "显式初始中心形状必须为 状态数 × 特征数，且全部为有限数值。",
                    f"graph.nodes.{node.id}.parameters.initial_means",
                )
        initialized_means = initialize_gaussian_means_kernel(
            np.ascontiguousarray(training, dtype=np.float64),
            np.int64(components),
            np.int64(strategy_codes[initialization_strategy]),
            np.int64(random_seed),
            explicit_means,
        )
        if node.type == "model.gmm":
            weights, means, variances, _ = gmm_fit_kernel(
                training,
                components,
                iterations,
                initialized_means,
            )
            posterior = gmm_posterior_kernel(classify, weights, means, variances)
            estimation_method = "gaussian_mixture_em"
        elif node.type == "model.markov":
            initial, transition, means, variances, _ = markov_fit_kernel(
                training,
                components,
                iterations,
                initialized_means,
            )
            if mode == "realtime":
                posterior = markov_filtered_posterior_kernel(
                    training,
                    classify,
                    initial,
                    transition,
                    means,
                    variances,
                )
            else:
                posterior = markov_smoothed_posterior_kernel(
                    classify,
                    initial,
                    transition,
                    means,
                    variances,
                )
            estimation_method = "hard_gaussian_states_with_markov_transitions"
        else:
            initial, transition, means, variances, _ = hmm_fit_kernel(
                training,
                components,
                iterations,
                initialized_means,
            )
            if mode == "realtime":
                posterior = hmm_filtered_posterior_kernel(
                    training, classify, initial, transition, means, variances
                )
            else:
                posterior = hmm_smoothed_posterior_kernel(
                    classify, initial, transition, means, variances
                )
            estimation_method = "hidden_markov_baum_welch"
        assignments, _ = posterior_assignment_kernel(posterior)
        order = component_order_kernel(means)
        model_audits[node.id] = {
            "node_id": node.id,
            "model_type": node.type,
            "estimation_method": estimation_method,
            "fit_mode": (
                "initial_training_interval_locked"
                if mode == "realtime"
                else "full_sample_retrospective"
            ),
            "training_count": int(training_count),
            "training_start_index": int(training_positions[0]),
            "training_end_index": int(training_positions[-1]),
            "classification_start_index": (
                int(classify_positions[0]) if classify_positions.shape[0] else None
            ),
            "training_cutoff_available_at": training_as_of,
            "walk_forward_refit": training_as_of is not None,
            "components": int(components),
            "initialization_strategy": initialization_strategy,
            "random_seed": random_seed,
            "initial_means": initialized_means.tolist(),
            "initialization_fingerprint": _content_hash(initialized_means),
            "initialization_backend": "numba_njit_fixed_signature",
            "component_order": [int(item) for item in order.tolist()],
            "label_mapping_policy": "training_mean_order",
            "label_mapping_locked_before_classification": mode == "realtime",
            "uses_full_sample_for_label_mapping": mode != "realtime",
        }
        states, confidence, probabilities = scatter_component_model_kernel(
            assignments,
            posterior,
            order,
            classify_positions,
            np.int64(matrix.shape[0]),
            np.int64(state_count),
        )
        score = matrix_column_kernel(matrix, np.int64(0))
        return {
            "state": self._port(states, feature_port),
            "score": self._port(score, feature_port),
            "confidence": self._port(confidence, feature_port),
            "probabilities": self._port(probabilities, feature_port),
        }

    def _execute_numeric_node(
        self,
        node: Any,
        node_outputs: Mapping[str, Mapping[str, PortValue]],
        state_count: int,
        mode: str,
        as_of: str | None,
        latent_training_as_of: str | None,
        formula_plans: Mapping[str, Any],
        formula_audits: dict[str, Any],
        model_audits: dict[str, Any],
    ) -> dict[str, PortValue]:
        node_type = node.type
        parameters = node.parameters
        if node_type == "source.constant":
            anchor = self._input(node_outputs, node.inputs["anchor"])
            return {"value": self._port(constant_like_kernel(np.ascontiguousarray(anchor.values, dtype=np.float64), np.float64(parameters.get("value", 0.0))), anchor)}
        if node_type in {"align.strict_intersection", "align.pit_asof"}:
            if node_type == "align.strict_intersection":
                left, right = self._strict_align_ports([
                    self._input(node_outputs, node.inputs["left"]),
                    self._input(node_outputs, node.inputs["right"]),
                ])
                return {"left": left, "right": right}
            anchor = self._input(node_outputs, node.inputs["anchor"])
            feature = self._input(node_outputs, node.inputs["feature"])
            feature_order = stable_time_order_kernel(
                np.ascontiguousarray(feature.available, dtype=np.int64)
            )
            sorted_feature = self._take_port(
                feature,
                feature_order,
                take_int64_kernel(feature.dates, feature_order),
                take_int64_kernel(feature.available, feature_order),
            )
            positions = pit_asof_positions_kernel(
                anchor.dates,
                sorted_feature.available,
                np.int64(int(parameters.get("max_age_days", 400))),
            )
            available = aligned_available_kernel(
                anchor.available,
                sorted_feature.available,
                positions,
            )
            return {
                "anchor": PortValue(anchor.values, anchor.dates, available),
                "feature": PortValue(
                    take_float_or_nan_kernel(
                        np.ascontiguousarray(sorted_feature.values, dtype=np.float64),
                        positions,
                    ),
                    anchor.dates,
                    available,
                ),
            }
        if node_type == "align.resample":
            source = self._input(node_outputs, node.inputs["value"])
            if "frequency" not in parameters and "every" in parameters:
                positions = resample_positions_kernel(
                    np.int64(source.dates.shape[0]),
                    np.int64(int(parameters.get("every", 5))),
                    np.int64(int(parameters.get("offset", 0))),
                )
                dates = take_int64_kernel(source.dates, positions)
                available = take_int64_kernel(source.available, positions)
                return {"value": self._take_port(source, positions, dates, available)}
            frequency_name = str(parameters.get("frequency", "weekly"))
            aggregation_name = str(parameters.get("aggregation", "last"))
            frequency_code = {
                "daily": 0,
                "weekly": 1,
                "monthly": 2,
                "quarterly": 3,
                "yearly": 4,
            }[frequency_name]
            aggregation_code = {
                "first": 0,
                "last": 1,
                "mean": 2,
                "sum": 3,
            }[aggregation_name]
            values, dates, available = calendar_resample_kernel(
                np.ascontiguousarray(source.values, dtype=np.float64),
                np.ascontiguousarray(source.dates, dtype=np.int64),
                np.ascontiguousarray(source.available, dtype=np.int64),
                np.int64(frequency_code),
                np.int64(aggregation_code),
            )
            if (
                mode == "realtime"
                and frequency_name != "daily"
                and aggregation_name in {"last", "mean", "sum"}
                and values.shape[0] > 0
            ):
                cutoff_ns = np.int64(
                    _parse_date(as_of, "as_of").value
                    if as_of
                    else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize().value
                )
                final_bucket = calendar_bucket_kernel(dates[-1], np.int64(frequency_code))
                cutoff_bucket = calendar_bucket_kernel(cutoff_ns, np.int64(frequency_code))
                if final_bucket == cutoff_bucket:
                    # A partial final calendar bucket would change when new points arrive.
                    values = np.ascontiguousarray(values[:-1], dtype=np.float64)
                    dates = np.ascontiguousarray(dates[:-1], dtype=np.int64)
                    available = np.ascontiguousarray(available[:-1], dtype=np.int64)
            return {"value": PortValue(values, dates, available)}
        if node_type in {"align.cross_section", "feature.matrix"}:
            ports = [
                self._input(node_outputs, node.inputs[name])
                for name in (("feature_1", "feature_2", "feature_3", "feature_4") if node_type == "feature.formula" else node.inputs)
                if name in node.inputs
            ]
            if node_type == "align.cross_section":
                ports = self._strict_align_ports(ports)
            else:
                self._same_axis(*ports)
            filler = ports[0]
            padded = ports + [filler] * (4 - len(ports))
            matrix = feature_matrix_kernel(
                *(np.ascontiguousarray(port.values, dtype=np.float64) for port in padded),
                np.int64(len(ports)),
            )
            return {"features": self._port(matrix, ports[0])}
        if is_typed_formula_node(node, NODE_REGISTRY):
            ports = [
                self._input(node_outputs, node.inputs[name])
                for name in (("feature_1", "feature_2", "feature_3", "feature_4") if node_type == "feature.formula" else node.inputs)
                if name in node.inputs
            ]
            self._same_axis(*ports)
            columns = {
                    name: np.ascontiguousarray(
                        self._input(node_outputs, reference).values,
                        dtype=np.float64,
                    )
                    for name, reference in node.inputs.items()
                }
            aliases = parameters.get("variables")
            if isinstance(aliases, Mapping):
                for alias, port_name in aliases.items():
                    if str(alias).isidentifier() and str(port_name) in columns:
                        columns[str(alias)] = columns[str(port_name)]
            frame = pd.DataFrame(columns)
            results = {}
            for output_port, expression in typed_node_expressions(node, NODE_REGISTRY).items():
                key = formula_plan_key(node.id, output_port)
                prepared = formula_plans.get(key)
                if not isinstance(prepared, Mapping) or not prepared.get("compile_token"):
                    raise ValidationError("FORMULA_NJIT_PLAN_NOT_WARMED", "指标或公式节点没有可执行的预热计划。", f"graph.nodes.{node.id}")
                formula_result = evaluate_formula(expression, frame, compile_token=str(prepared["compile_token"]))
                results[output_port] = np.ascontiguousarray(formula_result.values.to_numpy(dtype=np.float64))
                formula_audits[key] = copy.deepcopy(formula_result.audit)
            available = ports[0].available
            for port in ports[1:]:
                available = maximum_int64_kernel(available, port.available)
            # Conservative for every dependency in the causal prefix, including
            # late publication of an earlier input consumed by rolling/scan nodes.
            available = causal_available_kernel(np.ascontiguousarray(available, dtype=np.int64))
            return {port: PortValue(values, ports[0].dates, available) for port, values in results.items()}
        if node_type in {"transform.identity", "transform.log", "transform.lag", "transform.diff", "transform.return", "transform.yoy", "transform.mom"}:
            source = self._input(node_outputs, node.inputs["value"])
            opcode = {
                "transform.identity": 0,
                "transform.log": 1,
                "transform.diff": 2,
                "transform.return": 3,
                "transform.yoy": 3,
                "transform.mom": 3,
                "transform.lag": 4,
            }[node_type]
            period = int(parameters.get("periods", parameters.get("window", 1)))
            return {"value": self._port(unary_transform_kernel(np.ascontiguousarray(source.values, dtype=np.float64), np.int64(opcode), np.int64(period)), source)}
        if node_type == "transform.drawdown":
            source = self._input(node_outputs, node.inputs["value"])
            return {"value": self._port(drawdown_series_kernel(np.ascontiguousarray(source.values, dtype=np.float64)), source)}
        if node_type == "transform.standardize":
            source = self._input(node_outputs, node.inputs["value"])
            return {"value": self._port(rolling_kernel(np.ascontiguousarray(source.values, dtype=np.float64), np.int64(int(parameters.get("window", 20))), np.int64(2)), source)}
        if node_type == "transform.clip":
            source = self._input(node_outputs, node.inputs["value"])
            return {"value": self._port(clip_kernel(np.ascontiguousarray(source.values, dtype=np.float64), np.float64(parameters.get("lower", -3.0)), np.float64(parameters.get("upper", 3.0))), source)}
        if node_type.startswith("math."):
            left = self._input(node_outputs, node.inputs["left"])
            right = self._input(node_outputs, node.inputs["right"])
            self._same_axis(left, right)
            opcode = {"math.add": 0, "math.subtract": 1, "math.multiply": 2, "math.divide": 3}[node_type]
            return {"value": self._port(binary_math_kernel(np.ascontiguousarray(left.values, dtype=np.float64), np.ascontiguousarray(right.values, dtype=np.float64), np.int64(opcode)), left)}
        if node_type == "pivot.local_extrema":
            source = self._input(node_outputs, node.inputs["value"])
            pivots, prices = local_extrema_kernel(np.ascontiguousarray(source.values, dtype=np.float64),
                np.int64(parameters.get("left_window", 8)), np.int64(parameters.get("right_window", 8)),
                np.int64(parameters.get("head_window", 6)), np.int64(parameters.get("tail_window", 6)))
            return {"pivot": self._port(pivots, source), "pivot_price": self._port(prices, source)}
        if node_type == "segment.between_pivots":
            source = self._input(node_outputs, node.inputs["pivot"])
            starts, ends = between_pivots_kernel(np.ascontiguousarray(source.values, dtype=np.float64))
            return {"start": self._port(starts, source), "end": self._port(ends, source)}
        if node_type in STATISTIC_IDS:
            source, starts, ends = [self._input(node_outputs, node.inputs[name]) for name in ("value", "start", "end")]
            self._same_axis(source, starts, ends)
            values = interval_statistic_kernel(np.ascontiguousarray(source.values, dtype=np.float64),
                np.ascontiguousarray(starts.values, dtype=np.int64), np.ascontiguousarray(ends.values, dtype=np.int64),
                np.int64(STATISTIC_IDS[node_type]), np.int64(parameters.get("ddof", 1)))
            return {"value": self._port(values, source)}
        if node_type == "model.range_threshold":
            source = self._input(node_outputs, node.inputs["value"])
            limits = []
            for name, parameter, default in (("upper_bound", "upper", .03), ("lower_bound", "lower", -.03)):
                bound = self._input(node_outputs, node.inputs[name]) if name in node.inputs else self._port(
                    constant_like_kernel(source.values, np.float64(parameters.get(parameter, default))), source)
                self._same_axis(source, bound)
                limits.append(bound)
            states, invalid = range_threshold_kernel(np.ascontiguousarray(source.values, dtype=np.float64),
                np.ascontiguousarray(limits[0].values, dtype=np.float64), np.ascontiguousarray(limits[1].values, dtype=np.float64))
            if invalid:
                raise ValidationError("INVALID_THRESHOLDS", "每个有效时点的下界必须小于上界。", f"graph.nodes.{node.id}.inputs")
            available = maximum_int64_kernel(source.available, maximum_int64_kernel(limits[0].available, limits[1].available))
            return {"state": PortValue(states, source.dates, available)}
        if node_type == "model.peak_trough":
            if mode != "retrospective":
                raise ValidationError("NON_CAUSAL_REALTIME_GRAPH", "峰谷定界法依赖后续数据，仅限事后识别。", f"graph.nodes.{node.id}")
            source = self._input(node_outputs, node.inputs["value"])
            values = peak_trough_asymmetric_kernel(
                np.ascontiguousarray(source.values, dtype=np.float64),
                np.int64(parameters.get("left_window", parameters.get("window", 8))),
                np.int64(parameters.get("right_window", parameters.get("window", 8))),
                np.int64(parameters.get("min_phase", 4)), np.int64(parameters.get("min_cycle", 16)),
                np.int64(parameters.get("head_window", parameters.get("endpoint_window", 6))),
                np.int64(parameters.get("tail_window", parameters.get("endpoint_window", 6))),
                np.float64(parameters.get("amplitude_exception", 0.2)),
            )
            sideways = peak_trough_sideways_kernel(
                np.ascontiguousarray(source.values, dtype=np.float64), values[0], values[2], values[3],
                np.int64(parameters.get("sideways_enabled", False)),
                np.int64(2 if state_count == 3 else 1), np.int64(1 if state_count == 3 else -1),
                np.float64(parameters.get("small_swing_threshold", 0.03)),
                np.float64(parameters.get("sideways_max_range", 0.06)),
                np.float64(parameters.get("sideways_max_efficiency", 0.25)),
                np.int64(parameters.get("sideways_min_duration", 20)),
            )
            outputs = {name: self._port(value, source) for name, value in zip(
                ("state", "pivot", "phase_start_index", "phase_end_index", "phase_return", "boundary_line"), values)}
            outputs.update({name: self._port(value, source) for name, value in zip(
                ("state", "sideways_range", "sideways_efficiency", "sideways_start_index", "sideways_end_index", "sideways_swing_count"), sideways)})
            return outputs
        if node_type in {"filter.super_smoother", "filter.kama"}:
            source = self._input(node_outputs, node.inputs["value"])
            raw = np.ascontiguousarray(source.values, dtype=np.float64)
            if node_type == "filter.super_smoother":
                values = super_smoother_kernel(raw, np.int64(parameters.get("period", 126)))
            else:
                values = kama_kernel(raw, np.int64(parameters.get("window", 60)),
                                     np.int64(parameters.get("fast", 2)), np.int64(parameters.get("slow", 126)))
            return {"value": self._port(values, source)}
        if node_type == "feature.trend_metrics":
            source = self._input(node_outputs, node.inputs["log_price"])
            trend = self._input(node_outputs, node.inputs["trend"])
            self._same_axis(source, trend)
            values = trend_features_kernel(
                np.ascontiguousarray(source.values, dtype=np.float64),
                np.ascontiguousarray(trend.values, dtype=np.float64),
                np.int64(parameters.get("volatility_window", 60)), np.int64(parameters.get("slope_window", 20)),
                np.int64(parameters.get("efficiency_window", 60)), np.float64(parameters.get("scale_floor", 0.0001)),
                np.int64(parameters.get("shock_window", 5)), np.float64(parameters.get("drawdown_alert", 0.2)),
                np.float64(parameters.get("shock_alert", 0.08)),
            )
            return {name: self._port(value, source) for name, value in zip(
                ("distance", "slope", "efficiency", "scale", "drawdown", "risk", "index_value", "filtered_index"), values)}
        if node_type == "model.trend_regime":
            ports = [self._input(node_outputs, node.inputs[name]) for name in ("distance", "slope", "efficiency")]
            self._same_axis(*ports)
            values = trend_regime_kernel(
                *(np.ascontiguousarray(port.values, dtype=np.float64) for port in ports),
                np.float64(parameters.get("band", 1.0)), np.float64(parameters.get("trend_enter", 0.1)),
                np.float64(parameters.get("flat_threshold", 0.05)), np.float64(parameters.get("efficiency_ceiling", 0.25)),
                np.int64(parameters.get("confirmation", 3)),
            )
            return {name: self._port(value, ports[0]) for name, value in zip(
                ("state", "candidate", "pending_count", "phase"), values)}
        if node_type == "post.merge_short_regimes":
            # Defense in depth for direct node execution, in addition to the graph gate.
            if mode == "realtime":
                raise ValidationError("NON_CAUSAL_REALTIME_GRAPH", "实时识别禁止事后区间合并。", f"graph.nodes.{node.id}")
            source = self._input(node_outputs, node.inputs["state"])
            price = self._input(node_outputs, node.inputs["price"])
            self._same_axis(source, price)
            values = merge_short_regimes_kernel(
                np.ascontiguousarray(source.values, dtype=np.int64), np.ascontiguousarray(price.values, dtype=np.float64),
                np.int64(parameters.get("max_duration", 10)), np.float64(parameters.get("max_move", 0.08)),
                np.int64(parameters.get("following_confirmation", 3)),
            )
            return {"state": self._port(values, source)}
        if node_type in {"filter.ema", "filter.sma", "filter.kalman"}:
            source = self._input(node_outputs, node.inputs["value"])
            if node_type == "filter.ema":
                values = ema_kernel(np.ascontiguousarray(source.values, dtype=np.float64), np.int64(int(parameters.get("window", 20))))
            elif node_type == "filter.sma":
                values = rolling_kernel(np.ascontiguousarray(source.values, dtype=np.float64), np.int64(int(parameters.get("window", 20))), np.int64(0))
            else:
                values = kalman_filter_kernel(np.ascontiguousarray(source.values, dtype=np.float64), np.float64(parameters.get("process_variance", 1e-5)), np.float64(parameters.get("measurement_variance", 1e-2)))
            return {"value": self._port(values, source)}
        if node_type.startswith("rolling."):
            source = self._input(node_outputs, node.inputs["value"])
            opcode = {"rolling.mean": 0, "rolling.std": 1, "rolling.zscore": 2, "rolling.slope": 3, "rolling.min": 4, "rolling.max": 5}[node_type]
            return {"value": self._port(rolling_kernel(np.ascontiguousarray(source.values, dtype=np.float64), np.int64(int(parameters.get("window", 20))), np.int64(opcode)), source)}
        if node_type == "model.threshold":
            source = self._input(node_outputs, node.inputs["value"])
            score = np.ascontiguousarray(source.values, dtype=np.float64)
            states = threshold_state_kernel(score, np.float64(parameters.get("upper", 0.001)), np.float64(parameters.get("lower", -0.001)))
            recognition, _, reasons = temporal_output_kernel(states)
            return {"state": self._port(states, source), "score": self._port(unary_transform_kernel(score, np.int64(0), np.int64(1)), source), "confidence": self._port(state_confidence_kernel(states), source), "probabilities": self._port(state_probabilities_kernel(states, np.int64(state_count)), source), "recognition_index": self._port(recognition, source), "reason_code": self._port(reasons, source)}
        if node_type == "model.hysteresis":
            source = self._input(node_outputs, node.inputs["value"])
            score = np.ascontiguousarray(source.values, dtype=np.float64)
            states = hysteresis_state_kernel(
                score,
                np.float64(parameters.get("upper_enter", 0.0015)),
                np.float64(parameters.get("upper_exit", 0.0002)),
                np.float64(parameters.get("lower_enter", -0.0015)),
                np.float64(parameters.get("lower_exit", -0.0002)),
            )
            return {"state": self._port(states, source), "score": self._port(unary_transform_kernel(score, np.int64(0), np.int64(1)), source), "confidence": self._port(state_confidence_kernel(states), source), "probabilities": self._port(state_probabilities_kernel(states, np.int64(state_count)), source)}
        if node_type == "post.hysteresis":
            source = self._input(node_outputs, node.inputs["score"])
            score = np.ascontiguousarray(source.values, dtype=np.float64)
            states = hysteresis_state_kernel(
                score,
                np.float64(parameters.get("upper_enter", 0.0015)),
                np.float64(parameters.get("upper_exit", 0.0002)),
                np.float64(parameters.get("lower_enter", -0.0015)),
                np.float64(parameters.get("lower_exit", -0.0002)),
            )
            return {
                "state": self._port(states, source),
                "score": self._port(
                    unary_transform_kernel(score, np.int64(0), np.int64(1)),
                    source,
                ),
                "confidence": self._port(state_confidence_kernel(states), source),
                "probabilities": self._port(
                    state_probabilities_kernel(states, np.int64(state_count)),
                    source,
                ),
            }
        if node_type == "model.quadrant":
            growth = self._input(node_outputs, node.inputs["growth"])
            inflation = self._input(node_outputs, node.inputs["inflation"])
            self._same_axis(growth, inflation)
            states = quadrant_state_kernel(
                np.ascontiguousarray(growth.values, dtype=np.float64),
                np.ascontiguousarray(inflation.values, dtype=np.float64),
                np.float64(parameters.get("growth_threshold", 0.0)),
                np.float64(parameters.get("inflation_threshold", 0.0)),
            )
            score = binary_math_kernel(np.ascontiguousarray(growth.values, dtype=np.float64), np.ascontiguousarray(inflation.values, dtype=np.float64), np.int64(1))
            return {"state": self._port(states, growth), "score": self._port(score, growth), "confidence": self._port(state_confidence_kernel(states), growth), "probabilities": self._port(state_probabilities_kernel(states, np.int64(state_count)), growth)}
        if node_type == "model.turning_point":
            source = self._input(node_outputs, node.inputs["value"])
            states, recognition, confidence, segment_move, _ = turning_point_kernel(
                np.ascontiguousarray(source.values, dtype=np.float64),
                np.int64(int(parameters.get("window", 20))),
                np.float64(parameters.get("min_move", 0.08)),
            )
            _, _, reasons = temporal_output_kernel(states)
            return {"state": self._port(states, source), "score": self._port(segment_move, source), "confidence": self._port(confidence, source), "probabilities": self._port(state_probabilities_kernel(states, np.int64(state_count)), source), "recognition_index": self._port(recognition, source), "reason_code": self._port(reasons, source)}
        if node_type in {"model.hmm", "model.markov", "model.gmm"}:
            return self._execute_latent_model(
                node,
                self._input(node_outputs, node.inputs["features"]),
                state_count,
                mode,
                latent_training_as_of,
                model_audits,
            )
        if node_type == "model.change_point":
            source = self._input(node_outputs, node.inputs["value"])
            score, states, confidence, recognition, reasons, *_ = change_point_kernel(
                np.ascontiguousarray(source.values, dtype=np.float64),
                np.int64(int(parameters.get("window", 20))),
                np.float64(parameters.get("threshold", 1.5)),
                np.int64(int(parameters.get("confirmation", 2))),
                mode != "realtime",
            )
            return {"state": self._port(states, source), "score": self._port(score, source), "confidence": self._port(confidence, source), "probabilities": self._port(state_probabilities_kernel(states, np.int64(state_count)), source), "recognition_index": self._port(recognition, source), "reason_code": self._port(reasons, source)}
        if node_type == "model.ensemble":
            ports = [
                self._input(node_outputs, node.inputs[name])
                for name in ("state_1", "state_2", "state_3", "state_4")
                if name in node.inputs
            ]
            self._same_axis(*ports)
            padded = ports + [ports[0]] * (4 - len(ports))
            configured_weights = list(parameters.get("weights") or [])
            weights = np.ones(4, dtype=np.float64)
            for index in range(min(len(configured_weights), len(ports))):
                weights[index] = float(configured_weights[index])
            states, probabilities, confidence, _ = ensemble_state_kernel(
                *(np.ascontiguousarray(port.values, dtype=np.int64) for port in padded),
                np.ascontiguousarray(weights),
                np.int64(len(ports)),
                np.int64(state_count),
                np.float64(parameters.get("consensus_threshold", 0.6)),
            )
            return {"state": self._port(states, ports[0]), "confidence": self._port(confidence, ports[0]), "probabilities": self._port(probabilities, ports[0])}
        if node_type in {"post.confirmation", "post.min_duration"}:
            source = self._input(node_outputs, node.inputs["state"])
            states = confirmation_state_kernel(
                np.ascontiguousarray(source.values, dtype=np.int64),
                np.int64(int(parameters.get("confirmation", 1 if node_type == "post.min_duration" else 2))),
                np.int64(int(parameters.get("min_duration", 1))),
            )
            return {"state": self._port(states, source)}
        if node_type == "post.component_map":
            source = self._input(node_outputs, node.inputs["state"])
            mapping = np.ascontiguousarray(np.asarray(parameters.get("mapping"), dtype=np.int64))
            return {"state": self._port(component_map_kernel(np.ascontiguousarray(source.values, dtype=np.int64), mapping), source)}
        if node_type in {"post.priority", "post.conflict_reject"}:
            left_name, right_name = ("primary", "secondary") if node_type == "post.priority" else ("left", "right")
            left = self._input(node_outputs, node.inputs[left_name])
            right = self._input(node_outputs, node.inputs[right_name])
            self._same_axis(left, right)
            states = combine_states_kernel(np.ascontiguousarray(left.values, dtype=np.int64), np.ascontiguousarray(right.values, dtype=np.int64), np.int64(0 if node_type == "post.priority" else 1))
            return {"state": self._port(states, left)}
        if node_type == "post.confidence_gate":
            states_port = self._input(node_outputs, node.inputs["state"])
            confidence_port = self._input(node_outputs, node.inputs["confidence"])
            self._same_axis(states_port, confidence_port)
            states = confidence_gate_kernel(np.ascontiguousarray(states_port.values, dtype=np.int64), np.ascontiguousarray(confidence_port.values, dtype=np.float64), np.float64(parameters.get("floor", 0.5)))
            return {"state": self._port(states, states_port)}
        if node_type == "output.temporal":
            source = self._input(node_outputs, node.inputs["state"])
            recognition, effective, reasons = temporal_output_kernel(
                np.ascontiguousarray(source.values, dtype=np.int64)
            )
            return {
                "recognition_index": self._port(recognition, source),
                "effective_index": self._port(effective, source),
                "reason_code": self._port(reasons, source),
            }
        if node_type.startswith("output."):
            port_name = node_type.split(".", 1)[1]
            return {port_name: self._input(node_outputs, node.inputs[port_name])}
        if node_type == "model.external_optimized":
            raise ValidationError("THIRD_PARTY_MODEL_ADAPTER_UNAVAILABLE", "第三方模型适配器不可用。", f"graph.nodes.{node.id}")
        raise ValidationError("UNSUPPORTED_EXECUTION_NODE", f"节点 {node.type} 尚未接入执行器。", f"graph.nodes.{node.id}.type")

    def _evaluation_values(
        self,
        definition: RegimeDefinitionV2,
        output_axis: PortValue,
        source_values: Mapping[str, PortValue],
        mode: str,
        as_of: str | None,
        source_cache: dict[str, DataBundle] | None = None,
    ) -> tuple[
        np.ndarray,
        dict[str, Any] | None,
        str,
        dict[str, dict[str, Any]],
    ]:
        primary = next((target for target in definition.evaluation_targets if target.primary), None)
        if primary is None and definition.evaluation_targets:
            primary = definition.evaluation_targets[0]
        if primary is None:
            first_source_id = next(iter(source_values))
            source = source_values[first_source_id]
            values = left_align_float_kernel(output_axis.dates, source.dates, np.ascontiguousarray(source.values, dtype=np.float64))
            return values, None, first_source_id, {}
        cache = source_cache if source_cache is not None else {}
        resolver_as_of = as_of if mode == "retrospective" else None
        outputs: dict[str, dict[str, Any]] = {}
        for target in definition.evaluation_targets:
            target_kind = str(target.source.get("kind") or "")
            cache_key = _canonical_source_cache_key(
                target.source,
                mode,
                resolver_as_of,
            )
            full_bundle = cache.get(cache_key)
            if full_bundle is None:
                if target_kind == "upload":
                    full_bundle = self._upload_bundle(target.source, mode, resolver_as_of)
                elif target_kind == "macro":
                    data_root = self._bound_source_root("source.macro", target.source)
                    full_bundle = _macro_bundle(target.source, mode, resolver_as_of, data_root)
                else:
                    data_root = (
                        self._bound_source_root(f"source.{target_kind}", target.source)
                        if target_kind in {"index", "etf", "fund"}
                        else self.market_data_dir
                    )
                    full_bundle = resolve_target(
                        target.source,
                        mode,
                        resolver_as_of,
                        data_root,
                        self.indicator_service,
                    )
                cache[cache_key] = full_bundle
            if resolver_as_of is not None:
                resolver_snapshot = copy.deepcopy(full_bundle.snapshot)
                resolver_snapshot["as_of"] = resolver_as_of
                bundle = DataBundle(
                    frame=full_bundle.frame,
                    snapshot=resolver_snapshot,
                )
            else:
                bundle = self._slice_cached_bundle(full_bundle, as_of)
            if target_kind == "indicator" and target.source.get("data_fingerprint"):
                actual_fingerprint = str(bundle.snapshot.get("fingerprint") or "")
                if actual_fingerprint != str(target.source["data_fingerprint"]):
                    raise ValidationError(
                        "INDICATOR_DATA_FINGERPRINT_MISMATCH",
                        "评价指标底层数据已变化，与已保存定义不一致。",
                        f"evaluation_targets.{target.id}.source.data_fingerprint",
                    )
            values = left_align_float_kernel(
                output_axis.dates,
                _as_dates(bundle.frame, "observation_date"),
                _as_values(bundle.frame),
            )
            outputs[target.id] = {
                "id": target.id,
                "name": target.name,
                "primary": target.id == primary.id,
                "source": copy.deepcopy(target.source),
                "snapshot": copy.deepcopy(bundle.snapshot),
                "port": PortValue(values, output_axis.dates, output_axis.available),
            }
        primary_output = outputs[primary.id]
        return (
            np.ascontiguousarray(primary_output["port"].values, dtype=np.float64),
            copy.deepcopy(primary_output["snapshot"]),
            primary.id,
            outputs,
        )

    @staticmethod
    def _validate_realtime_graph(definition: RegimeDefinitionV2, mode: str) -> None:
        if mode != "realtime":
            return
        for node in definition.graph.nodes:
            schema = NODE_REGISTRY.get(node.type, {})
            if (
                schema.get("supports_realtime") is not True
                or schema.get("causal") is not True
                or schema.get("repaints") is not False
            ):
                raise ValidationError(
                    "NON_CAUSAL_REALTIME_GRAPH",
                    "实时识别已禁用事后分析算法；请移除该节点，或切换到事后研究。",
                    f"graph.nodes.{node.id}",
                )

    def _execute_graph(
        self,
        job_id: str | None,
        definition: RegimeDefinitionV2,
        mode: str,
        as_of: str | None,
        *,
        plan: Mapping[str, Any] | None = None,
        source_cache: dict[str, DataBundle] | None = None,
        latent_training_as_of: str | None = None,
    ) -> dict[str, Any]:
        inspection = validate_definition_v2(definition)
        required = _required_node_ids(definition)
        if job_id is not None:
            with self._lock:
                job = self._jobs[job_id]
            self._check_cancelled(job)
        self._validate_realtime_graph(definition, mode)
        execution_source_cache = source_cache if source_cache is not None else {}
        bundles, snapshots = self._resolve_sources(
            definition,
            required,
            mode,
            as_of,
            execution_source_cache,
        )
        if job_id is not None:
            self._update_job(job_id, status="running", stage="executing_graph", progress=0.2, message="正在执行 NJIT 图谱。")
        node_map = {node.id: node for node in definition.graph.nodes}
        source_values = {
            source_id: PortValue(
                _as_values(bundle.frame),
                _as_dates(bundle.frame, "observation_date"),
                _as_dates(bundle.frame, "available_at"),
            )
            for source_id, bundle in bundles.items()
        }
        node_outputs: dict[str, dict[str, PortValue]] = {
            source_id: {"value": value} for source_id, value in source_values.items()
        }
        formula_plans = dict((plan or {}).get("formula_plans") or {})
        formula_audits: dict[str, Any] = {}
        model_audits: dict[str, Any] = {}
        execution_order = [node_id for node_id in inspection["topological_order"] if node_id in required]
        executable_count = sum(1 for node_id in execution_order if node_id not in bundles)
        completed = 0
        for node_id in execution_order:
            node = node_map[node_id]
            if node_id in bundles:
                continue
            if job_id is not None:
                with self._lock:
                    current_job = self._jobs[job_id]
                self._check_cancelled(current_job)
            node_outputs[node_id] = self._execute_numeric_node(
                node,
                node_outputs,
                len(definition.states),
                mode,
                as_of,
                latent_training_as_of,
                formula_plans,
                formula_audits,
                model_audits,
            )
            completed += 1
            progress = 0.2 + 0.65 * float(completed) / float(max(executable_count, 1))
            if job_id is not None:
                self._update_job(job_id, progress=progress, message=f"已执行节点 {node_id}。")

        if definition.graph._node_preview:
            target = definition.graph.outputs["preview"]
            selected = node_outputs[target.node_id][target.port]
            return {
                "result": {
                    "result_kind": "node_preview", "preview_target": target.model_dump(),
                    "row_count": int(selected.values.shape[0]), "mode": mode,
                    "data_snapshots": snapshots,
                    "output_catalog": [
                        {"node_id": node_id, "port": port, "value_type": next(item["type"] for item in NODE_REGISTRY[node_map[node_id].type]["outputs"] if item["name"] == port)}
                        for node_id, outputs in node_outputs.items() for port in outputs
                    ],
                    "diagnostics": {"execution_audit": copy.deepcopy(self._runtime_audit),
                                    "formula_audits": formula_audits, "model_audits": model_audits,
                                    "required_node_ids": sorted(required), "request_time_compilation": 0, "python_fallback": 0},
                },
                "series": [], "node_outputs": node_outputs,
                "node_types": {node.id: node.type for node in definition.graph.nodes},
            }

        state_ref = definition.graph.outputs["state"]
        state_port = node_outputs[state_ref.node_id][state_ref.port]
        if state_port.dates.shape[0] < MIN_OBSERVATIONS:
            raise ValidationError("INSUFFICIENT_OBSERVATIONS", "最终状态序列至少需要 5 条观测。", "graph.outputs.state")
        state_axis_length = int(state_port.dates.shape[0])
        state_contract = {
            "one_dimensional": state_port.values.ndim == 1,
            "integer_dtype": bool(np.issubdtype(state_port.values.dtype, np.integer)),
            "value_length": int(state_port.values.shape[0]),
            "date_length": state_axis_length,
            "available_length": int(state_port.available.shape[0]),
        }
        state_contract["passed"] = bool(
            state_contract["one_dimensional"]
            and state_contract["integer_dtype"]
            and state_contract["value_length"] == state_axis_length
            and state_contract["available_length"] == state_axis_length
        )
        if not state_contract["passed"]:
            raise ValidationError(
                "REGIME_FINAL_OUTPUT_CONTRACT_VIOLATION",
                "最终状态输出未通过维度、类型或时间轴长度约束。",
                "graph.outputs.state",
                [state_contract],
            )
        final_states = np.ascontiguousarray(state_port.values, dtype=np.int64)
        probability_ref = definition.graph.outputs.get("probabilities")
        confidence_ref = definition.graph.outputs.get("confidence")
        if probability_ref is not None:
            probability_port = node_outputs[probability_ref.node_id][probability_ref.port]
            self._same_axis(state_port, probability_port)
            raw_probabilities = probability_port.values
            if raw_probabilities.ndim != 2:
                raise ValidationError(
                    "REGIME_PROBABILITY_CONTRACT_VIOLATION",
                    "概率输出必须是二维 time × state 矩阵。",
                    "graph.outputs.probabilities",
                )
            probabilities = np.ascontiguousarray(raw_probabilities, dtype=np.float64)
        else:
            probabilities = state_probabilities_kernel(
                final_states,
                np.int64(len(definition.states)),
            )
        probability_contract_values = probability_contract_kernel(
            probabilities,
            final_states,
            np.int64(len(definition.states)),
        )
        probability_contract = {
            "passed": not any(
                float(probability_contract_values[index]) > 0.0
                for index in range(probability_contract_values.shape[0])
            ),
            "shape_mismatches": int(probability_contract_values[0]),
            "nonfinite_values": int(probability_contract_values[1]),
            "out_of_range_values": int(probability_contract_values[2]),
            "invalid_row_sums": int(probability_contract_values[3]),
            "invalid_state_codes": int(probability_contract_values[4]),
        }
        if not probability_contract["passed"]:
            raise ValidationError(
                "REGIME_PROBABILITY_CONTRACT_VIOLATION",
                "概率输出未通过形状、有限值、区间或逐行和约束。",
                "graph.outputs.probabilities",
                [probability_contract],
            )
        def final_port(reference: Any, output_name: str, *, integer: bool) -> PortValue:
            port = node_outputs[reference.node_id][reference.port]
            try:
                self._same_axis(state_port, port)
            except ValidationError as exc:
                raise ValidationError(
                    "FINAL_OUTPUT_AXIS_MISMATCH",
                    f"最终输出 {output_name} 与 state 不在同一时间轴。",
                    f"graph.outputs.{output_name}",
                ) from exc
            if port.values.ndim != 1:
                raise ValidationError(
                    "FINAL_OUTPUT_SHAPE_MISMATCH",
                    f"最终输出 {output_name} 必须是一维时序。",
                    f"graph.outputs.{output_name}",
                )
            if integer and not np.issubdtype(port.values.dtype, np.integer):
                raise ValidationError(
                    "FINAL_OUTPUT_DTYPE_MISMATCH",
                    f"最终输出 {output_name} 必须使用整数编码。",
                    f"graph.outputs.{output_name}",
                )
            return port

        confidence = (
            np.ascontiguousarray(
                final_port(confidence_ref, "confidence", integer=False).values,
                dtype=np.float64,
            )
            if confidence_ref is not None
            else state_confidence_kernel(final_states)
        )
        default_recognition, default_effective, default_reasons = temporal_output_kernel(final_states)
        recognition_ref = definition.graph.outputs.get("recognition_index")
        effective_ref = definition.graph.outputs.get("effective_index")
        reason_ref = definition.graph.outputs.get("reason_code")
        recognition_indices = (
            np.ascontiguousarray(
                final_port(recognition_ref, "recognition_index", integer=True).values,
                dtype=np.int64,
            )
            if recognition_ref is not None
            else default_recognition
        )
        effective_indices = (
            np.ascontiguousarray(
                final_port(effective_ref, "effective_index", integer=True).values,
                dtype=np.int64,
            )
            if effective_ref is not None
            else (
                effective_from_recognition_kernel(recognition_indices, final_states)
                if recognition_ref is not None
                else default_effective
            )
        )
        reason_codes = (
            np.ascontiguousarray(
                final_port(reason_ref, "reason_code", integer=True).values,
                dtype=np.int64,
            )
            if reason_ref is not None
            else default_reasons
        )
        final_dependencies = set()
        pending = [ref.node_id for ref in definition.graph.outputs.values()]
        while pending:
            current = pending.pop()
            if current in final_dependencies:
                continue
            final_dependencies.add(current)
            pending.extend(ref.node_id for ref in node_map[current].inputs.values())
        dating_nodes = [node_id for node_id in final_dependencies
                        if NODE_REGISTRY[node_map[node_id].type].get("knowledge_scope") == "full_input"]
        retrospective_dating = bool(dating_nodes)
        dating_knowledge_at = None
        if retrospective_dating:
            # Applies even when a custom graph omits or replaces temporal outputs.
            # A downstream slice must not erase knowledge used by the dating node.
            for node_id in sorted(dating_nodes):
                dating_source = self._input(node_outputs, node_map[node_id].inputs["value"])
                recognition_indices, effective_indices, knowledge_at = retrospective_dating_timing_kernel(
                    final_states, np.ascontiguousarray(dating_source.available, dtype=np.int64))
                dating_knowledge_at = knowledge_at if dating_knowledge_at is None else max(dating_knowledge_at, knowledge_at)
        final_contract_values = final_output_contract_kernel(
            confidence,
            recognition_indices,
            effective_indices,
            reason_codes,
            final_states,
        )
        final_contract = {
            "passed": not any(int(value) > 0 for value in final_contract_values),
            "confidence_length_mismatches": int(final_contract_values[0]),
            "invalid_confidence_values": int(final_contract_values[1]),
            "recognition_length_mismatches": int(final_contract_values[2]),
            "invalid_recognition_indices": int(final_contract_values[3]),
            "classified_without_recognition": int(final_contract_values[4]),
            "effective_length_mismatches": int(final_contract_values[5]),
            "invalid_effective_indices": int(final_contract_values[6]),
            "reason_length_mismatches": int(final_contract_values[7]),
            "invalid_reason_codes": int(final_contract_values[8]),
        }
        if not final_contract["passed"]:
            raise ValidationError(
                "REGIME_FINAL_OUTPUT_CONTRACT_VIOLATION",
                "置信度、识别日、生效日或判定原因未通过长度与取值约束。",
                "graph.outputs",
                [final_contract],
            )
        counts = state_count_kernel(final_states, np.int64(len(definition.states)))
        (
            display_values,
            evaluation_snapshot,
            display_source,
            evaluation_outputs,
        ) = self._evaluation_values(
            definition,
            state_port,
            source_values,
            mode,
            as_of,
            execution_source_cache,
        )
        state_definitions = definition.states
        # Only attach evidence from the final state's actual trend lineage.
        # Do not confuse an unrelated diagnostic model with the displayed state.
        evidence: dict[str, PortValue] = {}
        evidence_node = node_map[state_ref.node_id]
        while evidence_node.type in {"post.confirmation", "post.merge_short_regimes"}:
            evidence_node = node_map[evidence_node.inputs["state"].node_id]
        if evidence_node.type == "model.peak_trough":
            evidence = {name: port for name, port in node_outputs[evidence_node.id].items() if name != "state"}
            self._same_axis(state_port, node_outputs[evidence_node.id]["state"])
            evidence["index_value"] = self._input(node_outputs, evidence_node.inputs["value"])
        if evidence_node.type == "model.trend_regime":
            model_ports = node_outputs[evidence_node.id]
            self._same_axis(state_port, model_ports["state"])
            evidence = {name: model_ports[name] for name in ("candidate", "pending_count", "phase")}
            evidence["raw_trend_state"] = model_ports["state"]
            refs = [evidence_node.inputs[name] for name in ("distance", "slope", "efficiency")]
            for name, ref in zip(("distance", "slope", "efficiency"), refs):
                evidence[name] = self._input(node_outputs, ref)
            if (len({ref.node_id for ref in refs}) == 1
                    and [ref.port for ref in refs] == ["distance", "slope", "efficiency"]
                    and node_map[refs[0].node_id].type == "feature.trend_metrics"):
                evidence.update(node_outputs[refs[0].node_id])
        interval_nodes = [node_map[node_id] for node_id in final_dependencies if node_map[node_id].type in STATISTIC_IDS]
        segment_ids = {node.inputs["start"].node_id for node in interval_nodes}
        if len(segment_ids) == 1:
            segment_id = next(iter(segment_ids))
            segment = node_map[segment_id]
            pivot_node = node_map[segment.inputs["pivot"].node_id]
            price_port = self._input(node_outputs, pivot_node.inputs["value"])
            if price_port.dates is state_port.dates:
                evidence.update(phase_start_index=node_outputs[segment_id]["start"],
                                phase_end_index=node_outputs[segment_id]["end"], index_value=price_port,
                                pivot=node_outputs[pivot_node.id]["pivot"], pivot_price=node_outputs[pivot_node.id]["pivot_price"])
                for candidate in (node_map[node_id] for node_id in required if node_map[node_id].type in STATISTIC_IDS):
                    if (candidate.inputs["start"].node_id != segment_id
                            or candidate.inputs["value"] != pivot_node.inputs["value"]
                            or self._input(node_outputs, candidate.inputs["value"]).dates is not state_port.dates):
                        continue
                    key = "phase_return" if candidate.type == "segment.change" else candidate.type.replace(".", "_")
                    evidence[key] = node_outputs[candidate.id]["value"]
        series: list[dict[str, Any]] = []
        channel_values = {}
        for name in definition.graph.channel_metadata:
            if name in {"state", "probabilities", "confidence", "recognition_index", "effective_index", "reason_code"}:
                continue
            port = self._input(node_outputs, definition.graph.outputs[name])
            self._same_axis(state_port, port)
            channel_values[name] = port
        for index in range(state_port.dates.shape[0]):
            observation_date = pd.Timestamp(int(state_port.dates[index]), unit="ns").date().isoformat()
            available_at = pd.Timestamp(int(state_port.available[index]), unit="ns").date().isoformat()
            recognition_index = int(recognition_indices[index])
            effective_index = int(effective_indices[index])
            recognized_at = available_at
            if 0 <= recognition_index < state_port.dates.shape[0]:
                recognized_ns = max(
                    int(state_port.available[index]),
                    int(state_port.available[recognition_index]),
                )
                recognized_at = pd.Timestamp(recognized_ns, unit="ns").date().isoformat()
            effective_date = None
            if retrospective_dating and recognition_index >= 0:
                recognized_at = max(recognized_at, pd.Timestamp(int(dating_knowledge_at), unit="ns").date().isoformat())
            if 0 <= effective_index < state_port.dates.shape[0]:
                effective_ns = max(int(state_port.dates[effective_index]), int(state_port.available[effective_index]))
                effective_date = pd.Timestamp(effective_ns, unit="ns").date().isoformat()
            state_code = int(final_states[index])
            state = state_definitions[state_code] if 0 <= state_code < len(state_definitions) else None
            series.append(
                {
                    "index": index,
                    "observation_date": observation_date,
                    "date": observation_date,
                    "available_at": available_at,
                    "data_available_at": available_at,
                    "recognized_at": recognized_at,
                    "signal_date": recognized_at,
                    "effective_date": effective_date if state is not None else None,
                    "executable": effective_date is not None and state is not None,
                    "value": _safe_number(display_values[index]),
                    "state_code": state_code,
                    "recognition_index": recognition_index,
                    "effective_index": effective_index,
                    "reason_code": int(reason_codes[index]),
                    "state_id": state.id if state is not None else "unclassified",
                    "state_label": state.label if state is not None else "未分类",
                    "confidence": _safe_number(confidence[index]),
                    "probabilities": {
                        state_item.id: _safe_number(probabilities[index, state_index])
                        for state_index, state_item in enumerate(state_definitions)
                    },
                    "probability_source": "model" if probability_ref is not None else "deterministic_state",
                    "filtered_value": None,
                    "score": None,
                    "features": {**{name: _safe_number(port.values[index]) for name, port in evidence.items()},
                                 **{f"channel:{name}": _safe_number(port.values[index]) for name, port in channel_values.items()}},
                    "reasons": (["事后峰谷定界：相邻小幅反向波段满足整段振幅、方向效率及最短长度约束，合并为震荡；不是当时的交易信号。"]
                                if state is not None and state.role == "neutral" and "sideways_range" in evidence and np.isfinite(evidence["sideways_range"].values[index])
                                else ["事后峰谷分段：依据独立区间统计与分类规则判断；不提供当时交易信号。"] if state is not None and interval_nodes else ["事后峰谷定界；依赖全样本筛选，不是当时的交易信号。"] if state is not None else
                                ["未分类：未形成完整保留峰谷区间、处于首尾边界或存在无效数据。"])
                               if retrospective_dating else (["状态已识别"] if int(reason_codes[index]) == 0 else ["输入不足或状态被拒识"]),
                    "revision": 1,
                    "vintage": None,
                    "is_final": not retrospective_dating,
                }
            )
        output_catalog: list[dict[str, Any]] = []
        for node_id, outputs in node_outputs.items():
            node_type = node_map[node_id].type
            port_types = {item["name"]: item["type"] for item in NODE_REGISTRY[node_type]["outputs"]}
            for port, port_value in outputs.items():
                values = port_value.values
                output_catalog.append(
                    {
                        "node_id": node_id,
                        "node_type": node_type,
                        "port": port,
                        "value_type": port_types[port],
                        "shape": list(values.shape),
                        "dtype": str(values.dtype),
                    }
                )
        if job_id is not None:
            self._update_job(job_id, stage="finalizing", progress=0.9, message="正在生成诊断摘要。")
        result = {
            "schema_version": "2.0",
            "graph_hash": inspection["graph_hash"],
            "definition_hash": inspection["definition_hash"],
            "mode": mode,
            "row_count": int(state_port.dates.shape[0]),
            "display_source": display_source,
            "state_counts": {
                state.id: int(counts[index]) for index, state in enumerate(state_definitions)
            },
            "classified_count": int(counts[len(state_definitions)]),
            "state_switches": int(counts[len(state_definitions) + 1]),
            "data_snapshots": snapshots,
            "evaluation_snapshot": evaluation_snapshot,
            "evaluation_results": {
                target_id: {
                    "id": item["id"],
                    "name": item["name"],
                    "primary": item["primary"],
                    "snapshot": copy.deepcopy(item["snapshot"]),
                }
                for target_id, item in evaluation_outputs.items()
            },
            "output_catalog": output_catalog,
            "series_outputs": regime_series_outputs(definition),
            "diagnostics": {
                "topological_order": inspection["topological_order"],
                "required_node_ids": sorted(required),
                "alignment": "explicit_nodes_only",
                "missing_value_policy": "preserve_nan_and_unclassified",
                "execution_audit": copy.deepcopy(self._runtime_audit),
                "formula_audits": formula_audits,
                "model_audits": model_audits,
                "probability_contract": probability_contract,
                "final_output_contract": final_contract,
                "request_time_compilation": 0,
                "python_fallback": 0,
            },
        }
        return {
            "result": result,
            "series": series,
            "node_outputs": node_outputs,
            "node_types": {node.id: node.type for node in definition.graph.nodes},
            "evaluation_outputs": evaluation_outputs,
        }

    def _persist_node_outputs(
        self,
        node_outputs: Mapping[str, Mapping[str, PortValue]],
        node_types: Mapping[str, str],
    ) -> dict[str, Any]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        catalog: list[dict[str, Any]] = []
        for node_position, (node_id, outputs) in enumerate(sorted(node_outputs.items())):
            for port_position, (port, value) in enumerate(sorted(outputs.items())):
                prefix = f"n{node_position}_p{port_position}"
                arrays[f"{prefix}_values"] = np.ascontiguousarray(value.values)
                arrays[f"{prefix}_dates"] = np.ascontiguousarray(value.dates, dtype=np.int64)
                arrays[f"{prefix}_available"] = np.ascontiguousarray(value.available, dtype=np.int64)
                catalog.append(
                    {
                        "node_id": node_id,
                        "node_type": node_types[node_id],
                        "port": port,
                        "array_prefix": prefix,
                        "shape": list(value.values.shape),
                        "dtype": str(value.values.dtype),
                    }
                )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".regime-output-",
            suffix=".npz.tmp",
            dir=self.artifact_dir,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            with temporary_path.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            digest = _sha256_file(temporary_path)
            final_path = self.artifact_dir / f"{digest}.npz"
            if not final_path.exists():
                try:
                    os.link(temporary_path, final_path)
                except FileExistsError:
                    pass
            size_bytes = final_path.stat().st_size
        finally:
            temporary_path.unlink(missing_ok=True)
        return {
            "artifact_id": f"regime-output-sha256-{digest}",
            "checksum": f"sha256:{digest}",
            "format": "npz",
            "schema_version": "regime-node-output-v1",
            "content_addressed": True,
            "size_bytes": int(size_bytes),
            "arrays": catalog,
            "uri": f"historical-regime-artifact://regime-output-sha256-{digest}",
        }

    def _persist_series(self, series: list[dict[str, Any]]) -> dict[str, Any]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for item in series:
            row = copy.deepcopy(item)
            row["probabilities_json"] = json.dumps(
                row.pop("probabilities", {}),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            row["features_json"] = json.dumps(
                row.pop("features", {}),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            row["reasons_json"] = json.dumps(
                row.pop("reasons", []),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            rows.append(row)
        table = pa.Table.from_pylist(rows)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".regime-series-",
            suffix=".parquet.tmp",
            dir=self.artifact_dir,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            pq.write_table(table, temporary_path, compression="zstd")
            metadata = dict(table.schema.metadata or {})
            metadata.update(
                {
                    b"historical_regime_schema": b"regime-series-v2",
                }
            )
            pq.write_table(
                table.replace_schema_metadata(metadata),
                temporary_path,
                compression="zstd",
            )
            digest = _sha256_file(temporary_path)
            artifact_id = f"regime-series-sha256-{digest}"
            checksum = f"sha256:{digest}"
            final_path = self.artifact_dir / f"{digest}.parquet"
            if not final_path.exists():
                try:
                    os.link(temporary_path, final_path)
                except FileExistsError:
                    pass
            size_bytes = final_path.stat().st_size
        finally:
            temporary_path.unlink(missing_ok=True)
        return {
            "artifact_id": artifact_id,
            "checksum": checksum,
            "format": "parquet",
            "schema_version": "regime-series-v2",
            "content_addressed": True,
            "row_count": len(series),
            "size_bytes": int(size_bytes),
            "uri": f"historical-regime-artifact://{artifact_id}",
        }

    def _load_series(self, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
        hydrated = hydrate_v2_run_snapshot(
            {"schema_version": "2.0", "series_artifact": dict(manifest)},
            artifact_dir=self.artifact_dir,
        )
        return list(hydrated["series"])

    def _hydrate_run(self, run: Mapping[str, Any]) -> dict[str, Any]:
        hydrated = hydrate_v2_run_snapshot(run, artifact_dir=self.artifact_dir)
        definition = hydrated.get("definition")
        if str(hydrated.get("schema_version")) == "2.0" and isinstance(definition, Mapping) and definition.get("states"):
            diagnostics = hydrated.get("algorithm_diagnostics") or {}
            hydrated["overview"] = build_result_overview(
                run_kind="saved",
                run_id=str(hydrated["id"]),
                definition=definition,
                series=hydrated["series"],
                definition_id=hydrated.get("definition_id"),
                revision=hydrated.get("definition_revision") or hydrated.get("revision"),
                definition_hash=hydrated.get("definition_snapshot_hash") or diagnostics.get("definition_hash") or "",
                graph_hash=diagnostics.get("graph_hash") or hydrated.get("algorithm", {}).get("parameters", {}).get("graph_hash") or "",
                mode=hydrated.get("mode") or "realtime",
                as_of=hydrated.get("as_of"),
                data_snapshots=hydrated.get("data_snapshots"),
                series_endpoint=f"/api/historical-regimes/runs/{hydrated['id']}",
                series_artifact=hydrated.get("series_artifact"),
                result={**diagnostics, "evaluation_results": hydrated.get("evaluation_results") or {}},
                created_at=hydrated.get("created_at"),
            )
        return hydrated

    def hydrate_run_snapshot(self, run: Mapping[str, Any]) -> dict[str, Any]:
        """Hydrate a previously integrity-checked raw snapshot for downstream consumers."""

        return self._hydrate_run(run)

    def _formal_source_gate(self, definition: RegimeDefinitionV2) -> None:
        for node in definition.graph.nodes:
            if node.type == "source.inline":
                raise ValidationError(
                    "INLINE_SOURCE_FORMAL_RUN_BLOCKED",
                    "正式运行不能引用内联数据；请先在研究数据实验室登记不可变上传版本。",
                    f"graph.nodes.{node.id}.type",
                )
            if node.type in {"source.index", "source.macro", "source.etf", "source.fund"}:
                self._bound_source_root(node.type, node.parameters)
            elif node.type == "source.upload":
                self._upload_bundle(node.parameters, "retrospective", None)
            elif node.type == "source.indicator":
                if (
                    not node.parameters.get("data_fingerprint")
                    or not isinstance(node.parameters.get("indicator_data_snapshot"), Mapping)
                ):
                    raise ValidationError(
                        "INDICATOR_DATA_FINGERPRINT_REQUIRED",
                        "正式运行要求锁定指标版本及其底层数据指纹与数据快照。",
                        f"graph.nodes.{node.id}.parameters.data_fingerprint",
                    )
            elif node.type == "source.relative":
                raise ValidationError(
                    "RELATIVE_SOURCE_FORMAL_RUN_BLOCKED",
                    "相对强弱正式运行必须改用两个已锁定数据源、显式对齐与 NJIT 数学节点。",
                    f"graph.nodes.{node.id}.type",
                )
        for target in definition.evaluation_targets:
            source = target.source
            kind = str(source.get("kind") or "")
            if kind == "inline":
                raise ValidationError(
                    "INLINE_SOURCE_FORMAL_RUN_BLOCKED",
                    "正式运行的评价标的不能引用内联数据。",
                    f"evaluation_targets.{target.id}.source",
                )
            if kind in {"index", "macro", "etf", "fund"}:
                self._bound_source_root(f"source.{kind}", source)
            elif kind == "upload":
                self._upload_bundle(source, "retrospective", None)
            elif kind == "indicator" and (
                not source.get("data_fingerprint")
                or not isinstance(source.get("indicator_data_snapshot"), Mapping)
            ):
                raise ValidationError(
                    "INDICATOR_DATA_FINGERPRINT_REQUIRED",
                    "正式运行要求锁定评价指标及其底层数据快照。",
                    f"evaluation_targets.{target.id}.source.data_fingerprint",
                )
            elif kind == "relative":
                raise ValidationError(
                    "RELATIVE_SOURCE_FORMAL_RUN_BLOCKED",
                    "正式运行的评价标的暂不接受隐式相对强弱数据源。",
                    f"evaluation_targets.{target.id}.source",
                )

    @staticmethod
    def _causality_payload(
        definition: RegimeDefinitionV2,
        mode: str,
        series: list[dict[str, Any]],
    ) -> dict[str, Any]:
        required = _required_node_ids(definition)
        noncausal = [
            node.id
            for node in definition.graph.nodes
            if node.id in required and NODE_REGISTRY[node.type].get("causal") is not True
        ]
        if mode == "retrospective":
            noncausal.extend(
                node.id
                for node in definition.graph.nodes
                if node.id in required
                and node.type in {"model.hmm", "model.markov", "model.gmm"}
                and node.id not in noncausal
            )
        temporal_violations: list[int] = []
        for index, item in enumerate(series):
            dates = [
                pd.Timestamp(item["observation_date"]),
                pd.Timestamp(item["data_available_at"]),
                pd.Timestamp(item["recognized_at"]),
            ]
            if item.get("effective_date"):
                dates.append(pd.Timestamp(item["effective_date"]))
            if dates != sorted(dates):
                temporal_violations.append(index)
        causal = not noncausal and not temporal_violations
        eligible = ["research_display", "product_research"]
        if mode == "realtime" and causal:
            eligible.extend(["formal_backtest", "taa"])
        blockers: list[str] = []
        if noncausal:
            blockers.append(f"图谱包含非因果节点: {', '.join(noncausal)}")
        if temporal_violations:
            blockers.append("存在观测日、可得日、识别日、生效日倒序。")
        return {
            "classification": "causal" if causal else "non_causal",
            "is_causal": causal,
            "uses_future_data": bool(noncausal),
            "repaints": bool(noncausal),
            "realtime_eligible": mode == "realtime" and causal,
            "publish_eligible_usages": eligible,
            "blockers": blockers,
            "warnings": [],
            "checks": [
                {"id": "temporal_order", "passed": not temporal_violations, "violations": temporal_violations[:20]},
                {"id": "future_data", "passed": not noncausal},
                {"id": "historical_repaint", "passed": not noncausal},
                {"id": "last_point_not_executable", "passed": not bool(series[-1].get("executable")) if series else True},
            ],
        }

    @staticmethod
    def _series_codes(
        definition: RegimeDefinitionV2,
        series: list[dict[str, Any]],
    ) -> np.ndarray:
        state_index = {state.id: index for index, state in enumerate(definition.states)}
        return np.ascontiguousarray(
            [state_index.get(str(item.get("state_id")), -1) for item in series],
            dtype=np.int64,
        )

    @staticmethod
    def _comparison_metrics(
        base_codes: np.ndarray,
        candidate_codes: np.ndarray,
    ) -> dict[str, Any]:
        stability = prefix_stability_kernel(base_codes, candidate_codes)
        pair = comparison_pair_kernel(base_codes, candidate_codes)
        return {
            "comparable_observations": int(stability[0]),
            "revisions": int(stability[1]),
            "agreement": _safe_number(stability[2]),
            "revision_rate": _safe_number(stability[3]),
            "mean_boundary_distance_observations": _safe_number(pair[3]),
        }

    def _perturbed_definitions(
        self,
        definition: RegimeDefinitionV2,
        perturbation: float,
        maximum_candidates: int,
    ) -> list[tuple[RegimeDefinitionV2, dict[str, Any]]]:
        if perturbation <= 0.0:
            return []
        payload = definition.model_dump(mode="json")
        candidates: list[tuple[RegimeDefinitionV2, dict[str, Any]]] = []
        preferred_parameters = (
            "upper",
            "lower",
            "upper_enter",
            "lower_enter",
            "threshold",
            "min_move",
            "window",
            "periods",
            "confirmation",
            "min_duration",
            "process_variance",
            "measurement_variance",
        )
        for node_index, node in enumerate(definition.graph.nodes):
            schema = NODE_REGISTRY[node.type].get("parameter_schema", {})
            properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
            for parameter_name in preferred_parameters:
                if parameter_name not in node.parameters:
                    continue
                raw_value = node.parameters[parameter_name]
                if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                    continue
                field_schema = properties.get(parameter_name, {})
                field_type = str(field_schema.get("type") or "number")
                minimum = float(field_schema.get("minimum", -1.0e300))
                changed = float(
                    perturb_scalar_kernel(
                        np.float64(raw_value),
                        np.float64(perturbation),
                        np.float64(minimum),
                        np.uint8(1 if field_type == "integer" else 0),
                    )
                )
                maximum = field_schema.get("maximum")
                if maximum is not None and changed > float(maximum):
                    continue
                if changed == float(raw_value):
                    continue
                candidate_payload = copy.deepcopy(payload)
                candidate_parameters = candidate_payload["graph"]["nodes"][node_index].setdefault(
                    "parameters", {}
                )
                candidate_parameters[parameter_name] = (
                    int(changed) if field_type == "integer" else changed
                )
                candidate = parse_definition_v2(candidate_payload)
                validate_definition_v2(candidate)
                candidates.append(
                    (
                        candidate,
                        {
                            "node_id": node.id,
                            "parameter": parameter_name,
                            "base_value": raw_value,
                            "candidate_value": candidate_parameters[parameter_name],
                            "perturbation": perturbation,
                        },
                    )
                )
                if len(candidates) >= maximum_candidates:
                    return candidates
        return candidates

    def _validation_reports(
        self,
        definition: RegimeDefinitionV2,
        mode: str,
        as_of: str | None,
        plan: Mapping[str, Any],
        series: list[dict[str, Any]],
        source_cache: dict[str, DataBundle] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        validation = definition.validation
        folds = int(validation.get("folds", 4))
        perturbation = float(validation.get("stability_perturbation", 0.1))
        maximum_candidates = int(validation.get("sensitivity_candidates", 4))
        if maximum_candidates < 1 or maximum_candidates > 8:
            raise ValidationError(
                "INVALID_SENSITIVITY_CANDIDATES",
                "validation.sensitivity_candidates 必须在 1 到 8 之间。",
                "validation.sensitivity_candidates",
            )
        base_codes = self._series_codes(definition, series)
        label_summary = label_summary_kernel(base_codes)
        recognition = np.ascontiguousarray(
            [int(item.get("recognition_index", -1)) for item in series],
            dtype=np.int64,
        )
        recognition_summary = recognition_delay_summary_kernel(base_codes, recognition)
        prefix_length, windows = validation_windows_kernel(
            np.int64(len(series)),
            np.int64(folds),
        )
        prefix_length = int(prefix_length)
        prefix_report: dict[str, Any]
        try:
            prefix_as_of = (
                str(series[prefix_length - 1]["data_available_at"])
                if prefix_length > 0 and prefix_length <= len(series)
                else as_of
            )
            prefix_execution = self._execute_graph(
                None,
                definition,
                mode,
                prefix_as_of,
                plan=plan,
                source_cache=source_cache,
            )
            prefix_codes = self._series_codes(definition, prefix_execution["series"])
            expected_dates = [
                item["observation_date"]
                for item in series[: prefix_codes.shape[0]]
            ]
            actual_dates = [
                item["observation_date"]
                for item in prefix_execution["series"]
            ]
            axis_matches = expected_dates == actual_dates
            prefix_report = {
                "status": "passed" if axis_matches else "axis_changed",
                "prefix_observations": int(prefix_codes.shape[0]),
                "date_axis_matches": axis_matches,
                **self._comparison_metrics(base_codes[: prefix_codes.shape[0]], prefix_codes),
            }
            if int(prefix_report["revisions"]) > 0 and axis_matches:
                prefix_report["status"] = "revisions_detected"
        except IndicatorDomainError as exc:
            prefix_report = {
                "status": "insufficient",
                "prefix_observations": prefix_length,
                "error": exc.detail(),
            }

        sensitivity_candidates: list[dict[str, Any]] = []
        for candidate, descriptor in self._perturbed_definitions(
            definition,
            perturbation,
            maximum_candidates,
        ):
            try:
                candidate_execution = self._execute_graph(
                    None,
                    candidate,
                    mode,
                    as_of,
                    plan=plan,
                    source_cache=source_cache,
                )
                candidate_codes = self._series_codes(
                    candidate,
                    candidate_execution["series"],
                )
                sensitivity_candidates.append(
                    {
                        **descriptor,
                        "status": "completed",
                        **self._comparison_metrics(base_codes, candidate_codes),
                    }
                )
            except IndicatorDomainError as exc:
                sensitivity_candidates.append(
                    {**descriptor, "status": "failed", "error": exc.detail()}
                )

        if mode == "realtime":
            mode_difference = {
                "status": "disabled",
                "base_mode": mode,
                "alternate_mode": "retrospective",
                "reason": "实时识别已禁用事后分析；模式对比仅在事后研究中执行。",
            }
        else:
            alternate_mode = "realtime"
            try:
                alternate_execution = self._execute_graph(
                    None,
                    definition,
                    alternate_mode,
                    as_of,
                    plan=plan,
                    source_cache=source_cache,
                )
                alternate_codes = self._series_codes(
                    definition,
                    alternate_execution["series"],
                )
                mode_difference = {
                    "status": "completed",
                    "base_mode": mode,
                    "alternate_mode": alternate_mode,
                    **self._comparison_metrics(base_codes, alternate_codes),
                }
            except IndicatorDomainError as exc:
                mode_difference = {
                    "status": "blocked",
                    "base_mode": mode,
                    "alternate_mode": alternate_mode,
                    "error": exc.detail(),
                }

        fold_reports: list[dict[str, Any]] = []
        classified_by_fold: list[int] = []
        if bool(validation.get("walk_forward", True)):
            for fold_index in range(windows.shape[0]):
                train_end = int(windows[fold_index, 0])
                test_end = int(windows[fold_index, 1])
                if test_end <= train_end or test_end > len(series):
                    continue
                fold_as_of = str(series[test_end - 1]["data_available_at"])
                training_as_of = str(series[train_end - 1]["data_available_at"])
                try:
                    fold_execution = self._execute_graph(
                        None,
                        definition,
                        "realtime",
                        fold_as_of,
                        plan=plan,
                        source_cache=source_cache,
                        latent_training_as_of=training_as_of,
                    )
                    fold_codes = self._series_codes(definition, fold_execution["series"])
                    available_end = min(test_end, int(fold_codes.shape[0]))
                    test_codes = np.ascontiguousarray(
                        fold_codes[train_end:available_end],
                        dtype=np.int64,
                    )
                    state_counts = state_counts_kernel(
                        test_codes,
                        np.int64(len(definition.states)),
                    )
                    summary = label_summary_kernel(test_codes)
                    classified = int(summary[0])
                    classified_by_fold.append(classified)
                    fold_reports.append(
                        {
                            "fold": fold_index + 1,
                            "status": "ok" if classified > 0 else "insufficient",
                            "train_end_index": int(windows[fold_index, 2]),
                            "model_training_available_through": training_as_of,
                            "model_refit": "expanding_window_at_fold_boundary",
                            "model_audits": copy.deepcopy(
                                fold_execution["result"]
                                .get("diagnostics", {})
                                .get("model_audits", {})
                            ),
                            "test_start_index": train_end,
                            "test_end_index": available_end - 1,
                            "test_observations": int(test_codes.shape[0]),
                            "classified_observations": classified,
                            "state_distribution": {
                                state.id: int(state_counts[index])
                                for index, state in enumerate(definition.states)
                            },
                            "full_sample_comparison": self._comparison_metrics(
                                base_codes[train_end:available_end],
                                test_codes,
                            ),
                        }
                    )
                except IndicatorDomainError as exc:
                    fold_reports.append(
                        {
                            "fold": fold_index + 1,
                            "status": "insufficient",
                            "train_end_index": int(windows[fold_index, 2]),
                            "test_end_index": int(windows[fold_index, 4]),
                            "error": exc.detail(),
                        }
                    )
            classified_total = int(
                integer_sum_kernel(
                    np.ascontiguousarray(classified_by_fold, dtype=np.int64)
                )
            )
            walk_forward = {
                "status": "completed" if fold_reports and classified_total > 0 else "insufficient",
                "method": "expanding_prefix_point_in_time",
                "fold_count": len(fold_reports),
                "classified_observations": classified_total,
                "folds": fold_reports,
            }
        else:
            walk_forward = {
                "status": "disabled_by_policy",
                "method": "expanding_prefix_point_in_time",
                "fold_count": 0,
                "folds": [],
            }

        stability = {
            "status": "validated",
            "prefix_invariance": prefix_report,
            "parameter_sensitivity": {
                "status": "completed" if sensitivity_candidates else "no_eligible_parameter",
                "perturbation": perturbation,
                "candidates": sensitivity_candidates,
            },
            "realtime_vs_retrospective": mode_difference,
            "state_switches": int(label_summary[1]),
            "classified_ratio": _safe_number(label_summary[2]),
            "recognition_delay": {
                "classified_observations": int(recognition_summary[0]),
                "mean_observations": _safe_number(recognition_summary[1]),
                "max_observations": _safe_number(recognition_summary[2]),
                "delayed_rate": _safe_number(recognition_summary[3]),
            },
            "realtime_monitoring": {
                "prefix_revisions": prefix_report.get("revisions"),
                "prefix_revision_rate": prefix_report.get("revision_rate"),
                "label_flips": int(label_summary[1]),
                "label_flip_rate": _safe_number(label_summary[3]),
                "classified_observations": int(label_summary[0]),
            },
        }
        audit = execution_audit(
            "regime_graph_v2_validation",
            [
                "comparison_pair",
                "integer_sum",
                "label_summary",
                "perturb_scalar",
                "prefix_stability",
                "state_counts",
                "validation_windows",
            ],
        )
        return stability, walk_forward, audit

    @staticmethod
    def _audit_gate(calculation_audits: list[Mapping[str, Any]]) -> dict[str, Any]:
        failures: list[dict[str, Any]] = []
        for index, audit in enumerate(calculation_audits):
            try:
                validate_execution_audit(audit)
            except ComputePolicyError as exc:
                failures.append(
                    {
                        "index": index,
                        "engine": audit.get("engine") or audit.get("name"),
                        "message": str(exc),
                    }
                )
        return {
            "passed": not failures,
            "audit_count": len(calculation_audits),
            "failures": failures,
            "required": {
                "nopython": True,
                "object_mode": 0,
                "python_fallback": 0,
                "request_time_compilation": 0,
            },
        }

    @staticmethod
    def _publication_governance(
        definition: RegimeDefinitionV2,
        mode: str,
        row_count: int,
        classified_count: int,
        data_snapshots: Mapping[str, Any],
        causality: Mapping[str, Any],
        stability: Mapping[str, Any],
        walk_forward: Mapping[str, Any],
        calculation_audits: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        validation = definition.validation
        classified_ratio = (
            float(classified_count) / float(row_count) if row_count > 0 else 0.0
        )
        min_coverage = float(validation.get("min_classified_ratio", 0.5))
        min_oos_coverage = float(
            validation.get("min_walk_forward_classified_ratio", 0.5)
        )
        min_sensitivity_agreement = float(
            validation.get("min_parameter_agreement", 0.7)
        )
        max_prefix_revision_rate = float(
            validation.get("max_prefix_revision_rate", 0.0)
        )
        max_label_flip_rate = float(validation.get("max_label_flip_rate", 0.5))

        snapshot_items = [
            item for item in data_snapshots.values() if isinstance(item, Mapping)
        ]
        non_pit_sources = [
            source_id
            for source_id, snapshot in data_snapshots.items()
            if not isinstance(snapshot, Mapping)
            or snapshot.get("revision_policy") == "latest_vintage"
            or snapshot.get("availability_status") == "release_date_unknown"
        ]
        pit_passed = mode == "realtime" and not non_pit_sources and bool(snapshot_items)

        folds = [
            fold
            for fold in walk_forward.get("folds", [])
            if isinstance(fold, Mapping)
        ]
        test_observations = sum(int(fold.get("test_observations") or 0) for fold in folds)
        oos_classified = sum(
            int(fold.get("classified_observations") or 0) for fold in folds
        )
        oos_ratio = (
            float(oos_classified) / float(test_observations)
            if test_observations > 0
            else 0.0
        )
        walk_forward_passed = (
            walk_forward.get("status") == "completed"
            and bool(folds)
            and all(fold.get("status") == "ok" for fold in folds)
        )

        prefix = stability.get("prefix_invariance")
        prefix = prefix if isinstance(prefix, Mapping) else {}
        prefix_revision_rate = prefix.get("revision_rate")
        prefix_revision_rate = (
            float(prefix_revision_rate) if prefix_revision_rate is not None else np.inf
        )
        sensitivity = stability.get("parameter_sensitivity")
        sensitivity = sensitivity if isinstance(sensitivity, Mapping) else {}
        sensitivity_candidates = [
            candidate
            for candidate in sensitivity.get("candidates", [])
            if isinstance(candidate, Mapping) and candidate.get("status") == "completed"
        ]
        agreements = [
            float(candidate["agreement"])
            for candidate in sensitivity_candidates
            if candidate.get("agreement") is not None
        ]
        sensitivity_passed = not agreements or min(agreements) >= min_sensitivity_agreement
        monitoring = stability.get("realtime_monitoring")
        monitoring = monitoring if isinstance(monitoring, Mapping) else {}
        flip_rate = monitoring.get("label_flip_rate")
        flip_rate = float(flip_rate) if flip_rate is not None else np.inf
        stability_passed = (
            prefix.get("status") == "passed"
            and prefix_revision_rate <= max_prefix_revision_rate
            and sensitivity_passed
            and flip_rate <= max_label_flip_rate
        )
        audit_gate = RegimeGraphV2Service._audit_gate(calculation_audits)
        checks = {
            "pit": {
                "passed": pit_passed,
                "mode": mode,
                "non_pit_source_ids": non_pit_sources,
            },
            "causality": {
                "passed": causality.get("is_causal") is True,
                "classification": causality.get("classification"),
            },
            "no_repaint": {
                "passed": causality.get("repaints") is False
                and causality.get("uses_future_data") is False,
            },
            "sample_out": {
                "passed": walk_forward_passed and oos_ratio >= min_oos_coverage,
                "walk_forward_status": walk_forward.get("status"),
                "fold_count": len(folds),
                "classified_ratio": oos_ratio,
                "minimum": min_oos_coverage,
            },
            "stability": {
                "passed": stability_passed,
                "prefix_revision_rate": (
                    prefix_revision_rate if np.isfinite(prefix_revision_rate) else None
                ),
                "max_prefix_revision_rate": max_prefix_revision_rate,
                "minimum_parameter_agreement": min_sensitivity_agreement,
                "label_flip_rate": flip_rate if np.isfinite(flip_rate) else None,
                "max_label_flip_rate": max_label_flip_rate,
            },
            "data_coverage": {
                "passed": row_count >= MIN_OBSERVATIONS
                and classified_ratio >= min_coverage,
                "row_count": row_count,
                "classified_count": classified_count,
                "classified_ratio": classified_ratio,
                "minimum": min_coverage,
            },
            "njit_call_graph": audit_gate,
        }
        formal_required = (
            "pit",
            "causality",
            "no_repaint",
            "sample_out",
            "stability",
            "data_coverage",
            "njit_call_graph",
        )
        formal_failures = [
            check_id for check_id in formal_required if checks[check_id]["passed"] is not True
        ]
        eligible = ["research_display", "product_research"]
        if not formal_failures:
            eligible.extend(["formal_backtest", "taa"])
        restrictions: list[str] = []
        if formal_failures:
            restrictions.append(
                "仅允许研究展示或产品研究；正式回测与 TAA 门禁未全部通过。"
            )
        if mode != "realtime":
            restrictions.append("事后模式结果不可作为实时可交易信号。")
        return {
            "checks": checks,
            "formal_required_checks": list(formal_required),
            "formal_failures": formal_failures,
            "formal_gate_passed": not formal_failures,
            "publish_eligible_usages": eligible,
            "research_restrictions": restrictions,
            "usage_intent": definition.usage_intent,
        }

    def run_saved(
        self,
        reference: Mapping[str, Any],
        mode: str,
        as_of: str | None = None,
        compile_token: str | None = None,
    ) -> dict[str, Any]:
        if mode not in {"realtime", "retrospective"}:
            raise ValidationError("INVALID_RUN_MODE", "mode 必须是 realtime 或 retrospective。", "mode")
        if not isinstance(reference, Mapping) or set(reference) - {"schema_version", "id", "revision"}:
            raise ValidationError(
                "V2_RUN_REQUIRES_SAVED_REFERENCE",
                "正式运行只接受已保存定义的 id 与 revision，不接受草稿正文。",
                "definition",
            )
        if str(reference.get("schema_version") or "") != "2.0" or not reference.get("id") or reference.get("revision") is None:
            raise ValidationError(
                "V2_RUN_REQUIRES_SAVED_REFERENCE",
                "正式运行必须明确指定 schema_version=2.0、id 和 revision。",
                "definition",
            )
        revision = int(reference["revision"])
        stored = self.definitions.get(str(reference["id"]), revision)
        definition = parse_definition_v2(stored)
        validate_definition_v2(definition)
        self._validate_realtime_graph(definition, mode)
        self._formal_source_gate(definition)
        try:
            plan = self._validate_plan(definition, compile_token)
        except ValidationError as exc:
            if exc.code in {"REGIME_GRAPH_PREPARE_REQUIRED", "REGIME_GRAPH_PLAN_NOT_WARM"}:
                raise ConflictError(
                    "REGIME_PLAN_NOT_WARMED",
                    "正式运行所需的执行计划未在当前进程预热。",
                    field="compile_token",
                ) from exc
            raise
        source_cache: dict[str, DataBundle] = {}
        execution = self._execute_graph(
            None,
            definition,
            mode,
            as_of,
            plan=plan,
            source_cache=source_cache,
        )
        artifact = self._persist_node_outputs(
            execution["node_outputs"],
            execution["node_types"],
        )
        series = execution["series"]
        series_artifact = self._persist_series(series)
        states = [item.model_dump(mode="json") for item in definition.states]
        source_nodes = [
            node
            for node in definition.graph.nodes
            if node.type.startswith("source.") and node.type != "source.constant"
        ]
        frequency = str(source_nodes[0].parameters.get("frequency") or "daily")
        conditional = conditional_statistics(series, states, frequency)
        evaluation_outputs = execution.get("evaluation_outputs") or {}
        evaluation_artifact = None
        evaluation_results: dict[str, Any] = {}
        if evaluation_outputs:
            evaluation_artifact = self._persist_node_outputs(
                {
                    target_id: {"value": item["port"]}
                    for target_id, item in evaluation_outputs.items()
                },
                {target_id: "evaluation_target" for target_id in evaluation_outputs},
            )
            artifact_entries = {
                str(item["node_id"]): item
                for item in evaluation_artifact.get("arrays", [])
            }
            for target_id, item in evaluation_outputs.items():
                values = np.ascontiguousarray(item["port"].values, dtype=np.float64)
                target_series = [
                    {**point, "value": _safe_number(values[index])}
                    for index, point in enumerate(series)
                ]
                target_frequency = str(item["source"].get("frequency") or frequency)
                target_conditional = conditional_statistics(
                    target_series,
                    states,
                    target_frequency,
                )
                if item["primary"]:
                    conditional = target_conditional
                evaluation_results[target_id] = {
                    "id": item["id"],
                    "name": item["name"],
                    "primary": item["primary"],
                    "source": copy.deepcopy(item["source"]),
                    "snapshot": copy.deepcopy(item["snapshot"]),
                    "conditional_metrics": target_conditional,
                    "artifact": copy.deepcopy(artifact_entries.get(target_id)),
                }
        causality = self._causality_payload(definition, mode, series)
        analytics_audit = analytics_execution_audit()
        graph_execution_audit = copy.deepcopy(self._runtime_audit)
        graph_execution_audit["executed_kernel_ids"] = list(plan.get("kernel_ids") or [])
        graph_execution_audit["execution_plan_coverage"] = "complete"
        shared_kernel_ids = list(plan.get("shared_kernel_ids") or [])
        shared_model_audit = (
            execution_audit("regime_graph_v2_shared_models", shared_kernel_ids)
            if shared_kernel_ids
            else None
        )
        stability, walk_forward, validation_audit = self._validation_reports(
            definition,
            mode,
            as_of,
            plan,
            series,
            source_cache,
        )
        formula_audits = list(
            (execution["result"].get("diagnostics", {}).get("formula_audits") or {}).values()
        )
        model_audits = list(
            (execution["result"].get("diagnostics", {}).get("model_audits") or {}).values()
        )
        snapshots = execution["result"].get("data_snapshots") or {}
        first_snapshot = next(iter(snapshots.values()), {})
        classified = int(execution["result"].get("classified_count") or 0)
        row_count = int(execution["result"].get("row_count") or 0)
        calculation_audits = [
            graph_execution_audit,
            analytics_audit,
            validation_audit,
            *([shared_model_audit] if shared_model_audit is not None else []),
            *formula_audits,
        ]
        governance = self._publication_governance(
            definition,
            mode,
            row_count,
            classified,
            snapshots,
            causality,
            stability,
            walk_forward,
            calculation_audits,
        )
        run_payload = {
            "schema_version": "2.0",
            "definition_id": definition.id,
            "definition_revision": revision,
            "revision": revision,
            "definition_source": "repository_reference",
            "definition_snapshot_hash": definition_content_hash(definition),
            "name": definition.name,
            "mode": mode,
            "as_of": as_of,
            "definition": definition.model_dump(mode="json"),
            "states": states,
            "target": next(
                (
                    target.source
                    for target in definition.evaluation_targets
                    if target.primary
                ),
                definition.evaluation_targets[0].source
                if definition.evaluation_targets
                else {},
            ),
            "algorithm": {
                "family": "regime_graph_v2",
                "parameters": {
                    "graph_hash": execution["result"]["graph_hash"],
                    "registry_version": REGISTRY_VERSION,
                    "plan_id": plan["plan_id"],
                },
            },
            "data_snapshot": first_snapshot,
            "data_snapshots": snapshots,
            "evaluation_snapshot": execution["result"].get("evaluation_snapshot"),
            "evaluation_results": evaluation_results,
            "series_artifact": series_artifact,
            "series_summary": {
                "row_count": row_count,
                "first_observation_date": series[0]["observation_date"] if series else None,
                "last_observation_date": series[-1]["observation_date"] if series else None,
                "state_counts": execution["result"].get("state_counts") or {},
            },
            "segments": build_segments(series, frequency),
            "conditional_metrics": conditional,
            "conditional_stats": conditional,
            "transition": transition_matrix(series, states),
            "causality": causality,
            "stability": stability,
            "walk_forward": walk_forward,
            "governance": governance,
            "evidence": [
                {"kind": "formula_compilation", **audit} for audit in formula_audits
            ] + [
                {"kind": "latent_model_mapping", **audit} for audit in model_audits
            ] + [{"kind": "node_output_artifact", **artifact}] + (
                [{"kind": "evaluation_output_artifact", **evaluation_artifact}]
                if evaluation_artifact is not None
                else []
            ),
            "diagnostics": [
                {
                    "code": "REGIME_GRAPH_V2_EXECUTED",
                    "level": "info",
                    "message": "历史情景图谱已通过预热的固定签名 NJIT 计划执行。",
                },
                {
                    "code": "NODE_OUTPUTS_EXTERNALIZED",
                    "level": "info",
                    "message": "节点大数组已写入内容寻址制品，未嵌入正式运行 JSON。",
                },
            ],
            "algorithm_diagnostics": execution["result"],
            "formula_diagnostics": formula_audits or None,
            "calculation_audit": graph_execution_audit,
            "calculation_audits": calculation_audits,
            "artifact_manifest": {
                "node_outputs": artifact,
                "series": series_artifact,
                "evaluation_targets": evaluation_artifact,
            },
            "application_bindings": [],
        }
        run_payload["content_hash"] = _content_hash(run_payload)
        return self._hydrate_run(self.runs.create(_json_safe(run_payload)))

    def list_runs(self, definition_id: str | None = None) -> list[dict[str, Any]]:
        items = self.runs.list(definition_id)
        for item in items:
            if str(item.get("schema_version")) == "2.0":
                item.pop("series", None)
                item["series_included"] = False
                item["series_detail_endpoint"] = f"/api/historical-regimes/runs/{item.get('id')}"
        return items

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._hydrate_run(self.runs.get(run_id))

    @staticmethod
    def _experiment_grid_candidates(
        definition: RegimeDefinitionV2,
        parameter_grid: list[Mapping[str, Any]],
    ) -> list[tuple[RegimeDefinitionV2, list[dict[str, Any]]]]:
        if not parameter_grid or len(parameter_grid) > 6:
            raise ValidationError(
                "INVALID_EXPERIMENT_GRID",
                "parameter_grid 必须包含 1 到 6 个参数维度。",
                "parameter_grid",
            )
        node_map = {node.id: node for node in definition.graph.nodes}
        dimensions: list[tuple[str, str, list[Any]]] = []
        combination_count = 1
        seen: set[tuple[str, str]] = set()
        for index, raw_dimension in enumerate(parameter_grid):
            node_id = str(raw_dimension.get("node_id") or "")
            parameter = str(raw_dimension.get("parameter") or "")
            values = raw_dimension.get("values")
            key = (node_id, parameter)
            node = node_map.get(node_id)
            if node is None or not parameter or key in seen:
                raise ValidationError(
                    "INVALID_EXPERIMENT_GRID_DIMENSION",
                    "参数网格包含未知节点、空参数或重复维度。",
                    f"parameter_grid.{index}",
                )
            metadata = NODE_REGISTRY[node.type]
            if metadata.get("category") in {"source", "alignment"} or is_typed_formula_node(node, NODE_REGISTRY):
                raise ValidationError(
                    "EXPERIMENT_STRUCTURE_PARAMETER_BLOCKED",
                    "批量实验只允许改变不影响数据版本、端口和公式编译计划的参数。",
                    f"parameter_grid.{index}",
                )
            properties = (metadata.get("parameter_schema") or {}).get("properties") or {}
            if parameter not in properties:
                raise ValidationError(
                    "UNKNOWN_EXPERIMENT_PARAMETER",
                    f"节点 {node_id} 不支持参数 {parameter}。",
                    f"parameter_grid.{index}.parameter",
                )
            if not isinstance(values, list) or not values or len(values) > 12:
                raise ValidationError(
                    "INVALID_EXPERIMENT_VALUES",
                    "每个实验维度必须包含 1 到 12 个候选值。",
                    f"parameter_grid.{index}.values",
                )
            unique_values: list[Any] = []
            for value in values:
                if value not in unique_values:
                    unique_values.append(copy.deepcopy(value))
            combination_count *= len(unique_values)
            if combination_count > 64:
                raise ValidationError(
                    "EXPERIMENT_GRID_TOO_LARGE",
                    "参数网格展开后不能超过 64 个候选。",
                    "parameter_grid",
                )
            seen.add(key)
            dimensions.append((node_id, parameter, unique_values))

        base_payload = definition.model_dump(mode="json")
        node_positions = {
            str(node["id"]): index
            for index, node in enumerate(base_payload["graph"]["nodes"])
        }
        candidates: list[tuple[RegimeDefinitionV2, list[dict[str, Any]]]] = []
        for combination in itertools.product(*(dimension[2] for dimension in dimensions)):
            payload = copy.deepcopy(base_payload)
            differences: list[dict[str, Any]] = []
            is_baseline = True
            for (node_id, parameter, _), value in zip(dimensions, combination):
                node = payload["graph"]["nodes"][node_positions[node_id]]
                previous = node.setdefault("parameters", {}).get(parameter)
                node["parameters"][parameter] = copy.deepcopy(value)
                if previous != value:
                    is_baseline = False
                differences.append(
                    {
                        "node_id": node_id,
                        "parameter": parameter,
                        "baseline": previous,
                        "candidate": copy.deepcopy(value),
                    }
                )
            if is_baseline:
                continue
            candidate = parse_definition_v2(payload)
            validate_definition_v2(candidate)
            candidates.append((candidate, differences))
        return candidates

    @staticmethod
    def _experiment_rank_value(metrics: Mapping[str, Any], ranking_metric: str) -> float:
        if ranking_metric == "classified_ratio":
            value = metrics.get("classified_ratio")
        elif ranking_metric == "low_flip_rate":
            flip_rate = metrics.get("flip_rate")
            value = None if flip_rate is None else 1.0 - float(flip_rate)
        elif ranking_metric == "boundary_distance":
            distance = metrics.get("mean_boundary_distance_observations")
            value = None if distance is None else -float(distance)
        else:
            value = metrics.get("agreement")
        return float(value) if value is not None and np.isfinite(float(value)) else -np.inf

    def run_batch_experiment(
        self,
        reference: Mapping[str, Any],
        parameter_grid: list[Mapping[str, Any]],
        *,
        compile_token: str | None,
        mode: str = "realtime",
        as_of: str | None = None,
        ranking_metric: str = "agreement",
    ) -> dict[str, Any]:
        if mode not in {"realtime", "retrospective"}:
            raise ValidationError("INVALID_RUN_MODE", "mode 必须是 realtime 或 retrospective。", "mode")
        if ranking_metric not in {
            "agreement",
            "classified_ratio",
            "low_flip_rate",
            "boundary_distance",
        }:
            raise ValidationError(
                "INVALID_EXPERIMENT_RANKING_METRIC",
                "ranking_metric 不受支持。",
                "ranking_metric",
            )
        if (
            not isinstance(reference, Mapping)
            or str(reference.get("schema_version") or "") != "2.0"
            or not reference.get("id")
            or reference.get("revision") is None
        ):
            raise ValidationError(
                "EXPERIMENT_REQUIRES_SAVED_REFERENCE",
                "批量实验必须引用已保存的 v2 定义版本。",
                "definition",
            )
        definition = parse_definition_v2(
            self.definitions.get(str(reference["id"]), int(reference["revision"]))
        )
        validate_definition_v2(definition)
        plan = self._validate_plan(definition, compile_token)
        candidates = self._experiment_grid_candidates(definition, parameter_grid)
        if not candidates:
            raise ValidationError(
                "EXPERIMENT_GRID_EQUALS_BASELINE",
                "参数网格没有产生区别于基准的候选。",
                "parameter_grid",
            )
        source_cache: dict[str, DataBundle] = {}
        baseline_execution = self._execute_graph(
            None,
            definition,
            mode,
            as_of,
            plan=plan,
            source_cache=source_cache,
        )
        baseline_series = baseline_execution["series"]
        baseline_codes = self._series_codes(definition, baseline_series)
        baseline_summary = label_summary_kernel(baseline_codes)
        results: list[dict[str, Any]] = []
        for candidate_index, (candidate, differences) in enumerate(candidates, start=1):
            candidate_execution = self._execute_graph(
                None,
                candidate,
                mode,
                as_of,
                plan=plan,
                source_cache=source_cache,
            )
            candidate_series = candidate_execution["series"]
            candidate_codes = self._series_codes(candidate, candidate_series)
            summary = label_summary_kernel(candidate_codes)
            metrics = {
                **self._comparison_metrics(baseline_codes, candidate_codes),
                "classified_observations": int(summary[0]),
                "state_switches": int(summary[1]),
                "classified_ratio": _safe_number(summary[2]),
                "flip_rate": _safe_number(summary[3]),
            }
            spans = disagreement_spans_kernel(baseline_codes, candidate_codes)
            intervals: list[dict[str, Any]] = []
            for start_index, end_index in spans[:200]:
                start = int(start_index)
                end = int(end_index)
                intervals.append(
                    {
                        "start_index": start,
                        "end_index": end,
                        "start_date": baseline_series[start]["observation_date"],
                        "end_date": baseline_series[end]["observation_date"],
                        "observations": end - start + 1,
                    }
                )
            results.append(
                {
                    "candidate_id": f"candidate-{candidate_index}",
                    "parameter_differences": differences,
                    "definition_hash": definition_content_hash(candidate),
                    "metrics": metrics,
                    "disagreement_intervals": intervals,
                    "disagreement_intervals_truncated": int(spans.shape[0]) > len(intervals),
                    "rank_value": self._experiment_rank_value(metrics, ranking_metric),
                }
            )
        results.sort(key=lambda item: (-float(item["rank_value"]), item["candidate_id"]))
        for rank, item in enumerate(results, start=1):
            item["rank"] = rank
            if not np.isfinite(float(item["rank_value"])):
                item["rank_value"] = None
        experiment_audit = copy.deepcopy(regime_graph_numba_status())
        experiment_audit["engine"] = "regime_graph_v2_experiment"
        experiment_audit["executed_kernel_ids"] = [
            "disagreement_spans",
        ]
        experiment_audit["execution_plan_coverage"] = "complete"
        experiment_comparison_audit = execution_audit(
            "regime_graph_v2_experiment_comparison",
            ["comparison_pair", "label_summary", "prefix_stability"],
        )
        payload = {
            "schema_version": "2.0",
            "definition_id": definition.id,
            "definition_revision": definition.revision,
            "definition_snapshot_hash": definition_content_hash(definition),
            "mode": mode,
            "as_of": as_of,
            "plan_id": plan["plan_id"],
            "graph_hash": plan["graph_hash"],
            "parameter_grid": _json_safe(parameter_grid),
            "ranking_metric": ranking_metric,
            "baseline": {
                "definition_hash": definition_content_hash(definition),
                "classified_observations": int(baseline_summary[0]),
                "state_switches": int(baseline_summary[1]),
                "classified_ratio": _safe_number(baseline_summary[2]),
                "flip_rate": _safe_number(baseline_summary[3]),
            },
            "candidate_count": len(results),
            "ranking": results,
            "source_scan_cache": {
                "unique_source_views": len(source_cache),
                "scope": "shared_across_baseline_and_candidates",
            },
            "calculation_audit": experiment_audit,
            "calculation_audits": [experiment_audit, experiment_comparison_audit],
            "request_time_compilation": 0,
            "python_fallback": 0,
        }
        payload["content_hash"] = _content_hash(payload)
        return self.experiments.create(_json_safe(payload))

    def list_experiments(self, definition_id: str | None = None) -> list[dict[str, Any]]:
        return self.experiments.list(definition_id)

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        return self.experiments.get(experiment_id)

    def publish(
        self,
        run_id: str,
        usage: str | list[str],
        note: str = "",
    ) -> dict[str, Any]:
        run = self.runs.get(run_id)
        if str(run.get("schema_version")) != "2.0":
            raise ValidationError("NOT_V2_REGIME_RUN", "该运行不是 v2 图谱运行。", "run_id")
        usages = [usage] if isinstance(usage, str) else list(usage or [])
        usages = list(dict.fromkeys(str(item) for item in usages))
        if not usages or any(item not in PUBLICATION_USAGES for item in usages):
            raise ValidationError("INVALID_PUBLICATION_USAGE", "usage 包含不支持的应用目标。", "usage")
        if len(note) > 500:
            raise ValidationError("PUBLICATION_NOTE_TOO_LONG", "发布说明不能超过 500 个字符。", "note")
        if run.get("immutable") is not True or _stored_run_snapshot_hash(run) != run.get("content_hash"):
            raise ValidationError(
                "REGIME_RUN_SNAPSHOT_MISMATCH",
                "历史情景运行快照完整性校验失败，不能发布。",
                "run_id",
            )
        stored = self.definitions.get(
            str(run.get("definition_id")),
            int(run.get("definition_revision") or 0),
        )
        definition = parse_definition_v2(stored)
        if definition_content_hash(definition) != run.get("definition_snapshot_hash"):
            raise ValidationError(
                "RUN_DEFINITION_LINEAGE_MISMATCH",
                "运行快照与已保存定义版本不一致，不能发布。",
                "run_id",
            )
        governance = run.get("governance")
        governance = governance if isinstance(governance, Mapping) else {}
        eligible = set(governance.get("publish_eligible_usages") or [])
        blocked = [item for item in usages if item not in eligible]
        if blocked:
            raise ValidationError(
                "REGIME_PUBLICATION_GATE_FAILED",
                f"该运行未通过 {', '.join(blocked)} 所需的综合发布门禁。",
                "usage",
                diagnostics=[
                    {
                        "requested": usages,
                        "eligible": sorted(eligible),
                        "formal_failures": governance.get("formal_failures") or [],
                        "checks": governance.get("checks") or {},
                    }
                ],
            )
        publications = [
            {
                "id": f"publication-{uuid.uuid4().hex}",
                "usage": item,
                "published_at": _iso(_utc_now()),
                "note": note,
                "run_id": run_id,
                "definition_revision": int(run["definition_revision"]),
                "run_content_hash": run["content_hash"],
                "gate": (
                    "comprehensive_formal_gate_passed"
                    if item in FORMAL_USAGE_INTENTS
                    else "research_with_recorded_restrictions"
                ),
                "gate_checks": copy.deepcopy(governance.get("checks") or {}),
                "research_restrictions": copy.deepcopy(
                    governance.get("research_restrictions") or []
                ),
            }
            for item in usages
        ]
        updated = self.runs.add_publications(run_id, publications)
        return {
            "run_id": run_id,
            "publication": publications[0],
            "publications": updated.get("publications", []),
            "application_bindings": updated.get("application_bindings", []),
        }

    def taa_backtest(self, run_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        raw = self.runs.get(run_id)
        if str(raw.get("schema_version") or "") != "2.0":
            raise ValidationError("NOT_V2_REGIME_RUN", "该运行不是 v2 图谱运行。", "run_id")
        if raw.get("immutable") is not True or raw.get("mode") != "realtime":
            raise ValidationError(
                "TAA_REQUIRES_REALTIME_RUN",
                "TAA 回测只接受不可变的 realtime 历史情景运行。",
                "run_id",
            )
        causality = raw.get("causality") if isinstance(raw.get("causality"), dict) else {}
        if (
            not causality.get("is_causal")
            or causality.get("uses_future_data")
            or causality.get("repaints")
            or not causality.get("realtime_eligible")
        ):
            raise ValidationError("TAA_REQUIRES_CAUSAL_RUN", "TAA 回测只接受因果且不重绘的运行。", "run_id")
        analytical = copy.deepcopy(raw)
        for key in ("id", "created_at", "immutable", "publications", "content_hash"):
            analytical.pop(key, None)
        analytical["application_bindings"] = []
        if _content_hash(analytical) != raw.get("content_hash"):
            raise ValidationError("REGIME_RUN_SNAPSHOT_MISMATCH", "历史情景运行快照校验失败。", "run_id")
        accepted = [
            item
            for item in raw.get("publications") or []
            if item.get("usage") in FORMAL_USAGE_INTENTS
            and item.get("run_id") == run_id
            and item.get("run_content_hash") == raw.get("content_hash")
            and item.get("definition_revision") == raw.get("definition_revision")
            and item.get("gate") == "comprehensive_formal_gate_passed"
        ]
        if not accepted:
            raise ValidationError(
                "TAA_RUN_NOT_PUBLISHED",
                "历史情景运行必须先发布到 TAA 或正式回测。",
                "run_id",
            )
        gate = {
            "passed": True,
            "immutable": True,
            "mode": "realtime",
            "causal": True,
            "publication_usages": sorted({str(item["usage"]) for item in accepted}),
            "publication_ids": sorted(str(item["id"]) for item in accepted),
            "run_content_hash": raw["content_hash"],
        }
        return execute_taa_backtest(self._hydrate_run(raw), dict(request), gate)

    def compare(self, run_ids: list[str], reference_run_id: str | None = None) -> dict[str, Any]:
        unique_ids = list(dict.fromkeys(str(item) for item in run_ids))
        if len(unique_ids) < 2 or len(unique_ids) > 8:
            raise ValidationError("INVALID_COMPARE_RUNS", "模型比较需要 2 至 8 个不同运行。", "run_ids")
        if reference_run_id and reference_run_id not in unique_ids:
            raise ValidationError("INVALID_REFERENCE_RUN", "reference_run_id 必须包含在 run_ids 中。", "reference_run_id")
        return self.compare_snapshots(
            [self.get_run(run_id) for run_id in unique_ids],
            reference_run_id,
        )

    def compare_snapshots(
        self,
        runs: list[dict[str, Any]],
        reference_run_id: str | None = None,
    ) -> dict[str, Any]:
        unique_ids = list(dict.fromkeys(str(item.get("id")) for item in runs))
        if len(unique_ids) < 2 or len(unique_ids) > 8 or len(unique_ids) != len(runs):
            raise ValidationError("INVALID_COMPARE_RUNS", "模型比较需要 2 至 8 个不同运行。", "run_ids")
        if reference_run_id and reference_run_id not in unique_ids:
            raise ValidationError("INVALID_REFERENCE_RUN", "reference_run_id 必须包含在 run_ids 中。", "reference_run_id")
        result = compare_run_snapshots(runs, reference_run_id)
        return {**result, "compared_at": _iso(_utc_now())}

    def get_preview(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            self._prune_jobs()
            job = self._jobs.get(job_id)
            if job is None:
                raise NotFoundError("REGIME_PREVIEW_NOT_FOUND", "未找到试算任务，任务可能已过期。")
            return self._public_job(job)

    def cancel_preview(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            self._prune_jobs()
            job = self._jobs.get(job_id)
            if job is None:
                raise NotFoundError("REGIME_PREVIEW_NOT_FOUND", "未找到试算任务，任务可能已过期。")
            if job["status"] not in TERMINAL_JOB_STATUSES:
                job["cancel_event"].set()
                job["status"] = "cancelled"
                job["stage"] = "cancelled"
                job["progress"] = 1.0
                job["message"] = "试算已取消。"
                job["updated_at"] = _iso(_utc_now())
                job["expires_at"] = _utc_now() + timedelta(seconds=int(job["ttl_seconds"]))
            return self._public_job(job)

    def preview_overview(self, job_id: str) -> dict[str, Any]:
        """Read the frozen final result without executing the graph again."""
        with self._lock:
            self._prune_jobs()
            job = self._jobs.get(job_id)
            if job is None:
                raise NotFoundError("REGIME_PREVIEW_NOT_FOUND", "未找到试算任务，任务可能已过期。")
            if job["status"] != "completed":
                raise ConflictError("REGIME_PREVIEW_NOT_COMPLETE", "试算尚未完成，暂不能读取结果总览。")
            if job.get("_series") is None or job.get("result") is None:
                raise NotFoundError("REGIME_PREVIEW_RESULT_UNAVAILABLE", "试算结果已不可用，请重新运行。")
            if job.get("preview_target") is not None:
                raise ConflictError("NODE_PREVIEW_HAS_NO_REGIME_OVERVIEW", "这是节点数据预览；完整情景结果需要运行识别算法。")
            if job.get("_overview") is None:
                definition = job["definition"]
                job["_overview"] = build_result_overview(
                    run_kind="preview",
                    run_id=job_id,
                    definition=definition.model_dump(mode="json"),
                    series=job["_series"],
                    definition_id=definition.id,
                    revision=definition.revision,
                    definition_hash=job["definition_hash"],
                    graph_hash=job["graph_hash"],
                    mode=job["mode"],
                    as_of=job["as_of"],
                    data_snapshots=job["result"].get("data_snapshots"),
                    series_endpoint=f"/api/historical-regimes/preview-runs/{job_id}/series",
                    result=job["result"],
                    created_at=job["created_at"],
                )
            return copy.deepcopy(job["_overview"])

    def normalized_preview_chart(
        self, job_id: str, *, node_id: str, port: str = "value", base_index: int = 0,
    ) -> dict[str, Any]:
        """Rebase a frozen numeric output for display; never change stored values."""
        with self._lock:
            self._prune_jobs()
            job = self._jobs.get(job_id)
            if job is None:
                raise NotFoundError("REGIME_PREVIEW_NOT_FOUND", "预览已过期，请重新预览节点。")
            if job["status"] != "completed":
                raise ConflictError("REGIME_PREVIEW_NOT_COMPLETE", "节点预览尚未完成。")
            output = (job["_node_outputs"].get(node_id) or {}).get(port)
            if output is None:
                raise NotFoundError("REGIME_PREVIEW_PORT_NOT_FOUND", "未找到所选节点的输出。")
            metadata = NODE_REGISTRY[job["_node_types"][node_id]]
            value_type = next(item["type"] for item in metadata["outputs"] if item["name"] == port)
        if value_type != "series<float64>" or output.values.ndim != 1:
            raise ValidationError("NON_NUMERIC_NORMALIZATION", "仅数值序列支持区间归一化，市场状态等枚举值不适用。", "port")
        values = np.ascontiguousarray(output.values, dtype=np.float64)
        if not 0 <= base_index < values.size:
            raise ValidationError("INVALID_NORMALIZATION_BASE", "所选区间没有可用观测，请调整时间范围。", "base_index")
        base = float(values[base_index])
        if not np.isfinite(base) or base <= 0.0:
            raise ValidationError("INVALID_NORMALIZATION_BASE", "区间首日数值须大于 0 且非缺失，请调整区间起点。", "base_index")
        # Reuse the already warmed fixed-signature arithmetic lane.
        normalized = binary_math_kernel(values, np.full(values.size, base, dtype=np.float64), 3)
        relative = binary_math_kernel(normalized, np.ones(values.size, dtype=np.float64), 1)
        changes = binary_math_kernel(relative, np.full(values.size, 100.0, dtype=np.float64), 2)
        return {
            "run_id": job_id, "node_id": node_id, "port": port,
            "base_index": base_index,
            "base_date": pd.Timestamp(int(output.dates[base_index]), unit="ns").date().isoformat(),
            "base_value": base,
            "values": [_safe_number(value) for value in normalized],
            "change_pct": [_safe_number(value) for value in changes],
            "execution": copy.deepcopy(self._runtime_audit),
        }

    def preview_series(
        self,
        job_id: str,
        *,
        node_id: str | None = None,
        port: str | None = None,
        offset: int = 0,
        limit: int = 1000,
    ) -> dict[str, Any]:
        with self._lock:
            self._prune_jobs()
            job = self._jobs.get(job_id)
            if job is None:
                raise NotFoundError("REGIME_PREVIEW_NOT_FOUND", "未找到试算任务，任务可能已过期。")
            if job["status"] != "completed":
                raise ConflictError("REGIME_PREVIEW_NOT_COMPLETE", "试算尚未完成，暂不能读取序列。")
            series = job["_series"]
            node_outputs = job["_node_outputs"]
            node_types = job["_node_types"]
            expires_at = job["expires_at"]
            if node_id is None and job.get("preview_target") is not None:
                node_id = job["preview_target"]["node_id"]
                port = port or job["preview_target"]["port"]
        if offset < 0 or limit < 1 or limit > 5000:
            raise ValidationError("INVALID_SERIES_PAGE", "offset 必须非负且 limit 必须在 1 到 5000 之间。", "limit")
        if node_id is None:
            total = len(series)
            return {
                "id": job_id,
                "node_id": None,
                "port": "state",
                "offset": offset,
                "limit": limit,
                "total": total,
                "items": copy.deepcopy(series[offset : offset + limit]),
                "expires_at": _iso(expires_at),
            }
        outputs = node_outputs.get(node_id)
        if outputs is None:
            raise NotFoundError("REGIME_PREVIEW_NODE_NOT_FOUND", "试算结果中没有该节点。")
        selected_port = port or ("value" if "value" in outputs else next(iter(outputs)))
        port_value = outputs.get(selected_port)
        if port_value is None:
            raise NotFoundError("REGIME_PREVIEW_PORT_NOT_FOUND", "试算结果中没有该节点端口。")
        values = port_value.values
        total = int(values.shape[0])
        items: list[dict[str, Any]] = []
        stop = min(offset + limit, total)
        for absolute_index in range(offset, stop):
            value = values[absolute_index]
            item = {
                "index": absolute_index,
                "observation_date": pd.Timestamp(
                    int(port_value.dates[absolute_index]), unit="ns"
                ).date().isoformat(),
            }
            if values.ndim == 1:
                if np.issubdtype(values.dtype, np.integer):
                    item["value"] = int(value)
                else:
                    item["value"] = _safe_number(value)
            else:
                item["values"] = [_safe_number(part) for part in value]
            items.append(item)
        context = preview_output_context(job["definition"], node_outputs, node_id)
        metadata = NODE_REGISTRY[node_types[node_id]]
        port_type = next(item["type"] for item in metadata["outputs"] if item["name"] == selected_port)
        return {
            "id": job_id,
            "node_id": node_id,
            "node_type": node_types[node_id],
            **context,
            "port": selected_port,
            "value_type": port_type,
            "offset": offset,
            "limit": limit,
            "total": total,
            "items": items,
            "expires_at": _iso(expires_at),
        }


__all__ = [
    "DEFAULT_PREVIEW_TTL_SECONDS",
    "JOB_STATUSES",
    "RegimeGraphV2Service",
    "TERMINAL_JOB_STATUSES",
    "hydrate_v2_run_snapshot",
]
