from __future__ import annotations

"""Automatic asset-class construction from a product pool.

Orchestration only: this module resolves products, loads market data, hands
every numerical step to :mod:`auto_class_numba`, and serialises a draft that is
shape-compatible with ``asset_alloc_info`` so the result can be opened in the
manual asset-class workspace or saved through ``/api/save-allocation``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

try:
    from backend.auto_class_numba import (
        LINKAGE_AVERAGE,
        LINKAGE_COMPLETE,
        LINKAGE_WARD,
        WEIGHT_AFFINITY,
        WEIGHT_EQUAL,
        WEIGHT_INV_VAR,
        WEIGHT_INV_VOL,
        affinity_from_distance_kernel,
        agglomerative_linkage_kernel,
        allocate_block_clusters_kernel,
        apply_weight_caps_kernel,
        auto_class_execution_audit,
        capacity_assign_kernel,
        correlation_eigenvalues_kernel,
        correlation_matrix_kernel,
        corr_to_distance_kernel,
        cross_class_corr_kernel,
        cut_linkage_kernel,
        euclidean_distance_kernel,
        intra_class_mean_corr_kernel,
        intra_class_weights_kernel,
        kmeans_kernel,
        kmedoids_kernel,
        denoise_correlation_kernel,
        spectral_labels_kernel,
        gmm_kernel,
        mean_offdiagonal_kernel,
        pca_features_kernel,
        robust_standardize_kernel,
        silhouette_kernel,
        winsorize_returns_kernel,
    )
    from backend.fit import _map_to_ts, _returns_wide, load_adj_nav_pit
    from backend.fund_taxonomy import (
        TAXONOMY_LEVEL_LABELS,
        TAXONOMY_LEVELS,
        TaxonomyLabel,
        classify,
        taxonomy_tree,
    )
    from backend.market_data import resolve_market_data_file
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from auto_class_numba import (
        LINKAGE_AVERAGE,
        LINKAGE_COMPLETE,
        LINKAGE_WARD,
        WEIGHT_AFFINITY,
        WEIGHT_EQUAL,
        WEIGHT_INV_VAR,
        WEIGHT_INV_VOL,
        affinity_from_distance_kernel,
        agglomerative_linkage_kernel,
        allocate_block_clusters_kernel,
        apply_weight_caps_kernel,
        auto_class_execution_audit,
        capacity_assign_kernel,
        correlation_eigenvalues_kernel,
        correlation_matrix_kernel,
        corr_to_distance_kernel,
        cross_class_corr_kernel,
        cut_linkage_kernel,
        euclidean_distance_kernel,
        intra_class_mean_corr_kernel,
        intra_class_weights_kernel,
        kmeans_kernel,
        kmedoids_kernel,
        denoise_correlation_kernel,
        spectral_labels_kernel,
        gmm_kernel,
        mean_offdiagonal_kernel,
        pca_features_kernel,
        robust_standardize_kernel,
        silhouette_kernel,
        winsorize_returns_kernel,
    )
    from fit import _map_to_ts, _returns_wide, load_adj_nav_pit
    from fund_taxonomy import (
        TAXONOMY_LEVEL_LABELS,
        TAXONOMY_LEVELS,
        TaxonomyLabel,
        classify,
        taxonomy_tree,
    )
    from market_data import resolve_market_data_file



# The PIT package is imported bare-first, unlike the modules above. Startup
# warms `pit.audit`'s dataset cache; reaching the same code through
# `backend.pit.audit` would load a second copy with a cold cache and turn the
# first strict-mode run into a full 37M-row rescan.
try:
    from pit.context import (
        PitContextError,
        ResearchContext,
        build_context,
        require_usable,
    )
    from pit.frame import LATEST_ONLY, NOT_APPLICABLE, REPLAYED, read_pit
except ModuleNotFoundError:  # pragma: no cover - imported as a backend.* module
    from backend.pit.context import (
        PitContextError,
        ResearchContext,
        build_context,
        require_usable,
    )
    from backend.pit.frame import LATEST_ONLY, NOT_APPLICABLE, REPLAYED, read_pit


MIN_OBSERVATIONS = 60
# 25 robust sigma: real Chinese ETF fat tails top out near 18 sigma, while an
# unadjusted split/dividend print lands in the hundreds.
WINSOR_SIGMA = 25.0
MAX_AUTO_K = 8
# Kaufman/Rousseeuw read silhouette below 0.25 as "no substantial structure".
# Inside a contract block that is the difference between finding two real
# sub-styles and cutting five 沪深300 trackers into look-alike halves.
BLOCK_SPLIT_SILHOUETTE = 0.25
PCA_COMPONENTS = 5
DEFAULT_SEED = 20260101

ALGORITHMS = {
    "rule": "合同分类规则映射",
    "hierarchical": "相关性层次聚类",
    "kmedoids": "K-medoids（代表产品）",
    "kmeans": "K-means（特征质心）",
    "spectral": "谱聚类（相关性图）",
    "gmm": "高斯混合（软分配）",
}
FEATURE_SETS = {
    "correlation": "收益相关性距离",
    "denoised": "去噪相关性距离（RMT）",
    "metrics": "风险收益画像",
    "pca": "主成分载荷",
    "blend": "画像 + 主成分",
}
LINKAGE_METHODS = {"average": LINKAGE_AVERAGE, "complete": LINKAGE_COMPLETE, "ward": LINKAGE_WARD}
WEIGHT_MODES = {
    "equal": WEIGHT_EQUAL,
    "inv_vol": WEIGHT_INV_VOL,
    "inv_var": WEIGHT_INV_VAR,
    "affinity": WEIGHT_AFFINITY,
}
METRIC_FEATURE_COLUMNS = (
    "return_1y",
    "return_3y",
    "annual_volatility_1y",
    "max_drawdown_3y",
    "sharpe_1y",
    "calmar_3y",
    "premium_discount_latest",
    "amount_avg_20d",
)

# Which taxonomy level names the classes and, for ``block_by``, which level is a
# hard partition the statistics may not cross.  Tables live in `fund_taxonomy`.
BLOCK_MODES = {
    "none": "不分层（纯统计聚类）",
    **{level: TAXONOMY_LEVEL_LABELS[level] for level in TAXONOMY_LEVELS},
}


class AutoClassError(ValueError):
    """Raised when the request cannot produce an honest classification."""


@dataclass
class AutoClassRequestSpec:
    codes: list[str]
    names: list[str] = field(default_factory=list)
    start_date: str = "2020-01-01"
    algorithm: str = "hierarchical"
    features: str = "correlation"
    linkage: str = "average"
    k: Optional[int] = None
    size_min: int = 2
    size_max: int = 8
    unassigned_policy: str = "park"
    weight_mode: str = "inv_vol"
    seed: int = DEFAULT_SEED
    # Which contract-taxonomy level names the classes.
    taxonomy_level: str = "asset_class"
    # Taxonomy level the statistics may not cross; "none" keeps pure clustering.
    block_by: str = "none"
    # Research day: only NAV rows announced on or before it may be used.
    as_of: Optional[str] = None
    # RESEARCH tolerates approximate availability; STRICT_PIT fails closed.
    run_mode: str = "RESEARCH"
    data_release_id: Optional[str] = None
    # product code -> product-pool max weight as a fraction in (0, 1]
    max_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class _Product:
    code: str
    name: str
    column: Optional[str]
    instrument_type: str = "etf"
    fund_type: str = ""
    invest_type: str = ""
    benchmark: str = ""
    index_name: str = ""
    management: str = ""
    max_weight: Optional[float] = None
    taxonomy: Optional[TaxonomyLabel] = None


def _finite(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


def product_taxonomy(product: _Product) -> TaxonomyLabel:
    """Contract taxonomy for one product, cached on the product itself."""

    if product.taxonomy is None:
        product.taxonomy = classify(
            (
                product.fund_type,
                product.invest_type,
                product.benchmark,
                product.index_name,
                product.name,
            )
        )
    return product.taxonomy


def rule_label(product: _Product, level: str = "asset_class") -> str:
    """Map contract metadata to a taxonomy label; the deterministic baseline."""

    return product_taxonomy(product).at(level)


def _lookup_max_weight(limits: dict[str, float], code: str) -> Optional[float]:
    """Resolve a pool restriction by exact code, then by the bare code."""

    if not limits:
        return None
    value = limits.get(code)
    if value is None:
        value = limits.get(code.split(".")[0])
    if value is None:
        upper = code.upper()
        value = limits.get(upper) or limits.get(upper.split(".")[0])
    parsed = _finite(value)
    return parsed if parsed is not None and 0.0 < parsed <= 1.0 else None


METADATA_COLUMNS = (
    "ts_code",
    "code",
    "name",
    "instrument_type",
    "fund_type",
    "invest_type",
    "benchmark",
    "index_name",
    "management",
)


def _load_instrument_metadata(
    data_dir: Path, codes: Iterable[str], context: ResearchContext
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Contract metadata as it was knowable on the research day.

    Read through :func:`pit.frame.read_pit` rather than off the latest-state
    file, because this table is not decoration: ``rule`` classifies entirely on
    it, ``blockBy`` blocks on it, and it names every class. A fund reclassified
    from 债券型 to 混合型 in 2024 would otherwise be blocked as 混合型 in a 2018
    run — the clusters themselves change, not just their labels.

    `read_pit` replays the dimension snapshot log when one exists and reports
    LATEST_ONLY when it does not, so hindsight can no longer pass as a cut.
    """

    wanted = {str(code).strip() for code in codes if str(code).strip()}
    bare = {code.split(".")[0] for code in wanted}
    columns = list(METADATA_COLUMNS)
    frames: list[pd.DataFrame] = []
    coverages: list[str] = []
    warnings: list[str] = []
    snapshots: dict[str, Optional[str]] = {}
    for dataset_id in ("etf_info", "fund_info"):
        read = read_pit(dataset_id, data_dir, context, columns=columns)
        if not int(read.lineage.get("rows_before_cut") or 0):
            # A table this deployment simply does not carry says nothing about
            # coverage; counting it would flag every 场内-only install as
            # latest-state on evidence it never had.
            continue
        coverages.append(str(read.lineage.get("coverage")))
        warnings.extend(read.lineage.get("warnings") or [])
        snapshots[dataset_id] = read.lineage.get("snapshot_used")
        frame = read.frame
        if frame.empty or "ts_code" not in frame.columns:
            continue
        ts_codes = frame["ts_code"].astype(str)
        mask = ts_codes.isin(wanted) | ts_codes.str.split(".").str[0].isin(bare)
        if "code" in frame.columns:
            mask = mask | frame["code"].astype(str).isin(bare)
        frames.append(frame[mask])
    lineage: dict[str, Any] = {
        # The weaker of the two decides: one replayed table beside one
        # latest-state table is still a latest-state classification.
        "coverage": LATEST_ONLY if LATEST_ONLY in coverages else (
            REPLAYED if REPLAYED in coverages else NOT_APPLICABLE
        ),
        "snapshot_used": snapshots,
        "warnings": warnings,
    }
    if not frames:
        return pd.DataFrame(columns=columns), lineage
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["ts_code"], keep="first"
    )
    return merged, lineage


def _load_metric_features(data_dir: Path, codes: list[str]) -> pd.DataFrame:
    path = resolve_market_data_file("instrument_metrics_snapshot.parquet", data_dir)
    if not path.exists():
        return pd.DataFrame(columns=["ts_code", *METRIC_FEATURE_COLUMNS])
    try:
        frame = pd.read_parquet(path, columns=["ts_code", *METRIC_FEATURE_COLUMNS])
    except Exception:
        frame = pd.read_parquet(path)
        keep = [column for column in ("ts_code", *METRIC_FEATURE_COLUMNS) if column in frame.columns]
        frame = frame[keep]
    ts_codes = frame["ts_code"].astype(str)
    bare = {code.split(".")[0] for code in codes}
    mask = ts_codes.isin(set(codes)) | ts_codes.str.split(".").str[0].isin(bare)
    return frame[mask].drop_duplicates(subset=["ts_code"], keep="last")


def _resolve_products(
    metadata: pd.DataFrame,
    spec: AutoClassRequestSpec,
    nav_frame: pd.DataFrame,
    available: list[str],
) -> tuple[list[_Product], list[dict[str, str]]]:
    by_ts: dict[str, dict[str, Any]] = {}
    by_bare: dict[str, dict[str, Any]] = {}
    for record in metadata.to_dict("records"):
        ts_code = _text(record.get("ts_code"))
        if not ts_code:
            continue
        by_ts[ts_code] = record
        by_bare.setdefault(ts_code.split(".")[0], record)

    names = list(spec.names) + [""] * max(0, len(spec.codes) - len(spec.names))
    products: list[_Product] = []
    skipped: list[dict[str, str]] = []
    seen_columns: set[str] = set()
    for index, raw_code in enumerate(spec.codes):
        code = str(raw_code).strip()
        if not code:
            continue
        requested_name = _text(names[index])
        row = by_ts.get(code) or by_bare.get(code.split(".")[0])
        display_name = requested_name or (_text(row.get("name")) if row is not None else "") or code
        column = _map_to_ts(nav_frame, available, code, display_name)
        if column is None:
            skipped.append({"code": code, "name": display_name, "reason": "NO_SERIES", "detail": "样本期内没有可用净值"})
            continue
        if column in seen_columns:
            skipped.append({"code": code, "name": display_name, "reason": "DUPLICATE", "detail": "与已选产品指向同一条净值序列"})
            continue
        seen_columns.add(column)
        products.append(
            _Product(
                code=code,
                name=display_name,
                column=column,
                instrument_type=(_text(row.get("instrument_type")) if row is not None else "") or ("fund" if code.upper().endswith(".OF") else "etf"),
                fund_type=_text(row.get("fund_type")) if row is not None else "",
                invest_type=_text(row.get("invest_type")) if row is not None else "",
                benchmark=_text(row.get("benchmark")) if row is not None else "",
                index_name=_text(row.get("index_name")) if row is not None else "",
                management=_text(row.get("management")) if row is not None else "",
                max_weight=_lookup_max_weight(spec.max_weights, code),
            )
        )
    return products, skipped


def _build_feature_space(
    data_dir: Path,
    spec: AutoClassRequestSpec,
    products: list[_Product],
    returns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (correlation, distance, standardized_features) for the chosen lane.

    `returns` must already be winsorised; the raw series stays with the display
    pipeline in ``/api/fit-classes``.
    """

    correlation = correlation_matrix_kernel(np.ascontiguousarray(returns))
    if spec.features in {"correlation", "denoised"}:
        # The denoised matrix only shapes the clustering geometry; every
        # reported correlation stays the raw observed one.
        basis = correlation
        if spec.features == "denoised":
            basis = denoise_correlation_kernel(
                np.ascontiguousarray(correlation), int(returns.shape[0])
            )
        distance = corr_to_distance_kernel(np.ascontiguousarray(basis))
        features = robust_standardize_kernel(np.ascontiguousarray(basis))
        return correlation, distance, features

    blocks: list[np.ndarray] = []
    if spec.features in {"metrics", "blend"}:
        metric_frame = _load_metric_features(data_dir, [product.code for product in products])
        lookup: dict[str, dict[str, Any]] = {}
        for record in metric_frame.to_dict("records"):
            ts_code = _text(record.get("ts_code"))
            if ts_code:
                lookup[ts_code] = record
                lookup.setdefault(ts_code.split(".")[0], record)
        raw = np.full((len(products), len(METRIC_FEATURE_COLUMNS)), np.nan, dtype=np.float64)
        for index, product in enumerate(products):
            row = lookup.get(product.code) or lookup.get(product.code.split(".")[0])
            if row is None:
                continue
            for column_index, column in enumerate(METRIC_FEATURE_COLUMNS):
                value = _finite(row.get(column))
                if value is not None:
                    raw[index, column_index] = value
        if not np.any(np.isfinite(raw)):
            raise AutoClassError("所选产品在指标快照中没有任何可用的风险收益特征，请改用收益相关性特征")
        blocks.append(robust_standardize_kernel(np.ascontiguousarray(raw)))
    if spec.features in {"pca", "blend"}:
        components = min(PCA_COMPONENTS, len(products))
        loadings = pca_features_kernel(np.ascontiguousarray(correlation), int(components))
        blocks.append(robust_standardize_kernel(np.ascontiguousarray(loadings)))
    if not blocks:
        raise AutoClassError(f"不支持的特征集：{spec.features}")
    features = np.ascontiguousarray(np.hstack(blocks) if len(blocks) > 1 else blocks[0])
    distance = euclidean_distance_kernel(features)
    return correlation, distance, features


def _linkage(spec: AutoClassRequestSpec, distance: np.ndarray) -> np.ndarray:
    method = LINKAGE_METHODS.get(spec.linkage)
    if method is None:
        raise AutoClassError(f"不支持的连接方式：{spec.linkage}")
    return agglomerative_linkage_kernel(np.ascontiguousarray(distance), int(method))


def _cluster(
    spec: AutoClassRequestSpec,
    distance: np.ndarray,
    features: np.ndarray,
    clusters: int,
    linkage: Optional[np.ndarray] = None,
) -> np.ndarray:
    if spec.algorithm == "kmedoids":
        labels, _ = kmedoids_kernel(np.ascontiguousarray(distance), int(clusters), int(spec.seed), 100)
        return labels
    if spec.algorithm == "kmeans":
        return kmeans_kernel(np.ascontiguousarray(features), int(clusters), int(spec.seed), 200)
    if spec.algorithm == "spectral":
        return spectral_labels_kernel(
            np.ascontiguousarray(distance), int(clusters), int(spec.seed), 200
        )
    if spec.algorithm == "gmm":
        # Posteriors stay inside the kernel: the downstream affinity, capacity
        # and weighting chain is distance-scaled and must not mix in a [0, 1]
        # probability.
        labels, _ = gmm_kernel(np.ascontiguousarray(features), int(clusters), int(spec.seed), 200)
        return labels
    # The merge tree does not depend on K, so it is built once and only cut here.
    tree = _linkage(spec, distance) if linkage is None else linkage
    return cut_linkage_kernel(np.ascontiguousarray(tree), distance.shape[0], int(clusters))


def _taxonomy_groups(products: list[_Product], level: str) -> tuple[np.ndarray, list[str]]:
    """Group products by their contract taxonomy label at `level`.

    Order follows first appearance so the same pool always produces the same
    class ordering, which the manual workspace relies on.
    """

    order: list[str] = []
    labels = np.full(len(products), -1, dtype=np.int64)
    for index, product in enumerate(products):
        label = rule_label(product, level)
        if label not in order:
            order.append(label)
        labels[index] = order.index(label)
    return np.ascontiguousarray(labels), order


def _block_best_k(
    spec: AutoClassRequestSpec,
    distance: np.ndarray,
    features: np.ndarray,
    capacity: int,
) -> tuple[int, Optional[float]]:
    """Silhouette-chosen K inside one taxonomy block.

    A block only splits when some K actually scores positive: a homogeneous
    block (five 沪深300 trackers) must stay one class instead of being cut into
    look-alike halves.
    """

    best_k, best_score = 1, BLOCK_SPLIT_SILHOUETTE
    for candidate in range(2, min(capacity, MAX_AUTO_K) + 1):
        labels = _cluster(spec, distance, features, candidate, None)
        score, _ = silhouette_kernel(
            np.ascontiguousarray(distance), np.ascontiguousarray(labels), int(candidate)
        )
        value = _finite(score)
        if value is not None and value > best_score:
            best_k, best_score = candidate, value
    return best_k, (best_score if best_k > 1 else None)


def _blocked_cluster(
    spec: AutoClassRequestSpec,
    distance: np.ndarray,
    features: np.ndarray,
    block_ids: np.ndarray,
    block_names: list[str],
    requested_k: Optional[int],
) -> tuple[np.ndarray, np.ndarray, int, list[dict[str, Any]]]:
    """Cluster *inside* each contract block; blocks never merge or trade members.

    This is the 「先分层、再聚类」 practice: a correlation window can make a gold
    ETF look like an equity ETF, but the contract cannot, so the taxonomy is a
    hard constraint and the statistics only refine what is inside it.
    """

    blocks = len(block_names)
    sizes = np.ascontiguousarray(
        np.array([int(np.sum(block_ids == block)) for block in range(blocks)], dtype=np.int64)
    )
    # A target beyond total capacity is clamped to it, so this reads back the
    # per-block ceiling without duplicating the kernel's capacity rule.
    capacity = allocate_block_clusters_kernel(sizes, np.int64(1 << 40), np.int64(max(1, spec.size_min)))
    if requested_k is None:
        per_block = np.zeros(blocks, dtype=np.int64)
    else:
        per_block = allocate_block_clusters_kernel(
            sizes, np.int64(requested_k), np.int64(max(1, spec.size_min))
        )

    labels = np.full(block_ids.shape[0], -1, dtype=np.int64)
    cluster_block: list[int] = []
    report: list[dict[str, Any]] = []
    offset = 0
    for block in range(blocks):
        rows = np.flatnonzero(block_ids == block)
        if rows.size == 0:
            continue
        sub_distance = np.ascontiguousarray(distance[np.ix_(rows, rows)])
        sub_features = np.ascontiguousarray(features[rows, :])
        if requested_k is None:
            count, score = _block_best_k(spec, sub_distance, sub_features, int(capacity[block]))
        else:
            count, score = int(per_block[block]), None
        if count <= 1 or rows.size < 2:
            labels[rows] = offset
            cluster_block.append(block)
            report.append({"block": block_names[block], "size": int(rows.size), "k": 1, "silhouette": None})
            offset += 1
            continue
        sub_labels = _cluster(spec, sub_distance, sub_features, count, None)
        if score is None:
            raw, _ = silhouette_kernel(sub_distance, np.ascontiguousarray(sub_labels), int(count))
            score = _finite(raw)
        for position, row in enumerate(rows):
            labels[int(row)] = offset + int(sub_labels[position])
        cluster_block.extend([block] * count)
        report.append({"block": block_names[block], "size": int(rows.size), "k": count, "silhouette": score})
        offset += count
    return (
        np.ascontiguousarray(labels),
        np.ascontiguousarray(np.array(cluster_block, dtype=np.int64)),
        offset,
        report,
    )


def _suggest_k(
    spec: AutoClassRequestSpec,
    distance: np.ndarray,
    features: np.ndarray,
    upper: int,
    linkage: Optional[np.ndarray],
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    for candidate in range(2, upper + 1):
        labels = _cluster(spec, distance, features, candidate, linkage)
        score, _ = silhouette_kernel(np.ascontiguousarray(distance), np.ascontiguousarray(labels), int(candidate))
        suggestions.append({"k": candidate, "silhouette": _finite(score)})
    return suggestions


def _significant_eigenvalues(correlation: np.ndarray, observations: int) -> int:
    """Marchenko-Pastur upper bound: eigenvalues above it are structure, not noise."""

    size = correlation.shape[0]
    if size < 2 or observations <= size:
        return 0
    ratio = size / observations
    upper = (1.0 + np.sqrt(ratio)) ** 2
    eigenvalues = correlation_eigenvalues_kernel(np.ascontiguousarray(correlation))
    return int(np.sum(eigenvalues > upper))


def _class_names(
    products: list[_Product],
    labels: np.ndarray,
    clusters: int,
    medoids: dict[int, int],
    preset: Optional[list[str]],
    level: str = "asset_class",
) -> list[str]:
    """Name each class after its dominant contract label, disambiguated by medoid."""

    if preset is not None:
        return list(preset)
    names: list[str] = []
    used: dict[str, int] = {}
    for cluster in range(clusters):
        members = [products[index] for index in range(len(products)) if labels[index] == cluster]
        if not members:
            base = f"大类{cluster + 1}"
        else:
            counts: dict[str, int] = {}
            for member in members:
                label = rule_label(member, level)
                counts[label] = counts.get(label, 0) + 1
            base = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
        seen = used.get(base, 0)
        used[base] = seen + 1
        if seen == 0:
            names.append(base)
            continue
        # Same contract family split across clusters: let the representative
        # product name the class instead of a meaningless numeric suffix.
        qualifier = ""
        medoid_index = medoids.get(cluster, -1)
        if medoid_index >= 0:
            medoid = products[medoid_index]
            qualifier = medoid.index_name or medoid.invest_type or medoid.name
        candidate = f"{base}-{qualifier}" if qualifier else f"{base}{seen + 1}"
        if candidate in names:
            candidate = f"{candidate}{seen + 1}"
        names.append(candidate)
    return names


def run_auto_classification(data_dir: Path, spec: AutoClassRequestSpec) -> dict[str, Any]:
    """Produce a class draft, its diagnostics and the NJIT execution audit."""

    if spec.algorithm not in ALGORITHMS:
        raise AutoClassError(f"不支持的算法：{spec.algorithm}")
    if spec.features not in FEATURE_SETS:
        raise AutoClassError(f"不支持的特征集：{spec.features}")
    if spec.weight_mode not in WEIGHT_MODES:
        raise AutoClassError(f"不支持的类内权重方式：{spec.weight_mode}")
    if spec.unassigned_policy not in {"park", "force"}:
        raise AutoClassError(f"不支持的未归类策略：{spec.unassigned_policy}")
    if spec.taxonomy_level not in TAXONOMY_LEVELS:
        raise AutoClassError(f"不支持的合同分类层级：{spec.taxonomy_level}")
    if spec.block_by not in BLOCK_MODES:
        raise AutoClassError(f"不支持的分层方式：{spec.block_by}")
    try:
        # Validates as_of/run_mode together, and refuses STRICT_PIT without a
        # research day rather than letting it behave like research mode.
        context = build_context(spec.as_of, spec.run_mode, spec.data_release_id)
        # Classification names classes from the contract taxonomy, which lives in
        # a latest-state dimension table; strict mode must not let that pass.
        require_usable(data_dir, context, ["etf_nav", "fund_nav", "etf_info", "fund_info"])
        if spec.features in {"metrics", "blend"} and context.as_of and context.strict:
            # The indicator snapshot is one row per product, recomputed in place:
            # no availability column, no snapshot log, nothing to cut on. Paired
            # with a research day it is not a degraded read, it is look-ahead —
            # 2026 的三年最大回撤拿去给 2018 年分类。Research mode warns instead.
            raise PitContextError(
                "严格 PIT 模式下不能使用「风险收益画像」特征："
                f"指标快照只有当期一版，无法还原 {context.as_of} 当日的画像。"
            )
    except PitContextError as exc:
        raise AutoClassError(str(exc)) from exc
    if spec.size_min < 1:
        raise AutoClassError("每类最少产品数不能小于 1")
    if spec.size_max < spec.size_min:
        raise AutoClassError("每类最多产品数不能小于最少产品数")
    codes = [str(code).strip() for code in spec.codes if str(code).strip()]
    if len(codes) < 2:
        raise AutoClassError("自动分类至少需要 2 个产品")

    try:
        start = pd.to_datetime(spec.start_date)
    except Exception as exc:  # noqa: BLE001 - surfaced as a 400 by the route
        raise AutoClassError("startDate 格式错误，应为 YYYY-MM-DD") from exc

    loaded = load_adj_nav_pit(
        data_dir, codes, spec.names, as_of=spec.as_of, run_mode=spec.run_mode
    )
    nav_frame = loaded.frame
    pit_lineage = loaded.lineage
    if nav_frame.empty:
        raise AutoClassError("所选产品在数据集中没有净值记录")
    wide = _returns_wide(nav_frame, start)
    if wide.empty:
        raise AutoClassError("样本期内没有所有产品共同覆盖的交易日，请放宽开始日期或缩减产品")
    available = list(wide.columns.astype(str))
    try:
        # Strict mode refuses a latest-state contract table here rather than
        # relabelling history; research mode warns further down.
        metadata, taxonomy_lineage = _load_instrument_metadata(
            data_dir, spec.codes, context
        )
    except PitContextError as exc:
        raise AutoClassError(str(exc)) from exc
    products, skipped = _resolve_products(metadata, spec, nav_frame, available)
    if len(products) < 2:
        raise AutoClassError("可用于分类的产品不足 2 个，请检查产品代码或样本期")

    returns = np.ascontiguousarray(wide[[product.column for product in products]].to_numpy(dtype=np.float64))
    observations = int(returns.shape[0])
    if observations < MIN_OBSERVATIONS:
        raise AutoClassError(f"共同样本仅 {observations} 个交易日，少于 {MIN_OBSERVATIONS} 日的最低要求")

    total = len(products)
    # Correlation needs a window every product covers, so the latest-listed
    # product silently decides the sample. Name it instead of hiding it.
    first_dates = nav_frame.groupby("ts_code")["date"].min()
    window_start = wide.index[0]
    sample_binding = sorted(
        (
            {
                "code": product.code,
                "name": product.name,
                "first_date": str(first_dates[product.column].date()),
            }
            for product in products
            # Compare against the requested start, not the resulting window: the common
        # window opens one observation after the latest first NAV, so a product
        # that defines it never satisfies `first_date >= window_start`.
        if product.column in first_dates.index and first_dates[product.column] > start
        ),
        key=lambda item: item["first_date"],
        reverse=True,
    )
    clean_returns, clipped_counts, clipped_worst = winsorize_returns_kernel(returns, WINSOR_SIGMA)
    correlation, distance, features = _build_feature_space(
        data_dir, spec, products, np.ascontiguousarray(clean_returns)
    )

    preset_names: Optional[list[str]] = None
    k_suggestions: list[dict[str, Any]] = []
    block_report: list[dict[str, Any]] = []
    cluster_block: Optional[np.ndarray] = None
    block_ids: Optional[np.ndarray] = None
    if spec.algorithm == "rule":
        raw_labels, preset_names = _taxonomy_groups(products, spec.taxonomy_level)
        clusters = len(preset_names)
    elif spec.block_by != "none":
        block_ids, block_names = _taxonomy_groups(products, spec.block_by)
        raw_labels, cluster_block, clusters, block_report = _blocked_cluster(
            spec, distance, features, block_ids, block_names, spec.k
        )
        if clusters < 1:
            raise AutoClassError("分层后没有可用的大类，请放宽分层层级")
    else:
        upper_bound = max(2, min(MAX_AUTO_K, total // max(1, spec.size_min)))
        linkage = _linkage(spec, distance) if spec.algorithm == "hierarchical" else None
        if spec.k is None:
            k_suggestions = _suggest_k(spec, distance, features, upper_bound, linkage)
            best = max(
                (item for item in k_suggestions if item["silhouette"] is not None),
                key=lambda item: item["silhouette"],
                default=None,
            )
            clusters = int(best["k"]) if best else 2
        else:
            clusters = int(spec.k)
        if clusters < 2:
            raise AutoClassError("大类个数至少为 2")
        if clusters > total:
            raise AutoClassError(f"大类个数 {clusters} 超过可用产品数 {total}")
        if clusters * spec.size_min > total:
            raise AutoClassError(
                f"{clusters} 个大类 × 每类最少 {spec.size_min} 个 = {clusters * spec.size_min}，超过可用产品数 {total}"
            )
        raw_labels = _cluster(spec, distance, features, clusters, linkage)

    affinity = affinity_from_distance_kernel(
        np.ascontiguousarray(distance), np.ascontiguousarray(raw_labels), int(clusters)
    )
    if block_ids is not None and cluster_block is not None and cluster_block.size:
        # capacity_assign_kernel skips non-finite affinities, so masking the
        # cross-block pairs is what stops the size_min backfill and the "force"
        # policy from quietly moving a bond ETF into an equity class.
        affinity = np.ascontiguousarray(
            np.where(block_ids[:, None] == cluster_block[None, :], affinity, -np.inf)
        )
    # A parked product may only backfill a short class when it is closer than an
    # average pool pair; otherwise the class stays short and says so.
    similarity_floor = -float(mean_offdiagonal_kernel(np.ascontiguousarray(distance)))
    labels = capacity_assign_kernel(
        np.ascontiguousarray(affinity),
        np.ascontiguousarray(raw_labels),
        int(spec.size_min),
        int(spec.size_max),
        0 if spec.unassigned_policy == "force" else 1,
        similarity_floor,
    )
    weights = intra_class_weights_kernel(
        np.ascontiguousarray(clean_returns),
        np.ascontiguousarray(labels),
        np.ascontiguousarray(affinity),
        int(clusters),
        int(WEIGHT_MODES[spec.weight_mode]),
    )

    caps = np.ascontiguousarray(
        np.array(
            [
                100.0 if product.max_weight is None else product.max_weight * 100.0
                for product in products
            ],
            dtype=np.float64,
        )
    )
    raw_weights = weights
    weights, class_capacity = apply_weight_caps_kernel(
        np.ascontiguousarray(weights), caps, np.ascontiguousarray(labels), int(clusters)
    )

    silhouette, per_cluster = silhouette_kernel(
        np.ascontiguousarray(distance), np.ascontiguousarray(labels), int(clusters)
    )
    mean_corr = intra_class_mean_corr_kernel(
        np.ascontiguousarray(correlation), np.ascontiguousarray(labels), int(clusters)
    )
    cross_corr = cross_class_corr_kernel(
        np.ascontiguousarray(correlation), np.ascontiguousarray(labels), int(clusters)
    )
    medoid_index: dict[int, int] = {}
    for cluster in range(clusters):
        members = [index for index in range(total) if labels[index] == cluster]
        if not members:
            continue
        medoid_index[cluster] = min(
            members, key=lambda index: sum(distance[index, other] for other in members)
        )
    medoids = {cluster: products[index].code for cluster, index in medoid_index.items()}
    class_names = _class_names(
        products, labels, clusters, medoid_index, preset_names, spec.taxonomy_level
    )

    classes: list[dict[str, Any]] = []
    deviations: list[dict[str, str]] = []
    for cluster in range(clusters):
        members = [index for index in range(total) if labels[index] == cluster]
        if not members:
            continue
        name = class_names[cluster]
        entries = []
        for index in members:
            product = products[index]
            taxonomy = product_taxonomy(product)
            contract = taxonomy.at(spec.taxonomy_level)
            entries.append({
                "code": product.code,
                "name": product.name,
                "weight": _finite(weights[index]) or 0.0,
                "instrument_type": product.instrument_type,
                "fund_type": product.fund_type,
                "invest_type": product.invest_type,
                "management": product.management,
                "contract_label": contract,
                "taxonomy": {
                    "asset_class": taxonomy.asset_class,
                    "category": taxonomy.category,
                    "detail": taxonomy.detail,
                    "path": taxonomy.path,
                    "matched": taxonomy.matched,
                },
                "affinity": _finite(affinity[index, cluster]),
                "is_medoid": product.code == medoids.get(cluster),
                "max_weight": product.max_weight,
                "capped": bool(
                    product.max_weight is not None
                    and raw_weights[index] > weights[index] + 1e-9
                ),
            })
            if preset_names is None and contract not in name:
                deviations.append({
                    "code": product.code,
                    "name": product.name,
                    "assigned_class": name,
                    "contract_label": contract,
                })
        entries.sort(key=lambda item: item["weight"], reverse=True)
        capacity = _finite(class_capacity[cluster]) if cluster < class_capacity.shape[0] else None
        classes.append({
            "id": f"auto-{cluster}",
            "name": name,
            "size": len(members),
            # Largest SAA weight for this class that still keeps every member
            # inside its product-pool restriction.
            "max_class_weight": capacity,
            "medoid": medoids.get(cluster, ""),
            "silhouette": _finite(per_cluster[cluster]) if cluster < per_cluster.shape[0] else None,
            "mean_corr": _finite(mean_corr[cluster]) if cluster < mean_corr.shape[0] else None,
            "etfs": entries,
        })

    unassigned = [
        {
            "code": products[index].code,
            "name": products[index].name,
            "reason": "CAPACITY",
            "detail": "所有候选大类已达上限或亲和度不足",
        }
        for index in range(total)
        if labels[index] < 0
    ]

    warnings: list[str] = list(pit_lineage.get("warnings") or [])
    warnings.extend(taxonomy_lineage.get("warnings") or [])
    if context.as_of and taxonomy_lineage["coverage"] == LATEST_ONLY:
        # Named, not blocked: with no dimension history on disk yet, blocking
        # would stop every historical classification in the platform on day one.
        uses = "类名" if spec.block_by == "none" and spec.algorithm != "rule" else "分类结果"
        warnings.append(
            f"合同分类信息只有最新态（缺维表历史快照），研究日 {context.as_of} 的"
            f"{uses}用的是今天的分类口径"
        )
    if context.as_of and spec.features in {"metrics", "blend"}:
        warnings.append(
            f"「{FEATURE_SETS[spec.features]}」取自当期指标快照，该快照没有历史版本；"
            f"研究日 {context.as_of} 的聚类实际用的是今天的收益与回撤"
        )
    if pit_lineage.get("as_of_applied") and pit_lineage.get("rows_dropped_by_as_of"):
        warnings.append(
            f"按研究日 {pit_lineage['as_of']} 的公告时点截断，剔除 "
            f"{pit_lineage['rows_dropped_by_as_of']} 行当时尚未公告的净值"
        )
    if not pit_lineage.get("as_of_applied"):
        warnings.append("未指定研究日，使用了磁盘上的全部净值；该结果不具备时点可复现性")
    if spec.algorithm == "rule" and spec.k is not None and spec.k != clusters:
        warnings.append(f"规则映射按合同分类自然产生 {clusters} 个大类，已忽略指定的 {spec.k} 类")
    if spec.algorithm == "rule" and spec.block_by != "none":
        warnings.append("规则映射本身就是合同分层，已忽略额外的分层设置")
    if block_report:
        warnings.append(
            f"已按「{BLOCK_MODES[spec.block_by]}」分层：{len(block_report)} 个合同块，"
            f"块内统计聚类共产生 {clusters} 个大类；不同合同块的产品不会被合并进同一大类"
        )
        if spec.k is not None and clusters != spec.k:
            warnings.append(
                f"每个合同块至少保留 1 个大类，分层后实际 {clusters} 类，与指定的 {spec.k} 类不一致"
            )
    if unassigned:
        warnings.append(f"{len(unassigned)} 个产品未归类，已进入待观察池")
    if skipped:
        warnings.append(f"{len(skipped)} 个产品缺少可用净值或重复，已排除在分类之外")
    if sample_binding and window_start > start:
        latest = sample_binding[0]
        warnings.append(
            f"共同样本从 {window_start.date()} 开始，晚于所选起始日；"
            f"最晚有净值的产品是【{latest['name']}】（{latest['first_date']}）"
        )
    restricted = [
        member
        for group in classes
        for member in group["etfs"]
        if member["max_weight"] is not None
    ]
    if restricted:
        capped = [member for member in restricted if member["capped"]]
        warnings.append(
            f"{len(restricted)} 个产品带有产品池限额"
            + (f"，其中 {len(capped)} 个的类内权重已被限额压低" if capped else "，当前类内权重均未触及限额")
        )
    for group in classes:
        capacity = group["max_class_weight"]
        if capacity is not None and capacity < 100.0 - 1e-9:
            warnings.append(
                f"大类【{group['name']}】成员限额合计仅 {capacity:.1f}%，"
                f"该大类的 SAA 权重不得超过 {capacity:.1f}%，否则会突破产品池限额"
            )
    flagged = int(np.sum(clipped_counts > 0))
    if flagged:
        warnings.append(f"{flagged} 个产品的复权净值存在疑似异常跳变，已在聚类特征中稳健处理（展示净值与指标不受影响）")
    empty_classes = clusters - len(classes)
    if empty_classes > 0:
        warnings.append(f"{empty_classes} 个大类在容量约束下为空，实际输出 {len(classes)} 类")
    for cluster in range(clusters):
        natural = int(np.sum(raw_labels == cluster))
        kept = int(np.sum(labels == cluster))
        if 0 < kept < spec.size_min:
            warnings.append(
                f"大类【{class_names[cluster]}】只有 {kept} 个足够相似的产品，未达到每类最少 {spec.size_min} 个；"
                f"平台不会为了凑数塞入不相关产品"
            )
        if natural > kept and kept > 0:
            name = class_names[cluster]
            warnings.append(
                f"大类【{name}】自然聚类有 {natural} 个候选，按每类上限保留 {kept} 个代表产品；"
                f"如需覆盖更多产品请调高每类上限或增加大类数"
            )

    # Empty clusters are dropped from `classes`, so the cross-class matrix has to
    # drop them too or its row/column labels stop matching what the UI shows.
    populated = [cluster for cluster in range(clusters) if int(np.sum(labels == cluster)) > 0]
    label_list = [class_names[cluster] for cluster in populated]
    cross_matrix = [
        [_finite(cross_corr[left, right]) for right in populated]
        for left in populated
    ]

    return {
        "algorithm": spec.algorithm,
        "features": spec.features,
        "taxonomy_level": spec.taxonomy_level,
        "as_of": spec.as_of,
        "run_mode": spec.run_mode,
        "data_release_id": spec.data_release_id,
        "pit": {**pit_lineage, "taxonomy": taxonomy_lineage},
        "block_by": spec.block_by,
        "k": clusters,
        "weight_mode": spec.weight_mode,
        "observations": observations,
        "start_date": str(wide.index[0].date()),
        "end_date": str(wide.index[-1].date()),
        "classes": classes,
        "unassigned": unassigned,
        "skipped": skipped,
        "warnings": warnings,
        "diagnostics": {
            "silhouette": _finite(silhouette),
            "cross_class_corr": cross_matrix,
            "cross_class_labels": label_list,
            "significant_eigenvalues": _significant_eigenvalues(correlation, observations),
            "k_suggestions": k_suggestions,
            "blocks": block_report,
            "contract_deviations": deviations,
            "sample_binding": sample_binding,
            "winsorized": [
                {
                    "code": products[index].code,
                    "name": products[index].name,
                    "clipped": int(clipped_counts[index]),
                    "max_raw_return": _finite(clipped_worst[index]),
                }
                for index in range(total)
                if int(clipped_counts[index]) > 0
            ],
        },
        "execution": auto_class_execution_audit(),
    }


def auto_classification_meta() -> dict[str, Any]:
    return {
        "algorithms": [{"id": key, "label": value} for key, value in ALGORITHMS.items()],
        "features": [{"id": key, "label": value} for key, value in FEATURE_SETS.items()],
        "linkages": [{"id": key, "label": key} for key in LINKAGE_METHODS],
        "taxonomy_levels": [
            {"id": level, "label": TAXONOMY_LEVEL_LABELS[level]} for level in TAXONOMY_LEVELS
        ],
        "block_modes": [{"id": key, "label": value} for key, value in BLOCK_MODES.items()],
        "taxonomy": taxonomy_tree(),
        "weight_modes": [
            {"id": "equal", "label": "等权"},
            {"id": "inv_vol", "label": "逆波动率"},
            {"id": "inv_var", "label": "逆方差"},
            {"id": "affinity", "label": "按亲和度"},
        ],
        "limits": {
            "min_observations": MIN_OBSERVATIONS,
            "max_auto_k": MAX_AUTO_K,
            "min_products": 2,
        },
        "defaults": {
            "algorithm": "hierarchical",
            "features": "correlation",
            "linkage": "average",
            "weight_mode": "inv_vol",
            "size_min": 2,
            "size_max": 8,
            "unassigned_policy": "park",
            "start_date": "2020-01-01",
            "taxonomy_level": "asset_class",
            "block_by": "none",
        },
    }


__all__ = [
    "ALGORITHMS",
    "BLOCK_MODES",
    "AutoClassError",
    "AutoClassRequestSpec",
    "FEATURE_SETS",
    "auto_classification_meta",
    "product_taxonomy",
    "rule_label",
    "run_auto_classification",
]
