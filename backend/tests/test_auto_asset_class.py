"""Auto asset-class construction: NJIT kernel contracts and orchestration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, get_args

import numpy as np
import pandas as pd
import pytest
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend import auto_class_numba as kernels
from backend import auto_asset_class as service
from backend import fund_taxonomy as taxonomy
from backend.services import auto_class_routes as routes
from product_pools.constants import UNIVERSE_SNAPSHOT_STORE
from product_pools.repository import InvestableUniverseRepository


@pytest.fixture(scope="module", autouse=True)
def _warm_kernels() -> None:
    kernels.warm_auto_class_numba_kernels()


def _c(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64))


def _grouped_returns(periods: int = 400, groups: int = 3, per_group: int = 4, seed: int = 0) -> np.ndarray:
    """Factor-driven returns whose true grouping is known by construction."""

    rng = np.random.default_rng(seed)
    factors = rng.normal(0.0, 0.01, (periods, groups))
    columns = [factors[:, group] + rng.normal(0.0, 0.002, periods) for group in range(groups) for _ in range(per_group)]
    return _c(np.array(columns).T)


def _write_universe(
    data_dir: Path,
    codes: list[str],
    limits: Dict[str, float] | None = None,
) -> dict[str, Any]:
    return InvestableUniverseRepository(data_dir / UNIVERSE_SNAPSHOT_STORE).create(
        {
            "name": "自动分类测试域",
            "research_date": "2026-09-04",
            "version_refs": [{"pool_id": "pool-1", "version_id": "version-1"}],
            "members": [
                {
                    "kind": "etf",
                    "product_id": code,
                    "name": code,
                    "eligible": True,
                    "eligibility_reasons": [],
                    "max_weight": (limits or {}).get(code),
                }
                for code in codes
            ],
            "summary": {
                "pool_count": 1,
                "member_count": len(codes),
                "eligible_count": len(codes),
                "restricted_count": 0,
                "watch_count": 0,
            },
            "content_hash": "universe-hash",
        }
    )


def _json(resp: JSONResponse) -> Dict[str, Any]:
    return json.loads(resp.body.decode("utf-8"))


# --------------------------------------------------------------------------
# Execution policy
# --------------------------------------------------------------------------

def test_execution_audit_is_fixed_signature_njit() -> None:
    audit = kernels.auto_class_execution_audit()
    assert audit["execution_backend"] == "numba_njit_fixed_signature"
    assert audit["nopython"] is True
    assert audit["python_fallback"] == 0
    assert audit["object_mode"] == 0
    assert audit["request_time_compilation"] == 0
    assert audit["kernel_signatures"]
    for signatures in audit["kernel_signatures"].values():
        assert signatures and len(set(signatures)) == len(signatures)


def test_every_public_kernel_refuses_new_signatures() -> None:
    for kernel in kernels._PUBLIC_KERNELS:
        assert kernel._can_compile is False, kernel.py_func.__name__


# --------------------------------------------------------------------------
# Numerical kernels vs controlled references
# --------------------------------------------------------------------------

def test_correlation_matches_numpy_reference() -> None:
    returns = _grouped_returns(periods=250, groups=2, per_group=3, seed=11)
    produced = kernels.correlation_matrix_kernel(returns)
    expected = np.corrcoef(returns, rowvar=False)
    assert np.allclose(produced, expected, atol=1e-10)


def test_correlation_handles_constant_and_nonfinite_columns() -> None:
    returns = _c(np.array([[0.01, 1.0], [-0.02, 1.0], [0.03, 1.0], [0.00, 1.0]]))
    produced = kernels.correlation_matrix_kernel(returns)
    assert np.all(np.isfinite(produced))
    assert produced[0, 0] == pytest.approx(1.0)
    # A zero-variance column cannot correlate with anything; it must not be NaN.
    assert produced[0, 1] == pytest.approx(0.0)


def test_corr_distance_is_a_metric_on_the_unit_range() -> None:
    correlation = _c(np.array([[1.0, 0.5, -1.0], [0.5, 1.0, 0.0], [-1.0, 0.0, 1.0]]))
    distance = kernels.corr_to_distance_kernel(correlation)
    assert np.allclose(np.diag(distance), 0.0)
    assert np.allclose(distance, distance.T)
    assert distance[0, 2] == pytest.approx(1.0)
    assert distance[0, 1] == pytest.approx(np.sqrt(0.25))
    assert distance.min() >= 0.0 and distance.max() <= 1.0


def test_euclidean_distance_matches_reference() -> None:
    features = _c(np.array([[0.0, 0.0], [3.0, 4.0], [-1.0, 1.0]]))
    produced = kernels.euclidean_distance_kernel(features)
    expected = np.sqrt(((features[:, None, :] - features[None, :, :]) ** 2).sum(axis=2))
    assert np.allclose(produced, expected)


def test_robust_standardize_centers_on_median_and_survives_degenerate_columns() -> None:
    features = _c(np.array([[1.0, 5.0, np.nan], [2.0, 5.0, 1.0], [3.0, 5.0, np.inf], [100.0, 5.0, 2.0]]))
    produced = kernels.robust_standardize_kernel(features)
    assert np.all(np.isfinite(produced))
    # Median of column 0 is 2.5, so the two central rows straddle zero.
    assert produced[1, 0] < 0.0 < produced[2, 0]
    # A constant column carries no information and must collapse to zero.
    assert np.allclose(produced[:, 1], 0.0)
    # Winsorisation caps the outlier instead of letting it dominate the distance.
    assert produced[3, 0] == pytest.approx(5.0)


def test_winsorize_clips_only_data_errors_and_reports_them() -> None:
    rng = np.random.default_rng(3)
    clean = rng.normal(0.0, 0.001, (300, 2))
    clean[10, 0] = 0.99  # unadjusted split print
    returns = _c(clean)
    produced, clipped, worst = kernels.winsorize_returns_kernel(returns, 25.0)
    assert clipped[0] == 1
    assert clipped[1] == 0
    assert worst[0] == pytest.approx(0.99)
    assert produced[10, 0] < 0.05
    # Everything else is untouched.
    untouched = np.delete(np.arange(300), 10)
    assert np.allclose(produced[untouched, 0], returns[untouched, 0])


def test_winsorize_handles_empty_and_constant_input() -> None:
    empty, counts, worst = kernels.winsorize_returns_kernel(_c(np.zeros((0, 2))), 25.0)
    assert empty.shape == (0, 2) and counts.tolist() == [0, 0] and worst.tolist() == [0.0, 0.0]
    constant, counts, _ = kernels.winsorize_returns_kernel(_c(np.full((5, 1), 0.01)), 25.0)
    assert np.allclose(constant, 0.01)
    assert counts[0] == 0


def test_pca_features_reproduce_the_leading_eigenvectors() -> None:
    returns = _grouped_returns(periods=300, groups=2, per_group=3, seed=5)
    correlation = kernels.correlation_matrix_kernel(returns)
    loadings = kernels.pca_features_kernel(_c(correlation), 2)
    values, vectors = np.linalg.eigh(correlation)
    expected = vectors[:, -1] * np.sqrt(values[-1])
    # Eigenvector sign is arbitrary; compare up to a global flip.
    assert np.allclose(loadings[:, 0], expected) or np.allclose(loadings[:, 0], -expected)


def test_eigenvalues_are_descending_and_sum_to_the_trace() -> None:
    returns = _grouped_returns(periods=300, groups=3, per_group=2, seed=7)
    correlation = kernels.correlation_matrix_kernel(returns)
    values = kernels.correlation_eigenvalues_kernel(_c(correlation))
    assert np.all(np.diff(values) <= 1e-9)
    assert values.sum() == pytest.approx(correlation.shape[0])


# --------------------------------------------------------------------------
# Clustering recovers a known structure
# --------------------------------------------------------------------------

def _grouping_is_exact(labels: np.ndarray, groups: int, per_group: int) -> bool:
    for group in range(groups):
        block = labels[group * per_group:(group + 1) * per_group]
        if len(set(block.tolist())) != 1:
            return False
    return len(set(labels.tolist())) == groups


@pytest.mark.parametrize("method", [kernels.LINKAGE_AVERAGE, kernels.LINKAGE_COMPLETE, kernels.LINKAGE_WARD])
def test_linkage_recovers_planted_groups(method: int) -> None:
    returns = _grouped_returns()
    distance = kernels.corr_to_distance_kernel(_c(kernels.correlation_matrix_kernel(returns)))
    linkage = kernels.agglomerative_linkage_kernel(_c(distance), method)
    assert linkage.shape == (11, 4)
    assert np.all(np.diff(linkage[:, 2]) >= -1e-12), "merge distances must be monotone"
    assert linkage[-1, 3] == 12
    labels = kernels.cut_linkage_kernel(_c(linkage), 12, 3)
    assert _grouping_is_exact(labels, 3, 4)


def test_kmedoids_and_kmeans_recover_planted_groups() -> None:
    returns = _grouped_returns()
    correlation = kernels.correlation_matrix_kernel(returns)
    distance = kernels.corr_to_distance_kernel(_c(correlation))
    medoid_labels, medoids = kernels.kmedoids_kernel(_c(distance), 3, 7, 100)
    assert _grouping_is_exact(medoid_labels, 3, 4)
    assert len(set(medoids.tolist())) == 3
    features = kernels.robust_standardize_kernel(_c(correlation))
    assert _grouping_is_exact(kernels.kmeans_kernel(features, 3, 7, 200), 3, 4)


def test_spectral_and_gmm_recover_planted_groups() -> None:
    returns = _grouped_returns()
    correlation = kernels.correlation_matrix_kernel(returns)
    distance = kernels.corr_to_distance_kernel(_c(correlation))
    features = kernels.robust_standardize_kernel(_c(correlation))

    spectral = kernels.spectral_labels_kernel(_c(distance), 3, 7, 200)
    assert _grouping_is_exact(spectral, 3, 4)
    assert spectral.tolist() == kernels.spectral_labels_kernel(_c(distance), 3, 7, 200).tolist()

    labels, posterior = kernels.gmm_kernel(features, 3, 7, 200)
    assert _grouping_is_exact(labels, 3, 4)
    assert posterior.shape == (12, 3)
    assert np.allclose(posterior.sum(axis=1), 1.0)
    assert np.all(posterior >= 0.0)
    # argmax of the posterior is what becomes the label.
    assert posterior.argmax(axis=1).tolist() == labels.tolist()
    assert labels.tolist() == kernels.gmm_kernel(features, 3, 7, 200)[0].tolist()


def test_spectral_and_gmm_degrade_safely_on_tiny_pools() -> None:
    returns = _grouped_returns(periods=120, groups=2, per_group=1, seed=5)
    correlation = kernels.correlation_matrix_kernel(returns)
    distance = kernels.corr_to_distance_kernel(_c(correlation))
    features = kernels.robust_standardize_kernel(_c(correlation))
    # Two products cannot support a graph cut; one class is the honest answer.
    assert set(kernels.spectral_labels_kernel(_c(distance), 3, 7, 50).tolist()) == {0}
    single, posterior = kernels.gmm_kernel(features, 1, 7, 50)
    assert set(single.tolist()) == {0}
    assert np.allclose(posterior, 1.0)


def test_denoise_correlation_flattens_the_noise_spectrum() -> None:
    returns = _grouped_returns()
    correlation = _c(kernels.correlation_matrix_kernel(returns))
    denoised = kernels.denoise_correlation_kernel(correlation, returns.shape[0])

    assert np.allclose(np.diag(denoised), 1.0)
    assert np.allclose(denoised, denoised.T)
    assert np.trace(denoised) == pytest.approx(np.trace(correlation))

    raw_eigenvalues = np.linalg.eigvalsh(correlation)
    denoised_eigenvalues = np.linalg.eigvalsh(denoised)
    # Three planted factors survive; the other nine collapse to one level.
    assert denoised_eigenvalues[-3:] == pytest.approx(raw_eigenvalues[-3:], abs=0.05)
    noise = denoised_eigenvalues[:-3]
    assert noise.max() - noise.min() < 0.02
    assert noise.max() - noise.min() < (raw_eigenvalues[:-3].max() - raw_eigenvalues[:-3].min())
    # Clustering the denoised distance still finds the planted groups.
    distance = kernels.corr_to_distance_kernel(_c(denoised))
    linkage = kernels.agglomerative_linkage_kernel(_c(distance), kernels.LINKAGE_AVERAGE)
    assert _grouping_is_exact(kernels.cut_linkage_kernel(_c(linkage), 12, 3), 3, 4)


def test_denoise_correlation_leaves_rank_deficient_samples_alone() -> None:
    returns = _grouped_returns(periods=400)
    correlation = _c(kernels.correlation_matrix_kernel(returns))
    # T <= N: no signal/noise split is identifiable, so nothing may be clipped.
    assert np.allclose(kernels.denoise_correlation_kernel(correlation, 12), correlation)
    assert np.allclose(kernels.denoise_correlation_kernel(correlation, 4), correlation)


def test_clustering_is_deterministic_across_repeated_runs() -> None:
    returns = _grouped_returns(seed=21)
    distance = kernels.corr_to_distance_kernel(_c(kernels.correlation_matrix_kernel(returns)))
    first, _ = kernels.kmedoids_kernel(_c(distance), 3, 42, 100)
    second, _ = kernels.kmedoids_kernel(_c(distance), 3, 42, 100)
    assert first.tolist() == second.tolist()


def test_cut_linkage_clamps_out_of_range_cluster_counts() -> None:
    returns = _grouped_returns(periods=200, groups=2, per_group=2, seed=9)
    distance = kernels.corr_to_distance_kernel(_c(kernels.correlation_matrix_kernel(returns)))
    linkage = kernels.agglomerative_linkage_kernel(_c(distance), kernels.LINKAGE_AVERAGE)
    assert len(set(kernels.cut_linkage_kernel(_c(linkage), 4, 0).tolist())) == 1
    assert len(set(kernels.cut_linkage_kernel(_c(linkage), 4, 99).tolist())) == 4


def test_single_observation_pool_produces_an_empty_linkage() -> None:
    linkage = kernels.agglomerative_linkage_kernel(_c(np.zeros((1, 1))), kernels.LINKAGE_AVERAGE)
    assert linkage.shape == (0, 4)
    assert kernels.cut_linkage_kernel(_c(linkage), 1, 1).tolist() == [0]


# --------------------------------------------------------------------------
# Capacity-constrained selection
# --------------------------------------------------------------------------

def _affinity(distance: np.ndarray, labels: np.ndarray, clusters: int) -> np.ndarray:
    return _c(kernels.affinity_from_distance_kernel(_c(distance), np.ascontiguousarray(labels), clusters))


def test_capacity_selection_keeps_the_most_representative_members() -> None:
    returns = _grouped_returns()
    distance = kernels.corr_to_distance_kernel(_c(kernels.correlation_matrix_kernel(returns)))
    labels = kernels.cut_linkage_kernel(
        _c(kernels.agglomerative_linkage_kernel(_c(distance), kernels.LINKAGE_AVERAGE)), 12, 3
    )
    affinity = _affinity(distance, labels, 3)
    selected = kernels.capacity_assign_kernel(affinity, np.ascontiguousarray(labels), 1, 2, 1, -np.inf)
    counts = np.bincount(selected[selected >= 0], minlength=3)
    assert counts.tolist() == [2, 2, 2]
    assert int(np.sum(selected < 0)) == 6
    # Selection never moves a product out of the cluster it was clustered into.
    for index in range(12):
        if selected[index] >= 0:
            assert selected[index] == labels[index]


def test_capacity_selection_respects_the_similarity_floor() -> None:
    # Cluster 0 has two members, cluster 1 only one; the leftover product in
    # cluster 0 is far from cluster 1 and must not be dragged in to fill it.
    distance = _c(np.array([
        [0.0, 0.05, 0.06, 0.90],
        [0.05, 0.0, 0.07, 0.92],
        [0.06, 0.07, 0.0, 0.95],
        [0.90, 0.92, 0.95, 0.0],
    ]))
    labels = np.ascontiguousarray(np.array([0, 0, 0, 1], dtype=np.int64))
    affinity = _affinity(distance, labels, 2)
    floor = -float(kernels.mean_offdiagonal_kernel(distance))
    strict = kernels.capacity_assign_kernel(affinity, labels, 2, 2, 1, floor)
    assert int(np.sum(strict == 1)) == 1, "an unrelated product must not backfill a short class"
    loose = kernels.capacity_assign_kernel(affinity, labels, 2, 2, 1, -np.inf)
    assert int(np.sum(loose == 1)) == 2, "without a floor the same call does fill the class"


def test_force_policy_assigns_every_product() -> None:
    returns = _grouped_returns()
    distance = kernels.corr_to_distance_kernel(_c(kernels.correlation_matrix_kernel(returns)))
    labels = kernels.cut_linkage_kernel(
        _c(kernels.agglomerative_linkage_kernel(_c(distance), kernels.LINKAGE_AVERAGE)), 12, 3
    )
    affinity = _affinity(distance, labels, 3)
    forced = kernels.capacity_assign_kernel(affinity, np.ascontiguousarray(labels), 1, 2, 0, -np.inf)
    assert int(np.sum(forced < 0)) == 0


# --------------------------------------------------------------------------
# Product-pool weight restrictions
# --------------------------------------------------------------------------

def _caps_case(weights, caps, labels, clusters=2):
    return kernels.apply_weight_caps_kernel(
        _c(np.asarray(weights)).copy(),
        _c(np.asarray(caps)).copy(),
        np.ascontiguousarray(np.asarray(labels, dtype=np.int64)),
        clusters,
    )


def test_weight_caps_water_fill_respects_every_limit() -> None:
    weights = [70.0, 20.0, 10.0, 90.0, 10.0]
    caps = [30.0, 100.0, 100.0, 60.0, 100.0]
    labels = [0, 0, 0, 1, 1]
    output, capacity = _caps_case(weights, caps, labels)
    assert np.all(output <= np.asarray(caps) + 1e-9)
    assert output[:3].sum() == pytest.approx(100.0)
    assert output[3:].sum() == pytest.approx(100.0)
    # The freed weight is redistributed in proportion to the original scores.
    assert output[0] == pytest.approx(30.0)
    assert output[1] / output[2] == pytest.approx(2.0)
    assert capacity.tolist() == pytest.approx([100.0, 100.0])


def test_weight_caps_report_capacity_when_a_class_cannot_be_filled() -> None:
    # 20 + 30 + 10 = 60 < 100: the class simply cannot hold a full allocation.
    output, capacity = _caps_case(
        [70.0, 20.0, 10.0, 90.0, 10.0], [20.0, 30.0, 10.0, 100.0, 100.0], [0, 0, 0, 1, 1]
    )
    assert capacity[0] == pytest.approx(60.0)
    assert output[:3].sum() == pytest.approx(100.0)
    # Weights proportional to the caps mean the limits bind exactly at the
    # reported class capacity and stay satisfied below it.
    assert (output[:3] * 0.60).tolist() == pytest.approx([20.0, 30.0, 10.0])
    assert np.all(output[:3] * 0.50 <= np.array([20.0, 30.0, 10.0]) + 1e-9)


def test_weight_caps_leave_unrestricted_classes_untouched() -> None:
    weights = [70.0, 20.0, 10.0, 90.0, 10.0]
    output, capacity = _caps_case(weights, [100.0] * 5, [0, 0, 0, 1, 1])
    assert output.tolist() == pytest.approx(weights)
    assert capacity.tolist() == pytest.approx([100.0, 100.0])


def test_weight_caps_survive_degenerate_inputs() -> None:
    zeroed, capacity = _caps_case([70.0, 30.0], [0.0, 0.0], [0, 0], clusters=1)
    assert zeroed.tolist() == pytest.approx([0.0, 0.0])
    assert capacity[0] == pytest.approx(0.0)
    # Non-finite caps are treated as "unrestricted" rather than poisoning output.
    messy, _ = _caps_case([70.0, 30.0], [np.nan, np.inf], [0, 0], clusters=1)
    assert np.all(np.isfinite(messy))
    assert messy.sum() == pytest.approx(100.0)
    empty, capacity = _caps_case([], [], [], clusters=2)
    assert empty.shape == (0,) and np.all(np.isnan(capacity))


def test_weight_caps_ignore_unassigned_products() -> None:
    output, capacity = _caps_case([50.0, 50.0, 99.0], [100.0, 100.0, 1.0], [0, 0, -1], clusters=1)
    assert output[:2].sum() == pytest.approx(100.0)
    assert output[2] == pytest.approx(99.0), "parked products keep their (unused) score"
    assert capacity[0] == pytest.approx(100.0)


def test_mean_offdiagonal_ignores_the_diagonal() -> None:
    distance = _c(np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert kernels.mean_offdiagonal_kernel(distance) == pytest.approx(1.0)
    assert np.isnan(kernels.mean_offdiagonal_kernel(_c(np.zeros((1, 1)))))


# --------------------------------------------------------------------------
# Diagnostics and weights
# --------------------------------------------------------------------------

def test_silhouette_rewards_the_true_grouping() -> None:
    returns = _grouped_returns()
    distance = _c(kernels.corr_to_distance_kernel(_c(kernels.correlation_matrix_kernel(returns))))
    truth = np.ascontiguousarray(np.repeat(np.arange(3), 4).astype(np.int64))
    shuffled = np.ascontiguousarray(np.tile(np.arange(3), 4).astype(np.int64))
    good, per_cluster = kernels.silhouette_kernel(distance, truth, 3)
    bad, _ = kernels.silhouette_kernel(distance, shuffled, 3)
    assert good > 0.5 > bad
    assert per_cluster.shape == (3,) and np.all(np.isfinite(per_cluster))


def test_silhouette_skips_unassigned_rows_and_undefined_cases() -> None:
    distance = _c(np.array([[0.0, 0.2, 0.9], [0.2, 0.0, 0.9], [0.9, 0.9, 0.0]]))
    labels = np.ascontiguousarray(np.array([0, 0, -1], dtype=np.int64))
    score, _ = kernels.silhouette_kernel(distance, labels, 1)
    # Only one populated cluster and one parked product: nothing is scorable.
    assert np.isnan(score)


def test_silhouette_scores_singletons_as_zero_not_as_perfect() -> None:
    # Three tight products; splitting them 2 + 1 must not beat keeping the
    # tight pair together, otherwise the automatic K suggestion over-splits.
    distance = _c(np.array([
        [0.0, 0.05, 0.06, 0.90],
        [0.05, 0.0, 0.07, 0.92],
        [0.06, 0.07, 0.0, 0.95],
        [0.90, 0.92, 0.95, 0.0],
    ]))
    grouped = np.ascontiguousarray(np.array([0, 0, 0, 1], dtype=np.int64))
    split = np.ascontiguousarray(np.array([0, 0, 2, 1], dtype=np.int64))
    grouped_score, _ = kernels.silhouette_kernel(distance, grouped, 2)
    split_score, per_cluster = kernels.silhouette_kernel(distance, split, 3)
    assert grouped_score > split_score
    assert per_cluster[2] == pytest.approx(0.0)


def test_intra_and_cross_class_correlation_summaries() -> None:
    correlation = _c(np.array([
        [1.0, 0.8, 0.1],
        [0.8, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ]))
    labels = np.ascontiguousarray(np.array([0, 0, 1], dtype=np.int64))
    intra = kernels.intra_class_mean_corr_kernel(correlation, labels, 2)
    assert intra[0] == pytest.approx(0.8)
    assert np.isnan(intra[1]), "a singleton class has no internal pair"
    cross = kernels.cross_class_corr_kernel(correlation, labels, 2)
    assert cross[0, 1] == pytest.approx((0.1 + 0.2) / 2)
    assert cross[0, 0] == pytest.approx(0.8)


@pytest.mark.parametrize(
    "mode", [kernels.WEIGHT_EQUAL, kernels.WEIGHT_INV_VOL, kernels.WEIGHT_INV_VAR, kernels.WEIGHT_AFFINITY]
)
def test_intra_class_weights_normalise_to_one_hundred_per_class(mode: int) -> None:
    returns = _grouped_returns()
    distance = kernels.corr_to_distance_kernel(_c(kernels.correlation_matrix_kernel(returns)))
    labels = np.ascontiguousarray(np.repeat(np.arange(3), 4).astype(np.int64))
    affinity = _affinity(distance, labels, 3)
    weights = kernels.intra_class_weights_kernel(returns, labels, affinity, 3, mode)
    assert np.all(weights >= 0.0)
    for cluster in range(3):
        assert weights[labels == cluster].sum() == pytest.approx(100.0)


def test_inverse_volatility_favours_the_calmer_product() -> None:
    rng = np.random.default_rng(4)
    returns = _c(np.column_stack([rng.normal(0, 0.001, 300), rng.normal(0, 0.02, 300)]))
    labels = np.ascontiguousarray(np.zeros(2, dtype=np.int64))
    affinity = _c(np.zeros((2, 1)))
    weights = kernels.intra_class_weights_kernel(returns, labels, affinity, 1, kernels.WEIGHT_INV_VOL)
    assert weights[0] > weights[1]
    assert weights.sum() == pytest.approx(100.0)


def test_weights_fall_back_to_equal_when_every_variance_is_degenerate() -> None:
    returns = _c(np.zeros((10, 2)))
    labels = np.ascontiguousarray(np.zeros(2, dtype=np.int64))
    weights = kernels.intra_class_weights_kernel(returns, labels, _c(np.zeros((2, 1))), 1, kernels.WEIGHT_INV_VOL)
    assert weights.tolist() == pytest.approx([50.0, 50.0])


def test_unassigned_products_receive_no_weight() -> None:
    returns = _grouped_returns(periods=200, groups=1, per_group=3, seed=13)
    labels = np.ascontiguousarray(np.array([0, 0, -1], dtype=np.int64))
    weights = kernels.intra_class_weights_kernel(returns, labels, _c(np.zeros((3, 1))), 1, kernels.WEIGHT_EQUAL)
    assert weights[2] == 0.0
    assert weights[:2].sum() == pytest.approx(100.0)


# --------------------------------------------------------------------------
# Contract rule table
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("fund_type", "invest_type", "name", "expected"),
    [
        ("货币市场型", "", "华宝现金添益", "货币类"),
        ("其他", "黄金现货合约", "华安易富黄金ETF", "商品类"),
        ("股票型", "被动指数型", "国泰纳斯达克100ETF(QDII)", "海外类"),
        ("债券型", "被动指数型", "国泰上证5年期国债ETF", "固收类"),
        ("股票型", "被动指数型", "华泰柏瑞沪深300ETF", "权益类"),
        ("混合型", "灵活配置型", "某混合基金", "混合类"),
        ("", "", "未知产品", "其他类"),
    ],
)
def test_rule_label_ordering(fund_type: str, invest_type: str, name: str, expected: str) -> None:
    product = service._Product(code="x", name=name, column="x", fund_type=fund_type, invest_type=invest_type)
    assert service.rule_label(product) == expected


# --------------------------------------------------------------------------
# Orchestration on a synthetic parquet fixture
# --------------------------------------------------------------------------

def _write_fixture(tmp_path: Path) -> list[str]:
    """Three equity-like, two bond-like and two gold-like synthetic ETFs."""

    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2021-01-01", periods=400)
    blocks = {"EQ": (3, 0.012), "BD": (2, 0.001), "AU": (2, 0.008)}
    rows: list[dict[str, Any]] = []
    codes: list[str] = []
    for prefix, (count, scale) in blocks.items():
        factor = rng.normal(0.0, scale, len(dates))
        for index in range(count):
            code = f"{prefix}{index}.SH"
            codes.append(code)
            returns = factor + rng.normal(0.0, scale / 8.0, len(dates))
            nav = np.cumprod(1.0 + returns)
            rows.extend(
                {"ts_code": code, "name": f"{prefix} Fund {index}", "date": date, "adj_nav": value}
                for date, value in zip(dates, nav)
            )
    pd.DataFrame(rows).to_parquet(tmp_path / "etf_daily_df.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ts_code": code,
                "code": code.split(".")[0],
                "name": f"{code[:2]} Fund",
                "instrument_type": "etf",
                "fund_type": {"EQ": "股票型", "BD": "债券型", "AU": "其他"}[code[:2]],
                "invest_type": {"EQ": "被动指数型", "BD": "被动指数型", "AU": "黄金现货合约"}[code[:2]],
                "benchmark": "",
                "index_name": "",
                "management": "Test AMC",
            }
            for code in codes
        ]
    ).to_parquet(tmp_path / "etf_info_df.parquet", index=False)
    profile = {
        "EQ": {"return_1y": 0.15, "annual_volatility_1y": 0.20, "max_drawdown_3y": -0.30, "sharpe_1y": 0.7},
        "BD": {"return_1y": 0.03, "annual_volatility_1y": 0.02, "max_drawdown_3y": -0.02, "sharpe_1y": 1.4},
        "AU": {"return_1y": 0.10, "annual_volatility_1y": 0.14, "max_drawdown_3y": -0.18, "sharpe_1y": 0.6},
    }
    pd.DataFrame(
        [
            {
                "ts_code": code,
                **profile[code[:2]],
                "return_3y": profile[code[:2]]["return_1y"] * 2.5,
                "calmar_3y": 0.4,
                "premium_discount_latest": 0.001,
                "amount_avg_20d": 1_000_000.0,
            }
            for code in codes
        ]
    ).to_parquet(tmp_path / "instrument_metrics_snapshot.parquet", index=False)
    return codes


def test_run_auto_classification_recovers_the_fixture_structure(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(codes=codes, start_date="2021-01-01", k=3, size_min=2, size_max=3),
    )
    assert result["k"] == 3
    assert len(result["classes"]) == 3
    assert result["execution"]["python_fallback"] == 0
    grouping = {
        member["code"][:2]
        for group in result["classes"]
        for member in group["etfs"]
    }
    assert grouping == {"EQ", "BD", "AU"}
    for group in result["classes"]:
        prefixes = {member["code"][:2] for member in group["etfs"]}
        assert len(prefixes) == 1, f"class {group['name']} mixed {prefixes}"
        assert sum(member["weight"] for member in group["etfs"]) == pytest.approx(100.0)
        assert any(member["is_medoid"] for member in group["etfs"])


def test_auto_k_reports_suggestions_and_picks_the_best_silhouette(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(codes=codes, start_date="2021-01-01", k=None, size_min=1, size_max=4),
    )
    suggestions = result["diagnostics"]["k_suggestions"]
    assert [item["k"] for item in suggestions] == list(range(2, len(suggestions) + 2))
    scorable = [item for item in suggestions if item["silhouette"] is not None]
    best = max(scorable, key=lambda item: item["silhouette"])
    assert result["k"] == best["k"] == 3
    # Splitting the pool all the way down to singletons must score no better
    # than zero, so the suggestion cannot run away to the largest K.
    assert suggestions[-1]["k"] == len(codes)
    assert suggestions[-1]["silhouette"] == pytest.approx(0.0)


def test_capacity_overflow_parks_products_and_warns(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(codes=codes, start_date="2021-01-01", k=3, size_min=1, size_max=1),
    )
    assert all(group["size"] == 1 for group in result["classes"])
    assert len(result["unassigned"]) == len(codes) - 3
    assert any("待观察池" in warning for warning in result["warnings"])


def test_force_policy_leaves_nothing_unassigned(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(
            codes=codes, start_date="2021-01-01", k=3, size_min=1, size_max=1, unassigned_policy="force"
        ),
    )
    assert result["unassigned"] == []
    assert sum(group["size"] for group in result["classes"]) == len(codes)


def test_rule_algorithm_uses_contract_labels(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(codes=codes, start_date="2021-01-01", algorithm="rule", size_min=1, size_max=5),
    )
    assert {group["name"] for group in result["classes"]} == {"权益类", "固收类", "商品类"}
    assert result["diagnostics"]["contract_deviations"] == []


@pytest.mark.parametrize("algorithm", ["spectral", "gmm"])
def test_spectral_and_gmm_run_end_to_end(tmp_path: Path, algorithm: str) -> None:
    codes = _write_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(
            codes=codes, start_date="2021-01-01", algorithm=algorithm, k=3, size_min=1, size_max=5
        ),
    )
    assert result["algorithm"] == algorithm
    assert len(result["classes"]) == 3
    assert sum(group["size"] for group in result["classes"]) + len(result["unassigned"]) == len(codes)


def test_denoised_feature_set_runs_end_to_end(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(
            codes=codes, start_date="2021-01-01", features="denoised", k=3, size_min=1, size_max=5
        ),
    )
    assert result["features"] == "denoised"
    assert len(result["classes"]) == 3
    # Denoising only reshapes the clustering geometry; reported correlations
    # must stay the raw observed ones.
    raw = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(
            codes=codes, start_date="2021-01-01", features="correlation", k=3, size_min=1, size_max=5
        ),
    )
    assert (
        result["diagnostics"]["significant_eigenvalues"]
        == raw["diagnostics"]["significant_eigenvalues"]
    )


def test_unknown_codes_are_reported_not_silently_dropped(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(codes=[*codes, "NOPE.SH"], start_date="2021-01-01", k=3, size_min=1, size_max=3),
    )
    assert [item["code"] for item in result["skipped"]] == ["NOPE.SH"]
    assert any("缺少可用净值" in warning for warning in result["warnings"])


def test_duplicate_codes_are_deduplicated(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(codes=[*codes, codes[0]], start_date="2021-01-01", k=3, size_min=1, size_max=3),
    )
    assigned = [member["code"] for group in result["classes"] for member in group["etfs"]]
    assert len(assigned) == len(set(assigned))


def test_metric_lane_fails_closed_without_a_metrics_snapshot(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    (tmp_path / "instrument_metrics_snapshot.parquet").unlink()
    with pytest.raises(service.AutoClassError, match="没有任何可用的风险收益特征"):
        service.run_auto_classification(
            tmp_path,
            service.AutoClassRequestSpec(codes=codes, start_date="2021-01-01", features="metrics", k=2),
        )


def test_late_listed_product_that_truncates_the_sample_is_named(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    frame = pd.read_parquet(tmp_path / "etf_daily_df.parquet")
    cutoff = frame["date"].sort_values().unique()[150]
    frame = frame[(frame["ts_code"] != "AU1.SH") | (frame["date"] >= cutoff)]
    frame.to_parquet(tmp_path / "etf_daily_df.parquet", index=False)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(codes=codes, start_date="2021-01-01", k=3, size_min=1, size_max=3),
    )
    binding = result["diagnostics"]["sample_binding"]
    assert [item["code"] for item in binding] == ["AU1.SH"]
    assert any("最晚有净值的产品" in warning for warning in result["warnings"])
    assert result["start_date"] >= binding[0]["first_date"]


def test_cross_class_matrix_matches_the_returned_classes(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(codes=codes, start_date="2021-01-01", k=3, size_min=1, size_max=1),
    )
    labels = result["diagnostics"]["cross_class_labels"]
    matrix = result["diagnostics"]["cross_class_corr"]
    assert labels == [group["name"] for group in result["classes"]]
    assert len(matrix) == len(labels)
    assert all(len(row) == len(labels) for row in matrix)


@pytest.mark.parametrize("features", ["correlation", "metrics", "pca", "blend"])
def test_every_feature_lane_produces_a_serialisable_result(tmp_path: Path, features: str) -> None:
    codes = _write_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(codes=codes, start_date="2021-01-01", features=features, k=2, size_min=1, size_max=4),
    )
    json.dumps(result)  # every float must already be finite-or-None
    assert result["features"] == features
    assert result["classes"]


@pytest.mark.parametrize(
    ("spec_kwargs", "message"),
    [
        ({"codes": ["EQ0.SH"]}, "至少需要 2 个产品"),
        ({"algorithm": "nope"}, "不支持的算法"),
        ({"features": "nope"}, "不支持的特征集"),
        ({"weight_mode": "nope"}, "不支持的类内权重"),
        ({"unassigned_policy": "nope"}, "不支持的未归类策略"),
        ({"size_min": 0}, "不能小于 1"),
        ({"size_min": 5, "size_max": 2}, "不能小于最少产品数"),
        ({"k": 1}, "至少为 2"),
        ({"k": 99}, "超过可用产品数"),
        ({"k": 3, "size_min": 4}, "超过可用产品数"),
        ({"start_date": "not-a-date"}, "startDate 格式错误"),
    ],
)
def test_invalid_requests_fail_closed(tmp_path: Path, spec_kwargs: Dict[str, Any], message: str) -> None:
    codes = _write_fixture(tmp_path)
    kwargs: Dict[str, Any] = {"codes": codes, "start_date": "2021-01-01", "k": 3, **spec_kwargs}
    with pytest.raises(service.AutoClassError) as excinfo:
        service.run_auto_classification(tmp_path, service.AutoClassRequestSpec(**kwargs))
    assert message in str(excinfo.value)


def test_short_sample_is_rejected_rather_than_computed(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    with pytest.raises(service.AutoClassError, match="最低要求"):
        service.run_auto_classification(
            tmp_path,
            service.AutoClassRequestSpec(codes=codes, start_date="2022-06-01", k=2),
        )


def test_bad_nav_print_does_not_scatter_the_classification(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    frame = pd.read_parquet(tmp_path / "etf_daily_df.parquet")
    # Double one bond fund's NAV mid-series, the way an unadjusted split looks.
    mask = (frame["ts_code"] == "BD0.SH") & (frame["date"] >= frame["date"].iloc[200])
    frame.loc[mask, "adj_nav"] *= 2.0
    frame.to_parquet(tmp_path / "etf_daily_df.parquet", index=False)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(codes=codes, start_date="2021-01-01", k=3, size_min=2, size_max=3),
    )
    assert [item["code"] for item in result["diagnostics"]["winsorized"]] == ["BD0.SH"]
    bond_class = next(
        group for group in result["classes"] if any(member["code"] == "BD0.SH" for member in group["etfs"])
    )
    assert {member["code"] for member in bond_class["etfs"]} == {"BD0.SH", "BD1.SH"}
    assert any("异常跳变" in warning for warning in result["warnings"])


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

def test_pool_limit_caps_the_intra_class_weight(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    # BD0/BD1 form one class; restrict BD0 to 10% of the portfolio.
    limited = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(
            codes=codes, start_date="2021-01-01", k=3, size_min=2, size_max=3,
            max_weights={"BD0.SH": 0.10},
        ),
    )
    bond = next(
        group for group in limited["classes"]
        if {member["code"] for member in group["etfs"]} == {"BD0.SH", "BD1.SH"}
    )
    capped = next(member for member in bond["etfs"] if member["code"] == "BD0.SH")
    assert capped["max_weight"] == pytest.approx(0.10)
    assert capped["capped"] is True
    assert capped["weight"] <= 10.0 + 1e-9
    assert sum(member["weight"] for member in bond["etfs"]) == pytest.approx(100.0)
    assert any("产品池限额" in warning for warning in limited["warnings"])


def test_class_capacity_is_reported_when_limits_cannot_fill_a_class(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(
            codes=codes, start_date="2021-01-01", k=3, size_min=2, size_max=3,
            max_weights={"BD0.SH": 0.10, "BD1.SH": 0.15},
        ),
    )
    bond = next(
        group for group in result["classes"]
        if {member["code"] for member in group["etfs"]} == {"BD0.SH", "BD1.SH"}
    )
    assert bond["max_class_weight"] == pytest.approx(25.0)
    assert any("SAA 权重不得超过 25.0%" in warning for warning in result["warnings"])
    # At that class weight every product lands exactly on its own limit.
    weight = bond["max_class_weight"] / 100.0
    limits = {"BD0.SH": 10.0, "BD1.SH": 15.0}
    for member in bond["etfs"]:
        assert member["weight"] * weight <= limits[member["code"]] + 1e-9


def test_unrestricted_pool_leaves_class_capacity_at_full(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(codes=codes, start_date="2021-01-01", k=3, size_min=1, size_max=3),
    )
    assert all(group["max_class_weight"] == pytest.approx(100.0) for group in result["classes"])
    assert all(member["max_weight"] is None and member["capped"] is False
               for group in result["classes"] for member in group["etfs"])
    assert not any("产品池限额" in warning for warning in result["warnings"])


def test_request_model_accepts_every_option_the_meta_route_advertises() -> None:
    """A registry entry the request schema rejects reaches the UI as an
    unrenderable 422, not as a usable option."""

    fields = routes.AutoClassPreviewRequest.model_fields
    for field, registry in (
        ("algorithm", service.ALGORITHMS),
        ("features", service.FEATURE_SETS),
        ("linkage", service.LINKAGE_METHODS),
        ("weightMode", service.WEIGHT_MODES),
        ("blockBy", service.BLOCK_MODES),
        ("taxonomyLevel", taxonomy.TAXONOMY_LEVELS),
    ):
        assert set(get_args(fields[field].annotation)) == set(registry), field


def test_meta_route_exposes_every_supported_option() -> None:
    meta = routes.auto_class_meta()
    assert {item["id"] for item in meta["algorithms"]} == set(service.ALGORITHMS)
    assert {item["id"] for item in meta["features"]} == set(service.FEATURE_SETS)
    assert meta["defaults"]["algorithm"] in service.ALGORITHMS
    assert meta["limits"]["min_observations"] == service.MIN_OBSERVATIONS


def test_preview_route_returns_a_draft(tmp_path: Path, monkeypatch) -> None:
    codes = _write_fixture(tmp_path)
    universe = _write_universe(tmp_path, codes)
    monkeypatch.setattr(routes, "DATA_DIR", tmp_path)
    payload = routes.AutoClassPreviewRequest(
        universe_snapshot_id=universe["id"],
        products=[routes.PoolProduct(code=code, kind="etf") for code in codes],
        startDate="2021-01-01",
        k=3,
        sizeMin=2,
        sizeMax=3,
    )
    result = routes.auto_class_preview(payload)
    assert not isinstance(result, JSONResponse)
    assert len(result["classes"]) == 3
    assert result["universe_snapshot"]["id"] == universe["id"]


def test_preview_route_applies_pool_limits_from_the_snapshot(tmp_path: Path, monkeypatch) -> None:
    codes = _write_fixture(tmp_path)
    universe = _write_universe(tmp_path, codes, limits={"BD0.SH": 0.10})
    monkeypatch.setattr(routes, "DATA_DIR", tmp_path)
    payload = routes.AutoClassPreviewRequest(
        universe_snapshot_id=universe["id"],
        products=[routes.PoolProduct(code=code, kind="etf") for code in codes],
        startDate="2021-01-01",
        k=3,
        sizeMin=2,
        sizeMax=3,
    )
    result = routes.auto_class_preview(payload)
    assert not isinstance(result, JSONResponse)
    restricted = [
        member
        for group in result["classes"]
        for member in group["etfs"]
        if member["code"] == "BD0.SH"
    ]
    assert len(restricted) == 1
    assert restricted[0]["max_weight"] == pytest.approx(0.10)
    assert restricted[0]["weight"] <= 10.0 + 1e-9


def test_preview_route_reports_bad_input_as_400(tmp_path: Path, monkeypatch) -> None:
    universe = _write_universe(tmp_path, ["EQ0.SH"])
    monkeypatch.setattr(routes, "DATA_DIR", tmp_path)
    payload = routes.AutoClassPreviewRequest(
        universe_snapshot_id=universe["id"],
        products=[routes.PoolProduct(code="EQ0.SH", kind="etf")],
    )
    response = routes.auto_class_preview(payload)
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert "至少需要 2 个产品" in _json(response)["detail"]


def test_preview_route_reports_missing_market_data_as_404(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "empty"
    universe = _write_universe(data_dir, ["EQ0.SH", "EQ1.SH"])
    monkeypatch.setattr(routes, "DATA_DIR", data_dir)
    payload = routes.AutoClassPreviewRequest(
        universe_snapshot_id=universe["id"],
        products=[
            routes.PoolProduct(code="EQ0.SH", kind="etf"),
            routes.PoolProduct(code="EQ1.SH", kind="etf"),
        ],
    )
    response = routes.auto_class_preview(payload)
    assert isinstance(response, JSONResponse)
    assert response.status_code == 404


def test_preview_route_rejects_product_outside_universe(tmp_path: Path, monkeypatch) -> None:
    universe = _write_universe(tmp_path, ["EQ0.SH"])
    monkeypatch.setattr(routes, "DATA_DIR", tmp_path)
    payload = routes.AutoClassPreviewRequest(
        universe_snapshot_id=universe["id"],
        products=[
            routes.PoolProduct(code="EQ0.SH", kind="etf"),
            routes.PoolProduct(code="EQ1.SH", kind="etf"),
        ],
    )

    response = routes.auto_class_preview(payload)

    assert isinstance(response, JSONResponse)
    # 422 is the shared product-pool contract for a semantically invalid payload
    # (PRODUCT_OUTSIDE_INVESTABLE_UNIVERSE), not a generic 400.
    assert response.status_code == 422
    detail = _json(response)["detail"]
    assert "存在不在可投资域内或当前不可用的产品" in detail
    assert "EQ1.SH: not_in_universe" in detail
    assert "EQ0.SH" not in detail, "只应报告违规产品"


def _write_live_shaped_universe(data_dir: Path, codes: list[str]) -> dict[str, Any]:
    """A snapshot exactly as ``ProductPoolService.create_universe_snapshot`` stores it.

    Every other fixture here writes the decorated ``members`` shape, which is why
    the whole suite stayed green while the running app answered
    「未找到指定可投资域快照。」 for its own snapshots.
    """

    return InvestableUniverseRepository(data_dir / UNIVERSE_SNAPSHOT_STORE).create(
        {
            "name": "投前研究可投资域",
            "research_date": "2026-09-04",
            "version_ids": ["pool-version-1"],
            "pool_ids": ["pool-1"],
            "excluded_product_keys": [],
            "groups": [],
            "products": [
                {
                    "key": f"etf:{code}",
                    "kind": "etf",
                    "product_id": code,
                    "code": code,
                    "name": code,
                    "usage_status": "normal",
                    "max_weight": None,
                    "valid_until": None,
                    "substitute_group": "",
                    "reasons": [],
                    "source_version_ids": ["pool-version-1"],
                    "source_pool_ids": ["pool-1"],
                }
                for code in codes
            ],
            "product_count": len(codes),
        }
    )


def test_preview_route_accepts_a_snapshot_stored_by_the_product_pool_service(
    tmp_path: Path,
    monkeypatch,
) -> None:
    codes = _write_fixture(tmp_path)
    universe = _write_live_shaped_universe(tmp_path, codes)
    monkeypatch.setattr(routes, "DATA_DIR", tmp_path)

    result = routes.auto_class_preview(
        routes.AutoClassPreviewRequest(
            universe_snapshot_id=universe["id"],
            products=[routes.PoolProduct(code=code, kind="etf") for code in codes],
            startDate="2021-01-01",
            k=3,
            sizeMin=2,
            sizeMax=3,
        )
    )

    assert not isinstance(result, JSONResponse)
    assert len(result["classes"]) == 3
    # The lineage is stored as parallel id lists in this shape, not version_refs.
    assert result["universe_snapshot"]["version_refs"] == [
        {"version_id": "pool-version-1", "pool_id": "pool-1"}
    ]


def test_products_shaped_snapshot_still_blocks_unavailable_products(tmp_path: Path) -> None:
    from product_pools.errors import ProductPoolValidationError
    from product_pools.membership import InvestableUniverseMembership

    snapshot = _write_live_shaped_universe(tmp_path, ["EQ0.SH", "EQ1.SH"])
    store = InvestableUniverseRepository(tmp_path / UNIVERSE_SNAPSHOT_STORE)
    with store.store.locked():
        raw = store.store.read_unlocked()
        raw["universe_snapshots"][0]["products"][1]["usage_status"] = "unavailable"
        store.store.write_unlocked(raw)
    validator = InvestableUniverseMembership(store)

    ok = validator.validate(snapshot["id"], [{"kind": "etf", "product_id": "EQ0.SH"}])
    assert [item["product_id"] for item in ok.members] == ["EQ0.SH"]

    with pytest.raises(ProductPoolValidationError) as error:
        validator.validate(snapshot["id"], [{"kind": "etf", "product_id": "EQ1.SH"}])
    assert error.value.diagnostics[0]["reason"] == "not_eligible"


def test_auto_class_route_and_product_pool_service_share_one_snapshot_store() -> None:
    """The original defect: the route read a file the writer never created."""

    from services import product_pool_routes

    assert (
        Path(product_pool_routes.product_pool_service.repository.store.path).name
        == UNIVERSE_SNAPSHOT_STORE
    )


# --------------------------------------------------------------------------
# Contract taxonomy (fund_taxonomy)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # 宽基规模: the size buckets share prefixes, so the longest name must win.
        ("华泰柏瑞沪深300ETF", ("权益类", "宽基规模", "大盘")),
        ("南方中证500ETF", ("权益类", "宽基规模", "中盘")),
        ("华夏中证1000ETF", ("权益类", "宽基规模", "小盘")),
        ("易方达中证100ETF", ("权益类", "宽基规模", "大盘")),
        ("华夏上证科创板50ETF", ("权益类", "宽基规模", "科创板")),
        ("易方达创业板ETF", ("权益类", "宽基规模", "创业板")),
        # 风格因子 (Smart Beta) beats the index name it is built on.
        ("华泰柏瑞中证红利低波动ETF", ("权益类", "风格因子", "红利")),
        ("景顺长城中证500低波动ETF", ("权益类", "风格因子", "低波")),
        ("华宝中证800自由现金流ETF", ("权益类", "风格因子", "自由现金流")),
        # 主题 crosses industries and is resolved before the industry table.
        ("华夏中证人工智能主题ETF", ("权益类", "主题", "数字化与人工智能")),
        ("华泰柏瑞中证光伏产业ETF", ("权益类", "主题", "低碳转型")),
        ("汇添富中证国新央企科技引领ETF", ("权益类", "主题", "国企改革")),
        # 行业: the eight buckets.
        ("华宝中证全指证券公司ETF", ("权益类", "行业", "金融")),
        ("国联安中证全指半导体ETF", ("权益类", "行业", "科技")),
        ("汇添富中证主要消费ETF", ("权益类", "行业", "消费")),
        ("永赢中证沪深港黄金产业股票ETF", ("权益类", "行业", "周期")),
        # 固收
        ("鹏扬中债30年期国债ETF", ("固收类", "利率债", "国债")),
        ("平安中证公司债ETF", ("固收类", "信用债", "公司债")),
        ("海富通上证城投债ETF", ("固收类", "信用债", "城投债")),
        ("博时可转债ETF", ("固收类", "可转债", "可转债")),
        ("华富中证同业存单AAA指数基金", ("固收类", "同业存单", "同业存单")),
        # 商品 / 海外 / 货币
        ("华安易富黄金ETF", ("商品类", "贵金属", "黄金")),
        ("华夏饲料豆粕期货ETF", ("商品类", "农产品", "豆粕")),
        ("华夏恒生科技ETF(QDII)", ("海外类", "港股", "港股科技")),
        ("广发纳斯达克100ETF(QDII)", ("海外类", "美股", "纳斯达克")),
        ("华宝现金添益货币", ("货币类", "货币", "货币")),
        ("某未知产品", ("其他类", "其他类", "其他类")),
    ],
)
def test_taxonomy_resolves_the_researched_buckets(name: str, expected: tuple[str, str, str]) -> None:
    label = taxonomy.classify((name,))
    assert (label.asset_class, label.category, label.detail) == expected


def test_gold_sector_equity_is_not_filed_as_a_commodity() -> None:
    """The trap the old flat table fell into: 黄金股/有色金属 are equity sectors."""

    assert taxonomy.classify(("永赢中证沪深港黄金股票ETF",)).asset_class == "权益类"
    assert taxonomy.classify(("有色金属ETF",)).asset_class == "权益类"
    assert taxonomy.classify(("华安黄金ETF",)).asset_class == "商品类"


def test_science_innovation_bond_is_not_a_science_board_equity() -> None:
    assert taxonomy.classify(("科创债ETF",)).category == "信用债"
    assert taxonomy.classify(("科创50ETF",)).detail == "科创板"


def test_taxonomy_path_collapses_repeated_levels() -> None:
    assert taxonomy.classify(("博时可转债ETF",)).path == "固收类 / 可转债"
    assert taxonomy.classify(("沪深300ETF",)).path == "权益类 / 宽基规模 / 大盘"


def test_taxonomy_tree_is_serialisable_and_lists_every_asset_class() -> None:
    tree = taxonomy.taxonomy_tree()
    json.dumps(tree, ensure_ascii=False)
    assert {item["asset_class"] for item in tree} >= {"权益类", "固收类", "商品类", "海外类", "货币类", "混合类"}


# --------------------------------------------------------------------------
# Block allocation kernel
# --------------------------------------------------------------------------

def test_block_allocation_gives_every_block_at_least_one_class() -> None:
    sizes = np.ascontiguousarray(np.array([10, 3, 1], dtype=np.int64))
    counts = kernels.allocate_block_clusters_kernel(sizes, np.int64(6), np.int64(2))
    assert counts.tolist() == [4, 1, 1]
    assert counts.sum() == 6


def test_block_allocation_never_exceeds_a_block_capacity() -> None:
    sizes = np.ascontiguousarray(np.array([4, 2], dtype=np.int64))
    # size_min=2 caps the blocks at 2 and 1 classes, so K=9 is clamped to 3.
    counts = kernels.allocate_block_clusters_kernel(sizes, np.int64(9), np.int64(2))
    assert counts.tolist() == [2, 1]


def test_block_allocation_keeps_blocks_when_k_is_below_the_block_count() -> None:
    """The taxonomy is a hard partition: K may be raised, never a block merged."""

    sizes = np.ascontiguousarray(np.array([5, 5, 5], dtype=np.int64))
    counts = kernels.allocate_block_clusters_kernel(sizes, np.int64(2), np.int64(2))
    assert counts.tolist() == [1, 1, 1]


def test_block_allocation_handles_empty_and_degenerate_input() -> None:
    empty = kernels.allocate_block_clusters_kernel(
        np.ascontiguousarray(np.array([], dtype=np.int64)), np.int64(3), np.int64(2)
    )
    assert empty.shape == (0,)
    zeros = kernels.allocate_block_clusters_kernel(
        np.ascontiguousarray(np.array([0, 2], dtype=np.int64)), np.int64(4), np.int64(1)
    )
    assert zeros.tolist() == [0, 2]


# --------------------------------------------------------------------------
# Taxonomy-blocked classification
# --------------------------------------------------------------------------

def _asset_classes(result: Dict[str, Any]) -> list[set[str]]:
    return [
        {member["taxonomy"]["asset_class"] for member in group["etfs"]}
        for group in result["classes"]
    ]


def test_blocking_never_mixes_asset_classes_even_under_the_force_policy(tmp_path: Path) -> None:
    """A correlation window can make gold look like equity; the contract cannot."""

    codes = _write_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(
            codes=codes,
            start_date="2021-01-01",
            block_by="asset_class",
            size_min=1,
            size_max=8,
            unassigned_policy="force",
        ),
    )
    assert all(len(classes) == 1 for classes in _asset_classes(result))
    assert not result["unassigned"]
    assert {item["block"] for item in result["diagnostics"]["blocks"]} == {"权益类", "固收类", "商品类"}
    assert result["block_by"] == "asset_class"


def test_blocking_raises_k_rather_than_merging_two_contract_blocks(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(
            codes=codes, start_date="2021-01-01", k=2, block_by="asset_class", size_min=1, size_max=8
        ),
    )
    assert result["k"] == 3
    assert any("与指定的 2 类不一致" in warning for warning in result["warnings"])
    assert all(len(classes) == 1 for classes in _asset_classes(result))


def test_blocking_without_k_splits_only_blocks_that_have_structure(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(
            codes=codes, start_date="2021-01-01", block_by="asset_class", size_min=1, size_max=8
        ),
    )
    # Each fixture block is one factor, so no block earns a split.
    assert [item["k"] for item in result["diagnostics"]["blocks"]] == [1, 1, 1]
    assert result["k"] == 3


def test_unblocked_classification_is_unchanged_by_the_taxonomy_work(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(codes=codes, start_date="2021-01-01", k=3, size_min=2, size_max=3),
    )
    assert result["k"] == 3
    assert result["block_by"] == "none"
    assert result["diagnostics"]["blocks"] == []


def test_taxonomy_level_drives_the_rule_algorithm_granularity(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    coarse = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(
            codes=codes, start_date="2021-01-01", algorithm="rule", size_min=1, size_max=5
        ),
    )
    fine = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(
            codes=codes,
            start_date="2021-01-01",
            algorithm="rule",
            taxonomy_level="detail",
            size_min=1,
            size_max=5,
        ),
    )
    assert {group["name"] for group in coarse["classes"]} == {"权益类", "固收类", "商品类"}
    assert {group["name"] for group in fine["classes"]} == {"宽基规模", "综合债", "黄金"}


def test_members_carry_the_full_taxonomy_path(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(codes=codes, start_date="2021-01-01", k=3, size_min=2, size_max=3),
    )
    json.dumps(result, ensure_ascii=False)
    members = [member for group in result["classes"] for member in group["etfs"]]
    assert all(set(member["taxonomy"]) == {"asset_class", "category", "detail", "path", "matched"}
               for member in members)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"taxonomy_level": "sector"}, "不支持的合同分类层级"),
        ({"block_by": "industry"}, "不支持的分层方式"),
    ],
)
def test_invalid_taxonomy_options_fail_closed(tmp_path: Path, kwargs: Dict[str, Any], message: str) -> None:
    codes = _write_fixture(tmp_path)
    with pytest.raises(service.AutoClassError, match=message):
        service.run_auto_classification(
            tmp_path, service.AutoClassRequestSpec(codes=codes, start_date="2021-01-01", **kwargs)
        )


def test_meta_route_exposes_the_taxonomy_options() -> None:
    meta = routes.auto_class_meta()
    assert {item["id"] for item in meta["taxonomy_levels"]} == set(taxonomy.TAXONOMY_LEVELS)
    assert {item["id"] for item in meta["block_modes"]} == set(service.BLOCK_MODES)
    assert meta["defaults"]["block_by"] == "none"
    json.dumps(meta, ensure_ascii=False)


def test_preview_route_passes_the_taxonomy_options_through(tmp_path: Path, monkeypatch) -> None:
    codes = _write_fixture(tmp_path)
    universe = _write_universe(tmp_path, codes)
    monkeypatch.setattr(routes, "DATA_DIR", tmp_path)
    result = routes.auto_class_preview(routes.AutoClassPreviewRequest(
        universe_snapshot_id=universe["id"],
        products=[routes.PoolProduct(code=code, kind="etf") for code in codes],
        startDate="2021-01-01",
        blockBy="asset_class",
        taxonomyLevel="category",
        sizeMin=1,
        sizeMax=8,
    ))
    assert not isinstance(result, JSONResponse)
    assert result["block_by"] == "asset_class"
    assert result["taxonomy_level"] == "category"
    assert all(len(classes) == 1 for classes in _asset_classes(result))


def _write_two_style_equity_fixture(tmp_path: Path) -> list[str]:
    """Equity block holding two independent styles, plus a bond block."""

    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2021-01-01", periods=400)
    blocks = {"EA": (3, 0.012), "EB": (3, 0.010), "BD": (2, 0.001)}
    rows: list[dict[str, Any]] = []
    codes: list[str] = []
    for prefix, (count, scale) in blocks.items():
        factor = rng.normal(0.0, scale, len(dates))
        for index in range(count):
            code = f"{prefix}{index}.SH"
            codes.append(code)
            returns = factor + rng.normal(0.0, scale / 10.0, len(dates))
            nav = np.cumprod(1.0 + returns)
            rows.extend(
                {"ts_code": code, "name": f"{prefix} Fund {index}", "date": date, "adj_nav": value}
                for date, value in zip(dates, nav)
            )
    pd.DataFrame(rows).to_parquet(tmp_path / "etf_daily_df.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ts_code": code,
                "code": code.split(".")[0],
                "name": f"{code[:2]} Fund",
                "instrument_type": "etf",
                "fund_type": "债券型" if code.startswith("BD") else "股票型",
                "invest_type": "被动指数型",
                "benchmark": "",
                "index_name": "",
                "management": "Test AMC",
            }
            for code in codes
        ]
    ).to_parquet(tmp_path / "etf_info_df.parquet", index=False)
    return codes


def test_a_block_with_two_real_styles_still_splits(tmp_path: Path) -> None:
    """The 0.25 silhouette floor must not turn auto-K inside a block off."""

    codes = _write_two_style_equity_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(
            codes=codes, start_date="2021-01-01", block_by="asset_class", size_min=1, size_max=8
        ),
    )
    blocks = {item["block"]: item for item in result["diagnostics"]["blocks"]}
    assert blocks["权益类"]["k"] == 2
    assert blocks["权益类"]["silhouette"] > service.BLOCK_SPLIT_SILHOUETTE
    assert blocks["固收类"]["k"] == 1
    assert result["k"] == 3
    assert all(len(classes) == 1 for classes in _asset_classes(result))


# --------------------------------------------------------------------------
# Point-in-time: the contract table, the metrics snapshot and the universe
# --------------------------------------------------------------------------

def _write_taxonomy_history(
    tmp_path: Path, codes: list[str], *, snapshots: tuple[tuple[str, str], ...]
) -> None:
    """Dated snapshots of the contract table, as the refresh appends them.

    Each entry is (snapshot_date, the fund_type the BD funds carried that day),
    so a replayed run can be caught classifying on the label of its own time
    rather than on today's.
    """

    rows: list[dict[str, Any]] = []
    for snapshot_date, bond_type in snapshots:
        for code in codes:
            rows.append(
                {
                    "pit_snapshot_date": pd.Timestamp(snapshot_date),
                    "ts_code": code,
                    "code": code.split(".")[0],
                    "name": f"{code[:2]} Fund",
                    "instrument_type": "etf",
                    "fund_type": {"EQ": "股票型", "BD": bond_type, "AU": "其他"}[code[:2]],
                    "invest_type": {
                        "EQ": "被动指数型",
                        "BD": "被动指数型",
                        "AU": "黄金现货合约",
                    }[code[:2]],
                    "benchmark": "",
                    "index_name": "",
                    "management": "Test AMC",
                }
            )
    (tmp_path / "pit_dim").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(
        tmp_path / "pit_dim" / "etf_info_df_history.parquet", index=False
    )


def _stamp_nav_announcements(tmp_path: Path) -> None:
    """Give the fixture NAV a publication column so strict mode can get past it.

    Strict PIT refuses NAV without `ann_date` before any classification runs, so
    a test about the contract table has to clear that gate first.
    """

    path = tmp_path / "etf_daily_df.parquet"
    frame = pd.read_parquet(path)
    frame["ann_date"] = pd.to_datetime(frame["date"]) + pd.Timedelta(days=1)
    frame.to_parquet(path, index=False)


def test_rule_classification_replays_the_contract_table_of_the_research_day(
    tmp_path: Path,
) -> None:
    """`rule` classifies entirely on the contract table, so replay must reach it.

    The fixture's live table calls the BD funds 债券型 (固收类). In 2021 they were
    filed as 股票型 (权益类), which is what the dated snapshot holds. A 2021 run
    that reads today's table produces three classes; one that stands on 2021
    produces two — the clusters change, not just their names.
    """

    codes = _write_fixture(tmp_path)
    _write_taxonomy_history(
        tmp_path, codes, snapshots=(("2021-03-01", "股票型"), ("2026-01-01", "债券型"))
    )

    replayed = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(
            codes=codes, start_date="2021-01-01", algorithm="rule", as_of="2021-09-01"
        ),
    )
    latest = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(
            codes=codes, start_date="2021-01-01", algorithm="rule"
        ),
    )

    assert {group["name"] for group in replayed["classes"]} == {"权益类", "商品类"}
    assert {group["name"] for group in latest["classes"]} == {"权益类", "固收类", "商品类"}
    # It is the research day that picks the snapshot, not the dataset: the same
    # log answers 2021 with the 2021 labels and answers "no cut-off" with today's.
    assert replayed["pit"]["taxonomy"]["snapshot_used"]["etf_info"] == "2021-03-01"
    assert latest["pit"]["taxonomy"]["snapshot_used"]["etf_info"] == "2026-01-01"
    assert replayed["pit"]["taxonomy"]["coverage"] == "REPLAYED"


def test_research_mode_names_a_latest_state_contract_table(tmp_path: Path) -> None:
    codes = _write_fixture(tmp_path)
    result = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(
            codes=codes, start_date="2021-01-01", k=3, as_of="2021-09-01"
        ),
    )
    assert result["pit"]["taxonomy"]["coverage"] == "LATEST_ONLY"
    assert any("合同分类信息只有最新态" in warning for warning in result["warnings"])


def test_strict_mode_refuses_a_contract_history_that_starts_after_the_research_day(
    tmp_path: Path,
) -> None:
    """The trap `require_usable` alone cannot catch.

    Two snapshots on disk earn the table a grade of B, so the dataset gate lets
    strict mode through — but neither snapshot is at or before the research day,
    so there is nothing to replay and the read would fall back to today's
    labels. Failing closed here is the whole point of reading through `read_pit`
    instead of off the latest-state file.
    """

    codes = _write_fixture(tmp_path)
    _stamp_nav_announcements(tmp_path)
    _write_taxonomy_history(
        tmp_path, codes, snapshots=(("2022-01-01", "股票型"), ("2026-01-01", "债券型"))
    )
    with pytest.raises(service.AutoClassError, match="严格 PIT 模式"):
        service.run_auto_classification(
            tmp_path,
            service.AutoClassRequestSpec(
                codes=codes,
                start_date="2021-01-01",
                algorithm="rule",
                as_of="2021-09-01",
                run_mode="STRICT_PIT",
            ),
        )


def test_the_metrics_feature_lane_is_look_ahead_and_says_so(tmp_path: Path) -> None:
    """The indicator snapshot has no clock at all — one row per product, in place.

    Pairing it with a research day is not a degraded PIT read, it is 2026 的三年
    最大回撤 deciding a 2021 classification. Strict refuses; research names it.
    """

    codes = _write_fixture(tmp_path)
    _write_taxonomy_history(tmp_path, codes, snapshots=(("2021-03-01", "债券型"),))
    spec = dict(codes=codes, start_date="2021-01-01", features="metrics", k=2)

    with pytest.raises(service.AutoClassError, match="风险收益画像"):
        service.run_auto_classification(
            tmp_path,
            service.AutoClassRequestSpec(
                **spec, as_of="2021-09-01", run_mode="STRICT_PIT"
            ),
        )
    recorded = service.run_auto_classification(
        tmp_path, service.AutoClassRequestSpec(**spec, as_of="2021-09-01")
    )
    assert any("没有历史版本" in warning for warning in recorded["warnings"])
    # A correlation run reads no snapshot, so it must not inherit the warning.
    clean = service.run_auto_classification(
        tmp_path,
        service.AutoClassRequestSpec(
            codes=codes, start_date="2021-01-01", k=2, as_of="2021-09-01"
        ),
    )
    assert not any("没有历史版本" in warning for warning in clean["warnings"])


def test_preview_judges_the_locked_universe_against_the_research_day(
    tmp_path: Path, monkeypatch
) -> None:
    """A pool screened on 2026 numbers, replayed over 2021.

    Every individual computation is perfectly causal, which is why this can only
    be caught at the candidate set. Research mode records it on the result;
    strict mode refuses the run.
    """

    codes = _write_fixture(tmp_path)
    _write_taxonomy_history(tmp_path, codes, snapshots=(("2021-03-01", "债券型"),))
    universe = _write_universe(tmp_path, codes)  # research_date 2026-09-04
    monkeypatch.setattr(routes, "DATA_DIR", tmp_path)
    request = dict(
        universe_snapshot_id=universe["id"],
        products=[routes.PoolProduct(code=code, kind="etf") for code in codes],
        startDate="2021-01-01",
        k=3,
        sizeMin=2,
        sizeMax=3,
    )

    recorded = routes.auto_class_preview(
        routes.AutoClassPreviewRequest(**request, asOf="2021-09-01")
    )
    assert not isinstance(recorded, JSONResponse)
    universe_pit = recorded["pit"]["universe"]
    assert universe_pit["clean"] is False
    assert universe_pit["established_at"] == "2026-09-04"
    assert [item["code"] for item in universe_pit["findings"]] == ["UNIVERSE_LOOKAHEAD"]

    refused = routes.auto_class_preview(
        routes.AutoClassPreviewRequest(
            **request, asOf="2021-09-01", runMode="STRICT_PIT"
        )
    )
    assert isinstance(refused, JSONResponse)
    assert refused.status_code == 400
    assert "未来信息" in _json(refused)["detail"]
