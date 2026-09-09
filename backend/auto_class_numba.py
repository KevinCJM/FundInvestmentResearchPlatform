from __future__ import annotations

"""Fixed-signature nopython kernels for automatic asset-class construction.

Every numerical step of the auto-classification chain lives here: feature
standardisation, correlation/distance matrices, agglomerative linkage, k-means,
k-medoids, capacity-constrained assignment, silhouette diagnostics and
intra-class weighting.  The Python orchestration layer only loads data, converts
frames to arrays and serialises results.
"""

import hashlib

import numpy as np
from numba import float64, int64, njit, types, uint8

try:
    from backend.compute_policy import validate_execution_audit
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from compute_policy import validate_execution_audit


AUTO_CLASS_ENGINE_VERSION = "auto-asset-class-njit-1.0.0"
AUTO_CLASS_KERNEL_VERSION = "auto-asset-class-kernels-3"

_F1 = float64[::1]
_F2 = float64[:, ::1]
_I1 = int64[::1]
_U1 = uint8[::1]

LINKAGE_AVERAGE = 0
LINKAGE_COMPLETE = 1
LINKAGE_WARD = 2

WEIGHT_EQUAL = 0
WEIGHT_INV_VOL = 1
WEIGHT_INV_VAR = 2
WEIGHT_AFFINITY = 3


@njit(_F2(_F2), cache=False, nogil=True)
def robust_standardize_kernel(features: np.ndarray) -> np.ndarray:
    """Median/MAD z-score per column; non-finite inputs collapse to 0.0."""

    rows, columns = features.shape
    output = np.zeros((rows, columns), dtype=np.float64)
    if rows == 0 or columns == 0:
        return output
    buffer = np.empty(rows, dtype=np.float64)
    deviations = np.empty(rows, dtype=np.float64)
    for column in range(columns):
        count = 0
        for row in range(rows):
            value = features[row, column]
            if np.isfinite(value):
                buffer[count] = value
                count += 1
        if count == 0:
            continue
        median = np.median(buffer[:count])
        for index in range(count):
            deviations[index] = abs(buffer[index] - median)
        mad = np.median(deviations[:count])
        scale = mad * 1.4826
        if not np.isfinite(scale) or scale <= 1e-12:
            # Degenerate spread: fall back to standard deviation, then to 1.0.
            total = 0.0
            for index in range(count):
                total += (buffer[index] - median) * (buffer[index] - median)
            scale = np.sqrt(total / count)
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
        for row in range(rows):
            value = features[row, column]
            if np.isfinite(value):
                normalized = (value - median) / scale
                # Winsorise so a single outlier cannot dominate every distance.
                if normalized > 5.0:
                    normalized = 5.0
                elif normalized < -5.0:
                    normalized = -5.0
                output[row, column] = normalized
    return output


@njit(types.Tuple((_F2, _I1, _F1))(_F2, float64), cache=False, nogil=True)
def winsorize_returns_kernel(returns: np.ndarray, sigma_multiple: float64):
    """Clip each column to median +/- `sigma_multiple` robust sigma.

    Source adjusted-NAV series occasionally carry an unadjusted split/dividend
    jump (a single +99% day on a treasury ETF, for example).  One such point is
    enough to drive that product's correlation with everything else to zero and
    scatter it across the classification.  Only the clustering feature lane is
    winsorized; reported NAV and performance metrics keep the raw series.

    Returns the clipped matrix, the per-column count of clipped observations
    and the largest raw magnitude that had to be clipped.
    """

    rows, columns = returns.shape
    output = np.zeros((rows, columns), dtype=np.float64)
    clipped = np.zeros(columns, dtype=np.int64)
    worst = np.zeros(columns, dtype=np.float64)
    if rows == 0 or columns == 0:
        return output, clipped, worst
    buffer = np.empty(rows, dtype=np.float64)
    deviations = np.empty(rows, dtype=np.float64)
    for column in range(columns):
        count = 0
        for row in range(rows):
            value = returns[row, column]
            if np.isfinite(value):
                buffer[count] = value
                count += 1
        if count == 0:
            continue
        median = np.median(buffer[:count])
        for index in range(count):
            deviations[index] = abs(buffer[index] - median)
        sigma = np.median(deviations[:count]) * 1.4826
        if not np.isfinite(sigma) or sigma <= 1e-12:
            for row in range(rows):
                value = returns[row, column]
                output[row, column] = value if np.isfinite(value) else 0.0
            continue
        band = sigma_multiple * sigma
        upper = median + band
        lower = median - band
        for row in range(rows):
            value = returns[row, column]
            if not np.isfinite(value):
                output[row, column] = median
                continue
            if value > upper or value < lower:
                output[row, column] = upper if value > upper else lower
                clipped[column] += 1
                magnitude = abs(value)
                if magnitude > worst[column]:
                    worst[column] = magnitude
            else:
                output[row, column] = value
    return output, clipped, worst


@njit(_F2(_F2), cache=False, nogil=True)
def correlation_matrix_kernel(returns: np.ndarray) -> np.ndarray:
    """Pearson correlation across columns of an already aligned return matrix."""

    rows, columns = returns.shape
    output = np.zeros((columns, columns), dtype=np.float64)
    means = np.zeros(columns, dtype=np.float64)
    deviations = np.zeros(columns, dtype=np.float64)
    for column in range(columns):
        total = 0.0
        for row in range(rows):
            total += returns[row, column]
        mean = total / rows if rows > 0 else 0.0
        means[column] = mean
        variance = 0.0
        for row in range(rows):
            diff = returns[row, column] - mean
            variance += diff * diff
        deviations[column] = np.sqrt(variance)
    for left in range(columns):
        output[left, left] = 1.0
        for right in range(left + 1, columns):
            covariance = 0.0
            for row in range(rows):
                covariance += (returns[row, left] - means[left]) * (returns[row, right] - means[right])
            denominator = deviations[left] * deviations[right]
            value = covariance / denominator if denominator > 1e-18 else 0.0
            if not np.isfinite(value):
                value = 0.0
            elif value > 1.0:
                value = 1.0
            elif value < -1.0:
                value = -1.0
            output[left, right] = value
            output[right, left] = value
    return output


@njit(_F2(_F2), cache=False, nogil=True)
def corr_to_distance_kernel(correlation: np.ndarray) -> np.ndarray:
    """Lopez de Prado correlation distance sqrt(0.5 * (1 - rho))."""

    size = correlation.shape[0]
    output = np.zeros((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(size):
            if left == right:
                continue
            value = correlation[left, right]
            if not np.isfinite(value):
                value = 0.0
            distance = 0.5 * (1.0 - value)
            if distance < 0.0:
                distance = 0.0
            output[left, right] = np.sqrt(distance)
    return output


@njit(_F2(_F2), cache=False, nogil=True)
def euclidean_distance_kernel(features: np.ndarray) -> np.ndarray:
    rows, columns = features.shape
    output = np.zeros((rows, rows), dtype=np.float64)
    for left in range(rows):
        for right in range(left + 1, rows):
            total = 0.0
            for column in range(columns):
                diff = features[left, column] - features[right, column]
                total += diff * diff
            distance = np.sqrt(total)
            output[left, right] = distance
            output[right, left] = distance
    return output


@njit(_F2(_F2, int64), cache=False, nogil=True)
def pca_features_kernel(correlation: np.ndarray, components: int) -> np.ndarray:
    """Top-`components` eigenvector loadings scaled by sqrt(eigenvalue).

    Rows are products, columns are principal components.  This is the
    dimension-reduction feature lane: two products with similar loadings share
    the same systematic drivers.
    """

    size = correlation.shape[0]
    wanted = components
    if wanted < 1:
        wanted = 1
    if wanted > size:
        wanted = size
    output = np.zeros((size, wanted), dtype=np.float64)
    if size == 0:
        return output
    symmetric = np.zeros((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(size):
            value = 0.5 * (correlation[left, right] + correlation[right, left])
            if not np.isfinite(value):
                value = 0.0
            symmetric[left, right] = value
    values, vectors = np.linalg.eigh(symmetric)
    # eigh returns ascending eigenvalues; take the largest `wanted`.
    for component in range(wanted):
        source = size - 1 - component
        eigenvalue = values[source]
        if not np.isfinite(eigenvalue) or eigenvalue < 0.0:
            eigenvalue = 0.0
        scale = np.sqrt(eigenvalue)
        for row in range(size):
            loading = vectors[row, source]
            if not np.isfinite(loading):
                loading = 0.0
            output[row, component] = loading * scale
    return output


@njit(_F1(_F2), cache=False, nogil=True)
def correlation_eigenvalues_kernel(correlation: np.ndarray) -> np.ndarray:
    """Descending eigenvalues of the correlation matrix (for the K suggestion)."""

    size = correlation.shape[0]
    output = np.zeros(size, dtype=np.float64)
    if size == 0:
        return output
    symmetric = np.zeros((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(size):
            value = 0.5 * (correlation[left, right] + correlation[right, left])
            if not np.isfinite(value):
                value = 0.0
            symmetric[left, right] = value
    values = np.linalg.eigvalsh(symmetric)
    for index in range(size):
        output[index] = values[size - 1 - index]
    return output


@njit(_F2(_F2, int64), cache=False, nogil=True)
def agglomerative_linkage_kernel(distance: np.ndarray, method: int64) -> np.ndarray:
    """Lance-Williams agglomerative clustering in scipy linkage layout.

    Returns an (n-1, 4) matrix of [left_id, right_id, merge_distance, size].
    Cluster ids follow the scipy convention: originals are 0..n-1 and merge m
    creates id n+m.

    ponytail: naive O(n^3) nearest-pair scan; swap in nearest-neighbour chain if
    pools ever exceed a few thousand products.
    """

    size = distance.shape[0]
    merges = size - 1 if size > 0 else 0
    output = np.zeros((merges, 4), dtype=np.float64)
    if size < 2:
        return output
    working = np.zeros((size, size), dtype=np.float64)
    squared = method == LINKAGE_WARD
    for left in range(size):
        for right in range(size):
            value = distance[left, right]
            if not np.isfinite(value):
                value = 0.0
            working[left, right] = value * value if squared else value
    active = np.ones(size, dtype=np.uint8)
    counts = np.ones(size, dtype=np.float64)
    cluster_id = np.empty(size, dtype=np.int64)
    for index in range(size):
        cluster_id[index] = index
    for step in range(merges):
        best_left = -1
        best_right = -1
        best_value = np.inf
        for left in range(size):
            if active[left] == 0:
                continue
            for right in range(left + 1, size):
                if active[right] == 0:
                    continue
                value = working[left, right]
                if value < best_value:
                    best_value = value
                    best_left = left
                    best_right = right
        if best_left < 0:
            break
        size_left = counts[best_left]
        size_right = counts[best_right]
        merged = size_left + size_right
        pair_distance = working[best_left, best_right]
        for other in range(size):
            if active[other] == 0 or other == best_left or other == best_right:
                continue
            left_value = working[best_left, other]
            right_value = working[best_right, other]
            if method == LINKAGE_COMPLETE:
                updated = left_value if left_value > right_value else right_value
            elif method == LINKAGE_WARD:
                size_other = counts[other]
                total = merged + size_other
                updated = (
                    (size_left + size_other) * left_value
                    + (size_right + size_other) * right_value
                    - size_other * pair_distance
                ) / total
                if updated < 0.0:
                    updated = 0.0
            else:
                updated = (size_left * left_value + size_right * right_value) / merged
            working[best_left, other] = updated
            working[other, best_left] = updated
        output[step, 0] = float(cluster_id[best_left])
        output[step, 1] = float(cluster_id[best_right])
        output[step, 2] = np.sqrt(pair_distance) if squared else pair_distance
        output[step, 3] = merged
        counts[best_left] = merged
        cluster_id[best_left] = size + step
        active[best_right] = 0
    return output


@njit(_I1(_F2, int64, int64), cache=False, nogil=True)
def cut_linkage_kernel(linkage: np.ndarray, observations: int64, clusters: int64) -> np.ndarray:
    """Cut a linkage tree into `clusters` groups and return compacted labels."""

    labels = np.full(observations, -1, dtype=np.int64)
    if observations <= 0:
        return labels
    wanted = clusters
    if wanted < 1:
        wanted = 1
    if wanted > observations:
        wanted = observations
    parent = np.empty(observations, dtype=np.int64)
    for index in range(observations):
        parent[index] = index
    representative = np.empty(2 * observations, dtype=np.int64)
    for index in range(observations):
        representative[index] = index
    for index in range(observations, 2 * observations):
        representative[index] = -1
    applied = observations - wanted
    total_merges = linkage.shape[0]
    if applied > total_merges:
        applied = total_merges
    for step in range(applied):
        left_id = int(linkage[step, 0])
        right_id = int(linkage[step, 1])
        left_root = representative[left_id]
        right_root = representative[right_id]
        if left_root < 0 or right_root < 0:
            continue
        while parent[left_root] != left_root:
            left_root = parent[left_root]
        while parent[right_root] != right_root:
            right_root = parent[right_root]
        if left_root != right_root:
            parent[right_root] = left_root
        representative[observations + step] = left_root
    remap = np.full(observations, -1, dtype=np.int64)
    next_label = 0
    for index in range(observations):
        root = index
        while parent[root] != root:
            root = parent[root]
        if remap[root] < 0:
            remap[root] = next_label
            next_label += 1
        labels[index] = remap[root]
    return labels


@njit(_I1(_F2, int64, int64, int64), cache=False, nogil=True)
def kmeans_kernel(features: np.ndarray, clusters: int64, seed: int64, max_iter: int64) -> np.ndarray:
    """Lloyd k-means with deterministic seeded k-means++ initialisation."""

    rows, columns = features.shape
    labels = np.zeros(rows, dtype=np.int64)
    wanted = clusters
    if wanted < 1:
        wanted = 1
    if wanted > rows:
        wanted = rows
    if rows == 0 or columns == 0:
        return labels
    np.random.seed(seed)
    centroids = np.zeros((wanted, columns), dtype=np.float64)
    first = int(np.random.random() * rows)
    if first >= rows:
        first = rows - 1
    for column in range(columns):
        centroids[0, column] = features[first, column]
    closest = np.full(rows, np.inf, dtype=np.float64)
    for center in range(1, wanted):
        total = 0.0
        for row in range(rows):
            distance = 0.0
            for column in range(columns):
                diff = features[row, column] - centroids[center - 1, column]
                distance += diff * diff
            if distance < closest[row]:
                closest[row] = distance
            total += closest[row]
        picked = rows - 1
        if total > 0.0:
            threshold = np.random.random() * total
            running = 0.0
            for row in range(rows):
                running += closest[row]
                if running >= threshold:
                    picked = row
                    break
        else:
            picked = center % rows
        for column in range(columns):
            centroids[center, column] = features[picked, column]
    counts = np.zeros(wanted, dtype=np.int64)
    for _ in range(max_iter):
        changed = 0
        for row in range(rows):
            best = 0
            best_distance = np.inf
            for center in range(wanted):
                distance = 0.0
                for column in range(columns):
                    diff = features[row, column] - centroids[center, column]
                    distance += diff * diff
                if distance < best_distance:
                    best_distance = distance
                    best = center
            if labels[row] != best:
                changed += 1
            labels[row] = best
        for center in range(wanted):
            counts[center] = 0
            for column in range(columns):
                centroids[center, column] = 0.0
        for row in range(rows):
            center = labels[row]
            counts[center] += 1
            for column in range(columns):
                centroids[center, column] += features[row, column]
        for center in range(wanted):
            if counts[center] > 0:
                for column in range(columns):
                    centroids[center, column] /= counts[center]
            else:
                # Re-seed an empty cluster on the point farthest from its centroid.
                worst = 0
                worst_distance = -1.0
                for row in range(rows):
                    owner = labels[row]
                    distance = 0.0
                    for column in range(columns):
                        diff = features[row, column] - centroids[owner, column]
                        distance += diff * diff
                    if distance > worst_distance:
                        worst_distance = distance
                        worst = row
                for column in range(columns):
                    centroids[center, column] = features[worst, column]
        if changed == 0:
            break
    return labels


@njit(types.Tuple((_I1, _I1))(_F2, int64, int64, int64), cache=False, nogil=True)
def kmedoids_kernel(distance: np.ndarray, clusters: int64, seed: int64, max_iter: int64):
    """Voronoi-iteration k-medoids; medoids are real products, not virtual points."""

    size = distance.shape[0]
    labels = np.zeros(size, dtype=np.int64)
    wanted = clusters
    if wanted < 1:
        wanted = 1
    if wanted > size:
        wanted = size
    medoids = np.zeros(wanted, dtype=np.int64)
    if size == 0:
        return labels, medoids
    np.random.seed(seed)
    first = int(np.random.random() * size)
    if first >= size:
        first = size - 1
    medoids[0] = first
    closest = np.full(size, np.inf, dtype=np.float64)
    for center in range(1, wanted):
        total = 0.0
        for row in range(size):
            value = distance[row, medoids[center - 1]]
            squared = value * value
            if squared < closest[row]:
                closest[row] = squared
            total += closest[row]
        picked = center % size
        if total > 0.0:
            threshold = np.random.random() * total
            running = 0.0
            for row in range(size):
                running += closest[row]
                if running >= threshold:
                    picked = row
                    break
        medoids[center] = picked
    for _ in range(max_iter):
        changed = 0
        for row in range(size):
            best = 0
            best_distance = np.inf
            for center in range(wanted):
                value = distance[row, medoids[center]]
                if value < best_distance:
                    best_distance = value
                    best = center
            if labels[row] != best:
                changed += 1
            labels[row] = best
        for center in range(wanted):
            best_member = -1
            best_cost = np.inf
            for candidate in range(size):
                if labels[candidate] != center:
                    continue
                cost = 0.0
                for member in range(size):
                    if labels[member] != center:
                        continue
                    cost += distance[candidate, member]
                if cost < best_cost:
                    best_cost = cost
                    best_member = candidate
            if best_member >= 0:
                medoids[center] = best_member
        if changed == 0:
            break
    return labels, medoids


@njit(_F2(_F2, int64), cache=False, nogil=True)
def denoise_correlation_kernel(correlation: np.ndarray, observations: int64) -> np.ndarray:
    """Marchenko-Pastur eigenvalue clipping: noise modes collapse to one level.

    For N products over T observations the eigenvalues of a pure-noise
    correlation matrix stay below (1 + sqrt(N/T))^2.  Everything at or under
    that bound says nothing about which products actually move together, so it
    is flattened to a single average level (trace preserved) and only the
    systematic modes survive to drive the distance matrix.  This is the standard
    pre-step for correlation-based asset clustering; with 40 ETFs over one year
    of daily data roughly half the spectrum is noise.
    """

    size = correlation.shape[0]
    output = np.zeros((size, size), dtype=np.float64)
    if size == 0:
        return output
    for left in range(size):
        for right in range(size):
            value = 0.5 * (correlation[left, right] + correlation[right, left])
            if not np.isfinite(value):
                value = 0.0
            output[left, right] = value
    # q >= 1 leaves the sample correlation rank-deficient: no signal/noise split
    # is identifiable there, so the raw matrix is the honest answer.
    if size < 2 or observations <= size:
        return output
    values, vectors = np.linalg.eigh(output)
    bound = (1.0 + np.sqrt(size / observations)) ** 2
    noise_total = 0.0
    noise_count = 0
    for index in range(size):
        if values[index] <= bound:
            noise_total += values[index]
            noise_count += 1
    # All-signal or all-noise: clipping would either change nothing or erase the
    # whole structure into an identity matrix.  Leave the input alone.
    if noise_count == 0 or noise_count == size:
        return output
    flat = noise_total / noise_count
    filtered = np.zeros(size, dtype=np.float64)
    for index in range(size):
        filtered[index] = flat if values[index] <= bound else values[index]
    rebuilt = np.zeros((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(left, size):
            total = 0.0
            for mode in range(size):
                total += vectors[left, mode] * filtered[mode] * vectors[right, mode]
            rebuilt[left, right] = total
            rebuilt[right, left] = total
    # Clipping breaks the unit diagonal; rescale back to a correlation matrix.
    scale = np.zeros(size, dtype=np.float64)
    for index in range(size):
        diagonal = rebuilt[index, index]
        scale[index] = np.sqrt(diagonal) if diagonal > 0.0 else 1.0
    for left in range(size):
        for right in range(size):
            value = rebuilt[left, right] / (scale[left] * scale[right])
            if value > 1.0:
                value = 1.0
            elif value < -1.0:
                value = -1.0
            output[left, right] = value
        output[left, left] = 1.0
    return output


@njit(_I1(_F2, int64, int64, int64), cache=False, nogil=True)
def spectral_labels_kernel(
    distance: np.ndarray, clusters: int64, seed: int64, max_iter: int64
) -> np.ndarray:
    """Normalised-cut spectral clustering (Ng-Jordan-Weiss) on the distance graph.

    A Gaussian kernel turns distances into graph weights, the symmetric
    normalised affinity is embedded by its leading eigenvectors, and k-means
    cuts that embedding.  Unlike k-means/k-medoids in the raw space this makes
    no round-cluster assumption: a chain of products drifting from 沪深300
    towards 创业板 stays one connected class instead of being split at an
    arbitrary radius.
    """

    size = distance.shape[0]
    labels = np.zeros(size, dtype=np.int64)
    wanted = clusters
    if wanted < 1:
        wanted = 1
    if wanted > size:
        wanted = size
    if size < 3 or wanted < 2:
        return labels
    # Kernel width from the pool's own scale, so the graph neither saturates nor
    # disconnects when the distance unit changes.
    total = 0.0
    pairs = 0
    for left in range(size):
        for right in range(size):
            if left == right:
                continue
            value = distance[left, right]
            if np.isfinite(value):
                total += value
                pairs += 1
    sigma = total / pairs if pairs > 0 else 0.0
    if sigma <= 0.0:
        return labels
    scale = 2.0 * sigma * sigma
    weights = np.zeros((size, size), dtype=np.float64)
    degree = np.zeros(size, dtype=np.float64)
    for left in range(size):
        for right in range(size):
            if left == right:
                continue
            value = distance[left, right]
            if not np.isfinite(value):
                continue
            weight = np.exp(-(value * value) / scale)
            weights[left, right] = weight
            degree[left] += weight
    for index in range(size):
        # An isolated node would divide by zero; park it at unit degree so it
        # embeds near the origin and joins whichever class claims it.
        if degree[index] <= 0.0:
            degree[index] = 1.0
        degree[index] = 1.0 / np.sqrt(degree[index])
    normalized = np.zeros((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(size):
            normalized[left, right] = weights[left, right] * degree[left] * degree[right]
    _values, vectors = np.linalg.eigh(normalized)
    embedding = np.zeros((size, wanted), dtype=np.float64)
    for component in range(wanted):
        source = size - 1 - component
        for row in range(size):
            loading = vectors[row, source]
            if not np.isfinite(loading):
                loading = 0.0
            embedding[row, component] = loading
    for row in range(size):
        norm = 0.0
        for component in range(wanted):
            norm += embedding[row, component] * embedding[row, component]
        norm = np.sqrt(norm)
        if norm > 0.0:
            for component in range(wanted):
                embedding[row, component] /= norm
    return kmeans_kernel(embedding, wanted, seed, max_iter)


@njit(types.Tuple((_I1, _F2))(_F2, int64, int64, int64), cache=False, nogil=True)
def gmm_kernel(features: np.ndarray, clusters: int64, seed: int64, max_iter: int64):
    """Diagonal-covariance Gaussian mixture by EM, started from k-means.

    Returns hard labels plus the posterior membership matrix.  k-means forces
    every class to be a ball of the same radius; a mixture lets a tight 货币类
    cluster and a sprawling 主题类 cluster coexist, and the posteriors say how
    confidently a product belongs rather than only where it landed.

    ponytail: diagonal covariance only; full covariance needs a per-component
    Cholesky and only pays off once features outnumber products per class.
    """

    rows, columns = features.shape
    wanted = clusters
    if wanted < 1:
        wanted = 1
    if wanted > rows:
        wanted = rows
    labels = np.zeros(rows, dtype=np.int64)
    responsibility = np.zeros((rows, wanted), dtype=np.float64)
    if rows == 0 or columns == 0 or wanted < 2:
        for row in range(rows):
            responsibility[row, 0] = 1.0
        return labels, responsibility
    labels = kmeans_kernel(features, wanted, seed, max_iter)

    # Variance floor keeps a degenerate component (one member, or duplicate
    # products) from collapsing to a delta and swallowing the likelihood.
    floor = 0.0
    for column in range(columns):
        mean = 0.0
        for row in range(rows):
            mean += features[row, column]
        mean /= rows
        spread = 0.0
        for row in range(rows):
            diff = features[row, column] - mean
            spread += diff * diff
        floor += spread / rows
    floor = floor / columns * 1e-6
    if floor <= 0.0 or not np.isfinite(floor):
        floor = 1e-9

    weights = np.zeros(wanted, dtype=np.float64)
    means = np.zeros((wanted, columns), dtype=np.float64)
    variances = np.zeros((wanted, columns), dtype=np.float64)
    counts = np.zeros(wanted, dtype=np.int64)
    for row in range(rows):
        component = labels[row]
        counts[component] += 1
        for column in range(columns):
            means[component, column] += features[row, column]
    for component in range(wanted):
        if counts[component] > 0:
            for column in range(columns):
                means[component, column] /= counts[component]
    for row in range(rows):
        component = labels[row]
        for column in range(columns):
            diff = features[row, column] - means[component, column]
            variances[component, column] += diff * diff
    weight_total = 0.0
    for component in range(wanted):
        if counts[component] > 0:
            for column in range(columns):
                value = variances[component, column] / counts[component]
                variances[component, column] = value if value > floor else floor
            weights[component] = counts[component] / rows
        else:
            for column in range(columns):
                variances[component, column] = floor
            # A component k-means left empty is kept alive at one phantom point
            # so EM can still claim it back instead of dividing by zero.
            weights[component] = 1.0 / rows
        weight_total += weights[component]
    for component in range(wanted):
        weights[component] /= weight_total

    log_two_pi = np.log(2.0 * np.pi)
    scores = np.zeros(wanted, dtype=np.float64)
    previous = 0.0
    for iteration in range(max_iter):
        likelihood = 0.0
        for row in range(rows):
            best = -np.inf
            for component in range(wanted):
                value = np.log(weights[component])
                for column in range(columns):
                    variance = variances[component, column]
                    diff = features[row, column] - means[component, column]
                    value -= 0.5 * (log_two_pi + np.log(variance) + diff * diff / variance)
                scores[component] = value
                if value > best:
                    best = value
            # log-sum-exp: the raw densities underflow long before the ratios do.
            total = 0.0
            for component in range(wanted):
                scores[component] = np.exp(scores[component] - best)
                total += scores[component]
            if total > 0.0:
                for component in range(wanted):
                    responsibility[row, component] = scores[component] / total
                likelihood += best + np.log(total)
            else:
                for component in range(wanted):
                    responsibility[row, component] = 1.0 / wanted
        for component in range(wanted):
            mass = 0.0
            for row in range(rows):
                mass += responsibility[row, component]
            if mass <= 0.0:
                weights[component] = 1.0 / rows
                continue
            weights[component] = mass / rows
            for column in range(columns):
                mean = 0.0
                for row in range(rows):
                    mean += responsibility[row, component] * features[row, column]
                mean /= mass
                means[component, column] = mean
                spread = 0.0
                for row in range(rows):
                    diff = features[row, column] - mean
                    spread += responsibility[row, component] * diff * diff
                spread /= mass
                variances[component, column] = spread if spread > floor else floor
        weight_total = 0.0
        for component in range(wanted):
            weight_total += weights[component]
        if weight_total > 0.0:
            for component in range(wanted):
                weights[component] /= weight_total
        if iteration > 0 and likelihood - previous <= 1e-9 * (np.abs(previous) + 1.0):
            break
        previous = likelihood

    for row in range(rows):
        best_component = 0
        best_value = -1.0
        for component in range(wanted):
            value = responsibility[row, component]
            if value > best_value:
                best_value = value
                best_component = component
        labels[row] = best_component
    return labels, responsibility


@njit(_F2(_F2, _I1, int64), cache=False, nogil=True)
def affinity_from_distance_kernel(distance: np.ndarray, labels: np.ndarray, clusters: int64) -> np.ndarray:
    """Affinity[i, k] = -mean distance from i to the members of cluster k.

    Higher is better.  Self-distance is excluded so a product is not rewarded
    simply for sitting in a small cluster.
    """

    size = distance.shape[0]
    output = np.full((size, clusters), -np.inf, dtype=np.float64)
    if size == 0 or clusters <= 0:
        return output
    counts = np.zeros(clusters, dtype=np.int64)
    for index in range(size):
        label = labels[index]
        if 0 <= label < clusters:
            counts[label] += 1
    global_mean = 0.0
    pairs = 0
    for left in range(size):
        for right in range(size):
            if left != right:
                global_mean += distance[left, right]
                pairs += 1
    if pairs > 0:
        global_mean /= pairs
    for row in range(size):
        for cluster in range(clusters):
            total = 0.0
            members = 0
            for other in range(size):
                if other == row:
                    continue
                if labels[other] != cluster:
                    continue
                total += distance[row, other]
                members += 1
            if members == 0:
                # An empty (or singleton-self) cluster is scored at the pool
                # average so it stays reachable without being attractive.
                output[row, cluster] = -global_mean
            else:
                output[row, cluster] = -(total / members)
    return output


@njit(float64(_F2), cache=False, nogil=True)
def mean_offdiagonal_kernel(distance: np.ndarray) -> float:
    """Average pairwise distance of the pool; the reference for 'more similar than random'."""

    size = distance.shape[0]
    total = 0.0
    pairs = 0
    for left in range(size):
        for right in range(size):
            if left == right:
                continue
            value = distance[left, right]
            if np.isfinite(value):
                total += value
                pairs += 1
    return total / pairs if pairs > 0 else np.nan


@njit(_I1(_F2, _I1, int64, int64, int64, float64), cache=False, nogil=True)
def capacity_assign_kernel(
    affinity: np.ndarray,
    labels: np.ndarray,
    size_min: int64,
    size_max: int64,
    allow_unassigned: int64,
    min_affinity: float64,
) -> np.ndarray:
    """Trim each cluster to its most representative members under size bounds.

    This is a *selection* over the clustering, not a re-partition: a cluster
    keeps at most `size_max` members ranked by affinity, the overflow is parked,
    and a cluster below `size_min` pulls back the nearest parked products.  A
    global re-assignment would push an overflowing equity cluster's leftovers
    into whatever cluster still had room, which silently corrupts the classes.
    """

    rows, clusters = affinity.shape
    output = np.full(rows, -1, dtype=np.int64)
    if rows == 0 or clusters <= 0:
        return output
    upper = size_max
    if upper < 1:
        upper = rows
    lower = size_min
    if lower < 0:
        lower = 0
    counts = np.zeros(clusters, dtype=np.int64)

    for cluster in range(clusters):
        while counts[cluster] < upper:
            best = -1
            best_value = -np.inf
            for row in range(rows):
                if labels[row] != cluster or output[row] >= 0:
                    continue
                value = affinity[row, cluster]
                if np.isfinite(value) and value > best_value:
                    best_value = value
                    best = row
            if best < 0:
                break
            output[best] = cluster
            counts[cluster] += 1

    floor = min_affinity if np.isfinite(min_affinity) else -np.inf
    for cluster in range(clusters):
        while counts[cluster] < lower:
            best = -1
            best_value = -np.inf
            for row in range(rows):
                if output[row] >= 0:
                    continue
                value = affinity[row, cluster]
                if np.isfinite(value) and value >= floor and value > best_value:
                    best_value = value
                    best = row
            if best < 0:
                # Nothing left that is genuinely similar: report a short class
                # rather than manufacture one.
                break
            output[best] = cluster
            counts[cluster] += 1

    if allow_unassigned == 0:
        for row in range(rows):
            if output[row] >= 0:
                continue
            best = -1
            best_value = -np.inf
            for cluster in range(clusters):
                if counts[cluster] >= upper:
                    continue
                value = affinity[row, cluster]
                if np.isfinite(value) and value > best_value:
                    best_value = value
                    best = cluster
            if best < 0:
                for cluster in range(clusters):
                    value = affinity[row, cluster]
                    if np.isfinite(value) and value > best_value:
                        best_value = value
                        best = cluster
            if best >= 0:
                output[row] = best
                counts[best] += 1
    return output


@njit(types.Tuple((float64, _F1))(_F2, _I1, int64), cache=False, nogil=True)
def silhouette_kernel(distance: np.ndarray, labels: np.ndarray, clusters: int64):
    """Overall and per-cluster mean silhouette width; unassigned rows are skipped."""

    size = distance.shape[0]
    per_cluster = np.full(clusters if clusters > 0 else 0, np.nan, dtype=np.float64)
    if size == 0 or clusters <= 0:
        return np.nan, per_cluster
    totals = np.zeros(clusters, dtype=np.float64)
    counts = np.zeros(clusters, dtype=np.int64)
    overall = 0.0
    scored = 0
    for row in range(size):
        own = labels[row]
        if own < 0 or own >= clusters:
            continue
        own_total = 0.0
        own_members = 0
        best_other = np.inf
        for cluster in range(clusters):
            total = 0.0
            members = 0
            for other in range(size):
                if other == row or labels[other] != cluster:
                    continue
                total += distance[row, other]
                members += 1
            if cluster == own:
                own_total = total
                own_members = members
            elif members > 0:
                mean = total / members
                if mean < best_other:
                    best_other = mean
        if not np.isfinite(best_other):
            # Only one populated cluster: there is nothing to separate from.
            continue
        if own_members == 0:
            # Standard convention: a lone member scores 0 rather than being
            # dropped.  Skipping it would reward splitting a pool into
            # singletons and quietly bias the automatic K suggestion upward.
            score = 0.0
        else:
            cohesion = own_total / own_members
            denominator = cohesion if cohesion > best_other else best_other
            score = 0.0 if denominator <= 1e-18 else (best_other - cohesion) / denominator
        totals[own] += score
        counts[own] += 1
        overall += score
        scored += 1
    for cluster in range(clusters):
        if counts[cluster] > 0:
            per_cluster[cluster] = totals[cluster] / counts[cluster]
    return (overall / scored if scored > 0 else np.nan), per_cluster


@njit(_F1(_F2, _I1, int64), cache=False, nogil=True)
def intra_class_mean_corr_kernel(correlation: np.ndarray, labels: np.ndarray, clusters: int64) -> np.ndarray:
    """Average off-diagonal correlation inside each cluster."""

    size = correlation.shape[0]
    output = np.full(clusters if clusters > 0 else 0, np.nan, dtype=np.float64)
    for cluster in range(clusters):
        total = 0.0
        pairs = 0
        for left in range(size):
            if labels[left] != cluster:
                continue
            for right in range(left + 1, size):
                if labels[right] != cluster:
                    continue
                value = correlation[left, right]
                if np.isfinite(value):
                    total += value
                    pairs += 1
        if pairs > 0:
            output[cluster] = total / pairs
    return output


@njit(_F2(_F2, _I1, int64), cache=False, nogil=True)
def cross_class_corr_kernel(correlation: np.ndarray, labels: np.ndarray, clusters: int64) -> np.ndarray:
    """Mean pairwise correlation between every pair of clusters."""

    size = correlation.shape[0]
    output = np.full((clusters, clusters), np.nan, dtype=np.float64)
    for left_cluster in range(clusters):
        for right_cluster in range(clusters):
            total = 0.0
            pairs = 0
            for left in range(size):
                if labels[left] != left_cluster:
                    continue
                for right in range(size):
                    if labels[right] != right_cluster:
                        continue
                    if left_cluster == right_cluster and left == right:
                        continue
                    value = correlation[left, right]
                    if np.isfinite(value):
                        total += value
                        pairs += 1
            if pairs > 0:
                output[left_cluster, right_cluster] = total / pairs
    return output


@njit(_F1(_F2, _I1, _F2, int64, int64), cache=False, nogil=True)
def intra_class_weights_kernel(
    returns: np.ndarray,
    labels: np.ndarray,
    affinity: np.ndarray,
    clusters: int64,
    mode: int64,
) -> np.ndarray:
    """Per-product weight in percent, normalised to 100 inside each cluster."""

    periods, assets = returns.shape
    output = np.zeros(assets, dtype=np.float64)
    if assets == 0 or clusters <= 0:
        return output
    scores = np.zeros(assets, dtype=np.float64)
    for asset in range(assets):
        if labels[asset] < 0:
            continue
        if mode == WEIGHT_EQUAL:
            scores[asset] = 1.0
            continue
        if mode == WEIGHT_AFFINITY:
            value = affinity[asset, labels[asset]]
            # Affinity is a negative mean distance; shift it into a positive score.
            scores[asset] = 1.0 / (1.0 + abs(value)) if np.isfinite(value) else 0.0
            continue
        mean = 0.0
        for period in range(periods):
            mean += returns[period, asset]
        mean = mean / periods if periods > 0 else 0.0
        variance = 0.0
        for period in range(periods):
            diff = returns[period, asset] - mean
            variance += diff * diff
        variance = variance / (periods - 1) if periods > 1 else 0.0
        if not np.isfinite(variance) or variance <= 1e-18:
            scores[asset] = 0.0
            continue
        if mode == WEIGHT_INV_VAR:
            scores[asset] = 1.0 / variance
        else:
            scores[asset] = 1.0 / np.sqrt(variance)
    for cluster in range(clusters):
        total = 0.0
        members = 0
        for asset in range(assets):
            if labels[asset] == cluster:
                total += scores[asset]
                members += 1
        if members == 0:
            continue
        if total <= 1e-18:
            # Degenerate scores (zero variance everywhere) fall back to equal weight.
            for asset in range(assets):
                if labels[asset] == cluster:
                    output[asset] = 100.0 / members
            continue
        for asset in range(assets):
            if labels[asset] == cluster:
                output[asset] = 100.0 * scores[asset] / total
    return output


@njit(types.Tuple((_F1, _F1))(_F1, _F1, _I1, int64), cache=False, nogil=True)
def apply_weight_caps_kernel(
    weights: np.ndarray,
    caps: np.ndarray,
    labels: np.ndarray,
    clusters: int64,
):
    """Re-solve intra-class weights so no product exceeds its product-pool cap.

    `caps` is a per-product upper bound in percent (100 means unrestricted).
    Capping the *intra-class* weight is the conservative way to honour a
    portfolio-level restriction here: the final portfolio weight is
    ``class_weight * intra_weight`` and ``class_weight <= 1``, so an intra-class
    weight within the cap can never breach it downstream.

    Returns the adjusted weights and, per cluster, the largest class weight that
    keeps every member inside its cap (percent).  When the caps inside a class
    sum to less than 100 the class cannot be filled without breaching one of
    them, so weights are set proportional to the caps and the reported capacity
    is that sum -- the class stays usable as long as SAA gives it no more.
    """

    size = weights.shape[0]
    output = np.zeros(size, dtype=np.float64)
    capacity = np.full(clusters if clusters > 0 else 0, np.nan, dtype=np.float64)
    for index in range(size):
        output[index] = weights[index]
    if size == 0 or clusters <= 0:
        return output, capacity

    frozen = np.zeros(size, dtype=np.uint8)
    for cluster in range(clusters):
        members = 0
        total_cap = 0.0
        for index in range(size):
            if labels[index] != cluster:
                continue
            members += 1
            cap = caps[index]
            if not np.isfinite(cap) or cap < 0.0:
                cap = 100.0
            if cap > 100.0:
                cap = 100.0
            total_cap += cap
        if members == 0:
            continue
        capacity[cluster] = total_cap if total_cap < 100.0 else 100.0
        if total_cap <= 1e-12:
            for index in range(size):
                if labels[index] == cluster:
                    output[index] = 0.0
            continue
        if total_cap <= 100.0:
            # Cannot reach 100 without breaching a cap: stay proportional to the
            # caps, which keeps every product inside its limit for any class
            # weight up to `total_cap`.
            for index in range(size):
                if labels[index] != cluster:
                    continue
                cap = caps[index]
                if not np.isfinite(cap) or cap < 0.0 or cap > 100.0:
                    cap = 100.0
                output[index] = 100.0 * cap / total_cap
            continue

        for index in range(size):
            if labels[index] == cluster:
                frozen[index] = 0
        remaining = 100.0
        for _ in range(members + 1):
            score_total = 0.0
            active = 0
            for index in range(size):
                if labels[index] != cluster or frozen[index] == 1:
                    continue
                active += 1
                value = weights[index]
                if np.isfinite(value) and value > 0.0:
                    score_total += value
            if active == 0:
                break
            for index in range(size):
                if labels[index] != cluster or frozen[index] == 1:
                    continue
                if score_total > 1e-12:
                    value = weights[index]
                    if not np.isfinite(value) or value < 0.0:
                        value = 0.0
                    output[index] = remaining * value / score_total
                else:
                    output[index] = remaining / active
            violations = 0
            for index in range(size):
                if labels[index] != cluster or frozen[index] == 1:
                    continue
                cap = caps[index]
                if not np.isfinite(cap) or cap < 0.0 or cap > 100.0:
                    cap = 100.0
                if output[index] > cap + 1e-12:
                    output[index] = cap
                    frozen[index] = 1
                    remaining -= cap
                    violations += 1
            if violations == 0:
                break
            if remaining < 0.0:
                remaining = 0.0
    return output, capacity


@njit(_I1(_I1, int64, int64), cache=False, nogil=True)
def allocate_block_clusters_kernel(
    block_sizes: np.ndarray, total_clusters: int64, size_min: int64
) -> np.ndarray:
    """Split a requested class count across contract-taxonomy blocks.

    Every non-empty block keeps at least one class -- the taxonomy is a hard
    partition, so a block can never be merged away to hit K.  The remaining
    classes go to whichever block currently has the largest products-per-class
    load (highest averages), which is deterministic and needs no tie-break.
    """

    blocks = block_sizes.shape[0]
    output = np.zeros(blocks, dtype=np.int64)
    if blocks == 0:
        return output
    lower = size_min if size_min > 0 else 1
    capacity = np.zeros(blocks, dtype=np.int64)
    total_capacity = 0
    for block in range(blocks):
        size = block_sizes[block]
        if size <= 0:
            continue
        room = size // lower
        if room < 1:
            # Too small to satisfy size_min on its own: it still forms one
            # (short) class rather than being dissolved into another block.
            room = 1
        if room > size:
            room = size
        capacity[block] = room
        total_capacity += room

    target = total_clusters
    if target < 1:
        target = blocks
    if target > total_capacity:
        target = total_capacity

    assigned = 0
    for block in range(blocks):
        if capacity[block] > 0:
            output[block] = 1
            assigned += 1
    remaining = target - assigned
    while remaining > 0:
        best = -1
        best_score = -1.0
        for block in range(blocks):
            if output[block] >= capacity[block]:
                continue
            score = block_sizes[block] / (output[block] + 1)
            if score > best_score:
                best_score = score
                best = block
        if best < 0:
            break
        output[best] += 1
        remaining -= 1
    return output


@njit(_U1(_F2), cache=False, nogil=True)
def finite_rows_kernel(values: np.ndarray) -> np.ndarray:
    """Row mask used before any covariance work; keeps serialization honest."""

    rows, columns = values.shape
    output = np.zeros(rows, dtype=np.uint8)
    for row in range(rows):
        ok = 1
        for column in range(columns):
            if not np.isfinite(values[row, column]):
                ok = 0
                break
        output[row] = ok
    return output


_PUBLIC_KERNELS = (
    robust_standardize_kernel,
    allocate_block_clusters_kernel,
    winsorize_returns_kernel,
    correlation_matrix_kernel,
    corr_to_distance_kernel,
    euclidean_distance_kernel,
    pca_features_kernel,
    correlation_eigenvalues_kernel,
    agglomerative_linkage_kernel,
    cut_linkage_kernel,
    kmeans_kernel,
    kmedoids_kernel,
    denoise_correlation_kernel,
    spectral_labels_kernel,
    gmm_kernel,
    affinity_from_distance_kernel,
    capacity_assign_kernel,
    mean_offdiagonal_kernel,
    silhouette_kernel,
    intra_class_mean_corr_kernel,
    cross_class_corr_kernel,
    intra_class_weights_kernel,
    apply_weight_caps_kernel,
    finite_rows_kernel,
)

for _kernel in _PUBLIC_KERNELS:
    _kernel.disable_compile()


def auto_class_execution_audit() -> dict[str, object]:
    signatures = {
        kernel.py_func.__name__: [str(signature) for signature in kernel.signatures]
        for kernel in _PUBLIC_KERNELS
    }
    material = "|".join(
        [AUTO_CLASS_ENGINE_VERSION, AUTO_CLASS_KERNEL_VERSION]
        + [f"{name}:{','.join(values)}" for name, values in sorted(signatures.items())]
    )
    return validate_execution_audit({
        "engine": AUTO_CLASS_ENGINE_VERSION,
        "backend": "numba_njit_fixed_signature",
        "kernel_version": AUTO_CLASS_KERNEL_VERSION,
        "kernel_signatures": signatures,
        "kernel_fingerprint": hashlib.sha256(material.encode("utf-8")).hexdigest(),
        "nopython": all(
            bool(kernel.nopython_signatures)
            and all(
                not compilation.objectmode
                for compilation in kernel.overloads.values()
            )
            for kernel in _PUBLIC_KERNELS
        ),
        "object_mode": 0,
        "python_fallback": 0,
        "request_time_compilation": 0,
    })


def warm_auto_class_numba_kernels() -> dict[str, object]:
    """Compile every auto-classification signature before readiness is reported."""

    returns = np.ascontiguousarray(
        np.array(
            [
                [0.01, 0.012, -0.004, -0.003],
                [-0.005, -0.006, 0.002, 0.003],
                [0.008, 0.009, -0.001, -0.002],
                [0.002, 0.003, 0.004, 0.005],
            ],
            dtype=np.float64,
        )
    )
    finite_rows_kernel(returns)
    allocate_block_clusters_kernel(np.ascontiguousarray(np.array([3, 1], dtype=np.int64)), 3, 1)
    clean, _clipped, _worst = winsorize_returns_kernel(returns, 25.0)
    correlation = correlation_matrix_kernel(np.ascontiguousarray(clean))
    distance = corr_to_distance_kernel(correlation)
    features = robust_standardize_kernel(np.ascontiguousarray(correlation))
    euclidean_distance_kernel(features)
    pca_features_kernel(np.ascontiguousarray(correlation), 2)
    correlation_eigenvalues_kernel(np.ascontiguousarray(correlation))
    for method in (LINKAGE_AVERAGE, LINKAGE_COMPLETE, LINKAGE_WARD):
        linkage = agglomerative_linkage_kernel(np.ascontiguousarray(distance), method)
        labels = cut_linkage_kernel(np.ascontiguousarray(linkage), distance.shape[0], 2)
    kmeans_kernel(features, 2, 7, 32)
    kmedoids_kernel(np.ascontiguousarray(distance), 2, 7, 32)
    # T well above N so the warmup walks the eigen-clipping branch, not the guard.
    denoised = denoise_correlation_kernel(np.ascontiguousarray(correlation), 250)
    spectral_labels_kernel(np.ascontiguousarray(corr_to_distance_kernel(denoised)), 2, 7, 32)
    gmm_kernel(features, 2, 7, 32)
    affinity = affinity_from_distance_kernel(np.ascontiguousarray(distance), labels, 2)
    floor = mean_offdiagonal_kernel(np.ascontiguousarray(distance))
    assigned = capacity_assign_kernel(np.ascontiguousarray(affinity), labels, 1, 3, 1, -floor)
    silhouette_kernel(np.ascontiguousarray(distance), assigned, 2)
    intra_class_mean_corr_kernel(np.ascontiguousarray(correlation), assigned, 2)
    cross_class_corr_kernel(np.ascontiguousarray(correlation), assigned, 2)
    for mode in (WEIGHT_EQUAL, WEIGHT_INV_VOL, WEIGHT_INV_VAR, WEIGHT_AFFINITY):
        intra_class_weights_kernel(returns, assigned, np.ascontiguousarray(affinity), 2, mode)
    warm_weights = intra_class_weights_kernel(returns, assigned, np.ascontiguousarray(affinity), 2, WEIGHT_INV_VOL)
    apply_weight_caps_kernel(
        np.ascontiguousarray(warm_weights),
        np.ascontiguousarray(np.array([60.0, 100.0, 40.0, 100.0], dtype=np.float64)),
        assigned,
        2,
    )
    audit = validate_execution_audit(auto_class_execution_audit())
    if not audit["nopython"] or audit["python_fallback"] != 0:
        raise RuntimeError("自动构建大类 NJIT 内核未进入 nopython 模式")
    return audit


__all__ = [
    "AUTO_CLASS_ENGINE_VERSION",
    "AUTO_CLASS_KERNEL_VERSION",
    "LINKAGE_AVERAGE",
    "LINKAGE_COMPLETE",
    "LINKAGE_WARD",
    "WEIGHT_AFFINITY",
    "WEIGHT_EQUAL",
    "WEIGHT_INV_VAR",
    "WEIGHT_INV_VOL",
    "affinity_from_distance_kernel",
    "allocate_block_clusters_kernel",
    "agglomerative_linkage_kernel",
    "apply_weight_caps_kernel",
    "auto_class_execution_audit",
    "capacity_assign_kernel",
    "correlation_eigenvalues_kernel",
    "correlation_matrix_kernel",
    "corr_to_distance_kernel",
    "cross_class_corr_kernel",
    "cut_linkage_kernel",
    "denoise_correlation_kernel",
    "euclidean_distance_kernel",
    "finite_rows_kernel",
    "gmm_kernel",
    "intra_class_mean_corr_kernel",
    "intra_class_weights_kernel",
    "kmeans_kernel",
    "kmedoids_kernel",
    "mean_offdiagonal_kernel",
    "pca_features_kernel",
    "robust_standardize_kernel",
    "silhouette_kernel",
    "spectral_labels_kernel",
    "warm_auto_class_numba_kernels",
    "winsorize_returns_kernel",
]
