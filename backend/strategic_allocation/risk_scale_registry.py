"""Capabilities derived from the actual M1 registry and callable defaults."""
import inspect
from backend import frontier_moments as frontier
from . import risk_scale_kernels as numeric

VERSION = 'risk-scale-service/1.0.0'
PRIMARY_POINTS = 101
STABILITY_POINTS = 200
STABILITY_TOLERANCE = 0.02
ENDPOINT_TOLERANCE = 1e-8
LIMITS = {'assets': 30, 'components': 300, 'observations': 10000, 'input_elements': 3000000,
          'primary_points': PRIMARY_POINTS, 'stability_points': STABILITY_POINTS,
          'concurrent_computations': 1, 'cache_entries': 16, 'cache_bytes': 32 * 1024 * 1024,
          'cache_ttl_seconds': 300, 'max_iterations': 1000}
LABELS = {'frontier_shape_dp_v2': '前沿形状', 'equal_volatility_v1': '风险等距',
          'equal_arclength_v1': '前沿弧长等分', 'equal_return_v1': '收益等分',
          'manual_volatility_bands_v1': '人工风险阈值'}


def algorithms():
    params = inspect.signature(numeric.segment_frontier).parameters
    hints = {k: params[k].default for k in ('min_edges', 'min_span', 'tie_tolerance', 'near_linear_threshold')}
    return [{'id': key, 'version': numeric.VERSION, 'name': LABELS[key],
             'default': key == params['algorithm_id'].default, 'parameters': hints,
             'boundary_tolerance': numeric.BOUNDARY_TOL, 'span_tolerance': numeric.SPAN_TOL,
             'stability_tolerance': STABILITY_TOLERANCE, 'endpoint_tolerance': ENDPOINT_TOLERANCE,
             'frontier_primal_tolerance': frontier.PRIMAL_TOL, 'manual': key == 'manual_volatility_bands_v1'}
            for key, value in numeric.ALGORITHMS.items()]
