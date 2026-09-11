"""Reproducible prepared v2/v3 built-in comparison; no network or live data.

Run with PYTHONPATH=.:backend NUMBA_NRT_STATS=1 python this_file.py.
Compilation and fixture setup are excluded from timed numerical execution.
"""
from __future__ import annotations

import gc
import json
import statistics
import time

import numpy as np
from numba.core.runtime import rtsys

from cal_indicators.typed_dsl import compose_typed_series_bundle
from cal_indicators.typed_numba_plan import compile_numba_series_plan
from custom_indicators.service import _built_in_indicators
from custom_indicators.variable_registry import variable_types


def main() -> None:
    size, repeats = 4096, 9
    x = np.arange(size, dtype=np.float64)
    close = 100 + .01 * x + 2 * np.sin(x / 7)
    nav = 1 + .0001 * x + .03 * np.sin(x / 7)
    context = {'market_close': close, 'market_high': close + 1, 'market_low': close - 1,
               'volume': 1000 + 100 * np.cos(x / 3), 'adjusted_nav': nav,
               'returns': np.r_[np.nan, nav[1:] / nav[:-1] - 1], 'observation_dates': 20000 + x,
               'annual_risk_free_rate_decimal': .015, 'risk_free_rate_per_observation': 1.015 ** (1 / 252) - 1,
               'periods_per_year': 252.}
    for value in context.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    definitions = {(item['id'], item['revision']): item for item in _built_in_indicators()
                   if item.get('result_kind') == 'time_series'}
    rows = []
    for indicator_id in sorted({key[0] for key in definitions}):
        values, timings = {}, {}
        for revision in (2, 3):
            definition = definitions[indicator_id, revision]
            plan = compose_typed_series_bundle({item['id']: item['expression'] for item in definition['series_outputs']},
                variable_types=variable_types('single_product', definition['dsl_version']),
                dsl_version=definition['dsl_version'], operator_registry_version=definition['operator_registry_version'])
            compiled = compile_numba_series_plan(plan)
            args = tuple(context[name] for name in compiled.context_names)
            values[revision] = compiled.compute(args)
            signature_count = len(compiled.dispatcher.signatures)
            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                result = compiled.compute(args)
                samples.append((time.perf_counter() - start) * 1000)
                del result
            assert len(compiled.dispatcher.signatures) == signature_count
            timings[revision] = statistics.median(samples)
            if revision == 3:
                gc.collect()
                before = rtsys.get_allocation_stats()
                for _ in range(40):
                    result = compiled.compute(args)
                    del result
                gc.collect()
                after = rtsys.get_allocation_stats()
                retained = (after.alloc - after.free) - (before.alloc - before.free)
                assert retained == 0, (indicator_id, retained)
        for old, new in zip(values[2], values[3]):
            np.testing.assert_allclose(new, old, equal_nan=True, rtol=1e-8, atol=1e-9)
        rows.append({'id': indicator_id, 'v2_median_ms': round(timings[2], 4),
                     'v3_median_ms': round(timings[3], 4), 'v3_over_v2_ratio': round(timings[3] / timings[2], 2),
                     'retained_nrt_allocations_after_40_calls': retained,
                     'readonly_input': True, 'request_time_compilation': 0, 'python_fallback': 0})
    print(json.dumps({'size': size, 'repeats': repeats, 'includes_compilation': False, 'cases': rows}, indent=2))


if __name__ == '__main__':
    main()
