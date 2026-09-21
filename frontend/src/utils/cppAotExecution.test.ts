import { describe, expect, it } from 'vitest'
import {
  assertCppAotExecution, assertNativeNumericalExecution,
  assertNativeNumericalExecutionLanes, assertCompliantExecutionGraph,
} from './fixedNjitExecution'

import { cppAotAudit as cpp } from '../test/cppAotFixture'

const njit = {
  execution_backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0,
  python_fallback: 0, request_time_compilation: 0, kernel_signatures: { k: ['float64[:]'] },
}

describe('C++ AOT execution policy', () => {
  it('accepts a native proof without fabricated NJIT signatures', () => {
    expect(() => assertCppAotExecution(cpp)).not.toThrow()
    expect(() => assertNativeNumericalExecution(cpp)).not.toThrow()
    expect(() => assertNativeNumericalExecution(njit)).not.toThrow()
    expect(() => assertNativeNumericalExecutionLanes({ metrics: cpp, weights: njit })).not.toThrow()
    expect(() => assertCompliantExecutionGraph([cpp, { plan: njit }])).not.toThrow()
  })
  it.each(Object.keys(cpp))('rejects a missing %s', (key) => {
    const incomplete = { ...cpp } as Record<string, unknown>
    delete incomplete[key]
    expect(() => assertCppAotExecution(incomplete)).toThrow()
  })
  it.each([
    { backend: 'numba_njit_fixed_signature' }, { engine_build_id: '' },
    { native_aot: false }, { python_fallback: true }, { python_operator_calls: 1 },
    { python_worker_callbacks: 1 }, { request_time_compilation: 1 },
    { cpu_tokens: 0 }, { cpu_tokens: 3 }, { cpu_budget: 1.5 },
    { operator_registry_version: 'unknown' }, { result_lifetime: 'unknown' },
  ])('fails closed on contradictory or unsupported proof', (override) => {
    expect(() => assertNativeNumericalExecution({ ...cpp, ...override })).toThrow()
    expect(() => assertNativeNumericalExecutionLanes({ good: njit, bad: { ...cpp, ...override } })).toThrow()
  })
})
