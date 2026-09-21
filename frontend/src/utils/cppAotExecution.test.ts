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

const model = {
  execution_backend: 'optimized_third_party_model', model_family: 'machine_learning',
  package: 'scikit-learn', package_version: '1.7.0', model_name: 'Tree', model_version: '1',
  native_backend: 'compiled_tree_inference', model_fingerprint: 'model-1',
  input_dtype: 'float64[C]', output_dtype: 'int64[C]', third_party_package: true,
  native_optimized: true, isolated_array_contract: true, model_engine_scope: 'training_or_inference_only',
  feature_pipeline_backend: 'cpp_aot', postprocess_backend: 'cpp_aot',
  feature_pipeline_audit: cpp, postprocess_audit: cpp,
  python_callback: false, python_fallback: 0, exemption_reason: 'optimized_ml_dl_nn_model_engine',
}

describe('C++ AOT execution policy', () => {
  it('accepts model stages with complete independent AOT proofs', () => {
    expect(() => assertCompliantExecutionGraph([model])).not.toThrow()
    expect(() => assertCompliantExecutionGraph([{ ...model, backend: 'numba_njit_fixed_signature' }])).toThrow()
  })
  it.each(['feature_pipeline', 'postprocess'])('requires a valid AOT proof for %s', stage => {
    for (const audit of [undefined, njit, { ...cpp, engine_build_id: '' },
      { ...cpp, python_worker_callbacks: 1 }, { ...cpp, backend: 'numba_njit_fixed_signature' }]) {
      expect(() => assertCompliantExecutionGraph([{ ...model, [`${stage}_audit`]: audit }])).toThrow()
    }
  })
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
