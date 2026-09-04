import { describe, expect, it } from 'vitest'
import {
  assertCompliantExecutionGraph,
  assertCompliantNumericalExecution,
  assertFixedNjitExecution,
  assertOptimizedThirdPartyExecution,
} from './fixedNjitExecution'

const validAudit = {
  execution_backend: 'numba_njit_fixed_signature',
  nopython: true,
  object_mode: 0,
  python_fallback: 0,
  request_time_compilation: 0,
  kernel_signatures: { financial_kernel: ['(Array(float64, 1, C),) -> float64'] },
}

describe('assertFixedNjitExecution', () => {
  it('接受完整且固定的 NJIT 执行证明', () => {
    expect(() => assertFixedNjitExecution(validAudit, '产品统计')).not.toThrow()
  })

  it.each([
    undefined,
    { ...validAudit, python_fallback: 1 },
    { ...validAudit, object_mode: 1 },
    { ...validAudit, request_time_compilation: 1 },
    { ...validAudit, kernel_signatures: {} },
    { ...validAudit, kernel_signatures: { financial_kernel: [] } },
  ])('拒绝缺失、回退、请求期编译或空签名的执行证明', (audit) => {
    expect(() => assertFixedNjitExecution(audit, '产品统计'))
      .toThrow('产品统计未提供有效的固定签名 NJIT 执行证明')
  })
})

const validThirdPartyAudit = {
  execution_backend: 'optimized_third_party_model',
  model_family: 'machine_learning',
  package: 'scikit-learn',
  package_version: '1.7.0',
  model_name: 'GradientBoostingClassifier',
  model_version: 'regime-model-3',
  native_backend: 'compiled_tree_inference',
  model_fingerprint: 'sha256:model',
  input_dtype: 'float64[C]',
  output_dtype: 'int64[C]',
  third_party_package: true,
  native_optimized: true,
  isolated_array_contract: true,
  model_engine_scope: 'training_or_inference_only',
  feature_pipeline_backend: 'numba_njit_fixed_signature',
  postprocess_backend: 'numba_njit_fixed_signature',
  python_callback: false,
  python_fallback: 0,
  exemption_reason: 'optimized_ml_dl_nn_model_engine',
}

describe('mixed numerical execution policy', () => {
  it('只接受完整隔离的第三方优化模型训练或推理通道', () => {
    expect(() => assertOptimizedThirdPartyExecution(validThirdPartyAudit, '情景模型')).not.toThrow()
    expect(() => assertCompliantNumericalExecution(validThirdPartyAudit, '情景模型')).not.toThrow()
  })

  it.each([
    { ...validThirdPartyAudit, model_family: 'regression' },
    { ...validThirdPartyAudit, third_party_package: false },
    { ...validThirdPartyAudit, native_backend: 'python' },
    { ...validThirdPartyAudit, feature_pipeline_backend: 'python' },
    { ...validThirdPartyAudit, postprocess_backend: 'pandas' },
    { ...validThirdPartyAudit, python_callback: true },
  ])('拒绝把普通数学或 Python 回退伪装成第三方模型豁免', (audit) => {
    expect(() => assertOptimizedThirdPartyExecution(audit, '情景模型'))
      .toThrow('情景模型未提供有效的第三方优化模型执行证明')
  })

  it('校验混合 NJIT 与第三方模型的完整执行图，并支持包装的 plan', () => {
    expect(() => assertCompliantExecutionGraph([
      validAudit,
      { source_kind: 'formula', plan: validAudit },
      validThirdPartyAudit,
    ], '历史情景')).not.toThrow()
  })

  it('拒绝空执行图或其中任一不合规通道', () => {
    expect(() => assertCompliantExecutionGraph([], '历史情景'))
      .toThrow('历史情景未提供完整的数值执行证明链')
    expect(() => assertCompliantExecutionGraph([validAudit, { backend: 'numpy' }], '历史情景'))
      .toThrow('历史情景第 2 段未提供合规的数值执行证明')
  })
})
