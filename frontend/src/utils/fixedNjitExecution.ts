export interface FixedNjitExecutionAudit {
  backend?: string
  execution_backend?: string
  nopython?: boolean
  object_mode?: number
  python_fallback?: number
  request_time_compilation?: number
  kernel_signatures?: Record<string, readonly string[]>
}

export interface OptimizedThirdPartyExecutionAudit {
  backend?: string
  execution_backend?: string
  model_family?: string
  package?: string
  package_version?: string
  model_name?: string
  model_version?: string
  native_backend?: string
  model_fingerprint?: string
  input_dtype?: string
  output_dtype?: string
  third_party_package?: boolean
  native_optimized?: boolean
  isolated_array_contract?: boolean
  model_engine_scope?: string
  feature_pipeline_backend?: string
  postprocess_backend?: string
  python_callback?: boolean
  python_fallback?: number
  exemption_reason?: string
}

const optimizedModelFamilies = new Set([
  'machine_learning',
  'deep_learning',
  'neural_network',
])

const disallowedNativeModelBackends = new Set([
  '',
  'interpreted',
  'numpy',
  'pandas',
  'python',
  'scipy',
  'numba',
])

const nonEmptyText = (value: unknown) => typeof value === 'string' && value.trim().length > 0

export function assertFixedNjitExecution(
  value: unknown,
  calculationLabel = '数值计算',
): asserts value is FixedNjitExecutionAudit {
  const audit = value as FixedNjitExecutionAudit | null | undefined
  const backend = audit?.execution_backend ?? audit?.backend
  const signatures = audit?.kernel_signatures
  const fixedSignatures = signatures
    && Object.keys(signatures).length > 0
    && Object.values(signatures).every((items) => (
      Array.isArray(items)
      && items.length > 0
      && items.every((item) => typeof item === 'string' && item.trim().length > 0)
      && new Set(items).size === items.length
    ))

  if (
    backend !== 'numba_njit_fixed_signature'
    || audit?.nopython !== true
    || audit.object_mode !== 0
    || audit.python_fallback !== 0
    || audit.request_time_compilation !== 0
    || fixedSignatures !== true
  ) {
    throw new Error(`${calculationLabel}未提供有效的固定签名 NJIT 执行证明`)
  }
}

/**
 * Some endpoints prove several independent numeric lanes in one response --
 * ``/api/fit-classes`` returns ``{ fit_analytics, performance_metrics }``.
 * Every lane has to carry its own fixed-signature NJIT proof.
 */
export type FixedNjitExecutionAuditLanes = Record<string, FixedNjitExecutionAudit>

export function assertFixedNjitExecutionLanes(
  value: unknown,
  calculationLabel = '数值计算',
): asserts value is FixedNjitExecutionAudit | FixedNjitExecutionAuditLanes {
  const audit = value as FixedNjitExecutionAudit | null | undefined
  if (nonEmptyText(audit?.execution_backend) || nonEmptyText(audit?.backend)) {
    assertFixedNjitExecution(value, calculationLabel)
    return
  }
  const lanes = audit && typeof audit === 'object' && !Array.isArray(audit)
    ? Object.entries(audit as Record<string, unknown>)
    : []
  if (lanes.length === 0) {
    throw new Error(`${calculationLabel}未提供有效的固定签名 NJIT 执行证明`)
  }
  lanes.forEach(([lane, item]) => assertFixedNjitExecution(item, `${calculationLabel}（${lane}）`))
}

export function assertOptimizedThirdPartyExecution(
  value: unknown,
  calculationLabel = '模型计算',
): asserts value is OptimizedThirdPartyExecutionAudit {
  const audit = value as OptimizedThirdPartyExecutionAudit | null | undefined
  const backend = audit?.execution_backend ?? audit?.backend
  const requiredText = [
    audit?.package,
    audit?.package_version,
    audit?.model_name,
    audit?.model_version,
    audit?.native_backend,
    audit?.model_fingerprint,
    audit?.input_dtype,
    audit?.output_dtype,
  ]
  if (
    backend !== 'optimized_third_party_model'
    || !optimizedModelFamilies.has(audit?.model_family ?? '')
    || requiredText.some((item) => !nonEmptyText(item))
    || disallowedNativeModelBackends.has(String(audit?.native_backend ?? '').trim().toLowerCase())
    || audit?.third_party_package !== true
    || audit.native_optimized !== true
    || audit.isolated_array_contract !== true
    || audit.model_engine_scope !== 'training_or_inference_only'
    || audit.feature_pipeline_backend !== 'numba_njit_fixed_signature'
    || audit.postprocess_backend !== 'numba_njit_fixed_signature'
    || audit.python_callback !== false
    || audit.python_fallback !== 0
    || audit.exemption_reason !== 'optimized_ml_dl_nn_model_engine'
  ) {
    throw new Error(`${calculationLabel}未提供有效的第三方优化模型执行证明`)
  }
}

export function assertCompliantNumericalExecution(
  value: unknown,
  calculationLabel = '数值计算',
): asserts value is FixedNjitExecutionAudit | OptimizedThirdPartyExecutionAudit {
  const audit = value as { backend?: unknown; execution_backend?: unknown } | null | undefined
  const backend = audit?.execution_backend ?? audit?.backend
  if (backend === 'numba_njit_fixed_signature') {
    assertFixedNjitExecution(value, calculationLabel)
    return
  }
  if (backend === 'optimized_third_party_model') {
    assertOptimizedThirdPartyExecution(value, calculationLabel)
    return
  }
  throw new Error(`${calculationLabel}未提供合规的数值执行证明`)
}

export function assertCompliantExecutionGraph(
  value: unknown,
  calculationLabel = '数值计算链路',
): void {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${calculationLabel}未提供完整的数值执行证明链`)
  }
  value.forEach((item, index) => {
    const wrapper = item as { plan?: unknown } | null | undefined
    const audit = wrapper?.plan && typeof wrapper.plan === 'object' ? wrapper.plan : item
    assertCompliantNumericalExecution(audit, `${calculationLabel}第 ${index + 1} 段`)
  })
}
