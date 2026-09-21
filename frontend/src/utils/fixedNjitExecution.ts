export interface FixedNjitExecutionAudit {
  backend?: string
  execution_backend?: string
  nopython?: boolean
  object_mode?: number
  python_fallback?: number
  request_time_compilation?: number
  kernel_signatures?: Record<string, readonly string[]>
}

export interface CppAotExecutionAudit {
  backend?: string
  execution_backend?: string
  audit_schema?: string
  engine?: string
  engine_version?: string
  engine_build_id?: string
  operator_registry_version?: string
  typed_ir_version?: string
  plan_fingerprint?: string
  native_aot?: boolean
  python_fallback?: number
  python_operator_calls?: number
  python_worker_callbacks?: number
  request_time_compilation?: number
  input_dtype?: string
  output_dtype?: string
  cpu_budget?: number
  cpu_tokens?: number
  result_lifetime?: string
}

export type NativeNumericalExecutionAudit = FixedNjitExecutionAudit | CppAotExecutionAudit

export type NativeNumericalExecutionAuditLanes = Record<string, NativeNumericalExecutionAudit>

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
    || (audit?.backend && audit.execution_backend && audit.backend !== audit.execution_backend)
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
    || !['numba_njit_fixed_signature', 'cpp_aot'].includes(audit.feature_pipeline_backend ?? '')
    || !['numba_njit_fixed_signature', 'cpp_aot'].includes(audit.postprocess_backend ?? '')
    || audit.python_callback !== false
    || audit.python_fallback !== 0
    || audit.exemption_reason !== 'optimized_ml_dl_nn_model_engine'
  ) {
    throw new Error(`${calculationLabel}未提供有效的第三方优化模型执行证明`)
  }
}

export function assertCppAotExecution(
  value: unknown,
  calculationLabel = '数值计算',
): asserts value is CppAotExecutionAudit {
  const audit = value as CppAotExecutionAudit | null | undefined
  const cpu = audit?.cpu_budget
  const tokens = audit?.cpu_tokens
  if (
    (audit?.execution_backend ?? audit?.backend) !== 'cpp_aot'
    || (audit?.backend && audit.execution_backend && audit.backend !== audit.execution_backend)
    || audit?.audit_schema !== 'cpp-aot-execution-1'
    || audit.engine !== 'calmetrics_engine'
    || !nonEmptyText(audit.engine_version)
    || !/^[0-9a-f]{64}$/.test(audit.engine_build_id ?? '')
    || !/^native-[1-9][0-9]*-[0-9a-f]{32}$/.test(audit.plan_fingerprint ?? '')
    || audit.operator_registry_version !== 'canonical-native-1'
    || audit.typed_ir_version !== 'cpp-typed-ir-1'
    || audit.native_aot !== true
    || audit.python_fallback !== 0
    || audit.python_operator_calls !== 0
    || audit.python_worker_callbacks !== 0
    || audit.request_time_compilation !== 0
    || audit.input_dtype !== 'float64' || audit.output_dtype !== 'float64'
    || !Number.isInteger(cpu) || !Number.isInteger(tokens)
    || !(tokens! >= 1 && tokens! <= cpu! && cpu! <= 1024)
    || !['independent', 'borrowed_until_next_run'].includes(audit.result_lifetime ?? '')
  ) throw new Error(`${calculationLabel}未提供有效的 C++ AOT 执行证明`)
}

/** Both backends are native numerical lanes; this does not grant model exemptions. */
export function assertNativeNumericalExecution(
  value: unknown,
  calculationLabel = '数值计算',
): asserts value is NativeNumericalExecutionAudit {
  const audit = value as { backend?: unknown; execution_backend?: unknown } | null | undefined
  if ((audit?.execution_backend ?? audit?.backend) === 'cpp_aot') {
    assertCppAotExecution(value, calculationLabel)
  } else {
    assertFixedNjitExecution(value, calculationLabel)
  }
}

export function assertNativeNumericalExecutionLanes(
  value: unknown,
  calculationLabel = '数值计算',
): asserts value is NativeNumericalExecutionAudit | Record<string, NativeNumericalExecutionAudit> {
  const audit = value as { backend?: unknown; execution_backend?: unknown } | null | undefined
  if (nonEmptyText(audit?.execution_backend) || nonEmptyText(audit?.backend)) {
    assertNativeNumericalExecution(value, calculationLabel)
    return
  }
  const lanes = audit && typeof audit === 'object' && !Array.isArray(audit) ? Object.entries(audit) : []
  if (lanes.length === 0) throw new Error(`${calculationLabel}未提供有效的数值执行证明`)
  lanes.forEach(([lane, item]) => assertNativeNumericalExecution(item, `${calculationLabel}（${lane}）`))
}

export function assertCompliantNumericalExecution(
  value: unknown,
  calculationLabel = '数值计算',
): asserts value is NativeNumericalExecutionAudit | OptimizedThirdPartyExecutionAudit {
  const audit = value as { backend?: unknown; execution_backend?: unknown } | null | undefined
  const backend = audit?.execution_backend ?? audit?.backend
  if (backend === 'numba_njit_fixed_signature' || backend === 'cpp_aot') {
    assertNativeNumericalExecution(value, calculationLabel)
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

export function isNativeNumericalExecution(value: unknown): value is NativeNumericalExecutionAudit {
  try { assertNativeNumericalExecution(value); return true } catch { return false }
}
