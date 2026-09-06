import {
  assertFixedNjitExecution,
  type FixedNjitExecutionAudit,
} from '../utils/fixedNjitExecution'

export type ProductKind = 'etf' | 'fund'

export type IndicatorSource = 'built_in' | 'custom'
export type IndicatorDisplayFormat = 'number' | 'percent'
export type IndicatorDirection = 'higher_better' | 'lower_better'
export type EvaluationStatus = 'ok' | 'warning' | 'unavailable' | 'error'
export type IndicatorShape = 'scalar' | 'series' | 'vector' | 'matrix' | 'mask' | 'tuple' | 'unknown'
export type IndicatorContextDomain = 'single_product' | 'portfolio'
export type IndicatorResultKind = 'scalar' | 'time_series'
export type IndicatorType = 'return' | 'risk' | 'risk_adjusted' | 'path' | 'market_liquidity' | 'technical' | 'other'
export type SeriesOutputMeasureId =
  | 'auto'
  | 'raw_market_price'
  | 'adjusted_nav'
  | 'reported_nav'
  | 'virtual_nav'
  | 'normalized'
  | 'bounded_0_1'
  | 'bounded_minus1_1'
  | 'oscillator_0_100'
  | 'return_decimal'
  | 'rate_decimal'
  | 'volume'
  | 'currency_amount'
  | 'count'
  | 'calendar_days'
  | 'dimensionless'
  | 'derived'

export interface SeriesOutputMeasureOption {
  id: SeriesOutputMeasureId
  label: string
  description: string
  semantic_dimensions: string[]
  range: [number | null, number | null] | null
  default_unit: string
  default_display_format: IndicatorDisplayFormat
}

export interface FixedSeriesParameter {
  id: string
  label: string
  value: number
  type: 'integer' | 'number'
  source?: string
}

export interface RollingSourceDefinition {
  kind: 'rolling_scalar'
  /** Canonical rolling-transform protocol version. */
  transform_version: '1.0.0'
  /** Compatibility field accepted from older generated drafts. */
  version?: '1.0.0'
  indicator_id: string
  indicator_revision: number
  indicator_name: string
  /** Canonical hash of the locked scalar calculation contract. */
  definition_hash: string
  /** Compatibility field accepted from older generated drafts. */
  source_definition_hash?: string
  source_dsl_version: string
  window_observations: number
  minimum_observations: number
  detached: boolean
}

export interface RollingSeriesDerivationResponse {
  definition: IndicatorDraft
  validation: ValidationResponse
  source: {
    indicator_id: string
    indicator_revision: number
    indicator_name: string
    window_observations: number
  }
}

export interface MetricPresentation {
  indicator_id: string | null
  revision: number | null
  name: string
  source: IndicatorSource | 'inline' | string
  indicator_type?: IndicatorType
  category: string
  category_label: string
  context_kind: IndicatorContextDomain
  result_kind?: IndicatorResultKind
  catalog_status: 'current' | 'compatibility' | string
  display_format: IndicatorDisplayFormat
  precision: number
  unit: string
  notation: 'standard' | 'compact'
  value_scale: number
  output_measure: string
  direction: IndicatorDirection
  description: string
  methodology: string
  data_basis: string
  minimum_observations: number
  applicable_product_kinds: Array<ProductKind | 'portfolio'>
  axis_anchor?: string | null
  parameter_schema?: SeriesParameterDefinition[]
  fixed_parameters?: FixedSeriesParameter[]
  series_outputs?: SeriesOutputDefinition[]
  history_policy?: SeriesHistoryPolicy | null
  history_inference_source?: string | null
  lookback_observations?: number
}

export interface IndicatorTemplateOrigin {
  template_id: string
  template_version?: number
  bindings?: ComposeArgument[]
  detached?: boolean
}

export type SeriesHistoryPolicy = 'lookback' | 'full_history'

export interface SeriesParameterDefinition {
  id: string
  label: string
  type: 'integer' | 'number'
  default: number
  minimum: number
  maximum: number
  step: number
  description?: string
}

export interface SeriesOutputDefinition {
  id: string
  label: string
  expression: string
  unit: string
  display_format: IndicatorDisplayFormat
  precision: number
  output_measure: SeriesOutputMeasureId
  inferred_output_measure?: SeriesOutputMeasureId | null
  resolved_output_measure?: SeriesOutputMeasureId | null
  output_measure_source?: 'inferred' | 'explicit' | string | null
  semantic_dimension?: string | null
  price_basis?: string | null
  value_range?: [number | null, number | null] | null
}

export interface SeriesOutputInference {
  id: string
  label: string
  expression: string
  latex: string
  display_latex?: string | null
  math_notation_version?: string | null
  python_expression?: string | null
  inferred_type: string
  shape: IndicatorShape
  semantic_dimension?: string | null
  price_basis?: string | null
  output_measure?: SeriesOutputMeasureId | null
  inferred_output_measure?: SeriesOutputMeasureId | null
  resolved_output_measure?: SeriesOutputMeasureId | null
  output_measure_source?: 'inferred' | 'explicit' | string | null
  value_range?: [number | null, number | null] | null
  dependencies: string[]
  root_id: string | number
}

export interface IndicatorDraft {
  name: string
  description: string
  expression: string
  /** Legacy response compatibility only. v2 definitions do not persist runtime periods. */
  periods?: readonly string[]
  unit: string
  display_format: IndicatorDisplayFormat
  precision: number
  direction: IndicatorDirection
  indicator_type?: IndicatorType
  annual_risk_free_rate_percent: number
  dsl_version?: string
  operator_registry_version?: string
  numeric_kernel_version?: string
  variable_registry_version?: string
  context_schema_version?: string
  data_contract_version?: string
  period_policy?: 'all_supported'
  context_kind?: IndicatorContextDomain
  result_kind?: IndicatorResultKind
  output_contract?: 'scalar' | 'series_bundle'
  output_measure?: string
  parameter_schema?: SeriesParameterDefinition[]
  fixed_parameters?: FixedSeriesParameter[]
  series_outputs?: SeriesOutputDefinition[]
  axis_anchor?: string | null
  history_policy?: SeriesHistoryPolicy | null
  lookback_parameter?: string | null
  lookback_observations?: number
  history_inference_source?: string | null
  minimum_observations?: number
  methodology?: string
  data_basis?: string
  template_origin?: IndicatorTemplateOrigin | string | null
  rolling_source?: RollingSourceDefinition | null
  rolling_transform?: {
    version?: string
    rewritten_reductions?: string[]
    source_variables?: string[]
  } | null
  rolling_series_compatibility?: {
    supported: boolean
    protocol_version: string
    rewritten_reductions?: string[]
    code?: string
    message?: string
  }
}

export interface IndicatorDefinition extends IndicatorDraft {
  id: string
  revision: number
  source: IndicatorSource
  read_only: boolean
  created_at: string
  updated_at: string
  category_id?: string
  category_label?: string
  minimum_observations?: number
  formula_version?: string
  methodology?: string
  data_basis?: string
  applicable_product_kinds?: Array<ProductKind | 'portfolio'>
  catalog_status?: 'current' | 'compatibility' | string
  ui_exposed?: boolean
  presentation?: MetricPresentation
  display_latex?: string | null
  math_notation_version?: string
  required_variables?: string[]
  channel_types?: Record<string, {
    kind?: string
    dtype?: string
    axes?: string[]
    shape?: Array<string | number>
    semantic_dimension?: string
    price_basis?: string | null
  }>
  compiled_series_plan_id?: string
  availability_policy?: 'runtime_required' | string
  availability_status?: 'runtime_check' | 'ready' | string
  product_kind_hint?: {
    product_kind: ProductKind
    status: 'likely_available' | 'runtime_check' | string
    message: string
  }
}

export interface SnapshotIndicatorConfigItem {
  indicator_id: string
  indicator_revision: number
  period: string
  field: string
  channel_id?: string | null
  reducer?: 'last_finite' | null
  name?: string
  source?: IndicatorSource
  status?: 'ready' | 'definition_missing' | string
  status_message?: string
  presentation?: MetricPresentation
}

export interface SnapshotIndicatorConfig {
  schema_version: number
  revision: number
  updated_at?: string | null
  max_items: number
  items: SnapshotIndicatorConfigItem[]
  snapshot?: {
    generated_at?: string | null
    config_revision?: number | null
    configured_count?: number | null
    data_generation?: string | null
  } | null
  snapshot_status?: 'ready' | 'stale' | 'missing' | string
}

export interface IndicatorVariable {
  name: string
  label: string
  value_type: string
  dtype: string
  latex: string
  /** v2 optional metadata; absent fields are inferred from the v1 contract. */
  shape?: IndicatorShape
  structural_type?: Exclude<IndicatorShape, 'tuple' | 'unknown'> | 'mask'
  axes?: string[]
  symbolic_shape?: Array<string | number>
  semantic?: string
  semantic_role?: string
  measure?: string
  price_basis?: string | null
  scale?: string
  timing?: string
  availability_tier?: 'core' | 'conditional' | 'experimental' | string
  missing_policy?: string
  description?: string
  source?: string
  domains?: IndicatorContextDomain[]
  category?: string
  category_id?: string
  category_label?: string
  aliases?: string[]
  tags?: string[]
  unit?: string
  frequency?: string
  source_dataset?: string
  source_field?: string
  canonical_field?: string
  source_bindings?: Partial<Record<ProductKind, {
    configured: boolean
    source_dataset?: string | null
    source_field?: string | null
    transform?: string | null
    reason?: string | null
  }>>
  availability_policy?: string
  alternative_variables?: Array<string | { variable_id: string; label: string }>
  data_basis?: string
  product_kinds?: Array<ProductKind | 'portfolio'>
  availability?: 'available' | 'conditional' | 'unavailable' | string
}

export interface IndicatorOperatorParameter {
  name: string
  label?: string
  description?: string
  shape?: IndicatorShape
  allowed_shapes?: IndicatorShape[]
  allowed_types?: string[]
  allowed_semantic_roles?: string[]
  optional?: boolean
  default?: number | string | null
  source_policy?: 'fixed_constant' | string
  constant_kind?: 'integer' | 'number' | string
  minimum?: number
  maximum?: number
}

export interface IndicatorOperatorParameterSet {
  arity: number
  parameters: IndicatorOperatorParameter[]
  output?: string
  shape_rule?: string
}

export interface IndicatorOperator {
  name: string
  label: string
  signature: string
  latex_template: string
  display_latex_template?: string
  source_latex_template?: string
  return_type: string
  output_shape?: IndicatorShape
  input_shapes?: IndicatorShape[]
  mathematical_essence?: string
  semantic?: string
  domains?: IndicatorContextDomain[]
  parameters?: IndicatorOperatorParameter[]
  parameter_sets?: IndicatorOperatorParameterSet[]
  category?: string
  family?: string
  category_id?: string
  category_label?: string
  aliases?: string[]
  tags?: string[]
  examples?: string[]
  cost_estimate?: string | number
  type_rules?: Array<Record<string, unknown>>
  semantic_rules?: Record<string, unknown>
  output_rule?: Record<string, unknown>
  latex_alias?: string
  cost?: Record<string, unknown>
  minimum_samples?: number
  nan_policy?: string
  ddof?: number | null
  version?: string
  execution_backend?: 'numba_njit_fixed_signature' | 'numpy_blas_lapack' | 'numpy' | string
  njit_supported?: boolean
  kernel_version?: string
  execution_lane?: 'numba' | 'numba_blas' | string
  compiled_signatures?: string[]
  warmup_status?: 'pending' | 'ready' | string
  opcode?: number
  kernel_input_signatures?: string[]
  kernel_output_signature?: string
  parallel_policy?: string
  status_contract?: string
}

export interface IndicatorTemplate extends IndicatorDraft {
  id?: string
  label?: string
  template_kind?: 'indicator' | 'fragment'
  mathematical_essence?: string
  semantic?: string
  output_shape?: IndicatorShape
  domains?: IndicatorContextDomain[]
  parameters?: IndicatorOperatorParameter[]
}

export interface IndicatorPeriod {
  value: string
  label: string
  description: string
  kind?: 'rolling' | 'calendar' | 'lifetime' | string
  group?: 'rolling' | 'calendar_week' | 'calendar_month' | 'calendar_year' | 'lifetime' | string
}

export interface IndicatorMeta {
  engine_version: string
  workspace_scope: 'shared'
  variables: IndicatorVariable[]
  operators: IndicatorOperator[]
  periods: IndicatorPeriod[]
  templates: IndicatorTemplate[]
  limits: Record<string, number>
  dsl_version?: string
  compiler_version?: string
  operator_registry_version?: string
  numeric_kernel_version?: string
  variable_registry_version?: string
  data_contract_version?: string
  context_schema_version?: string
  math_notation_version?: string
  context_kinds?: IndicatorContextDomain[]
  indicator_result_kinds?: Array<{ id: IndicatorResultKind; label: string }>
  series_output_measures?: SeriesOutputMeasureOption[]
  indicator_categories?: Array<{ id: string; label: string }>
  indicator_types?: Array<{ id: IndicatorType; label: string }>
  numeric_backend?: {
    version: string
    warmed: boolean
    numba_version: string
    numpy_version: string
    policy: Record<string, string>
    compiled_signatures: Record<string, string[]>
    operator_coverage?: string
    python_fallback?: number
    python_operator_calls?: number
  }
  predefined_calculations?: IndicatorTemplate[]
}

export interface IndicatorDiagnostic {
  code: string
  message: string
  field?: string | null
  node_id?: string | number
  expected?: unknown
  actual?: unknown
  details?: Record<string, unknown>
}

export interface IndicatorDagNode {
  id: string | number
  label: string
  kind: string
  period?: string | null
  value_type?: string
  shape?: IndicatorShape
  /** Raw typed compose/infer responses keep the complete inferred value contract nested here. */
  inferred_type?: string | {
    kind?: string
    dtype?: string
    axes?: string[]
    shape?: Array<string | number>
    semantic_dimension?: string
    price_basis?: string | null
    is_mask?: boolean
    display?: string
  }
  semantic?: string
  signature?: string
  operator_id?: string
  operator_version?: string
  /** Typed DSL returns symbolic axes as an array; legacy DAGs may still return bracket notation. */
  symbolic_shape?: string | Array<string | number>
  actual_shape?: number[]
  formula_fragment?: string
  latex_fragment?: string | null
  cost_estimate?: string | number
  diagnostics?: IndicatorDiagnostic[]
  inputs?: Array<string | number>
  arguments?: Array<{ name: string; input_node_id: string | number }>
  operator?: { id: string; version?: string } | null
  cost?: { model?: string; expression?: string | number }
}

export interface IndicatorDagEdge {
  source: string | number
  target: string | number
  parameter?: string
  parameter_name?: string
  input_name?: string
  order?: number
}

export interface IndicatorDag {
  nodes: IndicatorDagNode[]
  edges: IndicatorDagEdge[]
  roots: Record<string, string | number>
}

export interface ValidationResponse {
  valid: boolean
  diagnostics: IndicatorDiagnostic[]
  dependencies: string[]
  python_expression: string | null
  dag: IndicatorDag | null
  latex?: string | null
  display_latex?: string | null
  math_notation_version?: string | null
  compile_token?: string | null
  compile_token_scope?: 'current_process_warm_cache' | string
  result_kind?: IndicatorResultKind
  output_contract?: 'scalar' | 'series_bundle'
  output_channels?: SeriesOutputDefinition[]
  output_inferences?: Record<string, SeriesOutputInference>
  channel_types?: IndicatorDefinition['channel_types']
  parameter_schema?: SeriesParameterDefinition[]
  fixed_parameters?: FixedSeriesParameter[]
  history_policy?: SeriesHistoryPolicy | null
  history_inference_source?: string | null
  lookback_observations?: number
  minimum_observations?: number
  compiled_series_plan_id?: string
  compile_status?: string
  execution?: FixedNjitExecutionAudit
}

export interface ComposeArgument {
  parameter: string
  source: 'variable' | 'constant' | 'expression'
  value: string | number
}

export interface ComposeIndicatorRequest {
  operator_id?: string
  template_id?: string
  indicator_id?: string
  indicator_revision?: number
  arguments: ComposeArgument[]
  context: IndicatorContextDomain
  dsl_version?: string
  operator_registry_version?: string
  variable_registry_version?: string
  data_contract_version?: string
  context_schema_version?: string
  parameter_schema?: SeriesParameterDefinition[]
}

export interface InferenceResponse {
  latex: string
  expression?: string
  normalized_expression?: string
  display_latex?: string
  math_notation_version?: string
  inferred_type: string
  shape: IndicatorShape
  semantic_warnings: IndicatorDiagnostic[]
  dependencies?: string[]
  dag?: IndicatorDag
  template_origin?: IndicatorTemplateOrigin | null
  indicator_origin?: {
    indicator_id: string
    indicator_revision: number
    name: string
    source: IndicatorSource | string
  } | null
}

export interface EvaluationTarget {
  kind: ProductKind
  product_id: string
}

export interface PortfolioRun {
  id: string
  target_name: string
  target_revision: number | string
  effective_as_of: string | null
  [key: string]: unknown
}

export interface EvaluationWarning {
  code: string
  message: string
}

export type InputRequirementStatus =
  | 'available'
  | 'partial'
  | 'source_unavailable'
  | 'field_missing'
  | 'no_observations'
  | 'insufficient_window'
  | 'unavailable'

export interface InputRequirementItem {
  variable_id: string
  label: string
  canonical_field: string
  status: InputRequirementStatus | string
  reason_code?: string | null
  reason?: string | null
  source_configured?: boolean
  source_dataset?: string | null
  source_field?: string | null
  coverage_ratio?: number | null
  non_null_count?: number | null
  first_date?: string | null
  latest_date?: string | null
  actual_shape?: number[] | null
  alternative_variables?: Array<{ variable_id: string; label: string }>
}

export interface InputRequirements {
  status: 'ready' | 'partial' | 'blocked' | 'insufficient' | string
  required_count: number
  available_count: number
  items: InputRequirementItem[]
  blocking_inputs: InputRequirementItem[]
  partial_inputs: InputRequirementItem[]
  reason?: EvaluationWarning | null
}

export interface TargetDataSummary {
  available_datasets: string[]
  available_variables: string[]
  data_latest_date: string | null
}

export interface EvaluationWindow {
  requested_as_of: string | null
  effective_as_of: string | null
  start_date: string | null
  end_date: string | null
  observation_count: number
  data_latest_date: string | null
}

export interface VariableAvailabilityItem {
  variable_id: string
  label?: string
  canonical_field?: string
  status: 'available' | 'partial' | 'source_unavailable' | 'field_missing' | 'no_observations' | 'unavailable' | 'not_applicable' | 'insufficient_window' | 'declared' | string
  coverage?: Record<string, unknown>
  coverage_ratio?: number | null
  non_null_count?: number | null
  first_date?: string | null
  latest_date?: string | null
  actual_shape?: number[] | null
  reason?: { code?: string; message?: string } | null
  reason_code?: string | null
  source_dataset?: string | null
  source_field?: string | null
  source_configured?: boolean
  available_target_count?: number
  target_count?: number
  target_statuses?: Array<{
    target: { kind: ProductKind; product_id: string; name: string }
    status: string
    reason?: { code?: string; message?: string } | null
    coverage?: Record<string, unknown>
    actual_shape?: number[] | null
  }>
  window?: EvaluationWindow | null
}

export interface VariableAvailabilityResponse {
  target: { kind: ProductKind; product_id: string; name: string } | null
  period: string
  as_of?: string | null
  data_latest_date?: string | null
  items: VariableAvailabilityItem[]
  targets?: VariableAvailabilityResponse[]
  summary?: { target_count: number; variable_count: number }
}

export interface EvaluationSeriesPoint {
  date: string
  value: number | null
}

export interface EvaluationResult {
  indicator_id: string | null
  indicator_revision: number | null
  indicator_name: string
  target: { kind: ProductKind | 'portfolio'; product_id: string; name: string }
  period: string
  value: number | null
  status: EvaluationStatus
  warnings: EvaluationWarning[]
  window: EvaluationWindow
  presentation: MetricPresentation
  input_requirements?: InputRequirements | null
  target_data?: TargetDataSummary | null
  series?: EvaluationSeriesPoint[]
}

export interface EvaluateIndicatorsRequest {
  indicator_ids?: string[]
  inline_definition?: IndicatorDraft
  compile_token?: string
  targets: EvaluationTarget[]
  period: string
  as_of?: string
  include_series?: boolean
}

export interface SeriesIndicatorInstance {
  indicator_id?: string
  indicator_revision?: number
  inline_definition?: IndicatorDraft
  compile_token?: string
  /** Deprecated compatibility field. New time-series definitions lock constants in the formula. */
  parameters?: Record<string, number>
}

export interface EvaluateTimeSeriesIndicatorsRequest {
  indicator_instances: SeriesIndicatorInstance[]
  target: EvaluationTarget
  period: string
  as_of?: string
  max_points?: number
}

export interface TimeSeriesChannelResult {
  id: string
  label: string
  unit: string
  display_format: IndicatorDisplayFormat
  precision: number
  output_measure: SeriesOutputMeasureId | string
  semantic_dimension?: string | null
  price_basis?: string | null
  value_range?: [number | null, number | null] | null
  null_count?: number
  values: Array<number | null>
}

export interface TimeSeriesIndicatorResult {
  indicator_id: string | null
  indicator_revision: number | null
  indicator_name: string
  result_kind: 'time_series'
  target: { kind: ProductKind; product_id: string; name: string }
  period: string
  parameters: Record<string, number>
  axis_anchor: string
  history_policy: SeriesHistoryPolicy
  lookback_observations?: number
  minimum_observations?: number
  status: EvaluationStatus
  warnings: EvaluationWarning[]
  window: EvaluationWindow
  dates: string[]
  channels: TimeSeriesChannelResult[]
  presentation: MetricPresentation
  execution?: FixedNjitExecutionAudit
}

export interface EvaluateTimeSeriesIndicatorsResponse {
  results: TimeSeriesIndicatorResult[]
  summary: { total: number; ok: number; warning: number; error: number; unavailable: number }
  cache: { hits: number; misses: number }
  execution: FixedNjitExecutionAudit & {
    compiled_plan_ids?: string[]
    request_time_compilation?: number
  }
}

export interface ExportIndicatorExcelRequest {
  indicator_ids?: string[]
  inline_definition?: IndicatorDraft
  compile_token?: string
  targets: EvaluationTarget[]
  period: string
  as_of?: string
  parameters?: Record<string, number>
}

export interface DownloadedFile {
  blob: Blob
  filename: string
}

export interface EvaluatePortfolioIndicatorsRequest {
  run_id: string
  indicator_ids?: string[]
  inline_definition?: IndicatorDraft
  compile_token?: string
}

export interface EvaluateIndicatorsResponse {
  results: EvaluationResult[]
  summary: { total: number; ok: number; warning: number; error: number; unavailable?: number }
  cache: { hits: number; misses: number }
  execution: FixedNjitExecutionAudit
}

export interface EvaluationPlanIndicator {
  indicator_id: string
  indicator_revision: number
  period: string
  weight: number
  direction: IndicatorDirection
}

export type InstrumentProductFilterKey = 'fund_type' | 'invest_type' | 'qdii_type' | 'market' | 'status' | 'management' | 'custodian'

export type InstrumentProductFilterState = Record<InstrumentProductFilterKey, string[]>

export interface EvaluationProductSelection {
  query: string
  filters: InstrumentProductFilterState
  conditions: ProductCondition[]
  selection_mode: 'manual' | 'all_matching'
}

export interface EvaluationPlanDraft {
  name: string
  description: string
  product_kind: ProductKind
  indicators: EvaluationPlanIndicator[]
  targets: EvaluationTarget[]
  product_selection?: EvaluationProductSelection | null
  missing_policy: 'strict'
}

export interface EvaluationPlan extends EvaluationPlanDraft {
  id: string
  revision: number
  created_at: string
  updated_at: string
}

export interface EvaluationPlanValue {
  indicator_id: string
  indicator_revision: number
  indicator_name: string
  period: string
  value: number | null
  status: EvaluationStatus
  warnings: EvaluationWarning[]
  window: EvaluationWindow | null
  presentation: MetricPresentation
  input_requirements?: InputRequirements | null
  target_data?: TargetDataSummary | null
  direction: IndicatorDirection
  definition_direction: IndicatorDirection
  direction_overridden: boolean
  configured_weight: number
  effective_weight: number | null
  normalized_score: number | null
  weighted_contribution: number | null
}

export interface EvaluationPlanRunRow {
  rank: number | null
  target: EvaluationTarget & { name: string }
  score: number | null
  status: 'ranked' | 'excluded'
  missing_indicators: string[]
  exclusion_reasons?: EvaluationWarning[]
  values: EvaluationPlanValue[]
}

export interface EvaluationPlanRunResponse {
  plan_id: string
  plan_revision: number
  run_at: string
  as_of: string | null
  rows: EvaluationPlanRunRow[]
  ranked_count: number
  excluded_count: number
  normalization: {
    method: 'min_max_0_100' | string
    configured_weight_total: number
    effective_weight_total: number
    missing_policy: 'strict'
  }
  result_id?: string
  pagination?: {
    page: number
    page_size: number
    total: number
    page_count: number
    has_next: boolean
    expires_at: string
  }
  execution: FixedNjitExecutionAudit & {
    engine_version?: string
    data_generation?: string
    execution_lanes?: Record<string, number>
    worker_processes?: number
    numba_threads?: number
    shared_memory_bytes?: number
    combinations?: number
    cache?: Record<string, number>
    timings_ms?: Record<string, number>
    parallel_scoring?: boolean
  }
}

export interface InstrumentSearchItem {
  code: string | null
  ts_code: string | null
  name: string | null
  management: string | null
  found_date: string | null
  instrument_type: ProductKind | string | null
}

export interface InstrumentSearchResponse {
  items: InstrumentSearchItem[]
  total: number
  page: number
  page_size: number
  kind: ProductKind | 'all'
}

export type ProductConditionOperator = 'gte' | 'lte' | 'gt' | 'lt' | 'eq'

export interface ProductCondition {
  field: string
  operator: ProductConditionOperator
  value: string
}

export interface ProductConditionField {
  field: string
  label: string
  data_type: 'date' | 'number'
  unit_label?: string | null
  input_scale?: number
  source: 'fund_basic' | 'instrument_metrics_snapshot' | string
  available: boolean
}

export interface ProductConditionOperatorOption {
  value: ProductConditionOperator
  label: string
  symbol: string
}

export interface InstrumentFilterOption {
  value: string
  label: string
  count?: number
}

export interface InstrumentProductItem extends InstrumentSearchItem {
  custodian?: string | null
  fund_type?: string | null
  type?: string | null
  invest_type?: string | null
  qdii_type?: 'QDII' | '非QDII' | string | null
  qdii_source?: string | null
  market?: string | null
  status?: string | null
  list_date?: string | null
  issue_date?: string | null
  condition_values?: Record<string, number | string | null>
  snapshot_values?: Record<string, number | string | null>
  snapshot_value_dates?: Record<string, string | null>
  snapshot_statuses?: Record<string, string | null>
  snapshot_warnings?: Record<string, string | null>
}

export interface SnapshotMetricField extends ProductConditionField {
  unit: string
  description: string
}

export interface InstrumentProductsResponse {
  items: InstrumentProductItem[]
  total: number
  page: number
  page_size: number
  kind: ProductKind
  summary: {
    universe_total: number
    filtered_total: number
    active_count?: number | null
  }
  available_filters: Record<string, InstrumentFilterOption[]>
  condition_fields?: ProductConditionField[]
  condition_operators?: ProductConditionOperatorOption[]
  snapshot_metric_fields?: SnapshotMetricField[]
  selected_snapshot_metrics?: string[]
  snapshot?: { status?: string | null; as_of?: string | null }
  sort_by: string
  sort_dir: 'asc' | 'desc' | string
}

export interface InstrumentProductSelectionResponse {
  items: InstrumentProductItem[]
  total: number
  kind: ProductKind
}

export interface InstrumentProductQueryOptions {
  kind: ProductKind
  query?: string
  page?: number
  pageSize?: number
  sortBy?: string
  sortDir?: 'asc' | 'desc'
  filters?: Partial<Record<'fund_type' | 'type' | 'invest_type' | 'qdii_type' | 'market' | 'status' | 'management' | 'custodian', string[]>>
  conditions?: ProductCondition[]
  snapshotMetrics?: string[]
  signal?: AbortSignal
}

export interface ApiErrorDetail {
  code: string
  message: string
  field?: string | null
  diagnostics?: IndicatorDiagnostic[]
}

export class CustomIndicatorApiError extends Error {
  readonly status: number
  readonly code: string
  readonly field?: string | null
  readonly diagnostics: IndicatorDiagnostic[]

  constructor(status: number, detail: ApiErrorDetail) {
    super(detail.message)
    this.name = 'CustomIndicatorApiError'
    this.status = status
    this.code = detail.code
    this.field = detail.field
    this.diagnostics = detail.diagnostics ?? []
  }
}

const DEFAULT_ERROR: ApiErrorDetail = {
  code: 'REQUEST_FAILED',
  message: '请求失败，请稍后重试。',
}

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers)
  if (init?.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }
  const response = await fetch(path, { ...init, headers })
  if (!response.ok) {
    let payload: unknown
    try {
      payload = await response.json()
    } catch {
      payload = null
    }
    const maybeDetail = (payload as { detail?: unknown } | null)?.detail
    const detail = typeof maybeDetail === 'object' && maybeDetail !== null
      ? maybeDetail as ApiErrorDetail
      : typeof maybeDetail === 'string'
        ? { code: `HTTP_${response.status}`, message: maybeDetail }
        : { ...DEFAULT_ERROR, code: `HTTP_${response.status}` }
    throw new CustomIndicatorApiError(response.status, {
      ...DEFAULT_ERROR,
      ...detail,
    })
  }
  if (response.status === 204) {
    return undefined as T
  }
  return response.json() as Promise<T>
}

async function calculationRequest<T extends { execution: unknown }>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const result = await apiRequest<T>(path, init)
  assertFixedNjitExecution(result.execution, '指标与评价计算')
  return result
}

const downloadFilename = (disposition: string | null) => {
  if (!disposition) return 'indicator-calculation.xlsx'
  const encoded = disposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1]
  if (encoded) {
    try {
      return decodeURIComponent(encoded)
    } catch {
      return encoded
    }
  }
  return disposition.match(/filename="?([^";]+)"?/i)?.[1] ?? 'indicator-calculation.xlsx'
}

async function fileRequest(path: string, init?: RequestInit): Promise<DownloadedFile> {
  const headers = new Headers(init?.headers)
  if (init?.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  const response = await fetch(path, { ...init, headers })
  if (!response.ok) {
    let payload: unknown
    try {
      payload = await response.json()
    } catch {
      payload = null
    }
    const maybeDetail = (payload as { detail?: unknown } | null)?.detail
    const detail = typeof maybeDetail === 'object' && maybeDetail !== null
      ? maybeDetail as ApiErrorDetail
      : typeof maybeDetail === 'string'
        ? { code: `HTTP_${response.status}`, message: maybeDetail }
        : { ...DEFAULT_ERROR, code: `HTTP_${response.status}` }
    throw new CustomIndicatorApiError(response.status, { ...DEFAULT_ERROR, ...detail })
  }
  return {
    blob: await response.blob(),
    filename: downloadFilename(response.headers.get('Content-Disposition')),
  }
}

export const getCustomIndicatorMeta = () =>
  apiRequest<IndicatorMeta>('/api/custom-indicators/meta')

export const indicatorsForContext = (
  items: IndicatorDefinition[],
  context: IndicatorContextDomain,
) => items.filter((item) => (item.context_kind ?? 'single_product') === context)

export interface IndicatorListOptions {
  contextKind?: IndicatorContextDomain
  productKind?: ProductKind | 'portfolio'
  source?: IndicatorSource
  indicatorType?: IndicatorType
  /** Compatibility alias. Prefer indicatorType. */
  category?: string
  includeCompatibility?: boolean
}

export const listCustomIndicators = (options: IndicatorListOptions = {}) => {
  const params = new URLSearchParams()
  if (options.contextKind) params.set('context_kind', options.contextKind)
  if (options.productKind) params.set('product_kind', options.productKind)
  if (options.source) params.set('source', options.source)
  if (options.indicatorType) params.set('indicator_type', options.indicatorType)
  if (options.category) params.set('category', options.category)
  if (options.includeCompatibility) params.set('include_compatibility', 'true')
  const query = params.toString()
  return apiRequest<{ items: IndicatorDefinition[]; total: number }>(
    `/api/custom-indicators${query ? `?${query}` : ''}`,
  )
}

export const getCustomIndicator = (id: string) =>
  apiRequest<IndicatorDefinition>(`/api/custom-indicators/${encodeURIComponent(id)}`)

export const createCustomIndicator = (draft: IndicatorDraft) =>
  apiRequest<IndicatorDefinition>('/api/custom-indicators', {
    method: 'POST',
    body: JSON.stringify(draft),
  })

export const updateCustomIndicator = (id: string, draft: IndicatorDraft, revision: number) =>
  apiRequest<IndicatorDefinition>(`/api/custom-indicators/${encodeURIComponent(id)}`, {
    method: 'PUT',
    body: JSON.stringify({ ...draft, revision }),
  })

export const deleteCustomIndicator = (id: string, revision: number) =>
  apiRequest<{ deleted_id: string }>(`/api/custom-indicators/${encodeURIComponent(id)}?revision=${revision}`, {
    method: 'DELETE',
  })

export const getSnapshotIndicatorConfig = () =>
  apiRequest<SnapshotIndicatorConfig>('/api/custom-indicators/snapshot-config')

export const updateSnapshotIndicatorConfig = (
  revision: number,
  items: Array<Pick<SnapshotIndicatorConfigItem, 'indicator_id' | 'indicator_revision' | 'period' | 'channel_id' | 'reducer'>>,
) => apiRequest<SnapshotIndicatorConfig>('/api/custom-indicators/snapshot-config', {
  method: 'PUT',
  body: JSON.stringify({ revision, items }),
})

export const buildRollingScalarDraft = (input: {
  indicator_id: string
  indicator_revision?: number
  window_observations: number
  min_periods?: number
  name?: string
}) => apiRequest<{
  definition: IndicatorDraft
  validation: ValidationResponse
  source: {
    indicator_id: string
    indicator_revision: number
    name: string
  }
}>('/api/custom-indicators/rolling-scalar-draft', {
  method: 'POST',
  body: JSON.stringify(input),
})

export const validateCustomIndicator = (input: IndicatorDraft) =>
  apiRequest<ValidationResponse>('/api/custom-indicators/validate', {
    method: 'POST',
    body: JSON.stringify(input),
  })

export const deriveRollingSeriesIndicator = (input: {
  indicator_id: string
  indicator_revision: number
  window_observations: number
  name?: string
  description?: string
}) => apiRequest<RollingSeriesDerivationResponse>('/api/custom-indicators/derive-rolling-series', {
  method: 'POST',
  body: JSON.stringify(input),
})

/** v2 compositional helper. Older servers may not expose this endpoint yet. */
export const composeCustomIndicator = (input: ComposeIndicatorRequest) =>
  apiRequest<InferenceResponse>('/api/custom-indicators/compose', {
    method: 'POST',
    body: JSON.stringify(input),
  })

/** Infers the current expression shape and semantic warnings without evaluating data. */
export const inferCustomIndicator = (input: { expression: string; context: IndicatorContextDomain; dsl_version?: string; operator_registry_version?: string; parameter_schema?: SeriesParameterDefinition[] }) =>
  apiRequest<InferenceResponse>('/api/custom-indicators/infer', {
    method: 'POST',
    body: JSON.stringify(input),
  })

export const getVariableAvailability = (input: {
  kind?: ProductKind
  product_id?: string
  targets?: EvaluationTarget[]
  variable_ids: string[]
  period: string
  as_of?: string
}) => apiRequest<VariableAvailabilityResponse>('/api/custom-indicators/variables/availability', {
  method: 'POST',
  body: JSON.stringify(input),
})

export const evaluateCustomIndicators = (input: EvaluateIndicatorsRequest) =>
  calculationRequest<EvaluateIndicatorsResponse>('/api/custom-indicators/evaluate', {
    method: 'POST',
    body: JSON.stringify(input),
  })

export const evaluateTimeSeriesIndicators = (
  input: EvaluateTimeSeriesIndicatorsRequest,
) => calculationRequest<EvaluateTimeSeriesIndicatorsResponse>(
  '/api/custom-indicators/evaluate-series',
  {
    method: 'POST',
    body: JSON.stringify(input),
  },
)

export const exportCustomIndicatorExcel = (input: ExportIndicatorExcelRequest) =>
  fileRequest('/api/custom-indicators/export-excel', {
    method: 'POST',
    body: JSON.stringify(input),
  })

export const getPortfolioRuns = () =>
  apiRequest<{ items: PortfolioRun[] }>('/api/portfolio-runs')

export const evaluatePortfolioCustomIndicators = (input: EvaluatePortfolioIndicatorsRequest) =>
  calculationRequest<EvaluateIndicatorsResponse>('/api/custom-indicators/evaluate-portfolio', {
    method: 'POST',
    body: JSON.stringify(input),
  })

export const searchInstruments = (options: {
  kind?: ProductKind | 'all'
  query?: string
  page?: number
  pageSize?: number
  signal?: AbortSignal
} = {}) => {
  const params = new URLSearchParams({
    kind: options.kind ?? 'all',
    q: options.query?.trim() ?? '',
    page: String(options.page ?? 1),
    page_size: String(options.pageSize ?? 20),
    sort_by: 'name',
    sort_dir: 'asc',
  })
  return apiRequest<InstrumentSearchResponse>(`/api/instruments/search?${params.toString()}`, {
    signal: options.signal,
  })
}

const instrumentProductParams = (options: InstrumentProductQueryOptions, includePage: boolean) => {
  const params = new URLSearchParams({
    kind: options.kind,
    q: options.query?.trim() ?? '',
    sort_by: options.sortBy ?? 'name',
    sort_dir: options.sortDir ?? 'asc',
  })
  if (includePage) {
    params.set('page', String(options.page ?? 1))
    params.set('page_size', String(options.pageSize ?? 10))
  }
  Object.entries(options.filters ?? {}).forEach(([key, values]) => {
    values?.forEach((value) => params.append(key, value))
  })
  options.conditions?.forEach((condition) => {
    params.append('condition', `${condition.field}|${condition.operator}|${condition.value}`)
  })
  if (includePage) {
    options.snapshotMetrics?.forEach((metric) => params.append('snapshot_metric', metric))
  }
  return params
}

export const listInstrumentProducts = (options: InstrumentProductQueryOptions) => {
  const params = instrumentProductParams(options, true)
  return apiRequest<InstrumentProductsResponse>(`/api/instruments/products?${params.toString()}`, {
    signal: options.signal,
  })
}

export const selectAllInstrumentProducts = (options: InstrumentProductQueryOptions) => {
  const params = instrumentProductParams(options, false)
  return apiRequest<InstrumentProductSelectionResponse>(`/api/instruments/products/selection?${params.toString()}`, {
    signal: options.signal,
  })
}

export const listEvaluationPlans = (kind?: ProductKind) => {
  const query = kind ? `?kind=${encodeURIComponent(kind)}` : ''
  return apiRequest<{ items: EvaluationPlan[]; total: number }>(`/api/evaluation-plans${query}`)
}

export const getEvaluationPlan = (id: string) =>
  apiRequest<EvaluationPlan>(`/api/evaluation-plans/${encodeURIComponent(id)}`)

export const createEvaluationPlan = (draft: EvaluationPlanDraft) =>
  apiRequest<EvaluationPlan>('/api/evaluation-plans', {
    method: 'POST',
    body: JSON.stringify(draft),
  })

export const updateEvaluationPlan = (id: string, draft: EvaluationPlanDraft, revision: number) =>
  apiRequest<EvaluationPlan>(`/api/evaluation-plans/${encodeURIComponent(id)}`, {
    method: 'PUT',
    body: JSON.stringify({ ...draft, revision }),
  })

export const deleteEvaluationPlan = (id: string, revision: number) =>
  apiRequest<{ deleted_id: string }>(`/api/evaluation-plans/${encodeURIComponent(id)}?revision=${revision}`, {
    method: 'DELETE',
  })

export const runEvaluationPlan = (id: string, asOf?: string) =>
  calculationRequest<EvaluationPlanRunResponse>(`/api/evaluation-plans/${encodeURIComponent(id)}/run`, {
    method: 'POST',
    body: JSON.stringify(asOf ? { as_of: asOf } : {}),
  })

export const getEvaluationPlanRunPage = (resultId: string, page = 1, pageSize = 100) =>
  calculationRequest<EvaluationPlanRunResponse>(
    `/api/evaluation-plan-runs/${encodeURIComponent(resultId)}?page=${page}&page_size=${pageSize}`,
  )
