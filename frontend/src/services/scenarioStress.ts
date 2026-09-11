import {
  assertFixedNjitExecution,
  type FixedNjitExecutionAudit,
} from "../utils/fixedNjitExecution";

export type ScenarioMethod =
  | "historical_replay"
  | "factor_path"
  | "monte_carlo"
  | "regime_conditioned"
  | "reverse_stress";

export type ScenarioPublicationUsage =
  | "research_display"
  | "product_research"
  | "portfolio_backtest"
  | "taa"
  | "risk_monitoring";
export type MissingPolicy = "block" | "degrade";

export interface ScenarioFactor {
  id: string;
  label: string;
  unit: string;
  description?: string;
}

export interface ScenarioAsset {
  id: string;
  label: string;
  description?: string;
}

export interface ScenarioPortfolio {
  id: string;
  name: string;
  weights: Record<string, number>;
}

export type HistoricalReturnTransform = "simple_return" | "forward_value";

export interface HistoricalEvaluationAssetReturnSource {
  kind: "historical_evaluation_targets";
  run_content_hash: string;
  evaluation_artifact_checksum: string;
  asset_target_map: Record<string, {
    target_id: string;
    return_transform: HistoricalReturnTransform;
  }>;
  sampling: "empirical_bootstrap";
  minimum_observations_per_state: number;
  inline_policy: "forbid" | "override";
}

export interface ScenarioDefinition {
  id?: string;
  revision?: number;
  status?: string;
  name: string;
  description: string;
  template_id?: string;
  method: ScenarioMethod;
  horizon: number;
  initial_nav?: number;
  usage_intent?: ScenarioPublicationUsage;
  factors: ScenarioFactor[];
  assets: ScenarioAsset[];
  portfolios: ScenarioPortfolio[];
  mapping: {
    factor_to_asset: Record<string, Record<string, number | null>>;
    missing_policy: MissingPolicy;
    minimum_coverage: number;
    response_space: "simple_return" | "log_return" | "direct_simple_return";
  };
  scenario: Record<string, unknown>;
  limits: ScenarioLimit[];
  created_at?: string;
  updated_at?: string;
}

export interface ScenarioTemplate {
  id: string;
  name: string;
  description?: string;
  category?: string;
  phase?: string;
  definition?: ScenarioDefinition;
  default_definition?: ScenarioDefinition;
}

export interface ScenarioLimit {
  id: string;
  label: string;
  metric:
    | "terminal_return"
    | "max_drawdown"
    | "worst_step_return"
    | "var_95"
    | "es_95"
    | "loss_probability"
    | "target_hit_probability";
  operator: "gt" | "gte" | "lt" | "lte";
  threshold: number;
}

export interface ScenarioMethodMeta {
  id: ScenarioMethod;
  label: string;
  description?: string;
  probabilistic?: boolean;
}

export interface ScenarioApplicationTarget {
  id: ScenarioPublicationUsage;
  label?: string;
  name?: string;
  description?: string;
}

export interface ScenarioStressMeta {
  schema_version?: string;
  templates: ScenarioTemplate[];
  methods: ScenarioMethodMeta[];
  application_targets: ScenarioApplicationTarget[];
  limits?: Record<string, number | string>;
  factor_catalog?: ScenarioFactor[];
  asset_catalog?: ScenarioAsset[];
  mapping_contract?: {
    orientation?: string;
    missing_policy?: MissingPolicy[];
    degrade_semantics?: string;
    response_space_by_method?: Partial<
      Record<ScenarioMethod, ScenarioDefinition["mapping"]["response_space"]>
    >;
  };
  historical_regime_distribution_contract?: {
    location?: string;
    kind?: "historical_evaluation_targets";
    required_content_locks?: string[];
    sampling?: Array<"empirical_bootstrap">;
    return_transforms?: HistoricalReturnTransform[];
    inline_policies?: Array<"forbid" | "override">;
    default_inline_policy?: "forbid";
    state_alignment?: string;
    joint_sampling?: string;
    missing_value_policy?: string;
  };
}

export interface ScenarioPathPoint {
  step: number;
  date?: string;
  return: number;
  nav: number;
  drawdown: number;
  factor_shocks?: Record<string, number>;
  asset_returns: Record<string, number | null>;
  contributions: Record<string, number | null>;
  coverage_ratio?: number;
}

export interface ScenarioFan {
  steps: number[];
  quantiles: Record<string, number[]>;
}

export interface ScenarioDistribution {
  fan: ScenarioFan;
  sample_paths?: Array<
    number[] | { id?: string; values?: number[]; nav?: number[] }
  >;
  terminal: {
    p05?: number;
    p25?: number;
    p50?: number;
    p75?: number;
    p95?: number;
    return_p05?: number;
    return_p50?: number;
    return_p95?: number;
    var_95?: number;
    es_95?: number;
    loss_probability?: number;
    target_hit_probability?: number;
    average_max_drawdown?: number;
    p95_max_drawdown?: number;
  };
  path_count?: number;
  seed?: number;
  state_probabilities?: Array<Record<string, number>>;
  sample_state_paths?: string[][];
}

export interface ReverseStressCandidate {
  kind: string;
  label: string;
  factor_shocks: Record<string, number>;
  asset_impacts: Record<string, number | null>;
  contributions: Record<string, number | null>;
  portfolio_return: number;
  max_drawdown: number;
  severity: number;
  meets_target: boolean;
}

export interface ScenarioCoverage {
  status: "complete" | "degraded" | string;
  ratio: number;
  covered_assets: string[];
  missing_assets: string[];
  policy: MissingPolicy;
  minimum_coverage?: number;
  renormalized?: boolean;
  by_state?: Record<string, ScenarioCoverage>;
}

export interface ScenarioResult {
  portfolio_id: string;
  name: string;
  coverage: ScenarioCoverage;
  summary: {
    initial_nav?: number;
    terminal_nav?: number;
    terminal_return?: number;
    max_drawdown?: number;
    worst_step_return?: number | null;
    breach_count?: number;
    recovery_steps?: number | null;
    recovery_step?: number | null;
    recovered?: boolean | null;
    max_drawdown_start_step?: number | null;
    max_drawdown_trough_step?: number | null;
    first_breach_step?: number | null;
    first_breach_date?: string | null;
    var_95?: number | null;
    es_95?: number | null;
    loss_probability?: number | null;
    target_hit_probability?: number | null;
  };
  path?: ScenarioPathPoint[];
  contributions: {
    by_asset: Record<string, number | null>;
    method: string;
  };
  distribution?: ScenarioDistribution;
  reverse_stress?: {
    target_metric: string;
    threshold: number;
    candidates: ReverseStressCandidate[];
  };
  limits?: Array<
    {
      id?: string;
      label?: string;
      breached?: boolean | null;
      first_breach_step?: number | null;
      first_breach_date?: string | null;
      evaluation_scope?: string;
      value?: number | null;
      threshold?: number | null;
    }
  >;
}

export interface ScenarioPublication {
  id: string;
  usage: ScenarioPublicationUsage;
  published_at: string;
  definition_revision: number;
  run_id: string;
  note?: string;
  run_content_hash?: string;
  portfolio_ids?: string[];
  gate?: string;
}

export interface ScenarioApplicationBinding {
  usage: ScenarioPublicationUsage;
  name?: string;
  path?: string;
  status?: string;
  run_id?: string;
  revision?: number;
  definition_revision?: number;
  publication_id?: string;
  run_content_hash?: string;
}

export interface ScenarioComputeAudit extends FixedNjitExecutionAudit {
  fully_warmed?: boolean;
  kernel_coverage?: string;
  fingerprint?: string;
  optimized_third_party_model?: Record<string, unknown> | null;
}

export interface ScenarioStressRun {
  id: string;
  name?: string;
  method: ScenarioMethod;
  probabilistic: boolean;
  definition_id: string | null;
  definition_revision: number | null;
  definition_source?: string;
  definition_snapshot_hash?: string;
  definition?: ScenarioDefinition;
  created_at?: string;
  schema_version?: string;
  batch?: boolean;
  batch_size?: number;
  data_snapshot?: Record<string, unknown>;
  compute_audit?: ScenarioComputeAudit;
  results: ScenarioResult[];
  mapping_diagnostics?:
    | Record<string, unknown>
    | Array<{ code?: string; message: string; level?: string; field?: string }>;
  diagnostics: Array<{
    code?: string;
    message: string;
    level?: string;
    field?: string;
  }>;
  content_hash?: string;
  immutable: boolean;
  publications: ScenarioPublication[];
  application_bindings?: ScenarioApplicationBinding[];
  governance?: {
    publish_eligible_usages?: ScenarioPublicationUsage[];
    publication_blockers?: string[];
    coverage_gate?: string;
    probability_semantics?: string;
  };
}

export interface ScenarioRunComparison {
  run_ids: string[];
  reference_run_id?: string;
  rows?: Array<{
    run_id: string;
    method?: ScenarioMethod;
    portfolio_id?: string;
    terminal_return?: number | null;
    p05?: number | null;
    es_95?: number | null;
    max_drawdown?: number | null;
    breach_count?: number | null;
    coverage?: number | null;
  }>;
  pairwise?: Array<Record<string, unknown>>;
  runs?: Array<{
    run_id: string;
    name?: string;
    method?: ScenarioMethod;
    probabilistic?: boolean;
    metrics: Record<string, {
      terminal_return?: number | null;
      max_drawdown?: number | null;
      var_95?: number | null;
      es_95?: number | null;
      loss_probability?: number | null;
      coverage_ratio?: number | null;
      breach_count?: number | null;
    }>;
    deltas_to_reference?: Record<string, Record<string, number | null>>;
  }>;
  execution: FixedNjitExecutionAudit;
}

export interface ScenarioWeightSummary {
  weight_count: number;
  total_weight: number;
  net_exposure: number;
  gross_exposure: number;
  largest_absolute_weight: number;
  finite: boolean;
  sums_to_one: boolean;
  single_weight_valid: boolean;
  gross_limit_valid: boolean;
  within_tolerance: boolean;
  status_code: number;
  limits: {
    expected_net_exposure: number;
    absolute_tolerance: number;
    maximum_absolute_weight: number;
    maximum_gross_exposure: number;
  };
  execution: FixedNjitExecutionAudit;
}

export class ScenarioStressApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
    this.name = "ScenarioStressApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try {
      const body = await response.json();
      const detail = body?.detail;
      if (typeof detail?.message === "string") message = detail.message;
      else if (typeof detail === "string") message = detail;
      else if (typeof body?.message === "string") message = body.message;
      else if (Array.isArray(detail)) {
        message = "提交内容未通过校验，请检查必填项与参数范围。";
      }
    } catch { /* retain stable Chinese fallback */ }
    throw new ScenarioStressApiError(response.status, message);
  }
  return response.json() as Promise<T>;
}

function listFrom<T>(value: unknown, key = "items"): T[] {
  if (Array.isArray(value)) return value as T[];
  if (
    value && typeof value === "object" &&
    Array.isArray((value as Record<string, unknown>)[key])
  ) {
    return (value as Record<string, unknown>)[key] as T[];
  }
  return [];
}

function validatedScenarioRun(
  run: ScenarioStressRun,
  label: string,
): ScenarioStressRun {
  assertFixedNjitExecution(run.compute_audit, label);
  return run;
}

export async function getScenarioStressMeta(): Promise<ScenarioStressMeta> {
  const raw = await request<Record<string, unknown>>(
    "/api/scenario-stress/meta",
  );
  return {
    schema_version: typeof raw.schema_version === "string"
      ? raw.schema_version
      : undefined,
    templates: listFrom<ScenarioTemplate>(raw.templates),
    methods: listFrom<ScenarioMethodMeta>(raw.methods ?? raw.scenario_methods),
    application_targets: listFrom<ScenarioApplicationTarget>(
      raw.application_targets ?? raw.usages,
    ),
    limits: raw.limits && typeof raw.limits === "object"
      ? raw.limits as Record<string, number | string>
      : undefined,
    factor_catalog: listFrom<ScenarioFactor>(raw.factor_catalog ?? raw.factors),
    asset_catalog: listFrom<ScenarioAsset>(raw.asset_catalog ?? raw.assets),
    mapping_contract:
      raw.mapping_contract && typeof raw.mapping_contract === "object"
        ? raw.mapping_contract as ScenarioStressMeta["mapping_contract"]
        : undefined,
  };
}

export async function listScenarioStressDefinitions(): Promise<
  ScenarioDefinition[]
> {
  return listFrom<ScenarioDefinition>(
    await request<unknown>("/api/scenario-stress/definitions"),
  );
}

export async function getScenarioStressDefinition(
  id: string,
  revision?: number,
): Promise<ScenarioDefinition> {
  const query = revision ? `?revision=${revision}` : "";
  return request<ScenarioDefinition>(
    `/api/scenario-stress/definitions/${encodeURIComponent(id)}${query}`,
  );
}

export async function createScenarioStressDefinition(
  definition: ScenarioDefinition,
): Promise<ScenarioDefinition> {
  return request<ScenarioDefinition>("/api/scenario-stress/definitions", {
    method: "POST",
    body: JSON.stringify(definition),
  });
}

export async function updateScenarioStressDefinition(
  definition: ScenarioDefinition,
): Promise<ScenarioDefinition> {
  if (!definition.id || !definition.revision) {
    throw new Error("保存修订版前需要定义 ID 和修订号。");
  }
  return request<ScenarioDefinition>(
    `/api/scenario-stress/definitions/${encodeURIComponent(definition.id)}`,
    {
      method: "PUT",
      body: JSON.stringify(definition),
    },
  );
}

export async function runScenarioStress(
  definition: ScenarioDefinition | { id: string; revision: number },
): Promise<ScenarioStressRun> {
  const run = await request<ScenarioStressRun>("/api/scenario-stress/run", {
    method: "POST",
    body: JSON.stringify({ definition }),
  });
  return validatedScenarioRun(run, "情景模拟运行");
}

export async function listScenarioStressRuns(
  definitionId?: string,
): Promise<ScenarioStressRun[]> {
  const query = definitionId
    ? `?definition_id=${encodeURIComponent(definitionId)}`
    : "";
  return listFrom<ScenarioStressRun>(
    await request<unknown>(`/api/scenario-stress/runs${query}`),
  ).map((run) => validatedScenarioRun(run, "情景模拟历史运行"));
}

export async function getScenarioStressRun(
  id: string,
): Promise<ScenarioStressRun> {
  const run = await request<ScenarioStressRun>(
    `/api/scenario-stress/runs/${encodeURIComponent(id)}`,
  );
  return validatedScenarioRun(run, "情景模拟运行");
}

export async function publishScenarioStressRun(
  id: string,
  usage: ScenarioPublicationUsage | ScenarioPublicationUsage[],
  note = "",
): Promise<
  {
    run_id: string;
    publications: ScenarioPublication[];
    application_bindings?: ScenarioApplicationBinding[];
  }
> {
  return request(
    `/api/scenario-stress/runs/${encodeURIComponent(id)}/publish`,
    {
      method: "POST",
      body: JSON.stringify({ usage, ...(note ? { note } : {}) }),
    },
  );
}

export async function compareScenarioStressRuns(
  runIds: string[],
  referenceRunId?: string,
): Promise<ScenarioRunComparison> {
  const result = await request<ScenarioRunComparison>(
    "/api/scenario-stress/compare",
    {
      method: "POST",
      body: JSON.stringify({
        run_ids: runIds,
        ...(referenceRunId ? { reference_run_id: referenceRunId } : {}),
      }),
    },
  );
  assertFixedNjitExecution(result.execution, "情景运行比较");
  if (!result.rows?.length && result.runs?.length) {
    result.rows = result.runs.flatMap((run) =>
      Object.entries(run.metrics ?? {}).map(([portfolioId, metrics]) => ({
        run_id: run.run_id,
        method: run.method,
        portfolio_id: portfolioId,
        terminal_return: metrics.terminal_return,
        max_drawdown: metrics.max_drawdown,
        breach_count: metrics.breach_count,
        coverage: metrics.coverage_ratio,
        es_95: metrics.es_95,
      }))
    );
  }
  return result;
}

export async function batchRunScenarioStress(
  definition: ScenarioDefinition | { id: string; revision: number },
): Promise<ScenarioStressRun> {
  const run = await request<ScenarioStressRun>("/api/scenario-stress/batch-run", {
    method: "POST",
    body: JSON.stringify({ definition }),
  });
  return validatedScenarioRun(run, "情景批量压测运行");
}

export async function summarizeScenarioStressWeights(
  weights: Record<string, number>,
  signal?: AbortSignal,
): Promise<ScenarioWeightSummary> {
  const summary = await request<ScenarioWeightSummary>(
    "/api/scenario-stress/weight-summary",
    {
      method: "POST",
      body: JSON.stringify({ weights }),
      signal,
    },
  );
  assertFixedNjitExecution(summary.execution, "情景组合权重核验");
  return summary;
}
