import {
  type KeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import ReactECharts from "echarts-for-react";
import { EmptyState } from "../components/ui";
import type { EChartsOption } from "echarts";
import {
  type HistoricalRegimeRun,
  listHistoricalRegimeRuns,
} from "../services/historicalRegimes";
import {
  batchRunScenarioStress,
  compareScenarioStressRuns,
  createScenarioStressDefinition,
  getScenarioStressDefinition,
  getScenarioStressMeta,
  listScenarioStressDefinitions,
  listScenarioStressRuns,
  publishScenarioStressRun,
  type ReverseStressCandidate,
  runScenarioStress,
  type ScenarioApplicationBinding,
  type ScenarioDefinition,
  type ScenarioLimit,
  type ScenarioMethod,
  type ScenarioPortfolio,
  type ScenarioPublicationUsage,
  type ScenarioResult,
  type ScenarioRunComparison,
  type ScenarioStressMeta,
  type ScenarioStressRun,
  type ScenarioTemplate,
  type ScenarioWeightSummary,
  summarizeScenarioStressWeights,
  updateScenarioStressDefinition,
} from "../services/scenarioStress";

type WorkspaceTab = "configure" | "result" | "compare" | "publish";

const tabs: Array<{ id: WorkspaceTab; label: string; helper: string }> = [
  { id: "configure", label: "情景定义", helper: "方法、路径与研究对象" },
  { id: "result", label: "结果与归因", helper: "净值、尾部与约束" },
  { id: "compare", label: "版本 / 运行对比", helper: "跨版本和跨情景复核" },
  { id: "publish", label: "批量压测与发布", helper: "组合覆盖和应用关系" },
];

const methodLabels: Record<ScenarioMethod, string> = {
  historical_replay: "历史事件重演",
  factor_path: "确定性因子路径",
  monte_carlo: "蒙特卡罗模拟",
  regime_conditioned: "历史状态条件模拟",
  reverse_stress: "反向压力测试",
};

const methodDescriptions: Record<ScenarioMethod, string> = {
  historical_replay: "复现真实历史窗口，并按当前权重与暴露重新计价。",
  factor_path: "逐期设定宏观或市场因子冲击，得到一条可解释的确定性路径。",
  monte_carlo: "冻结随机种子与路径数，输出净值分位扇形和尾部损失。",
  regime_conditioned: "引用已发布历史状态，按状态条件校准收益、波动和相关性。",
  reverse_stress: "从损失或回撤目标反推多组能触发阈值的因子组合。",
};

const usageLabels: Record<ScenarioPublicationUsage, string> = {
  research_display: "研究展示",
  product_research: "产品研究",
  portfolio_backtest: "组合回测",
  taa: "TAA 研究",
  risk_monitoring: "风险监控",
};

const historicalUsageLabels = {
  research_display: "研究展示",
  product_research: "产品研究",
  formal_backtest: "正式回测",
  taa: "TAA",
};

const limitMetricLabels: Record<ScenarioLimit["metric"], string> = {
  terminal_return: "期末收益",
  max_drawdown: "最大回撤",
  worst_step_return: "最差单步收益",
  var_95: "VaR 95%",
  es_95: "ES 95%",
  loss_probability: "损失概率",
  target_hit_probability: "目标达成概率",
};

const deterministicMethods = new Set<ScenarioMethod>([
  "historical_replay",
  "factor_path",
  "reverse_stress",
]);
const probabilisticLimitMetrics = new Set<ScenarioLimit["metric"]>([
  "var_95",
  "es_95",
  "loss_probability",
  "target_hit_probability",
]);
const chartColors = [
  "#4f46e5",
  "#0891b2",
  "#059669",
  "#d97706",
  "#e11d48",
  "#7c3aed",
];

function responseSpaceFor(
  method: ScenarioMethod,
): ScenarioDefinition["mapping"]["response_space"] {
  if (method === "monte_carlo") return "log_return";
  if (method === "factor_path" || method === "reverse_stress") {
    return "simple_return";
  }
  return "direct_simple_return";
}

function supportsHistoricalUsage(
  run: HistoricalRegimeRun,
  publication: HistoricalRegimeRun["publications"][number],
  usage: ScenarioPublicationUsage,
) {
  const causality = run.causality
  const causalRealtime = run.mode === "realtime" &&
    causality?.is_causal === true &&
    causality.repaints !== true &&
    causality.uses_future_data === false;
  if (usage === "research_display") return true;
  if (usage === "product_research") {
    return publication.usage === "product_research";
  }
  if (usage === "portfolio_backtest") {
    return publication.usage === "formal_backtest" && causalRealtime;
  }
  if (usage === "taa") {
    return publication.usage === "taa" && causalRealtime;
  }
  return causalRealtime &&
    (publication.usage === "formal_backtest" || publication.usage === "taa");
}

type HistoricalEvaluationTargetOption = { id: string; name: string };

function historicalEvaluationDistribution(run: HistoricalRegimeRun | undefined) {
  const artifact = run?.artifact_manifest?.evaluation_targets;
  const checksum = typeof artifact?.checksum === "string" ? artifact.checksum : "";
  const artifactId = typeof artifact?.artifact_id === "string" ? artifact.artifact_id : "";
  const contentHash = typeof run?.content_hash === "string" ? run.content_hash : "";
  const arrayTargets = new Set(
    (artifact?.arrays ?? []).flatMap((item) =>
      item?.port === "value" && typeof item.node_id === "string"
        ? [item.node_id]
        : []
    ),
  );
  const targets: HistoricalEvaluationTargetOption[] = Object.entries(
    run?.evaluation_results ?? {},
  ).flatMap(([key, value]) => {
    const id = String(value?.id || key);
    if (!id || !arrayTargets.has(id)) return [];
    return [{ id, name: String(value?.name || id) }];
  });
  const available = run?.schema_version === "2.0" &&
    /^[a-f0-9]{64}$/.test(contentHash) &&
    /^sha256:[a-f0-9]{64}$/.test(checksum) &&
    artifactId === `regime-output-sha256-${checksum.slice(7)}` &&
    artifact?.format === "npz" &&
    artifact?.schema_version === "regime-node-output-v1" &&
    arrayTargets.size > 0 &&
    targets.length > 0;
  return { artifact, checksum, contentHash, targets, available };
}

function cx(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

function cloneDefinition(definition: ScenarioDefinition): ScenarioDefinition {
  return JSON.parse(JSON.stringify(definition)) as ScenarioDefinition;
}

function definitionSignature(definition: ScenarioDefinition | null) {
  if (!definition) return "";
  const copy = cloneDefinition(definition);
  delete copy.updated_at;
  delete copy.created_at;
  delete copy.status;
  return JSON.stringify(copy);
}

function formatPercent(value: number | null | undefined, digits = 2) {
  return typeof value === "number" && Number.isFinite(value)
    ? new Intl.NumberFormat("zh-CN", {
      style: "percent",
      maximumFractionDigits: digits,
    }).format(value)
    : "—";
}

function formatNumber(value: number | null | undefined, digits = 2) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";
}

function normalizeTemplate(
  template: ScenarioTemplate,
): ScenarioDefinition | null {
  const source = template.definition ?? template.default_definition;
  if (!source) return null;
  const definition = cloneDefinition(source);
  definition.template_id = definition.template_id || template.id;
  definition.name ||= template.name;
  definition.description ||= template.description ?? "";
  definition.horizon = Number(definition.horizon || 1);
  definition.factors = Array.isArray(definition.factors)
    ? definition.factors
    : [];
  definition.assets = Array.isArray(definition.assets) ? definition.assets : [];
  definition.portfolios = Array.isArray(definition.portfolios)
    ? definition.portfolios
    : [];
  definition.mapping = {
    ...definition.mapping,
    factor_to_asset: definition.mapping?.factor_to_asset ?? {},
    missing_policy: definition.mapping?.missing_policy ?? "block",
    minimum_coverage: definition.mapping?.minimum_coverage ?? 1,
    response_space: definition.mapping?.response_space ??
      responseSpaceFor(definition.method),
  };
  definition.scenario =
    definition.scenario && typeof definition.scenario === "object"
      ? definition.scenario
      : {};
  definition.limits = Array.isArray(definition.limits) ? definition.limits : [];
  return definition;
}

function SectionHeading(
  { eyebrow, title, detail }: {
    eyebrow: string;
    title: string;
    detail: string;
  },
) {
  return (
    <div>
      <p className="text-xs font-bold uppercase tracking-[0.16em] text-accent-600">
        {eyebrow}
      </p>
      <h3 className="mt-1 text-lg font-bold text-slate-950">{title}</h3>
      <p className="mt-1 text-xs leading-5 text-slate-600">{detail}</p>
    </div>
  );
}

function Message({ error, notice }: { error: string; notice: string }) {
  if (!error && !notice) return null;
  return (
    <p
      role={error ? "alert" : "status"}
      className={cx(
        "rounded-xl border px-4 py-3 text-sm",
        error
          ? "border-rose-200 bg-rose-50 text-rose-900"
          : "border-accent-200 bg-accent-50 text-accent-900",
      )}
    >
      {error || notice}
    </p>
  );
}

function DefinitionBar(
  { definitions, draft, saved, dirty, busy, onSelect, onSave, onRun }: {
    definitions: ScenarioDefinition[];
    draft: ScenarioDefinition;
    saved: ScenarioDefinition | null;
    dirty: boolean;
    busy: string;
    onSelect: (id: string) => void;
    onSave: () => void;
    onRun: () => void;
  },
) {
  const runnable = Boolean(saved?.id && saved.revision && !dirty);
  const responseSpaceLabel = draft.mapping.response_space === "log_return"
    ? "对数收益响应"
    : draft.mapping.response_space === "simple_return"
    ? "简单收益响应"
    : "资产收益直接输入";
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="grid gap-3 xl:grid-cols-[minmax(260px,1fr)_auto] xl:items-end">
        <label className="text-sm font-semibold text-slate-700">
          已保存定义
          <select
            aria-label="已保存情景定义"
            value={draft.id ?? ""}
            onChange={(event) => onSelect(event.target.value)}
            className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3 focus:border-accent-500 focus:outline-none focus:ring-2 focus:ring-accent-500"
          >
            <option value="">新建定义</option>
            {definitions.map((definition) => (
              <option
                key={`${definition.id}-${definition.revision}`}
                value={definition.id}
              >
                {definition.name} · R{definition.revision ?? "—"}
              </option>
            ))}
          </select>
        </label>
        <div className="flex flex-wrap gap-2">
          <span className="inline-flex min-h-11 items-center rounded-xl bg-slate-100 px-3 text-xs font-bold text-slate-700">
            映射口径：{responseSpaceLabel}
          </span>
          <span
            className={cx(
              "inline-flex min-h-11 items-center rounded-xl px-3 text-xs font-bold",
              dirty
                ? "bg-amber-100 text-amber-900"
                : "bg-emerald-100 text-emerald-800",
            )}
          >
            {dirty
              ? "参数已变化 · 结果会过期"
              : `已保存版本 R${saved?.revision ?? "—"}`}
          </span>
          <button
            type="button"
            disabled={Boolean(busy)}
            onClick={onSave}
            className="min-h-11 rounded-xl border border-accent-300 px-4 text-sm font-bold text-accent-700 hover:bg-accent-50 disabled:opacity-50"
          >
            {busy === "save"
              ? "保存中…"
              : saved?.id
              ? "保存新修订"
              : "保存定义"}
          </button>
          <button
            type="button"
            disabled={!runnable || Boolean(busy)}
            onClick={onRun}
            className="min-h-11 rounded-xl bg-accent-600 px-4 text-sm font-bold text-white hover:bg-accent-500 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {busy === "run" ? "运行中…" : "按当前版本运行"}
          </button>
        </div>
      </div>
      {!runnable
        ? (
          <p className="mt-2 text-xs text-amber-800">
            请先保存当前参数，再按不可变版本运行；页面内未保存草稿不能发布。
          </p>
        )
        : null}
    </section>
  );
}

function BlueprintPanel(
  { meta, draft, onChange }: {
    meta: ScenarioStressMeta;
    draft: ScenarioDefinition;
    onChange: (next: ScenarioDefinition) => void;
  },
) {
  const methods = meta.methods.length
    ? meta.methods.map((item) => item.id)
    : Object.keys(methodLabels) as ScenarioMethod[];
  return (
    <section className="space-y-5">
      <SectionHeading
        eyebrow="01 / Blueprint"
        title="选择模板和计算方法"
        detail="模板只是可复制起点；保存后形成独立版本，运行结果冻结数据、参数与随机种子。"
      />
      <div
        role="radiogroup"
        aria-label="情景压力模板"
        className="grid gap-2 sm:grid-cols-2"
      >
        {meta.templates.map((template) => (
          <button
            key={template.id}
            type="button"
            role="radio"
            aria-checked={draft.template_id === template.id}
            onClick={() => {
              const next = normalizeTemplate(template);
              if (next) onChange(next);
            }}
            className={cx(
              "min-h-24 rounded-xl border p-3 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500",
              draft.template_id === template.id
                ? "border-accent-500 bg-accent-50"
                : "border-slate-200 hover:border-slate-400",
            )}
          >
            <span className="text-xs font-bold uppercase tracking-wide text-accent-600">
              {template.category || template.phase || "情景模板"}
            </span>
            <span className="mt-1 block text-sm font-bold text-slate-950">
              {template.name}
            </span>
            <span className="mt-1 block text-xs leading-5 text-slate-600">
              {template.description ||
                methodDescriptions[
                  (template.definition ?? template.default_definition)
                    ?.method ?? "factor_path"
                ]}
            </span>
          </button>
        ))}
      </div>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        <label className="text-sm font-semibold text-slate-700">
          定义名称<input
            aria-label="定义名称"
            value={draft.name}
            onChange={(event) =>
              onChange({ ...draft, name: event.target.value })}
            className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 px-3"
          />
        </label>
        <label className="text-sm font-semibold text-slate-700">
          说明<input
            aria-label="定义说明"
            value={draft.description}
            onChange={(event) =>
              onChange({ ...draft, description: event.target.value })}
            className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 px-3"
          />
        </label>
        <label className="text-sm font-semibold text-slate-700">
          预期应用用途<select
            aria-label="预期应用用途"
            value={draft.usage_intent ?? "research_display"}
            onChange={(event) =>
              onChange({
                ...draft,
                usage_intent: event.target.value as ScenarioPublicationUsage,
              })}
            className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3"
          >
            {(meta.application_targets.length
              ? meta.application_targets
              : Object.keys(usageLabels).map((id) => ({
                id: id as ScenarioPublicationUsage,
                label: usageLabels[id as ScenarioPublicationUsage],
              }))).map((target) => (
                <option key={target.id} value={target.id}>
                  {target.label ?? usageLabels[target.id]}
                </option>
              ))}
          </select>
        </label>
      </div>
      <fieldset>
        <legend className="text-xs font-bold text-slate-600">情景方法</legend>
        <div className="mt-2 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
          {methods.map((method) => (
            <button
              key={method}
              type="button"
              aria-pressed={draft.method === method}
              onClick={() =>
                onChange({
                  ...draft,
                  method,
                  horizon: method === "reverse_stress" ? 1 : draft.horizon,
                  mapping: {
                    ...draft.mapping,
                    response_space:
                      meta.mapping_contract?.response_space_by_method?.[
                        method
                      ] ?? responseSpaceFor(method),
                  },
                  limits: draft.limits.filter((item) =>
                    deterministicMethods.has(method)
                      ? !probabilisticLimitMetrics.has(item.metric)
                      : item.metric !== "worst_step_return"
                  ),
                })}
              className={cx(
                "min-h-24 rounded-xl border p-3 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500",
                draft.method === method
                  ? "border-slate-950 bg-slate-950 text-white"
                  : "border-slate-200 bg-white hover:border-slate-400",
              )}
            >
              <span className="block text-sm font-bold">
                {methodLabels[method]}
              </span>
              <span
                className={cx(
                  "mt-1 block text-xs leading-5",
                  draft.method === method ? "text-slate-600" : "text-slate-600",
                )}
              >
                {methodDescriptions[method]}
              </span>
            </button>
          ))}
        </div>
      </fieldset>
    </section>
  );
}

function PathPanel(
  { draft, historicalRuns, onChange }: {
    draft: ScenarioDefinition;
    historicalRuns: HistoricalRegimeRun[];
    onChange: (next: ScenarioDefinition) => void;
  },
) {
  const scenario = draft.scenario;
  const update = (patch: Record<string, unknown>) =>
    onChange({ ...draft, scenario: { ...scenario, ...patch } });
  const shocks =
    (scenario.shocks && typeof scenario.shocks === "object"
      ? scenario.shocks
      : {}) as Record<string, number>;
  const means =
    (scenario.factor_means && typeof scenario.factor_means === "object"
      ? scenario.factor_means
      : {}) as Record<string, number>;
  const volatilities = (scenario.factor_volatilities &&
      typeof scenario.factor_volatilities === "object"
    ? scenario.factor_volatilities
    : {}) as Record<string, number>;
  const historicalRef = (scenario.historical_run_ref &&
      typeof scenario.historical_run_ref === "object"
    ? scenario.historical_run_ref
    : {}) as Record<string, string>;
  const transition = (scenario.transition &&
      typeof scenario.transition === "object"
    ? scenario.transition
    : {}) as Record<string, unknown>;
  const assetReturnSource = (transition.asset_return_source &&
      typeof transition.asset_return_source === "object"
    ? transition.asset_return_source
    : {}) as Record<string, unknown>;
  const assetTargetMap = (assetReturnSource.asset_target_map &&
      typeof assetReturnSource.asset_target_map === "object"
    ? assetReturnSource.asset_target_map
    : {}) as Record<string, Record<string, string>>;
  const updateTransition = (patch: Record<string, unknown>) =>
    update({ transition: { ...transition, ...patch } });
  const publishedHistoricalRuns = historicalRuns.flatMap((item) =>
    (item.publications ?? []).map((publication) => ({ run: item, publication }))
  ).filter(({ run, publication }) =>
    supportsHistoricalUsage(
      run,
      publication,
      draft.usage_intent ?? "research_display",
    )
  );
  const hasHistoricalPublications = historicalRuns.some((item) =>
    (item.publications?.length ?? 0) > 0
  );
  const selectedHistorical = publishedHistoricalRuns.find((item) =>
    item.run.id === historicalRef.run_id &&
    item.publication.id === historicalRef.publication_id
  );
  const historicalDistribution = historicalEvaluationDistribution(
    selectedHistorical?.run,
  );
  const historicalDistributionLocked = historicalDistribution.available &&
    selectedHistorical?.publication.run_content_hash ===
      historicalDistribution.contentHash;
  const usesHistoricalDistribution =
    assetReturnSource.kind === "historical_evaluation_targets";
  const sourceMatchesSelectedRun = !usesHistoricalDistribution || (
    assetReturnSource.run_content_hash === historicalDistribution.contentHash &&
    assetReturnSource.evaluation_artifact_checksum === historicalDistribution.checksum
  );
  const unmappedHistoricalAssets = usesHistoricalDistribution
    ? draft.assets.filter((asset) => !assetTargetMap[asset.id]?.target_id)
    : [];
  const buildHistoricalDistributionSource = () => {
    if (!selectedHistorical || !historicalDistributionLocked) return;
    const targetIds = new Set(
      historicalDistribution.targets.map((target) => target.id),
    );
    const nextMap = Object.fromEntries(draft.assets.map((asset) => {
      const current = assetTargetMap[asset.id] ?? {};
      const exactMatch = historicalDistribution.targets.find((target) =>
        target.id === asset.id || target.name === asset.label
      )?.id ?? "";
      return [asset.id, {
        target_id: targetIds.has(current.target_id) ? current.target_id : exactMatch,
        return_transform: current.return_transform === "forward_value"
          ? "forward_value"
          : "simple_return",
      }];
    }));
    updateTransition({
      asset_return_source: {
        kind: "historical_evaluation_targets",
        run_content_hash: historicalDistribution.contentHash,
        evaluation_artifact_checksum: historicalDistribution.checksum,
        asset_target_map: nextMap,
        sampling: "empirical_bootstrap",
        minimum_observations_per_state: Number(
          assetReturnSource.minimum_observations_per_state ?? 5,
        ),
        inline_policy: assetReturnSource.inline_policy === "override"
          ? "override"
          : "forbid",
      },
    });
  };
  const selectedHistoricalCausal = Boolean(
    selectedHistorical?.run.mode === "realtime" &&
      selectedHistorical.run.causality?.is_causal === true &&
      selectedHistorical.run.causality.repaints !== true &&
      selectedHistorical.run.causality.uses_future_data === false,
  );
  const reverseConfig =
    (scenario.reverse_stress && typeof scenario.reverse_stress === "object"
      ? scenario.reverse_stress
      : scenario) as Record<string, unknown>;
  const updateReverse = (patch: Record<string, unknown>) =>
    update({ reverse_stress: { ...reverseConfig, ...patch } });

  return (
    <section className="space-y-5">
      <SectionHeading
        eyebrow="02 / Path"
        title="设定期限、随机种子与因子路径"
        detail="确定性方法只给出路径结果；只有随机模拟才显示概率、VaR、ES 与分位扇形。"
      />
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="block text-sm font-semibold text-slate-700">
          期限步数<input
            aria-label="期限步数"
            type="number"
            min="1"
            max={draft.method === "reverse_stress" ? 1 : 1200}
            disabled={draft.method === "reverse_stress"}
            value={draft.horizon}
            onChange={(event) =>
              onChange({ ...draft, horizon: Number(event.target.value) })}
            className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 px-3 disabled:bg-slate-100"
          />
          {draft.method === "reverse_stress"
            ? (
              <span className="mt-1 block text-xs font-normal text-slate-600">
                当前反向压力求解器采用单期线性近似。
              </span>
            )
            : null}
        </label>
        <label className="block text-sm font-semibold text-slate-700">
          每期口径<select
            aria-label="时间步长"
            value={String(scenario.step_unit ?? "period")}
            onChange={(event) => update({ step_unit: event.target.value })}
            className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3"
          >
            <option value="period">数据自然周期</option>
            <option value="trading_day">交易日</option>
            <option value="week">周</option>
            <option value="month">月</option>
            <option value="quarter">季度</option>
          </select>
        </label>
      </div>
      {draft.method === "historical_replay"
        ? (
          <JsonEditor
            label="历史收益序列 JSON"
            value={scenario.historical_returns ?? []}
            helper="每期包含 date 与 returns；空值会按缺失策略阻断或降级。"
            onApply={(historical_returns) => update({ historical_returns })}
          />
        )
        : null}
      {draft.method === "factor_path"
        ? (
          <>
            <div className="grid gap-3 sm:grid-cols-2">
              <label className="text-sm font-semibold text-slate-700">
                路径形状<select
                  aria-label="因子路径形状"
                  value={String(scenario.path_shape ?? "linear")}
                  onChange={(event) =>
                    update({ path_shape: event.target.value })}
                  className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3"
                >
                  <option value="linear">线性传导</option>
                  <option value="instant">首期冲击</option>
                  <option value="custom">自定义逐期路径</option>
                </select>
              </label>
              <label className="text-sm font-semibold text-slate-700">
                严重度<input
                  aria-label="情景严重度"
                  type="number"
                  min="0"
                  step="0.1"
                  value={Number(scenario.severity ?? 1)}
                  onChange={(event) =>
                    update({ severity: Number(event.target.value) })}
                  className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 px-3"
                />
              </label>
            </div>
            <FactorValueTable
              title="终点冲击"
              factors={draft.factors}
              values={shocks}
              onChange={(id, value) =>
                update({ shocks: { ...shocks, [id]: value } })}
            />
            <JsonEditor
              label="可选逐期冲击路径 JSON"
              value={scenario.factor_path ?? []}
              helper="自定义时，每期包含 step、可选 date 与 shocks。"
              onApply={(factor_path) => update({ factor_path })}
            />
          </>
        )
        : null}
      {draft.method === "monte_carlo"
        ? (
          <>
            <div className="grid gap-3 sm:grid-cols-4">
              <label className="text-sm font-semibold text-slate-700">
                分布<select
                  aria-label="模拟分布"
                  value={String(scenario.distribution ?? "normal")}
                  onChange={(event) =>
                    update({ distribution: event.target.value })}
                  className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3"
                >
                  <option value="normal">正态</option>
                  <option value="student_t">Student-t</option>
                </select>
              </label>
              <label className="text-sm font-semibold text-slate-700">
                路径数<input
                  aria-label="模拟路径数"
                  type="number"
                  min="100"
                  max="50000"
                  step="100"
                  value={Number(scenario.path_count ?? 2000)}
                  onChange={(event) =>
                    update({ path_count: Number(event.target.value) })}
                  className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 px-3"
                />
              </label>
              <label className="text-sm font-semibold text-slate-700">
                随机种子<input
                  aria-label="随机种子"
                  type="number"
                  value={Number(scenario.seed ?? 42)}
                  onChange={(event) =>
                    update({ seed: Number(event.target.value) })}
                  className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 px-3"
                />
              </label>
              <label className="text-sm font-semibold text-slate-700">
                目标收益<input
                  aria-label="目标收益"
                  type="number"
                  step="0.01"
                  value={Number(scenario.target_return ?? 0)}
                  onChange={(event) =>
                    update({ target_return: Number(event.target.value) })}
                  className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 px-3"
                />
              </label>
            </div>
            <div className="grid gap-4 lg:grid-cols-2">
              <FactorValueTable
                title="因子均值"
                factors={draft.factors}
                values={means}
                onChange={(id, value) =>
                  update({ factor_means: { ...means, [id]: value } })}
              />
              <FactorValueTable
                title="因子波动率"
                factors={draft.factors}
                values={volatilities}
                onChange={(id, value) =>
                  update({
                    factor_volatilities: { ...volatilities, [id]: value },
                  })}
              />
            </div>
            <JsonEditor
              label="因子相关矩阵 JSON"
              value={scenario.correlation ?? []}
              helper="矩阵维度与因子顺序严格一致。"
              onApply={(correlation) => update({ correlation })}
            />
          </>
        )
        : null}
      {draft.method === "regime_conditioned"
        ? (
          <>
            <label className="block text-sm font-semibold text-slate-700">
              历史状态版本<select
                aria-label="历史状态版本"
                value={historicalRef.run_id && historicalRef.publication_id
                  ? `${historicalRef.run_id}|${historicalRef.publication_id}`
                  : ""}
                onChange={(event) => {
                  const [run_id, publication_id] = event.target.value.split(
                    "|",
                  );
                  const nextTransition = { ...transition };
                  delete nextTransition.asset_return_source;
                  update({
                    historical_run_ref: event.target.value
                      ? { run_id, publication_id }
                      : undefined,
                    transition: nextTransition,
                  });
                }}
                className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3"
              >
                <option value="">不引用，使用下方内联转移矩阵</option>
                {publishedHistoricalRuns.map(({ run, publication }) => (
                  <option
                    key={`${run.id}-${publication.id}`}
                    value={`${run.id}|${publication.id}`}
                  >
                    {run.name} · R{run.definition_revision ?? "—"} ·{" "}
                    {run.mode === "realtime" ? "实时" : "事后"} ·{" "}
                    {historicalUsageLabels[publication.usage]}
                  </option>
                ))}
              </select>
            </label>
            {selectedHistorical
              ? (
                <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-3 text-xs text-emerald-950">
                  <p className="font-bold">
                    已锁定 {selectedHistorical.run.name}{" "}
                    · R{selectedHistorical.run.definition_revision ?? "—"}
                  </p>
                  <p className="mt-1">
                    {selectedHistorical.run.mode === "realtime"
                      ? "实时识别"
                      : "事后识别"} ·{" "}
                    {selectedHistoricalCausal ? "因果门禁合格" : "仅研究使用"}
                    {" "}
                    · 发布用途 {historicalUsageLabels[
                      selectedHistorical.publication.usage
                    ]}
                  </p>
                </div>
              )
              : null}
            {selectedHistorical
              ? (
                <div className="rounded-xl border border-slate-200 p-4">
                  <fieldset>
                    <legend className="text-sm font-bold text-slate-900">
                      状态收益来源
                    </legend>
                    <p className="mt-1 text-xs leading-5 text-slate-600">
                      状态转移来自上方发布运行；资产收益可以手工填写，也可以从同一运行的评价目标制品按状态联合抽样。
                    </p>
                    <div className="mt-3 grid gap-2 sm:grid-cols-2">
                      <label className="flex min-h-12 cursor-pointer items-center gap-3 rounded-xl border border-slate-200 px-3 text-xs font-bold text-slate-700">
                        <input
                          type="radio"
                          name="historical-return-source"
                          checked={!usesHistoricalDistribution}
                          onChange={() => updateTransition({ asset_return_source: undefined })}
                        />
                        手工状态收益参数
                      </label>
                      <label className={cx(
                        "flex min-h-12 items-center gap-3 rounded-xl border px-3 text-xs font-bold",
                        historicalDistributionLocked && draft.assets.length
                          ? "cursor-pointer border-accent-200 text-accent-800"
                          : "cursor-not-allowed border-slate-200 bg-slate-50 text-slate-600",
                      )}>
                        <input
                          type="radio"
                          name="historical-return-source"
                          checked={usesHistoricalDistribution}
                          disabled={!historicalDistributionLocked || !draft.assets.length}
                          onChange={buildHistoricalDistributionSource}
                        />
                        历史评价目标联合经验分布
                      </label>
                    </div>
                  </fieldset>
                  {!historicalDistributionLocked
                    ? (
                      <p role="status" className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-900">
                        此发布运行没有可用的、与发布内容一致的 v2 评价目标制品，不能选择历史收益分布；仍可使用手工状态收益参数。
                      </p>
                    )
                    : (
                      <p className="mt-3 text-xs leading-5 text-slate-600">
                        可用评价目标 {historicalDistribution.targets.length} 项 · 内容锁 {historicalDistribution.contentHash.slice(0, 12)}… · 制品锁 {historicalDistribution.checksum.slice(7, 19)}…
                      </p>
                    )}
                </div>
              )
              : null}
            {usesHistoricalDistribution && historicalDistributionLocked
              ? (
                <div className="space-y-4 rounded-xl border border-accent-200 bg-accent-50/40 p-4">
                  {!sourceMatchesSelectedRun
                    ? (
                      <div role="alert" className="flex flex-col gap-2 rounded-lg bg-rose-50 px-3 py-2 text-xs leading-5 text-rose-900 sm:flex-row sm:items-center sm:justify-between">
                        <span>当前草稿的收益制品锁与所选历史运行不一致，必须重新锁定后才能运行。</span>
                        <button
                          type="button"
                          onClick={buildHistoricalDistributionSource}
                          className="min-h-9 shrink-0 rounded-lg border border-rose-300 bg-white px-3 font-bold"
                        >
                          重新锁定当前运行
                        </button>
                      </div>
                    )
                    : null}
                  <div>
                    <h4 className="text-sm font-bold text-slate-950">资产与评价目标映射</h4>
                    <p className="mt-1 text-xs leading-5 text-slate-600">
                      联合抽样只保留所有已映射资产同时有效的观测，缺失值不会填零。
                    </p>
                  </div>
                  {unmappedHistoricalAssets.length
                    ? (
                      <p role="alert" className="rounded-lg bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-900">
                        尚有 {unmappedHistoricalAssets.length} 项资产未映射评价目标；完成逐项选择后才能保存并运行。
                      </p>
                    )
                    : null}
                  <div className="space-y-2">
                    {draft.assets.map((asset) => {
                      const mapping = assetTargetMap[asset.id] ?? {};
                      return (
                        <div key={asset.id} className="grid gap-2 rounded-xl border border-accent-100 bg-white p-3 sm:grid-cols-[minmax(120px,.8fr)_minmax(170px,1.2fr)_minmax(170px,1fr)] sm:items-end">
                          <p className="text-sm font-bold text-slate-800">{asset.label}</p>
                          <label className="text-xs font-semibold text-slate-600">
                            评价目标<select
                              aria-label={`${asset.label}评价目标`}
                              value={mapping.target_id ?? ""}
                              onChange={(event) => updateTransition({
                                asset_return_source: {
                                  ...assetReturnSource,
                                  asset_target_map: {
                                    ...assetTargetMap,
                                    [asset.id]: {
                                      ...mapping,
                                      target_id: event.target.value,
                                      return_transform: mapping.return_transform || "simple_return",
                                    },
                                  },
                                },
                              })}
                              className="mt-1 min-h-10 w-full rounded-xl border border-slate-300 bg-white px-2"
                            >
                              <option value="">请选择评价目标</option>
                              {historicalDistribution.targets.map((target) => (
                                <option key={target.id} value={target.id}>{target.name} · {target.id}</option>
                              ))}
                            </select>
                          </label>
                          <label className="text-xs font-semibold text-slate-600">
                            序列口径<select
                              aria-label={`${asset.label}收益转换`}
                              value={mapping.return_transform ?? "simple_return"}
                              onChange={(event) => updateTransition({
                                asset_return_source: {
                                  ...assetReturnSource,
                                  asset_target_map: {
                                    ...assetTargetMap,
                                    [asset.id]: {
                                      ...mapping,
                                      return_transform: event.target.value,
                                    },
                                  },
                                },
                              })}
                              className="mt-1 min-h-10 w-full rounded-xl border border-slate-300 bg-white px-2"
                            >
                              <option value="simple_return">净值 / 价格转下一期简单收益</option>
                              <option value="forward_value">序列值已是下一期简单收益</option>
                            </select>
                          </label>
                        </div>
                      );
                    })}
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <label className="text-xs font-semibold text-slate-600">
                      每状态最少样本<input
                        aria-label="每状态最少样本"
                        type="number"
                        min="2"
                        max="20000"
                        step="1"
                        value={Number(assetReturnSource.minimum_observations_per_state ?? 5)}
                        onChange={(event) => updateTransition({
                          asset_return_source: {
                            ...assetReturnSource,
                            minimum_observations_per_state: Number(event.target.value),
                          },
                        })}
                        className="mt-1 min-h-10 w-full rounded-xl border border-slate-300 bg-white px-3"
                      />
                    </label>
                    <label className="text-xs font-semibold text-slate-600">
                      内联参数策略<select
                        aria-label="内联状态收益策略"
                        value={String(assetReturnSource.inline_policy ?? "forbid")}
                        onChange={(event) => updateTransition({
                          asset_return_source: {
                            ...assetReturnSource,
                            inline_policy: event.target.value,
                          },
                        })}
                        className="mt-1 min-h-10 w-full rounded-xl border border-slate-300 bg-white px-3"
                      >
                        <option value="forbid">禁止手工收益覆盖（默认）</option>
                        <option value="override">允许手工收益显式覆盖</option>
                      </select>
                    </label>
                  </div>
                  {assetReturnSource.inline_policy === "override"
                    ? (
                      <p role="status" className="rounded-lg bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-900">
                        已允许手工覆盖：高级 JSON 中的内联状态收益会覆盖对应资产的历史分布，运行审计将明确披露。
                      </p>
                    )
                    : null}
                </div>
              )
              : null}
            {!publishedHistoricalRuns.length
              ? (
                <p className="rounded-xl bg-amber-50 p-3 text-xs text-amber-900">
                  {hasHistoricalPublications
                    ? `已有历史状态版本，但没有满足“${
                      usageLabels[
                        draft.usage_intent ?? "research_display"
                      ]
                    }”用途和因果门禁的发布；可改用内联转移矩阵或重新发布合格版本。`
                    : "尚无已发布历史状态版本；可先使用内联转移矩阵，或到“历史情景识别”保存、运行并发布。"}
                </p>
              )
              : null}
            <details className="rounded-xl border border-slate-200 p-3">
              <summary className="cursor-pointer text-xs font-bold text-slate-700">
                手工填写历史状态版本 ID（兼容入口）
              </summary>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <label className="text-xs font-semibold text-slate-600">
                  运行 ID<input
                    aria-label="历史情景运行 ID"
                    value={historicalRef.run_id ?? ""}
                    onChange={(event) =>
                      {
                        const nextTransition = { ...transition };
                        delete nextTransition.asset_return_source;
                        update({
                          historical_run_ref: {
                            ...historicalRef,
                            run_id: event.target.value,
                          },
                          transition: nextTransition,
                        });
                      }}
                    className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-3"
                  />
                </label>
                <label className="text-xs font-semibold text-slate-600">
                  发布 ID<input
                    aria-label="历史情景发布 ID"
                    value={historicalRef.publication_id ?? ""}
                    onChange={(event) =>
                      {
                        const nextTransition = { ...transition };
                        delete nextTransition.asset_return_source;
                        update({
                          historical_run_ref: {
                            ...historicalRef,
                            publication_id: event.target.value,
                          },
                          transition: nextTransition,
                        });
                      }}
                    className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-3"
                  />
                </label>
              </div>
            </details>
            <JsonEditor
              label="高级状态转移配置 JSON"
              value={scenario.transition ??
                {
                  states: [],
                  matrix: [],
                  initial_state: "",
                  path_count: 2000,
                  seed: 42,
                }}
              helper="高级入口：配置状态收益、转移矩阵、初始状态、路径数与随机种子；已选择历史分布时请保留内容锁字段。"
              onApply={(transition) => update({ transition })}
            />
          </>
        )
        : null}
      {draft.method === "reverse_stress"
        ? (
          <>
            <div className="grid gap-3 sm:grid-cols-2">
              <label className="text-sm font-semibold text-slate-700">
                目标指标<select
                  aria-label="反向压力目标"
                  value={String(reverseConfig.target_metric ?? "loss")}
                  onChange={(event) =>
                    updateReverse({ target_metric: event.target.value })}
                  className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3"
                >
                  <option value="loss">累计损失</option>
                  <option value="drawdown">最大回撤</option>
                </select>
              </label>
              <label className="text-sm font-semibold text-slate-700">
                正数阈值<input
                  aria-label="反向压力阈值"
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={Number(reverseConfig.threshold ?? 0.1)}
                  onChange={(event) =>
                    updateReverse({ threshold: Number(event.target.value) })}
                  className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 px-3"
                />
              </label>
            </div>
            <JsonEditor
              label="因子搜索边界 JSON"
              value={reverseConfig.bounds ?? {}}
              helper="每个因子填写 [下界, 上界]；后端返回多组达到目标的候选。"
              onApply={(bounds) => updateReverse({ bounds })}
            />
          </>
        )
        : null}
    </section>
  );
}

function FactorValueTable(
  { title, factors, values, onChange }: {
    title: string;
    factors: ScenarioDefinition["factors"];
    values: Record<string, number>;
    onChange: (id: string, value: number) => void;
  },
) {
  return (
    <div className="overflow-x-auto rounded-xl border border-slate-200">
      <table className="w-full text-sm">
        <caption className="px-3 py-2 text-left text-xs font-bold text-slate-600">
          {title}
        </caption>
        <tbody className="divide-y divide-slate-100">
          {factors.map((factor) => (
            <tr key={factor.id}>
              <th scope="row" className="px-3 py-2 text-left font-semibold">
                {factor.label}
              </th>
              <td className="px-3 py-2 text-slate-600">{factor.unit}</td>
              <td className="px-3 py-2 text-right">
                <input
                  aria-label={`${title}${factor.label}`}
                  type="number"
                  step="0.01"
                  value={values[factor.id] ?? 0}
                  onChange={(event) =>
                    onChange(factor.id, Number(event.target.value))}
                  className="min-h-9 w-28 rounded-lg border border-slate-300 px-2 text-right"
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function JsonEditor(
  { label, value, helper, onApply }: {
    label: string;
    value: unknown;
    helper: string;
    onApply: (value: unknown) => void;
  },
) {
  const [text, setText] = useState(() => JSON.stringify(value, null, 2));
  const [error, setError] = useState("");
  useEffect(() => setText(JSON.stringify(value, null, 2)), [value]);
  const apply = () => {
    try {
      onApply(JSON.parse(text) as unknown);
      setError("");
    } catch {
      setError("JSON 格式不正确，请检查括号、逗号和引号。");
    }
  };
  return (
    <div>
      <label className="text-sm font-semibold text-slate-700">
        {label}
        <textarea
          aria-label={label}
          rows={6}
          value={text}
          onChange={(event) => setText(event.target.value)}
          spellCheck={false}
          className="mt-1 w-full rounded-xl border border-slate-300 p-3 font-mono text-xs leading-5"
        />
      </label>
      <div className="mt-2 flex items-center justify-between gap-3">
        <span
          className={cx(
            "text-xs",
            error ? "text-rose-700" : "text-slate-600",
          )}
        >
          {error || helper}
        </span>
        <button
          type="button"
          onClick={apply}
          className="min-h-9 rounded-lg bg-slate-950 px-3 text-xs font-bold text-white"
        >
          解析并应用
        </button>
      </div>
    </div>
  );
}

function ExposurePanel(
  { draft, onChange }: {
    draft: ScenarioDefinition;
    onChange: (next: ScenarioDefinition) => void;
  },
) {
  const [portfolioId, setPortfolioId] = useState(draft.portfolios[0]?.id ?? "");
  const [weightSummary, setWeightSummary] = useState<ScenarioWeightSummary | null>(null);
  const [weightSummaryError, setWeightSummaryError] = useState("");
  const portfolio = draft.portfolios.find((item) => item.id === portfolioId) ??
    draft.portfolios[0];
  useEffect(() => {
    if (!draft.portfolios.some((item) => item.id === portfolioId)) {
      setPortfolioId(draft.portfolios[0]?.id ?? "");
    }
  }, [draft.portfolios, portfolioId]);
  useEffect(() => {
    if (!portfolio || Object.keys(portfolio.weights).length === 0) {
      setWeightSummary(null);
      setWeightSummaryError("");
      return;
    }
    const controller = new AbortController();
    setWeightSummary(null);
    setWeightSummaryError("");
    const timer = window.setTimeout(() => {
      summarizeScenarioStressWeights(portfolio.weights, controller.signal)
        .then((summary) => {
          if (!controller.signal.aborted) setWeightSummary(summary);
        })
        .catch((reason) => {
          if (controller.signal.aborted) return;
          setWeightSummaryError(
            reason instanceof Error ? reason.message : "组合权重核验不可用",
          );
        });
    }, 180);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [portfolio]);
  const updatePortfolio = (patch: Partial<ScenarioPortfolio>) =>
    portfolio &&
    onChange({
      ...draft,
      portfolios: draft.portfolios.map((item) =>
        item.id === portfolio.id ? { ...item, ...patch } : item
      ),
    });
  const addPortfolio = () => {
    const id = `portfolio-${draft.portfolios.length + 1}`;
    onChange({
      ...draft,
      portfolios: [...draft.portfolios, {
        id,
        name: `研究组合 ${draft.portfolios.length + 1}`,
        weights: Object.fromEntries(draft.assets.map((asset) => [asset.id, 0])),
      }],
    });
    setPortfolioId(id);
  };
  const updateBeta = (assetId: string, factorId: string, raw: string) => {
    const parsed = raw === "" ? null : Number(raw);
    const row = draft.mapping.factor_to_asset[assetId] ?? {};
    onChange({
      ...draft,
      mapping: {
        ...draft.mapping,
        factor_to_asset: {
          ...draft.mapping.factor_to_asset,
          [assetId]: {
            ...row,
            [factorId]: Number.isFinite(parsed) ? parsed : null,
          },
        },
      },
    });
  };
  return (
    <section className="space-y-5">
      <SectionHeading
        eyebrow="03 / Exposure"
        title="配置目标权重、暴露和缺失策略"
        detail="冲击先映射到资产，再由组合权重汇总损益；未覆盖资产不会被静默当作零风险。"
      />
      <div
        className="flex flex-wrap gap-2"
        role="tablist"
        aria-label="研究组合"
      >
        {draft.portfolios.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={portfolio?.id === item.id}
            onClick={() => setPortfolioId(item.id)}
            className={cx(
              "min-h-10 rounded-xl px-3 text-xs font-bold",
              portfolio?.id === item.id
                ? "bg-slate-950 text-white"
                : "border border-slate-200 text-slate-700",
            )}
          >
            {item.name}
          </button>
        ))}
        <button
          type="button"
          onClick={addPortfolio}
          className="min-h-10 rounded-xl border border-dashed border-accent-300 px-3 text-xs font-bold text-accent-700"
        >
          + 添加组合
        </button>
      </div>
      {portfolio
        ? (
          <div className="rounded-xl border border-slate-200 p-4">
            <div className="flex gap-3">
              <label className="flex-1 text-sm font-semibold text-slate-700">
                组合名称<input
                  aria-label="组合名称"
                  value={portfolio.name}
                  onChange={(event) =>
                    updatePortfolio({ name: event.target.value })}
                  className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-3"
                />
              </label>
              {draft.portfolios.length > 1
                ? (
                  <button
                    type="button"
                    onClick={() =>
                      onChange({
                        ...draft,
                        portfolios: draft.portfolios.filter((item) =>
                          item.id !== portfolio.id
                        ),
                      })}
                    className="self-end min-h-10 rounded-lg px-3 text-xs font-bold text-rose-700"
                  >
                    移除组合
                  </button>
                )
                : null}
            </div>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full min-w-[420px] text-sm">
                <caption className="sr-only">组合目标权重</caption>
                <thead className="text-left text-xs text-slate-600">
                  <tr>
                    <th scope="col" className="pb-2">资产</th>
                    <th scope="col" className="pb-2 text-right">目标权重</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {draft.assets.map((asset) => (
                    <tr key={asset.id}>
                      <th scope="row" className="py-2 text-left">
                        {asset.label}
                      </th>
                      <td className="py-2 text-right">
                        <input
                          aria-label={`${portfolio.name}${asset.label}目标权重`}
                          type="number"
                          min="-2"
                          max="2"
                          step="0.01"
                          value={portfolio.weights[asset.id] ?? 0}
                          onChange={(event) =>
                            updatePortfolio({
                              weights: {
                                ...portfolio.weights,
                                [asset.id]: Number(event.target.value),
                              },
                            })}
                          className="min-h-9 w-28 rounded-lg border border-slate-300 px-2 text-right"
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr>
                    <th scope="col" className="pt-3 text-left">权重合计</th>
                    <td className="pt-3 text-right font-bold">
                      {weightSummary
                        ? formatPercent(weightSummary.total_weight)
                        : weightSummaryError
                        ? "不可用"
                        : "核验中…"}
                    </td>
                  </tr>
                </tfoot>
              </table>
            </div>
            {weightSummaryError
              ? (
                <p role="alert" className="mt-3 text-xs text-rose-700">
                  {weightSummaryError}
                </p>
              )
              : weightSummary && !weightSummary.within_tolerance
              ? (
                <p role="status" className="mt-3 text-xs text-amber-700">
                  后端核验未通过：净敞口、单项权重或总敞口超出约束。
                </p>
              )
              : null}
          </div>
        )
        : null}
      <div className="rounded-xl border border-accent-100 bg-accent-50 px-4 py-3 text-xs leading-5 text-accent-950">
        当前传导口径：<strong>
          {draft.mapping.response_space === "log_return"
            ? "对数收益响应"
            : draft.mapping.response_space === "simple_return"
            ? "简单收益响应"
            : "资产收益直接输入"}
        </strong>。方法切换时系统会同步切换口径，防止同一系数被误用于不同收益空间。
      </div>
      <details className="rounded-xl border border-slate-200 bg-slate-50 p-4">
        <summary className="cursor-pointer text-sm font-bold text-slate-800">
          资产 × 因子传导矩阵
        </summary>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[680px] text-sm">
            <caption className="sr-only">资产因子传导矩阵</caption>
            <thead>
              <tr>
                <th scope="col" className="p-2 text-left text-xs text-slate-600">资产</th>
                {draft.factors.map((factor) => (
                  <th scope="col"
                    key={factor.id}
                    className="p-2 text-right text-xs text-slate-600"
                  >
                    {factor.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {draft.assets.map((asset) => (
                <tr key={asset.id}>
                  <th scope="row" className="p-2 text-left">{asset.label}</th>
                  {draft.factors.map((factor) => (
                    <td key={factor.id} className="p-2 text-right">
                      <input
                        aria-label={`${asset.label}对${factor.label}传导系数`}
                        type="number"
                        step="0.0001"
                        value={draft.mapping.factor_to_asset[asset.id]
                          ?.[factor.id] ?? ""}
                        onChange={(event) =>
                          updateBeta(asset.id, factor.id, event.target.value)}
                        className="min-h-9 w-24 rounded-lg border border-slate-300 px-2 text-right"
                      />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="text-sm font-semibold text-slate-700">
          缺失暴露处理<select
            aria-label="缺失暴露处理"
            value={draft.mapping.missing_policy}
            onChange={(event) =>
              onChange({
                ...draft,
                mapping: {
                  ...draft.mapping,
                  missing_policy: event.target.value as "block" | "degrade",
                },
              })}
            className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3"
          >
            <option value="block">阻断运行</option>
            <option value="degrade">降级并披露</option>
          </select>
        </label>
        <label className="text-sm font-semibold text-slate-700">
          最低覆盖率<input
            aria-label="最低覆盖率"
            type="number"
            min="0.5"
            max="1"
            step="0.01"
            value={draft.mapping.minimum_coverage}
            onChange={(event) =>
              onChange({
                ...draft,
                mapping: {
                  ...draft.mapping,
                  minimum_coverage: Number(event.target.value),
                },
              })}
            className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 px-3"
          />
          <span className="mt-1 block text-xs font-normal text-slate-600">
            当前 {formatPercent(draft.mapping.minimum_coverage)}
          </span>
        </label>
      </div>
    </section>
  );
}

function LimitsPanel(
  { draft, onChange }: {
    draft: ScenarioDefinition;
    onChange: (next: ScenarioDefinition) => void;
  },
) {
  const metrics = Object.keys(limitMetricLabels) as ScenarioLimit["metric"][];
  const update = (index: number, patch: Partial<ScenarioLimit>) =>
    onChange({
      ...draft,
      limits: draft.limits.map((item, itemIndex) =>
        itemIndex === index ? { ...item, ...patch } : item
      ),
    });
  const add = () =>
    onChange({
      ...draft,
      limits: [...draft.limits, {
        id: `limit-${draft.limits.length + 1}`,
        label: "新约束",
        metric: "terminal_return",
        operator: "lt",
        threshold: -0.1,
      }],
    });
  return (
    <section className="space-y-5">
      <SectionHeading
        eyebrow="04 / Limits"
        title="设定损失、回撤和概率约束"
        detail="确定性情景不能配置伪概率、VaR 或 ES；后端记录首次突破与期限内恢复情况。"
      />
      <div className="overflow-x-auto">
        <table className="w-full min-w-[820px] text-sm">
          <caption className="sr-only">情景约束</caption>
          <thead className="text-left text-xs text-slate-600">
            <tr>
              <th scope="col" className="pb-2">名称</th>
              <th scope="col" className="pb-2">指标</th>
              <th scope="col" className="pb-2">关系</th>
              <th scope="col" className="pb-2 text-right">阈值</th>
              <th scope="col" />
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {draft.limits.map((limit, index) => (
              <tr key={limit.id}>
                <td className="py-2">
                  <input
                    aria-label={`约束 ${index + 1} 名称`}
                    value={limit.label}
                    onChange={(event) =>
                      update(index, { label: event.target.value })}
                    className="min-h-9 rounded-lg border border-slate-300 px-2"
                  />
                </td>
                <td className="py-2">
                  <select
                    aria-label={`约束 ${index + 1} 指标`}
                    value={limit.metric}
                    onChange={(event) =>
                      update(index, {
                        metric: event.target.value as ScenarioLimit["metric"],
                      })}
                    className="min-h-9 rounded-xl border border-slate-300 bg-white px-2"
                  >
                    {metrics.filter((metric) =>
                      deterministicMethods.has(draft.method)
                        ? !probabilisticLimitMetrics.has(metric)
                        : metric !== "worst_step_return"
                    ).map((metric) => (
                      <option key={metric} value={metric}>
                        {limitMetricLabels[metric]}
                      </option>
                    ))}
                  </select>
                </td>
                <td className="py-2">
                  <select
                    aria-label={`约束 ${index + 1} 关系`}
                    value={limit.operator}
                    onChange={(event) =>
                      update(index, {
                        operator: event.target
                          .value as ScenarioLimit["operator"],
                      })}
                    className="min-h-9 rounded-xl border border-slate-300 bg-white px-2"
                  >
                    <option value="lt">小于</option>
                    <option value="lte">小于等于</option>
                    <option value="gt">大于</option>
                    <option value="gte">大于等于</option>
                  </select>
                </td>
                <td className="py-2 text-right">
                  <input
                    aria-label={`约束 ${index + 1} 阈值`}
                    type="number"
                    step="0.01"
                    value={limit.threshold}
                    onChange={(event) =>
                      update(index, { threshold: Number(event.target.value) })}
                    className="min-h-9 w-28 rounded-lg border border-slate-300 px-2 text-right"
                  />
                </td>
                <td className="py-2 text-right">
                  <button
                    type="button"
                    onClick={() =>
                      onChange({
                        ...draft,
                        limits: draft.limits.filter((_, itemIndex) =>
                          itemIndex !== index
                        ),
                      })}
                    className="rounded-lg px-2 py-1 text-xs font-bold text-rose-700"
                  >
                    移除
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {!draft.limits.length
          ? (
            <p className="rounded-xl bg-slate-50 p-5 text-center text-sm text-slate-600">
              尚未配置约束。
            </p>
          )
          : null}
      </div>
      <button
        type="button"
        onClick={add}
        className="min-h-10 rounded-xl border border-accent-300 px-3 text-xs font-bold text-accent-700"
      >
        + 添加约束
      </button>
    </section>
  );
}

function MetricCard(
  { label, value, hint, danger = false }: {
    label: string;
    value: string;
    hint: string;
    danger?: boolean;
  },
) {
  return (
    <article
      className={cx(
        "rounded-xl border p-3",
        danger ? "border-rose-200 bg-rose-50" : "border-slate-200 bg-white",
      )}
    >
      <p className="text-xs font-medium text-slate-600">{label}</p>
      <p
        className={cx(
          "mt-1 text-xl font-bold tabular-nums",
          danger ? "text-rose-800" : "text-slate-950",
        )}
      >
        {value}
      </p>
      <p className="mt-1 text-xs leading-4 text-slate-600">{hint}</p>
    </article>
  );
}

function ReverseCandidatesTable(
  { candidates }: { candidates: ReverseStressCandidate[] },
) {
  return (
    <table className="w-full min-w-[820px] text-xs">
      <caption className="sr-only">反向压力候选组合</caption>
      <thead className="text-left text-slate-600">
        <tr>
          <th scope="col" className="p-2">候选</th>
          <th scope="col" className="p-2 text-right">严重度</th>
          <th scope="col" className="p-2 text-right">组合收益</th>
          <th scope="col" className="p-2 text-right">最大回撤</th>
          <th scope="col" className="p-2">因子冲击</th>
          <th scope="col" className="p-2">达到目标</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-slate-100">
        {candidates.map((candidate, index) => (
          <tr key={`${candidate.kind}-${index}`}>
            <th scope="row" className="p-2 text-left">{candidate.label}</th>
            <td className="p-2 text-right">
              {formatNumber(candidate.severity)}
            </td>
            <td className="p-2 text-right font-bold text-rose-700">
              {formatPercent(candidate.portfolio_return)}
            </td>
            <td className="p-2 text-right">
              {formatPercent(candidate.max_drawdown)}
            </td>
            <td className="p-2 font-mono">
              {Object.entries(candidate.factor_shocks).map(([key, value]) =>
                `${key} ${value}`
              ).join("；")}
            </td>
            <td className="p-2">{candidate.meets_target ? "是" : "否"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ResultChart(
  { run, result }: { run: ScenarioStressRun; result: ScenarioResult },
) {
  const option = useMemo<EChartsOption>(() => {
    if (result.reverse_stress?.candidates.length) {
      return {
        aria: { enabled: true, description: "反向压力候选损失排序图。" },
        tooltip: { trigger: "axis" },
        grid: { left: 55, right: 24, top: 25, bottom: 75 },
        xAxis: {
          type: "category",
          data: result.reverse_stress.candidates.map((item) => item.label),
          axisLabel: { rotate: 20 },
        },
        yAxis: { type: "value", axisLabel: { formatter: "{value}%" } },
        series: [{
          name: "组合收益",
          type: "bar",
          data: result.reverse_stress.candidates.map((item) =>
            item.portfolio_return * 100
          ),
          itemStyle: { color: "#e11d48" },
        }],
      };
    }
    if (run.probabilistic && result.distribution?.fan) {
      const fan = result.distribution.fan;
      const names = Object.keys(fan.quantiles).sort();
      return {
        aria: { enabled: true, description: "概率模拟净值分位扇形图。" },
        tooltip: { trigger: "axis" },
        legend: { data: names.map((name) => name.toUpperCase()) },
        grid: { left: 55, right: 24, top: 45, bottom: 45 },
        xAxis: { type: "category", data: fan.steps },
        yAxis: { type: "value", scale: true, name: "净值" },
        series: names.map((name, index) => ({
          name: name.toUpperCase(),
          type: "line" as const,
          showSymbol: false,
          data: fan.quantiles[name],
          lineStyle: {
            color: chartColors[index % chartColors.length],
            width: name === "p50" ? 3 : 1.5,
          },
          areaStyle: name === "p25" || name === "p75"
            ? { opacity: 0.05 }
            : undefined,
        })),
      };
    }
    const path = result.path ?? [];
    return {
      aria: { enabled: true, description: "确定性情景净值与回撤路径图。" },
      tooltip: { trigger: "axis" },
      legend: { data: ["净值", "回撤"] },
      grid: { left: 55, right: 55, top: 45, bottom: 45 },
      xAxis: {
        type: "category",
        data: path.map((point) => point.date || point.step),
      },
      yAxis: [{ type: "value", scale: true, name: "净值" }, {
        type: "value",
        name: "回撤",
        axisLabel: { formatter: "{value}%" },
      }],
      series: [
        {
          name: "净值",
          type: "line",
          showSymbol: false,
          data: path.map((point) => point.nav),
          lineStyle: { color: "#4f46e5", width: 3 },
        },
        {
          name: "回撤",
          type: "line",
          yAxisIndex: 1,
          showSymbol: false,
          data: path.map((point) => point.drawdown * 100),
          lineStyle: { color: "#e11d48", width: 1.5 },
          areaStyle: { color: "#fecdd3", opacity: 0.3 },
        },
      ],
    };
  }, [result, run.probabilistic]);
  return (
    <figure className="rounded-xl border border-slate-200 bg-white p-2 shadow-sm">
      <figcaption className="sr-only">{result.name}情景结果图</figcaption>
      <ReactECharts
        option={option}
        style={{ height: 390 }}
        notMerge
        lazyUpdate
      />
      <details className="mx-2 mb-2 rounded-lg bg-slate-50 px-3 py-2 text-xs">
        <summary className="cursor-pointer font-bold text-accent-700">
          查看图表数据表
        </summary>
        <div className="mt-3 max-h-64 overflow-auto">
          {result.reverse_stress
            ? (
              <ReverseCandidatesTable
                candidates={result.reverse_stress.candidates}
              />
            )
            : run.probabilistic && result.distribution
            ? (
              <table className="w-full">
                <caption className="sr-only">分位净值逐期数据</caption>
                <thead>
                  <tr>
                    <th scope="col" className="p-2 text-left">步数</th>
                    {Object.keys(result.distribution.fan.quantiles).map((
                      name,
                    ) => (
                      <th scope="col" key={name} className="p-2 text-right">
                        {name.toUpperCase()}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.distribution.fan.steps.map((step, index) => (
                    <tr key={step}>
                      <td className="p-2">{step}</td>
                      {Object.entries(result.distribution!.fan.quantiles).map((
                        [name, values],
                      ) => (
                        <td key={name} className="p-2 text-right tabular-nums">
                          {formatNumber(values[index], 4)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            )
            : (
              <table className="w-full">
                <caption className="sr-only">确定性净值逐期数据</caption>
                <thead>
                  <tr>
                    <th scope="col" className="p-2 text-left">步数</th>
                    <th scope="col" className="p-2 text-right">净值</th>
                    <th scope="col" className="p-2 text-right">收益</th>
                    <th scope="col" className="p-2 text-right">回撤</th>
                  </tr>
                </thead>
                <tbody>
                  {result.path?.map((point) => (
                    <tr key={point.step}>
                      <td className="p-2">{point.date || point.step}</td>
                      <td className="p-2 text-right">
                        {formatNumber(point.nav, 4)}
                      </td>
                      <td className="p-2 text-right">
                        {formatPercent(point.return)}
                      </td>
                      <td className="p-2 text-right">
                        {formatPercent(point.drawdown)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
        </div>
      </details>
    </figure>
  );
}

function ResultPanel({ run }: { run: ScenarioStressRun | null }) {
  const [portfolioId, setPortfolioId] = useState("");
  useEffect(() => {
    if (run?.results.length) setPortfolioId(run.results[0].portfolio_id);
  }, [run?.id]);
  const result =
    run?.results.find((item) => item.portfolio_id === portfolioId) ??
      run?.results[0] ?? null;
  if (!run || !result) {
    return (
      <EmptyState title="尚无真实运行结果" hint="保存定义并按版本运行后，这里才会展示路径、归因和约束结论。" />
    );
  }
  const terminal = result.distribution?.terminal;
  const computeAudit = run.compute_audit;
  const computeBackend = computeAudit?.execution_backend ?? computeAudit?.backend;
  const computeCompliant = computeBackend === "numba_njit_fixed_signature" &&
    computeAudit?.nopython === true && computeAudit?.python_fallback === 0;
  const compiledKernelCount = Object.keys(
    computeAudit?.kernel_signatures ?? {},
  ).length;
  const firstBreachStep = result.summary.first_breach_step ?? null;
  const exportRun = () => {
    const objectUrl = URL.createObjectURL(
      new Blob([JSON.stringify(run, null, 2)], {
        type: "application/json;charset=utf-8",
      }),
    );
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = `scenario-stress-${run.id}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(objectUrl);
  };
  return (
    <div className="space-y-5">
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <SectionHeading
            eyebrow="Immutable result"
            title={`${result.name} · ${methodLabels[run.method]}`}
            detail={`${run.id} · ${run.definition_id ?? "试运行"} R${
              run.definition_revision ?? "—"
            } · ${run.immutable ? "不可变结果" : "临时结果"}`}
          />
          <div className="flex flex-wrap items-end gap-2">
            {run.results.length > 1
              ? (
                <label className="text-xs font-bold text-slate-600">
                  查看对象<select
                    aria-label="结果研究对象"
                    value={result.portfolio_id}
                    onChange={(event) => setPortfolioId(event.target.value)}
                    className="mt-1 min-h-10 w-full rounded-xl border border-slate-300 bg-white px-3"
                  >
                    {run.results.map((item) => (
                      <option key={item.portfolio_id} value={item.portfolio_id}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                </label>
              )
              : null}
            <button
              type="button"
              onClick={exportRun}
              className="min-h-10 rounded-xl border border-slate-300 bg-white px-3 text-xs font-bold text-slate-700 hover:border-accent-400 hover:text-accent-700"
            >
              导出完整运行 JSON
            </button>
          </div>
        </div>
        <div className="mt-5 grid grid-cols-2 gap-2 lg:grid-cols-5">
          {run.probabilistic
            ? (
              <>
                <MetricCard
                  label="P05 期末收益"
                  value={formatPercent(terminal?.return_p05)}
                  hint="尾部第五分位"
                  danger={(terminal?.return_p05 ?? 0) < -0.1}
                />
                <MetricCard
                  label="中位期末收益"
                  value={formatPercent(terminal?.return_p50)}
                  hint="P50 路径终值"
                />
                <MetricCard
                  label="损失概率"
                  value={formatPercent(terminal?.loss_probability)}
                  hint="期末收益低于零"
                  danger={(terminal?.loss_probability ?? 0) > 0.5}
                />
                <MetricCard
                  label="ES 95%"
                  value={formatPercent(terminal?.es_95)}
                  hint="最差 5% 平均损失"
                  danger
                />
                <MetricCard
                  label="平均最大回撤"
                  value={formatPercent(terminal?.average_max_drawdown)}
                  hint="模拟路径回撤均值"
                  danger={(terminal?.average_max_drawdown ?? 0) > 0.1}
                />
              </>
            )
            : (
              <>
                <MetricCard
                  label="期末收益"
                  value={formatPercent(result.summary.terminal_return)}
                  hint="确定性路径终值"
                  danger={(result.summary.terminal_return ?? 0) < -0.1}
                />
                <MetricCard
                  label="期末净值"
                  value={formatNumber(result.summary.terminal_nav, 4)}
                  hint="起始净值归一化"
                />
                <MetricCard
                  label="最大回撤"
                  value={formatPercent(result.summary.max_drawdown)}
                  hint="路径内峰谷损失"
                  danger={(result.summary.max_drawdown ?? 0) > 0.1}
                />
                <MetricCard
                  label="最差单步"
                  value={formatPercent(result.summary.worst_step_return)}
                  hint="最差一期收益"
                  danger
                />
                <MetricCard
                  label="约束突破"
                  value={`${result.summary.breach_count ?? 0} 项`}
                  hint="按定义阈值检测"
                  danger={Boolean(result.summary.breach_count)}
                />
              </>
            )}
        </div>
      </section>
      <section className="grid gap-5 xl:grid-cols-[minmax(0,1.4fr)_minmax(300px,.6fr)]">
        <ResultChart run={run} result={result} />
        <aside className="space-y-4">
          <div
            className={cx(
              "rounded-xl border p-4",
              computeCompliant
                ? "border-emerald-200 bg-emerald-50"
                : "border-rose-200 bg-rose-50",
            )}
          >
            <p className="text-xs font-bold uppercase tracking-wide text-slate-600">
              计算执行审计
            </p>
            <p className="mt-1 text-lg font-bold text-slate-950">
              {computeCompliant ? "固定签名 NJIT · 已通过" : "执行证据缺失或不合规"}
            </p>
            <dl className="mt-3 grid grid-cols-2 gap-2 text-xs">
              <div className="rounded-lg bg-white/70 p-2">
                <dt className="text-slate-600">内核覆盖</dt>
                <dd className="mt-1 font-bold text-slate-900">
                  {computeAudit?.kernel_coverage ?? `${compiledKernelCount} 个固定签名`}
                </dd>
              </div>
              <div className="rounded-lg bg-white/70 p-2">
                <dt className="text-slate-600">Python 回退</dt>
                <dd className="mt-1 font-bold text-slate-900">
                  {computeAudit?.python_fallback ?? "未知"}
                </dd>
              </div>
            </dl>
            <p className="mt-2 break-all text-xs leading-5 text-slate-600">
              nopython={String(computeAudit?.nopython ?? false)} · 指纹 {computeAudit?.fingerprint?.slice(0, 16) ?? "未提供"}
            </p>
          </div>
          <div
            className={cx(
              "rounded-xl border p-4",
              result.coverage.status === "complete"
                ? "border-emerald-200 bg-emerald-50"
                : "border-amber-200 bg-amber-50",
            )}
          >
            <p className="text-xs font-bold uppercase tracking-wide text-slate-600">
              映射覆盖率
            </p>
            <p className="mt-1 text-3xl font-bold text-slate-950">
              {formatPercent(result.coverage.ratio)}
            </p>
            <p className="mt-2 text-xs leading-5 text-slate-600">
              策略：{result.coverage.policy === "block"
                ? "缺失即阻断"
                : "允许降级但必须披露"}；状态：{result.coverage.status ===
                  "complete"
                ? "完整"
                : "降级"}。
            </p>
            {result.coverage.missing_assets.length
              ? (
                <p className="mt-2 rounded-lg bg-white/70 px-2 py-2 text-xs text-rose-800">
                  失败 / 缺失对象：{result.coverage.missing_assets.join("、")}
                </p>
              )
              : null}
          </div>
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <h4 className="text-sm font-bold text-slate-900">
              回撤恢复与突破时间
            </h4>
            <dl className="mt-3 space-y-3 text-xs">
              <div className="flex justify-between">
                <dt className="text-slate-600">首次突破</dt>
                <dd className="font-bold">
                  {firstBreachStep == null
                    ? "未突破或不适用"
                    : `第 ${firstBreachStep} 期`}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-600">恢复耗时</dt>
                <dd className="font-bold">
                  {result.summary.recovery_steps == null
                    ? "期限内未恢复或不适用"
                    : `${result.summary.recovery_steps} 期`}
                </dd>
              </div>
            </dl>
            {result.limits?.length
              ? (
                <ul className="mt-3 space-y-2">
                  {result.limits.map((limit, index) => (
                    <li
                      key={limit.id ?? index}
                      className={cx(
                        "rounded-lg px-3 py-2 text-xs",
                        limit.breached
                          ? "bg-rose-50 text-rose-900"
                          : "bg-slate-50 text-slate-600",
                      )}
                    >
                      {limit.label || limit.id || "约束"}：{limit.breached
                        ? `突破${
                          limit.first_breach_step != null
                            ? `于第 ${limit.first_breach_step} 期`
                            : ""
                        }`
                        : limit.breached == null
                        ? "不适用"
                        : "未突破"}
                    </li>
                  ))}
                </ul>
              )
              : null}
          </div>
        </aside>
      </section>
      {result.reverse_stress?.candidates.length
        ? (
          <section className="overflow-x-auto rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <SectionHeading
              eyebrow="Reverse candidates"
              title="多组反向压力候选"
              detail="候选是达到目标阈值的不同冲击组合，不是唯一预测。"
            />
            <div className="mt-4">
              <ReverseCandidatesTable
                candidates={result.reverse_stress.candidates}
              />
            </div>
          </section>
        )
        : null}
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <SectionHeading
          eyebrow="Contribution"
          title="损益贡献与失败对象"
          detail={`贡献口径：${result.contributions.method}；缺失传导保留为空，不会被展示成零贡献。`}
        />
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[560px] text-sm">
            <caption className="sr-only">资产损益贡献</caption>
            <thead className="text-left text-xs text-slate-600">
              <tr>
                <th scope="col" className="pb-2">资产</th>
                <th scope="col" className="pb-2 text-right">损益贡献</th>
                <th scope="col" className="pb-2">状态</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {Object.entries(result.contributions.by_asset).map((
                [asset, contribution],
              ) => (
                <tr key={asset}>
                  <th scope="row" className="py-3 text-left font-semibold">
                    {asset}
                  </th>
                  <td
                    className={cx(
                      "py-3 text-right font-bold tabular-nums",
                      contribution == null
                        ? "text-slate-600"
                        : contribution < 0
                        ? "text-rose-700"
                        : "text-emerald-700",
                    )}
                  >
                    {contribution == null
                      ? "不可计算"
                      : formatPercent(contribution)}
                  </td>
                  <td className="py-3 text-slate-600">
                    {contribution == null ? "缺少暴露或传导系数" : "已覆盖"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      {run.diagnostics.length
        ? (
          <section className="rounded-xl border border-slate-200 bg-slate-50 p-4">
            <h4 className="text-sm font-bold text-slate-900">运行说明</h4>
            <ul className="mt-2 space-y-2 text-xs text-slate-600">
              {run.diagnostics.map((item, index) => (
                <li key={index}>{item.message}</li>
              ))}
            </ul>
          </section>
        )
        : null}
    </div>
  );
}

function ComparePanel(
  { runs, comparison, busy, onCompare }: {
    runs: ScenarioStressRun[];
    comparison: ScenarioRunComparison | null;
    busy: boolean;
    onCompare: (ids: string[]) => void;
  },
) {
  const [selected, setSelected] = useState<string[]>([]);
  const toggle = (id: string) =>
    setSelected((current) =>
      current.includes(id)
        ? current.filter((item) => item !== id)
        : [...current, id]
    );
  return (
    <div className="space-y-5">
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <SectionHeading
            eyebrow="Run comparison"
            title="比较真实运行，而不是页面内临时数字"
            detail="至少选择两次不可变运行；后端按同一指标口径返回差异。"
          />
          <button
            type="button"
            disabled={selected.length < 2 || busy}
            onClick={() => onCompare(selected)}
            className="min-h-11 rounded-xl bg-accent-600 px-4 text-sm font-bold text-white disabled:bg-slate-300"
          >
            {busy ? "比较中…" : `比较 ${selected.length} 次运行`}
          </button>
        </div>
        <div className="mt-5 overflow-x-auto">
          <table className="w-full min-w-[820px] text-sm">
            <caption className="sr-only">可比较情景运行</caption>
            <thead className="bg-slate-50 text-left text-xs text-slate-600">
              <tr>
                <th scope="col" className="p-3">选择</th>
                <th scope="col" className="p-3">运行 / 版本</th>
                <th scope="col" className="p-3">方法</th>
                <th scope="col" className="p-3">对象</th>
                <th scope="col" className="p-3">生成时间</th>
                <th scope="col" className="p-3">发布</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {runs.map((item) => (
                <tr key={item.id}>
                  <td className="p-3">
                    <input
                      aria-label={`选择运行 ${item.id}`}
                      type="checkbox"
                      checked={selected.includes(item.id)}
                      onChange={() => toggle(item.id)}
                    />
                  </td>
                  <td className="p-3 font-mono text-xs">
                    {item.id}
                    <span className="ml-2 text-slate-600">
                      R{item.definition_revision ?? "—"}
                    </span>
                  </td>
                  <td className="p-3">{methodLabels[item.method]}</td>
                  <td className="p-3">{item.results.length}</td>
                  <td className="p-3 text-slate-600">
                    {item.created_at ?? "—"}
                  </td>
                  <td className="p-3">
                    {item.publications.length
                      ? `${item.publications.length} 项`
                      : "未发布"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {!runs.length
            ? (
              <p className="p-8 text-center text-sm text-slate-600">
                暂无历史运行。
              </p>
            )
            : null}
        </div>
      </section>
      {comparison
        ? (
          <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <SectionHeading
              eyebrow="Comparison result"
              title="版本与运行差异"
              detail={`已比较 ${comparison.run_ids.length} 次运行；基准 ${
                comparison.reference_run_id ?? comparison.run_ids[0] ?? "—"
              }。`}
            />
            {comparison.rows?.length
              ? (
                <div className="mt-4 overflow-x-auto">
                  <table className="w-full min-w-[780px] text-sm">
                    <thead className="text-left text-xs text-slate-600">
                      <tr>
                        <th scope="col" className="pb-2">运行</th>
                        <th scope="col" className="pb-2">对象</th>
                        <th scope="col" className="pb-2 text-right">期末收益</th>
                        <th scope="col" className="pb-2 text-right">ES 95%</th>
                        <th scope="col" className="pb-2 text-right">最大回撤</th>
                        <th scope="col" className="pb-2 text-right">突破</th>
                        <th scope="col" className="pb-2 text-right">覆盖率</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {comparison.rows.map((row, index) => (
                        <tr key={`${row.run_id}-${row.portfolio_id ?? index}`}>
                          <td className="py-3 font-mono text-xs">
                            {row.run_id}
                          </td>
                          <td className="py-3">{row.portfolio_id ?? "—"}</td>
                          <td className="py-3 text-right">
                            {formatPercent(row.terminal_return)}
                          </td>
                          <td className="py-3 text-right">
                            {formatPercent(row.es_95)}
                          </td>
                          <td className="py-3 text-right">
                            {formatPercent(row.max_drawdown)}
                          </td>
                          <td className="py-3 text-right">
                            {row.breach_count ?? "—"}
                          </td>
                          <td className="py-3 text-right">
                            {formatPercent(row.coverage)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )
              : (
                <p className="mt-4 rounded-xl bg-slate-50 p-4 text-sm text-slate-600">
                  后端已完成比较，但当前算法未返回逐对象指标行。运行
                  ID：{comparison.run_ids.join("、")}
                </p>
              )}
          </section>
        )
        : null}
    </div>
  );
}

function PublishPanel({
  meta,
  draft,
  run,
  dirty,
  busy,
  bindings,
  onBatch,
  onPublish,
}: {
  meta: ScenarioStressMeta;
  draft: ScenarioDefinition;
  run: ScenarioStressRun | null;
  dirty: boolean;
  busy: string;
  bindings: ScenarioApplicationBinding[];
  onBatch: () => void;
  onPublish: (usages: ScenarioPublicationUsage[], note: string) => void;
}) {
  const usages = meta.application_targets.length
    ? meta.application_targets.map((item) => item.id)
    : Object.keys(usageLabels) as ScenarioPublicationUsage[];
  const [selected, setSelected] = useState<ScenarioPublicationUsage[]>([]);
  const [note, setNote] = useState("");
  const eligibleUsages = new Set(
    run?.governance?.publish_eligible_usages ?? usages,
  );
  const eligibleUsageKey = [...eligibleUsages].sort().join("|");
  useEffect(() => {
    setSelected((current) =>
      current.filter((usage) => eligibleUsages.has(usage))
    );
  }, [run?.id, eligibleUsageKey]);
  const exactVersion = Boolean(
    run?.definition_id &&
      run.definition_id === draft.id &&
      run.definition_revision === draft.revision &&
      run.immutable &&
      !dirty &&
      run.definition &&
      definitionSignature(run.definition) === definitionSignature(draft),
  );
  return (
    <div className="grid gap-5 xl:grid-cols-2">
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <SectionHeading
          eyebrow="Batch stress"
          title="一次压测多个产品或组合"
          detail="批量运行复用同一情景快照，逐对象返回覆盖率、失败原因和损益结果。"
        />
        <div className="mt-5 rounded-xl bg-slate-50 p-4">
          <p className="text-sm font-bold text-slate-900">
            当前定义包含 {draft.portfolios.length} 个对象
          </p>
          <ul className="mt-3 space-y-2 text-xs text-slate-600">
            {draft.portfolios.map((portfolio) => (
              <li key={portfolio.id}>• {portfolio.name}</li>
            ))}
          </ul>
        </div>
        <button
          type="button"
          disabled={dirty || draft.portfolios.length < 2 || !draft.id ||
            !draft.revision || Boolean(busy)}
          onClick={onBatch}
          className="mt-4 min-h-11 w-full rounded-xl bg-slate-950 px-4 text-sm font-bold text-white disabled:bg-slate-300"
        >
          {busy === "batch" ? "批量运行中…" : "运行批量组合压测"}
        </button>
        {draft.portfolios.length < 2
          ? (
            <p className="mt-2 text-xs text-amber-800">
              至少配置两个对象后才能批量运行。
            </p>
          )
          : null}
      </section>
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <SectionHeading
          eyebrow="Publish gate"
          title="按用途发布当前不可变运行"
          detail="只有与当前定义修订完全一致的后端运行可以发布；没有模拟审批或页面内假状态。"
        />
        {run
          ? (
            <div
              className={cx(
                "mt-5 rounded-xl border p-4 text-sm",
                exactVersion
                  ? "border-emerald-200 bg-emerald-50"
                  : "border-amber-200 bg-amber-50",
              )}
            >
              <p className="font-bold">
                {run.id} · R{run.definition_revision ?? "—"}
              </p>
              <p className="mt-1 text-xs">
                {exactVersion
                  ? "版本一致，等待后端执行用途门禁。"
                  : "运行与当前草稿不一致，请保存并重新运行。"}
              </p>
            </div>
          )
          : (
            <p className="mt-5 rounded-xl bg-slate-50 p-4 text-sm text-slate-600">
              尚无可发布运行。
            </p>
          )}
        <fieldset className="mt-4">
          <legend className="text-xs font-bold text-slate-600">应用用途</legend>
          <div className="mt-2 grid gap-2 sm:grid-cols-2">
            {usages.map((usage) => (
              <label
                key={usage}
                className={cx(
                  "flex min-h-11 items-center gap-2 rounded-xl border border-slate-200 px-3 text-sm",
                  !eligibleUsages.has(usage) && "bg-slate-50 text-slate-600",
                )}
              >
                <input
                  type="checkbox"
                  disabled={!eligibleUsages.has(usage)}
                  checked={selected.includes(usage)}
                  onChange={() =>
                    setSelected((current) =>
                      current.includes(usage)
                        ? current.filter((item) => item !== usage)
                        : [...current, usage]
                    )}
                />
                {usageLabels[usage] ?? usage}
                {!eligibleUsages.has(usage)
                  ? <span className="ml-auto text-xs">门禁阻断</span>
                  : null}
              </label>
            ))}
          </div>
        </fieldset>
        {run?.governance?.publication_blockers?.length
          ? (
            <ul className="mt-3 space-y-1 rounded-xl bg-amber-50 p-3 text-xs text-amber-900">
              {run.governance.publication_blockers.map((message) => (
                <li key={message}>• {message}</li>
              ))}
            </ul>
          )
          : null}
        <label className="mt-4 block text-sm font-semibold text-slate-700">
          发布说明<textarea
            aria-label="发布说明"
            rows={2}
            value={note}
            onChange={(event) => setNote(event.target.value)}
            className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2"
          />
        </label>
        <button
          type="button"
          disabled={!exactVersion || !selected.length || Boolean(busy)}
          onClick={() => onPublish(selected, note)}
          className="mt-4 min-h-11 w-full rounded-xl bg-accent-600 px-4 text-sm font-bold text-white disabled:bg-slate-300"
        >
          {busy === "publish" ? "发布中…" : "发布到选定用途"}
        </button>
      </section>
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm xl:col-span-2">
        <SectionHeading
          eyebrow="Application lineage"
          title="真实应用绑定关系"
          detail="这里只展示后端返回的发布和消费关系，不推断业务系统已接入。"
        />
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[680px] text-sm">
            <caption className="sr-only">情景应用绑定关系</caption>
            <thead className="text-left text-xs text-slate-600">
              <tr>
                <th scope="col" className="pb-2">用途</th>
                <th scope="col" className="pb-2">消费者</th>
                <th scope="col" className="pb-2">运行</th>
                <th scope="col" className="pb-2">修订</th>
                <th scope="col" className="pb-2">状态</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {bindings.map((binding, index) => (
                <tr key={`${binding.publication_id ?? binding.usage}-${index}`}>
                  <td className="py-3">
                    {usageLabels[binding.usage] ?? binding.usage}
                  </td>
                  <td className="py-3">
                    {binding.name ?? binding.path ?? "—"}
                  </td>
                  <td className="py-3 font-mono text-xs">
                    {binding.run_id ?? run?.id ?? "—"}
                  </td>
                  <td className="py-3">
                    R{binding.definition_revision ?? binding.revision ??
                      run?.definition_revision ?? "—"}
                  </td>
                  <td className="py-3">{binding.status ?? "已发布"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!bindings.length
            ? (
              <p className="p-8 text-center text-sm text-slate-600">
                当前运行没有后端返回的应用绑定。
              </p>
            )
            : null}
        </div>
      </section>
    </div>
  );
}

export function ScenarioSimulationCenter() {
  const [activeTab, setActiveTab] = useState<WorkspaceTab>("configure");
  const [configureStep, setConfigureStep] = useState(0);
  const [meta, setMeta] = useState<ScenarioStressMeta | null>(null);
  const [definitions, setDefinitions] = useState<ScenarioDefinition[]>([]);
  const [runs, setRuns] = useState<ScenarioStressRun[]>([]);
  const [historicalRuns, setHistoricalRuns] = useState<HistoricalRegimeRun[]>(
    [],
  );
  const [draft, setDraft] = useState<ScenarioDefinition | null>(null);
  const [saved, setSaved] = useState<ScenarioDefinition | null>(null);
  const [run, setRun] = useState<ScenarioStressRun | null>(null);
  const [comparison, setComparison] = useState<ScenarioRunComparison | null>(
    null,
  );
  const [bindings, setBindings] = useState<ScenarioApplicationBinding[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);

  useEffect(() => {
    let active = true;
    Promise.all([
      getScenarioStressMeta(),
      listScenarioStressDefinitions(),
      listScenarioStressRuns(),
      listHistoricalRegimeRuns().catch(() => []),
    ])
      .then(([nextMeta, nextDefinitions, nextRuns, nextHistoricalRuns]) => {
        if (!active) return;
        setMeta(nextMeta);
        setDefinitions(nextDefinitions);
        setRuns(nextRuns);
        setHistoricalRuns(nextHistoricalRuns);
        const initialSaved = nextDefinitions[0] ?? null;
        const initialDraft = initialSaved
          ? cloneDefinition(initialSaved)
          : nextMeta.templates.map(normalizeTemplate).find((
            item,
          ): item is ScenarioDefinition => Boolean(item)) ?? null;
        setSaved(initialSaved ? cloneDefinition(initialSaved) : null);
        setDraft(initialDraft ? cloneDefinition(initialDraft) : null);
        const initialRun = initialSaved
          ? nextRuns.find((item) => item.definition_id === initialSaved.id) ??
            null
          : null;
        setRun(initialRun);
        setBindings(initialRun?.application_bindings ?? []);
      })
      .catch((reason) => {
        if (active) {
          setError(
            reason instanceof Error ? reason.message : "情景工作台加载失败。",
          );
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const dirty = useMemo(
    () => definitionSignature(draft) !== definitionSignature(saved),
    [draft, saved],
  );
  const changeDraft = (next: ScenarioDefinition) => {
    setDraft(next);
    setError("");
    setNotice("");
  };
  const withBusy = async (kind: string, action: () => Promise<void>) => {
    setBusy(kind);
    setError("");
    setNotice("");
    try {
      await action();
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "操作失败，请稍后再试。",
      );
    } finally {
      setBusy("");
    }
  };
  const save = () =>
    draft && void withBusy("save", async () => {
      const next = draft.id
        ? await updateScenarioStressDefinition(draft)
        : await createScenarioStressDefinition(draft);
      setSaved(cloneDefinition(next));
      setDraft(cloneDefinition(next));
      setDefinitions((current) => [
        next,
        ...current.filter((item) => item.id !== next.id),
      ]);
      setNotice(`已保存 ${next.name} 修订版 R${next.revision ?? "—"}。`);
    });
  const execute = () =>
    saved?.id && saved.revision && !dirty && void withBusy("run", async () => {
      const next = await runScenarioStress({
        id: saved.id!,
        revision: saved.revision!,
      });
      setRun(next);
      setRuns((
        current,
      ) => [next, ...current.filter((item) => item.id !== next.id)]);
      setBindings(next.application_bindings ?? []);
      setActiveTab("result");
      setNotice(`已完成不可变运行 ${next.id}。`);
    });
  const selectDefinition = (id: string) =>
    void withBusy("load", async () => {
      if (!id) {
        const initial = meta?.templates.map(normalizeTemplate).find((
          item,
        ): item is ScenarioDefinition => Boolean(item));
        if (initial) {
          const next = cloneDefinition(initial);
          delete next.id;
          delete next.revision;
          setDraft(next);
          setSaved(null);
          setRun(null);
          setBindings([]);
        }
        return;
      }
      const next = await getScenarioStressDefinition(id);
      setDraft(cloneDefinition(next));
      setSaved(cloneDefinition(next));
      const matchingRun = runs.find((item) => item.definition_id === id) ??
        null;
      setRun(matchingRun);
      setBindings(matchingRun?.application_bindings ?? []);
    });
  const compare = (ids: string[]) =>
    void withBusy(
      "compare",
      async () => setComparison(await compareScenarioStressRuns(ids, ids[0])),
    );
  const batch = () => {
    if (dirty || !saved?.id || !saved.revision) {
      setError("请先保存当前参数，再按精确版本运行批量压测。");
      return;
    }
    void withBusy("batch", async () => {
      const next = await batchRunScenarioStress({
        id: saved.id!,
        revision: saved.revision!,
      });
      setRun(next);
      setRuns((
        current,
      ) => [next, ...current.filter((item) => item.id !== next.id)]);
      setBindings(next.application_bindings ?? []);
      setActiveTab("result");
      setNotice(`已完成 ${next.results.length} 个对象的批量压测。`);
    });
  };
  const publish = (usages: ScenarioPublicationUsage[], note: string) =>
    run && void withBusy("publish", async () => {
      const response = await publishScenarioStressRun(run.id, usages, note);
      setBindings(response.application_bindings ?? []);
      const publishedRun = {
        ...run,
        publications: response.publications,
        application_bindings: response.application_bindings ?? [],
      };
      setRun(publishedRun);
      setRuns((current) =>
        current.map((item) => item.id === run.id ? publishedRun : item)
      );
      setNotice(`已按 ${usages.length} 个用途发布 ${run.id}。`);
    });
  const handleTabKeyDown = (
    event: KeyboardEvent<HTMLButtonElement>,
    index: number,
  ) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const nextIndex = event.key === "Home"
      ? 0
      : event.key === "End"
      ? tabs.length - 1
      : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) %
        tabs.length;
    setActiveTab(tabs[nextIndex].id);
    tabRefs.current[nextIndex]?.focus();
  };

  if (loading) {
    return (
      <div
        role="status"
        className="rounded-xl border border-slate-200 bg-white p-10 text-center text-sm text-slate-600 shadow-sm"
      >
        正在加载情景模板、版本与运行记录…
      </div>
    );
  }
  if (!meta || !draft) {
    return (
      <div className="space-y-4">
        <Message error={error || "接口没有返回可用情景模板。"} notice="" />
        <button
          type="button"
          onClick={() => window.location.reload()}
          className="min-h-11 rounded-xl bg-slate-950 px-4 text-sm font-bold text-white"
        >
          重新加载
        </button>
      </div>
    );
  }
  const panels = [
    <BlueprintPanel
      key="blueprint"
      meta={meta}
      draft={draft}
      onChange={changeDraft}
    />,
    <PathPanel
      key="path"
      draft={draft}
      historicalRuns={historicalRuns}
      onChange={changeDraft}
    />,
    <ExposurePanel key="exposure" draft={draft} onChange={changeDraft} />,
    <LimitsPanel key="limits" draft={draft} onChange={changeDraft} />,
  ];

  return (
    <div className="space-y-5" data-testid="scenario-algorithm-center">
      <header className="rounded-xl border border-slate-800 bg-slate-950 px-5 py-5 text-white shadow-sm sm:px-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.2em] text-accent-300">
              Forward scenario & stress engine
            </p>
            <h2 className="mt-1 text-2xl font-bold">情景模拟与压测</h2>
            <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-200">
              定义未来冲击、历史重演或反向压力目标，对版本化研究对象计算净值路径、尾部风险、损益来源和约束突破。
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-xs">
            <span
              className="h-2 w-2 rounded-full bg-emerald-400"
              aria-hidden="true"
            />
            <span className="text-slate-600">后端计算协议</span>
            <strong>{meta.schema_version ?? "已连接"}</strong>
          </div>
        </div>
      </header>
      <DefinitionBar
        definitions={definitions}
        draft={draft}
        saved={saved}
        dirty={dirty}
        busy={busy}
        onSelect={selectDefinition}
        onSave={save}
        onRun={execute}
      />
      <Message error={error} notice={notice} />
      <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white p-1.5 shadow-sm">
        <div
          role="tablist"
          aria-label="情景模拟与压测工作区"
          className="grid min-w-[720px] grid-cols-4 gap-1"
        >
          {tabs.map((tab, index) => (
            <button
              key={tab.id}
              ref={(node) => {
                tabRefs.current[index] = node;
              }}
              type="button"
              role="tab"
              id={`scenario-tab-${tab.id}`}
              aria-selected={activeTab === tab.id}
              aria-controls={`scenario-panel-${tab.id}`}
              tabIndex={activeTab === tab.id ? 0 : -1}
              onClick={() => setActiveTab(tab.id)}
              onKeyDown={(event) => handleTabKeyDown(event, index)}
              className={cx(
                "min-h-14 rounded-xl px-4 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500",
                activeTab === tab.id
                  ? "bg-slate-950 text-white"
                  : "text-slate-600 hover:bg-slate-50",
              )}
            >
              <span className="block text-sm font-bold">{tab.label}</span>
              <span
                className={cx(
                  "mt-0.5 block text-xs",
                  activeTab === tab.id ? "text-slate-600" : "text-slate-600",
                )}
              >
                {tab.helper}
              </span>
            </button>
          ))}
        </div>
      </div>
      <section
        role="tabpanel"
        id={`scenario-panel-${activeTab}`}
        aria-labelledby={`scenario-tab-${activeTab}`}
      >
        {activeTab === "configure"
          ? (
            <div className="grid gap-5 xl:grid-cols-[250px_minmax(0,1fr)]">
              <nav
                aria-label="情景定义步骤"
                className="h-fit rounded-xl border border-slate-200 bg-white p-3 shadow-sm"
              >
                <ol className="space-y-1">
                  {["模板与方法", "期限与路径", "对象与映射", "约束与阈值"].map(
                    (label, index) => (
                      <li key={label}>
                        <button
                          type="button"
                          aria-current={configureStep === index
                            ? "step"
                            : undefined}
                          onClick={() => setConfigureStep(index)}
                          className={cx(
                            "min-h-12 w-full rounded-xl px-3 text-left text-sm font-bold",
                            configureStep === index
                              ? "bg-slate-950 text-white"
                              : "text-slate-700 hover:bg-slate-50",
                          )}
                        >
                          <span className="mr-2 text-xs opacity-60">
                            0{index + 1}
                          </span>
                          {label}
                        </button>
                      </li>
                    ),
                  )}
                </ol>
              </nav>
              <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
                {panels[configureStep]}
                <div className="mt-6 flex justify-between border-t border-slate-200 pt-4">
                  <button
                    type="button"
                    disabled={configureStep === 0}
                    onClick={() =>
                      setConfigureStep((step) => Math.max(0, step - 1))}
                    className="min-h-10 rounded-xl px-3 text-xs font-bold text-slate-600 disabled:opacity-30"
                  >
                    上一步
                  </button>
                  <button
                    type="button"
                    disabled={configureStep === panels.length - 1}
                    onClick={() =>
                      setConfigureStep((step) =>
                        Math.min(panels.length - 1, step + 1)
                      )}
                    className="min-h-10 rounded-xl bg-accent-50 px-3 text-xs font-bold text-accent-700 disabled:opacity-30"
                  >
                    下一步
                  </button>
                </div>
              </div>
            </div>
          )
          : null}
        {activeTab === "result" ? <ResultPanel run={run} /> : null}
        {activeTab === "compare"
          ? (
            <ComparePanel
              runs={runs}
              comparison={comparison}
              busy={busy === "compare"}
              onCompare={compare}
            />
          )
          : null}
        {activeTab === "publish"
          ? (
            <PublishPanel
              meta={meta}
              draft={draft}
              run={run}
              dirty={dirty}
              busy={busy}
              bindings={bindings}
              onBatch={batch}
              onPublish={publish}
            />
          )
          : null}
      </section>
    </div>
  );
}

export default ScenarioSimulationCenter;
