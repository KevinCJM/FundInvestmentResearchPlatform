import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import ScenarioAlgorithmCenter from "./ScenarioAlgorithmCenter";
import type {
  ScenarioDefinition,
  ScenarioStressRun,
} from "../services/scenarioStress";

vi.mock("echarts-for-react", () => ({
  default: ({ option }: { option: { series?: Array<{ name?: string }> } }) => (
    <div data-testid="stress-chart">
      {option.series?.map((series) => series.name).join("、")}
    </div>
  ),
}));

const fixedExecution = {
  backend: "numba_njit_fixed_signature",
  execution_backend: "numba_njit_fixed_signature",
  nopython: true,
  fully_warmed: true,
  python_fallback: 0,
  object_mode: 0,
  request_time_compilation: 0,
  kernel_signatures: { scenario_stress_numeric_core: ["fixed"] },
};

const definition: ScenarioDefinition = {
  id: "SCN-1",
  revision: 1,
  name: "滞胀多因子路径",
  description: "确定性多资产压力",
  template_id: "factor-path",
  method: "factor_path",
  horizon: 3,
  initial_nav: 1,
  usage_intent: "research_display",
  factors: [
    { id: "growth", label: "增长动能", unit: "σ" },
    { id: "rate", label: "利率变动", unit: "bp" },
  ],
  assets: [
    { id: "equity", label: "A股权益" },
    { id: "bond", label: "中长久期债券" },
  ],
  portfolios: [{
    id: "balanced",
    name: "平衡组合",
    weights: { equity: 0.5, bond: 0.5 },
  }],
  mapping: {
    factor_to_asset: {
      equity: { growth: 0.04, rate: -0.0002 },
      bond: { growth: -0.01, rate: -0.0005 },
    },
    missing_policy: "block",
    minimum_coverage: 1,
    response_space: "simple_return",
  },
  scenario: {
    shocks: { growth: -1, rate: 40 },
    path_shape: "linear",
    severity: 1,
  },
  limits: [{
    id: "loss",
    label: "累计亏损",
    metric: "terminal_return",
    operator: "lt",
    threshold: -0.1,
  }],
};

const deterministicRun: ScenarioStressRun = {
  id: "RUN-1",
  name: definition.name,
  method: "factor_path",
  probabilistic: false,
  definition_id: "SCN-1",
  definition_revision: 1,
  definition,
  created_at: "2026-09-03T09:00:00Z",
  immutable: true,
  content_hash: "hash-1",
  compute_audit: {
    ...fixedExecution,
    kernel_coverage: "22/22",
    fingerprint: "0123456789abcdef0123456789abcdef",
  },
  results: [{
    portfolio_id: "balanced",
    name: "平衡组合",
    coverage: {
      status: "complete",
      ratio: 1,
      covered_assets: ["equity", "bond"],
      missing_assets: [],
      policy: "block",
    },
    summary: {
      terminal_nav: 0.94,
      terminal_return: -0.06,
      max_drawdown: 0.06,
      worst_step_return: -0.025,
      breach_count: 0,
      first_breach_step: null,
      recovery_steps: null,
    },
    path: [
      {
        step: 1,
        return: -0.02,
        nav: 0.98,
        drawdown: 0.02,
        asset_returns: { equity: -0.03, bond: -0.01 },
        contributions: { equity: -0.015, bond: -0.005 },
      },
      {
        step: 2,
        return: -0.025,
        nav: 0.9555,
        drawdown: 0.0445,
        asset_returns: { equity: -0.04, bond: -0.01 },
        contributions: { equity: -0.02, bond: -0.005 },
      },
      {
        step: 3,
        return: -0.0162,
        nav: 0.94,
        drawdown: 0.06,
        asset_returns: { equity: -0.0224, bond: -0.01 },
        contributions: { equity: -0.0112, bond: -0.005 },
      },
    ],
    contributions: {
      by_asset: { equity: -0.0462, bond: -0.015 },
      method: "sum_of_period_contributions",
    },
    limits: [{
      id: "loss",
      label: "累计亏损",
      breached: false,
      value: -0.06,
      threshold: -0.1,
    }],
  }],
  mapping_diagnostics: [],
  diagnostics: [{
    level: "info",
    message: "这是确定性情景，结果不包含概率、VaR 或 ES。",
  }],
  publications: [],
  application_bindings: [],
};

const monteCarloDefinition: ScenarioDefinition = {
  ...definition,
  id: "SCN-MC",
  revision: 1,
  name: "条件蒙特卡罗",
  template_id: "monte-carlo",
  method: "monte_carlo",
  mapping: { ...definition.mapping, response_space: "log_return" },
  scenario: {
    distribution: "normal",
    factor_means: { growth: 0, rate: 0 },
    factor_volatilities: { growth: 0.1, rate: 4 },
    correlation: [[1, 0], [0, 1]],
    path_count: 2000,
    seed: 20260903,
    target_return: 0.03,
  },
  limits: [{
    id: "es",
    label: "尾部损失",
    metric: "es_95",
    operator: "gt",
    threshold: 0.1,
  }],
};

const monteCarloRun: ScenarioStressRun = {
  ...deterministicRun,
  id: "RUN-MC",
  name: monteCarloDefinition.name,
  method: "monte_carlo",
  probabilistic: true,
  definition_id: "SCN-MC",
  definition_revision: 1,
  definition: monteCarloDefinition,
  results: [{
    portfolio_id: "balanced",
    name: "平衡组合",
    coverage: {
      status: "degraded",
      ratio: 0.75,
      covered_assets: ["equity"],
      missing_assets: ["bond"],
      policy: "degrade",
    },
    summary: {
      terminal_nav: 1.03,
      terminal_return: 0.03,
      max_drawdown: 0.14,
      worst_step_return: null,
      breach_count: 1,
    },
    contributions: {
      by_asset: { equity: 0.02, bond: null },
      method: "mean_sum_of_period_contributions",
    },
    distribution: {
      fan: {
        steps: [0, 1, 2, 3],
        quantiles: {
          p05: [1, 0.95, 0.93, 0.9],
          p50: [1, 1.01, 1.02, 1.03],
          p95: [1, 1.06, 1.1, 1.15],
        },
      },
      terminal: {
        p05: 0.9,
        p50: 1.03,
        p95: 1.15,
        return_p05: -0.1,
        return_p50: 0.03,
        return_p95: 0.15,
        var_95: 0.1,
        es_95: 0.13,
        loss_probability: 0.38,
        target_hit_probability: 0.5,
        average_max_drawdown: 0.08,
      },
    },
    limits: [{
      id: "es",
      label: "尾部损失",
      breached: true,
      value: 0.13,
      threshold: 0.1,
    }],
  }],
  diagnostics: [{
    level: "info",
    message: "概率来自带固定种子的随机路径频率。",
  }],
  governance: {
    publish_eligible_usages: ["research_display", "product_research"],
    publication_blockers: [
      "存在覆盖率降级；正式回测、TAA 与风险监控发布已阻断。",
    ],
  },
};

const template = (next: ScenarioDefinition, phase: string) => ({
  id: next.template_id!,
  name: next.name,
  phase,
  description: next.description,
  definition: { ...next, id: undefined, revision: undefined },
});

const meta = {
  schema_version: "1.0",
  methods: [
    { id: "historical_replay", label: "历史重演", probabilistic: false },
    { id: "factor_path", label: "因子路径", probabilistic: false },
    { id: "monte_carlo", label: "蒙特卡罗", probabilistic: true },
    { id: "regime_conditioned", label: "状态条件", probabilistic: true },
    { id: "reverse_stress", label: "反向压力", probabilistic: false },
  ],
  templates: [
    template(definition, "P0"),
    template({
      ...definition,
      template_id: "historical",
      name: "历史危机重演",
      method: "historical_replay",
      scenario: { historical_returns: [] },
    }, "P0"),
    template(monteCarloDefinition, "P1"),
    template({
      ...monteCarloDefinition,
      template_id: "regime",
      name: "牛熊状态条件路径",
      method: "regime_conditioned",
      scenario: {
        transition: {
          states: [],
          matrix: [],
          initial_state: "",
          path_count: 1000,
          seed: 42,
        },
      },
    }, "P1"),
    template({
      ...definition,
      template_id: "reverse",
      name: "损失阈值反推",
      method: "reverse_stress",
      scenario: {
        target_metric: "loss",
        threshold: 0.1,
        bounds: { growth: [-3, 3], rate: [-100, 100] },
      },
    }, "P1"),
  ],
  usages: [
    { id: "research_display", label: "研究展示" },
    { id: "product_research", label: "产品研究" },
    { id: "portfolio_backtest", label: "组合回测" },
    { id: "taa", label: "TAA" },
    { id: "risk_monitoring", label: "风险监控" },
  ],
  limits: { max_paths: 50000 },
};

const ok = (
  body: unknown,
) => ({ ok: true, status: 200, json: async () => body } as Response);

function makeFetch(
  options?: {
    initialDefinition?: ScenarioDefinition;
    initialRun?: ScenarioStressRun;
    failRun?: boolean;
    historicalRuns?: unknown[];
    invalidWeightAudit?: boolean;
  },
) {
  const initialDefinition = options?.initialDefinition ?? definition;
  const initialRun = options?.initialRun ?? deterministicRun;
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input);
    if (path.endsWith("/meta")) return ok(meta);
    if (path.endsWith("/weight-summary") && init?.method === "POST") {
      return ok({
        weight_count: 2,
        total_weight: 1,
        net_exposure: 1,
        gross_exposure: 1,
        largest_absolute_weight: 0.5,
        finite: true,
        sums_to_one: true,
        single_weight_valid: true,
        gross_limit_valid: true,
        within_tolerance: true,
        status_code: 0,
        limits: {
          expected_net_exposure: 1,
          absolute_tolerance: 1e-8,
          maximum_absolute_weight: 2,
          maximum_gross_exposure: 3,
        },
        execution: options?.invalidWeightAudit
          ? { ...fixedExecution, python_fallback: 1 }
          : fixedExecution,
      });
    }
    if (
      path.endsWith("/api/historical-regimes/runs") ||
      path.endsWith("/historical-regimes/runs")
    ) return ok({ items: options?.historicalRuns ?? [] });
    if (
      path.endsWith("/definitions") && (!init?.method || init.method === "GET")
    ) return ok({ items: [initialDefinition] });
    if (path.endsWith("/runs") && (!init?.method || init.method === "GET")) {
      return ok({ items: [initialRun, monteCarloRun] });
    }
    if (
      /\/definitions\/[^?]+/.test(path) &&
      (!init?.method || init.method === "GET")
    ) return ok(initialDefinition);
    if (/\/definitions\//.test(path) && init?.method === "PUT") {
      return ok({
        ...initialDefinition,
        ...JSON.parse(String(init.body)),
        revision: 2,
      });
    }
    if (path.endsWith("/definitions") && init?.method === "POST") {
      return ok({
        ...JSON.parse(String(init.body)),
        id: "SCN-NEW",
        revision: 1,
      });
    }
    if (path.endsWith("/run") && init?.method === "POST") {
      if (options?.failRun) {
        return {
          ok: false,
          status: 422,
          json: async () => ({
            detail: {
              code: "WEIGHTS_DO_NOT_SUM_TO_ONE",
              message: "组合权重之和必须等于 1。",
            },
          }),
        } as Response;
      }
      const body = JSON.parse(String(init.body));
      return ok({
        ...initialRun,
        id: "RUN-NEW",
        definition_id: body.definition.id,
        definition_revision: body.definition.revision,
      });
    }
    if (path.endsWith("/compare") && init?.method === "POST") {
      return ok({
        run_ids: ["RUN-1", "RUN-MC"],
        reference_run_id: "RUN-1",
        runs: [{
          run_id: "RUN-1",
          method: "factor_path",
          probabilistic: false,
          metrics: {
            balanced: {
              terminal_return: -0.06,
              max_drawdown: 0.06,
              breach_count: 0,
              coverage_ratio: 1,
            },
          },
        }, {
          run_id: "RUN-MC",
          method: "monte_carlo",
          probabilistic: true,
          metrics: {
            balanced: {
              terminal_return: 0.03,
              max_drawdown: 0.14,
              es_95: -0.12,
              breach_count: 1,
              coverage_ratio: 0.75,
            },
          },
        }],
        execution: fixedExecution,
      });
    }
    if (path.endsWith("/batch-run") && init?.method === "POST") {
      return ok({
        ...deterministicRun,
        id: "RUN-BATCH",
        definition: initialDefinition,
        results: [deterministicRun.results[0], {
          ...deterministicRun.results[0],
          portfolio_id: "second",
          name: "第二组合",
        }],
      });
    }
    if (path.includes("/publish") && init?.method === "POST") {
      return ok({
        run_id: initialRun.id,
        publications: [{
          id: "PUB-1",
          usage: "product_research",
          published_at: "2026-09-03",
          definition_revision: 1,
          run_id: initialRun.id,
        }],
        application_bindings: [{
          usage: "product_research",
          name: "产品研究净值区间",
          run_id: initialRun.id,
          revision: 1,
          status: "active",
        }],
      });
    }
    throw new Error(`Unexpected request: ${path} ${init?.method || "GET"}`);
  });
}

describe("ScenarioAlgorithmCenter", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("真实加载元数据、定义和运行，展示五类方法且不保留静态演示", async () => {
    const fetchMock = makeFetch();
    vi.stubGlobal("fetch", fetchMock);
    render(<ScenarioAlgorithmCenter />);

    expect(await screen.findByRole("heading", { name: "情景模拟与压测" }))
      .toBeInTheDocument();
    expect(screen.getByRole("button", { name: /历史事件重演/ }))
      .toBeInTheDocument();
    expect(screen.getByRole("button", { name: /确定性因子路径/ }))
      .toBeInTheDocument();
    expect(screen.getByRole("button", { name: /蒙特卡罗模拟/ }))
      .toBeInTheDocument();
    expect(screen.getByRole("button", { name: /历史状态条件模拟/ }))
      .toBeInTheDocument();
    expect(screen.getByRole("button", { name: /反向压力测试/ }))
      .toBeInTheDocument();
    expect(screen.queryByText(/静态计算引擎|不调用后端|INTERACTIVE PROTOTYPE/i))
      .not.toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/scenario-stress/meta",
      expect.any(Object),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/scenario-stress/definitions",
      expect.any(Object),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/scenario-stress/runs",
      expect.any(Object),
    );
  });

  it("权重合计只展示后端固定签名 NJIT 核验结果", async () => {
    const fetchMock = makeFetch();
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<ScenarioAlgorithmCenter />);
    await screen.findByRole("heading", { name: "情景模拟与压测" });

    await user.click(screen.getByRole("button", { name: /03.*对象与映射/ }));
    const totalRow = (await screen.findByText("权重合计")).closest("tr");
    expect(totalRow).not.toBeNull();
    expect(await within(totalRow!).findByText("100%")).toBeInTheDocument();
    const call = fetchMock.mock.calls.find(([path]) =>
      String(path).endsWith("/weight-summary")
    );
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({
      weights: { equity: 0.5, bond: 0.5 },
    });
  });

  it("权重核验缺少合规执行证明时失败关闭且不在浏览器回退求和", async () => {
    vi.stubGlobal("fetch", makeFetch({ invalidWeightAudit: true }));
    const user = userEvent.setup();
    render(<ScenarioAlgorithmCenter />);
    await screen.findByRole("heading", { name: "情景模拟与压测" });

    await user.click(screen.getByRole("button", { name: /03.*对象与映射/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "情景组合权重核验未提供有效的固定签名 NJIT 执行证明",
    );
    const totalRow = screen.getByText("权重合计").closest("tr");
    expect(within(totalRow!).getByText("不可用")).toBeInTheDocument();
  });

  it("参数改变后必须保存新修订，再按精确版本运行", async () => {
    const fetchMock = makeFetch();
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<ScenarioAlgorithmCenter />);
    await screen.findByRole("heading", { name: "情景模拟与压测" });

    fireEvent.change(screen.getByLabelText("定义名称"), {
      target: { value: "滞胀多因子路径修订" },
    });
    expect(screen.getByText("参数已变化 · 结果会过期")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "按当前版本运行" }))
      .toBeDisabled();
    await act(async () => {
      await user.click(screen.getByRole("button", { name: "保存新修订" }));
    });
    expect(await screen.findByText(/修订版 R2/)).toBeInTheDocument();
    await act(async () => {
      await user.click(screen.getByRole("button", { name: "按当前版本运行" }));
    });
    expect(await screen.findByText(/不可变运行 RUN-NEW/)).toBeInTheDocument();

    const runCall = fetchMock.mock.calls.find(([path, init]) =>
      String(path).endsWith("/run") && init?.method === "POST"
    );
    expect(JSON.parse(String(runCall?.[1]?.body))).toEqual({
      definition: { id: "SCN-1", revision: 2 },
    });
  });

  it("确定性结果不展示伪概率，概率型结果展示分位扇形和失败对象", async () => {
    vi.stubGlobal("fetch", makeFetch());
    const createObjectURL = vi.fn(() => "blob:scenario-run");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal(
      "URL",
      class extends URL {
        static createObjectURL = createObjectURL;
        static revokeObjectURL = revokeObjectURL;
      },
    );
    const download = vi.spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    const user = userEvent.setup();
    render(<ScenarioAlgorithmCenter />);
    await screen.findByRole("heading", { name: "情景模拟与压测" });
    await act(async () => {
      await user.click(screen.getByRole("tab", { name: /结果与归因/ }));
    });

    expect(screen.getByText("期末收益")).toBeInTheDocument();
    expect(screen.getByText("固定签名 NJIT · 已通过")).toBeInTheDocument();
    expect(screen.getByText("22/22")).toBeInTheDocument();
    expect(screen.queryByText("损失概率")).not.toBeInTheDocument();
    expect(screen.queryByText("ES 95%")).not.toBeInTheDocument();
    expect(screen.getByTestId("stress-chart")).toHaveTextContent("净值、回撤");
    await act(async () => {
      await user.click(
        screen.getByRole("button", { name: "导出完整运行 JSON" }),
      );
    });
    expect(createObjectURL).toHaveBeenCalledWith(expect.any(Blob));
    expect(download).toHaveBeenCalledOnce();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:scenario-run");

    await act(async () => {
      await user.click(screen.getByRole("tab", { name: /情景定义/ }));
    });
    fireEvent.change(screen.getByLabelText("已保存情景定义"), {
      target: { value: "SCN-1" },
    });
  });

  it("蒙特卡罗版本展示随机参数、真实概率指标、分位扇形与覆盖缺口", async () => {
    vi.stubGlobal(
      "fetch",
      makeFetch({
        initialDefinition: monteCarloDefinition,
        initialRun: monteCarloRun,
      }),
    );
    const user = userEvent.setup();
    render(<ScenarioAlgorithmCenter />);
    await screen.findByRole("heading", { name: "情景模拟与压测" });
    await act(async () => {
      await user.click(screen.getByRole("button", { name: "02 期限与路径" }));
    });
    expect(screen.getByLabelText("模拟路径数")).toHaveValue(2000);
    expect(screen.getByLabelText("随机种子")).toHaveValue(20260903);
    await act(async () => {
      await user.click(screen.getByRole("tab", { name: /结果与归因/ }));
    });
    expect(screen.getByText("损失概率")).toBeInTheDocument();
    expect(screen.getByText("ES 95%")).toBeInTheDocument();
    expect(screen.getByTestId("stress-chart")).toHaveTextContent("P05");
    expect(screen.getByText(/失败 \/ 缺失对象：bond/)).toBeInTheDocument();
    expect(screen.getByText("不可计算")).toBeInTheDocument();
    await act(async () => {
      await user.click(screen.getByRole("tab", { name: /批量压测与发布/ }));
    });
    expect(
      within(screen.getByText("TAA 研究").closest("label")!)
        .getByRole("checkbox"),
    ).toBeDisabled();
    expect(screen.getByText(/存在覆盖率降级/)).toBeInTheDocument();
  });

  it("状态条件模拟从已发布历史识别版本选择，并自动写入双 ID 血缘", async () => {
    const regimeDefinition: ScenarioDefinition = {
      ...monteCarloDefinition,
      id: "SCN-REGIME",
      name: "状态条件模拟",
      method: "regime_conditioned",
      usage_intent: "taa",
      mapping: {
        ...definition.mapping,
        factor_to_asset: {},
        response_space: "direct_simple_return",
      },
      factors: [],
      scenario: {
        transition: {
          states: [],
          matrix: [],
          initial_state: "",
          path_count: 1000,
          seed: 42,
        },
      },
    };
    const historicalRun = {
      id: "HIST-RUN-1",
      name: "沪深300牛熊识别",
      definition_revision: 3,
      mode: "realtime",
      calculation_audits: [{
        source_kind: "historical_regime_algorithm",
        ...fixedExecution,
      }],
      causality: {
        is_causal: true,
        repaints: false,
        uses_future_data: false,
      },
      publications: [{
        id: "HIST-PUB-1",
        usage: "taa",
        published_at: "2026-09-03",
        definition_revision: 3,
        run_id: "HIST-RUN-1",
      }],
    };
    const fetchMock = makeFetch({
      initialDefinition: regimeDefinition,
      historicalRuns: [historicalRun, {
        ...historicalRun,
        id: "HIST-RUN-BLOCKED",
        name: "事后牛熊划分",
        mode: "retrospective",
        causality: {
          is_causal: false,
          repaints: true,
          uses_future_data: true,
        },
        publications: [{
          id: "HIST-PUB-BLOCKED",
          usage: "research_display",
          published_at: "2026-09-03",
          definition_revision: 1,
          run_id: "HIST-RUN-BLOCKED",
        }],
      }],
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<ScenarioAlgorithmCenter />);
    await screen.findByRole("heading", { name: "情景模拟与压测" });
    await act(async () => {
      await user.click(screen.getByRole("button", { name: "02 期限与路径" }));
    });
    expect(screen.queryByRole("option", { name: /事后牛熊划分/ }))
      .not.toBeInTheDocument();

    await act(async () => {
      await user.selectOptions(
        screen.getByLabelText("历史状态版本"),
        "HIST-RUN-1|HIST-PUB-1",
      );
    });
    expect(screen.getByText(/已锁定 沪深300牛熊识别 · R3/)).toBeInTheDocument();
    expect(screen.getByText(/实时识别 · 因果门禁合格/)).toBeInTheDocument();
    expect(screen.getByText(/没有可用的、与发布内容一致的 v2 评价目标制品/))
      .toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "历史评价目标联合经验分布" }))
      .toBeDisabled();
    await act(async () => {
      await user.click(screen.getByRole("button", { name: "保存新修订" }));
    });
    const saveCall = fetchMock.mock.calls.find(([path, init]) =>
      String(path).includes("/definitions/") && init?.method === "PUT"
    );
    const savedBody = JSON.parse(String(saveCall?.[1]?.body));
    expect(savedBody.scenario.historical_run_ref)
      .toEqual({ run_id: "HIST-RUN-1", publication_id: "HIST-PUB-1" });
    expect(savedBody.mapping.response_space).toBe("direct_simple_return");
  });

  it("状态条件模拟锁定 v2 评价目标制品、配置逐资产口径且切换运行会清除旧锁", async () => {
    const regimeDefinition: ScenarioDefinition = {
      ...monteCarloDefinition,
      id: "SCN-REGIME-V2",
      name: "状态条件联合经验分布",
      method: "regime_conditioned",
      usage_intent: "taa",
      mapping: {
        ...definition.mapping,
        factor_to_asset: {},
        response_space: "direct_simple_return",
      },
      factors: [],
      scenario: {
        transition: { path_count: 1000, seed: 42 },
      },
    };
    const makeHistoricalRun = (suffix: "A" | "B") => {
      const runHash = (suffix === "A" ? "a" : "c").repeat(64);
      const artifactHash = (suffix === "A" ? "b" : "d").repeat(64);
      return {
        id: `HIST-RUN-${suffix}`,
        schema_version: "2.0",
        name: `历史状态图谱 ${suffix}`,
        definition_revision: suffix === "A" ? 3 : 4,
        mode: "realtime",
        content_hash: runHash,
        causality: {
          is_causal: true,
          repaints: false,
          uses_future_data: false,
        },
        evaluation_results: {
          equity_eval: { id: "equity_eval", name: "沪深300", primary: true },
          bond_eval: { id: "bond_eval", name: "中债综合", primary: false },
        },
        artifact_manifest: {
          evaluation_targets: {
            artifact_id: `regime-output-sha256-${artifactHash}`,
            checksum: `sha256:${artifactHash}`,
            format: "npz",
            schema_version: "regime-node-output-v1",
            arrays: [
              { node_id: "equity_eval", port: "value" },
              { node_id: "bond_eval", port: "value" },
            ],
          },
        },
        publications: [{
          id: `HIST-PUB-${suffix}`,
          usage: "taa",
          published_at: "2026-09-04",
          definition_revision: suffix === "A" ? 3 : 4,
          run_id: `HIST-RUN-${suffix}`,
          run_content_hash: runHash,
        }],
      };
    };
    const fetchMock = makeFetch({
      initialDefinition: regimeDefinition,
      historicalRuns: [makeHistoricalRun("A"), makeHistoricalRun("B")],
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<ScenarioAlgorithmCenter />);
    await screen.findByRole("heading", { name: "情景模拟与压测" });
    await user.click(screen.getByRole("button", { name: "02 期限与路径" }));

    await user.selectOptions(
      screen.getByLabelText("历史状态版本"),
      "HIST-RUN-A|HIST-PUB-A",
    );
    expect(screen.getByText(/可用评价目标 2 项/)).toBeInTheDocument();
    await user.click(screen.getByRole("radio", { name: "历史评价目标联合经验分布" }));
    expect(screen.getByText(/尚有 2 项资产未映射评价目标/)).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("A股权益评价目标"), "equity_eval");
    await user.selectOptions(screen.getByLabelText("中长久期债券评价目标"), "bond_eval");
    await user.selectOptions(screen.getByLabelText("A股权益收益转换"), "forward_value");
    fireEvent.change(screen.getByLabelText("每状态最少样本"), { target: { value: "8" } });
    await user.selectOptions(screen.getByLabelText("内联状态收益策略"), "override");
    expect(screen.getByText(/已允许手工覆盖/)).toBeInTheDocument();

    await user.selectOptions(
      screen.getByLabelText("历史状态版本"),
      "HIST-RUN-B|HIST-PUB-B",
    );
    expect(screen.getByRole("radio", { name: "手工状态收益参数" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "历史评价目标联合经验分布" }))
      .not.toBeChecked();
    expect(screen.queryByText(/已允许手工覆盖/)).not.toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: "历史评价目标联合经验分布" }));
    await user.selectOptions(screen.getByLabelText("A股权益评价目标"), "equity_eval");
    await user.selectOptions(screen.getByLabelText("中长久期债券评价目标"), "bond_eval");
    await user.click(screen.getByRole("button", { name: "保存新修订" }));
    const saveCall = fetchMock.mock.calls.find(([path, init]) =>
      String(path).includes("/definitions/") && init?.method === "PUT"
    );
    const savedBody = JSON.parse(String(saveCall?.[1]?.body));
    expect(savedBody.scenario.historical_run_ref).toEqual({
      run_id: "HIST-RUN-B",
      publication_id: "HIST-PUB-B",
    });
    expect(savedBody.scenario.transition.asset_return_source).toEqual({
      kind: "historical_evaluation_targets",
      run_content_hash: "c".repeat(64),
      evaluation_artifact_checksum: `sha256:${"d".repeat(64)}`,
      asset_target_map: {
        equity: { target_id: "equity_eval", return_transform: "simple_return" },
        bond: { target_id: "bond_eval", return_transform: "simple_return" },
      },
      sampling: "empirical_bootstrap",
      minimum_observations_per_state: 5,
      inline_policy: "forbid",
    });
  });

  it("比较运行调用真实端点并展示逐对象差异", async () => {
    const fetchMock = makeFetch();
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<ScenarioAlgorithmCenter />);
    await screen.findByRole("heading", { name: "情景模拟与压测" });
    await act(async () => {
      await user.click(screen.getByRole("tab", { name: /版本 \/ 运行对比/ }));
    });
    await act(async () => {
      await user.click(screen.getByLabelText("选择运行 RUN-1"));
      await user.click(screen.getByLabelText("选择运行 RUN-MC"));
    });
    await act(async () => {
      await user.click(screen.getByRole("button", { name: "比较 2 次运行" }));
    });
    expect(await screen.findByRole("heading", { name: "版本与运行差异" }))
      .toBeInTheDocument();
    expect(screen.getByText("-6%")).toBeInTheDocument();
    const call = fetchMock.mock.calls.find(([path]) =>
      String(path).endsWith("/compare")
    );
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({
      run_ids: ["RUN-1", "RUN-MC"],
      reference_run_id: "RUN-1",
    });
  });

  it("多对象定义可批量压测，并按用途发布精确运行", async () => {
    const twoPortfolios = {
      ...definition,
      portfolios: [...definition.portfolios, {
        id: "second",
        name: "第二组合",
        weights: { equity: 0.7, bond: 0.3 },
      }],
    };
    const fetchMock = makeFetch({
      initialDefinition: twoPortfolios,
      initialRun: deterministicRun,
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<ScenarioAlgorithmCenter />);
    await screen.findByRole("heading", { name: "情景模拟与压测" });
    await act(async () => {
      await user.click(screen.getByRole("tab", { name: /批量压测与发布/ }));
    });
    expect(screen.getByText("当前定义包含 2 个对象")).toBeInTheDocument();
    await act(async () => {
      await user.click(
        screen.getByRole("button", { name: "运行批量组合压测" }),
      );
    });
    expect(await screen.findByText(/完成 2 个对象的批量压测/))
      .toBeInTheDocument();

    await act(async () => {
      await user.click(screen.getByRole("tab", { name: /批量压测与发布/ }));
    });
    const productCheckbox = within(
      screen.getByText("产品研究").closest("label") as HTMLLabelElement,
    ).getByRole("checkbox");
    await act(async () => {
      await user.click(productCheckbox);
      await user.click(screen.getByRole("button", { name: "发布到选定用途" }));
    });
    expect(await screen.findByText(/已按 1 个用途发布 RUN-BATCH/))
      .toBeInTheDocument();
    expect(screen.getByText("产品研究净值区间")).toBeInTheDocument();
  });

  it("后端错误只展示中文业务信息，不暴露内部错误码", async () => {
    vi.stubGlobal("fetch", makeFetch({ failRun: true }));
    const user = userEvent.setup();
    render(<ScenarioAlgorithmCenter />);
    await screen.findByRole("heading", { name: "情景模拟与压测" });
    await act(async () => {
      await user.click(screen.getByRole("button", { name: "按当前版本运行" }));
    });
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "组合权重之和必须等于 1",
    );
    expect(screen.queryByText("WEIGHTS_DO_NOT_SUM_TO_ONE")).not
      .toBeInTheDocument();
  });
});
