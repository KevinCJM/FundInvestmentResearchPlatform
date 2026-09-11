import React, { useEffect, useMemo, useRef, useState } from 'react';
import TimeSeriesIndicatorPanel from '../components/indicator-parameters/TimeSeriesIndicatorPanel';
import { Link, useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';
import FactorEvidencePanel from '../components/FactorEvidencePanel';
import PublishedRiskPanel from '../components/risk-models/PublishedRiskPanel';
import ProductScenarioPanel from '../components/product-research/ProductScenarioPanel';
import {
  evaluateCustomIndicators,
  evaluateTimeSeriesIndicators,
  getCustomIndicatorMeta,
  indicatorsForContext,
  listCustomIndicators,
  type EvaluationResult,
  type IndicatorDefinition,
  type TimeSeriesIndicatorResult,
} from '../services/customIndicators';
import { latestProductResearchPublication, researchVersionChoices } from '../services/regimeResearchVersions';
import {
  getHistoricalRegimeRun,
  listHistoricalRegimeRuns,
  type HistoricalRegimeRun,
} from '../services/historicalRegimes';
import {
  MetricDefinitionDrawer,
  MetricResultCard,
  MetricSelector,
} from '../components/metrics/MetricDisplay';
import {
  groupIndicatorsByPeriod,
  metricPeriodFor,
  normalizeMetricPeriods,
  useMetricDisplayPreference,
  withSelectedIndicators,
} from '../components/metrics/useMetricDisplayPreference';
import {
  BOOTSTRAP_BLOCK_LENGTH_OPTIONS,
  FHS_EWMA_LAMBDA_OPTIONS,
  MIN_SIMULATION_OBSERVATIONS,
  MONTE_CARLO_HORIZON_OPTIONS,
  MONTE_CARLO_PATH_OPTIONS,
  SIMULATION_METHOD_META,
  SIMULATION_METHOD_ORDER,
  STATISTICS_PERIOD_OPTIONS,
  analyzeProduct,
  navDensityCurve,
  nearestNavDensityFrame,
  type DistributionInterpretation,
  type FuturePathSimulation,
  type ProductAnalysisResponse,
  type ProductAnalysisRequest,
  type ProductRegimeSegment,
  type RealizedFuturePath,
  type RealizedMethodScore,
  type SimulationMethod,
  type StatisticsPeriod,
} from '../services/productAnalysis';
import { useResearchDay } from '../app/ResearchContext';
import { readReturnNavigationState, returnToOrigin } from '../utils/returnNavigation';

interface TimeSeriesPoint {
  date: string;
  open: number | null;
  close: number;
  high: number | null;
  low: number | null;
  volume: number | null;
}

const SERIES_INDICATOR_IDS = {
  PRICE_MA: 'builtin-close-moving-average-series',
  VOLUME_MA: 'builtin-volume-moving-average-series',
  BOLL: 'builtin-bollinger-bands-series',
  KDJ: 'builtin-kdj-series',
} as const;

const CORE_RESEARCH_INDICATOR_IDS = [
  'builtin-total-return-v2',
  'builtin-annualized-return-v2',
  'builtin-annualized-volatility-v2',
  'builtin-drawdown-analysis-v3',
  'builtin-annualized-sharpe-v2',
];

interface ProductDetailResponse {
  product_id: string | null;
  name: string | null;
  management?: string | null;
  custodian?: string | null;
  status?: string | null;
  base_info: Record<string, string | number | null>;
  metrics: {
    issue_amount?: number | null;
    current_size?: number | null;
    current_size_as_of?: string | null;
    current_size_source?: 'instrument_metrics_snapshot' | null;
    current_share?: number | null;
    current_unit_nav?: number | null;
    m_fee?: number | null;
    c_fee?: number | null;
  };
  timeseries: TimeSeriesPoint[];
}

interface RegimeMarkAreaBoundary {
  name?: string;
  xAxis: string;
  itemStyle?: {
    color: string;
    opacity: number;
  };
}

type RegimeMarkArea = [RegimeMarkAreaBoundary, RegimeMarkAreaBoundary];

type OverlayId = 'PRICE_MA' | 'VOLUME_MA' | 'BOLL' | 'KDJ';

interface OverlayOption {
  id: OverlayId;
  label: string;
  description: string;
}

const overlayOptions: OverlayOption[] = [
  { id: 'PRICE_MA', label: '20 日收盘价均线', description: '固定 20 个交易日窗口观察价格趋势' },
  { id: 'VOLUME_MA', label: '10 日成交量均线', description: '固定 10 个交易日窗口观察量能节奏' },
  { id: 'BOLL', label: '20 日布林带', description: '20 日均值加减 2 倍总体标准差' },
  { id: 'KDJ', label: 'KDJ（9, 3, 3）', description: '固定 9 日 RSV、3 日 K 与 D 平滑' },
];

const histogramBinWidthOptions = [
  { label: '0.05%', value: 0.05 },
  { label: '0.10%', value: 0.1 },
  { label: '0.20%', value: 0.2 },
  { label: '0.50%', value: 0.5 },
  { label: '1.00%', value: 1 },
];

const FUTURE_SIMULATION_INITIAL_NAV = 1;
// Near-black against the pastel scenario palette: the overlay is the only line
// on the chart that is a fact, and it has to read that way at a glance.
const REALIZED_COLOR = '#0f172a';
const REALIZED_SERIES_NAME = '实际走势';

const decimalFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 2 });
const signedPercentFormatters: Record<number, Intl.NumberFormat> = {};
const ratioPercentFormatter = new Intl.NumberFormat('zh-CN', {
  style: 'percent',
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const formatIssueAmount = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  if (Math.abs(value) >= 10000) {
    return `${decimalFormatter.format(value / 10000)} 亿`;
  }
  return `${decimalFormatter.format(value)} 万`;
};

const formatPercent = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  return `${decimalFormatter.format(value)}%`;
};

const formatSignedPercent = (value?: number | null, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  if (!Number.isFinite(value)) {
    return '--';
  }
  if (!signedPercentFormatters[digits]) {
    signedPercentFormatters[digits] = new Intl.NumberFormat('zh-CN', {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
      signDisplay: 'always',
    });
  }
  return `${signedPercentFormatters[digits].format(value)}%`;
};

const signedDecimalFormatters: Record<number, Intl.NumberFormat> = {};
const decimalNumberFormatters: Record<number, Intl.NumberFormat> = {};

const formatSignedDecimal = (value?: number | null, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  if (!Number.isFinite(value)) {
    return '--';
  }
  if (!signedDecimalFormatters[digits]) {
    signedDecimalFormatters[digits] = new Intl.NumberFormat('zh-CN', {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
      signDisplay: 'always',
    });
  }
  return signedDecimalFormatters[digits].format(value);
};

const formatDecimal = (value?: number | null, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  if (!Number.isFinite(value)) {
    return '--';
  }
  if (!decimalNumberFormatters[digits]) {
    decimalNumberFormatters[digits] = new Intl.NumberFormat('zh-CN', {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }
  return decimalNumberFormatters[digits].format(value);
};

const formatRatioPercent = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  if (!Number.isFinite(value)) {
    return '--';
  }
  return ratioPercentFormatter.format(value);
};

const formatText = (value?: string | number | null) => {
  if (value === null || value === undefined) {
    return '未知';
  }
  const text = String(value).trim();
  return text.length > 0 ? text : '未知';
};

const formatDate = (value?: string | number | null) => {
  if (value === null || value === undefined) {
    return '未披露';
  }
  const text = String(value).trim();
  if (!text) {
    return '未披露';
  }
  if (/^\d{8}$/.test(text)) {
    return `${text.slice(0, 4)}-${text.slice(4, 6)}-${text.slice(6, 8)}`;
  }
  return text.slice(0, 10);
};

const normalizeDateKey = (value: string) => {
  const text = value.trim();
  if (/^\d{8}$/.test(text)) {
    return `${text.slice(0, 4)}-${text.slice(4, 6)}-${text.slice(6, 8)}`;
  }
  const date = text.slice(0, 10);
  return /^\d{4}-\d{2}-\d{2}$/.test(date) ? date : null;
};

const buildRegimeMarkAreas = (run: HistoricalRegimeRun | undefined, dates: string[]): RegimeMarkArea[] => {
  if (!run || dates.length === 0) {
    return [];
  }

  const datedCategories = dates
    .map((date) => ({ date, key: normalizeDateKey(date) }))
    .filter((item): item is { date: string; key: string } => item.key !== null);
  const statesById = new Map(run.states.map((state) => [state.id, state]));

  return run.segments.flatMap((segment): RegimeMarkArea[] => {
    const state = statesById.get(segment.state_id);
    const startDate = normalizeDateKey(segment.start_date);
    const endDate = normalizeDateKey(segment.end_date);
    if (!state?.label || !state.color || !startDate || !endDate || startDate > endDate) {
      return [];
    }

    const intersectingDates = datedCategories.filter(({ key }) => key >= startDate && key <= endDate);
    if (intersectingDates.length === 0) {
      return [];
    }

    return [[
      {
        name: state.label,
        xAxis: intersectingDates[0].date,
        itemStyle: { color: state.color, opacity: 0.12 },
      },
      { xAxis: intersectingDates[intersectingDates.length - 1].date },
    ]];
  });
};

const alignedChannelValues = (
  result: TimeSeriesIndicatorResult | undefined,
  channelId: string,
  dates: string[],
): Array<number | null> => {
  if (!result) return dates.map(() => null);
  const channel = result.channels.find((item) => item.id === channelId);
  if (!channel) return dates.map(() => null);
  const valuesByDate = new Map(
    result.dates.map((date, index) => [normalizeDateKey(date) ?? date, channel.values[index] ?? null]),
  );
  return dates.map((date) => valuesByDate.get(normalizeDateKey(date) ?? date) ?? null);
};

function MetricCard({ title, value, description }: { title: string; value: string; description?: string }) {
  return (
    <div className="rounded-2xl border border-transparent bg-gradient-to-br from-white via-slate-50 to-emerald-50 p-5 shadow-sm">
      <div className="text-xs font-semibold uppercase tracking-wide text-emerald-500">{title}</div>
      <div className="mt-2 text-2xl font-bold text-slate-900">{value}</div>
      {description && <div className="mt-1 text-xs text-slate-500">{description}</div>}
    </div>
  );
}

/**
 * What actually happened after the研究日, in one strip.
 *
 * Deliberately not another `MetricCard` row: the simulated quantiles above are
 * scenarios and these four numbers are facts, and a reader who cannot tell them
 * apart at a glance will quote a quantile as an outcome. One block, its own
 * frame, and the hindsight caveat attached to it rather than buried in a
 * footnote.
 */
/**
 * The fitted numbers behind whichever model is on screen.
 *
 * Three shapes, not five: the normal and four-moment lanes differ only in what
 * they matched, and the two filtered lanes differ only in where their
 * coefficients came from.
 */
function simulationAssumptionNote(simulation: FuturePathSimulation): string {
  const { assumptions: a, method } = simulation;
  if (method === 'block_bootstrap') {
    return ` 名义平均区块长度 ${a.averageBlockLength} 个收益观察值；遇缺口或区间末尾重新抽样，实际区块长度可能更短。`;
  }
  if (method === 'fhs_ewma' || method === 'fhs_garch') {
    const conditional = formatRatioPercent(a.conditionalVolatilityStart);
    const longRun = formatRatioPercent(a.dailyLogVolatility);
    const coefficients = method === 'fhs_ewma'
      // The panel above already says EWMA has no mean reversion; repeating it
      // here just made the same sentence appear twice on one screen.
      ? `λ = ${formatDecimal(a.ewmaLambda, 2)}（α = ${formatDecimal(a.garchAlpha, 3)}、β = ${formatDecimal(a.garchBeta, 3)}、ω = 0）`
      : `准极大似然拟合 α = ${formatDecimal(a.garchAlpha, 3)}、β = ${formatDecimal(a.garchBeta, 3)}、ω = ${(a.garchOmega ?? 0).toExponential(2)}，持续性 ${formatDecimal(a.volatilityPersistence, 3)}`;
    return ` 起始条件日波动 ${conditional}，长期日波动 ${longRun}${
      a.conditionalVolatilityStart !== null && a.dailyLogVolatility !== null
        ? `（当前是长期的 ${formatDecimal(a.conditionalVolatilityStart / a.dailyLogVolatility, 2)} 倍）`
        : ''
    }；${coefficients}。滤波后残差偏度 ${formatDecimal(a.residualSkewness, 2)}、超额峰度 ${formatDecimal(a.residualExcessKurtosis, 2)}——这些残差就是路径抽样的来源，峰度仍高说明肥尾不是波动聚集单独造成的。`;
  }
  const base = ` 日均对数收益 ${formatRatioPercent(a.meanDailyLogReturn)}，日波动 ${formatRatioPercent(a.dailyLogVolatility)}；历史对数收益偏度 ${formatDecimal(a.historicalLogSkewness, 2)}、超额峰度 ${formatDecimal(a.historicalLogExcessKurtosis, 2)}`;
  if (method === 'gaussian') {
    return `${base}——本模型不使用这两个形状参数，只用均值与波动率。`;
  }
  return `${base}，拟合值分别为 ${formatDecimal(a.fittedLogSkewness, 2)}、${formatDecimal(a.fittedLogExcessKurtosis, 2)}（${
    a.shapeCalibrationStatus === 'matched' ? '四矩校准已匹配'
      : a.shapeCalibrationStatus === 'approximate' ? '四矩近似校准' : '正态安全降级'
  }）。`;
}

function RealizedFutureStrip({
  realized,
  score,
  simulatedAverageMaxDrawdown,
}: {
  realized: RealizedFuturePath;
  score: RealizedMethodScore;
  /** The model's own average path drawdown, so 42% vs 22% reads as one figure. */
  simulatedAverageMaxDrawdown: number;
}) {
  const coverage = realized.complete
    ? `完整覆盖 ${realized.coveredDays}/${realized.requestedDays} 个交易日`
    : `仅覆盖 ${realized.coveredDays}/${realized.requestedDays} 个交易日`;
  const gapToMedian =
    realized.terminalNav !== null && score.simulatedP50 !== null
      ? realized.terminalNav / score.simulatedP50 - 1
      : null;
  return (
    <section
      data-testid="realized-future-strip"
      aria-label="研究日之后的实际走势回看"
      className="mt-4 overflow-hidden rounded-2xl border border-slate-900/15 bg-slate-50"
    >
      <div className="flex flex-col gap-1 border-b border-slate-200 px-4 py-3 sm:flex-row sm:items-baseline sm:justify-between">
        <h4 className="text-sm font-semibold text-slate-900">
          研究日之后实际发生了什么
          <span className="ml-2 text-xs font-normal text-slate-500">
            {realized.baseDate} → {realized.endDate}
          </span>
        </h4>
        <span
          className={`w-fit rounded-full px-3 py-1 text-xs font-semibold ${realized.complete ? 'bg-slate-900 text-white' : 'bg-amber-100 text-amber-800'}`}
        >
          {coverage}
        </span>
      </div>
      <dl className="grid gap-x-4 gap-y-3 px-4 py-3 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { label: '实际期末净值', value: formatDecimal(realized.terminalNav, 4), note: `基准日净值 ${formatDecimal(realized.baseNav, 2)}` },
          { label: '实际期末收益', value: formatRatioPercent(realized.terminalReturn), note: gapToMedian === null ? '—' : `相对模拟中位数 ${formatRatioPercent(gapToMedian)}` },
          {
            label: '落在模拟分布',
            value: score.percentileRank === null ? score.bandLabel ?? '--' : `${formatRatioPercent(score.percentileRank)} 分位`,
            note: score.percentileRank === null ? '区间未走完，只给所处区间' : score.bandLabel ?? '',
          },
          {
            label: '实际期间最大回撤',
            value: formatRatioPercent(realized.maxDrawdown),
            note: `模拟路径平均 ${formatRatioPercent(simulatedAverageMaxDrawdown)}`,
          },
        ].map((item) => (
          <div key={item.label}>
            <dt className="text-xs font-semibold uppercase tracking-wide text-slate-500">{item.label}</dt>
            <dd className="mt-1 text-2xl font-bold tabular-nums text-slate-900">{item.value}</dd>
            <dd className="mt-0.5 text-xs text-slate-500">{item.note}</dd>
          </div>
        ))}
      </dl>
      <div className="space-y-1 border-t border-slate-200 px-4 py-3 text-xs leading-5 text-slate-600">
        <p>
          {score.methodLabel}当时给出的第 {realized.coveredDays} 个交易日中位数为{' '}
          {formatDecimal(score.simulatedP50, 4)}，5%—95% 区间{' '}
          {formatDecimal(score.simulatedP05, 4)} — {formatDecimal(score.simulatedP95, 4)}。{score.verdict}
        </p>
        <p>
          实际路径有 {realized.observationDays - score.breachDays} / {realized.observationDays} 个已披露交易日留在 5%—95% 区间内
          （占比 {formatRatioPercent(score.containmentRatio)}，其中 {formatRatioPercent(score.aboveMedianRatio)} 的交易日位于中位数之上）。
          {score.worstBreachDay === null
            ? ' 期间没有走出该区间。'
            : ` 偏离最远出现在第 ${score.worstBreachDay} 个交易日（${realized.dates[score.worstBreachDay] ?? '日期缺失'}），越出区间 ${formatDecimal(Math.abs(score.worstBreachGap ?? 0), 4)} 个净值单位（${(score.worstBreachGap ?? 0) < 0 ? '向下' : '向上'}）。`}
          {' '}单条路径的区间内天数波动很大，只作描述用，模型好坏看期末分位。
        </p>
        {realized.observationDays < realized.coveredDays && (
          <p className="text-amber-700">
            其中 {realized.coveredDays - realized.observationDays} 个交易日没有披露净值，未纳入区间与回撤统计。
          </p>
        )}
        <p className="font-medium text-slate-700">
          这条走势在研究日 {realized.asOf} 当天不可得，只用于事后检验模型，不参与任何模拟输入。
        </p>
      </div>
    </section>
  );
}

function DistributionMetricCard({
  title,
  value,
  interpretation,
}: {
  title: string;
  value: string;
  interpretation: DistributionInterpretation;
}) {
  return (
    <div className="rounded-2xl border border-transparent bg-gradient-to-br from-white via-slate-50 to-emerald-50 p-5 shadow-sm">
      <div className="text-xs font-semibold uppercase tracking-wide text-emerald-500">{title}</div>
      <div className="mt-2 text-2xl font-bold text-slate-900">{value}</div>
      <div className="mt-3 inline-flex rounded-full bg-amber-50 px-2.5 py-1 text-xs font-semibold text-amber-700">
        {interpretation.label}
      </div>
      <p className="mt-2 text-xs leading-5 text-slate-600">{interpretation.meaning}</p>
    </div>
  );
}

function ExtremesCard({ best, worst }: { best: string; worst: string }) {
  const bestClass = best === '--' ? 'text-slate-400' : 'text-emerald-600';
  const worstClass = worst === '--' ? 'text-slate-400' : 'text-rose-500';
  return (
    <div className="rounded-2xl border border-transparent bg-gradient-to-br from-white via-slate-50 to-amber-50 p-5 shadow-sm">
      <div className="text-xs font-semibold uppercase tracking-wide text-emerald-500">单日极值</div>
      <div className="mt-4 grid grid-cols-2 gap-4">
        <div>
          <div className="text-xs text-slate-500">最佳日</div>
          <div className={`mt-1 text-xl font-bold ${bestClass}`}>{best}</div>
        </div>
        <div>
          <div className="text-xs text-slate-500">最差日</div>
          <div className={`mt-1 text-xl font-bold ${worstClass}`}>{worst}</div>
        </div>
      </div>
      <p className="mt-3 text-xs text-slate-500">观察收益极值，评估潜在的尾部风险。</p>
    </div>
  );
}

export default function ProductDetail() {
  const params = useParams<{ productId?: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const location = useLocation();
  const returnNavigation = readReturnNavigationState(location.state);
  const productKind = searchParams.get('kind') === 'fund' ? 'fund' : 'etf';
  const [detail, setDetail] = useState<ProductDetailResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<ProductAnalysisResponse | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [overlayResults, setOverlayResults] = useState<TimeSeriesIndicatorResult[]>([]);
  const [overlayLoading, setOverlayLoading] = useState(false);
  const [overlayError, setOverlayError] = useState<string | null>(null);
  const [selectedOverlays, setSelectedOverlays] = useState<OverlayId[]>(['PRICE_MA', 'VOLUME_MA']);
  const [histogramBinWidth, setHistogramBinWidth] = useState<number>(0.2);
  const [researchIndicators, setResearchIndicators] = useState<IndicatorDefinition[]>([]);
  const [seriesIndicators, setSeriesIndicators] = useState<IndicatorDefinition[]>([]);
  const [researchPeriods, setResearchPeriods] = useState<string[]>(['1Y']);
  const [researchResults, setResearchResults] = useState<EvaluationResult[]>([]);
  const [researchAsOf, setResearchAsOf] = useState('');
  const platformAsOf = useResearchDay();
  const [researchLoading, setResearchLoading] = useState(false);
  const [researchError, setResearchError] = useState<string | null>(null);
  const [definitionIndicator, setDefinitionIndicator] = useState<IndicatorDefinition | null>(null);
  const [statisticsPeriod, setStatisticsPeriod] = useState<StatisticsPeriod>('ALL');
  const [simulationMethod, setSimulationMethod] = useState<SimulationMethod>('parametric');
  const [simulationHorizon, setSimulationHorizon] = useState(252);
  const [simulationPathCount, setSimulationPathCount] = useState(500);
  const [bootstrapBlockLength, setBootstrapBlockLength] = useState(20);
  const [fhsEwmaLambda, setFhsEwmaLambda] = useState(0.94);
  const [simulationTargetReturn, setSimulationTargetReturn] = useState(5);
  const [simulationRun, setSimulationRun] = useState(0);
  /**
   * Which day the right-hand distribution answers for — the zoom window's
   * right edge, `null` while the whole horizon is shown.
   *
   * The window itself lives in a ref, not in state: the chart re-renders with
   * `notMerge`, so making every drag frame a render would rebuild the slider
   * under the user's cursor. Only a change of day is worth a render, and even
   * that waits for the drag to settle.
   */
  const [densityDay, setDensityDay] = useState<number | null>(null);
  const simulationZoomWindow = useRef<[number, number]>([0, 100]);
  const densityDayTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [activeTab, setActiveTab] = useState<'chart' | 'regime' | 'statistics' | 'simulation' | 'risk'>('chart');
  const [analysisBasis, setAnalysisBasis] = useState<'adjusted_nav' | 'price'>('adjusted_nav');
  const [analysisSettingsOpen, setAnalysisSettingsOpen] = useState(false);
  const [selectedStateId, setSelectedStateId] = useState('');
  const [selectedSegmentId, setSelectedSegmentId] = useState('');
  const [simulationRecord, setSimulationRecord] = useState<{ key: string; loading?: boolean; result?: ProductAnalysisResponse; error?: string } | null>(null);
  const simulationController = useRef<AbortController | null>(null);
  const simulationGeneration = useRef(0);
  const [regimeRefresh, setRegimeRefresh] = useState(0);
  const [historicalRegimeRuns, setHistoricalRegimeRuns] = useState<HistoricalRegimeRun[]>([]);
  const [historicalRegimeLoading, setHistoricalRegimeLoading] = useState(true);
  const [historicalRegimeError, setHistoricalRegimeError] = useState<string | null>(null);
  const [selectedHistoricalRegimeRunId, setSelectedHistoricalRegimeRunId] = useState('');
  const [selectedHistoricalRegimeRunDetail, setSelectedHistoricalRegimeRunDetail] = useState<HistoricalRegimeRun | null>(null);
  const [historicalRegimeDetailLoading, setHistoricalRegimeDetailLoading] = useState(false);
  const [historicalRegimeDetailError, setHistoricalRegimeDetailError] = useState<string | null>(null);
  const [researchPreference, setResearchPreference] = useMetricDisplayPreference(
    'product-detail',
    'single_product',
    CORE_RESEARCH_INDICATOR_IDS,
    '1Y',
    researchIndicators.map((indicator) => indicator.id),
  );

  const productId = useMemo(() => {
    if (!params.productId) {
      return '';
    }
    try {
      return decodeURIComponent(params.productId);
    } catch (err) {
      console.warn('Failed to decode productId from route', err);
      return params.productId;
    }
  }, [params.productId]);

  const selectedResearchIndicators = useMemo(
    () => researchPreference.indicatorIds
      .map((id) => researchIndicators.find((item) => item.id === id))
      .filter((item): item is IndicatorDefinition => Boolean(item)),
    [researchIndicators, researchPreference.indicatorIds],
  );

  const toggleOverlay = (overlayId: OverlayId) => {
    setSelectedOverlays((prev) => {
      if (prev.includes(overlayId)) {
        return prev.filter((item) => item !== overlayId);
      }
      return [...prev, overlayId];
    });
  };

  const restoreDefaultOverlays = () => {
    setSelectedOverlays(['PRICE_MA', 'VOLUME_MA']);
  };

  const renderOverlayControls = (optionId: OverlayId) => {
    const definition = overlayOptions.find((item) => item.id === optionId);
    return (
      <div className="rounded-xl border border-sky-100 bg-sky-50 px-3 py-2 text-xs leading-5 text-sky-800">
        <p><span className="font-semibold">固定指标版本：</span>{definition?.description}</p>
        <p className="mt-1">窗口、倍数和平滑周期不能在展示页面临时覆盖。需要其他参数时，请在指标中心复制并保存为另一个时序指标。</p>
        <Link to="/settings/indicators-models" className="mt-2 inline-flex font-semibold text-violet-700 underline decoration-violet-300 underline-offset-4">前往指标中心</Link>
      </div>
    );
  };

  useEffect(() => {
    if (!productId) {
      setError('未指定产品标识');
      setDetail(null);
      return;
    }
    const controller = new AbortController();
    const fetchDetail = async () => {
      try {
        setLoading(true);
        setError(null);
        const detailUrl = `/api/instruments/products/${encodeURIComponent(productId)}?kind=${productKind}`;
        const resp = await fetch(detailUrl, { signal: controller.signal });
        if (!resp.ok) {
          console.warn('Product detail request responded with non-OK status', resp.status);
          if (resp.status === 404) {
            setError('未找到对应的产品，请检查产品标识。');
          } else {
            setError('产品详情加载失败，请稍后重试。');
          }
          setDetail(null);
          return;
        }
        const data = (await resp.json()) as ProductDetailResponse;
        if (!Array.isArray(data.timeseries)) data.timeseries = [];
        setDetail(data);
      } catch (err) {
        if ((err as DOMException).name === 'AbortError') {
          return;
        }
        console.error('Failed to load product detail', err);
        setError('产品详情加载失败，请稍后重试。');
        setDetail(null);
      } finally {
        setLoading(false);
      }
    };
    fetchDetail();
    return () => controller.abort();
  }, [productId, productKind]);

  useEffect(() => {
    let active = true;
    setHistoricalRegimeLoading(true);
    setHistoricalRegimeError(null);
    listHistoricalRegimeRuns()
      .then((runs) => {
        if (active) {
          setHistoricalRegimeRuns(runs);
        }
      })
      .catch(() => {
        if (active) {
          setHistoricalRegimeRuns([]);
          setHistoricalRegimeError('历史情景版本加载失败，暂时无法叠加背景。');
        }
      })
      .finally(() => {
        if (active) {
          setHistoricalRegimeLoading(false);
        }
      });
    return () => { active = false; };
  }, [regimeRefresh]);

  useEffect(() => {
    const summary = historicalRegimeRuns.find((run) => run.id === selectedHistoricalRegimeRunId);
    if (!summary) {
      setSelectedHistoricalRegimeRunDetail(null);
      setHistoricalRegimeDetailLoading(false);
      setHistoricalRegimeDetailError(null);
      return undefined;
    }
    let active = true;
    setSelectedHistoricalRegimeRunDetail(null);
    setHistoricalRegimeDetailLoading(true);
    setHistoricalRegimeDetailError(null);
    getHistoricalRegimeRun(summary.id)
      .then((detailRun) => { if (active) setSelectedHistoricalRegimeRunDetail(detailRun); })
      .catch(() => {
        if (active) setHistoricalRegimeDetailError('所选历史情景完整结果加载失败，未向产品分析提交缺失区间。');
      })
      .finally(() => { if (active) setHistoricalRegimeDetailLoading(false); });
    return () => { active = false; };
  }, [historicalRegimeRuns, selectedHistoricalRegimeRunId]);

  useEffect(() => {
    let active = true;
    Promise.all([listCustomIndicators({ contextKind: 'single_product', productKind }), getCustomIndicatorMeta()])
      .then(([{ items }, metadata]) => {
        if (!active) return;
        const singleProductIndicators = indicatorsForContext(items, 'single_product');
        setResearchIndicators(singleProductIndicators);
        setSeriesIndicators(items.filter(item => item.result_kind === 'time_series'));
        const runtimePeriods = metadata.periods.map((item) => item.value);
        setResearchPeriods(runtimePeriods);
        setResearchPreference((current) => normalizeMetricPeriods(current, runtimePeriods, '1Y'));
      })
      .catch(() => {
        if (active) setResearchError('自定义指标库暂时不可用，请稍后重试。');
      });
    return () => { active = false; };
  }, [productKind]);

  useEffect(() => {
    if (selectedResearchIndicators.length === 0 || !productId) {
      setResearchResults([]);
      return;
    }
    let active = true;
    setResearchLoading(true);
    setResearchError(null);
    const selectedPreference = {
      ...researchPreference,
      indicatorIds: selectedResearchIndicators.map((indicator) => indicator.id),
    };
    Promise.all(groupIndicatorsByPeriod(selectedPreference, '1Y').map(({ indicatorIds, period }) => (
      evaluateCustomIndicators({
        indicator_ids: indicatorIds,
        targets: [{ kind: productKind, product_id: productId }],
        period,
        as_of: researchAsOf || undefined,
      })
    )))
      .then((responses) => {
        if (active) setResearchResults(responses.flatMap(({ results }) => results));
      })
      .catch(() => {
        if (active) {
          setResearchResults([]);
          setResearchError('该指标当前无法计算，请检查真实净值数据和样本窗口。');
        }
      })
      .finally(() => { if (active) setResearchLoading(false); });
    return () => { active = false; };
  }, [productId, productKind, researchAsOf, researchPreference.periodsByIndicator, selectedResearchIndicators]);

  const metrics = detail?.metrics ?? {};
  const baseInfo = detail?.base_info ?? {};
  const tsCode = baseInfo['ts_code'];
  const inceptionDateLabel = productKind === 'etf' ? '上市日期' : '成立日期';
  const inceptionDate = productKind === 'etf' ? baseInfo['list_date'] : baseInfo['found_date'];
  const endDate = productKind === 'etf'
    ? baseInfo['delist_date']
    : (baseInfo['due_date'] ?? baseInfo['delist_date']);
  const endDateLabel = productKind === 'etf'
    ? '退市日期'
    : (baseInfo['due_date'] ? '到期日期' : '终止日期');
  const inceptionDateText = formatDate(inceptionDate);
  const endDateText = formatDate(endDate);
  const currentSizeDescription = metrics.current_size_as_of
    ? `快照截至 ${formatDate(metrics.current_size_as_of)} · ${decimalFormatter.format(metrics.current_share ?? 0)} 万份 × ${decimalFormatter.format(metrics.current_unit_nav ?? 0)} 元/份`
    : productKind === 'fund'
      ? 'Tushare 暂无场外基金份额数据'
      : '暂无可匹配的份额与单位净值';

  const productResearchRegimeRuns = useMemo(
    () => researchVersionChoices(historicalRegimeRuns, selectedHistoricalRegimeRunId),
    [historicalRegimeRuns, selectedHistoricalRegimeRunId],
  );
  const selectedHistoricalRegimeRunSummary = useMemo(
    () => productResearchRegimeRuns.find((run) => run.id === selectedHistoricalRegimeRunId),
    [productResearchRegimeRuns, selectedHistoricalRegimeRunId],
  );
  const selectedHistoricalRegimeRun = selectedHistoricalRegimeRunDetail?.id === selectedHistoricalRegimeRunId
    ? selectedHistoricalRegimeRunDetail
    : undefined;
  const selectedHistoricalRegimePublication = useMemo(
    () => selectedHistoricalRegimeRunSummary
      ? latestProductResearchPublication(selectedHistoricalRegimeRunSummary)
      : undefined,
    [selectedHistoricalRegimeRunSummary],
  );
  const selectedSegment = analysis?.regimeAnalysis?.segments.find(item => item.id === selectedSegmentId);
  const selectedState = selectedHistoricalRegimeRun?.states.find(item => item.id === selectedStateId);
  const historicalRegimeMarkAreas = useMemo(() => {
    const windowStart = analysis?.researchContext?.windowStartDate ?? analysis?.researchContext?.startDate;
    const windowEnd = analysis?.researchContext?.windowEndDate ?? analysis?.researchContext?.endDate;
    const dates = (detail?.timeseries ?? []).filter(item => (!windowStart || item.date >= windowStart) && (!windowEnd || item.date <= windowEnd)).map(item => item.date);
    const run = selectedHistoricalRegimeRun;
    if (!run) return [];
    return buildRegimeMarkAreas({ ...run, segments: run.segments.filter(segment => (
      (!selectedStateId || segment.state_id === selectedStateId)
      && (!selectedSegment || (segment.start_date <= selectedSegment.endDate && segment.end_date >= selectedSegment.startDate))
    )) }, dates);
  }, [detail?.timeseries, selectedHistoricalRegimeRun, selectedStateId, selectedSegment, analysis?.researchContext]);
  const selectState = (id: string) => { setSelectedStateId(id); setSelectedSegmentId(''); };
  const selectSegment = (segment: ProductRegimeSegment) => { setSelectedStateId(segment.stateId); setSelectedSegmentId(segment.id); };
  const changeRegime = (id: string) => { setSelectedHistoricalRegimeRunId(id); setSelectedStateId(''); setSelectedSegmentId(''); };
  useEffect(() => { changeRegime(''); setActiveTab('chart'); setAnalysisBasis('adjusted_nav'); setAnalysisSettingsOpen(false); }, [productId, productKind]);
  useEffect(() => {
    if (!productId || productKind !== 'etf' || selectedOverlays.length === 0) {
      setOverlayResults([]);
      setOverlayLoading(false);
      setOverlayError(productKind === 'fund' ? '场外公募基金没有交易所 OHLCV，技术时序指标不可用。' : null);
      return undefined;
    }
    let active = true;
    const indicatorInstances = selectedOverlays.map((overlayId) => ({
      indicator_id: SERIES_INDICATOR_IDS[overlayId],
    }));
    setOverlayLoading(true);
    setOverlayError(null);
    setOverlayResults([]);
    evaluateTimeSeriesIndicators({
      indicator_instances: indicatorInstances,
      target: { kind: 'etf', product_id: productId },
      period: 'ALL',
      max_points: 5000,
    })
      .then((response) => {
        if (!active) return;
        setOverlayResults(response.results);
        const failed = response.results.filter((item) => item.status === 'unavailable' || item.status === 'error');
        if (failed.length > 0) {
          setOverlayError(failed.flatMap((item) => item.warnings.map((warning) => warning.message)).join('；'));
        }
      })
      .catch((requestError) => {
        if (!active) return;
        setOverlayResults([]);
        setOverlayError(requestError instanceof Error ? requestError.message : '技术时序指标计算失败。');
      })
      .finally(() => {
        if (active) setOverlayLoading(false);
      });
    return () => { active = false; };
  }, [productId, productKind, selectedOverlays]);

  const analysisRequest = useMemo<ProductAnalysisRequest>(() => ({
    statistics_period: statisticsPeriod,
    analysis_basis: analysisBasis,
    include_technical: false,
    include_simulation: false,
    price_ma_periods: [], volume_ma_periods: [],
    boll_period: 20, boll_multiplier: 2, kdj_period: 9, kdj_k_smoothing: 3, kdj_d_smoothing: 3,
    histogram_bin_width: histogramBinWidth,
    simulation_horizon: 252, simulation_path_count: 500,
    bootstrap_block_length: 20, fhs_ewma_lambda: 0.94,
    simulation_target_return: 5, simulation_run: 0,
    regime: selectedHistoricalRegimeRunSummary && selectedHistoricalRegimePublication ? {
      run_id: selectedHistoricalRegimeRunSummary.id,
      publication_id: selectedHistoricalRegimePublication.id,
      ...(selectedStateId ? { state_id: selectedStateId } : {}),
      ...(selectedSegmentId ? { segment_id: selectedSegmentId } : {}),
    } : null,
  }), [statisticsPeriod, analysisBasis, histogramBinWidth, selectedHistoricalRegimeRunSummary, selectedHistoricalRegimePublication, selectedStateId, selectedSegmentId]);
  const simulationKey = JSON.stringify([productId, productKind, analysisRequest, simulationHorizon, simulationPathCount, bootstrapBlockLength, fhsEwmaLambda, simulationTargetReturn]);
  const currentSimulation = simulationRecord?.key === simulationKey ? simulationRecord : null;
  const simulationAnalysis = currentSimulation?.result;
  useEffect(() => {
    simulationGeneration.current += 1;
    simulationController.current?.abort();
    setSimulationRecord(null);
    // A new run means a new day axis, so the old zoom window and the day the
    // right-hand panel was showing no longer refer to anything.
    simulationZoomWindow.current = [0, 100];
    setDensityDay(null);
    return () => { simulationController.current?.abort(); };
  }, [simulationKey]);
  useEffect(() => () => {
    if (densityDayTimer.current) clearTimeout(densityDayTimer.current);
  }, []);
  /**
   * The zoom window's right edge picks the day the distribution is drawn for.
   * Debounced rather than immediate: ECharts fires this continuously while the
   * slider is dragged, and each render rebuilds the chart.
   */
  const handleSimulationZoom = (params: unknown) => {
    const payload = params as { batch?: Array<{ start?: number; end?: number }>; start?: number; end?: number };
    const range = payload?.batch?.[0] ?? payload;
    const end = Number(range?.end);
    if (!Number.isFinite(end)) return;
    const start = Number(range?.start);
    simulationZoomWindow.current = [Number.isFinite(start) ? start : 0, end];
    const day = Math.round((end / 100) * simulationHorizon);
    if (densityDayTimer.current) clearTimeout(densityDayTimer.current);
    densityDayTimer.current = setTimeout(() => setDensityDay(day), 180);
  };
  const runSimulation = async () => {
    simulationController.current?.abort();
    const controller = new AbortController();
    simulationController.current = controller;
    const generation = ++simulationGeneration.current;
    const nextRun = simulationRun + 1;
    setSimulationRun(nextRun);
    setSimulationRecord({ key: simulationKey, loading: true });
    try {
      const result = await analyzeProduct(productId, productKind, {
        ...analysisRequest, include_simulation: true,
        simulation_horizon: simulationHorizon, simulation_path_count: simulationPathCount,
        bootstrap_block_length: bootstrapBlockLength, fhs_ewma_lambda: fhsEwmaLambda,
        simulation_target_return: simulationTargetReturn,
        simulation_run: nextRun,
      }, controller.signal);
      if (!controller.signal.aborted && generation === simulationGeneration.current) setSimulationRecord({ key: simulationKey, result });
    } catch (failure) {
      if (!controller.signal.aborted && generation === simulationGeneration.current) setSimulationRecord({ key: simulationKey, error: failure instanceof Error ? failure.message : '模拟未完成，请重试。' });
    }
  };
  useEffect(() => {
    if (!detail || !productId) { setAnalysis(null); setAnalysisLoading(false); return; }
    let active = true;
    const controller = new AbortController();
    setAnalysisLoading(true); setAnalysisError(null); setAnalysis(null);
    analyzeProduct(productId, productKind, analysisRequest, controller.signal)
      .then(payload => { if (active) setAnalysis(payload); })
      .catch(failure => { if (active && !controller.signal.aborted) setAnalysisError(failure instanceof Error ? failure.message : '产品分析失败，请稍后重试。'); })
      .finally(() => { if (active) setAnalysisLoading(false); });
    return () => { active = false; controller.abort(); };
  }, [detail, productId, productKind, analysisRequest]);
  const rawTechnicalAvailability = useMemo(() => {
    const points = detail?.timeseries ?? [];
    return {
      ohlc: points.length > 0 && points.every((item) => (
        item.open !== null
        && item.high !== null
        && item.low !== null
        && Number.isFinite(item.open)
        && Number.isFinite(item.high)
        && Number.isFinite(item.low)
        && Number.isFinite(item.close)
      )),
      volume: points.some((item) => item.volume !== null && Number.isFinite(item.volume)),
    };
  }, [detail?.timeseries]);

  const chartOption = useMemo(() => {
    if (!detail?.timeseries || detail.timeseries.length === 0) {
      return undefined;
    }

    const dates = detail.timeseries.map((item) => item.date);
    const findOverlayResult = (indicatorId: string) => overlayResults.find((item) => (
      item.indicator_id === indicatorId
    ));
    const { ohlc: hasOhlc, volume: hasVolume } = rawTechnicalAvailability;
    const totalPoints = dates.length;
    const defaultWindow = 252;
    const startIndex = Math.max(0, totalPoints - defaultWindow);
    const requestedStart = selectedSegment?.startDate ?? analysis?.researchContext?.windowStartDate;
    const requestedEnd = selectedSegment?.endDate ?? analysis?.researchContext?.windowEndDate;
    const startValue = requestedStart ? dates.find(date => date >= requestedStart) ?? dates[startIndex] : dates[startIndex];
    const endValue = requestedEnd ? dates.filter(date => date <= requestedEnd).pop() ?? dates[totalPoints - 1] : dates[totalPoints - 1];
    const klineValues = detail.timeseries.map((item) => [item.open, item.close, item.low, item.high]);
    const volumes = detail.timeseries.map((item) => ({
      value: item.volume,
      itemStyle: {
        color: item.open !== null && item.close >= item.open ? '#34d399' : '#94a3b8',
      },
    }));
    const priceMASeries = selectedOverlays.includes('PRICE_MA')
      ? [{
          name: '20 日收盘价均线',
          type: 'line',
          data: alignedChannelValues(
            findOverlayResult(SERIES_INDICATOR_IDS.PRICE_MA),
            'ma',
            dates,
          ),
          smooth: true,
          showSymbol: false,
          lineStyle: { width: 1.5 },
          emphasis: { focus: 'series' },
        }]
      : [];

    const volumeMASeries = selectedOverlays.includes('VOLUME_MA') && hasVolume
      ? [{
          name: '10 日成交量均线',
          type: 'line',
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: alignedChannelValues(
            findOverlayResult(SERIES_INDICATOR_IDS.VOLUME_MA),
            'volume_ma',
            dates,
          ),
          smooth: true,
          showSymbol: false,
          lineStyle: { width: 1 },
          emphasis: { focus: 'series' },
        }]
      : [];

    const bollingerResult = findOverlayResult(SERIES_INDICATOR_IDS.BOLL);
    const bollingerSeries = selectedOverlays.includes('BOLL')
      ? [
            {
              name: '布林上轨（20 日，2σ）',
              type: 'line',
              data: alignedChannelValues(bollingerResult, 'upper', dates),
              smooth: true,
              showSymbol: false,
              lineStyle: { width: 1, color: '#f97316' },
            },
            {
              name: '布林中轨',
              type: 'line',
              data: alignedChannelValues(bollingerResult, 'middle', dates),
              smooth: true,
              showSymbol: false,
              lineStyle: { width: 1, color: '#0ea5e9', type: 'dashed' },
            },
            {
              name: '布林下轨',
              type: 'line',
              data: alignedChannelValues(bollingerResult, 'lower', dates),
              smooth: true,
              showSymbol: false,
              lineStyle: { width: 1, color: '#10b981' },
            },
          ]
      : [];

    const kdjResult = findOverlayResult(SERIES_INDICATOR_IDS.KDJ);
    const hasKDJ = selectedOverlays.includes('KDJ')
      && Boolean(kdjResult)
      && kdjResult?.status !== 'unavailable'
      && kdjResult?.status !== 'error';
    const kValues = hasKDJ ? alignedChannelValues(kdjResult, 'k', dates) : [];
    const dValues = hasKDJ ? alignedChannelValues(kdjResult, 'd', dates) : [];
    const jValues = hasKDJ ? alignedChannelValues(kdjResult, 'j', dates) : [];

    const primaryTop = 50;
    const klineHeight = 260;
    const volumeHeight = 120;
    const extraPanelHeight = 110;
    const gridGap = 20;

    const grid = [
      { left: 12, right: 18, top: primaryTop, height: klineHeight, containLabel: true },
      { left: 12, right: 18, top: primaryTop + klineHeight + gridGap, height: volumeHeight, containLabel: true },
    ];
    const xAxis: any[] = [
      {
        type: 'category',
        data: dates,
        boundaryGap: false,
        axisLine: { lineStyle: { color: '#cbd5f5' } },
        axisLabel: { color: '#475569', fontSize: 10, hideOverlap: true, showMinLabel: false, showMaxLabel: false },
      },
      {
        type: 'category',
        gridIndex: 1,
        data: dates,
        boundaryGap: false,
        axisTick: { show: false },
        axisLine: { lineStyle: { color: '#cbd5f5' } },
        axisLabel: { show: false },
      },
    ];
    const yAxis: any[] = [
      {
        scale: true,
        axisLine: { lineStyle: { color: '#cbd5f5' } },
        splitLine: { lineStyle: { color: '#e2e8f0' } },
        axisLabel: { color: '#475569', fontSize: 10, hideOverlap: true },
      },
      {
        gridIndex: 1,
        axisLine: { lineStyle: { color: '#cbd5f5' } },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: '#e2e8f0' } },
        axisLabel: { color: '#475569', fontSize: 10, hideOverlap: true },
      },
    ];

    if (hasKDJ) {
      grid.push({ left: 12, right: 18, top: primaryTop + klineHeight + gridGap + volumeHeight + gridGap, height: extraPanelHeight, containLabel: true });
      xAxis.push({
        type: 'category',
        gridIndex: 2,
        data: dates,
        boundaryGap: false,
        axisTick: { show: false },
        axisLine: { lineStyle: { color: '#cbd5f5' } },
        axisLabel: { color: '#475569', fontSize: 10, hideOverlap: true },
      });
      yAxis.push({
        gridIndex: 2,
        axisLine: { lineStyle: { color: '#cbd5f5' } },
        splitLine: { lineStyle: { color: '#e2e8f0' } },
        axisLabel: { color: '#475569', fontSize: 10, hideOverlap: true },
      });
    }

    const dataZoom: any[] = [
      {
        type: 'inside',
        xAxisIndex: hasKDJ ? [0, 1, 2] : [0, 1],
        startValue,
        endValue,
      },
      {
        show: true,
        xAxisIndex: hasKDJ ? [0, 1, 2] : [0, 1],
        type: 'slider',
        height: 18,
        bottom: hasKDJ ? 50 : 40,
        startValue,
        endValue,
      },
    ];

    const historicalBackground = historicalRegimeMarkAreas.length > 0 ? {
          markArea: {
            silent: true,
            label: {
              show: true,
              position: 'insideTop',
              color: '#334155',
              fontSize: 10,
            },
            data: historicalRegimeMarkAreas,
          },
        } : {};
    const priceSeries = hasOhlc
      ? {
          name: '价格',
          type: 'candlestick',
          data: klineValues,
          itemStyle: {
            color: '#0ea5e9',
            color0: '#f87171',
            borderColor: '#0284c7',
            borderColor0: '#dc2626',
          },
          ...historicalBackground,
        }
      : {
          name: '价格',
          type: 'line',
          data: detail.timeseries.map((item) => item.close),
          showSymbol: false,
          lineStyle: { width: 1.6, color: '#0ea5e9' },
          ...historicalBackground,
        };
    const series: any[] = [
      priceSeries,
      {
        name: '成交量',
        type: 'bar',
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: hasVolume ? volumes : [],
        barWidth: '60%',
      },
      ...priceMASeries,
      ...volumeMASeries,
      ...bollingerSeries,
    ];

    if (hasKDJ) {
      series.push(
        {
          name: 'K值',
          type: 'line',
          xAxisIndex: 2,
          yAxisIndex: 2,
          data: kValues,
          smooth: true,
          showSymbol: false,
          lineStyle: { width: 1.2, color: '#34d399' },
        },
        {
          name: 'D值',
          type: 'line',
          xAxisIndex: 2,
          yAxisIndex: 2,
          data: dValues,
          smooth: true,
          showSymbol: false,
          lineStyle: { width: 1.2, color: '#3b82f6' },
        },
        {
          name: 'J值',
          type: 'line',
          xAxisIndex: 2,
          yAxisIndex: 2,
          data: jValues,
          smooth: true,
          showSymbol: false,
          lineStyle: { width: 1.2, color: '#f97316' },
        }
      );
    }

    return {
      backgroundColor: '#ffffff',
      animation: false,
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
          crossStyle: { color: '#94a3b8' },
        },
      },
      axisPointer: {
        link: [{ xAxisIndex: 'all' }],
      },
      legend: {
        type: 'scroll',
        top: 10,
        left: 'center',
        icon: 'roundRect',
        textStyle: { color: '#475569', fontSize: 12 },
      },
      grid,
      xAxis,
      yAxis,
      dataZoom,
      series,
    };
  }, [
    detail?.timeseries,
    historicalRegimeMarkAreas,
    analysis?.researchContext,
    selectedSegment,
    overlayResults,
    rawTechnicalAvailability,
    selectedOverlays,
  ]);

  const kdjOverlayAvailable = useMemo(() => overlayResults.some((item) => (
    item.indicator_id === SERIES_INDICATOR_IDS.KDJ
    && item.status !== 'unavailable'
    && item.status !== 'error'
  )), [overlayResults]);

  const chartHeight = useMemo(() => (
    selectedOverlays.includes('KDJ') && kdjOverlayAvailable ? 700 : 540
  ), [kdjOverlayAvailable, selectedOverlays]);

  const statisticsWindow = analysis?.window ?? {
    complete: false,
    requested_start_date: null,
    message: analysisLoading ? '正在使用 NJIT 计算产品分析结果…' : analysisError,
  };
  const dailyReturns = analysis?.dailyReturns ?? [];
  const returnStats = analysis?.returnStatistics ?? {
    mean: null,
    std: null,
    median: null,
    positiveRatio: null,
    best: null,
    worst: null,
    sampleSize: 0,
    skewness: null,
    kurtosis: null,
    jbStatistic: null,
    normalityPValue: null,
  };
  const histogramBins = analysis?.histogram ?? [];
  const hasNormalPdf = histogramBins.length > 0
    && histogramBins.every((bin) => bin.normalPdfCount !== null);
  const boxPlotData = analysis?.boxPlot ?? null;
  const normalQqData = analysis?.normalQq ?? null;
  const normalQqTableRows = normalQqData?.keyPoints ?? [];
  const skewnessInterpretation = analysis?.interpretation.skewness ?? {
    label: '样本不足',
    meaning: '正在等待后端 NJIT 分析结果。',
  };
  const kurtosisInterpretation = analysis?.interpretation.kurtosis ?? {
    label: '样本不足',
    meaning: '正在等待后端 NJIT 分析结果。',
  };
  const normalityConclusion = analysis?.interpretation.normality ?? '样本不足，无法进行检验';

  const returnLineOption = useMemo(() => {
    if (dailyReturns.length === 0) {
      return undefined;
    }
    return {
      backgroundColor: '#ffffff',
      tooltip: {
        trigger: 'axis',
        valueFormatter: (value: number | string) => `${Number(value).toFixed(2)}%`,
      },
      grid: { left: 12, right: 18, bottom: 60, top: 40, containLabel: true },
      xAxis: {
        type: 'category',
        data: dailyReturns.map((item) => item.date),
        boundaryGap: false,
        axisLabel: { color: '#475569' },
        axisLine: { lineStyle: { color: '#cbd5f5' } },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          color: '#475569',
          formatter: (value: number) => `${value.toFixed(1)}%`,
        },
        splitLine: { lineStyle: { color: '#e2e8f0' } },
      },
      dataZoom: [
        { type: 'inside', start: 60, end: 100 },
        { start: 60, end: 100 },
      ],
      series: [
        {
          name: '单期收益率',
          type: analysis?.researchContext?.scope === 'full' ? 'line' : 'bar',
          smooth: false,
          showSymbol: false,
          lineStyle: { width: 1.5, color: '#10b981' },
          areaStyle: { opacity: 0.08, color: '#34d399' },
          data: dailyReturns.map((item) => Number(item.return.toFixed(2))),
          markLine: {
            symbol: 'none',
            data: [
              {
                yAxis: 0,
                lineStyle: { type: 'dashed', color: '#94a3b8' },
                label: { show: false },
              },
            ],
          },
        },
      ],
    };
  }, [dailyReturns]);

  const histogramOption = useMemo(() => {
    if (histogramBins.length === 0) {
      return undefined;
    }
    const categories = histogramBins.map(
      (bin) => `${formatSignedPercent(bin.start, 1)} ~ ${formatSignedPercent(bin.end, 1)}`
    );
    return {
      backgroundColor: '#ffffff',
      legend: {
        data: ['出现次数', '正态拟合'],
        top: 10,
        left: 'center',
        textStyle: { color: '#475569', fontSize: 11 },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: any) => {
          const items = Array.isArray(params) ? params : [params];
          const first = items[0];
          if (!first || typeof first.dataIndex !== 'number') {
            return '';
          }
          const index = first.dataIndex;
          const bin = histogramBins[index];
          const startLabel = formatSignedPercent(bin.start, 2);
          const endLabel = formatSignedPercent(bin.end, 2);
          const lines = [`${startLabel} ~ ${endLabel}`];
          lines.push(`出现次数：${bin.count}`);
          if (hasNormalPdf) lines.push(`正态拟合：${formatDecimal(bin.normalPdfCount, 2)} 次`);
          lines.push(`频率：${formatDecimal(bin.frequency, 3)}`);
          return lines.join('<br/>');
        },
      },
      grid: { left: 12, right: 18, bottom: 70, top: 60, containLabel: true },
      xAxis: {
        type: 'category',
        data: categories,
        axisLabel: { color: '#475569', fontSize: 10, rotate: -35 },
        axisTick: { alignWithLabel: true },
        axisLine: { lineStyle: { color: '#cbd5f5' } },
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: '#475569' },
        splitLine: { lineStyle: { color: '#e2e8f0' } },
      },
      series: [
        {
          type: 'bar',
          name: '出现次数',
          data: histogramBins.map((bin) => bin.count),
          barMaxWidth: 28,
          itemStyle: {
            color: '#38bdf8',
            borderRadius: [6, 6, 0, 0],
          },
        },
        ...(hasNormalPdf
          ? [
              {
                type: 'line',
                name: '正态拟合',
                data: histogramBins.map((bin) => bin.normalPdfCount),
                smooth: true,
                symbol: 'none',
                lineStyle: { width: 2, color: '#f97316' },
                areaStyle: { opacity: 0 },
              },
            ]
          : []),
      ],
    };
  }, [hasNormalPdf, histogramBins]);

  const boxPlotOption = useMemo(() => {
    if (!boxPlotData) {
      return undefined;
    }
    return {
      backgroundColor: '#ffffff',
      tooltip: {
        trigger: 'item',
        formatter: () => {
          if (!boxPlotData) {
            return '';
          }
          const lines = [
            `下须：${formatSignedPercent(boxPlotData.whiskers.lower, 2)}`,
            `Q1：${formatSignedPercent(boxPlotData.quartiles.q1, 2)}`,
            `中位数：${formatSignedPercent(boxPlotData.quartiles.median, 2)}`,
            `Q3：${formatSignedPercent(boxPlotData.quartiles.q3, 2)}`,
            `上须：${formatSignedPercent(boxPlotData.whiskers.upper, 2)}`,
          ];
          return lines.join('<br/>');
        },
      },
      grid: { left: 20, right: 28, bottom: 56, top: 28, containLabel: true },
      xAxis: {
        type: 'value',
        splitNumber: 3,
        name: '单期收益率',
        nameLocation: 'middle',
        nameGap: 36,
        axisLabel: {
          color: '#475569',
          fontSize: 10,
          hideOverlap: true,
          formatter: (value: number) => `${value.toFixed(1)}%`,
        },
        axisLine: { lineStyle: { color: '#cbd5f5' } },
        splitLine: { lineStyle: { color: '#e2e8f0' } },
      },
      yAxis: {
        type: 'category',
        data: ['单期收益率'],
        axisLabel: { show: false },
        axisTick: { show: false },
        axisLine: { lineStyle: { color: '#cbd5f5' } },
      },
      series: [
        {
          name: '箱形图',
          type: 'boxplot',
          data: [[
            Number(boxPlotData.whiskers.lower.toFixed(2)),
            Number(boxPlotData.quartiles.q1.toFixed(2)),
            Number(boxPlotData.quartiles.median.toFixed(2)),
            Number(boxPlotData.quartiles.q3.toFixed(2)),
            Number(boxPlotData.whiskers.upper.toFixed(2)),
          ]],
          itemStyle: {
            color: '#bae6fd',
            borderColor: '#0ea5e9',
          },
          markLine: {
            symbol: 'none',
            data: [{ xAxis: 0 }],
            lineStyle: { type: 'dashed', color: '#94a3b8' },
            label: { show: false },
          },
        },
        ...(boxPlotData.outliers.length > 0
          ? [
              {
                name: '离群值',
                type: 'scatter',
                data: boxPlotData.outliers.map((value) => [Number(value.toFixed(2)), 0]),
                symbolSize: 8,
                itemStyle: { color: '#f97316' },
              },
            ]
          : []),
      ],
    };
  }, [boxPlotData]);

  const normalQqOption = useMemo(() => {
    if (!normalQqData) {
      return undefined;
    }
    const pointsForTail = (tail: 'lower' | 'center' | 'upper') => normalQqData.points
      .filter((point) => point.tail === tail)
      .map((point) => [
        Number(point.theoreticalQuantile.toFixed(4)),
        Number(point.observedReturn.toFixed(4)),
        point.percentile,
      ]);
    return {
      backgroundColor: '#ffffff',
      aria: {
        enabled: true,
        decal: { show: true },
        description: `正态 Q-Q 图，比较 ${normalQqData.sampleSize} 个实际单期收益率分位点与理论正态分位点。`,
      },
      legend: {
        data: ['正态参考线', '下行尾部', '中部样本', '上行尾部'],
        type: 'scroll',
        top: 0,
        textStyle: { color: '#475569', fontSize: 10 },
      },
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          const data = Array.isArray(params?.data) ? params.data : [];
          if (data.length < 3) {
            return params?.seriesName ?? '';
          }
          return [
            `${params.seriesName} · 第 ${(Number(data[2]) * 100).toFixed(1)} 百分位`,
            `理论正态分位数：${Number(data[0]).toFixed(2)}`,
            `实际单期收益率：${formatSignedPercent(Number(data[1]), 2)}`,
          ].join('<br/>');
        },
      },
      grid: { left: 20, right: 24, bottom: 56, top: 52, containLabel: true },
      xAxis: {
        type: 'value',
        name: '理论正态分位数',
        splitNumber: 3,
        nameLocation: 'middle',
        nameGap: 34,
        axisLabel: { color: '#475569', fontSize: 10, hideOverlap: true },
        axisLine: { lineStyle: { color: '#cbd5f5' } },
        splitLine: { lineStyle: { color: '#e2e8f0' } },
      },
      yAxis: {
        type: 'value',
        name: '实际单期收益率',
        nameLocation: 'middle',
        nameGap: 38,
        splitNumber: 3,
        axisLabel: {
          color: '#475569',
          fontSize: 10,
          hideOverlap: true,
          formatter: (value: number) => `${value.toFixed(1)}%`,
        },
        splitLine: { lineStyle: { color: '#e2e8f0' } },
      },
      series: [
        {
          name: '正态参考线',
          type: 'line',
          data: normalQqData.points.map((point) => [
            Number(point.theoreticalQuantile.toFixed(4)),
            Number(point.referenceReturn.toFixed(4)),
          ]),
          symbol: 'none',
          silent: true,
          lineStyle: { width: 2, type: 'dashed', color: '#64748b' },
          tooltip: { show: false },
        },
        {
          name: '下行尾部',
          type: 'scatter',
          data: pointsForTail('lower'),
          symbolSize: 7,
          itemStyle: { color: '#f43f5e' },
        },
        {
          name: '中部样本',
          type: 'scatter',
          data: pointsForTail('center'),
          symbolSize: 5,
          itemStyle: { color: '#38bdf8', opacity: 0.72 },
        },
        {
          name: '上行尾部',
          type: 'scatter',
          data: pointsForTail('upper'),
          symbolSize: 7,
          itemStyle: { color: '#10b981' },
        },
      ],
    };
  }, [normalQqData]);

  const simulationInitialNav = simulationAnalysis?.simulation?.initialNav ?? FUTURE_SIMULATION_INITIAL_NAV;
  const simulationMethods = simulationAnalysis?.simulation?.methods ?? SIMULATION_METHOD_ORDER;
  const activeSimulation = simulationAnalysis?.simulation?.byMethod[simulationMethod] ?? null;
  const simulationComparison = simulationAnalysis?.simulation?.comparison ?? null;
  const navDensity = simulationAnalysis?.simulation?.densities[simulationMethod] ?? null;
  /**
   * The distribution the right-hand panel draws: the checkpoint day nearest
   * the zoom window's right edge, or the horizon's own when nothing is zoomed.
   * Frames are strided, so the day shown is named on the chart rather than
   * assumed to be the day the reader dragged to.
   */
  const densityFrame = useMemo(() => {
    const frames = navDensity?.frames ?? [];
    if (frames.length === 0) return null;
    if (densityDay === null) return frames[frames.length - 1];
    return nearestNavDensityFrame(frames, densityDay);
  }, [navDensity, densityDay]);
  const realized = simulationAnalysis?.simulation?.realized ?? null;
  const realizedStatus = simulationAnalysis?.simulation?.realizedStatus ?? 'off';
  const realizedScore = realized?.byMethod[simulationMethod] ?? null;
  /**
   * Whose median sat closest to what actually happened, once the horizon is
   * fully covered. One product on one research day proves nothing about the
   * models in general — but it turns the model strip into a scoreboard, which
   * is the only reason a reader compares five lanes at all.
   */
  const closestMethod = useMemo<SimulationMethod | null>(() => {
    if (!realized?.complete) return null;
    let best: SimulationMethod | null = null;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const method of simulationMethods) {
      const rank = realized.byMethod[method]?.percentileRank;
      if (rank === null || rank === undefined) continue;
      const distance = Math.abs(rank - 0.5);
      if (distance < bestDistance) { bestDistance = distance; best = method; }
    }
    return best;
  }, [realized, simulationMethods]);

  const simulationOption = useMemo(() => {
    if (!activeSimulation || !navDensity || !densityFrame || simulationInitialNav === null) {
      return undefined;
    }
    const percentileSeries = [
      { name: '5% 分位', data: activeSimulation.percentiles.p05, color: '#f43f5e', type: 'dashed', width: 1.5 },
      { name: '25% 分位', data: activeSimulation.percentiles.p25, color: '#f59e0b', type: 'dashed', width: 1 },
      { name: '中位路径', data: activeSimulation.percentiles.p50, color: '#7c3aed', type: 'solid', width: 2.5 },
      { name: '75% 分位', data: activeSimulation.percentiles.p75, color: '#0ea5e9', type: 'dashed', width: 1 },
      { name: '95% 分位', data: activeSimulation.percentiles.p95, color: '#10b981', type: 'dashed', width: 1.5 },
    ];
    const densityCountPoints = navDensityCurve(densityFrame, simulationInitialNav);
    const countAxisMax = densityFrame.countAxisMax;
    const frameDay = densityFrame.day;
    const densityTitle = frameDay >= activeSimulation.days[activeSimulation.days.length - 1]
      ? '期末净值分布'
      : `第 ${frameDay} 日净值分布`;
    // Both reference lines belong to the day being drawn, not to the horizon:
    // a median from day 252 laid over day 40's distribution would sit outside
    // it and read as a model error.
    const frameMedian = activeSimulation.percentiles.p50[frameDay] ?? activeSimulation.terminal.p50;
    const frameRealizedNav = realized?.nav[frameDay] ?? null;
    // The realised path is exactly what may fall outside the simulated range,
    // and that case matters most — an axis fitted to the simulation alone would
    // clip the evidence out of sight.
    const realizedNavValues = (realized?.nav ?? []).filter((value): value is number => value !== null);
    const navAxisMin = Math.min(navDensity.navAxisMin, ...realizedNavValues);
    const navAxisMax = Math.max(navDensity.navAxisMax, ...realizedNavValues);
    return {
      animation: false,
      aria: {
        enabled: true,
        decal: { show: true },
        description: `${activeSimulation.methodLabel}虚拟净值路径图，右侧叠加 ${navDensity.sampleSize} 条模拟路径在第 ${frameDay} 个未来交易日的净值横向直方图与概率密度曲线；拖动下方缩放条可切换到任意日期。${realized ? `另叠加研究日 ${realized.asOf} 之后 ${realized.coveredDays} 个交易日的实际净值走势。` : ''}`,
      },
      tooltip: {
        trigger: 'axis',
        valueFormatter: (value: number | string) => formatDecimal(Number(value), 4),
      },
      legend: {
        data: [...percentileSeries.map((series) => series.name), ...(realized ? [REALIZED_SERIES_NAME] : [])],
        top: 4,
        textStyle: { color: '#475569', fontSize: 11 },
      },
      graphic: [{
        type: 'text',
        left: '84%',
        top: 48,
        silent: true,
        style: { text: densityTitle, fill: '#64748b', fontSize: 11, fontWeight: 600 },
      }],
      grid: [
        { left: 52, right: '19%', bottom: 58, top: 58 },
        { left: '83%', right: '2.5%', bottom: 58, top: 58 },
      ],
      xAxis: [
        {
          type: 'category',
          gridIndex: 0,
          name: '未来交易日',
          data: activeSimulation.days,
          boundaryGap: false,
          axisLabel: { color: '#475569' },
          axisLine: { lineStyle: { color: '#cbd5e1' } },
        },
        {
          type: 'value',
          gridIndex: 1,
          name: '路径数',
          nameLocation: 'middle',
          nameGap: 28,
          min: 0,
          max: countAxisMax,
          minInterval: 1,
          splitNumber: 2,
          axisLabel: { show: true, color: '#64748b', fontSize: 9, formatter: (value: number) => Math.round(value) },
          axisTick: { show: false },
          splitLine: { show: false },
          axisLine: { lineStyle: { color: '#cbd5e1' } },
          nameTextStyle: { color: '#64748b', fontSize: 10 },
        },
      ],
      yAxis: [
        {
          type: 'value',
          gridIndex: 0,
          name: '虚拟净值',
          min: navAxisMin,
          max: navAxisMax,
          axisLabel: { color: '#475569', formatter: (value: number) => formatDecimal(value, 3) },
          splitLine: { lineStyle: { color: '#e2e8f0' } },
        },
        {
          type: 'value',
          gridIndex: 1,
          min: navAxisMin,
          max: navAxisMax,
          axisLabel: { show: false },
          axisTick: { show: false },
          splitLine: { show: false },
          axisLine: { show: true, lineStyle: { color: '#cbd5e1' } },
        },
      ],
      // Always present now: the slider is how the reader picks which day the
      // right-hand distribution answers for, so hiding it on short horizons
      // would hide the feature. Position comes from the ref so a re-render
      // does not throw the window back to the full horizon.
      dataZoom: [
        { type: 'inside', xAxisIndex: 0, start: simulationZoomWindow.current[0], end: simulationZoomWindow.current[1] },
        { xAxisIndex: 0, start: simulationZoomWindow.current[0], end: simulationZoomWindow.current[1], left: 52, right: '19%' },
      ],
      series: [
        ...activeSimulation.samplePaths.map((path, index) => ({
          name: `样本路径 ${index + 1}`,
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: path,
          showSymbol: false,
          silent: true,
          lineStyle: { width: 0.7, color: '#94a3b8', opacity: 0.22 },
          emphasis: { disabled: true },
        })),
        ...percentileSeries.map((series) => ({
          name: series.name,
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: series.data,
          showSymbol: false,
          lineStyle: { width: series.width, color: series.color, type: series.type },
        })),
        {
          name: '期末净值密度填充',
          type: 'custom',
          coordinateSystem: 'cartesian2d',
          xAxisIndex: 1,
          yAxisIndex: 1,
          silent: true,
          z: 1,
          data: [[countAxisMax, densityFrame.navLow]],
          renderItem: (_params: any, api: any) => {
            const curve = densityCountPoints.map((point) => api.coord([point.count, point.nav]));
            const baseline = densityCountPoints
              .slice()
              .reverse()
              .map((point) => api.coord([0, point.nav]));
            return {
              type: 'polygon',
              shape: { points: [...curve, ...baseline] },
              style: { fill: 'rgba(124, 58, 237, 0.08)' },
            };
          },
        },
        {
          name: '期末净值直方图',
          type: 'custom',
          coordinateSystem: 'cartesian2d',
          xAxisIndex: 1,
          yAxisIndex: 1,
          z: 2,
          data: densityFrame.bins.map((count, index) => [
            count,
            densityFrame.navLow + index * densityFrame.binWidth,
            densityFrame.navLow + (index + 1) * densityFrame.binWidth,
          ]),
          renderItem: (_params: any, api: any) => {
            const lower = api.coord([0, api.value(1)]);
            const upper = api.coord([api.value(0), api.value(2)]);
            const height = Math.max(1, lower[1] - upper[1] - 1);
            return {
              type: 'rect',
              shape: { x: lower[0], y: upper[1] + 0.5, width: Math.max(0, upper[0] - lower[0]), height },
              style: { fill: 'rgba(124, 58, 237, 0.18)', stroke: 'rgba(124, 58, 237, 0.34)', lineWidth: 0.5 },
            };
          },
          tooltip: {
            trigger: 'item',
            formatter: (params: any) => {
              const data = Array.isArray(params?.data) ? params.data : [];
              return [
                densityTitle,
                `${formatDecimal(Number(data[1]), 4)} ~ ${formatDecimal(Number(data[2]), 4)}`,
                `路径数：${Number(data[0]) || 0}`,
              ].join('<br/>');
            },
          },
        },
        {
          name: '期末净值概率密度',
          type: 'line',
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: densityCountPoints.map((point) => [point.count, point.nav, point.simulatedReturn]),
          showSymbol: false,
          lineStyle: { width: 2, color: '#7c3aed' },
          emphasis: { disabled: true },
          z: 3,
          tooltip: {
            trigger: 'item',
            formatter: (params: any) => {
              const nav = Number(Array.isArray(params?.data) ? params.data[1] : Number.NaN);
              const estimatedCount = Number(Array.isArray(params?.data) ? params.data[0] : Number.NaN);
              const simulatedReturn = Number(Array.isArray(params?.data) ? params.data[2] : Number.NaN);
              return [
                `${densityTitle} · 概率密度`,
                `虚拟净值：${formatDecimal(nav, 4)}`,
                `区间估算路径数：${Number.isFinite(estimatedCount) ? formatDecimal(estimatedCount, 1) : '--'}`,
                `相对当前收益率：${Number.isFinite(simulatedReturn) ? formatRatioPercent(simulatedReturn) : '--'}`,
              ].join('<br/>');
            },
          },
        },
        {
          name: '期末中位数参考线',
          type: 'line',
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: [[0, frameMedian], [countAxisMax, frameMedian]],
          showSymbol: false,
          silent: true,
          lineStyle: { width: 1.5, type: 'dotted', color: '#7c3aed' },
          z: 4,
        },
        // Drawn last and darkest: one line here is a fact and the others are
        // scenarios, and a reader must never quote a quantile as what happened.
        ...(realized ? [{
          name: REALIZED_SERIES_NAME,
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          data: realized.nav,
          showSymbol: false,
          connectNulls: false,
          lineStyle: { width: 2.6, color: REALIZED_COLOR },
          z: 6,
          tooltip: {
            valueFormatter: (value: number | string) => formatDecimal(Number(value), 4),
          },
        }] : []),
        // Now drawn for any day the realised path reaches, not only a fully
        // covered horizon: a partial path still has a position inside the
        // distribution of the day it got to.
        ...(frameRealizedNav !== null ? [{
          name: '实际净值参考线',
          type: 'line',
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: [[0, frameRealizedNav], [countAxisMax, frameRealizedNav]],
          showSymbol: false,
          silent: true,
          lineStyle: { width: 2, color: REALIZED_COLOR },
          z: 6,
        }] : []),
      ],
    };
  }, [activeSimulation, densityFrame, navDensity, realized, simulationInitialNav]);

  const statisticsRange = useMemo(() => {
    if (dailyReturns.length === 0) {
      return null;
    }
    return {
      start: dailyReturns[0].date,
      end: dailyReturns[dailyReturns.length - 1].date,
      count: dailyReturns.length,
    };
  }, [dailyReturns]);

  const tabs = [
    { id: 'chart', label: '走势与指标' }, { id: 'regime', label: '情景表现' },
    { id: 'statistics', label: '收益统计' }, { id: 'simulation', label: '未来模拟' },
    { id: 'risk', label: '风险与压测' },
  ] as const;
  useEffect(() => { if (searchParams.get('tab') === 'risk') setActiveTab('risk'); }, [searchParams]);
  const context = analysis?.researchContext;
  useEffect(() => { if (activeTab === 'chart') window.dispatchEvent(new Event('resize')); }, [activeTab]);
  const contextLabel = selectedStateId ? `${selectedState?.label ?? context?.stateLabel ?? '所选状态'}${selectedSegmentId ? ' · 单个连续区间' : ' · 全部连续区间'}` : '完整样本';
  return (
    <div className="mx-auto min-w-0 max-w-7xl space-y-5 px-3 py-5 sm:px-6 sm:py-8">
      <button type="button" onClick={() => returnToOrigin(navigate, location, `/product-research/products?kind=${productKind}`)} className="inline-flex min-h-10 items-center rounded-lg px-1 text-sm font-medium text-slate-500 hover:text-sky-700">← {returnNavigation?.returnLabel ?? '返回上一页'}</button>
      {loading ? (
        <div className="flex h-96 items-center justify-center text-slate-400">
          <div className="flex items-center gap-3">
            <svg className="h-5 w-5 animate-spin text-emerald-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle className="opacity-25" cx="12" cy="12" r="10" />
              <path className="opacity-75" d="M4 12a8 8 0 018-8" />
            </svg>
            加载产品详情...
          </div>
        </div>
      ) : error ? (
        <div className="rounded-2xl bg-white p-12 text-center shadow-sm">
          <div className="mx-auto max-w-xl space-y-4">
            <div className="inline-flex rounded-full bg-rose-50 px-4 py-1 text-sm font-semibold text-rose-500">提示</div>
            <p className="text-lg font-semibold text-slate-800">{error}</p>
            <p className="text-sm text-slate-500">如需进一步帮助，请联系系统管理员或返回列表重新选择产品。</p>
          </div>
        </div>
      ) : !detail ? (
        <div className="rounded-2xl bg-white p-12 text-center text-slate-500 shadow-sm">暂无可展示的产品详情。</div>
      ) : (
        <>

          <header className="rounded-2xl border border-slate-200 bg-white px-5 py-5 sm:px-6">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="min-w-0"><p className="text-xs font-semibold tracking-wide text-slate-400">单产品研究 · {productKind === 'fund' ? '公募基金' : 'ETF'}</p><h1 className="mt-1 break-words text-2xl font-semibold tracking-tight text-slate-900">{detail.name ?? '--'}</h1><div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500"><span className="font-medium text-slate-700">{tsCode}</span>{baseInfo['qdii_type'] && <span className="rounded bg-slate-100 px-2 py-0.5">{String(baseInfo['qdii_type'])}</span>}<span>{detail.management}</span><span>{detail.status}</span></div></div>
              <div className="flex gap-6 text-right"><div><p className="text-xs text-slate-500">当前规模</p><p className="mt-1 text-xl font-semibold tabular-nums text-slate-900">{formatIssueAmount(metrics.current_size)}</p></div><div><p className="text-xs text-slate-500">管理 / 托管费</p><p className="mt-2 text-sm font-medium tabular-nums text-slate-700">{formatPercent(metrics.m_fee)} / {formatPercent(metrics.c_fee)}</p></div></div>
            </div>
            {productKind === 'etf' && <Link to={`/product-research/timing?product_id=${encodeURIComponent(String(tsCode || productId))}&kind=etf`} className="mt-4 inline-flex min-h-10 items-center rounded-lg border border-slate-200 px-3 text-sm font-medium text-sky-700 hover:bg-sky-50">研究这个 ETF 的买入与退出规则 →</Link>}
            <details className="mt-4 border-t border-slate-100 pt-3"><summary className="cursor-pointer text-xs font-medium text-slate-500">产品资料与规模口径</summary><div className="mt-3 grid gap-3 text-xs text-slate-600 sm:grid-cols-2 lg:grid-cols-3"><span>管理人：{formatText(detail.management)}</span><span>托管人：{formatText(detail.custodian)}</span><span aria-label={`${inceptionDateLabel}：${inceptionDateText}`}>{inceptionDateLabel}：{inceptionDateText}</span>{endDate && <span aria-label={`${endDateLabel}：${endDateText}`}>{endDateLabel}：{endDateText}</span>}<span>发行规模：{formatIssueAmount(metrics.issue_amount)}</span><span className="sm:col-span-2">{currentSizeDescription}</span></div></details>
          </header>
          <div role="tablist" aria-label="产品研究工作区" className="grid grid-cols-2 sm:grid-cols-5 rounded-xl border border-slate-200 bg-slate-100/70 p-1">{tabs.map((tab, index) => <button key={tab.id} role="tab" id={`product-tab-${tab.id}`} aria-controls={`product-panel-${tab.id}`} aria-selected={activeTab === tab.id} tabIndex={activeTab === tab.id ? 0 : -1} onClick={() => setActiveTab(tab.id)} onKeyDown={event => { const offset = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0; const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + offset + tabs.length) % tabs.length; if (offset || event.key === 'Home' || event.key === 'End') { event.preventDefault(); setActiveTab(tabs[next].id); document.getElementById(`product-tab-${tabs[next].id}`)?.focus(); } }} className={`min-h-11 whitespace-nowrap rounded-lg px-1 py-2 text-[11px] font-semibold transition sm:px-4 sm:text-sm ${activeTab === tab.id ? 'bg-white text-slate-900 shadow-sm ring-1 ring-slate-200/70' : 'text-slate-500 hover:text-slate-800'}`}>{tab.label}</button>)}</div>
          {activeTab !== 'chart' && activeTab !== 'risk' && <section aria-label="分析样本" className="min-w-0 rounded-xl border border-slate-200 bg-white px-3 py-3 sm:px-4">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
              <label className="flex min-w-0 items-center gap-2 text-xs font-medium text-slate-600">
                分析样本区间
                <select aria-label="分析样本区间" value={statisticsPeriod} onChange={event => { setStatisticsPeriod(event.target.value as StatisticsPeriod); setSelectedSegmentId(''); }} className="min-h-10 rounded-lg border border-slate-200 bg-white px-2 text-sm text-slate-800 focus:ring-2 focus:ring-sky-500">
                  {STATISTICS_PERIOD_OPTIONS.map(option => <option key={option.value} value={option.value}>{option.value === 'ALL' ? '全部历史' : option.label}</option>)}
                </select>
              </label>
              <div className="ml-auto flex items-center gap-3 text-xs">
                <span aria-label="当前收益口径" className="text-slate-500">{analysisBasis === 'adjusted_nav' ? '复权净值' : '交易价格'}</span>
                {productKind === 'etf' && <button type="button" aria-expanded={analysisSettingsOpen} aria-controls="product-analysis-settings" onClick={() => setAnalysisSettingsOpen(open => !open)} className="min-h-10 rounded-lg px-2 font-medium text-slate-600 hover:bg-slate-50 hover:text-sky-700 focus:ring-2 focus:ring-sky-500">分析设置 <span aria-hidden="true">{analysisSettingsOpen ? '−' : '+'}</span></button>}
              </div>
            </div>
            {productKind === 'etf' && analysisSettingsOpen && <div id="product-analysis-settings" className="mt-3 flex flex-wrap items-center gap-3 border-t border-slate-100 pt-3">
              <label className="flex items-center gap-2 text-xs text-slate-600">收益数据口径<select aria-label="收益数据口径" value={analysisBasis} onChange={event => { setAnalysisBasis(event.target.value as 'adjusted_nav' | 'price'); setSelectedSegmentId(''); }} className="min-h-10 rounded-lg border border-slate-200 bg-white px-2 text-sm text-slate-800"><option value="adjusted_nav">复权净值</option><option value="price">交易价格</option></select></label>
              <p className="text-xs leading-5 text-slate-500">用于情景表现、收益统计与模拟取样。指标卡片保留各自的计算口径与周期。</p>
            </div>}
            {productResearchRegimeRuns.length > 0 && <div className="mt-3 grid min-w-0 gap-3 border-t border-slate-100 pt-3 sm:grid-cols-2 xl:grid-cols-4">
              <label className="min-w-0 text-xs text-slate-500 sm:col-span-2">情景方案<select aria-label="历史情景背景" value={selectedHistoricalRegimeRunId} onChange={event => changeRegime(event.target.value)} className="mt-1 block min-h-10 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-800"><option value="">不使用情景 · 普通研究</option>{productResearchRegimeRuns.map(run => <option key={run.id} value={run.id}>{run.name} · v{run.definition_revision ?? latestProductResearchPublication(run)?.definition_revision ?? '—'} · {run.mode === 'realtime' ? '实时识别' : '事后研究'}</option>)}</select></label>
              <button type="button" disabled={historicalRegimeLoading} onClick={() => setRegimeRefresh(value => value + 1)} className="min-h-10 self-end justify-self-start rounded-lg border border-slate-200 px-3 text-xs font-medium text-slate-600 disabled:opacity-40">{historicalRegimeLoading ? '正在读取…' : '刷新情景'}</button>
              {selectedHistoricalRegimeRunId && <label className="min-w-0 text-xs text-slate-500">市场状态<select aria-label="市场状态" value={selectedStateId} onChange={event => selectState(event.target.value)} disabled={!selectedHistoricalRegimeRun} className="mt-1 block min-h-10 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-800"><option value="">全部状态 · 保留完整样本</option>{selectedHistoricalRegimeRun?.states.map(state => <option key={state.id} value={state.id}>{state.label}</option>)}</select></label>}
              {selectedStateId && <label className="min-w-0 text-xs text-slate-500">连续区间<select aria-label="连续区间" value={selectedSegmentId} onChange={event => setSelectedSegmentId(event.target.value)} className="mt-1 block min-h-10 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-800"><option value="">该状态的全部连续区间</option>{analysis?.regimeAnalysis?.segments.filter(segment => segment.stateId === selectedStateId).map(segment => <option key={segment.id} value={segment.id}>{segment.startDate} 至 {segment.endDate} · {segment.returnObservations} 个有效收益</option>)}</select></label>}
            </div>}
            {selectedHistoricalRegimeRunId && <div className="mt-2 flex flex-wrap items-start justify-between gap-2 text-xs leading-5 text-slate-500" aria-live="polite">
              {selectedHistoricalRegimeRun && <p className="w-full">{selectedHistoricalRegimeRun.mode === 'retrospective' ? '事后研究 · 用于解释历史表现' : '实时识别 · 按本次保存的数据范围'} · 版本 v{selectedHistoricalRegimeRun.definition_revision} · {selectedHistoricalRegimeRun.series?.[0]?.observation_date || '—'} — {selectedHistoricalRegimeRun.series?.[selectedHistoricalRegimeRun.series.length - 1]?.observation_date || '—'}</p>}
              {historicalRegimeDetailLoading && <p role="status">正在读取所选情景的状态与区间…</p>}{historicalRegimeDetailError && <p role="status" className="text-rose-700">{historicalRegimeDetailError}</p>}
              {selectedHistoricalRegimeRun && <details data-testid="historical-regime-selection-meta" className="min-w-0 flex-1"><summary className="cursor-pointer">情景来源详情</summary><p className="mt-2">不可变运行 · 定义版本 v{selectedHistoricalRegimeRun.definition_revision ?? selectedHistoricalRegimePublication?.definition_revision} · {selectedHistoricalRegimeRun.mode === 'realtime' ? '实时模式：历史研究结果，可得性以版本证据为准。' : '事后识别：用完整历史样本解释，不代表当时已知。'} · 发布于 {formatDate(selectedHistoricalRegimePublication?.published_at)}</p><dl className="mt-2 space-y-1 break-all"><div>运行 {selectedHistoricalRegimeRun.id}</div><div>发布 {selectedHistoricalRegimePublication?.id}</div><div>内容指纹 {selectedHistoricalRegimeRun.content_hash ?? '未提供'}</div></dl></details>}
              <button type="button" onClick={() => changeRegime('')} className="shrink-0 font-medium text-sky-700 hover:underline">清除情景</button>
            </div>}
          </section>}
          {activeTab !== 'risk' && (activeTab !== 'chart' || selectedHistoricalRegimeRunId) && <div className="flex flex-wrap items-center justify-between gap-2 px-1 text-xs text-slate-500" aria-live="polite">
            <span data-testid="product-research-scope" className="font-medium text-slate-700">{activeTab === 'chart' && selectedHistoricalRegimeRun ? `${selectedHistoricalRegimeRun.name} · ` : ''}{contextLabel}{context ? ` · ${context.returnObservations} 个有效收益${context.scope === 'full' ? '' : ` · ${context.segmentCount} 段`}` : ''}</span>
            {context && <span>{context.startDate ?? '—'} — {context.endDate ?? '—'} · {context.basisLabel}</span>}
            {activeTab === 'chart' && <button type="button" onClick={() => setActiveTab('regime')} className="min-h-10 font-medium text-sky-700 hover:underline">调整情景</button>}
          </div>}
          {activeTab !== 'risk' && analysisError && <div role="alert" className="rounded-xl border border-rose-100 bg-rose-50 px-4 py-3 text-sm text-rose-700"><strong>产品数值分析未完成</strong><p className="mt-1 text-xs">{analysisError}{productKind === 'etf' && analysisBasis === 'adjusted_nav' ? '；可检查复权净值数据，或在收益统计的“分析设置”中切换为交易价格。' : ''}</p></div>}
          {activeTab === 'risk' && <section role="tabpanel" id="product-panel-risk" aria-labelledby="product-tab-risk">
            <PublishedRiskPanel key={`${productKind}:${tsCode || productId}`} productKey={`${productKind}:${tsCode || productId}`} productName={detail.name ?? String(tsCode || productId)} />
          </section>}
          <div hidden={activeTab !== 'chart'} role="tabpanel" id="product-panel-chart" aria-labelledby="product-tab-chart" className="min-w-0 space-y-5">
          <section className="space-y-4 rounded-2xl border border-slate-200 bg-white p-3 sm:p-5">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-lg font-semibold text-slate-900">价格与成交量</h2>
              </div>
              <div className="flex flex-wrap gap-2 text-xs text-slate-500">
                <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-3 py-1">
                  <span className="h-2 w-2 rounded-full bg-sky-500" />
                  {rawTechnicalAvailability.ohlc ? 'K 线' : '真实收盘价 / 净值'}
                </span>
                <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-3 py-1">
                  <span className={`h-2 w-2 rounded-full ${rawTechnicalAvailability.volume ? 'bg-emerald-400' : 'bg-slate-300'}`} />
                  {rawTechnicalAvailability.volume ? '成交量' : '成交量未披露'}
                </span>
              </div>
            </div>
            {chartOption ? (
              <ReactECharts option={chartOption} style={{ height: chartHeight }} notMerge lazyUpdate />
            ) : (
              <div className="h-[320px] rounded-2xl bg-slate-50 text-center text-slate-400">暂无可视化数据</div>
            )}
            {detail && (!rawTechnicalAvailability.ohlc || !rawTechnicalAvailability.volume) && (
              <p className="rounded-xl bg-amber-50 px-4 py-3 text-xs leading-5 text-amber-800" role="status">
                原始数据未完整披露
                {!rawTechnicalAvailability.ohlc ? ' OHLC' : ''}
                {!rawTechnicalAvailability.volume ? ' 成交量' : ''}
                ；页面保留真实收盘价 / 净值，不使用 close 或 0 伪造缺失字段。相关 KDJ、成交量均线会保持不可用。
              </p>
            )}
            {overlayLoading && (
              <p className="rounded-xl bg-sky-50 px-4 py-3 text-xs text-sky-700" role="status">
                正在通过指标中心的固定签名 NJIT 计划计算技术时序指标…
              </p>
            )}
            {overlayError && (
              <p className="rounded-xl bg-amber-50 px-4 py-3 text-xs leading-5 text-amber-800" role="alert">
                {overlayError}
              </p>
            )}
            <details className="rounded-2xl bg-slate-50 p-4"><summary className="cursor-pointer text-sm font-semibold text-slate-700">技术辅助线设置</summary><div className="mt-4">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h3 className="text-base font-semibold text-slate-900">技术时序指标</h3>
                  <p className="text-sm text-slate-500">来自指标中心的内置时序指标；参数变化只重算辅助线，不重跑统计与模拟。</p>
                </div>
                <button
                  type="button"
                  onClick={restoreDefaultOverlays}
                  disabled={productKind !== 'etf'}
                  className="inline-flex items-center justify-center rounded-full border border-slate-200 px-3 py-1 text-xs font-medium text-slate-500 transition hover:border-emerald-400 hover:text-emerald-600 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  恢复默认
                </button>
              </div>
              <div className="mt-4 grid gap-3 md:grid-cols-2">
                {overlayOptions.map((option) => {
                  const active = selectedOverlays.includes(option.id);
                  return (
                    <div
                      key={option.id}
                      className={`flex flex-col rounded-2xl border px-5 py-4 transition ${
                        active ? 'border-emerald-400 bg-white shadow-sm' : 'border-transparent bg-white/70 hover:border-emerald-200'
                      }`}
                    >
                      <button
                        type="button"
                        disabled={productKind !== 'etf'}
                        onClick={() => toggleOverlay(option.id)}
                        className="flex items-center justify-between text-left disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        <div>
                          <span className={`text-sm font-semibold ${active ? 'text-emerald-600' : 'text-slate-700'}`}>{option.label}</span>
                          <p className="mt-1 text-xs text-slate-500">{option.description}</p>
                        </div>
                        <span
                          className={`inline-flex h-5 w-10 items-center rounded-full border px-1 transition ${
                            active ? 'border-emerald-400 bg-emerald-500' : 'border-slate-200 bg-slate-200'
                          }`}
                        >
                          <span className={`h-3.5 w-3.5 rounded-full bg-white transition-transform ${active ? 'translate-x-4' : ''}`} />
                        </span>
                      </button>
                      {active && <div className="mt-4 space-y-3 text-sm text-slate-600">{renderOverlayControls(option.id)}</div>}
                    </div>
                  );
                })}
              </div>
            </div>
            </details>
          </section>


          <section className="rounded-2xl border border-slate-200 bg-white p-5" aria-labelledby="custom-research-indicators-title">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 id="custom-research-indicators-title" className="text-lg font-semibold text-slate-900">自定义研究指标</h2>
                <p className="mt-1 max-w-2xl text-sm text-slate-600">
                  使用指标中心的保存版本。连续时间轴指标保留各自计算窗口；情景筛选用于走势高亮，以下指标不作为情景条件统计。
                </p>
              </div>
              <Link
                to={`/settings/indicators-models?kind=${productKind}&ids=${encodeURIComponent(productId)}`}
                className="inline-flex shrink-0 items-center justify-center rounded-lg bg-violet-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-violet-700 focus:outline-none focus:ring-2 focus:ring-violet-400 focus:ring-offset-2"
              >
                在指标中心分析
              </Link>
            </div>
            {researchIndicators.length > 0 && (
              <div className="mt-4 rounded-2xl border border-violet-100 bg-white p-4">
                <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                  <MetricSelector
                    indicators={researchIndicators}
                    selectedIds={researchPreference.indicatorIds}
                    onChange={(indicatorIds) => setResearchPreference((current) => withSelectedIndicators(current, indicatorIds, '1Y'))}
                    maxSelected={8}
                    label="选择研究指标"
                  />
                  <div className="grid gap-3">
                    <label className="text-sm font-medium text-slate-700">截止日（可选）
                      <input type="date" value={researchAsOf} onChange={(event) => setResearchAsOf(event.target.value)} className="mt-1 block min-h-11 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm" />
                      {/* 未填不等于"用全部数据"——它跟随平台研究日。说出来，
                          否则被截断的数字看起来像 bug。 */}
                      <span className="mt-1 block text-xs font-normal text-slate-500">
                        {researchAsOf
                          ? '仅本页生效，覆盖平台研究日。'
                          : platformAsOf === undefined
                            ? '平台 PIT 口径尚未确认；未填时仍由服务端确定口径。'
                          : platformAsOf
                            ? `未填则跟随平台研究日 ${platformAsOf}。`
                            : '未填则使用磁盘上的全部数据。'}
                      </span>
                    </label>
                  </div>
                </div>
                <p className="mt-3 text-xs text-slate-500">每个指标可独立选择计算区间；系统会按区间分组计算。</p>
                <div className="mt-4" aria-live="polite">
                  {researchLoading && <p className="mb-3 text-sm text-slate-500">正在基于真实数据批量计算…</p>}
                  {researchError ? <p className="text-sm text-rose-600">{researchError}</p> : (
                    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                      {selectedResearchIndicators.map((indicator) => <MetricResultCard
                        key={indicator.id}
                        indicator={indicator}
                        result={researchResults.find((result) => result.indicator_id === indicator.id)}
                        period={metricPeriodFor(researchPreference, indicator.id, '1Y')}
                        periodOptions={researchPeriods}
                        onPeriodChange={(period) => setResearchPreference((current) => ({
                          ...current,
                          periodsByIndicator: { ...current.periodsByIndicator, [indicator.id]: period },
                        }))}
                        onRemove={() => setResearchPreference((current) => withSelectedIndicators(
                          current,
                          current.indicatorIds.filter((id) => id !== indicator.id),
                          '1Y',
                        ))}
                        onDefinition={() => setDefinitionIndicator(indicator)}
                      />)}
                    </div>
                  )}
                </div>
              </div>
            )}
            {researchIndicators.length === 0 && !researchError && <div className="mt-4 rounded-2xl border border-dashed border-violet-200 bg-white px-4 py-3 text-sm text-slate-600">工作区尚无保存的自定义指标。请先在指标中心新建或复制内置指标。</div>}
            {researchIndicators.length === 0 && researchError && <p className="mt-4 text-sm text-rose-600" role="status">{researchError}</p>}
          </section>

          <TimeSeriesIndicatorPanel key={`${productKind}:${productId}`} indicators={seriesIndicators} productId={productId} productKind={productKind} periods={researchPeriods} />
            {productId && <FactorEvidencePanel contextType="product_research" contextId={`${productKind}:${productId}`} productId={String(tsCode || productId)} />}
          </div>
          {activeTab === 'regime' && <div role="tabpanel" id="product-panel-regime" aria-labelledby="product-tab-regime">
            {historicalRegimeLoading ? <p role="status" className="rounded-xl bg-white p-10 text-center text-sm text-slate-500">正在读取已保存的情景版本…</p>
              : historicalRegimeError ? <p role="status" className="rounded-xl bg-rose-50 p-6 text-sm text-rose-700">{historicalRegimeError}</p>
              : productResearchRegimeRuns.length === 0 ? <section className="rounded-2xl border border-dashed border-slate-300 bg-white px-5 py-12 text-center">
                <h2 className="text-lg font-semibold text-slate-900">暂无可用情景</h2>
                <button type="button" onClick={() => setRegimeRefresh(value => value + 1)} className="mt-2 min-h-10 px-3 text-sm font-medium text-sky-700">刷新情景</button>
                <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-500">在情景中心点击“保存情景”，即可在这里选择，并比较不同市场状态下的产品表现。</p>
                <Link to="/settings/scenario-algorithms" className="mt-5 inline-flex min-h-10 items-center rounded-lg bg-sky-700 px-4 text-sm font-medium text-white hover:bg-sky-800">前往情景中心</Link>
              </section>
              : <ProductScenarioPanel analysis={analysis?.regimeAnalysis ?? null} selectedStateId={selectedStateId} selectedSegmentId={selectedSegmentId} loading={analysisLoading} onStateChange={selectState} onSegmentChange={selectSegment} onLocate={segment => { selectSegment(segment); setActiveTab('chart'); }} />}
          </div>}
          {activeTab === 'statistics' && <section role="tabpanel" id="product-panel-statistics" aria-labelledby="product-tab-statistics" className="min-w-0 space-y-5 rounded-2xl border border-slate-200 bg-white p-4 sm:p-5"><div><h2 className="text-lg font-semibold text-slate-900">统计分析</h2><p className="mt-1 text-sm text-slate-500">{selectedStateId ? '基于所选状态连续区间内部的相邻有效收益，按观察值汇总。' : '基于研究窗口内的相邻有效收益，观察分布与波动。'}</p>{statisticsRange && <p className="mt-1 text-xs text-slate-500">{statisticsRange.start} — {statisticsRange.end} · {statisticsRange.count} 个有效收益观察值</p>}</div>
            {dailyReturns.length === 0 ? (
              <div className="rounded-2xl bg-slate-50 p-10 text-center text-slate-500">
                {statisticsWindow.message ?? '暂无足够的收益观察值用于统计分析。'}
              </div>
            ) : (
              <>
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
                  <MetricCard title="平均单期收益率" value={formatSignedPercent(returnStats.mean)} description="样本均值" />
                  <MetricCard title="单期波动率" value={formatSignedPercent(returnStats.std)} description="收益率标准差" />
                  <MetricCard title="收益率中位数" value={formatSignedPercent(returnStats.median)} description="样本中位数" />
                  <MetricCard title="正收益占比" value={formatRatioPercent(returnStats.positiveRatio)} description="单期收益率 &gt; 0" />
                  <ExtremesCard best={formatSignedPercent(returnStats.best)} worst={formatSignedPercent(returnStats.worst)} />
                </div>
                <div className="grid gap-4 md:grid-cols-3">
                  <DistributionMetricCard
                    title="偏度"
                    value={formatSignedDecimal(returnStats.skewness)}
                    interpretation={skewnessInterpretation}
                  />
                  <DistributionMetricCard
                    title="峰度（超额）"
                    value={formatSignedDecimal(returnStats.kurtosis)}
                    interpretation={kurtosisInterpretation}
                  />
                  <MetricCard
                    title="正态性检验"
                    value={
                      returnStats.normalityPValue === null
                        ? '--'
                        : `p=${formatDecimal(returnStats.normalityPValue, 3)}`
                    }
                    description={normalityConclusion}
                  />
                </div>
                <div className="grid gap-6 lg:grid-cols-2">
                  <div className="rounded-2xl border border-slate-100 bg-slate-50/60 p-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-semibold text-slate-900">相邻收益序列</h3>
                      <span className="text-xs text-slate-500">{context?.scope === 'full' ? '折线图' : '区间收益柱图'}</span>
                    </div>
                    <div className="mt-4">
                      {returnLineOption && (
                        <ReactECharts option={returnLineOption} style={{ height: 260 }} notMerge lazyUpdate />
                      )}
                    </div>
                  </div>
                  <div className="rounded-2xl border border-slate-100 bg-slate-50/60 p-4">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                      <div>
                        <h3 className="text-sm font-semibold text-slate-900">收益率分布</h3>
                        <p className="text-xs text-slate-500">柱状图 + 正态拟合曲线</p>
                      </div>
                      <label className="flex items-center gap-2 text-xs text-slate-500">
                        区间宽度
                        <select
                          className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 shadow-sm focus:border-emerald-400 focus:outline-none"
                          value={histogramBinWidth}
                          onChange={(event) => setHistogramBinWidth(Number(event.target.value))}
                        >
                          {histogramBinWidthOptions.map((option) => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                    <div className="mt-4">
                      {histogramOption && (
                        <ReactECharts option={histogramOption} style={{ height: 260 }} notMerge lazyUpdate />
                      )}
                      <div className="mt-3 text-xs text-slate-500">
                        当前共 {returnStats.sampleSize} 个样本，划分 {histogramBins.length} 个区间。
                      </div>
                    </div>
                  </div>
                </div>
                <div data-testid="distribution-diagnostics-grid" className="grid gap-6 lg:grid-cols-2">
                  <div className="min-w-0 rounded-2xl border border-slate-100 bg-slate-50/60 p-4">
                    <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
                      <div>
                        <h3 className="text-sm font-semibold text-slate-900">箱形图</h3>
                        <p className="text-xs text-slate-500">横向观察中位数、分位区间与离群值</p>
                      </div>
                      {boxPlotData && (
                        <span className="text-xs text-slate-500">离群值：{boxPlotData.outliers.length} 个</span>
                      )}
                    </div>
                    <div className="mt-4">
                      {boxPlotOption ? (
                        <ReactECharts option={boxPlotOption} style={{ height: 280 }} notMerge lazyUpdate />
                      ) : (
                        <div className="h-[280px] rounded-2xl bg-white/60 text-center text-sm leading-[280px] text-slate-400">
                          样本量不足，无法构建箱形图
                        </div>
                      )}
                    </div>
                    {boxPlotData && (
                      <dl className="mt-4 grid gap-4 text-xs text-slate-600 sm:grid-cols-3">
                        <div>
                          <dt className="font-medium text-slate-500">中位数</dt>
                          <dd className="mt-1 text-sm font-semibold text-slate-900">
                            {formatSignedPercent(boxPlotData.quartiles.median, 2)}
                          </dd>
                        </div>
                        <div>
                          <dt className="font-medium text-slate-500">四分位距 (IQR)</dt>
                          <dd className="mt-1 text-sm font-semibold text-slate-900">
                            {formatSignedPercent(boxPlotData.quartiles.iqr, 2)}
                          </dd>
                        </div>
                        <div>
                          <dt className="font-medium text-slate-500">箱须范围</dt>
                          <dd className="mt-1 text-sm font-semibold text-slate-900">
                            {formatSignedPercent(boxPlotData.whiskers.lower, 2)} ~ {formatSignedPercent(boxPlotData.whiskers.upper, 2)}
                          </dd>
                        </div>
                      </dl>
                    )}
                  </div>
                  <div className="min-w-0 rounded-2xl border border-slate-100 bg-slate-50/60 p-4">
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                      <div>
                        <h3 className="text-sm font-semibold text-slate-900">正态 Q-Q 图</h3>
                        <p className="text-xs text-slate-500">实际收益率分位点与理论正态分位点比较</p>
                      </div>
                      <span className="w-fit rounded-full bg-amber-50 px-2.5 py-1 text-xs font-medium text-amber-800 ring-1 ring-amber-100">
                        {skewnessInterpretation.label} · {kurtosisInterpretation.label}
                      </span>
                    </div>
                    <div className="mt-4">
                      {normalQqOption ? (
                        <ReactECharts option={normalQqOption} style={{ height: 280 }} notMerge lazyUpdate />
                      ) : (
                        <div className="flex h-[280px] items-center justify-center rounded-2xl bg-white/60 text-sm text-slate-400">
                          至少需要 3 个有效单期收益率才能构建 Q-Q 图
                        </div>
                      )}
                    </div>
                    <p className="mt-4 text-xs leading-5 text-slate-500">
                      左端低于参考线表示下行尾部更厚；右端高于参考线表示上行尾部更厚；两端同时外扩通常意味着极端涨跌多于正态分布。
                    </p>
                    {normalQqTableRows.length > 0 && (
                      <details className="mt-3 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs">
                        <summary className="cursor-pointer font-medium text-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                          查看关键分位点数据
                        </summary>
                        <div className="mt-2 max-h-56 overflow-auto">
                          <table className="min-w-full divide-y divide-slate-200 text-left">
                            <caption className="sr-only">正态 Q-Q 图关键分位点数据表</caption>
                            <thead>
                              <tr>
                                <th scope="col" className="px-2 py-2 text-slate-500">样本分位</th>
                                <th scope="col" className="px-2 py-2 text-right text-slate-500">理论分位数</th>
                                <th scope="col" className="px-2 py-2 text-right text-slate-500">实际单期收益率</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100">
                              {normalQqTableRows.map((point) => (
                                <tr key={point.percentile}>
                                  <td className="px-2 py-2 text-slate-600">{formatRatioPercent(point.percentile)}</td>
                                  <td className="px-2 py-2 text-right tabular-nums text-slate-600">{point.theoreticalQuantile.toFixed(2)}</td>
                                  <td className="px-2 py-2 text-right tabular-nums text-slate-900">{formatSignedPercent(point.observedReturn, 2)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </details>
                    )}
                  </div>
                </div>
              </>
            )}

          </section>}
          {activeTab === 'simulation' && <div role="tabpanel" id="product-panel-simulation" aria-labelledby="product-tab-simulation" className="min-w-0 space-y-4">
            <p className="rounded-xl border border-sky-100 bg-sky-50/60 px-4 py-3 text-sm leading-6 text-sky-900">{selectedStateId ? `假设未来持续处于“${selectedState?.label ?? context?.stateLabel ?? '所选状态'}”，使用${selectedSegmentId ? '所选连续区间' : '该状态'}的历史收益生成路径，不预测市场状态转换。` : '使用当前研究窗口的历史收益生成未来路径。选择市场状态，可进一步观察该状态持续时的条件结果。'}</p>
                <div className="rounded-2xl border border-violet-100 bg-violet-50/40 p-5" aria-labelledby="future-simulation-title">
                  <div>
                    <div className="max-w-2xl">
                      <h2 id="future-simulation-title" className="text-base font-semibold text-slate-900">未来虚拟净值模拟</h2>
                      {/* Each model now explains itself inside its own panel, so
                          the old "how the two methods use the sample" disclosure
                          was saying it twice. */}
                      <p className="mt-1.5 text-xs leading-5 text-slate-500">
                        所有路径统一从虚拟净值 1.0000 出发；期末 0.9000 表示亏损 10%，1.1000 表示盈利 10%。
                        模拟按一个收益观察值对应一个交易日建模，非日频净值需结合数据口径解释。
                      </p>
                    </div>
                    {/* Two sections, in reading order: what the experiment is,
                        then which model answers it. The old single row mixed
                        Bootstrap's block length in with the horizon and the
                        path budget, which read as though all five applied
                        everywhere — only one of them did. */}
                    <fieldset className="mt-5" data-testid="simulation-experiment-settings">
                      <legend className="text-[11px] font-semibold uppercase tracking-[0.08em] text-violet-800">
                        ① 实验设置 · 对所有模型一致
                      </legend>
                      <div className="mt-2 grid min-w-0 gap-3 rounded-xl border border-violet-200 bg-white p-3 sm:grid-cols-2 lg:grid-cols-4">
                        <label className="text-xs font-medium text-slate-600">
                          模拟未来区间
                          <select
                            aria-label="模拟未来区间"
                            value={simulationHorizon}
                            onChange={(event) => setSimulationHorizon(Number(event.target.value))}
                            className="mt-1 block min-h-10 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm"
                          >
                            {MONTE_CARLO_HORIZON_OPTIONS.map((option) => (
                              <option key={option.value} value={option.value}>{option.label}</option>
                            ))}
                          </select>
                        </label>
                        <label className="text-xs font-medium text-slate-600">
                          模拟路径数
                          <select
                            aria-label="模拟路径数"
                            value={simulationPathCount}
                            onChange={(event) => setSimulationPathCount(Number(event.target.value))}
                            className="mt-1 block min-h-10 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm"
                          >
                            {MONTE_CARLO_PATH_OPTIONS.map((count) => (
                              <option key={count} value={count}>{count} 条 / 每个模型</option>
                            ))}
                          </select>
                        </label>
                        <label className="text-xs font-medium text-slate-600">
                          目标期末收益率
                          <span className="relative mt-1 block">
                            <input
                              aria-label="目标期末收益率"
                              type="number"
                              min="-100"
                              max="1000"
                              step="1"
                              value={simulationTargetReturn}
                              onChange={(event) => setSimulationTargetReturn(Math.max(-100, Math.min(1000, Number(event.target.value) || 0)))}
                              className="min-h-10 w-full rounded-xl border border-slate-200 bg-white px-3 pr-8 text-sm"
                            />
                            <span className="pointer-events-none absolute right-3 top-2.5 text-sm text-slate-400">%</span>
                          </span>
                        </label>
                        <button
                          type="button"
                          onClick={() => void runSimulation()}
                          disabled={analysisLoading || !analysis?.researchContext?.simulationEligible || Boolean(currentSimulation?.loading)}
                          className="min-h-10 self-end rounded-xl bg-violet-600 px-4 text-sm font-semibold text-white shadow-sm transition hover:bg-violet-700 disabled:cursor-not-allowed disabled:opacity-40 focus:outline-none focus:ring-2 focus:ring-violet-400 focus:ring-offset-2"
                        >
                          {currentSimulation?.loading ? '正在模拟…' : activeSimulation ? '重新模拟' : '运行模拟'}
                        </button>
                      </div>
                      <p className="mt-1.5 text-[11px] leading-4 text-slate-500">
                        一次运行同时跑完所有模型，共用同一区间、路径预算与目标收益——否则对比表与实际走势叠加都失去意义。
                      </p>
                    </fieldset>
                  </div>
                  <fieldset className="mt-4" data-testid="simulation-model-picker">
                    <legend className="text-[11px] font-semibold uppercase tracking-[0.08em] text-violet-800">
                      ② 选择模型 · 参数各自独立
                    </legend>
                    <div className="mt-2 overflow-hidden rounded-xl border border-violet-200 bg-white">
                      <div className="flex flex-wrap gap-1 border-b border-violet-100 bg-violet-50/70 p-1.5" role="radiogroup" aria-label="模拟方法">
                        {simulationMethods.map((method) => {
                          const meta = SIMULATION_METHOD_META[method];
                          const active = simulationMethod === method;
                          return (
                            <label
                              key={method}
                              className={`cursor-pointer rounded-lg px-3 py-1.5 text-xs font-semibold transition ${active ? 'bg-violet-600 text-white shadow-sm' : 'text-slate-600 hover:bg-white'}`}
                            >
                              <input
                                className="sr-only"
                                type="radio"
                                name="simulation-method"
                                value={method}
                                checked={active}
                                onChange={() => setSimulationMethod(method)}
                              />
                              {meta.tab}
                              {closestMethod === method && (
                                <span className={`ml-1.5 rounded px-1 py-0.5 text-[10px] font-bold ${active ? 'bg-white/25' : 'bg-slate-900 text-white'}`}>
                                  最接近
                                </span>
                              )}
                            </label>
                          );
                        })}
                      </div>
                      <div className="flex flex-col gap-3 px-4 py-3 lg:flex-row lg:items-start lg:justify-between">
                        <div className="min-w-0 lg:max-w-2xl">
                          <p className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                            {activeSimulation?.methodLabel ?? SIMULATION_METHOD_META[simulationMethod].tab}
                            <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${SIMULATION_METHOD_META[simulationMethod].conditional ? 'border-violet-300 bg-violet-50 text-violet-800' : 'border-slate-200 bg-slate-50 text-slate-600'}`}>
                              {SIMULATION_METHOD_META[simulationMethod].conditional ? '条件模型 · 从当前波动状态出发' : '无条件模型 · 忽略当前波动状态'}
                            </span>
                          </p>
                          <p className="mt-1.5 text-xs leading-5 text-slate-600">{SIMULATION_METHOD_META[simulationMethod].summary}</p>
                        </div>
                        <div className="shrink-0 lg:w-64" data-testid="simulation-model-parameter">
                          {SIMULATION_METHOD_META[simulationMethod].parameter === 'bootstrap_block_length' ? (
                            <label className="text-xs font-medium text-slate-600">
                              本模型参数 · 平均区块长度
                              <select
                                aria-label="Bootstrap 平均区块长度"
                                value={bootstrapBlockLength}
                                onChange={(event) => setBootstrapBlockLength(Number(event.target.value))}
                                className="mt-1 block min-h-10 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm"
                              >
                                {BOOTSTRAP_BLOCK_LENGTH_OPTIONS.map((length) => (
                                  <option key={length} value={length}>{length} 个收益观察值</option>
                                ))}
                              </select>
                            </label>
                          ) : SIMULATION_METHOD_META[simulationMethod].parameter === 'fhs_ewma_lambda' ? (
                            <label className="text-xs font-medium text-slate-600">
                              本模型参数 · 衰减系数 λ
                              <select
                                aria-label="EWMA 衰减系数"
                                value={fhsEwmaLambda}
                                onChange={(event) => setFhsEwmaLambda(Number(event.target.value))}
                                className="mt-1 block min-h-10 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm"
                              >
                                {FHS_EWMA_LAMBDA_OPTIONS.map((option) => (
                                  <option key={option.value} value={option.value}>{option.label}</option>
                                ))}
                              </select>
                            </label>
                          ) : (
                            <p className="rounded-xl border border-dashed border-slate-200 px-3 py-2 text-[11px] leading-4 text-slate-500">
                              本模型无可调参数：所有取值都由样本自动确定。
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                    <p className="mt-1.5 text-[11px] leading-4 text-slate-500">
                      改动任一模型的参数都会让整批结果失效，需要重新运行——五个模型必须来自同一次运行才可比。
                    </p>
                  </fieldset>
                  {activeSimulation && simulationOption ? (
                    <>
                      <p className="mt-3 text-xs leading-5 text-slate-600" aria-live="polite">
                        当前方法：{activeSimulation.methodLabel}；样本 {activeSimulation.assumptions.sourceObservationCount} 个有效收益观察值。
                        {' '}{simulationAnalysis?.researchContext?.startDate} — {simulationAnalysis?.researchContext?.endDate} · {simulationAnalysis?.researchContext?.basisLabel}。

                        {simulationAssumptionNote(activeSimulation)}
                      </p>
                      <details className="mt-2 text-xs text-slate-500"><summary className="cursor-pointer">模拟样本与数据来源</summary><p className="mt-2 break-all">{simulationAnalysis?.researchContext?.scope === 'full' ? '完整样本' : simulationAnalysis?.researchContext?.stateLabel} · {simulationAnalysis?.researchContext?.returnObservations} 个有效收益 · 数据指纹 {simulationAnalysis?.researchContext?.dataFingerprint}</p>{simulationAnalysis?.researchContext?.warnings?.map((warning, index) => <p key={index} className="mt-1">{warning}</p>)}</details>
                      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4" aria-live="polite">
                        <MetricCard title="期末 5% 分位" value={formatDecimal(activeSimulation.terminal.p05, 4)} description="偏悲观情景，不等同于最大损失" />
                        <MetricCard title="期末中位净值" value={formatDecimal(activeSimulation.terminal.p50, 4)} description="一半路径高于该值" />
                        <MetricCard title="期末 95% 分位" value={formatDecimal(activeSimulation.terminal.p95, 4)} description="偏乐观情景，不等同于收益承诺" />
                        <MetricCard title="期末亏损概率" value={formatRatioPercent(activeSimulation.terminal.lossProbability)} description={`${simulationPathCount} 条虚拟路径中的样本比例`} />
                        <MetricCard title="95% VaR（损失）" value={formatRatioPercent(activeSimulation.terminal.valueAtRisk95)} description="期末收益 5% 分位对应的损失幅度" />
                        <MetricCard title="95% CVaR（预期短缺）" value={formatRatioPercent(activeSimulation.terminal.conditionalValueAtRisk95)} description="最差 5% 期末情景的平均损失" />
                        <MetricCard title="平均最大回撤" value={formatRatioPercent(activeSimulation.terminal.averageMaxDrawdown)} description="每条模拟路径最大回撤的平均值" />
                        <MetricCard title={`达到 ${simulationTargetReturn}% 概率`} value={formatRatioPercent(activeSimulation.terminal.targetHitProbability)} description="期末收益达到目标的路径比例" />
                      </div>
                      {realized && realizedScore ? (
                        <RealizedFutureStrip
                          realized={realized}
                          score={realizedScore}
                          simulatedAverageMaxDrawdown={activeSimulation.terminal.averageMaxDrawdown}
                        />
                      ) : (
                        <p className="mt-4 rounded-2xl border border-dashed border-slate-200 bg-white px-4 py-3 text-xs leading-5 text-slate-500">
                          {realizedStatus === 'off' ? (
                            <>
                              未启用 PIT 研究日，模拟没有可以对照的后续走势。把顶部的 PIT 标签切到历史某一天（只影响本标签页），或在
                              {' '}
                              <Link to="/settings/pit-snapshots" className="text-sky-700 underline hover:no-underline">数据版本管理</Link>
                              {' '}应用一个研究日靠前的版本，这里会自动叠加那之后的实际走势并给模型评分。
                            </>
                          ) : (
                            '研究日之后还没有该产品的行情数据，暂时无法回看实际走势。把研究日往前移，或等数据刷新到更晚的日期。'
                          )}
                        </p>
                      )}
                      <div
                        data-testid="monte-carlo-combined-chart"
                        aria-label={`${activeSimulation.methodLabel}：路径与期末净值概率分布组合图`}
                        className="mt-4 rounded-2xl bg-white p-3"
                      >
                        <ReactECharts
                          option={simulationOption}
                          style={{ height: 360 }}
                          notMerge
                          lazyUpdate
                          onEvents={{ datazoom: handleSimulationZoom }}
                        />
                      </div>
                      <p className="mt-2 text-xs leading-5 text-slate-500">
                        右侧分布画的是<strong className="font-semibold text-slate-700">第 {densityFrame?.day ?? simulationHorizon} 个未来交易日</strong>：{navDensity?.sampleSize ?? 0} 条模拟路径当天的净值落点，柱状图为区间路径数，紫色曲线为同一批结果的平滑概率密度，虚线为当天的模拟中位数。
                        {' '}拖动图表下方的缩放条即可换成任意一天——分布越靠前越窄，因为路径还没来得及分开。
                        {realized && `深色实线为研究日之后的实际净值走势（第 1 — ${realized.coveredDays} 个交易日）；右侧同色横线标出当天的实际净值在模拟分布中的位置。`}
                      </p>
                      {simulationComparison && simulationInitialNav !== null && (
                        <div data-testid="simulation-model-comparison" className="mt-5 overflow-hidden rounded-2xl border border-slate-200 bg-white">
                          <div className="flex flex-col gap-2 border-b border-slate-200 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
                            <div>
                              <h4 className="text-sm font-semibold text-slate-900">{simulationMethods.length} 个模型结果对比</h4>
                              <p className="mt-1 text-xs text-slate-500">同一历史区间、未来周期、路径数、目标收益与随机轮次；差异只来自模型假设本身。</p>
                            </div>
                            <span className={`w-fit rounded-full px-3 py-1 text-xs font-semibold ${simulationComparison.level === 'high' ? 'bg-rose-100 text-rose-700' : simulationComparison.level === 'medium' ? 'bg-amber-100 text-amber-700' : 'bg-emerald-100 text-emerald-700'}`}>
                              模型分歧度：{simulationComparison.level === 'high' ? '高' : simulationComparison.level === 'medium' ? '中' : '低'}
                            </span>
                          </div>
                          <div className="overflow-x-auto">
                            <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
                              <caption className="sr-only">全部模拟模型的结果对比</caption>
                              <thead className="bg-slate-50 text-slate-500">
                                <tr>
                                  <th scope="col" className="px-4 py-3">模型</th>
                                  <th scope="col" className="px-3 py-3 text-right">5% 分位收益</th>
                                  <th scope="col" className="px-3 py-3 text-right">中位收益</th>
                                  <th scope="col" className="px-3 py-3 text-right">亏损概率</th>
                                  <th scope="col" className="px-3 py-3 text-right">95% CVaR</th>
                                  <th scope="col" className="px-3 py-3 text-right">平均最大回撤</th>
                                  <th scope="col" className="px-4 py-3 text-right">目标达成概率</th>
                                  {realized && (
                                    <>
                                      <th scope="col" className="px-3 py-3 text-right">实际所处分位</th>
                                      <th scope="col" className="px-4 py-3 text-right">实际留在区间</th>
                                    </>
                                  )}
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-slate-100 text-slate-700">
                                {simulationMethods.map((method) => simulationAnalysis?.simulation?.byMethod[method]).filter((simulation): simulation is FuturePathSimulation => Boolean(simulation)).map((simulation) => (
                                  <tr key={simulation.method} className={simulation.method === simulationMethod ? 'bg-violet-50/60' : undefined}>
                                    <th scope="row" className="whitespace-nowrap px-4 py-3 font-semibold text-slate-900">
                                      {simulation.methodLabel}
                                      {closestMethod === simulation.method && (
                                        <span className="ml-1.5 rounded bg-slate-900 px-1 py-0.5 text-[10px] font-bold text-white">最接近</span>
                                      )}
                                    </th>
                                    <td className="px-3 py-3 text-right tabular-nums">{formatRatioPercent(simulation.terminal.p05Return)}</td>
                                    <td className="px-3 py-3 text-right tabular-nums">{formatRatioPercent(simulation.terminal.medianReturn)}</td>
                                    <td className="px-3 py-3 text-right tabular-nums">{formatRatioPercent(simulation.terminal.lossProbability)}</td>
                                    <td className="px-3 py-3 text-right tabular-nums">{formatRatioPercent(simulation.terminal.conditionalValueAtRisk95)}</td>
                                    <td className="px-3 py-3 text-right tabular-nums">{formatRatioPercent(simulation.terminal.averageMaxDrawdown)}</td>
                                    <td className="px-4 py-3 text-right tabular-nums">{formatRatioPercent(simulation.terminal.targetHitProbability)}</td>
                                    {realized && (
                                      <>
                                        <td className="px-3 py-3 text-right tabular-nums font-semibold text-slate-900">
                                          {realized.byMethod[simulation.method].percentileRank === null
                                            ? realized.byMethod[simulation.method].bandLabel ?? '--'
                                            : `${formatRatioPercent(realized.byMethod[simulation.method].percentileRank)} 分位`}
                                        </td>
                                        <td className="px-4 py-3 text-right tabular-nums">
                                          {formatRatioPercent(realized.byMethod[simulation.method].containmentRatio)}
                                        </td>
                                      </>
                                    )}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                          <p className="border-t border-slate-100 px-4 py-3 text-xs leading-5 text-slate-600">
                            {simulationComparison.message}
                            {' '}最保守与最乐观模型之间：5% 分位收益差 {formatRatioPercent(simulationComparison.p05ReturnGap)}、中位收益差 {formatRatioPercent(simulationComparison.medianReturnGap)}、亏损概率差 {formatRatioPercent(simulationComparison.lossProbabilityGap)}、95% CVaR 差 {formatRatioPercent(simulationComparison.conditionalValueAtRiskGap)}。
                            {realized && ' 右两列是同一条实际走势对每个模型的事后评分：期末分位越靠近 50%，该模型当时的中枢越接近后来发生的事——单个产品、单个研究日的一次命中不足以判定模型优劣。'}
                          </p>
                        </div>
                      )}
                    </>
                  ) : (
                    <div role="status" className="mt-5 rounded-2xl border border-dashed border-slate-200 bg-white px-5 py-12 text-center text-sm text-slate-500">
                      {currentSimulation?.loading ? '正在生成模拟路径…' : currentSimulation?.error ? <span className="text-rose-700">{currentSimulation.error}</span> : !analysis?.researchContext?.simulationEligible ? analysis?.researchContext?.simulationMessage ?? `至少需要 ${MIN_SIMULATION_OBSERVATIONS} 个有效收益观察值。` : '选择未来周期，点击“运行模拟”生成结果。修改研究条件或模拟参数后，需要重新运行。'}
                    </div>
                  )}
                  <p className="mt-3 text-xs leading-5 text-slate-500">
                    两种方法都是基于历史样本和模型假设的情景生成，不预测市场状态切换、结构性变化或未来事件，不构成收益预测或投资建议。
                  </p>
                </div>
          </div>}
          <details data-testid="product-analysis-execution" className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-xs text-slate-500"><summary className="cursor-pointer">{analysisLoading ? '正在计算研究结果…' : analysisError ? '计算详情 · 分析未完成' : analysis ? '高性能计算已验证' : '计算详情'}</summary><div className="mt-3 flex flex-wrap gap-x-4 gap-y-2">{analysis ? <><span>固定签名 NJIT</span><span>内核覆盖 {analysis.execution.kernel_coverage}</span><span>{analysis.execution.nopython ? 'nopython' : '执行模式异常'}</span><span>Object mode {analysis.execution.object_mode}</span><span>Python 回退 {analysis.execution.python_fallback}</span><span title={analysis.execution.kernel_fingerprint}>指纹 {analysis.execution.kernel_fingerprint.slice(0, 12)}</span></> : <span>{analysisError ?? '等待产品分析任务。'}；页面不会退回浏览器本地计算。</span>}</div></details>
          <MetricDefinitionDrawer indicator={definitionIndicator} onClose={() => setDefinitionIndicator(null)} />
        </>
      )}
    </div>
  );
}
