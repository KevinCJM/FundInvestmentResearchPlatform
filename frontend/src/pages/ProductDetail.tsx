import React, { useEffect, useMemo, useState } from 'react';
import { Link, useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';
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
import {
  getHistoricalRegimeRun,
  listHistoricalRegimeRuns,
  type HistoricalRegimeRun,
  type RegimePublication,
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
  MIN_SIMULATION_OBSERVATIONS,
  MONTE_CARLO_HORIZON_OPTIONS,
  MONTE_CARLO_PATH_OPTIONS,
  STATISTICS_PERIOD_OPTIONS,
  analyzeProduct,
  type DistributionInterpretation,
  type ProductAnalysisResponse,
  type ProductRegimeStatistic,
  type SimulationMethod,
  type StatisticsPeriod,
} from '../services/productAnalysis';
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
  'builtin-maximum-drawdown-v2',
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

const latestProductResearchPublication = (run: HistoricalRegimeRun): RegimePublication | undefined => (
  (run.publications ?? [])
    .filter((publication) => publication.usage === 'product_research' && publication.run_id === run.id)
    .sort((left, right) => right.published_at.localeCompare(left.published_at))[0]
);

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
  const [researchPeriods, setResearchPeriods] = useState<string[]>(['1Y']);
  const [researchResults, setResearchResults] = useState<EvaluationResult[]>([]);
  const [researchAsOf, setResearchAsOf] = useState('');
  const [researchLoading, setResearchLoading] = useState(false);
  const [researchError, setResearchError] = useState<string | null>(null);
  const [definitionIndicator, setDefinitionIndicator] = useState<IndicatorDefinition | null>(null);
  const [statisticsPeriod, setStatisticsPeriod] = useState<StatisticsPeriod>('ALL');
  const [simulationMethod, setSimulationMethod] = useState<SimulationMethod>('parametric');
  const [simulationHorizon, setSimulationHorizon] = useState(252);
  const [simulationPathCount, setSimulationPathCount] = useState(500);
  const [bootstrapBlockLength, setBootstrapBlockLength] = useState(20);
  const [simulationTargetReturn, setSimulationTargetReturn] = useState(5);
  const [simulationRun, setSimulationRun] = useState(0);
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
        if (!data?.timeseries || data.timeseries.length === 0) {
          console.warn('Received product detail without timeseries, unable to render chart');
          setError('产品详情数据缺失，无法展示。');
          setDetail(null);
          return;
        }
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
  }, []);

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
    () => historicalRegimeRuns
      .filter((run) => run.immutable && Boolean(latestProductResearchPublication(run)))
      .sort((left, right) => {
        const leftPublishedAt = latestProductResearchPublication(left)?.published_at ?? '';
        const rightPublishedAt = latestProductResearchPublication(right)?.published_at ?? '';
        return rightPublishedAt.localeCompare(leftPublishedAt);
      }),
    [historicalRegimeRuns],
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
  const historicalRegimeMarkAreas = useMemo(
    () => buildRegimeMarkAreas(
      selectedHistoricalRegimeRun,
      detail?.timeseries.map((item) => item.date) ?? [],
    ),
    [detail?.timeseries, selectedHistoricalRegimeRun],
  );
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

  useEffect(() => {
    if (!detail || !productId) {
      setAnalysis(null);
      setAnalysisLoading(false);
      return;
    }
    let active = true;
    const controller = new AbortController();
    setAnalysisLoading(true);
    setAnalysisError(null);
    setAnalysis(null);
    analyzeProduct(productId, productKind, {
      statistics_period: statisticsPeriod,
      include_technical: false,
      price_ma_periods: [],
      volume_ma_periods: [],
      boll_period: 20,
      boll_multiplier: 2,
      kdj_period: 9,
      kdj_k_smoothing: 3,
      kdj_d_smoothing: 3,
      histogram_bin_width: histogramBinWidth,
      simulation_horizon: simulationHorizon,
      simulation_path_count: simulationPathCount,
      bootstrap_block_length: bootstrapBlockLength,
      simulation_target_return: simulationTargetReturn,
      simulation_run: simulationRun,
      regime: selectedHistoricalRegimeRunSummary && selectedHistoricalRegimePublication
        ? {
            run_id: selectedHistoricalRegimeRunSummary.id,
            publication_id: selectedHistoricalRegimePublication.id,
          }
        : null,
    }, controller.signal)
      .then((payload) => {
        if (active) setAnalysis(payload);
      })
      .catch((requestError) => {
        if (!active || (requestError as DOMException).name === 'AbortError') return;
        setAnalysisError(requestError instanceof Error ? requestError.message : '产品分析失败，请稍后重试。');
      })
      .finally(() => {
        if (active) setAnalysisLoading(false);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [
    bootstrapBlockLength,
    detail,
    histogramBinWidth,
    productId,
    productKind,
    selectedHistoricalRegimePublication,
    selectedHistoricalRegimeRunSummary,
    simulationHorizon,
    simulationPathCount,
    simulationRun,
    simulationTargetReturn,
    statisticsPeriod,
  ]);
  const productRegimeStatistics: ProductRegimeStatistic[] = analysis?.regimeStatistics ?? [];
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
    const startValue = dates[startIndex];
    const endValue = dates[totalPoints - 1];
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
      { left: '6%', right: '4%', top: primaryTop, height: klineHeight },
      { left: '6%', right: '4%', top: primaryTop + klineHeight + gridGap, height: volumeHeight },
    ];
    const xAxis: any[] = [
      {
        type: 'category',
        data: dates,
        boundaryGap: false,
        axisLine: { lineStyle: { color: '#cbd5f5' } },
        axisLabel: { color: '#475569' },
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
        axisLabel: { color: '#475569' },
      },
      {
        gridIndex: 1,
        axisLine: { lineStyle: { color: '#cbd5f5' } },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: '#e2e8f0' } },
        axisLabel: { color: '#475569' },
      },
    ];

    if (hasKDJ) {
      grid.push({ left: '6%', right: '4%', top: primaryTop + klineHeight + gridGap + volumeHeight + gridGap, height: extraPanelHeight });
      xAxis.push({
        type: 'category',
        gridIndex: 2,
        data: dates,
        boundaryGap: false,
        axisTick: { show: false },
        axisLine: { lineStyle: { color: '#cbd5f5' } },
        axisLabel: { color: '#475569' },
      });
      yAxis.push({
        gridIndex: 2,
        axisLine: { lineStyle: { color: '#cbd5f5' } },
        splitLine: { lineStyle: { color: '#e2e8f0' } },
        axisLabel: { color: '#475569' },
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
      grid: { left: '6%', right: '4%', bottom: 60, top: 40 },
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
          name: '日收益率',
          type: 'line',
          smooth: true,
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
        data: ['出现天数', '正态拟合'],
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
          lines.push(`出现天数：${bin.count}`);
          if (hasNormalPdf) lines.push(`正态拟合：${formatDecimal(bin.normalPdfCount, 2)} 天`);
          lines.push(`频率：${formatDecimal(bin.frequency, 3)}`);
          return lines.join('<br/>');
        },
      },
      grid: { left: '6%', right: '4%', bottom: 70, top: 60 },
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
          name: '出现天数',
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
        name: '日收益率',
        nameLocation: 'middle',
        nameGap: 36,
        axisLabel: {
          color: '#475569',
          formatter: (value: number) => `${value.toFixed(1)}%`,
        },
        axisLine: { lineStyle: { color: '#cbd5f5' } },
        splitLine: { lineStyle: { color: '#e2e8f0' } },
      },
      yAxis: {
        type: 'category',
        data: ['日收益率'],
        axisLabel: { color: '#475569' },
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
        description: `正态 Q-Q 图，比较 ${normalQqData.sampleSize} 个实际日收益率分位点与理论正态分位点。`,
      },
      legend: {
        data: ['正态参考线', '下行尾部', '中部样本', '上行尾部'],
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
            `实际日收益率：${formatSignedPercent(Number(data[1]), 2)}`,
          ].join('<br/>');
        },
      },
      grid: { left: 20, right: 24, bottom: 56, top: 52, containLabel: true },
      xAxis: {
        type: 'value',
        name: '理论正态分位数',
        nameLocation: 'middle',
        nameGap: 34,
        axisLabel: { color: '#475569' },
        axisLine: { lineStyle: { color: '#cbd5f5' } },
        splitLine: { lineStyle: { color: '#e2e8f0' } },
      },
      yAxis: {
        type: 'value',
        name: '实际日收益率',
        nameLocation: 'middle',
        nameGap: 48,
        axisLabel: {
          color: '#475569',
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

  const simulationInitialNav = analysis?.simulation?.initialNav ?? FUTURE_SIMULATION_INITIAL_NAV;
  const parametricSimulation = analysis?.simulation?.parametric ?? null;
  const bootstrapSimulation = analysis?.simulation?.blockBootstrap ?? null;
  const activeSimulation = simulationMethod === 'parametric' ? parametricSimulation : bootstrapSimulation;
  const simulationComparison = analysis?.simulation?.comparison ?? null;
  const terminalNavDensity = analysis?.simulation?.densities[simulationMethod] ?? null;

  const simulationOption = useMemo(() => {
    if (!activeSimulation || !terminalNavDensity || simulationInitialNav === null) {
      return undefined;
    }
    const percentileSeries = [
      { name: '5% 分位', data: activeSimulation.percentiles.p05, color: '#f43f5e', type: 'dashed', width: 1.5 },
      { name: '25% 分位', data: activeSimulation.percentiles.p25, color: '#f59e0b', type: 'dashed', width: 1 },
      { name: '中位路径', data: activeSimulation.percentiles.p50, color: '#7c3aed', type: 'solid', width: 2.5 },
      { name: '75% 分位', data: activeSimulation.percentiles.p75, color: '#0ea5e9', type: 'dashed', width: 1 },
      { name: '95% 分位', data: activeSimulation.percentiles.p95, color: '#10b981', type: 'dashed', width: 1.5 },
    ];
    const densityCountPoints = terminalNavDensity.points.map((point) => ({
      nav: point.nav,
      count: point.estimatedCount,
      simulatedReturn: point.simulatedReturn,
    }));
    const countAxisMax = terminalNavDensity.countAxisMax;
    const navAxisMin = terminalNavDensity.navAxisMin;
    const navAxisMax = terminalNavDensity.navAxisMax;
    return {
      animation: false,
      aria: {
        enabled: true,
        decal: { show: true },
        description: `${activeSimulation.methodLabel}虚拟净值路径图，右侧叠加 ${terminalNavDensity.sampleSize} 条模拟期末净值的横向直方图与概率密度曲线。`,
      },
      tooltip: {
        trigger: 'axis',
        valueFormatter: (value: number | string) => formatDecimal(Number(value), 4),
      },
      legend: {
        data: percentileSeries.map((series) => series.name),
        top: 4,
        textStyle: { color: '#475569', fontSize: 11 },
      },
      graphic: [{
        type: 'text',
        left: '84%',
        top: 48,
        silent: true,
        style: { text: '期末净值分布', fill: '#64748b', fontSize: 11, fontWeight: 600 },
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
      dataZoom: simulationHorizon > 126
        ? [
            { type: 'inside', xAxisIndex: 0, start: 0, end: 100 },
            { xAxisIndex: 0, start: 0, end: 100, left: 52, right: '19%' },
          ]
        : [],
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
          data: [[countAxisMax, terminalNavDensity.minNav]],
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
          data: terminalNavDensity.histogram.map((bin) => [bin.count, bin.lowerNav, bin.upperNav]),
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
                '期末净值直方图',
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
                '期末净值概率密度',
                `期末净值：${formatDecimal(nav, 4)}`,
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
          data: [[0, activeSimulation.terminal.p50], [countAxisMax, activeSimulation.terminal.p50]],
          showSymbol: false,
          silent: true,
          lineStyle: { width: 1.5, type: 'dotted', color: '#7c3aed' },
          z: 4,
        },
      ],
    };
  }, [activeSimulation, simulationHorizon, terminalNavDensity]);

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

  return (
    <div className="mx-auto max-w-7xl space-y-8 px-6 py-10">
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => returnToOrigin(navigate, location, `/product-research/products?kind=${productKind}`)}
          className="inline-flex items-center rounded-full border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-600 shadow-sm hover:border-emerald-400 hover:text-emerald-600"
        >
          ← {returnNavigation?.returnLabel ?? '返回上一页'}
        </button>
      </div>

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
          <section className="space-y-6 rounded-3xl bg-white p-8 shadow-sm ring-1 ring-slate-100">
            <div className="flex flex-col gap-6 xl:flex-row xl:items-center xl:justify-between">
              <div>
                <div className="text-sm font-semibold uppercase tracking-wide text-emerald-500">产品研究</div>
                <h1 className="mt-2 text-3xl font-bold text-slate-900">{detail.name ?? '--'}</h1>
                <div className="mt-2 flex flex-wrap gap-3 text-sm text-slate-500">
                  {tsCode && <span className="inline-flex rounded-full bg-emerald-50 px-3 py-1 text-emerald-600">{tsCode}</span>}
                  {baseInfo['qdii_type'] && (
                    <span className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold ${baseInfo['qdii_type'] === 'QDII' ? 'bg-violet-100 text-violet-700' : 'bg-slate-100 text-slate-600'}`}>
                      {String(baseInfo['qdii_type'])}
                    </span>
                  )}
                  {detail.management && <span>管理人：{formatText(detail.management)}</span>}
                  {detail.custodian && <span>托管人：{formatText(detail.custodian)}</span>}
                  <span aria-label={`${inceptionDateLabel}：${inceptionDateText}`}>
                    {inceptionDateLabel}：{inceptionDateText}
                  </span>
                  {endDate && (
                    <span aria-label={`${endDateLabel}：${endDateText}`}>
                      {endDateLabel}：{endDateText}
                    </span>
                  )}
                  {detail.status && (
                    <span className="inline-flex items-center rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">
                      {formatText(detail.status)}
                    </span>
                  )}
                </div>
              </div>
              <div className="grid gap-4 sm:grid-cols-2 xl:min-w-[650px] xl:grid-cols-3">
                <MetricCard
                  title="发行规模"
                  value={formatIssueAmount(metrics.issue_amount)}
                  description="基于信息表披露的发行规模"
                />
                <MetricCard
                  title="当前规模"
                  value={formatIssueAmount(metrics.current_size)}
                  description={currentSizeDescription}
                />
                <MetricCard
                  title="管理 / 托管费"
                  value={`${formatPercent(metrics.m_fee)} / ${formatPercent(metrics.c_fee)}`}
                  description="产品费用率概览"
                />
              </div>
            </div>
          </section>

          <section
            data-testid="product-analysis-execution"
            className="rounded-2xl border border-emerald-100 bg-emerald-50/50 px-5 py-4"
            aria-live="polite"
          >
            {analysisLoading ? (
              <p className="text-sm font-medium text-emerald-800">正在由后端固定签名 NJIT 内核计算技术指标、统计分布与双模型模拟…</p>
            ) : analysisError ? (
              <div>
                <p className="text-sm font-semibold text-rose-700">产品数值分析未完成</p>
                <p className="mt-1 text-xs text-rose-600">{analysisError}；页面不会退回浏览器本地计算。</p>
              </div>
            ) : analysis ? (
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-emerald-800">
                <span className="font-semibold">高性能计算已验证</span>
                <span>固定签名 NJIT</span>
                <span>内核覆盖 {analysis.execution.kernel_coverage}</span>
                <span>{analysis.execution.nopython ? 'nopython' : '执行模式异常'}</span>
                <span>Object mode {analysis.execution.object_mode}</span>
                <span>Python 回退 {analysis.execution.python_fallback}</span>
                <span title={analysis.execution.kernel_fingerprint}>指纹 {analysis.execution.kernel_fingerprint.slice(0, 12)}</span>
              </div>
            ) : (
              <p className="text-sm text-slate-500">等待产品分析任务。</p>
            )}
          </section>

          <section className="rounded-3xl border border-violet-100 bg-violet-50/40 p-6 shadow-sm" aria-labelledby="custom-research-indicators-title">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 id="custom-research-indicators-title" className="text-lg font-semibold text-slate-900">自定义研究指标</h2>
                <p className="mt-1 max-w-2xl text-sm text-slate-600">
                  使用工作区已保存的公式，基于该产品的真实净值计算研究指标。它们与下方仅用于图表叠加的 MA、BOLL、KDJ 技术辅助线相互独立。
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
          <MetricDefinitionDrawer indicator={definitionIndicator} onClose={() => setDefinitionIndicator(null)} />

          <section className="space-y-6 rounded-3xl bg-white p-8 shadow-sm ring-1 ring-slate-100">
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
            <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4" aria-labelledby="historical-regime-background-title">
              <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                <div className="max-w-2xl">
                  <h3 id="historical-regime-background-title" className="text-sm font-semibold text-slate-900">历史情景背景</h3>
                  <p className="mt-1 text-xs leading-5 text-slate-500">
                    仅可叠加已发布到“产品研究”的不可变识别结果；色块按情景区间与本产品价格日期的真实交集绘制。
                  </p>
                </div>
                <label className="w-full text-xs font-medium text-slate-600 lg:w-[420px]">
                  选择历史情景版本
                  <select
                    aria-label="历史情景背景"
                    value={selectedHistoricalRegimeRunId}
                    onChange={(event) => setSelectedHistoricalRegimeRunId(event.target.value)}
                    disabled={historicalRegimeLoading || Boolean(historicalRegimeError) || productResearchRegimeRuns.length === 0}
                    className="mt-1 min-h-10 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm text-slate-700 shadow-sm disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400"
                  >
                    <option value="">关闭历史情景背景</option>
                    {productResearchRegimeRuns.map((run) => {
                      const publication = latestProductResearchPublication(run);
                      const revision = run.definition_revision ?? publication?.definition_revision;
                      const modeLabel = run.mode === 'realtime' ? '实时识别' : '事后识别';
                      return (
                        <option key={run.id} value={run.id}>
                          {run.name} · {revision === undefined || revision === null ? '版本未标明' : `v${revision}`} · {modeLabel}
                        </option>
                      );
                    })}
                  </select>
                </label>
              </div>
              <div className="mt-3" aria-live="polite">
                {historicalRegimeLoading ? (
                  <p className="text-xs text-slate-500">正在读取已发布的历史情景版本…</p>
                ) : historicalRegimeError ? (
                  <p className="text-xs text-rose-600" role="status">{historicalRegimeError}</p>
                ) : productResearchRegimeRuns.length === 0 ? (
                  <p className="text-xs text-slate-500" role="status">
                    暂无已发布到“产品研究”的不可变历史情景版本，请先在历史情景识别中心完成发布。
                  </p>
                ) : selectedHistoricalRegimeRunId && historicalRegimeDetailLoading ? (
                  <p className="text-xs text-indigo-700" role="status">正在按需读取所选情景的完整区间与状态…</p>
                ) : selectedHistoricalRegimeRunId && historicalRegimeDetailError ? (
                  <p className="text-xs text-rose-600" role="status">{historicalRegimeDetailError}</p>
                ) : selectedHistoricalRegimeRun && selectedHistoricalRegimePublication ? (
                  <div className="space-y-2" data-testid="historical-regime-selection-meta">
                    <p className="text-xs leading-5 text-slate-600">
                      不可变运行 · 定义版本 v{selectedHistoricalRegimeRun.definition_revision ?? selectedHistoricalRegimePublication.definition_revision}
                      {' · '}
                      {selectedHistoricalRegimeRun.mode === 'realtime'
                        ? '实时识别：按当时可得信息生成，可按当时视角解释。'
                        : '事后识别：基于完整历史样本划分，仅用于研究解释，不代表当时可获知。'}
                      {' · '}发布于 {formatDate(selectedHistoricalRegimePublication.published_at)}
                    </p>
                    <div className="flex flex-wrap gap-2" aria-label="历史情景图例">
                      {selectedHistoricalRegimeRun.states.map((state) => (
                        <span key={state.id} className="inline-flex items-center gap-1.5 rounded-full bg-white px-2.5 py-1 text-xs text-slate-600 ring-1 ring-slate-200">
                          <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: state.color }} />
                          {state.label}
                        </span>
                      ))}
                    </div>
                    {historicalRegimeMarkAreas.length === 0 && (
                      <p className="text-xs text-amber-700" role="status">所选情景与当前产品价格日期没有交集，图表未绘制背景。</p>
                    )}
                    {productRegimeStatistics.length > 0 && (
                      <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white">
                        <table className="w-full min-w-[720px] text-xs" aria-label="产品历史情景表现">
                          <caption className="px-3 py-2 text-left font-semibold text-slate-700">本产品在各历史情景区间的真实表现</caption>
                          <thead className="bg-slate-50 text-slate-500">
                            <tr>
                              <th className="px-3 py-2 text-left">情景</th>
                              <th className="px-3 py-2 text-right">价格点</th>
                              <th className="px-3 py-2 text-right">区间内收益</th>
                              <th className="px-3 py-2 text-right">年化波动</th>
                              <th className="px-3 py-2 text-right">最深回撤</th>
                              <th className="px-3 py-2 text-right">上涨日占比</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-100">
                            {productRegimeStatistics.map((item) => (
                              <tr key={item.stateId}>
                                <th scope="row" className="px-3 py-2 text-left font-semibold text-slate-800">
                                  <span className="mr-2 inline-block h-2.5 w-2.5 rounded-sm align-middle" style={{ backgroundColor: item.color }} />
                                  {item.stateLabel}
                                </th>
                                <td className="px-3 py-2 text-right tabular-nums text-slate-600">{item.observations}</td>
                                <td className="px-3 py-2 text-right font-semibold tabular-nums text-slate-800">{formatRatioPercent(item.cumulativeReturn)}</td>
                                <td className="px-3 py-2 text-right tabular-nums text-slate-600">{formatRatioPercent(item.annualizedVolatility)}</td>
                                <td className="px-3 py-2 text-right tabular-nums text-rose-700">{formatRatioPercent(item.maxDrawdown)}</td>
                                <td className="px-3 py-2 text-right tabular-nums text-slate-600">{formatRatioPercent(item.winRate)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        <p className="border-t border-slate-100 px-3 py-2 text-[11px] leading-4 text-slate-500">
                          仅复合每个连续情景段内部的相邻日收益；跨情景边界收益不会归入任一状态，避免边界跳变污染统计。
                        </p>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-xs text-slate-500">当前未叠加历史情景背景。</p>
                )}
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
            <div className="rounded-2xl bg-slate-50 p-6">
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
          </section>

          <section className="space-y-6 rounded-3xl bg-white p-8 shadow-sm ring-1 ring-slate-100">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <h2 className="text-lg font-semibold text-slate-900">统计分析</h2>
                <p className="text-sm text-slate-500">基于所选区间的日度收盘价计算收益率，辅助评估分布特征与波动水平。</p>
              </div>
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                <label className="flex items-center gap-2 text-sm font-medium text-slate-600">
                  统计区间
                  <select
                    aria-label="统计区间"
                    value={statisticsPeriod}
                    onChange={(event) => setStatisticsPeriod(event.target.value as StatisticsPeriod)}
                    className="min-h-10 rounded-xl border border-slate-200 bg-white px-3 text-sm text-slate-700 shadow-sm focus:border-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-100"
                  >
                    {STATISTICS_PERIOD_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>{option.label}</option>
                    ))}
                  </select>
                </label>
                {statisticsRange && (
                  <div className="text-xs text-slate-500">
                    样本区间：{statisticsRange.start} ~ {statisticsRange.end}（共 {statisticsRange.count} 个交易日）
                  </div>
                )}
              </div>
            </div>
            {dailyReturns.length === 0 ? (
              <div className="rounded-2xl bg-slate-50 p-10 text-center text-slate-500">
                {statisticsWindow.message ?? '暂无足够的日度收益数据用于统计分析。'}
              </div>
            ) : (
              <>
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
                  <MetricCard title="平均日收益率" value={formatSignedPercent(returnStats.mean)} description="样本均值" />
                  <MetricCard title="日波动率" value={formatSignedPercent(returnStats.std)} description="收益率标准差" />
                  <MetricCard title="收益率中位数" value={formatSignedPercent(returnStats.median)} description="样本中位数" />
                  <MetricCard title="正收益占比" value={formatRatioPercent(returnStats.positiveRatio)} description="日度收益率 &gt; 0" />
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
                      <h3 className="text-sm font-semibold text-slate-900">日收益率序列</h3>
                      <span className="text-xs text-slate-500">折线图</span>
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
                          至少需要 3 个有效日收益率才能构建 Q-Q 图
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
                                <th scope="col" className="px-2 py-2 text-right text-slate-500">实际日收益率</th>
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
                <div className="rounded-2xl border border-violet-100 bg-violet-50/40 p-5" aria-labelledby="future-simulation-title">
                  <div>
                    <div className="max-w-2xl">
                      <h3 id="future-simulation-title" className="text-base font-semibold text-slate-900">未来虚拟净值模拟</h3>
                      <p className="mt-1 text-sm leading-6 text-slate-600">
                        参数化蒙特卡洛使用 sinh-arcsinh 分布同时校准历史对数收益的均值、波动率、偏度与超额峰度；区块 Bootstrap 成段抽取历史收益，额外保留短期时序依赖。
                      </p>
                      <p className="mt-1 text-xs leading-5 text-slate-500">
                        所有路径统一从虚拟净值 1.0000 出发；期末 0.9000 表示亏损 10%，1.1000 表示盈利 10%。
                      </p>
                    </div>
                    <div className="mt-4 grid min-w-0 gap-3 sm:grid-cols-2 lg:grid-cols-5">
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
                            <option key={count} value={count}>{count} 条</option>
                          ))}
                        </select>
                      </label>
                      <label className="text-xs font-medium text-slate-600">
                        Bootstrap 平均区块
                        <select
                          aria-label="Bootstrap 平均区块长度"
                          value={bootstrapBlockLength}
                          onChange={(event) => setBootstrapBlockLength(Number(event.target.value))}
                          className="mt-1 block min-h-10 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm"
                        >
                          {BOOTSTRAP_BLOCK_LENGTH_OPTIONS.map((length) => (
                            <option key={length} value={length}>{length} 个交易日</option>
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
                        onClick={() => setSimulationRun((current) => current + 1)}
                        className="min-h-10 self-end rounded-xl bg-violet-600 px-4 text-sm font-semibold text-white shadow-sm transition hover:bg-violet-700 focus:outline-none focus:ring-2 focus:ring-violet-400 focus:ring-offset-2"
                      >
                        重新模拟
                      </button>
                    </div>
                  </div>
                  <fieldset className="mt-5">
                    <legend className="sr-only">模拟方法</legend>
                    <div className="inline-flex rounded-xl border border-violet-200 bg-white p-1" aria-label="模拟方法">
                      {([
                        ['parametric', '参数化蒙特卡洛'],
                        ['block_bootstrap', '区块 Bootstrap'],
                      ] as Array<[SimulationMethod, string]>).map(([method, label]) => (
                        <label
                          key={method}
                          className={`cursor-pointer rounded-lg px-4 py-2 text-sm font-semibold transition ${simulationMethod === method ? 'bg-violet-600 text-white shadow-sm' : 'text-slate-600 hover:bg-violet-50'}`}
                        >
                          <input
                            className="sr-only"
                            type="radio"
                            name="simulation-method"
                            value={method}
                            checked={simulationMethod === method}
                            onChange={() => setSimulationMethod(method)}
                          />
                          {label}
                        </label>
                      ))}
                    </div>
                  </fieldset>
                  {activeSimulation && simulationOption ? (
                    <>
                      <p className="mt-3 text-xs leading-5 text-slate-600" aria-live="polite">
                        当前方法：{activeSimulation.methodLabel}；样本 {activeSimulation.assumptions.sourceObservationCount} 个日收益观察值。
                        {activeSimulation.method === 'parametric'
                          ? ` 日均对数收益 ${formatRatioPercent(activeSimulation.assumptions.meanDailyLogReturn)}，日波动 ${formatRatioPercent(activeSimulation.assumptions.dailyLogVolatility)}；历史对数收益偏度 ${formatDecimal(activeSimulation.assumptions.historicalLogSkewness, 2)}、超额峰度 ${formatDecimal(activeSimulation.assumptions.historicalLogExcessKurtosis, 2)}，拟合值分别为 ${formatDecimal(activeSimulation.assumptions.fittedLogSkewness, 2)}、${formatDecimal(activeSimulation.assumptions.fittedLogExcessKurtosis, 2)}（${activeSimulation.assumptions.shapeCalibrationStatus === 'matched' ? '四矩校准已匹配' : activeSimulation.assumptions.shapeCalibrationStatus === 'approximate' ? '四矩近似校准' : '正态安全降级'}）。`
                          : ` 平均区块长度 ${activeSimulation.assumptions.averageBlockLength} 个交易日；区块边界随机，实际区块长度会变化。`}
                      </p>
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
                      <div
                        data-testid="monte-carlo-combined-chart"
                        aria-label={`${activeSimulation.methodLabel}：路径与期末净值概率分布组合图`}
                        className="mt-4 rounded-2xl bg-white p-3"
                      >
                        <ReactECharts option={simulationOption} style={{ height: 360 }} notMerge lazyUpdate />
                      </div>
                      <p className="mt-2 text-xs leading-5 text-slate-500">
                        右侧约占图表六分之一：横向柱状图按期末净值区间展示实际路径数，共计 {terminalNavDensity?.sampleSize ?? 0} 条；紫色曲线为同一批模拟结果的平滑概率密度，并按区间路径数尺度对齐。
                      </p>
                      {parametricSimulation && bootstrapSimulation && simulationComparison && simulationInitialNav !== null && (
                        <div className="mt-5 overflow-hidden rounded-2xl border border-slate-200 bg-white">
                          <div className="flex flex-col gap-2 border-b border-slate-200 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
                            <div>
                              <h4 className="text-sm font-semibold text-slate-900">双模型结果对比</h4>
                              <p className="mt-1 text-xs text-slate-500">同一历史区间、未来周期、路径数、目标收益与随机轮次。</p>
                            </div>
                            <span className={`w-fit rounded-full px-3 py-1 text-xs font-semibold ${simulationComparison.level === 'high' ? 'bg-rose-100 text-rose-700' : simulationComparison.level === 'medium' ? 'bg-amber-100 text-amber-700' : 'bg-emerald-100 text-emerald-700'}`}>
                              模型敏感度：{simulationComparison.level === 'high' ? '高' : simulationComparison.level === 'medium' ? '中' : '低'}
                            </span>
                          </div>
                          <div className="overflow-x-auto">
                            <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
                              <caption className="sr-only">参数化蒙特卡洛与区块 Bootstrap 模拟结果对比</caption>
                              <thead className="bg-slate-50 text-slate-500">
                                <tr>
                                  <th scope="col" className="px-4 py-3">模型</th>
                                  <th scope="col" className="px-3 py-3 text-right">5% 分位收益</th>
                                  <th scope="col" className="px-3 py-3 text-right">中位收益</th>
                                  <th scope="col" className="px-3 py-3 text-right">亏损概率</th>
                                  <th scope="col" className="px-3 py-3 text-right">95% CVaR</th>
                                  <th scope="col" className="px-3 py-3 text-right">平均最大回撤</th>
                                  <th scope="col" className="px-4 py-3 text-right">目标达成概率</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-slate-100 text-slate-700">
                                {[parametricSimulation, bootstrapSimulation].map((simulation) => (
                                  <tr key={simulation.method}>
                                    <th scope="row" className="whitespace-nowrap px-4 py-3 font-semibold text-slate-900">{simulation.methodLabel}</th>
                                    <td className="px-3 py-3 text-right tabular-nums">{formatRatioPercent(simulation.terminal.p05Return)}</td>
                                    <td className="px-3 py-3 text-right tabular-nums">{formatRatioPercent(simulation.terminal.medianReturn)}</td>
                                    <td className="px-3 py-3 text-right tabular-nums">{formatRatioPercent(simulation.terminal.lossProbability)}</td>
                                    <td className="px-3 py-3 text-right tabular-nums">{formatRatioPercent(simulation.terminal.conditionalValueAtRisk95)}</td>
                                    <td className="px-3 py-3 text-right tabular-nums">{formatRatioPercent(simulation.terminal.averageMaxDrawdown)}</td>
                                    <td className="px-4 py-3 text-right tabular-nums">{formatRatioPercent(simulation.terminal.targetHitProbability)}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                          <p className="border-t border-slate-100 px-4 py-3 text-xs leading-5 text-slate-600">{simulationComparison.message}</p>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="mt-5 rounded-2xl bg-white p-8 text-center text-sm text-slate-500">所选区间至少需要 {MIN_SIMULATION_OBSERVATIONS} 个有效日收益观察值，当前无法进行模拟。</div>
                  )}
                  <p className="mt-3 text-xs leading-5 text-slate-500">
                    两种方法都是基于历史样本和模型假设的情景生成，不预测市场状态切换、结构性变化或未来事件，不构成收益预测或投资建议。
                  </p>
                </div>
              </>
            )}
          </section>
        </>
      )}
    </div>
  );
}
