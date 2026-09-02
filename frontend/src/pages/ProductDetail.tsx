import React, { useEffect, useMemo, useState } from 'react';
import { Link, useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';
import {
  evaluateCustomIndicators,
  getCustomIndicatorMeta,
  indicatorsForContext,
  listCustomIndicators,
  type EvaluationResult,
  type IndicatorDefinition,
} from '../services/customIndicators';
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
  buildNormalQqData,
  buildTerminalNavDensity,
  compareSimulations,
  interpretExcessKurtosis,
  interpretSkewness,
  selectStatisticsWindow,
  simulateParametricMonteCarlo,
  simulateStationaryBlockBootstrap,
  type DistributionInterpretation,
  type SimulationMethod,
  type StatisticsPeriod,
} from '../utils/statisticalAnalysis';
import { readReturnNavigationState, returnToOrigin } from '../utils/returnNavigation';

interface TimeSeriesPoint {
  date: string;
  open: number;
  close: number;
  high: number;
  low: number;
  volume: number;
}

interface DailyReturnPoint {
  date: string;
  return: number;
}

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
    current_size_source?: 'total_netasset' | 'net_asset' | null;
    m_fee?: number | null;
    c_fee?: number | null;
  };
  timeseries: TimeSeriesPoint[];
}

type OverlayId = 'PRICE_MA' | 'VOLUME_MA' | 'BOLL' | 'KDJ';

interface OverlayOption {
  id: OverlayId;
  label: string;
  description: string;
}

interface ReturnStatistics {
  mean: number | null;
  std: number | null;
  median: number | null;
  positiveRatio: number | null;
  best: number | null;
  worst: number | null;
  sampleSize: number;
  skewness: number | null;
  kurtosis: number | null;
  jbStatistic: number | null;
  normalityPValue: number | null;
}

interface ReturnHistogramBin {
  start: number;
  end: number;
  count: number;
}

interface BoxPlotResult {
  stats: [number, number, number, number, number];
  outliers: number[];
  quartiles: { q1: number; q3: number; median: number };
  whiskers: { lower: number; upper: number };
}

const overlayOptions: OverlayOption[] = [
  { id: 'PRICE_MA', label: '收盘价均线', description: '自定义多个周期观察趋势' },
  { id: 'VOLUME_MA', label: '成交量均线', description: '识别量能变化节奏' },
  { id: 'BOLL', label: '布林带', description: '判断波动区间与突破' },
  { id: 'KDJ', label: 'KDJ 指标', description: '研判超买超卖信号' },
];

const histogramBinWidthOptions = [
  { label: '0.05%', value: 0.05 },
  { label: '0.10%', value: 0.1 },
  { label: '0.20%', value: 0.2 },
  { label: '0.50%', value: 0.5 },
  { label: '1.00%', value: 1 },
];

const FUTURE_SIMULATION_INITIAL_NAV = 1;

type OverlaySettings = {
  PRICE_MA: { periods: string };
  VOLUME_MA: { periods: string };
  BOLL: { period: number; multiplier: number };
  KDJ: { period: number; kSmoothing: number; dSmoothing: number };
};

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

const calculateMovingAverage = (values: number[], period: number) => {
  return values.map((_, index) => {
    if (index + 1 < period) {
      return null;
    }
    const window = values.slice(index - period + 1, index + 1);
    const sum = window.reduce((acc, cur) => acc + cur, 0);
    return Number((sum / period).toFixed(2));
  });
};

const parsePeriods = (input: string) => {
  return input
    .split(/[,，\s]+/)
    .map((item) => Number(item.trim()))
    .filter((num) => Number.isFinite(num) && num > 0)
    .map((num) => Math.round(num));
};

const calculateBollinger = (values: number[], period: number, multiplier: number) => {
  return values.map((_, index) => {
    if (index + 1 < period) {
      return { upper: null, middle: null, lower: null };
    }
    const window = values.slice(index - period + 1, index + 1);
    const mean = window.reduce((acc, cur) => acc + cur, 0) / period;
    const variance = window.reduce((acc, cur) => acc + (cur - mean) ** 2, 0) / period;
    const std = Math.sqrt(variance);
    return {
      upper: Number((mean + multiplier * std).toFixed(2)),
      middle: Number(mean.toFixed(2)),
      lower: Number((mean - multiplier * std).toFixed(2)),
    };
  });
};

const calculateKDJ = (series: TimeSeriesPoint[], period: number, kSmoothing: number, dSmoothing: number) => {
  const kValues: (number | null)[] = [];
  const dValues: (number | null)[] = [];
  const jValues: (number | null)[] = [];
  let prevK = 50;
  let prevD = 50;

  series.forEach((item, index) => {
    const start = Math.max(0, index - period + 1);
    const window = series.slice(start, index + 1);
    const high = Math.max(...window.map((point) => point.high));
    const low = Math.min(...window.map((point) => point.low));
    let rsv = 50;
    if (high !== low) {
      rsv = ((item.close - low) / (high - low)) * 100;
    }
    const k = ((kSmoothing - 1) * prevK + rsv) / kSmoothing;
    const d = ((dSmoothing - 1) * prevD + k) / dSmoothing;
    const j = 3 * k - 2 * d;
    const fixedK = Number(k.toFixed(2));
    const fixedD = Number(d.toFixed(2));
    const fixedJ = Number(j.toFixed(2));
    kValues.push(fixedK);
    dValues.push(fixedD);
    jValues.push(fixedJ);
    prevK = fixedK;
    prevD = fixedD;
  });

  return { kValues, dValues, jValues };
};

const calculateDailyReturns = (series: TimeSeriesPoint[]): DailyReturnPoint[] => {
  if (!Array.isArray(series) || series.length < 2) {
    return [];
  }
  const sorted = [...series].sort((a, b) => a.date.localeCompare(b.date));
  const points: DailyReturnPoint[] = [];
  for (let index = 1; index < sorted.length; index += 1) {
    const current = sorted[index];
    const previous = sorted[index - 1];
    if (!Number.isFinite(previous.close) || !Number.isFinite(current.close) || previous.close === 0) {
      continue;
    }
    const dailyReturn = ((current.close - previous.close) / previous.close) * 100;
    if (!Number.isFinite(dailyReturn)) {
      continue;
    }
    points.push({ date: current.date, return: Number(dailyReturn.toFixed(4)) });
  }
  return points;
};

const calculateReturnStatistics = (values: number[]): ReturnStatistics => {
  if (!Array.isArray(values) || values.length === 0) {
    return {
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
  }
  const filtered = values.filter((item) => Number.isFinite(item));
  if (filtered.length === 0) {
    return {
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
  }
  const sum = filtered.reduce((acc, cur) => acc + cur, 0);
  const mean = sum / filtered.length;
  const variance = filtered.reduce((acc, cur) => acc + (cur - mean) ** 2, 0) / filtered.length;
  const std = Math.sqrt(variance);
  const sorted = [...filtered].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  const positiveRatio = filtered.filter((item) => item > 0).length / filtered.length;
  const best = Math.max(...filtered);
  const worst = Math.min(...filtered);
  let skewness: number | null = null;
  let kurtosis: number | null = null;
  let jbStatistic: number | null = null;
  let normalityPValue: number | null = null;
  if (std > 0) {
    const n = filtered.length;
    const standardized = filtered.map((item) => (item - mean) / std);
    if (n > 2) {
      const skewNumerator = standardized.reduce((acc, cur) => acc + cur ** 3, 0);
      skewness = (Math.sqrt(n * (n - 1)) / (n - 2)) * (skewNumerator / n);
    }
    if (n > 3) {
      const kurtNumerator = standardized.reduce((acc, cur) => acc + cur ** 4, 0);
      const populationExcessKurtosis = kurtNumerator / n - 3;
      kurtosis = ((n - 1) / ((n - 2) * (n - 3)))
        * ((n + 1) * populationExcessKurtosis + 6);
    }
    if (skewness !== null && kurtosis !== null) {
      const jb = (n / 6) * ((skewness ** 2) + (kurtosis ** 2) / 4);
      jbStatistic = jb;
      normalityPValue = Math.exp(-jb / 2);
    }
  }
  return {
    mean,
    std,
    median,
    positiveRatio,
    best,
    worst,
    sampleSize: filtered.length,
    skewness,
    kurtosis,
    jbStatistic,
    normalityPValue,
  };
};

const calculateHistogram = (values: number[], binWidth: number): ReturnHistogramBin[] => {
  if (!Array.isArray(values) || values.length === 0) {
    return [];
  }
  const filtered = values.filter((item) => Number.isFinite(item));
  if (filtered.length === 0) {
    return [];
  }
  const safeWidth = Math.max(binWidth, 0.01);
  const min = Math.min(...filtered);
  const max = Math.max(...filtered);
  if (min === max) {
    return [{ start: min, end: min + safeWidth, count: filtered.length }];
  }
  const normalizedMin = Math.floor(min / safeWidth) * safeWidth;
  const normalizedMax = Math.ceil(max / safeWidth) * safeWidth;
  const binCount = Math.max(1, Math.round((normalizedMax - normalizedMin) / safeWidth));
  const bins: ReturnHistogramBin[] = Array.from({ length: binCount }, (_, index) => {
    const start = normalizedMin + index * safeWidth;
    const end = index === binCount - 1 ? normalizedMax : start + safeWidth;
    return { start, end, count: 0 };
  });
  filtered.forEach((value) => {
    let idx = Math.floor((value - normalizedMin) / safeWidth);
    if (idx < 0) {
      idx = 0;
    }
    if (idx >= binCount) {
      idx = binCount - 1;
    }
    bins[idx].count += 1;
  });
  return bins;
};

const calculateNormalPdfCounts = (
  mean: number | null,
  std: number | null,
  sampleSize: number,
  bins: ReturnHistogramBin[],
  binWidth: number
) => {
  if (mean === null || std === null || !Number.isFinite(mean) || !Number.isFinite(std) || std <= 0) {
    return [];
  }
  if (!Number.isFinite(sampleSize) || sampleSize <= 0) {
    return [];
  }
  const safeWidth = Math.max(binWidth, 0.01);
  const variance = std ** 2;
  return bins.map((bin) => {
    const center = (bin.start + bin.end) / 2;
    const exponent = -((center - mean) ** 2) / (2 * variance);
    const pdf = (1 / (Math.sqrt(2 * Math.PI * variance))) * Math.exp(exponent);
    return pdf * sampleSize * safeWidth;
  });
};

const calculateBoxPlot = (values: number[]): BoxPlotResult | null => {
  if (!Array.isArray(values) || values.length < 5) {
    return null;
  }
  const filtered = values.filter((item) => Number.isFinite(item));
  if (filtered.length < 5) {
    return null;
  }
  const sorted = [...filtered].sort((a, b) => a - b);
  const quantile = (q: number) => {
    const pos = (sorted.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    const lower = sorted[base];
    const upper = sorted[Math.min(sorted.length - 1, base + 1)];
    return lower + (upper - lower) * rest;
  };
  const q1 = quantile(0.25);
  const q3 = quantile(0.75);
  const median = quantile(0.5);
  const iqr = q3 - q1;
  const lowerFence = q1 - 1.5 * iqr;
  const upperFence = q3 + 1.5 * iqr;
  const lowerWhisker = sorted.find((value) => value >= lowerFence) ?? sorted[0];
  const upperWhisker = [...sorted].reverse().find((value) => value <= upperFence) ?? sorted[sorted.length - 1];
  const outliers = sorted.filter((value) => value < lowerWhisker || value > upperWhisker);
  return {
    stats: [lowerWhisker, q1, median, q3, upperWhisker],
    outliers,
    quartiles: { q1, q3, median },
    whiskers: { lower: lowerWhisker, upper: upperWhisker },
  };
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

const defaultOverlaySettings: OverlaySettings = {
  PRICE_MA: { periods: '5,10,20' },
  VOLUME_MA: { periods: '5,10' },
  BOLL: { period: 20, multiplier: 2 },
  KDJ: { period: 9, kSmoothing: 3, dSmoothing: 3 },
};

const cloneOverlaySettings = (): OverlaySettings => ({
  PRICE_MA: { ...defaultOverlaySettings.PRICE_MA },
  VOLUME_MA: { ...defaultOverlaySettings.VOLUME_MA },
  BOLL: { ...defaultOverlaySettings.BOLL },
  KDJ: { ...defaultOverlaySettings.KDJ },
});

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
  const [selectedOverlays, setSelectedOverlays] = useState<OverlayId[]>(['PRICE_MA', 'VOLUME_MA']);
  const [overlaySettings, setOverlaySettings] = useState<OverlaySettings>(() => cloneOverlaySettings());
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

  const handleOverlaySettingChange = <K extends OverlayId, Key extends keyof OverlaySettings[K]>(
    id: K,
    key: Key,
    value: OverlaySettings[K][Key]
  ) => {
    setOverlaySettings((prev) => ({
      ...prev,
      [id]: {
        ...prev[id],
        [key]: value,
      },
    }));
  };

  const restoreDefaultOverlays = () => {
    setSelectedOverlays(['PRICE_MA', 'VOLUME_MA']);
    setOverlaySettings(cloneOverlaySettings());
  };

  const renderOverlayControls = (optionId: OverlayId) => {
    switch (optionId) {
      case 'PRICE_MA':
        return (
          <label className="block text-xs text-slate-500">
            均线周期（逗号分隔）
            <input
              type="text"
              value={overlaySettings.PRICE_MA.periods}
              onChange={(event) => handleOverlaySettingChange('PRICE_MA', 'periods', event.target.value)}
              placeholder="例如：5,10,20"
              className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm text-slate-700 focus:border-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-100"
            />
            <span className="mt-1 block text-[11px] text-slate-400">支持一次性输入多个周期，系统将自动排序并生成多条均线。</span>
          </label>
        );
      case 'VOLUME_MA':
        return (
          <label className="block text-xs text-slate-500">
            均线周期（逗号分隔）
            <input
              type="text"
              value={overlaySettings.VOLUME_MA.periods}
              onChange={(event) => handleOverlaySettingChange('VOLUME_MA', 'periods', event.target.value)}
              placeholder="例如：5,10"
              className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm text-slate-700 focus:border-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-100"
            />
            <span className="mt-1 block text-[11px] text-slate-400">可组合不同窗口，以更精细地观察放量或缩量趋势。</span>
          </label>
        );
      case 'BOLL':
        return (
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="block text-xs text-slate-500">
              计算周期
              <input
                type="number"
                min={2}
                value={overlaySettings.BOLL.period}
                onChange={(event) =>
                  handleOverlaySettingChange('BOLL', 'period', Number(event.target.value) || overlaySettings.BOLL.period)
                }
                className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm text-slate-700 focus:border-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-100"
              />
            </label>
            <label className="block text-xs text-slate-500">
              标准差倍数
              <input
                type="number"
                step={0.1}
                min={0.5}
                value={overlaySettings.BOLL.multiplier}
                onChange={(event) =>
                  handleOverlaySettingChange('BOLL', 'multiplier', Number(event.target.value) || overlaySettings.BOLL.multiplier)
                }
                className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm text-slate-700 focus:border-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-100"
              />
            </label>
          </div>
        );
      case 'KDJ':
        return (
          <div className="grid gap-3 sm:grid-cols-3">
            <label className="block text-xs text-slate-500">
              计算周期
              <input
                type="number"
                min={2}
                value={overlaySettings.KDJ.period}
                onChange={(event) =>
                  handleOverlaySettingChange('KDJ', 'period', Number(event.target.value) || overlaySettings.KDJ.period)
                }
                className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm text-slate-700 focus:border-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-100"
              />
            </label>
            <label className="block text-xs text-slate-500">
              K 平滑系数
              <input
                type="number"
                min={1}
                value={overlaySettings.KDJ.kSmoothing}
                onChange={(event) =>
                  handleOverlaySettingChange('KDJ', 'kSmoothing', Number(event.target.value) || overlaySettings.KDJ.kSmoothing)
                }
                className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm text-slate-700 focus:border-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-100"
              />
            </label>
            <label className="block text-xs text-slate-500">
              D 平滑系数
              <input
                type="number"
                min={1}
                value={overlaySettings.KDJ.dSmoothing}
                onChange={(event) =>
                  handleOverlaySettingChange('KDJ', 'dSmoothing', Number(event.target.value) || overlaySettings.KDJ.dSmoothing)
                }
                className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm text-slate-700 focus:border-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-100"
              />
            </label>
          </div>
        );
      default:
        return null;
    }
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
    ? `截至 ${formatDate(metrics.current_size_as_of)} · ${metrics.current_size_source === 'total_netasset' ? '最新披露合计资产净值' : '最新披露资产净值'}（非实时）`
    : '净值数据暂未披露资产净值';

  const chartOption = useMemo(() => {
    if (!detail?.timeseries || detail.timeseries.length === 0) {
      return undefined;
    }

    const dates = detail.timeseries.map((item) => item.date);
    const totalPoints = dates.length;
    const defaultWindow = 252;
    const startIndex = Math.max(0, totalPoints - defaultWindow);
    const startValue = dates[startIndex];
    const endValue = dates[totalPoints - 1];
    const klineValues = detail.timeseries.map((item) => [item.open, item.close, item.low, item.high]);
    const volumes = detail.timeseries.map((item) => ({
      value: item.volume,
      itemStyle: {
        color: item.close >= item.open ? '#34d399' : '#f87171',
      },
    }));
    const closeValues = detail.timeseries.map((item) => item.close);
    const volumeValues = detail.timeseries.map((item) => item.volume);

    const priceMASeries = selectedOverlays.includes('PRICE_MA')
      ? (() => {
          const periods = Array.from(new Set(parsePeriods(overlaySettings.PRICE_MA.periods))).filter((num) => num > 1);
          return periods
            .sort((a, b) => a - b)
            .map((period) => ({
              name: `收盘价${period}日均线`,
              type: 'line',
              data: calculateMovingAverage(closeValues, period),
              smooth: true,
              showSymbol: false,
              lineStyle: { width: 1.5 },
              emphasis: { focus: 'series' },
            }));
        })()
      : [];

    const volumeMASeries = selectedOverlays.includes('VOLUME_MA')
      ? (() => {
          const periods = Array.from(new Set(parsePeriods(overlaySettings.VOLUME_MA.periods))).filter((num) => num > 1);
          return periods
            .sort((a, b) => a - b)
            .map((period) => ({
              name: `成交量${period}日均线`,
              type: 'line',
              xAxisIndex: 1,
              yAxisIndex: 1,
              data: calculateMovingAverage(volumeValues, period),
              smooth: true,
              showSymbol: false,
              lineStyle: { width: 1 },
              emphasis: { focus: 'series' },
            }));
        })()
      : [];

    const bollingerSeries = selectedOverlays.includes('BOLL')
      ? (() => {
          const period = Math.max(2, Math.round(overlaySettings.BOLL.period));
          const multiplier = Math.max(0.5, overlaySettings.BOLL.multiplier);
          const bands = calculateBollinger(closeValues, period, multiplier);
          return [
            {
              name: `布林上轨(${period}, ${multiplier.toFixed(1)}σ)`,
              type: 'line',
              data: bands.map((band) => band.upper),
              smooth: true,
              showSymbol: false,
              lineStyle: { width: 1, color: '#f97316' },
            },
            {
              name: '布林中轨',
              type: 'line',
              data: bands.map((band) => band.middle),
              smooth: true,
              showSymbol: false,
              lineStyle: { width: 1, color: '#0ea5e9', type: 'dashed' },
            },
            {
              name: '布林下轨',
              type: 'line',
              data: bands.map((band) => band.lower),
              smooth: true,
              showSymbol: false,
              lineStyle: { width: 1, color: '#10b981' },
            },
          ];
        })()
      : [];

    const hasKDJ = selectedOverlays.includes('KDJ');
    const kdjSettings = overlaySettings.KDJ;
    const kdjPeriod = Math.max(2, Math.round(kdjSettings.period));
    const kSmoothing = Math.max(1, Math.round(kdjSettings.kSmoothing));
    const dSmoothing = Math.max(1, Math.round(kdjSettings.dSmoothing));
    const { kValues, dValues, jValues } = hasKDJ
      ? calculateKDJ(detail.timeseries, kdjPeriod, kSmoothing, dSmoothing)
      : { kValues: [], dValues: [], jValues: [] };

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

    const series: any[] = [
      {
        name: '价格',
        type: 'candlestick',
        data: klineValues,
        itemStyle: {
          color: '#0ea5e9',
          color0: '#f87171',
          borderColor: '#0284c7',
          borderColor0: '#dc2626',
        },
      },
      {
        name: '成交量',
        type: 'bar',
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: volumes,
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
  }, [detail?.timeseries, selectedOverlays, overlaySettings]);

  const chartHeight = useMemo(() => {
    return selectedOverlays.includes('KDJ') ? 700 : 540;
  }, [selectedOverlays]);

  const statisticsWindow = useMemo(
    () => selectStatisticsWindow(detail?.timeseries ?? [], statisticsPeriod),
    [detail?.timeseries, statisticsPeriod],
  );

  const dailyReturns = useMemo(
    () => calculateDailyReturns(statisticsWindow.series),
    [statisticsWindow.series],
  );

  const dailyReturnValues = useMemo(() => dailyReturns.map((item) => item.return), [dailyReturns]);

  const returnStats = useMemo(() => calculateReturnStatistics(dailyReturnValues), [dailyReturnValues]);

  const histogramBins = useMemo(
    () => calculateHistogram(dailyReturnValues, histogramBinWidth),
    [dailyReturnValues, histogramBinWidth]
  );
  const histogramPdfValues = useMemo(
    () =>
      calculateNormalPdfCounts(
        returnStats.mean,
        returnStats.std,
        returnStats.sampleSize,
        histogramBins,
        histogramBinWidth
      ),
    [returnStats.mean, returnStats.std, returnStats.sampleSize, histogramBins, histogramBinWidth]
  );
  const boxPlotData = useMemo(() => calculateBoxPlot(dailyReturnValues), [dailyReturnValues]);
  const normalQqData = useMemo(() => buildNormalQqData(dailyReturnValues), [dailyReturnValues]);
  const normalQqTableRows = useMemo(() => {
    if (!normalQqData) {
      return [];
    }
    const targetPercentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99];
    return targetPercentiles
      .map((target) => normalQqData.points.reduce((closest, point) => (
        Math.abs(point.percentile - target) < Math.abs(closest.percentile - target) ? point : closest
      )))
      .filter((point, index, points) => points.findIndex((candidate) => candidate.percentile === point.percentile) === index);
  }, [normalQqData]);
  const skewnessInterpretation = useMemo(
    () => interpretSkewness(returnStats.skewness),
    [returnStats.skewness],
  );
  const kurtosisInterpretation = useMemo(
    () => interpretExcessKurtosis(returnStats.kurtosis),
    [returnStats.kurtosis],
  );
  const normalityConclusion = useMemo(() => {
    if (returnStats.normalityPValue === null || !Number.isFinite(returnStats.normalityPValue)) {
      return '样本不足，无法进行检验';
    }
    return returnStats.normalityPValue < 0.05 ? '拒绝正态假设（5% 显著性水平）' : '无法拒绝正态假设（5% 显著性水平）';
  }, [returnStats.normalityPValue]);

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
          if (histogramPdfValues.length === histogramBins.length) {
            lines.push(`正态拟合：${formatDecimal(histogramPdfValues[index], 2)} 天`);
          }
          if (returnStats.sampleSize > 0) {
            const frequency = bin.count / returnStats.sampleSize;
            lines.push(`频率：${formatDecimal(frequency, 3)}`);
          }
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
        ...(histogramPdfValues.length === histogramBins.length
          ? [
              {
                type: 'line',
                name: '正态拟合',
                data: histogramPdfValues.map((value) => Number(value.toFixed(2))),
                smooth: true,
                symbol: 'none',
                lineStyle: { width: 2, color: '#f97316' },
                areaStyle: { opacity: 0 },
              },
            ]
          : []),
      ],
    };
  }, [histogramBins, histogramPdfValues, returnStats.sampleSize]);

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

  const simulationInitialNav = FUTURE_SIMULATION_INITIAL_NAV;
  const simulationSeed = `${productId}-${statisticsPeriod}-${simulationHorizon}-${simulationPathCount}-${simulationRun}`;
  const parametricSimulation = useMemo(() => simulateParametricMonteCarlo({
    returnsPercent: dailyReturnValues,
    initialNav: simulationInitialNav,
    horizonDays: simulationHorizon,
    pathCount: simulationPathCount,
    targetReturnPercent: simulationTargetReturn,
    seed: `${simulationSeed}-parametric`,
  }), [
    dailyReturnValues,
    simulationHorizon,
    simulationInitialNav,
    simulationPathCount,
    simulationSeed,
    simulationTargetReturn,
  ]);
  const bootstrapSimulation = useMemo(() => simulateStationaryBlockBootstrap({
    returnsPercent: dailyReturnValues,
    initialNav: simulationInitialNav,
    horizonDays: simulationHorizon,
    pathCount: simulationPathCount,
    targetReturnPercent: simulationTargetReturn,
    averageBlockLength: bootstrapBlockLength,
    seed: `${simulationSeed}-bootstrap`,
  }), [
    bootstrapBlockLength,
    dailyReturnValues,
    simulationHorizon,
    simulationInitialNav,
    simulationPathCount,
    simulationSeed,
    simulationTargetReturn,
  ]);
  const activeSimulation = simulationMethod === 'parametric' ? parametricSimulation : bootstrapSimulation;
  const simulationComparison = useMemo(() => (
    parametricSimulation && bootstrapSimulation
      ? compareSimulations(parametricSimulation, bootstrapSimulation)
      : null
  ), [bootstrapSimulation, parametricSimulation]);
  const terminalNavDensity = useMemo(
    () => buildTerminalNavDensity(activeSimulation?.terminalValues ?? []),
    [activeSimulation],
  );

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
    const navValues = [
      ...activeSimulation.percentiles.p05,
      ...activeSimulation.percentiles.p95,
      terminalNavDensity.minNav,
      terminalNavDensity.maxNav,
    ];
    const observedMin = Math.min(...navValues);
    const observedMax = Math.max(...navValues);
    const navPadding = Math.max((observedMax - observedMin) * 0.04, Math.abs(observedMax) * 0.001, 0.0001);
    const navAxisMin = Math.max(0, observedMin - navPadding);
    const navAxisMax = observedMax + navPadding;
    const histogramBinWidth = Math.max(
      terminalNavDensity.histogram[0]?.upperNav - terminalNavDensity.histogram[0]?.lowerNav,
      Number.EPSILON,
    );
    const densityCountFactor = terminalNavDensity.sampleSize * histogramBinWidth;
    const densityCountPoints = terminalNavDensity.points.map((point) => ({
      nav: point.nav,
      count: point.density * densityCountFactor,
    }));
    const countAxisMax = Math.max(
      1,
      Math.ceil(
        Math.max(
          ...terminalNavDensity.histogram.map((bin) => bin.count),
          ...densityCountPoints.map((point) => point.count),
        ) * 1.08,
      ),
    );
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
          data: densityCountPoints.map((point) => [point.count, point.nav]),
          showSymbol: false,
          lineStyle: { width: 2, color: '#7c3aed' },
          emphasis: { disabled: true },
          z: 3,
          tooltip: {
            trigger: 'item',
            formatter: (params: any) => {
              const nav = Number(Array.isArray(params?.data) ? params.data[1] : Number.NaN);
              const estimatedCount = Number(Array.isArray(params?.data) ? params.data[0] : Number.NaN);
              const simulatedReturn = Number.isFinite(nav) ? nav / simulationInitialNav - 1 : Number.NaN;
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
  }, [activeSimulation, simulationHorizon, simulationInitialNav, terminalNavDensity]);

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
          onClick={() => returnToOrigin(navigate, location, `/research?kind=${productKind}`)}
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

          <section className="rounded-3xl border border-violet-100 bg-violet-50/40 p-6 shadow-sm" aria-labelledby="custom-research-indicators-title">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 id="custom-research-indicators-title" className="text-lg font-semibold text-slate-900">自定义研究指标</h2>
                <p className="mt-1 max-w-2xl text-sm text-slate-600">
                  使用工作区已保存的公式，基于该产品的真实净值计算研究指标。它们与下方仅用于图表叠加的 MA、BOLL、KDJ 技术辅助线相互独立。
                </p>
              </div>
              <Link
                to={`/indicator-studio?kind=${productKind}&ids=${encodeURIComponent(productId)}`}
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
                  <span className="h-2 w-2 rounded-full bg-sky-500" />K 线
                </span>
                <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-3 py-1">
                  <span className="h-2 w-2 rounded-full bg-emerald-400" />成交量
                </span>
              </div>
            </div>
            {chartOption ? (
              <ReactECharts option={chartOption} style={{ height: chartHeight }} notMerge lazyUpdate />
            ) : (
              <div className="h-[320px] rounded-2xl bg-slate-50 text-center text-slate-400">暂无可视化数据</div>
            )}
            <div className="rounded-2xl bg-slate-50 p-6">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h3 className="text-base font-semibold text-slate-900">自定义辅助线</h3>
                  <p className="text-sm text-slate-500">勾选需要叠加的技术指标，快速评估行情结构与量价关系。</p>
                </div>
                <button
                  type="button"
                  onClick={restoreDefaultOverlays}
                  className="inline-flex items-center justify-center rounded-full border border-slate-200 px-3 py-1 text-xs font-medium text-slate-500 transition hover:border-emerald-400 hover:text-emerald-600"
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
                      <button type="button" onClick={() => toggleOverlay(option.id)} className="flex items-center justify-between text-left">
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
                        当前共 {histogramBins.reduce((acc, bin) => acc + bin.count, 0)} 个样本，划分 {histogramBins.length} 个区间。
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
                            {formatSignedPercent(boxPlotData.quartiles.q3 - boxPlotData.quartiles.q1, 2)}
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
                                  <td className="px-2 py-2 text-slate-600">{(point.percentile * 100).toFixed(1)}%</td>
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
                                    <td className="px-3 py-3 text-right tabular-nums">{formatRatioPercent(simulation.terminal.p05 / simulationInitialNav - 1)}</td>
                                    <td className="px-3 py-3 text-right tabular-nums">{formatRatioPercent(simulation.terminal.p50 / simulationInitialNav - 1)}</td>
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
