import React, { useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';

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

interface ProductDetailResponse {
  product_id: string | null;
  name: string | null;
  management?: string | null;
  custodian?: string | null;
  status?: string | null;
  base_info: Record<string, string | number | null>;
  metrics: {
    issue_amount?: number | null;
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

const formatDateISO = (date: Date) => {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
};

const createSeededRandom = (seedText: string) => {
  let seed = 0;
  for (let index = 0; index < seedText.length; index += 1) {
    seed = (seed * 31 + seedText.charCodeAt(index)) >>> 0;
  }
  if (seed === 0) {
    seed = 1;
  }
  return () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 0x100000000;
  };
};

const generateMockTimeSeries = (seed: string, days = 180): TimeSeriesPoint[] => {
  const random = createSeededRandom(seed);
  const cursor = new Date();
  cursor.setHours(0, 0, 0, 0);
  const dates: Date[] = [];
  while (dates.length < days) {
    const day = cursor.getDay();
    if (day !== 0 && day !== 6) {
      dates.push(new Date(cursor));
    }
    cursor.setDate(cursor.getDate() - 1);
  }
  dates.reverse();

  const series: TimeSeriesPoint[] = [];
  let previousClose = 100 + random() * 20;

  dates.forEach((date) => {
    const drift = (random() - 0.45) * 0.04;
    const open = previousClose * (1 + (random() - 0.5) * 0.015);
    const close = Math.max(1, previousClose * (1 + drift));
    const high = Math.max(open, close) * (1 + random() * 0.012);
    const low = Math.min(open, close) * (1 - random() * 0.012);
    const volumeBase = 900000 + random() * 400000;
    series.push({
      date: formatDateISO(date),
      open: Number(open.toFixed(2)),
      close: Number(close.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      volume: Math.max(1, Math.round(volumeBase * (1 + ((close - previousClose) / previousClose) * 3))),
    });
    previousClose = close;
  });

  return series;
};

const generateMockProductDetail = (productId: string): ProductDetailResponse => {
  const normalizedId = productId || 'demo-etf';
  const nameSuffix = normalizedId.replace(/[^a-zA-Z0-9]/g, '').toUpperCase() || 'DEMO';
  const timeseries = generateMockTimeSeries(`${normalizedId}-series`);
  const random = createSeededRandom(`${normalizedId}-meta`);
  const issueAmount = Math.round((random() * 800000 + 200000) / 100) * 100; // 单位：万
  return {
    product_id: normalizedId,
    name: `示例ETF-${nameSuffix}`,
    management: '模拟资产管理有限公司',
    custodian: '示例银行股份有限公司',
    status: '存续',
    base_info: {
      ts_code: `${nameSuffix}.OF`,
      issue_date: '2021-01-15',
      benchmark: '沪深300指数',
      listing_exchange: '上海证券交易所',
      fund_type: 'ETF',
    },
    metrics: {
      issue_amount: issueAmount,
      m_fee: Number((random() * 0.3 + 0.2).toFixed(2)),
      c_fee: Number((random() * 0.1 + 0.05).toFixed(2)),
    },
    timeseries,
  };
};

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
      skewness = (n / ((n - 1) * (n - 2))) * skewNumerator;
    }
    if (n > 3) {
      const kurtNumerator = standardized.reduce((acc, cur) => acc + cur ** 4, 0);
      kurtosis =
        ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * kurtNumerator -
        (3 * (n - 1) ** 2) / ((n - 2) * (n - 3));
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
  const [detail, setDetail] = useState<ProductDetailResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedOverlays, setSelectedOverlays] = useState<OverlayId[]>(['PRICE_MA', 'VOLUME_MA']);
  const [overlaySettings, setOverlaySettings] = useState<OverlaySettings>(() => cloneOverlaySettings());
  const [histogramBinWidth, setHistogramBinWidth] = useState<number>(0.2);

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
      const hydrateWithMock = () => {
        const mockDetail = generateMockProductDetail(productId);
        setDetail(mockDetail);
        setError(null);
      };
      try {
        setLoading(true);
        setError(null);
        const resp = await fetch(`/api/etf/products/${encodeURIComponent(productId)}`, { signal: controller.signal });
        if (!resp.ok) {
          console.warn('Use mock product detail due to response status', resp.status);
          hydrateWithMock();
          return;
        }
        const data = (await resp.json()) as ProductDetailResponse;
        if (!data?.timeseries || data.timeseries.length === 0) {
          console.warn('Received product detail without timeseries, fallback to mock data');
          hydrateWithMock();
          return;
        }
        setDetail(data);
      } catch (err) {
        if ((err as DOMException).name === 'AbortError') {
          return;
        }
        console.error('Failed to load product detail', err);
        hydrateWithMock();
      } finally {
        setLoading(false);
      }
    };
    fetchDetail();
    return () => controller.abort();
  }, [productId]);

  const metrics = detail?.metrics ?? {};
  const baseInfo = detail?.base_info ?? {};
  const tsCode = baseInfo['ts_code'];

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

  const dailyReturns = useMemo(() => {
    return calculateDailyReturns(detail?.timeseries ?? []);
  }, [detail?.timeseries]);

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
  const normalityConclusion = useMemo(() => {
    if (!returnStats.normalityPValue || !Number.isFinite(returnStats.normalityPValue)) {
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
      grid: { left: '10%', right: '6%', bottom: 40, top: 30 },
      xAxis: {
        type: 'category',
        data: ['日收益率'],
        axisLabel: { color: '#475569' },
        axisTick: { show: false },
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
        },
        ...(boxPlotData.outliers.length > 0
          ? [
              {
                name: '离群值',
                type: 'scatter',
                data: boxPlotData.outliers.map((value) => [0, Number(value.toFixed(2))]),
                symbolSize: 8,
                itemStyle: { color: '#f97316' },
              },
            ]
          : []),
      ],
    };
  }, [boxPlotData]);

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
          onClick={() => window.open('/research', '_self')}
          className="inline-flex items-center rounded-full border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-600 shadow-sm hover:border-emerald-400 hover:text-emerald-600"
        >
          ← 返回
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
            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <div>
                <div className="text-sm font-semibold uppercase tracking-wide text-emerald-500">产品研究</div>
                <h1 className="mt-2 text-3xl font-bold text-slate-900">{detail.name ?? '--'}</h1>
                <div className="mt-2 flex flex-wrap gap-3 text-sm text-slate-500">
                  {tsCode && <span className="inline-flex rounded-full bg-emerald-50 px-3 py-1 text-emerald-600">{tsCode}</span>}
                  {detail.management && <span>管理人：{formatText(detail.management)}</span>}
                  {detail.custodian && <span>托管人：{formatText(detail.custodian)}</span>}
                  {detail.status && (
                    <span className="inline-flex items-center rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">
                      {formatText(detail.status)}
                    </span>
                  )}
                </div>
              </div>
              <div className="grid gap-4 sm:grid-cols-2">
                <MetricCard
                  title="发行规模"
                  value={formatIssueAmount(metrics.issue_amount)}
                  description="基于信息表披露的发行规模"
                />
                <MetricCard
                  title="管理 / 托管费"
                  value={`${formatPercent(metrics.m_fee)} / ${formatPercent(metrics.c_fee)}`}
                  description="产品费用率概览"
                />
              </div>
            </div>
          </section>

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
                <p className="text-sm text-slate-500">基于日度收盘价计算收益率，辅助评估分布特征与波动水平。</p>
              </div>
              {statisticsRange && (
                <div className="text-xs text-slate-500">
                  样本区间：{statisticsRange.start} ~ {statisticsRange.end}（共 {statisticsRange.count} 个交易日）
                </div>
              )}
            </div>
            {dailyReturns.length === 0 ? (
              <div className="rounded-2xl bg-slate-50 p-10 text-center text-slate-400">暂无足够的日度收益数据用于统计分析。</div>
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
                  <MetricCard
                    title="偏度"
                    value={formatSignedDecimal(returnStats.skewness)}
                    description="衡量分布左/右尾的偏移程度"
                  />
                  <MetricCard
                    title="峰度（超额）"
                    value={formatSignedDecimal(returnStats.kurtosis)}
                    description="评估尾部厚度与尖峰程度"
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
                <div className="rounded-2xl border border-slate-100 bg-slate-50/60 p-4">
                  <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <h3 className="text-sm font-semibold text-slate-900">箱形图</h3>
                      <p className="text-xs text-slate-500">观察中位数、分位区间与离群值</p>
                    </div>
                    {boxPlotData && (
                      <span className="text-xs text-slate-500">离群值：{boxPlotData.outliers.length} 个</span>
                    )}
                  </div>
                  <div className="mt-4">
                    {boxPlotOption ? (
                      <ReactECharts option={boxPlotOption} style={{ height: 260 }} notMerge lazyUpdate />
                    ) : (
                      <div className="h-[260px] rounded-2xl bg-white/60 text-center text-sm leading-[260px] text-slate-400">
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
                        <dt className="font-medium text-slate-500">箱体范围</dt>
                        <dd className="mt-1 text-sm font-semibold text-slate-900">
                          {formatSignedPercent(boxPlotData.whiskers.lower, 2)} ~ {formatSignedPercent(boxPlotData.whiskers.upper, 2)}
                        </dd>
                      </div>
                    </dl>
                  )}
                </div>
              </>
            )}
          </section>
        </>
      )}
    </div>
  );
}
