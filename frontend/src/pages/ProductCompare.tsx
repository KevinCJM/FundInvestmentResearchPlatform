import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';

interface TimeSeriesPoint {
  date: string;
  close: number;
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
    exp_return?: number | null;
  };
  timeseries: TimeSeriesPoint[];
}

const decimalFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 2 });

const formatPercent = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  return `${decimalFormatter.format(value)}%`;
};

const formatIssueAmount = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  if (Math.abs(value) >= 10000) {
    return `${decimalFormatter.format(value / 10000)} 亿`;
  }
  return `${decimalFormatter.format(value)} 万`;
};

const formatDate = (value?: string | number | null) => {
  if (value === null || value === undefined) {
    return '--';
  }
  const text = String(value).trim();
  return text.length === 0 ? '--' : text;
};

const formatText = (value?: string | number | null) => {
  if (value === null || value === undefined) {
    return '未知';
  }
  const text = String(value).trim();
  return text.length === 0 ? '未知' : text;
};

interface DerivedMetrics {
  cumulativeReturn: number | null;
  annualizedReturn: number | null;
  volatility: number | null;
  maxDrawdown: number | null;
  returnToFee: number | null;
  totalFee: number | null;
  sharpeRatio: number | null;
  calmarRatio: number | null;
}

type MetricColumn = {
  key: string;
  label: string;
  getValue: (context: { product: ProductDetailResponse; derived?: DerivedMetrics }) => number | null;
  render?: (
    value: number | null,
    context: { product: ProductDetailResponse; derived?: DerivedMetrics },
  ) => React.ReactNode;
  higherIsBetter?: boolean;
};

type RangePreset = '1M' | '3M' | '6M' | '1Y' | '3Y' | '5Y' | 'ALL' | 'CUSTOM';

interface RangeSelection {
  preset: RangePreset;
  customStart?: string;
  customEnd?: string;
}

interface RangeMeta {
  startIndex: number;
  endIndex: number;
  startDate: string | null;
  endDate: string | null;
  startTimestamp: number | null;
  endTimestamp: number | null;
  dates: string[];
}

const rangePresetOptions: { value: RangePreset; label: string }[] = [
  { value: '1M', label: '近 1 月' },
  { value: '3M', label: '近 3 月' },
  { value: '6M', label: '近 6 月' },
  { value: '1Y', label: '近 1 年' },
  { value: '3Y', label: '近 3 年' },
  { value: '5Y', label: '近 5 年' },
  { value: 'ALL', label: '全部' },
  { value: 'CUSTOM', label: '自定义' },
];

const METRIC_TABLE_VIEWPORT = {
  width: 960,
  height: 260,
} as const;

type EfficiencyQuadrantKey = 'sharpeRatio' | 'calmarRatio' | 'returnToFee';

type QuadrantMetricConfig = {
  key: EfficiencyQuadrantKey;
  label: string;
  metricLabel: string;
  metricAccessor: (context: { product: ProductDetailResponse; derived?: DerivedMetrics }) => number | null;
  metricFormatter: (value: number | null) => string;
  x: {
    label: string;
    accessor: (context: { product: ProductDetailResponse; derived?: DerivedMetrics }) => number | null;
    valueFormatter: (value: number | null) => string;
    axisFormatter: (value: number) => string;
    enforceNonNegative?: boolean;
  };
  y: {
    label: string;
    accessor: (context: { product: ProductDetailResponse; derived?: DerivedMetrics }) => number | null;
    valueFormatter: (value: number | null) => string;
    axisFormatter: (value: number) => string;
    enforceNonNegative?: boolean;
  };
};

const computeGradientStyles = (
  values: (number | null)[],
  higherIsBetter: boolean,
): React.CSSProperties[] => {
  const numericValues = values.filter((value): value is number => value !== null && Number.isFinite(value));
  if (numericValues.length === 0) {
    return values.map(() => ({}));
  }

  const min = Math.min(...numericValues);
  const max = Math.max(...numericValues);
  const range = max - min;

  if (Math.abs(range) < Number.EPSILON) {
    return values.map((value) =>
      value === null || !Number.isFinite(value)
        ? {}
        : {
            backgroundColor: 'hsl(160, 60%, 85%)',
            color: '#0f172a',
          },
    );
  }

  return values.map((value) => {
    if (value === null || !Number.isFinite(value)) {
      return {};
    }

    const normalized = Math.min(1, Math.max(0, (value - min) / range));
    const emphasis = higherIsBetter ? normalized : 1 - normalized;
    const hue = 10 + emphasis * 110; // red (10) -> green (120)
    const saturation = 68;
    const lightness = 92 - emphasis * 24; // keep high readability while increasing contrast

    return {
      backgroundColor: `hsl(${hue}, ${saturation}%, ${lightness}%)`,
      color: '#0f172a',
    };
  });
};

const efficiencyQuadrantConfigs: QuadrantMetricConfig[] = [
  {
    key: 'sharpeRatio',
    label: '夏普比率',
    metricLabel: '夏普比率',
    metricAccessor: ({ derived }) => derived?.sharpeRatio ?? null,
    metricFormatter: (value) => formatRatio(value),
    x: {
      label: '年化收益率 (%)',
      accessor: ({ derived }) => derived?.annualizedReturn ?? null,
      valueFormatter: (value) => formatPercent(value),
      axisFormatter: (value) => `${decimalFormatter.format(value)}%`,
    },
    y: {
      label: '年化波动率 (%)',
      accessor: ({ derived }) => derived?.volatility ?? null,
      valueFormatter: (value) => formatPercent(value),
      axisFormatter: (value) => `${decimalFormatter.format(value)}%`,
      enforceNonNegative: true,
    },
  },
  {
    key: 'calmarRatio',
    label: '卡玛比率',
    metricLabel: '卡玛比率',
    metricAccessor: ({ derived }) => derived?.calmarRatio ?? null,
    metricFormatter: (value) => formatRatio(value),
    x: {
      label: '年化收益率 (%)',
      accessor: ({ derived }) => derived?.annualizedReturn ?? null,
      valueFormatter: (value) => formatPercent(value),
      axisFormatter: (value) => `${decimalFormatter.format(value)}%`,
    },
    y: {
      label: '最大回撤幅度 (%)',
      accessor: ({ derived }) => {
        if (derived?.maxDrawdown === null || derived?.maxDrawdown === undefined) {
          return null;
        }
        return Math.abs(derived.maxDrawdown);
      },
      valueFormatter: (value) => formatPercent(value),
      axisFormatter: (value) => `${decimalFormatter.format(value)}%`,
      enforceNonNegative: true,
    },
  },
  {
    key: 'returnToFee',
    label: '收益费用比',
    metricLabel: '收益费用比',
    metricAccessor: ({ derived }) => derived?.returnToFee ?? null,
    metricFormatter: (value) => formatRatio(value, ' 倍'),
    x: {
      label: '累计收益率 (%)',
      accessor: ({ derived }) => derived?.cumulativeReturn ?? null,
      valueFormatter: (value) => formatPercent(value),
      axisFormatter: (value) => `${decimalFormatter.format(value)}%`,
    },
    y: {
      label: '管理费+托管费 (%)',
      accessor: ({ derived }) => derived?.totalFee ?? null,
      valueFormatter: (value) => formatPercent(value),
      axisFormatter: (value) => `${decimalFormatter.format(value)}%`,
      enforceNonNegative: true,
    },
  },
];

const generateSyntheticSeries = (
  startDate: string,
  length: number,
  options: { drift: number; volatility: number; seasonal?: number },
): TimeSeriesPoint[] => {
  const series: TimeSeriesPoint[] = [];
  const baseDate = new Date(startDate);
  const day = new Date(baseDate);
  let value = 1;
  let index = 0;
  const volatility = options.volatility;
  const seasonal = options.seasonal ?? 0.0025;

  while (series.length < length) {
    const weekDay = day.getDay();
    if (weekDay !== 0 && weekDay !== 6) {
      const wave = Math.sin(index / 7) * volatility + Math.cos(index / 13) * seasonal;
      const step = options.drift + wave;
      value = Math.max(0.2, value * (1 + step));
      series.push({ date: day.toISOString().slice(0, 10), close: Number(value.toFixed(4)) });
      index += 1;
    }
    day.setDate(day.getDate() + 1);
  }

  return series;
};

const DEMO_PRODUCTS: ProductDetailResponse[] = [
  {
    product_id: 'DEMO50',
    name: '虚拟稳健 50',
    management: '示例资产管理',
    custodian: '示例银行',
    status: '存续',
    base_info: {
      ts_code: 'DEMO50',
      code: 'DV0001',
      fund_type: '指数型',
      type: '公募基金',
      invest_type: '宽基指数',
      market: '上交所',
      management: '示例资产管理',
      custodian: '示例银行',
      status: '存续',
      benchmark: '示例宽基指数',
      list_date: '2019-02-15',
      found_date: '2018-12-20',
    },
    metrics: {
      issue_amount: 92000,
      m_fee: 0.45,
      c_fee: 0.1,
      exp_return: 12.3,
    },
    timeseries: generateSyntheticSeries('2019-01-02', 720, { drift: 0.00038, volatility: 0.0028 }),
  },
  {
    product_id: 'DEMOQH',
    name: '虚拟量化优选',
    management: '前瞻量化',
    custodian: '华夏银行',
    status: '存续',
    base_info: {
      ts_code: 'DEMOQH',
      code: 'DV0002',
      fund_type: '指数增强',
      type: '公募基金',
      invest_type: '量化增强',
      market: '深交所',
      management: '前瞻量化',
      custodian: '华夏银行',
      status: '存续',
      benchmark: '成长风格指数',
      list_date: '2020-04-10',
      found_date: '2020-03-05',
    },
    metrics: {
      issue_amount: 68000,
      m_fee: 0.65,
      c_fee: 0.12,
      exp_return: 15.6,
    },
    timeseries: generateSyntheticSeries('2020-01-06', 620, { drift: 0.00052, volatility: 0.0038, seasonal: 0.003 }),
  },
  {
    product_id: 'DEMOTECH',
    name: '虚拟科创先锋',
    management: '远见成长',
    custodian: '招商银行',
    status: '存续',
    base_info: {
      ts_code: 'DEMOTECH',
      code: 'DV0003',
      fund_type: '主题指数',
      type: '公募基金',
      invest_type: '科技创新',
      market: '上交所',
      management: '远见成长',
      custodian: '招商银行',
      status: '存续',
      benchmark: '科创精选指数',
      list_date: '2021-01-15',
      found_date: '2020-12-22',
    },
    metrics: {
      issue_amount: 54000,
      m_fee: 0.75,
      c_fee: 0.13,
      exp_return: 18.9,
    },
    timeseries: generateSyntheticSeries('2020-07-01', 520, { drift: 0.00065, volatility: 0.0046, seasonal: 0.0035 }),
  },
  {
    product_id: 'DEMOVALUE',
    name: '虚拟价值优选',
    management: '恒远价值',
    custodian: '中信银行',
    status: '存续',
    base_info: {
      ts_code: 'DEMOVALUE',
      code: 'DV0004',
      fund_type: '价值型',
      type: '公募基金',
      invest_type: '蓝筹价值',
      market: '上交所',
      management: '恒远价值',
      custodian: '中信银行',
      status: '存续',
      benchmark: '蓝筹价值指数',
      list_date: '2018-06-28',
      found_date: '2018-05-18',
    },
    metrics: {
      issue_amount: 83000,
      m_fee: 0.5,
      c_fee: 0.1,
      exp_return: 11.5,
    },
    timeseries: generateSyntheticSeries('2018-01-03', 780, { drift: 0.00032, volatility: 0.0025 }),
  },
  {
    product_id: 'DEMOESG',
    name: '虚拟绿能先锋',
    management: '绿色未来',
    custodian: '建设银行',
    status: '存续',
    base_info: {
      ts_code: 'DEMOESG',
      code: 'DV0005',
      fund_type: '主题指数',
      type: '公募基金',
      invest_type: '新能源',
      market: '深交所',
      management: '绿色未来',
      custodian: '建设银行',
      status: '存续',
      benchmark: '碳中和产业指数',
      list_date: '2020-09-22',
      found_date: '2020-08-17',
    },
    metrics: {
      issue_amount: 61000,
      m_fee: 0.7,
      c_fee: 0.12,
      exp_return: 17.8,
    },
    timeseries: generateSyntheticSeries('2020-05-06', 600, { drift: 0.00058, volatility: 0.0042, seasonal: 0.0038 }),
  },
  {
    product_id: 'DEMOMED',
    name: '虚拟医疗成长',
    management: '康泰成长',
    custodian: '浦发银行',
    status: '存续',
    base_info: {
      ts_code: 'DEMOMED',
      code: 'DV0006',
      fund_type: '行业指数',
      type: '公募基金',
      invest_type: '医疗健康',
      market: '上交所',
      management: '康泰成长',
      custodian: '浦发银行',
      status: '存续',
      benchmark: '医药创新指数',
      list_date: '2019-11-08',
      found_date: '2019-10-10',
    },
    metrics: {
      issue_amount: 57000,
      m_fee: 0.72,
      c_fee: 0.12,
      exp_return: 16.4,
    },
    timeseries: generateSyntheticSeries('2019-03-04', 680, { drift: 0.00055, volatility: 0.004, seasonal: 0.0032 }),
  },
  {
    product_id: 'DEMOCONS',
    name: '虚拟消费升级',
    management: '远航消费',
    custodian: '民生银行',
    status: '存续',
    base_info: {
      ts_code: 'DEMOCONS',
      code: 'DV0007',
      fund_type: '消费主题',
      type: '公募基金',
      invest_type: '消费升级',
      market: '深交所',
      management: '远航消费',
      custodian: '民生银行',
      status: '存续',
      benchmark: '新消费领先指数',
      list_date: '2018-04-12',
      found_date: '2018-03-05',
    },
    metrics: {
      issue_amount: 76000,
      m_fee: 0.55,
      c_fee: 0.1,
      exp_return: 13.8,
    },
    timeseries: generateSyntheticSeries('2018-01-08', 800, { drift: 0.00042, volatility: 0.0031 }),
  },
  {
    product_id: 'DEMOGLOBAL',
    name: '虚拟全球配置',
    management: '环球配置',
    custodian: '花旗银行',
    status: '存续',
    base_info: {
      ts_code: 'DEMOGLOBAL',
      code: 'DV0008',
      fund_type: 'QDII',
      type: '公募基金',
      invest_type: '全球多资产',
      market: '上交所',
      management: '环球配置',
      custodian: '花旗银行',
      status: '存续',
      benchmark: '全球资产配置指数',
      list_date: '2017-09-18',
      found_date: '2017-08-21',
    },
    metrics: {
      issue_amount: 88000,
      m_fee: 0.6,
      c_fee: 0.14,
      exp_return: 10.6,
    },
    timeseries: generateSyntheticSeries('2017-01-03', 920, { drift: 0.00028, volatility: 0.0022 }),
  },
  {
    product_id: 'DEMOALPHA',
    name: '虚拟多策略 Alpha',
    management: '量化多策略',
    custodian: '兴业银行',
    status: '存续',
    base_info: {
      ts_code: 'DEMOALPHA',
      code: 'DV0009',
      fund_type: '多策略',
      type: '公募基金',
      invest_type: '对冲增强',
      market: '深交所',
      management: '量化多策略',
      custodian: '兴业银行',
      status: '存续',
      benchmark: '绝对收益指数',
      list_date: '2021-05-20',
      found_date: '2021-04-15',
    },
    metrics: {
      issue_amount: 45000,
      m_fee: 0.85,
      c_fee: 0.15,
      exp_return: 14.2,
    },
    timeseries: generateSyntheticSeries('2021-01-04', 460, { drift: 0.00048, volatility: 0.0036, seasonal: 0.0028 }),
  },
  {
    product_id: 'DEMOFUTURE',
    name: '虚拟创新驱动',
    management: '前沿创新',
    custodian: '光大银行',
    status: '存续',
    base_info: {
      ts_code: 'DEMOFUTURE',
      code: 'DV0010',
      fund_type: '混合型',
      type: '公募基金',
      invest_type: '前沿科技',
      market: '上交所',
      management: '前沿创新',
      custodian: '光大银行',
      status: '存续',
      benchmark: '未来产业指数',
      list_date: '2022-02-18',
      found_date: '2022-01-20',
    },
    metrics: {
      issue_amount: 52000,
      m_fee: 0.78,
      c_fee: 0.13,
      exp_return: 19.4,
    },
    timeseries: generateSyntheticSeries('2021-09-01', 380, { drift: 0.00068, volatility: 0.0048, seasonal: 0.0039 }),
  },
];

const parseDateToTimestamp = (value?: string | null) => {
  if (!value) {
    return null;
  }
  const timestamp = new Date(value).getTime();
  return Number.isNaN(timestamp) ? null : timestamp;
};

const getPresetStartTimestamp = (endTimestamp: number, preset: RangePreset): number | null => {
  const end = new Date(endTimestamp);
  switch (preset) {
    case '1M': {
      const next = new Date(end);
      next.setMonth(next.getMonth() - 1);
      return next.getTime();
    }
    case '3M': {
      const next = new Date(end);
      next.setMonth(next.getMonth() - 3);
      return next.getTime();
    }
    case '6M': {
      const next = new Date(end);
      next.setMonth(next.getMonth() - 6);
      return next.getTime();
    }
    case '1Y': {
      const next = new Date(end);
      next.setFullYear(next.getFullYear() - 1);
      return next.getTime();
    }
    case '3Y': {
      const next = new Date(end);
      next.setFullYear(next.getFullYear() - 3);
      return next.getTime();
    }
    case '5Y': {
      const next = new Date(end);
      next.setFullYear(next.getFullYear() - 5);
      return next.getTime();
    }
    case 'ALL':
      return Number.NEGATIVE_INFINITY;
    case 'CUSTOM':
    default:
      return null;
  }
};

const computeRangeMeta = (
  selection: RangeSelection,
  dates: string[],
  timestamps: (number | null)[],
): RangeMeta => {
  const lastIndex = dates.length - 1;
  if (lastIndex < 0) {
    return {
      startIndex: 0,
      endIndex: -1,
      startDate: null,
      endDate: null,
      startTimestamp: null,
      endTimestamp: null,
      dates: [],
    };
  }

  const validTimestamps = timestamps.filter((value): value is number => value !== null);
  const minTimestamp = validTimestamps.length ? Math.min(...validTimestamps) : null;
  const maxTimestamp = validTimestamps.length ? Math.max(...validTimestamps) : null;

  let resolvedEndTimestamp = maxTimestamp;
  if (selection.preset === 'CUSTOM') {
    const parsedEnd = parseDateToTimestamp(selection.customEnd);
    if (parsedEnd !== null) {
      if (minTimestamp !== null && maxTimestamp !== null) {
        resolvedEndTimestamp = Math.min(Math.max(parsedEnd, minTimestamp), maxTimestamp);
      } else {
        resolvedEndTimestamp = parsedEnd;
      }
    }
  }

  if (resolvedEndTimestamp === null) {
    resolvedEndTimestamp = maxTimestamp;
  }

  let resolvedStartTimestamp = minTimestamp;
  if (selection.preset === 'CUSTOM') {
    const parsedStart = parseDateToTimestamp(selection.customStart);
    if (parsedStart !== null && resolvedEndTimestamp !== null) {
      if (minTimestamp !== null && maxTimestamp !== null) {
        resolvedStartTimestamp = Math.min(Math.max(parsedStart, minTimestamp), resolvedEndTimestamp);
      } else {
        resolvedStartTimestamp = Math.min(parsedStart, resolvedEndTimestamp);
      }
    }
  } else if (selection.preset !== 'ALL') {
    if (resolvedEndTimestamp !== null) {
      const presetStart = getPresetStartTimestamp(resolvedEndTimestamp, selection.preset);
      if (presetStart !== null) {
        if (minTimestamp !== null) {
          resolvedStartTimestamp = Math.max(presetStart, minTimestamp);
        } else {
          resolvedStartTimestamp = presetStart;
        }
      }
    }
  }

  if (
    resolvedStartTimestamp !== null &&
    resolvedEndTimestamp !== null &&
    resolvedStartTimestamp > resolvedEndTimestamp
  ) {
    resolvedStartTimestamp = resolvedEndTimestamp;
  }

  let startIndex = 0;
  if (resolvedStartTimestamp !== null) {
    for (let i = 0; i < timestamps.length; i += 1) {
      const ts = timestamps[i];
      if (ts === null) {
        startIndex = i;
        break;
      }
      if (ts >= resolvedStartTimestamp) {
        startIndex = i;
        break;
      }
    }
  }

  let endIndex = lastIndex;
  if (resolvedEndTimestamp !== null) {
    for (let i = timestamps.length - 1; i >= 0; i -= 1) {
      const ts = timestamps[i];
      if (ts === null) {
        endIndex = i;
        break;
      }
      if (ts <= resolvedEndTimestamp) {
        endIndex = i;
        break;
      }
    }
  }

  if (endIndex < startIndex) {
    endIndex = startIndex;
  }

  const selectedDates = dates.slice(startIndex, endIndex + 1);
  const startTimestamp = timestamps[startIndex] ?? parseDateToTimestamp(selectedDates[0]);
  const endTimestamp = timestamps[endIndex] ?? parseDateToTimestamp(selectedDates[selectedDates.length - 1]);

  return {
    startIndex,
    endIndex,
    startDate: selectedDates[0] ?? null,
    endDate: selectedDates[selectedDates.length - 1] ?? null,
    startTimestamp: startTimestamp ?? null,
    endTimestamp: endTimestamp ?? null,
    dates: selectedDates,
  };
};

interface RangeSelectorProps {
  id: string;
  selection: RangeSelection;
  onChange: (next: RangeSelection) => void;
  minDate: string | null;
  maxDate: string | null;
  resolvedStart: string | null;
  resolvedEnd: string | null;
}

const RangeSelector: React.FC<RangeSelectorProps> = ({
  id,
  selection,
  onChange,
  minDate,
  maxDate,
  resolvedStart,
  resolvedEnd,
}) => {
  return (
    <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500">
      <label htmlFor={id} className="font-medium text-slate-600">
        显示区间
      </label>
      <select
        id={id}
        value={selection.preset}
        onChange={(event) => {
          const preset = event.target.value as RangePreset;
          if (preset === 'CUSTOM') {
            onChange({ preset, customStart: selection.customStart, customEnd: selection.customEnd });
          } else {
            onChange({ preset, customStart: selection.customStart, customEnd: selection.customEnd });
          }
        }}
        className="h-9 rounded border border-slate-200 bg-white px-3 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
      >
        {rangePresetOptions.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      {selection.preset === 'CUSTOM' && (
        <>
          <input
            type="date"
            value={selection.customStart ?? ''}
            min={minDate ?? undefined}
            max={maxDate ?? undefined}
            onChange={(event) => {
              const value = event.target.value;
              onChange({
                preset: 'CUSTOM',
                customStart: value || undefined,
                customEnd: selection.customEnd,
              });
            }}
            className="h-9 rounded border border-slate-200 px-3 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
          />
          <span className="text-slate-400">至</span>
          <input
            type="date"
            value={selection.customEnd ?? ''}
            min={minDate ?? undefined}
            max={maxDate ?? undefined}
            onChange={(event) => {
              const value = event.target.value;
              onChange({
                preset: 'CUSTOM',
                customStart: selection.customStart,
                customEnd: value || undefined,
              });
            }}
            className="h-9 rounded border border-slate-200 px-3 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
          />
        </>
      )}
      <span className="text-slate-400">
        当前：{resolvedStart ?? '--'} ~ {resolvedEnd ?? '--'}
      </span>
      {minDate && maxDate && (
        <span className="text-slate-300">(可选范围：{minDate} - {maxDate})</span>
      )}
    </div>
  );
};

const MS_IN_DAY = 86_400_000;
const TRADING_DAYS_PER_YEAR = 252;
const DEFAULT_ROLLING_WINDOW_DAYS = 30;
const ROLLING_WINDOW_MIN = 5;
const ROLLING_WINDOW_MAX = 120;
const ROLLING_WINDOW_STEP = 1;

const computeDerivedMetrics = (
  product: ProductDetailResponse,
  range?: { startTimestamp?: number | null; endTimestamp?: number | null },
): DerivedMetrics => {
  const points = (product.timeseries ?? [])
    .filter((point): point is TimeSeriesPoint =>
      Boolean(point?.date) && Number.isFinite(point?.close ?? NaN),
    )
    .map((point) => {
      const date = String(point.date);
      const timestamp = new Date(date).getTime();
      return {
        date,
        close: Number(point.close),
        timestamp: Number.isNaN(timestamp) ? null : timestamp,
      };
    })
    .sort((a, b) => {
      if (a.timestamp !== null && b.timestamp !== null) {
        return a.timestamp - b.timestamp;
      }
      return a.date.localeCompare(b.date);
    });

  const filteredPoints = points.filter((point) => {
    if (range?.startTimestamp !== undefined && range?.startTimestamp !== null && point.timestamp !== null) {
      if (point.timestamp < range.startTimestamp) {
        return false;
      }
    }
    if (range?.endTimestamp !== undefined && range?.endTimestamp !== null && point.timestamp !== null) {
      if (point.timestamp > range.endTimestamp) {
        return false;
      }
    }
    return true;
  });

  if (filteredPoints.length === 0) {
    return {
      cumulativeReturn: null,
      annualizedReturn: null,
      volatility: null,
      maxDrawdown: null,
      returnToFee: null,
      totalFee: null,
      sharpeRatio: null,
      calmarRatio: null,
    };
  }

  let baseValue: number | null = null;
  for (const point of filteredPoints) {
    if (Number.isFinite(point.close) && point.close > 0) {
      baseValue = point.close;
      break;
    }
  }
  if (baseValue === null) {
    const fallback = filteredPoints[0]?.close;
    baseValue = fallback !== undefined && fallback > 0 ? fallback : null;
  }

  if (baseValue === null) {
    return {
      cumulativeReturn: null,
      annualizedReturn: null,
      volatility: null,
      maxDrawdown: null,
      returnToFee: null,
      totalFee: null,
      sharpeRatio: null,
      calmarRatio: null,
    };
  }

  const normalized = filteredPoints.map((point) => point.close / baseValue!);
  const lastClose = filteredPoints[filteredPoints.length - 1]?.close ?? null;
  const cumulativeReturn =
    lastClose !== null && Number.isFinite(lastClose)
      ? (lastClose / baseValue - 1) * 100
      : null;

  let annualizedReturn: number | null = null;
  const firstTimestamp = filteredPoints[0]?.timestamp;
  const lastTimestamp = filteredPoints[filteredPoints.length - 1]?.timestamp;
  if (
    cumulativeReturn !== null &&
    firstTimestamp !== null &&
    lastTimestamp !== null &&
    lastTimestamp > firstTimestamp
  ) {
    const diffDays = (lastTimestamp - firstTimestamp) / MS_IN_DAY;
    if (diffDays > 0) {
      const years = diffDays / 365;
      if (years > 0) {
        const totalGrowth = 1 + cumulativeReturn / 100;
        annualizedReturn = (Math.pow(totalGrowth, 1 / years) - 1) * 100;
      }
    }
  }

  const dailyReturns: number[] = [];
  for (let i = 1; i < filteredPoints.length; i += 1) {
    const prev = filteredPoints[i - 1];
    const current = filteredPoints[i];
    if (prev.close > 0 && current.close > 0) {
      dailyReturns.push(current.close / prev.close - 1);
    }
  }

  let volatility: number | null = null;
  if (dailyReturns.length > 1) {
    const mean = dailyReturns.reduce((acc, value) => acc + value, 0) / dailyReturns.length;
    const variance =
      dailyReturns.reduce((acc, value) => acc + (value - mean) ** 2, 0) / (dailyReturns.length - 1);
    const dailyVol = Math.sqrt(Math.max(variance, 0));
    volatility = dailyVol * Math.sqrt(TRADING_DAYS_PER_YEAR) * 100;
  }

  let maxDrawdown: number | null = null;
  if (normalized.length > 1) {
    let peak = -Infinity;
    let minDrawdown = 0;
    normalized.forEach((value) => {
      if (!Number.isFinite(value)) {
        return;
      }
      peak = Math.max(peak, value);
      if (peak <= 0) {
        return;
      }
      const drawdown = value / peak - 1;
      if (drawdown < minDrawdown) {
        minDrawdown = drawdown;
      }
    });
    maxDrawdown = minDrawdown * 100;
  }

  const managementFee = Number.isFinite(product.metrics?.m_fee ?? NaN)
    ? (product.metrics?.m_fee as number)
    : null;
  const custodyFee = Number.isFinite(product.metrics?.c_fee ?? NaN)
    ? (product.metrics?.c_fee as number)
    : null;
  const totalFee =
    managementFee === null && custodyFee === null
      ? null
      : (managementFee ?? 0) + (custodyFee ?? 0);

  let returnToFee: number | null = null;
  if (totalFee !== null && totalFee !== 0 && cumulativeReturn !== null) {
    returnToFee = cumulativeReturn / totalFee;
  }

  let sharpeRatio: number | null = null;
  if (annualizedReturn !== null && volatility !== null && volatility !== 0) {
    const annualizedReturnDecimal = annualizedReturn / 100;
    const volatilityDecimal = volatility / 100;
    if (volatilityDecimal !== 0) {
      sharpeRatio = annualizedReturnDecimal / volatilityDecimal;
    }
  }

  let calmarRatio: number | null = null;
  if (annualizedReturn !== null && maxDrawdown !== null && maxDrawdown !== 0) {
    const annualizedReturnDecimal = annualizedReturn / 100;
    const drawdownMagnitude = Math.abs(maxDrawdown) / 100;
    if (drawdownMagnitude > 0) {
      calmarRatio = annualizedReturnDecimal / drawdownMagnitude;
    }
  }

  return {
    cumulativeReturn,
    annualizedReturn,
    volatility,
    maxDrawdown,
    returnToFee,
    totalFee,
    sharpeRatio,
    calmarRatio,
  };
};

const formatRatio = (value?: number | null, suffix = '') => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  return `${decimalFormatter.format(value)}${suffix}`;
};

export default function ProductCompare() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [products, setProducts] = useState<ProductDetailResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [failedIds, setFailedIds] = useState<string[]>([]);
  const [performanceRange, setPerformanceRange] = useState<RangeSelection>({ preset: '1Y' });
  const [riskRange, setRiskRange] = useState<RangeSelection>({ preset: '1Y' });
  const [efficiencyRange, setEfficiencyRange] = useState<RangeSelection>({ preset: '1Y' });
  const [performanceZoomRange, setPerformanceZoomRange] = useState<{ start: number; end: number } | null>(null);
  const [riskZoomRange, setRiskZoomRange] = useState<{ start: number; end: number } | null>(null);
  const [efficiencyQuadrantKey, setEfficiencyQuadrantKey] = useState<EfficiencyQuadrantKey>('sharpeRatio');
  const [rollingWindowDays, setRollingWindowDays] = useState(DEFAULT_ROLLING_WINDOW_DAYS);
  const [rollingWindowInputValue, setRollingWindowInputValue] = useState<string>(
    String(DEFAULT_ROLLING_WINDOW_DAYS),
  );

  const idsFromQuery = useMemo(() => {
    const rawParam = searchParams.get('ids') ?? '';
    const rawTokens = rawParam
      .split(',')
      .map((item) => item.trim())
      .filter((item) => item.length > 0);
    const decoded = rawTokens.map((token) => {
      try {
        return decodeURIComponent(token);
      } catch (err) {
        console.warn('Failed to decode product id from query', token, err);
        return token;
      }
    });
    const unique = Array.from(new Set(decoded));
    return unique;
  }, [searchParams]);

  const previewMode = searchParams.get('preview') === 'demo';
  const limitedIds = useMemo(() => (previewMode ? [] : idsFromQuery.slice(0, 10)), [idsFromQuery, previewMode]);
  const truncatedIds = !previewMode && idsFromQuery.length > limitedIds.length;
  const hasRemoteIds = limitedIds.length > 0;

  useEffect(() => {
    if (previewMode) {
      setProducts(DEMO_PRODUCTS);
      setFailedIds([]);
      setError(null);
      setLoading(false);
      return;
    }
    if (!hasRemoteIds) {
      setProducts([]);
      setFailedIds([]);
      setError(null);
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        setFailedIds([]);
        const results = await Promise.allSettled(
          limitedIds.map(async (id) => {
            const resp = await fetch(`/api/etf/products/${encodeURIComponent(id)}`, { signal: controller.signal });
            if (!resp.ok) {
              throw new Error(`无法获取产品 ${id} 的详情`);
            }
            const data = (await resp.json()) as ProductDetailResponse;
            return data;
          }),
        );
        if (controller.signal.aborted) {
          return;
        }
        const succeeded: ProductDetailResponse[] = [];
        const failed: string[] = [];
        results.forEach((result, index) => {
          if (result.status === 'fulfilled') {
            succeeded.push(result.value);
          } else {
            const reason = result.reason as { name?: string } | undefined;
            if (reason?.name === 'AbortError') {
              return;
            }
            failed.push(limitedIds[index]);
          }
        });
        setProducts(succeeded);
        setFailedIds(failed);
        if (succeeded.length === 0) {
          setError('未能加载任何产品详情，请返回重新选择。');
        }
      } catch (err) {
        if ((err as DOMException).name === 'AbortError') {
          return;
        }
        console.error('Failed to load product comparisons', err);
        setError('加载产品对比数据时出现问题，请稍后重试。');
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    };
    fetchData();
    return () => controller.abort();
  }, [hasRemoteIds, limitedIds, previewMode]);

  const detailColumns = useMemo(
    () => [
      {
        key: 'code',
        label: '产品代码',
        render: (product: ProductDetailResponse) =>
          (product.base_info?.ts_code as string) ?? (product.base_info?.code as string) ?? product.product_id ?? '--',
      },
      {
        key: 'fund_type',
        label: '基金类型',
        render: (product: ProductDetailResponse) => formatText(product.base_info?.fund_type),
      },
      {
        key: 'type',
        label: '机构类型',
        render: (product: ProductDetailResponse) => formatText(product.base_info?.type),
      },
      {
        key: 'invest_type',
        label: '投资风格',
        render: (product: ProductDetailResponse) => formatText(product.base_info?.invest_type),
      },
      {
        key: 'market',
        label: '交易市场',
        render: (product: ProductDetailResponse) => formatText(product.base_info?.market),
      },
      {
        key: 'management',
        label: '管理人',
        render: (product: ProductDetailResponse) => formatText(product.base_info?.management ?? product.management),
      },
      {
        key: 'custodian',
        label: '托管人',
        render: (product: ProductDetailResponse) => formatText(product.base_info?.custodian ?? product.custodian),
      },
      {
        key: 'status',
        label: '产品状态',
        render: (product: ProductDetailResponse) => formatText(product.base_info?.status ?? product.status),
      },
      {
        key: 'issue_amount',
        label: '发行规模',
        render: (product: ProductDetailResponse) => formatIssueAmount(product.metrics?.issue_amount),
      },
      {
        key: 'fee',
        label: '管理费 / 托管费',
        render: (product: ProductDetailResponse) =>
          `${formatPercent(product.metrics?.m_fee)} · ${formatPercent(product.metrics?.c_fee)}`,
      },
      {
        key: 'benchmark',
        label: '业绩基准',
        render: (product: ProductDetailResponse) => formatText(product.base_info?.benchmark),
      },
      {
        key: 'list_date',
        label: '上市日',
        render: (product: ProductDetailResponse) => formatDate(product.base_info?.list_date),
      },
      {
        key: 'found_date',
        label: '成立日',
        render: (product: ProductDetailResponse) => formatDate(product.base_info?.found_date ?? product.base_info?.issue_date),
      },
      {
        key: 'recent_nav',
        label: '最新虚拟净值',
        render: (product: ProductDetailResponse) => {
          if (!product.timeseries || product.timeseries.length === 0) {
            return '--';
          }
          const latest = product.timeseries[product.timeseries.length - 1];
          return Number.isFinite(latest.close) ? decimalFormatter.format(latest.close) : '--';
        },
      },
    ],
    [],
  );

  const productPresentations = useMemo(
    () =>
      products.map((product, index) => {
        const candidateKey =
          (typeof product.product_id === 'string' && product.product_id.trim().length > 0 && product.product_id.trim()) ||
          (typeof product.base_info?.ts_code === 'string' && product.base_info.ts_code.trim().length > 0 && product.base_info.ts_code.trim()) ||
          (typeof product.base_info?.code === 'string' && product.base_info.code.trim().length > 0 && product.base_info.code.trim()) ||
          (typeof product.name === 'string' && product.name.trim().length > 0 && product.name.trim()) ||
          `product-${index}`;
        const key = `${candidateKey}-${index}`;
        const displayName =
          (typeof product.name === 'string' && product.name.trim().length > 0 && product.name.trim()) ||
          (typeof product.product_id === 'string' && product.product_id.trim().length > 0 && product.product_id.trim()) ||
          `产品${index + 1}`;
        const code =
          (product.base_info?.ts_code as string) ??
          (product.base_info?.code as string) ??
          product.product_id ??
          '--';
        return { key, product, displayName, code };
      }),
    [products],
  );

  const performanceColumns = useMemo<MetricColumn[]>(
    () => [
      {
        key: 'cumulativeReturn',
        label: '累计收益率',
        getValue: ({ derived }) => derived?.cumulativeReturn ?? null,
        render: (value) => formatPercent(value),
      },
      {
        key: 'annualizedReturn',
        label: '年化收益率',
        getValue: ({ derived }) => derived?.annualizedReturn ?? null,
        render: (value) => formatPercent(value),
      },
      {
        key: 'expectedReturn',
        label: '预期收益率',
        getValue: ({ product }) => (product.metrics?.exp_return as number | null) ?? null,
        render: (value) => formatPercent(value),
      },
    ],
    [],
  );

  const riskColumns = useMemo<MetricColumn[]>(
    () => [
      {
        key: 'volatility',
        label: '年化波动率',
        getValue: ({ derived }) => derived?.volatility ?? null,
        render: (value) => formatPercent(value),
        higherIsBetter: false,
      },
      {
        key: 'maxDrawdown',
        label: '最大回撤',
        getValue: ({ derived }) => derived?.maxDrawdown ?? null,
        render: (value) => formatPercent(value),
        higherIsBetter: false,
      },
    ],
    [],
  );

  const efficiencyColumns = useMemo<MetricColumn[]>(
    () => [
      {
        key: 'sharpeRatio',
        label: '夏普比率',
        getValue: ({ derived }) => derived?.sharpeRatio ?? null,
        render: (value) => formatRatio(value),
      },
      {
        key: 'calmarRatio',
        label: '卡玛比率',
        getValue: ({ derived }) => derived?.calmarRatio ?? null,
        render: (value) => formatRatio(value),
      },
      {
        key: 'returnToFee',
        label: '收益费用比',
        getValue: ({ derived }) => derived?.returnToFee ?? null,
        render: (value) => formatRatio(value, ' 倍'),
      },
    ],
    [],
  );

  const chartSource = useMemo(() => {
    if (!products.length) {
      return { allDates: [] as string[], sources: [] as { name: string; map: Map<string, number> }[] };
    }
    const dateSet = new Set<string>();
    const sources = products.map((product) => {
      const name = product.name ?? product.product_id ?? '未知产品';
      const map = new Map<string, number>();
      product.timeseries?.forEach((point) => {
        if (point.date && Number.isFinite(point.close)) {
          dateSet.add(point.date);
          map.set(point.date, point.close);
        }
      });
      return { name, map };
    });
    const allDates = Array.from(dateSet).sort();
    return { allDates, sources };
  }, [products]);

  const { allDates, sources } = chartSource;

  const dateTimestamps = useMemo(
    () =>
      allDates.map((date) => {
        const timestamp = new Date(date).getTime();
        return Number.isNaN(timestamp) ? null : timestamp;
      }),
    [allDates],
  );

  const performanceRangeMeta = useMemo(
    () => computeRangeMeta(performanceRange, allDates, dateTimestamps),
    [performanceRange, allDates, dateTimestamps],
  );

  const riskRangeMeta = useMemo(
    () => computeRangeMeta(riskRange, allDates, dateTimestamps),
    [riskRange, allDates, dateTimestamps],
  );

  const efficiencyRangeMeta = useMemo(
    () => computeRangeMeta(efficiencyRange, allDates, dateTimestamps),
    [efficiencyRange, allDates, dateTimestamps],
  );

  const renderMetricTable = useCallback(
    (columns: MetricColumn[], metricsByProduct: Map<string, DerivedMetrics>, tableKey: string) => (
      <table className="min-w-[720px] w-max divide-y divide-slate-100">
        <thead className="bg-slate-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">指标</th>
            {productPresentations.map(({ key, displayName, code }) => (
              <th key={`${tableKey}-head-${key}`} className="px-6 py-3 text-center text-xs font-semibold uppercase tracking-wider text-emerald-600">
                <div className="flex flex-col items-center gap-0.5">
                  <span className="text-sm font-semibold">{displayName}</span>
                  <span className="text-[11px] font-normal text-emerald-500">{code}</span>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100 bg-white">
          {columns.map((column) => {
            const values = productPresentations.map(({ key, product }) => {
              const derived = metricsByProduct.get(key);
              return column.getValue({ product, derived });
            });
            const gradientStyles = computeGradientStyles(values, column.higherIsBetter !== false);
            return (
              <tr key={`${tableKey}-row-${column.key}`} className="hover:bg-emerald-50/20">
                <th scope="row" className="whitespace-nowrap px-6 py-4 text-left text-sm font-semibold text-slate-600">
                  {column.label}
                </th>
                {productPresentations.map(({ key, product }, index) => {
                  const derived = metricsByProduct.get(key);
                  const rawValue = values[index];
                  const content = column.render
                    ? column.render(rawValue, { product, derived })
                    : rawValue;
                  const displayContent =
                    typeof content === 'number' || typeof content === 'string'
                      ? content
                      : content ?? '--';
                  const title =
                    typeof displayContent === 'string'
                      ? displayContent
                      : typeof rawValue === 'number'
                      ? column.render
                        ? undefined
                        : decimalFormatter.format(rawValue)
                      : undefined;
                  return (
                    <td
                      key={`${tableKey}-${column.key}-${key}`}
                      className="px-6 py-4 text-center text-sm font-medium text-slate-700 transition-colors"
                      style={gradientStyles[index] ?? {}}
                      title={title}
                    >
                      {displayContent}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    ),
    [productPresentations],
  );

  const performanceMetricsByProduct = useMemo(() => {
    const map = new Map<string, DerivedMetrics>();
    productPresentations.forEach(({ key, product }) => {
      map.set(
        key,
        computeDerivedMetrics(product, {
          startTimestamp: performanceRangeMeta.startTimestamp,
          endTimestamp: performanceRangeMeta.endTimestamp,
        }),
      );
    });
    return map;
  }, [performanceRangeMeta.endTimestamp, performanceRangeMeta.startTimestamp, productPresentations]);

  const riskMetricsByProduct = useMemo(() => {
    const map = new Map<string, DerivedMetrics>();
    productPresentations.forEach(({ key, product }) => {
      map.set(
        key,
        computeDerivedMetrics(product, {
          startTimestamp: riskRangeMeta.startTimestamp,
          endTimestamp: riskRangeMeta.endTimestamp,
        }),
      );
    });
    return map;
  }, [productPresentations, riskRangeMeta.endTimestamp, riskRangeMeta.startTimestamp]);

  const efficiencyMetricsByProduct = useMemo(() => {
    const map = new Map<string, DerivedMetrics>();
    productPresentations.forEach(({ key, product }) => {
      map.set(
        key,
        computeDerivedMetrics(product, {
          startTimestamp: efficiencyRangeMeta.startTimestamp,
          endTimestamp: efficiencyRangeMeta.endTimestamp,
        }),
      );
    });
    return map;
  }, [efficiencyRangeMeta.endTimestamp, efficiencyRangeMeta.startTimestamp, productPresentations]);

  const efficiencyQuadrantChartOption = useMemo(() => {
    const config =
      efficiencyQuadrantConfigs.find((item) => item.key === efficiencyQuadrantKey) ??
      efficiencyQuadrantConfigs[0];

    if (!config) {
      return null;
    }

    const points = productPresentations
      .map(({ key, displayName, code, product }) => {
        const derived = efficiencyMetricsByProduct.get(key);
        const xValue = config.x.accessor({ product, derived });
        const yValue = config.y.accessor({ product, derived });
        const metricValue = config.metricAccessor({ product, derived });
        return {
          name: displayName,
          code,
          x: typeof xValue === 'number' && Number.isFinite(xValue) ? xValue : null,
          y: typeof yValue === 'number' && Number.isFinite(yValue) ? yValue : null,
          metric: typeof metricValue === 'number' && Number.isFinite(metricValue) ? metricValue : null,
        };
      })
      .filter((point): point is { name: string; code: string; x: number; y: number; metric: number | null } =>
        point.x !== null && point.y !== null,
      );

    if (points.length === 0) {
      return null;
    }

    const computeBounds = (values: number[], enforceNonNegative?: boolean) => {
      const minRaw = Math.min(...values);
      const maxRaw = Math.max(...values);
      let padding = (maxRaw - minRaw) * 0.1;
      if (!Number.isFinite(padding) || Math.abs(padding) < 1e-6) {
        const magnitude = Math.max(Math.abs(maxRaw), Math.abs(minRaw));
        padding = magnitude === 0 ? 1 : magnitude * 0.1;
      }
      let min = minRaw - padding;
      let max = maxRaw + padding;
      if (enforceNonNegative) {
        min = Math.max(0, Math.min(min, minRaw));
      }
      if (!Number.isFinite(min)) {
        min = enforceNonNegative ? 0 : 0;
      }
      if (!Number.isFinite(max)) {
        max = enforceNonNegative ? Math.max(1, maxRaw) : Math.max(1, maxRaw);
      }
      if (max <= min) {
        const base = Math.max(Math.abs(maxRaw), Math.abs(minRaw), 1);
        const delta = base * 0.5;
        min = enforceNonNegative ? Math.max(0, minRaw - delta) : minRaw - delta;
        max = min + delta * 2;
      }
      if (enforceNonNegative && min < 0) {
        min = 0;
      }
      return { min, max };
    };

    const xValues = points.map((point) => point.x);
    const yValues = points.map((point) => point.y);
    const xBounds = computeBounds(xValues, config.x.enforceNonNegative);
    const yBounds = computeBounds(yValues, config.y.enforceNonNegative);

    const xBaseline = points.reduce((acc, point) => acc + point.x, 0) / points.length;
    const yBaseline = points.reduce((acc, point) => acc + point.y, 0) / points.length;

    const scatterData = points.map((point) => ({
      value: [point.x, point.y],
      name: point.name,
      code: point.code,
      metric: point.metric,
    }));

    return {
      color: ['#10b981'],
      title: {
        text: `${config.label}收益风险象限`,
        left: 0,
        top: 8,
        textStyle: {
          fontSize: 14,
          fontWeight: 600,
          color: '#0f172a',
        },
      },
      grid: { left: 60, right: 36, top: 64, bottom: 56 },
      tooltip: {
        trigger: 'item',
        backgroundColor: 'rgba(15, 23, 42, 0.92)',
        borderWidth: 0,
        textStyle: { color: '#f8fafc' },
        formatter: (params: any) => {
          const data = params?.data as { name: string; code: string; value: [number, number]; metric: number | null };
          if (!data) {
            return '';
          }
          const [xValue, yValue] = data.value ?? [];
          return [
            `<div class="text-sm font-semibold">${data.name} <span class="text-xs font-normal text-emerald-200">${data.code}</span></div>`,
            `<div class="text-xs text-slate-200">${config.metricLabel}：${config.metricFormatter(data.metric)}</div>`,
            `<div class="text-xs text-slate-200">${config.x.label}：${config.x.valueFormatter(
              typeof xValue === 'number' ? xValue : null,
            )}</div>`,
            `<div class="text-xs text-slate-200">${config.y.label}：${config.y.valueFormatter(
              typeof yValue === 'number' ? yValue : null,
            )}</div>`,
          ].join('');
        },
      },
      xAxis: {
        type: 'value',
        name: config.x.label,
        nameLocation: 'middle',
        nameGap: 36,
        min: xBounds.min,
        max: xBounds.max,
        axisLabel: {
          formatter: (value: number) => config.x.axisFormatter(value),
        },
        splitLine: {
          lineStyle: { type: 'dashed', color: '#cbd5f5' },
        },
      },
      yAxis: {
        type: 'value',
        name: config.y.label,
        nameLocation: 'middle',
        nameGap: 42,
        min: yBounds.min,
        max: yBounds.max,
        axisLabel: {
          formatter: (value: number) => config.y.axisFormatter(value),
        },
        splitLine: {
          lineStyle: { type: 'dashed', color: '#cbd5f5' },
        },
      },
      series: [
        {
          type: 'scatter',
          symbolSize: 16,
          itemStyle: {
            color: '#10b981',
            borderColor: '#047857',
            borderWidth: 1,
          },
          emphasis: {
            focus: 'self',
            scale: 1.15,
          },
          data: scatterData,
        },
      ],
      markLine: {
        silent: true,
        symbol: ['none', 'none'],
        lineStyle: {
          type: 'dashed',
          color: '#94a3b8',
        },
        data: [
          { xAxis: xBaseline },
          { yAxis: yBaseline },
        ],
      },
    };
  }, [efficiencyMetricsByProduct, efficiencyQuadrantKey, productPresentations]);

  const performanceDates = performanceRangeMeta.dates;
  const riskDates = riskRangeMeta.dates;

  const globalMinDate = allDates[0] ?? null;
  const globalMaxDate = allDates[allDates.length - 1] ?? null;

  const riskTimelineLength = riskDates.length;
  const rollingWindowMax =
    riskTimelineLength > 1
      ? Math.max(2, Math.min(ROLLING_WINDOW_MAX, riskTimelineLength - 1))
      : ROLLING_WINDOW_MAX;
  const rollingWindowMin = Math.max(2, Math.min(ROLLING_WINDOW_MIN, rollingWindowMax));
  const effectiveRollingWindowDays = Math.min(
    Math.max(Math.round(rollingWindowDays), rollingWindowMin),
    rollingWindowMax,
  );

  useEffect(() => {
    if (effectiveRollingWindowDays !== rollingWindowDays) {
      setRollingWindowDays(effectiveRollingWindowDays);
      setRollingWindowInputValue(String(effectiveRollingWindowDays));
    }
  }, [effectiveRollingWindowDays, rollingWindowDays]);

  useEffect(() => {
    setRollingWindowInputValue(String(rollingWindowDays));
  }, [rollingWindowDays]);

  useEffect(() => {
    setPerformanceZoomRange(null);
  }, [performanceRangeMeta.startDate, performanceRangeMeta.endDate]);

  useEffect(() => {
    setRiskZoomRange(null);
  }, [riskRangeMeta.startDate, riskRangeMeta.endDate]);

  const comparisonChartOption = useMemo(() => {
    if (!sources.length || !performanceDates.length) {
      return undefined;
    }

    const lastIndex = performanceDates.length - 1;
    const clamp = (value: number) => Math.min(Math.max(value, 0), lastIndex);
    const startIndex = performanceZoomRange ? clamp(performanceZoomRange.start) : 0;
    const endIndex = performanceZoomRange ? clamp(performanceZoomRange.end) : lastIndex;
    const visibleDates = performanceDates.slice(startIndex, endIndex + 1);
    const denominator = lastIndex > 0 ? lastIndex : 1;
    const startPercent = lastIndex > 0 ? (startIndex / denominator) * 100 : 0;
    const endPercent = lastIndex > 0 ? (endIndex / denominator) * 100 : 100;

    const series = sources.map(({ name, map }) => {
      let baseValue = 1;
      for (const date of visibleDates) {
        const candidate = map.get(date);
        if (candidate !== undefined && candidate !== null && Number.isFinite(candidate)) {
          baseValue = candidate === 0 ? 1 : candidate;
          break;
        }
      }
      const data = performanceDates.map((date) => {
        const value = map.get(date);
        if (value === undefined || value === null || !Number.isFinite(value)) {
          return null;
        }
        return Number((value / baseValue).toFixed(4));
      });
      return {
        name,
        type: 'line' as const,
        smooth: true,
        showSymbol: false,
        data,
      };
    });

    const yValues: number[] = [];
    series.forEach(({ data }) => {
      data.forEach((value, index) => {
        if (value !== null && Number.isFinite(value) && index >= startIndex && index <= endIndex) {
          yValues.push(value);
        }
      });
    });

    let yMin: number | undefined;
    let yMax: number | undefined;
    if (yValues.length) {
      const min = Math.min(...yValues);
      const max = Math.max(...yValues);
      if (min === max) {
        const offset = min === 0 ? 0.5 : Math.abs(min) * 0.1;
        yMin = min - offset;
        yMax = max + offset;
      } else {
        const padding = (max - min) * 0.1;
        yMin = min - padding;
        yMax = max + padding;
      }
    }

    if (yMin !== undefined && yMin < 0) {
      yMin = 0;
    }

    return {
      grid: { left: 48, right: 24, top: 40, bottom: 64 },
      tooltip: {
        trigger: 'axis',
        valueFormatter: (value: number | string | null) => {
          if (value === null || value === undefined) {
            return '--';
          }
          const num = Number(value);
          if (Number.isNaN(num)) {
            return value;
          }
          return decimalFormatter.format(num);
        },
      },
      legend: {
        top: 0,
      },
      xAxis: {
        type: 'category',
        data: performanceDates,
        axisLabel: { rotate: 45 },
      },
      yAxis: {
        type: 'value',
        min: yMin,
        max: yMax,
        axisLabel: {
          formatter: (value: number) => decimalFormatter.format(value),
        },
      },
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: 0,
          filterMode: 'none',
          start: startPercent,
          end: endPercent,
        },
        {
          type: 'slider',
          xAxisIndex: 0,
          filterMode: 'none',
          height: 28,
          bottom: 16,
          showDetail: false,
          start: startPercent,
          end: endPercent,
        },
      ],
      series,
    };
  }, [performanceDates, performanceZoomRange, sources]);

  const drawdownChartOption = useMemo(() => {
    if (!sources.length || !riskDates.length) {
      return undefined;
    }

    const lastIndex = riskDates.length - 1;
    const clamp = (value: number) => Math.min(Math.max(value, 0), lastIndex);
    const startIndex = riskZoomRange ? clamp(riskZoomRange.start) : 0;
    const endIndex = riskZoomRange ? clamp(riskZoomRange.end) : lastIndex;
    const visibleDates = riskDates.slice(startIndex, endIndex + 1);
    const denominator = lastIndex > 0 ? lastIndex : 1;
    const startPercent = lastIndex > 0 ? (startIndex / denominator) * 100 : 0;
    const endPercent = lastIndex > 0 ? (endIndex / denominator) * 100 : 100;

    let hasSeries = false;
    const series = sources.map(({ name, map }) => {
      const findBaseValue = (dates: string[]) => {
        for (const date of dates) {
          const candidate = map.get(date);
          if (
            candidate !== undefined &&
            candidate !== null &&
            Number.isFinite(candidate) &&
            candidate > 0
          ) {
            return candidate;
          }
        }
        return null;
      };

      const baseValue = findBaseValue(visibleDates) ?? findBaseValue(riskDates);
      if (baseValue === null || !Number.isFinite(baseValue) || baseValue <= 0) {
        return {
          name,
          type: 'line' as const,
          smooth: true,
          showSymbol: false,
          data: riskDates.map(() => null as number | null),
        };
      }

      let peak = 1;
      const data = riskDates.map((date, index) => {
        const value = map.get(date);
        if (
          value === undefined ||
          value === null ||
          !Number.isFinite(value) ||
          value <= 0 ||
          index < startIndex
        ) {
          return null;
        }

        const normalized = value / baseValue;
        if (!Number.isFinite(normalized) || normalized <= 0) {
          return null;
        }

        peak = Math.max(peak, normalized);
        if (peak <= 0) {
          return null;
        }

        const drawdown = (normalized / peak - 1) * 100;
        if (!Number.isFinite(drawdown)) {
          return null;
        }

        hasSeries = true;
        return Number(drawdown.toFixed(2));
      });
      return {
        name,
        type: 'line' as const,
        smooth: true,
        showSymbol: false,
        data,
      };
    });

    if (!hasSeries) {
      return undefined;
    }

    const yValues: number[] = [];
    series.forEach(({ data }) => {
      data.forEach((value, index) => {
        if (value !== null && Number.isFinite(value) && index >= startIndex && index <= endIndex) {
          yValues.push(value);
        }
      });
    });

    let yMin: number | undefined;
    let yMax: number | undefined;
    if (yValues.length) {
      const min = Math.min(...yValues);
      const max = Math.max(...yValues);
      if (max <= 0) {
        const padding = Math.max(Math.abs(min) * 0.1, 0.5);
        yMin = min - padding;
        yMax = 0;
      } else {
        const padding = (max - min) * 0.1 || 1;
        yMin = min - padding;
        yMax = max + padding;
      }
    }

    return {
      grid: { left: 48, right: 24, top: 40, bottom: 64 },
      tooltip: {
        trigger: 'axis',
        valueFormatter: (value: number | string | null) => {
          if (value === null || value === undefined) {
            return '--';
          }
          const num = Number(value);
          if (Number.isNaN(num)) {
            return value;
          }
          return formatPercent(num);
        },
      },
      legend: {
        top: 0,
      },
      xAxis: {
        type: 'category',
        data: riskDates,
        axisLabel: { rotate: 45 },
      },
      yAxis: {
        type: 'value',
        min: yMin,
        max: yMax,
        axisLabel: {
          formatter: (value: number) => formatPercent(value),
        },
      },
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: 0,
          filterMode: 'none',
          start: startPercent,
          end: endPercent,
        },
        {
          type: 'slider',
          xAxisIndex: 0,
          filterMode: 'none',
          height: 28,
          bottom: 16,
          showDetail: false,
          start: startPercent,
          end: endPercent,
        },
      ],
      series,
    };
  }, [riskDates, riskZoomRange, sources]);

  const rollingVolatilityChartOption = useMemo(() => {
    if (!sources.length || riskDates.length < 2) {
      return undefined;
    }

    const windowSize = effectiveRollingWindowDays;
    const lastIndex = riskDates.length - 1;
    const clamp = (value: number) => Math.min(Math.max(value, 0), lastIndex);
    const startIndex = riskZoomRange ? clamp(riskZoomRange.start) : 0;
    const endIndex = riskZoomRange ? clamp(riskZoomRange.end) : lastIndex;
    const denominator = lastIndex > 0 ? lastIndex : 1;
    const startPercent = lastIndex > 0 ? (startIndex / denominator) * 100 : 0;
    const endPercent = lastIndex > 0 ? (endIndex / denominator) * 100 : 100;

    let hasSeries = false;
    const series = sources.map(({ name, map }) => {
      const values = riskDates.map((date) => {
        const value = map.get(date);
        if (value === undefined || value === null || !Number.isFinite(value) || value <= 0) {
          return null;
        }
        return Number(value);
      });

      const returns = values.map((value, index) => {
        if (index === 0) {
          return null;
        }
        const prev = values[index - 1];
        if (value === null || prev === null || prev <= 0) {
          return null;
        }
        return value / prev - 1;
      });

      const data = riskDates.map((_, index) => {
        if (index === 0) {
          return null;
        }
        const start = Math.max(1, index - windowSize + 1);
        const windowReturns: number[] = [];
        for (let i = start; i <= index; i += 1) {
          const value = returns[i];
          if (value !== null && Number.isFinite(value)) {
            windowReturns.push(value);
          }
        }
        if (windowReturns.length < 2) {
          return null;
        }
        const mean = windowReturns.reduce((acc, value) => acc + value, 0) / windowReturns.length;
        const variance =
          windowReturns.reduce((acc, value) => acc + (value - mean) ** 2, 0) /
          (windowReturns.length - 1);
        const dailyVol = Math.sqrt(Math.max(variance, 0));
        const annualizedVol = dailyVol * Math.sqrt(TRADING_DAYS_PER_YEAR) * 100;
        if (!Number.isFinite(annualizedVol)) {
          return null;
        }
        hasSeries = true;
        return Number(annualizedVol.toFixed(2));
      });

      return {
        name,
        type: 'line' as const,
        smooth: true,
        showSymbol: false,
        data,
      };
    });

    if (!hasSeries) {
      return undefined;
    }

    const yValues: number[] = [];
    series.forEach(({ data }) => {
      data.forEach((value, index) => {
        if (value !== null && Number.isFinite(value) && index >= startIndex && index <= endIndex) {
          yValues.push(value);
        }
      });
    });

    let yMin: number | undefined;
    let yMax: number | undefined;
    if (yValues.length) {
      const min = Math.min(...yValues);
      const max = Math.max(...yValues);
      if (min === max) {
        const offset = min === 0 ? 5 : Math.abs(min) * 0.1;
        yMin = Math.max(min - offset, 0);
        yMax = max + offset;
      } else {
        const padding = (max - min) * 0.1;
        yMin = Math.max(min - padding, 0);
        yMax = max + padding;
      }
    }

    return {
      grid: { left: 48, right: 24, top: 40, bottom: 64 },
      tooltip: {
        trigger: 'axis',
        valueFormatter: (value: number | string | null) => {
          if (value === null || value === undefined) {
            return '--';
          }
          const num = Number(value);
          if (Number.isNaN(num)) {
            return value;
          }
          return formatPercent(num);
        },
      },
      legend: {
        top: 0,
      },
      xAxis: {
        type: 'category',
        data: riskDates,
        axisLabel: { rotate: 45 },
      },
      yAxis: {
        type: 'value',
        min: yMin,
        max: yMax,
        axisLabel: {
          formatter: (value: number) => formatPercent(value),
        },
      },
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: 0,
          filterMode: 'none',
          start: startPercent,
          end: endPercent,
        },
        {
          type: 'slider',
          xAxisIndex: 0,
          filterMode: 'none',
          height: 28,
          bottom: 16,
          showDetail: false,
          start: startPercent,
          end: endPercent,
        },
      ],
      series,
    };
  }, [effectiveRollingWindowDays, riskDates, riskZoomRange, sources]);

  const handlePerformanceDataZoom = useCallback(
    (event: any) => {
      if (!performanceDates.length) {
        return;
      }
      const payload = Array.isArray(event?.batch) ? event.batch[0] : event;
      if (!payload) {
        return;
      }
      const lastIndex = performanceDates.length - 1;
      const clamp = (value: number) => Math.min(Math.max(Math.round(value), 0), lastIndex);
      const resolveIndex = (value: unknown, percent: unknown) => {
        if (typeof value === 'string') {
          const idx = performanceDates.indexOf(value);
          if (idx !== -1) {
            return idx;
          }
        }
        if (typeof value === 'number' && Number.isFinite(value)) {
          return clamp(value);
        }
        if (typeof percent === 'number' && Number.isFinite(percent)) {
          return clamp(((performanceDates.length - 1) * percent) / 100);
        }
        return undefined;
      };

      const startIndex = resolveIndex(payload.startValue, payload.start);
      const endIndex = resolveIndex(payload.endValue, payload.end);
      if (startIndex === undefined && endIndex === undefined) {
        return;
      }

      const nextRange = {
        start: clamp(startIndex ?? 0),
        end: clamp(endIndex ?? lastIndex),
      };

      setPerformanceZoomRange((prev) => {
        if (prev && prev.start === nextRange.start && prev.end === nextRange.end) {
          return prev;
        }
        return nextRange;
      });
    },
    [performanceDates],
  );

  const handleRiskDataZoom = useCallback(
    (event: any) => {
      if (!riskDates.length) {
        return;
      }
      const payload = Array.isArray(event?.batch) ? event.batch[0] : event;
      if (!payload) {
        return;
      }
      const lastIndex = riskDates.length - 1;
      const clamp = (value: number) => Math.min(Math.max(Math.round(value), 0), lastIndex);
      const resolveIndex = (value: unknown, percent: unknown) => {
        if (typeof value === 'string') {
          const idx = riskDates.indexOf(value);
          if (idx !== -1) {
            return idx;
          }
        }
        if (typeof value === 'number' && Number.isFinite(value)) {
          return clamp(value);
        }
        if (typeof percent === 'number' && Number.isFinite(percent)) {
          return clamp(((riskDates.length - 1) * percent) / 100);
        }
        return undefined;
      };

      const startIndex = resolveIndex(payload.startValue, payload.start);
      const endIndex = resolveIndex(payload.endValue, payload.end);
      if (startIndex === undefined && endIndex === undefined) {
        return;
      }

      const nextRange = {
        start: clamp(startIndex ?? 0),
        end: clamp(endIndex ?? lastIndex),
      };

      setRiskZoomRange((prev) => {
        if (prev && prev.start === nextRange.start && prev.end === nextRange.end) {
          return prev;
        }
        return nextRange;
      });
    },
    [riskDates],
  );

  const performanceChartEvents = useMemo(
    () => ({ dataZoom: handlePerformanceDataZoom }),
    [handlePerformanceDataZoom],
  );

  const riskChartEvents = useMemo(() => ({ dataZoom: handleRiskDataZoom }), [handleRiskDataZoom]);

  const shouldShowEmptyState = !previewMode && !hasRemoteIds;
  const shouldShowLoading = !previewMode && hasRemoteIds && loading;
  const shouldShowError = !previewMode && hasRemoteIds && !loading && !!error;
  const showContent = products.length > 0 && !shouldShowLoading && !shouldShowError;

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">产品对比</h1>
          <p className="mt-2 text-base text-slate-600">
            对比已选 ETF 产品的基本信息、费用与虚拟净值走势，辅助判断配置优先级。
          </p>
          {truncatedIds && (
            <p className="mt-1 text-xs text-amber-600">
              已自动截取前 10 个产品参与对比，其余产品请返回筛选页调整选择。
            </p>
          )}
          {previewMode && (
            <p className="mt-1 text-xs text-emerald-600">
              当前展示为虚拟示例数据，便于预览页面布局与交互效果。
            </p>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => navigate(-1)}
            className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium text-slate-600 hover:border-emerald-400 hover:text-emerald-600"
          >
            返回上一页
          </button>
          <Link
            to="/research"
            className="rounded-lg bg-emerald-500 px-3 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-emerald-600"
          >
            回到产品研究
          </Link>
        </div>
      </div>

      {shouldShowEmptyState && (
        <div className="rounded-2xl bg-white p-10 text-center text-slate-500 shadow-sm ring-1 ring-slate-100">
          <p>未能加载任何产品详情。</p>
          <div className="mt-6 flex justify-center">
            <Link
              to="/product-compare?preview=demo"
              className="rounded-lg bg-emerald-500 px-4 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-emerald-600"
            >
              查看虚拟产品预览
            </Link>
          </div>
        </div>
      )}

      {shouldShowLoading && (
        <div className="rounded-2xl bg-white p-10 text-center text-slate-400 shadow-sm ring-1 ring-slate-100">
          <div className="flex items-center justify-center gap-3">
            <svg className="h-5 w-5 animate-spin text-emerald-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle className="opacity-25" cx="12" cy="12" r="10" />
              <path className="opacity-75" d="M4 12a8 8 0 018-8" />
            </svg>
            数据加载中...
          </div>
        </div>
      )}

      {shouldShowError && (
        <div className="rounded-2xl bg-white p-10 text-center text-rose-500 shadow-sm ring-1 ring-rose-100">
          {error}
        </div>
      )}

      {showContent && (
        <div className="space-y-8">
          {failedIds.length > 0 && (
            <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700">
              部分产品未能成功加载：{failedIds.join('、')}。
            </div>
          )}

          <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-100">
            <div className="border-b border-slate-100 px-6 py-4">
              <h2 className="text-lg font-semibold text-slate-900">基础信息对比</h2>
              <p className="mt-1 text-xs text-slate-500">按产品展示核心要素，便于快速识别发行与运作差异。</p>
            </div>
            <div className="h-[420px] w-full overflow-auto px-6 pb-6 pt-4">
              <table className="min-w-[1100px] w-max divide-y divide-slate-100">
                <thead className="bg-slate-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">产品</th>
                    {detailColumns.map((column) => (
                      <th key={column.key} className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                        {column.label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 bg-white">
                  {productPresentations.map(({ key, product, displayName, code }) => (
                    <tr key={`basic-${key}`} className="hover:bg-emerald-50/40">
                      <td className="px-6 py-4 text-sm font-semibold text-emerald-600">
                        <div className="flex flex-col">
                          <span>{displayName}</span>
                          <span className="text-xs font-normal text-emerald-500">{code}</span>
                        </div>
                      </td>
                      {detailColumns.map((column) => (
                        <td key={`${column.key}-${key}`} className="px-6 py-4 text-sm text-slate-600">
                          {column.render(product)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-100">
            <div className="border-b border-slate-100 px-6 py-4">
              <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">收益指标对比</h2>
                  <p className="mt-1 text-xs text-slate-500">集中查看收益表现及走势，评估短中期配置价值。</p>
                </div>
                <RangeSelector
                  id="performance-range"
                  selection={performanceRange}
                  onChange={setPerformanceRange}
                  minDate={globalMinDate}
                  maxDate={globalMaxDate}
                  resolvedStart={performanceRangeMeta.startDate}
                  resolvedEnd={performanceRangeMeta.endDate}
                />
              </div>
            </div>
            <div className="space-y-6 px-6 pb-6 pt-4">
              <div
                className="overflow-auto"
                style={{
                  height: METRIC_TABLE_VIEWPORT.height,
                  maxHeight: METRIC_TABLE_VIEWPORT.height,
                  width: METRIC_TABLE_VIEWPORT.width,
                  maxWidth: '100%',
                }}
              >
                {renderMetricTable(performanceColumns, performanceMetricsByProduct, 'performance')}
              </div>
              <div>
                <h3 className="text-base font-semibold text-slate-900">虚拟净值走势</h3>
                <p className="mt-1 text-xs text-slate-500">
                  当前可见区间会自动将首个交易日归一到 1，并实时调整 Y 轴范围。
                </p>
                <div className="mt-4 rounded-xl bg-slate-50 p-4">
                  {comparisonChartOption ? (
                    <ReactECharts
                      style={{ height: 360 }}
                      option={comparisonChartOption}
                      notMerge
                      lazyUpdate
                      onEvents={performanceChartEvents}
                    />
                  ) : (
                    <div className="rounded-lg bg-white/80 p-10 text-center text-slate-500">暂无可用的净值数据。</div>
                  )}
                </div>
              </div>
            </div>
          </section>

          <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-100">
            <div className="border-b border-slate-100 px-6 py-4">
              <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">风险指标对比</h2>
                  <p className="mt-1 text-xs text-slate-500">关注波动与回撤，衡量目标产品在震荡行情下的承压能力。</p>
                </div>
                <RangeSelector
                  id="risk-range"
                  selection={riskRange}
                  onChange={setRiskRange}
                  minDate={globalMinDate}
                  maxDate={globalMaxDate}
                  resolvedStart={riskRangeMeta.startDate}
                  resolvedEnd={riskRangeMeta.endDate}
                />
              </div>
            </div>
            <div className="px-6 pb-6 pt-4">
              <div
                className="overflow-auto"
                style={{
                  height: METRIC_TABLE_VIEWPORT.height,
                  maxHeight: METRIC_TABLE_VIEWPORT.height,
                  width: METRIC_TABLE_VIEWPORT.width,
                  maxWidth: '100%',
                }}
              >
                {renderMetricTable(riskColumns, riskMetricsByProduct, 'risk')}
              </div>
              <div className="mt-6 space-y-6">
                <div>
                  <h3 className="text-base font-semibold text-slate-900">最大回撤</h3>
                  <p className="mt-1 text-xs text-slate-500">回撤曲线相对于阶段高点的跌幅，辅助定位风控压力最大的时段。</p>
                  <div className="mt-4 rounded-xl bg-slate-50 p-4">
                    {drawdownChartOption ? (
                      <ReactECharts
                        style={{ height: 300 }}
                        option={drawdownChartOption}
                        notMerge
                        lazyUpdate
                        onEvents={riskChartEvents}
                      />
                    ) : (
                      <div className="rounded-lg bg-white/80 p-10 text-center text-slate-500">暂无可用的回撤数据。</div>
                    )}
                  </div>
                </div>
                <div>
                  <h3 className="text-base font-semibold text-slate-900">滚动波动率</h3>
                  <p className="mt-1 text-xs text-slate-500">{effectiveRollingWindowDays} 个交易日窗口年化的波动率，衡量短期震荡强度。</p>
                  <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-slate-500">
                    <label htmlFor="rolling-window-days" className="font-medium text-slate-600">
                      滚动窗口（交易日）
                    </label>
                    <input
                      id="rolling-window-days"
                      type="number"
                      inputMode="numeric"
                      min={rollingWindowMin}
                      max={rollingWindowMax}
                      step={ROLLING_WINDOW_STEP}
                      value={rollingWindowInputValue}
                      onChange={(event) => {
                        const { value } = event.target;
                        setRollingWindowInputValue(value);
                        if (!value.trim()) {
                          return;
                        }
                        const next = Number(value);
                        if (!Number.isNaN(next)) {
                          setRollingWindowDays(next);
                        }
                      }}
                      onBlur={() => {
                        if (!rollingWindowInputValue.trim()) {
                          setRollingWindowInputValue(String(effectiveRollingWindowDays));
                          return;
                        }
                        const parsed = Number(rollingWindowInputValue);
                        if (Number.isNaN(parsed)) {
                          setRollingWindowInputValue(String(effectiveRollingWindowDays));
                          return;
                        }
                        setRollingWindowDays(parsed);
                      }}
                      className="h-9 w-24 rounded border border-slate-200 px-3 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                    />
                    <span className="text-xs text-slate-400">{rollingWindowMin} - {rollingWindowMax} 天</span>
                    <span className="text-sm font-semibold text-emerald-600">当前：{effectiveRollingWindowDays} 天</span>
                  </div>
                  <div className="mt-4 rounded-xl bg-slate-50 p-4">
                    {rollingVolatilityChartOption ? (
                      <ReactECharts
                        style={{ height: 300 }}
                        option={rollingVolatilityChartOption}
                        notMerge
                        lazyUpdate
                        onEvents={riskChartEvents}
                      />
                    ) : (
                      <div className="rounded-lg bg-white/80 p-10 text-center text-slate-500">暂无可用的波动率数据。</div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-100">
            <div className="border-b border-slate-100 px-6 py-4">
              <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">性价比指标对比</h2>
                  <p className="mt-1 text-xs text-slate-500">综合费用与回报，筛选最具投入产出效率的候选产品。</p>
                </div>
                <RangeSelector
                  id="efficiency-range"
                  selection={efficiencyRange}
                  onChange={setEfficiencyRange}
                  minDate={globalMinDate}
                  maxDate={globalMaxDate}
                  resolvedStart={efficiencyRangeMeta.startDate}
                  resolvedEnd={efficiencyRangeMeta.endDate}
                />
              </div>
            </div>
            <div className="px-6 pb-6 pt-4">
              <div
                className="overflow-auto"
                style={{
                  height: METRIC_TABLE_VIEWPORT.height,
                  maxHeight: METRIC_TABLE_VIEWPORT.height,
                  width: METRIC_TABLE_VIEWPORT.width,
                  maxWidth: '100%',
                }}
              >
                {renderMetricTable(efficiencyColumns, efficiencyMetricsByProduct, 'efficiency')}
              </div>
            </div>
            <div className="px-6 pb-6">
              <div className="rounded-2xl bg-slate-50 p-5">
                <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                  <div>
                    <h3 className="text-base font-semibold text-slate-800">收益风险象限图</h3>
                    <p className="text-xs text-slate-500">
                      选择关注的性价比指标，观察不同产品在收益（横轴）与风险（纵轴）维度的分布。
                    </p>
                  </div>
                  <label className="flex flex-col gap-2 text-sm text-slate-600 sm:flex-row sm:items-center">
                    <span className="font-medium text-slate-700">指标选择</span>
                    <select
                      className="h-9 min-w-[160px] rounded border border-slate-200 px-3 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                      value={efficiencyQuadrantKey}
                      onChange={(event) => {
                        const next = event.target.value as EfficiencyQuadrantKey;
                        setEfficiencyQuadrantKey(next);
                      }}
                    >
                      {efficiencyQuadrantConfigs.map((option) => (
                        <option key={option.key} value={option.key}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <div className="mt-4 h-[360px] rounded-xl bg-white/80 p-4">
                  {efficiencyQuadrantChartOption ? (
                    <ReactECharts
                      style={{ height: '100%', width: '100%' }}
                      option={efficiencyQuadrantChartOption}
                      notMerge
                      lazyUpdate
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center text-sm text-slate-500">
                      暂无可用于绘制象限图的数据。
                    </div>
                  )}
                </div>
              </div>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
