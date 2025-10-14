import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';

interface TimeSeriesPoint {
  date: string;
  open: number;
  close: number;
  high: number;
  low: number;
  volume: number;
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
    duration_year?: number | null;
  };
  timeseries: TimeSeriesPoint[];
}

type OverlayId = 'PRICE_MA' | 'VOLUME_MA' | 'BOLL' | 'KDJ';

interface OverlayOption {
  id: OverlayId;
  label: string;
  description: string;
}

const overlayOptions: OverlayOption[] = [
  { id: 'PRICE_MA', label: '收盘价均线', description: '自定义多个周期观察趋势' },
  { id: 'VOLUME_MA', label: '成交量均线', description: '识别量能变化节奏' },
  { id: 'BOLL', label: '布林带', description: '判断波动区间与突破' },
  { id: 'KDJ', label: 'KDJ 指标', description: '研判超买超卖信号' },
];

type OverlaySettings = {
  PRICE_MA: { periods: string };
  VOLUME_MA: { periods: string };
  BOLL: { period: number; multiplier: number };
  KDJ: { period: number; kSmoothing: number; dSmoothing: number };
};

const decimalFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 2 });

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

const formatDuration = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  return `${decimalFormatter.format(value)} 年`;
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

function MetricCard({ title, value, description }: { title: string; value: string; description?: string }) {
  return (
    <div className="rounded-2xl border border-transparent bg-gradient-to-br from-white via-slate-50 to-emerald-50 p-5 shadow-sm">
      <div className="text-xs font-semibold uppercase tracking-wide text-emerald-500">{title}</div>
      <div className="mt-2 text-2xl font-bold text-slate-900">{value}</div>
      {description && <div className="mt-1 text-xs text-slate-500">{description}</div>}
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
  const navigate = useNavigate();
  const [detail, setDetail] = useState<ProductDetailResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedOverlays, setSelectedOverlays] = useState<OverlayId[]>(['PRICE_MA', 'VOLUME_MA']);
  const [overlaySettings, setOverlaySettings] = useState<OverlaySettings>(() => cloneOverlaySettings());

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
      try {
        setLoading(true);
        setError(null);
        const resp = await fetch(`/api/etf/products/${encodeURIComponent(productId)}`, { signal: controller.signal });
        if (!resp.ok) {
          if (resp.status === 404) {
            setError('未找到对应的ETF产品，请检查编号后重试。');
            setDetail(null);
            return;
          }
          throw new Error('加载产品详情失败');
        }
        const data = (await resp.json()) as ProductDetailResponse;
        setDetail(data);
      } catch (err) {
        if ((err as DOMException).name === 'AbortError') {
          return;
        }
        console.error('Failed to load product detail', err);
        setError('加载产品详情时出现问题，请稍后重试。');
        setDetail(null);
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
        start: 40,
        end: 100,
      },
      {
        show: true,
        xAxisIndex: hasKDJ ? [0, 1, 2] : [0, 1],
        type: 'slider',
        height: 18,
        bottom: hasKDJ ? 50 : 40,
        start: 40,
        end: 100,
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
        bottom: hasKDJ ? 90 : 70,
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

  return (
    <div className="mx-auto max-w-7xl space-y-8 px-6 py-10">
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => navigate(-1)}
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
                <div className="text-sm font-semibold uppercase tracking-wide text-emerald-500">ETF 产品研究</div>
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
                <MetricCard title="预期收益" value={formatPercent(metrics.exp_return)} description="披露的目标收益区间" />
                <MetricCard title="存续期" value={formatDuration(metrics.duration_year)} description="产品预计存续年限" />
              </div>
            </div>
          </section>

          <section className="space-y-6 rounded-3xl bg-white p-8 shadow-sm ring-1 ring-slate-100">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-lg font-semibold text-slate-900">价格与成交量</h2>
                <p className="text-sm text-slate-500">最近 {detail.timeseries.length} 个交易日</p>
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
        </>
      )}
    </div>
  );
}
