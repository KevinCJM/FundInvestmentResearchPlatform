import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import ReactECharts from 'echarts-for-react';
import HorizontalMetricComparison, {
  DEFAULT_METRIC_TABLE_HEIGHT,
  PerformanceQuadrantChart,
} from '../components/HorizontalMetricComparison';
import { buildAnnualMetricRows, computeAnnualMetrics } from '../utils/performance';
import { useNavigate } from 'react-router-dom';

// Helper component for section titles
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mt-6 rounded-2xl border border-gray-200 bg-white p-6">
      <h2 className="text-lg font-semibold text-gray-800">{title}</h2>
      <div className="mt-4">{children}</div>
    </div>
  );
}

interface ConfigDetail {
  className: string;
  code: string;
  name: string;
  weight: string;
}

const normalizeForKey = (value: any): any => {
  if (Array.isArray(value)) {
    return value.map((item) => normalizeForKey(item));
  }
  if (value && typeof value === 'object') {
    const entries = Object.entries(value)
      .filter(([, v]) => v !== undefined && v !== null)
      .map(([k, v]) => [k, normalizeForKey(v)] as const)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
    return entries.reduce<Record<string, any>>((acc, [k, v]) => {
      acc[k] = v;
      return acc;
    }, {});
  }
  return value;
};

const stableStringify = (value: any): string => JSON.stringify(normalizeForKey(value));

export default function ClassAllocation() {
  const navigate = useNavigate();
  // State for UI interaction
  const [returnMetric, setReturnMetric] = useState('annual_mean');
  const [riskMetric, setRiskMetric] = useState('annual_vol');
  const [startDate, setStartDate] = useState('2020-01-01');
  const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);

  // State for results
  const [frontierData, setFrontierData] = useState<any>(null);
  const [isCalculating, setIsCalculating] = useState(false);

  // State for loading data
  const [allocations, setAllocations] = useState<string[]>([]);
  const [selectedAlloc, setSelectedAlloc] = useState('');
  const [configDetails, setConfigDetails] = useState<ConfigDetail[] | null>(null);
  const [assetNames, setAssetNames] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // Form states for dynamic inputs
  const [annualDaysRet, setAnnualDaysRet] = useState(252);
  const [ewmAlpha, setEwmAlpha] = useState(0.94);
  const [ewmWindow, setEwmWindow] = useState(60);
  const [annualDaysRisk, setAnnualDaysRisk] = useState(252);
  const [ewmAlphaRisk, setEwmAlphaRisk] = useState(0.94);
  const [ewmWindowRisk, setEwmWindowRisk] = useState(60);
  const [confidence, setConfidence] = useState(95);
  const [returnType, setReturnType] = useState('simple');
  // 年化无风险收益率（%）
  const [riskFreePct, setRiskFreePct] = useState(1.5);

  // 约束：单一上下限 per asset
  const [singleLimits, setSingleLimits] = useState<Record<string, { lo: number; hi: number }>>({});
  // 约束：组上下限列表
  type GroupLimit = { id: string; assets: string[]; lo: number; hi: number };
  const [groupLimits, setGroupLimits] = useState<GroupLimit[]>([]);

  // 随机探索参数：多轮
  type RoundConf = { id: string; samples: number; step: number; buckets: number };
  const [rounds, setRounds] = useState<RoundConf[]>([
    { id: 'r0', samples: 1000, step: 0.5, buckets: 10 }, // 第0轮不使用分桶参数
    { id: 'r1', samples: 2000, step: 0.4, buckets: 10 },
    { id: 'r2', samples: 3000, step: 0.3, buckets: 20 },
    { id: 'r3', samples: 4000, step: 0.2, buckets: 30 },
    { id: 'r4', samples: 5000, step: 0.15, buckets: 40 },
    { id: 'r5', samples: 5000, step: 0.1, buckets: 50 },
  ]);
  // 权重量化
  const [quantStep, setQuantStep] = useState<'none' | '0.001' | '0.002' | '0.005'>('none');
  // SLSQP 精炼
  const [useRefine, setUseRefine] = useState(false);
  const [refineCount, setRefineCount] = useState(20);

  // ---- 策略制定与回测 ----
  type StrategyType = 'fixed' | 'risk_budget' | 'target';
  type StrategyRow = { id: string; type: StrategyType; name: string; rows: { className: string; weight?: number | null; budget?: number }[]; cfg: any; rebalance?: any };
  type ScheduleEntry = { markers: { date: string; weights: number[] }[]; cacheKey?: string | null; spec?: string | null };
  const [strategies, setStrategies] = useState<StrategyRow[]>([]);
  const [btStart, setBtStart] = useState<string>('');
  const [navCount, setNavCount] = useState<number>(0);
  const [btSeries, setBtSeries] = useState<any>(null);
  const [scheduleMarkers, setScheduleMarkers] = useState<Record<string, ScheduleEntry>>({});
  const [busyStrategy, setBusyStrategy] = useState<string | null>(null);
  const [showAddPicker, setShowAddPicker] = useState(false);
  const [btBusy, setBtBusy] = useState(false);
  const pageRef = useRef<HTMLDivElement | null>(null);
  const backtestButtonRef = useRef<HTMLButtonElement | null>(null);
  const [overlayOffset, setOverlayOffset] = useState<number | null>(null);
  const [btYAxisRange, setBtYAxisRange] = useState<{ min: number; max: number } | null>(null);

  const enterPortfolioResearch = useCallback(() => {
    const seen = new Set<string>();
    const constituents = (configDetails ?? []).flatMap((item) => {
      if (!item.code || seen.has(item.code)) return [];
      seen.add(item.code);
      return [{ kind: 'etf', product_id: item.code, code: item.code, name: item.name, weight: 0, risk_budget: 0 }];
    });
    sessionStorage.setItem('portfolioResearchImport', JSON.stringify({
      name: selectedAlloc ? `${selectedAlloc} · 产品研究组合` : '来自大类配置的研究组合',
      method: 'equal_weight',
      constituents,
    }));
    navigate('/portfolio-construction');
  }, [configDetails, navigate, selectedAlloc]);

  const formatWeightPercent = useCallback((value: number) => {
    if (!Number.isFinite(value)) return null;
    return Number((value * 100).toFixed(2));
  }, []);

  const applyWeightVector = useCallback((strategy: StrategyRow, weights: number[]): StrategyRow => {
    const nextRows = strategy.rows.map((row, idx) => {
      const raw = weights[idx];
      if (raw === undefined || raw === null || Number.isNaN(raw)) {
        return { ...row };
      }
      const formatted = formatWeightPercent(Number(raw));
      return { ...row, weight: formatted ?? row.weight ?? 0 };
    });
    return { ...strategy, rows: nextRows };
  }, [formatWeightPercent]);

  const buildTargetConstraints = useCallback(() => ({
    single_limits: Object.entries(singleLimits).reduce<Record<string, { lo: number; hi: number }>>((acc, [k, v]) => {
      acc[k] = { lo: Number(v?.lo ?? 0), hi: Number(v?.hi ?? 1) };
      return acc;
    }, {}),
    group_limits: groupLimits.map((g) => ({
      id: g.id,
      assets: g.assets,
      lo: Number(g.lo),
      hi: Number(g.hi),
    })),
  }), [singleLimits, groupLimits]);

  const buildSchedulePayload = useCallback((strategy: StrategyRow) => {
    if (!selectedAlloc) return null;
    const base: Record<string, any> = {
      alloc_name: selectedAlloc,
      start_date: btStart || undefined,
      strategy: {
        type: strategy.type,
        name: strategy.name,
        classes: strategy.rows.map((r) =>
          strategy.type === 'risk_budget'
            ? { name: r.className, budget: r.budget ?? 100 }
            : { name: r.className }
        ),
        rebalance: strategy.rebalance,
      },
    };
    if (strategy.type === 'risk_budget') {
      base.strategy.model = {
        risk_metric: strategy.cfg?.risk_metric || 'vol',
        days: strategy.cfg?.days ?? null,
        window: strategy.cfg?.window ?? null,
        confidence: strategy.cfg?.confidence ?? null,
        window_mode: strategy.cfg?.window_mode || 'rollingN',
        data_len: strategy.cfg?.data_len ?? null,
      };
    } else if (strategy.type === 'target') {
      base.strategy.model = {
        target: strategy.cfg?.target || 'min_risk',
        return_metric: strategy.cfg?.return_metric || 'annual',
        return_type: strategy.cfg?.return_type || 'simple',
        days: strategy.cfg?.ret_days ?? strategy.cfg?.days ?? 252,
        ret_alpha: strategy.cfg?.ret_alpha ?? null,
        ret_window: strategy.cfg?.ret_window ?? null,
        risk_metric: strategy.cfg?.risk_metric || 'vol',
        risk_days: strategy.cfg?.risk_days ?? strategy.cfg?.days ?? null,
        risk_alpha: strategy.cfg?.risk_alpha ?? null,
        risk_window: strategy.cfg?.risk_window ?? null,
        risk_confidence: strategy.cfg?.risk_confidence ?? null,
        risk_free_rate: Number((riskFreePct ?? 0) / 100),
        constraints: buildTargetConstraints(),
        target_return: strategy.cfg?.target_return ?? null,
        target_risk: strategy.cfg?.target_risk ?? null,
        window_mode: strategy.cfg?.window_mode || 'all',
        data_len: strategy.cfg?.data_len ?? null,
      };
    }
    return base;
  }, [selectedAlloc, btStart, buildTargetConstraints, riskFreePct]);

  const buildComputeWeightsPayload = useCallback((strategy: StrategyRow) => {
    if (!selectedAlloc) return null;
    if (strategy.type === 'fixed') return null;
    const windowMode = strategy.cfg?.window_mode || 'all';
    const base: any = {
      alloc_name: selectedAlloc,
      window_mode: windowMode,
      data_len: windowMode === 'all' ? undefined : strategy.cfg?.data_len ?? 60,
      strategy: {
        type: strategy.type,
        name: strategy.name,
        classes: strategy.rows.map((r) =>
          strategy.type === 'risk_budget'
            ? { name: r.className, budget: r.budget ?? 100 }
            : { name: r.className }
        ),
      },
    };
    if (strategy.type === 'risk_budget') {
      base.strategy.risk_metric = strategy.cfg?.risk_metric;
      base.strategy.confidence = strategy.cfg?.confidence;
      base.strategy.days = strategy.cfg?.days;
    } else if (strategy.type === 'target') {
      base.strategy = {
        ...base.strategy,
        target: strategy.cfg?.target,
        return_metric: strategy.cfg?.return_metric,
        return_type: strategy.cfg?.return_type,
        days: strategy.cfg?.ret_days ?? 252,
        risk_metric: strategy.cfg?.risk_metric || 'vol',
        window: strategy.cfg?.risk_window,
        confidence: strategy.cfg?.risk_confidence,
        risk_free_rate: Number((riskFreePct ?? 0) / 100),
        constraints: buildTargetConstraints(),
        target_return: strategy.cfg?.target_return,
        target_risk: strategy.cfg?.target_risk,
      };
    }
    return base;
  }, [selectedAlloc, buildTargetConstraints, riskFreePct]);

  const fetchPointWeights = useCallback(
    async (strategy: StrategyRow) => {
      const payload = buildComputeWeightsPayload(strategy);
      if (!payload) throw new Error('缺少方案配置，请先选择资产配置方案');
      const response = await fetch('/api/strategy/compute-weights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        const fallback = strategy.type === 'risk_budget' ? '风险预算权重计算失败' : '指定目标权重计算失败';
        throw new Error(data?.detail || fallback);
      }
      return (data.weights || []) as number[];
    },
    [buildComputeWeightsPayload]
  );

  const getScheduleSpecKey = useCallback(
    (strategy: StrategyRow): string | null => {
      if (!strategy.rebalance?.enabled || !strategy.rebalance?.recalc) return null;
      const schedule = buildSchedulePayload(strategy);
      if (!schedule) return null;
      const spec = schedule.strategy || {};
      return stableStringify({
        alloc_name: schedule.alloc_name,
        start_date: schedule.start_date ?? null,
        type: spec.type,
        rebalance: spec.rebalance || {},
        model: spec.model || {},
        classes: spec.classes || [],
      });
    },
    [buildSchedulePayload]
  );

  const fetchScheduleWeights = useCallback(
    async (strategy: StrategyRow) => {
      const payload = buildSchedulePayload(strategy);
      if (!payload) throw new Error('缺少方案配置，请先选择资产配置方案');
      const response = await fetch('/api/strategy/compute-schedule-weights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.detail || '批量调仓权重计算失败');
      }
      const markers = (data.dates || []).map((d: string, idx: number) => ({
        date: d,
        weights: (data.weights && data.weights[idx]) || [],
      }));
      const entry: ScheduleEntry = {
        markers,
        cacheKey: data.cache_key ?? null,
        spec: getScheduleSpecKey(strategy),
      };
      const lastWeights = markers.length ? markers[markers.length - 1].weights : [];
      return { entry, lastWeights };
    },
    [buildSchedulePayload, getScheduleSpecKey]
  );

  const buildBacktestModel = useCallback((strategy: StrategyRow) => {
    if (strategy.type === 'risk_budget') {
      return {
        risk_metric: strategy.cfg?.risk_metric,
        days: strategy.cfg?.days,
        confidence: strategy.cfg?.confidence,
        window_mode: strategy.cfg?.window_mode || 'all',
        data_len: strategy.cfg?.data_len,
      };
    }
    if (strategy.type === 'target') {
      return {
        target: strategy.cfg?.target,
        return_metric: strategy.cfg?.return_metric,
        return_type: strategy.cfg?.return_type,
        days: strategy.cfg?.ret_days ?? 252,
        ret_alpha: strategy.cfg?.ret_alpha,
        ret_window: strategy.cfg?.ret_window,
        risk_metric: strategy.cfg?.risk_metric,
        risk_days: strategy.cfg?.risk_days,
        risk_alpha: strategy.cfg?.risk_alpha,
        risk_window: strategy.cfg?.risk_window,
        risk_confidence: strategy.cfg?.risk_confidence,
        risk_free_rate: Number((riskFreePct ?? 0) / 100),
        constraints: buildTargetConstraints(),
        target_return: strategy.cfg?.target_return,
        target_risk: strategy.cfg?.target_risk,
        window_mode: strategy.cfg?.window_mode || 'all',
        data_len: strategy.cfg?.data_len,
      };
    }
    return undefined;
  }, [buildTargetConstraints, riskFreePct]);

  useEffect(() => {
    const handleReposition = () => {
      if (!pageRef.current || !backtestButtonRef.current) {
        setOverlayOffset(null);
        return;
      }
      const containerRect = pageRef.current.getBoundingClientRect();
      const buttonRect = backtestButtonRef.current.getBoundingClientRect();
      setOverlayOffset(Math.max(0, buttonRect.top - containerRect.top));
    };

    if (loading || isCalculating || btBusy) {
      handleReposition();
      window.addEventListener('resize', handleReposition);
      return () => window.removeEventListener('resize', handleReposition);
    } else {
      setOverlayOffset(null);
    }
  }, [loading, isCalculating, btBusy]);

  const parseMetricValue = useCallback((value: any) => {
    if (value === null || value === undefined) return NaN;
    const num = Number(value);
    return Number.isFinite(num) ? num : NaN;
  }, []);

  const backtestDateIndex = useMemo(() => {
    if (!btSeries?.dates || !Array.isArray(btSeries.dates)) return new Map<string, number>();
    const map = new Map<string, number>();
    btSeries.dates.forEach((d: string, idx: number) => {
      map.set(String(d), idx);
    });
    return map;
  }, [btSeries?.dates]);

  const computeBtYAxisRange = useCallback(
    (startIdx?: number, endIdx?: number) => {
      if (!btSeries?.series || !btSeries.dates) return null;
      const total = btSeries.dates.length;
      if (total === 0) return null;
      const lo = Math.max(0, startIdx ?? 0);
      const hi = Math.min(total - 1, endIdx ?? total - 1);
      if (lo > hi) return null;

      const values: number[] = [];

      Object.values(btSeries.series || {}).forEach((arr: any) => {
        if (!Array.isArray(arr)) return;
        for (let i = lo; i <= hi; i += 1) {
          const v = arr[i];
          if (Number.isFinite(v)) values.push(Number(v));
        }
      });

      Object.values(btSeries.markers || {}).forEach((arr: any) => {
        if (!Array.isArray(arr)) return;
        arr.forEach((item: any) => {
          const xVal = item?.date ?? (Array.isArray(item?.value) ? item.value[0] : undefined);
          const idx = xVal !== undefined ? backtestDateIndex.get(String(xVal)) : undefined;
          if (idx === undefined || idx < lo || idx > hi) return;
          const val = Array.isArray(item?.value) ? Number(item.value[1]) : Number(item?.value);
          if (Number.isFinite(val)) values.push(val);
        });
      });

      if (values.length === 0) return null;
      let min = Math.min(...values);
      let max = Math.max(...values);
      if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
      if (min === max) {
        const delta = min === 0 ? 1 : Math.abs(min) * 0.05;
        return { min: min - delta, max: max + delta };
      }
      const padding = (max - min) * 0.05;
      return { min: min - padding, max: max + padding };
    },
    [btSeries, backtestDateIndex]
  );

  useEffect(() => {
    if (!btSeries) {
      setBtYAxisRange(null);
      return;
    }
    setBtYAxisRange(computeBtYAxisRange());
  }, [btSeries, computeBtYAxisRange]);

  useEffect(() => {
    setScheduleMarkers((prev) => {
      let mutated = false;
      const next: Record<string, ScheduleEntry> = { ...prev };
      Object.entries(prev).forEach(([name, entry]) => {
        const strategy = strategies.find((item) => item.name === name);
        const specKey = strategy ? getScheduleSpecKey(strategy) : null;
        if (!strategy || !specKey || entry.spec !== specKey) {
          delete next[name];
          mutated = true;
        }
      });
      return mutated ? next : prev;
    });
  }, [strategies, getScheduleSpecKey]);

  const handleBacktestZoom = useCallback(
    (params: any) => {
      if (!btSeries?.dates || btSeries.dates.length === 0) return;
      const payload = Array.isArray(params?.batch) && params.batch.length > 0 ? params.batch[0] : params;
      const total = btSeries.dates.length;
      let startIdx: number | undefined;
      let endIdx: number | undefined;

      if (payload?.startValue !== undefined) {
        const idx = backtestDateIndex.get(String(payload.startValue));
        if (idx !== undefined) startIdx = idx;
      } else if (typeof payload?.start === 'number') {
        startIdx = Math.round((payload.start / 100) * (total - 1));
      }

      if (payload?.endValue !== undefined) {
        const idx = backtestDateIndex.get(String(payload.endValue));
        if (idx !== undefined) endIdx = idx;
      } else if (typeof payload?.end === 'number') {
        endIdx = Math.round((payload.end / 100) * (total - 1));
      }

      if (startIdx === undefined) startIdx = 0;
      if (endIdx === undefined) endIdx = total - 1;
      if (startIdx > endIdx) {
        const tmp = startIdx;
        startIdx = endIdx;
        endIdx = tmp;
      }

      const range = computeBtYAxisRange(startIdx, endIdx);
      setBtYAxisRange(range);
    },
    [btSeries, backtestDateIndex, computeBtYAxisRange]
  );

  const backtestMetricsSummary = useMemo(() => {
    const metrics = btSeries?.metrics;
    if (!Array.isArray(metrics) || metrics.length === 0) {
      return {
        columns: [] as string[],
        rows: [] as any[],
      };
    }
    const columns = metrics.map((m: any) => m?.name ?? '');
    const seriesMap = btSeries?.series || {};
    const toNavNumber = (value: any) => {
      if (value === null || value === undefined) return NaN;
      const num = Number(value);
      return Number.isFinite(num) ? num : NaN;
    };
    const computeCumulative = (name: string): number => {
      const values = seriesMap?.[name];
      if (!Array.isArray(values)) return NaN;
      const cleaned = values.map(toNavNumber).filter(v => Number.isFinite(v));
      if (cleaned.length === 0) return NaN;
      const first = cleaned.find(v => v !== 0) ?? cleaned[0];
      const last = cleaned[cleaned.length - 1];
      if (!Number.isFinite(first) || !Number.isFinite(last) || first === 0) return NaN;
      return last / first - 1;
    };
    const cumulativeValues = columns.map(name => computeCumulative(name));
    const cumulativePercentValues = cumulativeValues.map(v => (Number.isFinite(v) ? v * 100 : NaN));
    const rows = [
      { label: '累计收益率', values: cumulativeValues },
      { label: '累计收益率(%)', values: cumulativePercentValues },
      { label: '年化收益率(%)', values: metrics.map((m: any) => parseMetricValue(m.annual_return) * 100) },
      { label: '年化波动率(%)', values: metrics.map((m: any) => parseMetricValue(m.annual_vol) * 100) },
      { label: '夏普比率', values: metrics.map((m: any) => parseMetricValue(m.sharpe)) },
      { label: '99%VaR(日)(%)', values: metrics.map((m: any) => parseMetricValue(m.var99) * 100), reverseScale: true },
      { label: '99%ES(日)(%)', values: metrics.map((m: any) => parseMetricValue(m.es99) * 100), reverseScale: true },
      { label: '最大回撤(%)', values: metrics.map((m: any) => parseMetricValue(m.max_drawdown) * 100) },
      { label: '卡玛比率', values: metrics.map((m: any) => parseMetricValue(m.calmar)) },
    ];
    const annualSeriesMap: Record<string, Array<number | null | undefined>> = {};
    columns.forEach((name) => {
      annualSeriesMap[name] = Array.isArray(seriesMap?.[name]) ? seriesMap?.[name] : [];
    });
    const annualMetrics = computeAnnualMetrics(btSeries?.dates, annualSeriesMap);
    const annualRows = buildAnnualMetricRows(columns, annualMetrics);
    const mergedRows = annualRows.length > 0 ? [...rows, ...annualRows] : rows;
    return { columns, rows: mergedRows };
  }, [btSeries?.metrics, btSeries?.series, parseMetricValue]);

  const backtestMetricColumns = backtestMetricsSummary.columns;
  const backtestMetricRows = backtestMetricsSummary.rows;

  function computeEqualPercents(names: string[]): number[] {
    const n = Math.max(1, names.length);
    const base = Math.floor((100 / n) * 100) / 100; // 向下取两位
    const arr = Array(n).fill(base);
    const others = base * (n - 1);
    const first = parseFloat((100 - others).toFixed(2));
    arr[0] = first;
    return arr;
  }

  function uniqueStrategyName(base: string, list: StrategyRow[]): string {
    const exists = new Set(list.map(s => s.name));
    if (!exists.has(base)) return base;
    let k = 1;
    while (exists.has(`${base}${k}`)) k += 1;
    return `${base}${k}`;
  }


  // Fetch list of saved allocations on mount
  useEffect(() => {
    const fetchAllocations = async () => {
      try {
        setLoading(true);
        const res = await fetch('/api/list-allocations');
        if (!res.ok) throw new Error('无法获取方案列表');
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) {
          setAllocations(data);
          setSelectedAlloc(data[0]); // Default to the first one
        } else {
          setError('沒有找到已保存的大類構建方案。請先在“手動構建大類”頁面保存配置後，再進行大類資產配置。');
        }
      } catch (e: any) {
        setError(e.message || '获取方案列表失败');
      } finally {
        setLoading(false);
      }
    };
    fetchAllocations();
  }, []);

  // Handler to load details for the selected allocation
  const handleSelectAndLoad = async () => {
    if (!selectedAlloc) {
      alert('请选择一个方案');
      return;
    }
    try {
      setLoading(true);
      const res = await fetch(`/api/load-allocation?name=${encodeURIComponent(selectedAlloc)}`);
      if (!res.ok) throw new Error('加载方案详情失败');
      const data = await res.json();
      
      // Flatten the data for table display
      const details: ConfigDetail[] = [];
      data.forEach((ac: any) => {
        ac.etfs.forEach((etf: any) => {
          details.push({
            className: ac.name,
            code: etf.code,
            name: etf.name,
            weight: `${etf.weight.toFixed(2)}%`,
          });
        });
      });
      setConfigDetails(details);

      // 推导大类列表并初始化单项约束
      const aset = Array.from(new Set(details.map(d => d.className)));
      setAssetNames(aset);
      const initLimits: Record<string, { lo: number; hi: number }> = {};
      aset.forEach(n => { initLimits[n] = { lo: 0, hi: 1 }; });
      setSingleLimits(initLimits);

      // 初始化策略区域的行（按等权/空）
      setStrategies([]);

      // 获取默认回测开始日期（每个大类第一条净值的最大值）
      try {
        const r = await fetch(`/api/strategy/default-start?alloc_name=${encodeURIComponent(selectedAlloc)}`);
        const j = await r.json();
        if (r.ok) {
          if (j.default_start) setBtStart(j.default_start);
          if (typeof j.count === 'number') setNavCount(j.count);
        }
      } catch {}

    } catch (e: any) {
      alert(e.message || '加载失败');
      setConfigDetails(null);
    } finally {
      setLoading(false);
    }
  };

  // 当切换方案时，预取默认开始日期
  useEffect(() => {
    const fetchDefault = async () => {
      if (!selectedAlloc) return;
      try {
        const r = await fetch(`/api/strategy/default-start?alloc_name=${encodeURIComponent(selectedAlloc)}`);
        const j = await r.json();
        if (r.ok) {
          if (j.default_start) setBtStart(j.default_start);
          if (typeof j.count === 'number') setNavCount(j.count);
        }
      } catch {}
    };
    fetchDefault();
  }, [selectedAlloc]);

  const onCalculate = async () => {
    if (!selectedAlloc) {
      alert("请先选择一个大类构建方案并加载其详情");
      return;
    }
    
    const payload = {
      alloc_name: selectedAlloc,
      start_date: startDate,
      end_date: endDate,
      return_metric: {
        metric: returnMetric,
        type: returnType,
        days: annualDaysRet,
        alpha: ewmAlpha,
        window: ewmWindow,
      },
      risk_metric: {
        metric: riskMetric,
        type: returnType, // Risk metric uses the same return type
        days: annualDaysRisk,
        alpha: ewmAlphaRisk,
        window: ewmWindowRisk,
        confidence: confidence,
      },
      risk_free_rate: Number.isFinite(riskFreePct) ? riskFreePct / 100 : 0.0,
      constraints: {
        single_limits: singleLimits,
        group_limits: groupLimits.map(g => ({ assets: g.assets, lo: g.lo, hi: g.hi }))
      },
      exploration: {
        rounds: rounds.map((r, idx) => idx === 0 ? ({ samples: r.samples, step: r.step }) : ({ samples: r.samples, step: r.step, buckets: r.buckets }))
      },
      quantization: { step: quantStep === 'none' ? 'none' : Number(quantStep) },
      refine: { use_slsqp: useRefine, count: refineCount },
    };

    try {
      setIsCalculating(true);
      setFrontierData(null);
      const res = await fetch('/api/efficient-frontier', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '计算失败');
      setFrontierData(data);
    } catch (e: any) {
      alert(e.message);
    } finally {
      setIsCalculating(false);
    }
  };

  return (
    <div ref={pageRef} className="mx-auto max-w-5xl p-6 relative">
      {(loading || isCalculating || btBusy) && (
        <div className={`absolute inset-0 z-50 flex justify-center bg-black/40 rounded-2xl ${overlayOffset !== null ? 'items-start' : 'items-center'}`}>
          <div
            className="rounded-xl bg-white px-6 py-4 shadow text-sm"
            style={overlayOffset !== null ? { marginTop: overlayOffset } : undefined}
          >
            {btBusy ? '计算中...' : (isCalculating ? '正在计算，请稍候...' : '正在加载...')}
          </div>
        </div>
      )}

      
      <h1 className="text-2xl font-semibold">大类资产配置</h1>
      <p className="text-sm text-gray-500 mt-1">通过配置风险和收益指标，计算并可视化给定大类构建方案的可配置空间与有效前沿。</p>
      <button
        type="button"
        onClick={enterPortfolioResearch}
        disabled={!configDetails?.length}
        className="mt-3 rounded-lg border border-emerald-700 bg-white px-4 py-2 text-sm font-semibold text-emerald-800 hover:bg-emerald-50 disabled:cursor-not-allowed disabled:opacity-50"
      >
        保存为研究组合 / 进入组合指标
      </button>

      <Section title="选择大类构建方案">
        {loading && <p>正在加载方案列表...</p>}
        {error && <p className="text-red-600">{error}</p>}
        {!loading && !error && (
          <div className="flex items-center gap-4">
            <select 
              value={selectedAlloc}
              onChange={e => setSelectedAlloc(e.target.value)}
              className="flex-grow rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500">
              {allocations.map(name => <option key={name} value={name}>{name}</option>)}
            </select>
            <button 
              onClick={handleSelectAndLoad}
              className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700">
              选择该方案
            </button>
          </div>
        )}
        {configDetails && (
          <div className="mt-4 rounded-lg border">
            <div className="max-h-96 overflow-y-auto">
              <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">大类名称</th>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">ETF代码</th>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">ETF名称</th>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">资金权重</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 bg-white">
                {configDetails.map((item, index) => (
                  <tr key={index}>
                    <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-900">{item.className}</td>
                    <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-500 font-mono">{item.code}</td>
                    <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-900">{item.name}</td>
                    <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-500">{item.weight}</td>
                  </tr>
                ))}
              </tbody>
              </table>
            </div>
          </div>
        )}
      </Section>

      <Section title="刻画可配置空间与有效前沿的参数">
        <div className="grid grid-cols-1 gap-x-8 gap-y-6 md:grid-cols-2">
          {/* 收益指标 */}
          <div className="space-y-3 rounded-lg border p-4">
            <h3 className="font-medium text-gray-700">📈 收益指标</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-600">收益指标</label>
                <select onChange={e => setReturnMetric(e.target.value)} value={returnMetric} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm">
                  <option value="annual">年化收益率</option>
                  <option value="annual_mean">年化收益率均值</option>
                  <option value="cumulative">累计收益率</option>
                  <option value="mean">收益率均值</option>
                  <option value="ewm">指数加权收益率</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-600">收益类型</label>
                <select value={returnType} onChange={e => setReturnType(e.target.value)} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm">
                  <option value="simple">普通收益率</option>
                  <option value="log">对数收益率</option>
                </select>
              </div>
            </div>
            {(returnMetric === 'annual' || returnMetric === 'annual_mean') && (
              <div>
                <label className="block text-sm font-medium text-gray-600">年化天数</label>
                <input type="number" value={annualDaysRet} onChange={e => setAnnualDaysRet(Number(e.target.value))} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm" />
              </div>
            )}
            {returnMetric === 'ewm' && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-600">衰减因子 λ</label>
                  <input type="number" step="0.01" value={ewmAlpha} onChange={e => setEwmAlpha(Number(e.target.value))} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-600">窗口长度</label>
                  <input type="number" value={ewmWindow} onChange={e => setEwmWindow(Number(e.target.value))} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm" />
                </div>
              </div>
            )}
          </div>

          {/* 风险指标 */}
          <div className="space-y-3 rounded-lg border p-4">
            <h3 className="font-medium text-gray-700">⚠️ 风险指标</h3>
            <div className="grid grid-cols-2 gap-4">
                <div>
                    <label className="block text-sm font-medium text-gray-600">风险指标</label>
                    <select onChange={e => setRiskMetric(e.target.value)} value={riskMetric} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm">
                        <option value="vol">波动率</option>
                        <option value="annual_vol">年化波动率</option>
                        <option value="ewm_vol">指数加权波动率</option>
                        <option value="var">VaR</option>
                        <option value="es">ES</option>
                        <option value="max_drawdown">最大回撤</option>
                        <option value="downside_vol">下行波动率</option>
                    </select>
                </div>
                {(riskMetric === 'var' || riskMetric === 'es') && (
                    <div>
                        <label className="block text-sm font-medium text-gray-600">置信度 %</label>
                        <input type="number" value={confidence} onChange={e => setConfidence(Number(e.target.value))} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm" />
                    </div>
                )}
            </div>
            {(riskMetric === 'annual_vol') && (
                <div>
                    <label className="block text-sm font-medium text-gray-600">年化天数</label>
                    <input type="number" value={annualDaysRisk} onChange={e => setAnnualDaysRisk(Number(e.target.value))} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm" />
                </div>
            )}
            {(riskMetric === 'ewm_vol') && (
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-600">衰减因子 λ</label>
                        <input type="number" step="0.01" value={ewmAlphaRisk} onChange={e => setEwmAlphaRisk(Number(e.target.value))} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm" />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-600">窗口长度</label>
                    <input type="number" value={ewmWindowRisk} onChange={e => setEwmWindowRisk(Number(e.target.value))} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm" />
                    </div>
                </div>
            )}
          </div>

          {/* 夏普比率参数 */}
          <div className="space-y-3 rounded-lg border p-4 md:col-span-2">
            <h3 className="font-medium text-gray-700">📊 夏普比率参数</h3>
            <div>
              <label className="block text-sm font-medium text-gray-600">年化无风险收益率(%)</label>
              <input
                type="number"
                step="0.1"
                value={riskFreePct}
                onChange={e => setRiskFreePct(Number(e.target.value))}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
              />
              <p className="mt-1 text-xs text-gray-500">用于计算最大夏普率，默认 1.5%</p>
            </div>
            <p className="text-xs text-gray-500">
              计算逻辑：夏普比率 = (年化收益率均值 - 年化无风险利率) / 年化标准差。
            </p>
          </div>
        </div>

        {/* 权重约束设置 */}
        <div className="mt-6 grid grid-cols-1 gap-6 md:grid-cols-2">
          <div className="rounded-lg border p-4">
            <h3 className="font-medium text-gray-700">🔒 单个大类上下限</h3>
            {assetNames.length === 0 ? (
              <p className="text-sm text-gray-500 mt-2">请先选择并加载方案</p>
            ) : (
              <div className="mt-3 space-y-2">
                {assetNames.map(name => (
                  <div key={name} className="grid grid-cols-3 items-center gap-2">
                    <div className="text-sm text-gray-700">{name}</div>
                    <input type="number" min={0} max={1} step={0.01}
                      value={singleLimits[name]?.lo ?? 0}
                      onChange={e => setSingleLimits(prev => ({ ...prev, [name]: { ...(prev[name]||{lo:0,hi:1}), lo: Number(e.target.value) } }))}
                      className="rounded-md border-gray-300 px-2 py-1 text-sm" placeholder="下限(0-1)" />
                    <input type="number" min={0} max={1} step={0.01}
                      value={singleLimits[name]?.hi ?? 1}
                      onChange={e => setSingleLimits(prev => ({ ...prev, [name]: { ...(prev[name]||{lo:0,hi:1}), hi: Number(e.target.value) } }))}
                      className="rounded-md border-gray-300 px-2 py-1 text-sm" placeholder="上限(0-1)" />
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="rounded-lg border p-4">
            <h3 className="font-medium text-gray-700">🧩 多大类联合上下限</h3>
            <div className="mt-2 space-y-3">
              {groupLimits.map((g, idx) => (
                <div key={g.id} className="rounded border p-2">
                  <div className="flex flex-wrap gap-2">
                    {assetNames.map(n => (
                      <label key={n} className="flex items-center gap-1 text-xs">
                        <input type="checkbox" checked={g.assets.includes(n)} onChange={e => {
                          setGroupLimits(prev => prev.map(x => x.id===g.id ? { ...x, assets: e.target.checked ? [...x.assets, n] : x.assets.filter(a => a!==n) } : x))
                        }} />{n}
                      </label>
                    ))}
                  </div>
                  <div className="mt-2 grid grid-cols-3 gap-2">
                    <input type="number" min={0} max={1} step={0.01} value={g.lo} onChange={e => setGroupLimits(prev => prev.map(x => x.id===g.id ? { ...x, lo: Number(e.target.value) } : x))} className="rounded-md border-gray-300 px-2 py-1 text-sm" placeholder="下限(0-1)" />
                    <input type="number" min={0} max={1} step={0.01} value={g.hi} onChange={e => setGroupLimits(prev => prev.map(x => x.id===g.id ? { ...x, hi: Number(e.target.value) } : x))} className="rounded-md border-gray-300 px-2 py-1 text-sm" placeholder="上限(0-1)" />
                    <button onClick={() => setGroupLimits(prev => prev.filter(x => x.id !== g.id))} className="rounded bg-red-50 text-red-700 text-xs px-2">删除</button>
                  </div>
                </div>
              ))}
              <button onClick={() => setGroupLimits(prev => [...prev, { id: `g${Date.now()}`, assets: [], lo: 0, hi: 1 }])} className="rounded bg-gray-100 px-3 py-1 text-xs">+ 添加联合约束</button>
            </div>
          </div>
        </div>

        {/* 随机探索设置 */}
        <div className="mt-6 rounded-lg border p-4">
          <h3 className="font-medium text-gray-700">🎲 随机探索设置</h3>
          <p className="text-xs text-gray-500 mt-1">默认提供第0至第5轮，样本点从 1000~5000，步长从 0.5~0.1，分桶从 10~50（第0轮不分桶，可删除第1-5轮）。</p>
          <div className="mt-2 space-y-2">
            {rounds.map((r, idx) => (
              <div key={r.id} className="grid grid-cols-12 items-center gap-2">
                <div className="col-span-2 text-sm text-gray-600">第{idx}轮</div>
                <label className="col-span-3 text-xs text-gray-600 flex items-center gap-1">
                  <span className="whitespace-nowrap">样本点</span>
                  <input type="number" min={1} value={r.samples} onChange={e => setRounds(prev => prev.map(x => x.id===r.id ? { ...x, samples: Number(e.target.value) } : x))} className="ml-1 w-full rounded-md border-gray-300 px-2 py-1 text-xs" />
                </label>
                <label className="col-span-3 text-xs text-gray-600 flex items-center gap-1">
                  <span className="whitespace-nowrap">步长</span>
                  <input type="number" step={0.01} min={0} max={1} value={r.step} onChange={e => setRounds(prev => prev.map(x => x.id===r.id ? { ...x, step: Number(e.target.value) } : x))} className="ml-1 w-full rounded-md border-gray-300 px-2 py-1 text-xs" />
                </label>
                {idx > 0 && (
                  <label className="col-span-3 text-xs text-gray-600 flex items-center gap-1">
                    <span className="whitespace-nowrap">分桶</span>
                    <input type="number" min={1} value={(r as any).buckets ?? 50} onChange={e => setRounds(prev => prev.map(x => x.id===r.id ? { ...x, buckets: Number(e.target.value) } : x))} className="ml-1 w-full rounded-md border-gray-300 px-2 py-1 text-xs" />
                  </label>
                )}
                {idx > 0 && (
                  <button onClick={() => setRounds(prev => prev.filter(x => x.id !== r.id))} className="col-span-1 rounded bg-red-50 text-red-700 text-xs px-2">删</button>
                )}
              </div>
            ))}
            <button onClick={() => setRounds(prev => [...prev, { id: `r${Date.now()}`, samples: 200, step: 0.5, buckets: 50 }])} className="rounded bg-gray-100 px-3 py-1 text-xs">+ 增加一轮</button>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-600">权重量化</label>
              <select value={quantStep} onChange={e => setQuantStep(e.target.value as any)} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm">
                <option value="none">不量化</option>
                <option value="0.001">0.1%</option>
                <option value="0.002">0.2%</option>
                <option value="0.005">0.5%</option>
              </select>
            </div>
            <div className="flex items-end gap-2">
              <label className="flex items-center gap-2 text-sm text-gray-700">
                <input type="checkbox" checked={useRefine} onChange={e => setUseRefine(e.target.checked)} /> 使用 SLSQP 精炼
              </label>
              {useRefine && (
                <input type="number" min={1} value={refineCount} onChange={e => setRefineCount(Number(e.target.value))} className="w-28 rounded-md border-gray-300 px-2 py-1 text-sm" placeholder="精炼数量" />
              )}
            </div>
          </div>
        </div>
      </Section>

      <Section title="选择模型的构建区间">
        <div className="flex items-center gap-4">
            <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} className="w-full rounded-md border-gray-300 shadow-sm" />
            <span className="text-gray-500">至</span>
            <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} className="w-full rounded-md border-gray-300 shadow-sm" />
        </div>
      </Section>

      <div className="mt-8 flex justify-center">
        <button 
          onClick={onCalculate}
          disabled={isCalculating}
          className="rounded-lg bg-indigo-600 px-6 py-3 text-base font-semibold text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50">
          {isCalculating ? '计算中...' : '计算可配置空间与有效前沿'}
        </button>
      </div>

      {frontierData && (
        <Section title="结果展示">
          <ReactECharts
            style={{ height: 500 }}
            option={{
              title: {
                text: '可配置空间与有效前沿',
                left: 'center'
              },
              tooltip: {
                trigger: 'item',
                confine: true,
                formatter: (params: any) => {
                  const val = Array.isArray(params.value) ? params.value : (params.data?.value ?? params.value);
                  const risk = val?.[0];
                  const ret = val?.[1];
                  const names: string[] = frontierData.asset_names || [];
                  const ws: number[] | undefined = params.data?.weights;
                  const header = params.seriesName ? `${params.seriesName}<br/>` : '';
                  const rr = (Number.isFinite(risk) ? Number(risk).toFixed(4) : '-') + ' (风险)';
                  const re = (Number.isFinite(ret) ? Number(ret).toFixed(4) : '-') + ' (收益)';
                  if (ws && names && names.length === ws.length) {
                    const lines = names.map((n, i) => `${n}: ${(ws[i] * 100).toFixed(2)}%`);
                    return `${header}${lines.join('<br/>')}<br/>${rr}<br/>${re}`;
                  }
                  return `${header}${rr}<br/>${re}`;
                }
              },
              dataZoom: [
                { type: 'inside', xAxisIndex: 0, filterMode: 'none' },
                { type: 'inside', yAxisIndex: 0, filterMode: 'none' },
                { type: 'slider', xAxisIndex: 0, filterMode: 'none' },
                { type: 'slider', yAxisIndex: 0, filterMode: 'none' },
              ],
              legend: {
                top: 36,
                left: 'center',
                data: ['其他组合', '有效前沿', '最大夏普率', '最小方差', '最大收益']
              },
              grid: { top: 80 },
              xAxis: { type: 'value', name: `风险（${riskMetric}）`, scale: true },
              yAxis: { type: 'value', name: `收益（${returnMetric}/${returnType === 'log' ? 'log' : 'simple'}）`, scale: true },
              series: [
                ...(frontierData.scatter ? [{
                  name: '其他组合',
                  type: 'scatter',
                  symbolSize: 3,
                  data: frontierData.scatter,
                  itemStyle: { color: 'rgba(128, 128, 128, 0.35)' }
                }] : []),
                ...(frontierData.frontier ? [{
                  name: '有效前沿',
                  type: 'scatter',
                  symbolSize: 6,
                  data: frontierData.frontier,
                  itemStyle: { color: '#2563eb' } // Tailwind indigo-600
                }] : []),
                ...(frontierData.max_sharpe ? [{
                  name: '最大夏普率',
                  type: 'scatter',
                  symbolSize: 10,
                  data: [frontierData.max_sharpe]
                }] : []),
                ...(frontierData.min_variance ? [{
                  name: '最小方差',
                  type: 'scatter',
                  symbolSize: 10,
                  data: [frontierData.min_variance]
                }] : []),
                ...(frontierData.max_return ? [{
                  name: '最大收益',
                  type: 'scatter',
                  symbolSize: 10,
                  data: [frontierData.max_return]
                }] : []),
              ]
            }}
          />
        </Section>
      )}

      {/* 大类资产策略制定与回测 */}
      <Section title="大类资产策略制定与回测">
        <div className="space-y-4">
          {/* 顶部不再显示“添加策略”按钮，统一放在策略列表与回测之间 */}

          {strategies.map((s, idx) => (
            <div key={s.id} className="rounded-lg border p-4">
              <div className="flex flex-wrap items-center gap-3">
                <input value={s.name} onChange={e => setStrategies(prev => prev.map(x => x.id===s.id? { ...x, name: e.target.value } : x))} className="rounded-md border-gray-300 px-2 py-1 text-sm" />
                <span className="rounded bg-gray-100 px-2 py-1 text-xs text-gray-700">
                  {s.type === 'fixed' ? '固定比例' : s.type === 'risk_budget' ? '风险预算' : '指定目标'}
                </span>
                <button onClick={() => setStrategies(prev => prev.filter(x => x.id !== s.id))} className="ml-auto rounded bg-red-50 px-2 py-1 text-xs text-red-700">删除</button>
              </div>
              {/* 再平衡设置（通用） */}
              <div className="mt-3 rounded border p-3 text-sm">
                <label className="flex items-center gap-2"><input type="checkbox" checked={!!s.rebalance?.enabled} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, rebalance: { ...(x.rebalance||{}), enabled: e.target.checked } } : x))}/> 是否启用再平衡</label>
                {s.rebalance?.enabled && (
                  <div className="mt-2 grid grid-cols-12 items-center gap-2">
                    <div className="col-span-3">
                      <label className="block text-xs text-gray-600">再平衡方式</label>
                      <select value={s.rebalance?.mode||'monthly'} onChange={e=> setStrategies(prev=> prev.map(x=> {
                        if (x.id!==s.id) return x as any;
                        const mode = e.target.value;
                        const rb = { ...(x.rebalance||{}), mode } as any;
                        if (mode === 'fixed' && !rb.fixedInterval) {
                          rb.fixedInterval = Math.max(1, navCount||1);
                        }
                        return { ...x, rebalance: rb } as any;
                      }))} className="mt-1 w-full rounded border-gray-300">
                        <option value="weekly">每周</option>
                        <option value="monthly">每月</option>
                        <option value="yearly">每年</option>
                        <option value="fixed">固定区间</option>
                      </select>
                    </div>
                    {s.rebalance?.mode !== 'fixed' ? (
                      <>
                    <div className="col-span-2">
                      <label className="block text-xs text-gray-600">第N</label>
                      <input type="number" min={1} max={ s.rebalance?.mode==='weekly'?5: s.rebalance?.mode==='monthly'?30:360 } value={s.rebalance?.N ?? 1} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, rebalance: { ...(x.rebalance||{}), which:'nth', N: Number(e.target.value) } } : x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                    </div>
                        <div className="col-span-2">
                        <label className="block text-xs text-gray-600">个</label>
                        <div className="mt-1 text-sm text-gray-500">&nbsp;</div>
                        </div>
                        <div className="col-span-2">
                          <label className="block text-xs text-gray-600">单位</label>
                          <select value={s.rebalance?.unit||'trading'} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, rebalance: { ...(x.rebalance||{}), unit: e.target.value } } : x))} className="mt-1 w-full rounded border-gray-300">
                            <option value="trading">交易日</option>
                            <option value="natural">自然日</option>
                          </select>
                        </div>
                      </>
                    ) : (
                      <div className="col-span-3">
                        <label className="block text-xs text-gray-600">固定区间(天)</label>
                        <input type="number" min={1} value={s.rebalance?.fixedInterval ?? 20} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, rebalance: { ...(x.rebalance||{}), fixedInterval: Number(e.target.value) } } : x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                      </div>
                    )}
                    {s.type !== 'fixed' && (
                      <div className="col-span-12">
                        <label className="flex items-center gap-2"><input type="checkbox" checked={!!s.rebalance?.recalc} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, rebalance: { ...(x.rebalance||{}), recalc: e.target.checked } } : x))}/> 再平衡时是否重新模型计算</label>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* 固定比例 */}
              {s.type === 'fixed' && (
                <div className="mt-3 space-y-3">
                  <div className="flex items-center gap-3 text-sm">
                    <label className="flex items-center gap-2"><input type="radio" checked={(s.cfg?.mode||'equal')==='equal'} onChange={() => setStrategies(prev => prev.map(x=>{
                      if (x.id!==s.id) return x;
                      const eqArr = computeEqualPercents(assetNames);
                      return { ...x, cfg:{...x.cfg, mode:'equal'}, rows: assetNames.map((n,i)=> ({ className:n, weight:eqArr[i] })) } as StrategyRow;
                    }))}/> 等权重</label>
                    <label className="flex items-center gap-2"><input type="radio" checked={(s.cfg?.mode||'equal')==='custom'} onChange={() => setStrategies(prev => prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, mode:'custom'}, rows: x.rows.map((rr,i)=> ({...rr, weight: i===0?100:0})) }:x))}/> 自定义权重</label>
                  </div>
                  <div className="rounded border">
                    <table className="min-w-full">
                      <thead className="bg-gray-50 text-xs text-gray-600"><tr><th className="px-3 py-2 text-left">大类名称</th><th className="px-3 py-2 text-left">资金权重(%)</th></tr></thead>
                      <tbody className="text-sm">
                        {s.rows.map((r,i)=> (
                          <tr key={i} className="border-t">
                            <td className="px-3 py-2">{r.className}</td>
                            <td className="px-3 py-2"><input type="number" value={r.weight ?? ''} onChange={e=>{
                              const v = e.target.value === '' ? undefined : Number(e.target.value);
                              setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, rows:x.rows.map((rr,j)=> j===i?{...rr, weight:v}:rr)}:x))
                            }} className={`w-28 rounded border px-2 py-1 ${ (s.cfg?.mode||'equal')==='equal' ? 'bg-gray-50 text-gray-500 border-gray-200' : 'border-gray-300' }`} disabled={(s.cfg?.mode||'equal')==='equal'}/></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* 风险预算 */}
              {s.type === 'risk_budget' && (
                <div className="mt-3 space-y-3">
                  <div className="rounded border">
                    <table className="min-w-full">
                      <thead className="bg-gray-50 text-xs text-gray-600"><tr><th className="px-3 py-2 text-left">大类名称</th><th className="px-3 py-2 text-left">风险预算(%)</th><th className="px-3 py-2 text-left">资金权重(%)</th></tr></thead>
                      <tbody className="text-sm">
                        {s.rows.map((r,i)=> (
                          <tr key={i} className="border-t">
                            <td className="px-3 py-2">{r.className}</td>
                            <td className="px-3 py-2"><input type="number" value={r.budget ?? 100} onChange={e=>{
                              const v = Number(e.target.value);
                              setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, rows:x.rows.map((rr,j)=>j===i?{...rr, budget:v}:rr)}:x))
                            }} className="w-28 rounded border-gray-300 px-2 py-1"/></td>
                            <td className="px-3 py-2">{r.weight==null? '-' : (r.weight?.toFixed(2))}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <label className="block text-xs text-gray-600">风险指标</label>
                      <select value={s.cfg?.risk_metric||'vol'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_metric:e.target.value}}:x))} className="mt-1 w-full rounded border-gray-300">
                        <option value="vol">波动率</option>
                        <option value="var">VaR</option>
                        <option value="es">ES</option>
                        <option value="downside_vol">下行波动率</option>
                        <option value="max_drawdown">最大回撤</option>
                      </select>
                    </div>
                    {['var','es'].includes(s.cfg?.risk_metric) && (
                      <><div>
                        <label className="block text-xs text-gray-600">置信度(%)</label>
                        <input type="number" value={s.cfg?.confidence ?? 95} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, confidence:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                      </div>
                      <div>
                        <label className="block text-xs text-gray-600">天数</label>
                        <input type="number" value={s.cfg?.days ?? 252} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, days:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                      </div></>
                    )}
                  </div>
                  {/* 模型计算区间 */}
                  <div className="rounded border p-3 text-sm">
                    <div className="grid grid-cols-3 gap-3 items-end">
                      <div>
                        <label className="block text-xs text-gray-600">窗口模式</label>
                        <select value={s.cfg?.window_mode || 'all'} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, cfg: { ...(x.cfg||{}), window_mode: e.target.value } } : x))} className="mt-1 w-full rounded border-gray-300">
                          <option value="all">所有数据</option>
                          <option value="rollingN">最近N条</option>
                        </select>
                      </div>
                      { (s.cfg?.window_mode==='rollingN') && (
                        <div>
                          <label className="block text-xs text-gray-600">N（交易日）</label>
                          <input type="number" min={2} value={s.cfg?.data_len ?? 60} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, cfg: { ...(x.cfg||{}), data_len: Number(e.target.value) } } : x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                        </div>
                      )}
                    </div>
                    <p className="mt-2 text-xs text-gray-500">
                      模式说明：
                      <span className="ml-1 font-medium">所有数据</span> 使用回测开始至当期的全部样本；
                      <span className="ml-1 font-medium">最近N条</span> 使用当期之前最近 N 条的滚动窗口（推荐）。
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    <button
                      disabled={busyStrategy===s.id}
                      className="rounded bg-gray-100 px-3 py-1 text-sm disabled:opacity-50"
                      onClick={async ()=>{
                        try{
                          setBusyStrategy(s.id);
                          if (s.rebalance?.enabled && s.rebalance?.recalc) {
                            setBtBusy(true);
                            const { entry, lastWeights } = await fetchScheduleWeights(s);
                            setScheduleMarkers(prev => ({ ...prev, [s.name]: entry }));
                            if (lastWeights.length) {
                              setStrategies(prev => prev.map(x => x.id === s.id ? applyWeightVector(x, lastWeights) : x));
                            }
                          } else {
                            const weights = await fetchPointWeights(s);
                            setStrategies(prev => prev.map(x => x.id === s.id ? applyWeightVector(x, weights) : x));
                          }
                        }catch(e:any){
                          alert(e?.message || '反推失败');
                        }finally{
                          setBusyStrategy(null);
                          setBtBusy(false);
                        }
                      }}
                    >反推资金权重</button>
                  </div>
                  {busyStrategy===s.id && <div className="text-xs text-gray-500">计算中，请稍候…</div>}

                  {/* 再平衡横向权重表（来自回测后的 markers） */}
                  {(() => {
                    if (!s.rebalance?.enabled || !s.rebalance?.recalc) return null;
                    const entry = scheduleMarkers[s.name];
                    const markers = entry?.markers || (btSeries?.markers?.[s.name] || []);
                    if (!markers || markers.length === 0) return null;
                    const dates: string[] = markers.map((m: any) => m.date);
                    const names: string[] = (btSeries?.asset_names || assetNames);
                    const weightsByAsset: number[][] = names.map((_: any, i: number) => markers.map((m: any) => (m.weights?.[i] ?? 0)));
                    return (
                      <div className="mt-3">
                        <div className="rounded border overflow-x-auto">
                          <table className="min-w-full whitespace-nowrap text-sm">
                            <thead className="bg-gray-50">
                              <tr>
                                <th className="px-3 py-2 text-left text-xs text-gray-600">大类名称</th>
                                {dates.map((d) => (
                                  <th key={d} className="px-3 py-2 text-left text-xs text-gray-600">{d}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {names.map((n, rIdx) => (
                                <tr key={n} className="border-t">
                                  <td className="px-3 py-2">{n}</td>
                                  {weightsByAsset[rIdx].map((w, cIdx) => (
                                    <td key={cIdx} className="px-3 py-2">{(w*100).toFixed(2)}%</td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    );
                  })()}
                </div>
              )}

              {/* 指定目标 */}
              {s.type === 'target' && (
                <div className="mt-3 space-y-3">
                  {/* 目标类型 + 收益率类型 */}
                  <div className="grid grid-cols-2 gap-4 text-sm rounded border p-3">
                    <div>
                      <label className="block text-xs text-gray-600">目标类型</label>
                      <select value={s.cfg?.target||'min_risk'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, target:e.target.value}}:x))} className="mt-1 w-full rounded border-gray-300">
                        <option value="min_risk">最小风险</option>
                        <option value="max_return">最大收益</option>
                        <option value="max_sharpe">最大化收益风险性价比</option>
                        <option value="max_sharpe_traditional">最大化夏普比率</option>
                        <option value="risk_min_given_return">指定收益下最小风险</option>
                        <option value="return_max_given_risk">指定风险下最大收益</option>
                      </select>
                      
                      {/* Explanations for each target type */}
                      {(s.cfg?.target === 'min_risk' || !s.cfg?.target) && (
                        <p className="mt-2 text-xs text-gray-500 bg-gray-50 p-2 rounded-md">
                          目标：在满足所有约束条件下，寻找使组合风险（由指定的<strong>风险指标</strong>衡量）最小化的权重。
                        </p>
                      )}
                      {s.cfg?.target === 'max_return' && (
                        <p className="mt-2 text-xs text-gray-500 bg-gray-50 p-2 rounded-md">
                          目标：在满足所有约束条件下，寻找使组合收益（由指定的<strong>收益指标</strong>衡量）最大化的权重。
                        </p>
                      )}
                      {s.cfg?.target === 'max_sharpe' && (
                        <p className="mt-2 text-xs text-gray-500 bg-gray-50 p-2 rounded-md">
                          目标：寻找使 <strong>(指定收益指标) / (指定风险指标)</strong> 比值最大化的权重。这是一个广义的收益风险性价比优化。
                        </p>
                      )}
                      {s.cfg?.target === 'max_sharpe_traditional' && (
                        <p className="mt-2 text-xs text-gray-500 bg-gray-50 p-2 rounded-md">
                          目标：寻找使传统夏普比率 <code>(年化收益 - 无风险利率) / 年化波动率</code> 最大化的权重。
                        </p>
                      )}
                      {s.cfg?.target === 'risk_min_given_return' && (
                        <p className="mt-2 text-xs text-gray-500 bg-gray-50 p-2 rounded-md">
                          目标：在组合收益等于<strong>目标收益值</strong>的前提下，寻找使组合风险最小化的权重。
                        </p>
                      )}
                      {s.cfg?.target === 'return_max_given_risk' && (
                        <p className="mt-2 text-xs text-gray-500 bg-gray-50 p-2 rounded-md">
                          目标：在组合风险不高于<strong>目标风险值</strong>的前提下，寻找使组合收益最大化的权重。
                        </p>
                      )}
                    </div>
                    <div>
                      <label className="block text-xs text-gray-600">收益率类型</label>
                      <select value={s.cfg?.return_type||'simple'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, return_type:e.target.value}}:x))} className="mt-1 w-full rounded border-gray-300">
                        <option value="simple">普通收益率</option>
                        <option value="log">对数收益率</option>
                      </select>
                    </div>
                  </div>

                  {/* 根据目标类型显示不同UI */}
                  {s.cfg?.target === 'max_sharpe_traditional' ? (
                    <div className="space-y-3 rounded border p-3 text-sm">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-xs text-gray-600">收益指标 (固定)</label>
                          <input type="text" value="年化收益率均值" disabled className="mt-1 w-full rounded border-gray-200 bg-gray-100 px-2 py-1"/>
                        </div>
                        <div>
                          <label className="block text-xs text-gray-600">风险指标 (固定)</label>
                          <input type="text" value="年化波动率" disabled className="mt-1 w-full rounded border-gray-200 bg-gray-100 px-2 py-1"/>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-xs text-gray-600">年化天数</label>
                          <input type="number" value={s.cfg?.days ?? 252} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, days:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                        </div>
                        <div>
                          <label className="block text-xs text-gray-600">年化无风险利率(%)</label>
                          <input type="number" step="0.1" value={s.cfg?.risk_free_rate_pct ?? 1.5} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_free_rate_pct:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <>
                      {/* 收益指标配置 */}
                      <div className="rounded border p-3 text-sm space-y-3">
                        <div>
                          <label className="block text-xs text-gray-600">收益指标</label>
                          <select value={s.cfg?.return_metric||'cumulative'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, return_metric:e.target.value}}:x))} className="mt-1 w-full rounded border-gray-300">
                            <option value="annual">年化收益率</option>
                            <option value="annual_mean">年化收益率均值</option>
                            <option value="cumulative">累计收益率</option>
                            <option value="mean">收益率均值</option>
                            <option value="ewm">指数加权收益率</option>
                          </select>
                        </div>
                        {(s.cfg?.return_metric==='annual' || s.cfg?.return_metric==='annual_mean') && (
                          <div>
                            <label className="block text-xs text-gray-600">年化天数</label>
                            <input type="number" value={s.cfg?.days ?? 252} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, days:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                          </div>
                        )}
                        {s.cfg?.return_metric==='ewm' && (
                          <div className="grid grid-cols-2 gap-3">
                            <div>
                              <label className="block text-xs text-gray-600">衰减因子 λ</label>
                              <input type="number" step={0.01} value={s.cfg?.ret_alpha ?? 0.94} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, ret_alpha:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                            </div>
                            <div>
                              <label className="block text-xs text-gray-600">窗口长度</label>
                              <input type="number" value={s.cfg?.ret_window ?? 60} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, ret_window:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                            </div>
                          </div>
                        )}
                      </div>

                      {/* 风险指标配置 */}
                      <div className="rounded border p-3 text-sm space-y-3">
                        <div>
                          <label className="block text-xs text-gray-600">风险指标</label>
                          <select value={s.cfg?.risk_metric||'vol'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_metric:e.target.value}}:x))} className="mt-1 w-full rounded border-gray-300">
                            <option value="vol">波动率</option>
                            <option value="annual_vol">年化波动率</option>
                            <option value="ewm_vol">指数加权波动率</option>
                            <option value="var">VaR</option>
                            <option value="es">ES</option>
                            <option value="max_drawdown">最大回撤</option>
                            <option value="downside_vol">下行波动率</option>
                          </select>
                        </div>
                        {s.cfg?.risk_metric==='annual_vol' && (
                          <div>
                            <label className="block text-xs text-gray-600">年化天数</label>
                            <input type="number" value={s.cfg?.risk_days ?? 252} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_days:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                          </div>
                        )}
                        {s.cfg?.risk_metric==='ewm_vol' && (
                          <div className="grid grid-cols-2 gap-3">
                            <div>
                              <label className="block text-xs text-gray-600">衰减因子 λ</label>
                              <input type="number" step={0.01} value={s.cfg?.risk_alpha ?? 0.94} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_alpha:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                            </div>
                            <div>
                              <label className="block text-xs text-gray-600">窗口长度</label>
                              <input type="number" value={s.cfg?.risk_window ?? 60} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_window:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                            </div>
                          </div>
                        )}
                        {(s.cfg?.risk_metric==='var' || s.cfg?.risk_metric==='es') && (
                          <div>
                            <label className="block text-xs text-gray-600">置信度%</label>
                            <input type="number" value={s.cfg?.risk_confidence ?? 95} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_confidence:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                          </div>
                        )}
                      </div>
                    </>
                  )}
                  {(s.cfg?.target==='risk_min_given_return') && (
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <label className="block text-xs text-gray-600">目标收益值</label>
                        <input type="number" value={s.cfg?.target_return ?? ''} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, target_return: Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                      </div>
                    </div>
                  )}
                  {(s.cfg?.target==='return_max_given_risk') && (
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <label className="block text-xs text-gray-600">目标风险值</label>
                        <input type="number" value={s.cfg?.target_risk ?? ''} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, target_risk: Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                      </div>
                    </div>
                  )}

                  {/* 结果表格：若开启 recalc 且有回测数据，则横向展示每次再平衡的权重，否则展示当前权重 */}
                  {(() => {
                    if (!s.rebalance?.enabled || !s.rebalance?.recalc) return null;
                    const entry = scheduleMarkers[s.name];
                    const markers = entry?.markers || (btSeries?.markers?.[s.name] || []);
                    if (!markers || markers.length === 0) return null;
                    const dates: string[] = markers.map((m: any) => m.date);
                    const names: string[] = (btSeries?.asset_names || assetNames);
                    const weightsByAsset: number[][] = names.map((_: any, i: number) => markers.map((m: any) => (m.weights?.[i] ?? 0)));
                    return (
                      <div className="rounded border overflow-x-auto">
                        <table className="min-w-full whitespace-nowrap text-sm">
                          <thead className="bg-gray-50">
                            <tr>
                              <th className="px-3 py-2 text-left text-xs text-gray-600">大类名称</th>
                              {dates.map((d) => (
                                <th key={d} className="px-3 py-2 text-left text-xs text-gray-600">{d}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {names.map((n, rIdx) => (
                              <tr key={n} className="border-t">
                                <td className="px-3 py-2">{n}</td>
                                {weightsByAsset[rIdx].map((w, cIdx) => (
                                  <td key={cIdx} className="px-3 py-2">{(w*100).toFixed(2)}%</td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    );
                  })() || (
                    <div className="rounded border">
                      <table className="min-w-full">
                        <thead className="bg-gray-50 text-xs text-gray-600"><tr><th className="px-3 py-2 text-left">大类名称</th><th className="px-3 py-2 text-left">资金权重(%)</th></tr></thead>
                        <tbody className="text-sm">
                          {s.rows.map((r,i)=> (
                            <tr key={i} className="border-t">
                              <td className="px-3 py-2">{r.className}</td>
                              <td className="px-3 py-2">{r.weight==null? '-' : (r.weight?.toFixed(2))}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  {/* 模型计算区间 */}
                  <div className="rounded border p-3 text-sm">
                    <div className="grid grid-cols-3 gap-3 items-end">
                      <div>
                        <label className="block text-xs text-gray-600">窗口模式</label>
                        <select value={s.cfg?.window_mode || 'rollingN'} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, cfg: { ...(x.cfg||{}), window_mode: e.target.value } } : x))} className="mt-1 w-full rounded border-gray-300">
                          <option value="all">所有数据</option>
                          <option value="rollingN">最近N条</option>
                        </select>
                      </div>
                      { (s.cfg?.window_mode==='rollingN') && (
                        <div>
                          <label className="block text-xs text-gray-600">N（交易日）</label>
                          <input type="number" min={2} value={s.cfg?.data_len ?? 60} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, cfg: { ...(x.cfg||{}), data_len: Number(e.target.value) } } : x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                        </div>
                      )}
                    </div>
                    <p className="mt-2 text-xs text-gray-500">
                      模式说明：
                      <span className="ml-1 font-medium">所有数据</span> 使用回测开始至当期的全部样本；
                      <span className="ml-1 font-medium">最近N条</span> 使用当期之前最近 N 条的滚动窗口（推荐）。
                    </p>
                  </div>

                  <div>
                    <button
                      disabled={busyStrategy===s.id}
                      className="rounded bg-gray-100 px-3 py-1 text-sm disabled:opacity-50"
                      onClick={async ()=>{
                        try{
                          setBusyStrategy(s.id);
                          if (s.rebalance?.enabled && s.rebalance?.recalc) {
                            setBtBusy(true);
                            const { entry, lastWeights } = await fetchScheduleWeights(s);
                            setScheduleMarkers(prev => ({ ...prev, [s.name]: entry }));
                            if (lastWeights.length) {
                              setStrategies(prev => prev.map(x => x.id === s.id ? applyWeightVector(x, lastWeights) : x));
                            }
                          } else {
                            const weights = await fetchPointWeights(s);
                            setStrategies(prev => prev.map(x => x.id === s.id ? applyWeightVector(x, weights) : x));
                          }
                        }catch(e:any){
                          alert(e?.message || '反推失败');
                        }finally{
                          setBusyStrategy(null);
                          setBtBusy(false);
                        }
                      }}
                    >反推资金权重</button>
                  </div>

                  {/* 已替换为横向表格展示（见上）*/}
                </div>
              )}
            </div>
          ))}

          {/* 添加策略按钮与类型选择器，位于所有策略块的下方且位于回测模块上方 */}
          {!showAddPicker && (
            <div className="mt-4">
              <button
                onClick={() => {
                  if (assetNames.length === 0) { alert('请先加载方案'); return; }
                  setShowAddPicker(true);
                }}
                className="rounded bg-indigo-600 text-white px-3 py-2 text-sm">
                + 添加新的组合策略
              </button>
            </div>
          )}
              {showAddPicker && (
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <span className="text-sm text-gray-700">选择策略类型：</span>
                  {(['fixed','risk_budget','target'] as StrategyType[]).map(t => (
                    <button key={t} className="rounded bg-gray-100 px-3 py-1 text-sm" onClick={() => {
                      const id = `s${Date.now()}`;
                      if (t === 'fixed') {
                        setStrategies(prev => {
                          const eqArr = computeEqualPercents(assetNames);
                          const rows = assetNames.map((n,i)=> ({ className:n, weight: eqArr[i], budget: 100 }));
                          const name = uniqueStrategyName('固定比例策略', prev);
                          return [...prev, { id, type: t, name, rows, cfg: { mode: 'equal' } }];
                        });
                      } else if (t==='risk_budget') {
                        setStrategies(prev => {
                          const rows = assetNames.map(n=> ({ className:n, budget:100, weight: null }));
                          const name = uniqueStrategyName('风险预算策略', prev);
                          return [...prev, { id, type: t, name, rows, cfg: { risk_metric:'vol', window_mode:'rollingN', data_len:60 } }];
                        });
                      } else {
                        setStrategies(prev => {
                          const rows = assetNames.map(n=> ({ className:n, weight: null }));
                          const name = uniqueStrategyName('指定目标策略', prev);
                          return [...prev, { id, type: t, name, rows, cfg: { target:'min_risk', return_metric:'cumulative', window_mode:'all', data_len:60 } }];
                        });
                      }
                      setShowAddPicker(false);
                    }}>{t==='fixed'?'固定比例': t==='risk_budget'?'风险预算':'指定目标'}</button>
                  ))}
                  <button className="ml-2 rounded bg-white border px-2 py-1 text-xs" onClick={()=> setShowAddPicker(false)}>取消</button>
                </div>
              )}

          {/* 策略回测 */}
          <div className="rounded-lg border p-4">
            <h3 className="font-medium text-gray-700">策略回测</h3>
            <div className="mt-2 flex items-center gap-3">
              <label className="text-sm text-gray-600">选择开始日期</label>
              <input type="date" value={btStart} onChange={e=> setBtStart(e.target.value)} className="rounded border-gray-300 px-2 py-1"/>
              <button
                ref={backtestButtonRef}
                disabled={btBusy}
                className="rounded bg-indigo-600 px-3 py-2 text-sm text-white disabled:opacity-50"
                onClick={async ()=>{
                try{
                  if(!selectedAlloc){ alert('请先选择方案'); return; }
                  setBtBusy(true);
                  const markersDraft: Record<string, ScheduleEntry> = { ...scheduleMarkers };
                  const prepared: StrategyRow[] = [];
                  for (const strategy of strategies) {
                    let working: StrategyRow = { ...strategy, rows: strategy.rows.map(r => ({ ...r })) };
                    if (strategy.type !== 'fixed') {
                      if (strategy.rebalance?.enabled && strategy.rebalance?.recalc) {
                        const specKey = getScheduleSpecKey(strategy);
                        const cached = specKey ? markersDraft[strategy.name] : undefined;
                        if (!cached || cached.spec !== specKey) {
                          const { entry, lastWeights } = await fetchScheduleWeights(strategy);
                          markersDraft[strategy.name] = entry;
                          if (lastWeights.length) {
                            working = applyWeightVector(working, lastWeights);
                          }
                        } else {
                          const lastWeights = cached.markers.length ? cached.markers[cached.markers.length - 1].weights : [];
                          if (lastWeights.length) {
                            working = applyWeightVector(working, lastWeights);
                          }
                        }
                      } else {
                        const weights = await fetchPointWeights(strategy);
                        working = applyWeightVector(working, weights);
                      }
                    }
                    prepared.push(working);
                  }
                  setScheduleMarkers(markersDraft);
                  setStrategies(prepared);

                  const payload = {
                    alloc_name: selectedAlloc,
                    start_date: btStart || undefined,
                    strategies: prepared.map((s) => {
                      const specKey = getScheduleSpecKey(s);
                      const entry = specKey ? markersDraft[s.name] : undefined;
                      const precomputedKey = specKey && entry && entry.spec === specKey ? entry.cacheKey ?? undefined : undefined;
                      return {
                        type: s.type,
                        name: s.name,
                        classes: s.rows.map((r) => ({ name: r.className, weight: (r.weight ?? 0) / 100, budget: r.budget })),
                        rebalance: s.rebalance,
                        model: buildBacktestModel(s),
                        precomputed: precomputedKey,
                      };
                    }),
                  };
                  const res = await fetch('/api/strategy/backtest',{ method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
                  const dat = await res.json();
                  if(!res.ok) throw new Error(dat.detail||'回测失败');
                  setBtSeries(dat);
                }catch(e:any){ alert(e?.message||'回测失败'); }
                finally { setBtBusy(false); }
              }}>开始策略回测</button>
            </div>
            {btSeries && (
              <div className="mt-4 space-y-6">
                <ReactECharts
                  style={{height: 360}}
                  onEvents={{ datazoom: handleBacktestZoom }}
                  option={{
                  tooltip: { 
                    trigger:'axis',
                    formatter: (params: any) => {
                      const arr = Array.isArray(params) ? params : [params];
                      const header = arr[0]?.axisValue || '';
                      const lines = arr.map((p: any) => {
                        if (p.seriesType === 'scatter' && p.data && p.data.weights) {
                          const names = btSeries.asset_names || [];
                          const ws: number[] = p.data.weights || [];
                          const weightLines = names.map((n: string, i: number) => `${n}: ${(ws[i]*100).toFixed(2)}%`).join('<br/>');
                          return `${p.seriesName}: ${Number(p.data.value[1]).toFixed(2)}<br/>${weightLines}`;
                        }
                        return `${p.seriesName}: ${Number(p.data).toFixed(2)}`;
                      });
                      return `${header}<br/>${lines.join('<br/>')}`;
                    }
                  },
                  legend: {
                    type: 'scroll',
                    orient: 'horizontal',
                    top: 0,
                    left: 16,
                    right: 16,
                    height: 60,
                    itemWidth: 12,
                    itemHeight: 8,
                    itemGap: 16,
                    textStyle: { fontSize: 11, overflow: 'break', width: 80 },
                  },
                  dataZoom: [
                    { type: 'inside', xAxisIndex: 0, filterMode: 'none' },
                    { type: 'slider', xAxisIndex: 0, filterMode: 'none', bottom: 24, height: 20 }
                  ],
                  grid: { top: 90, right: 10, bottom: 80, left: 60 },
                  xAxis: { type:'category', data: btSeries.dates },
                  yAxis: {
                    type:'value',
                    name:'组合净值',
                    axisLabel: { formatter: (v: any) => Number(v).toFixed(2) },
                    scale: true,
                    ...(btYAxisRange ? { min: btYAxisRange.min, max: btYAxisRange.max } : {})
                  },
                  series: [
                    ...Object.keys(btSeries.series||{}).map((k:string)=> ({ name:k, type:'line', showSymbol:false, data: btSeries.series[k] })),
                    ...Object.keys(btSeries.markers||{}).flatMap((k:string)=> {
                      const raw = btSeries.markers[k] || [];
                      if (!Array.isArray(raw) || raw.length === 0) return [];
                      const arr = raw.map((m:any)=> ({ value: [m.date, m.value], weights: m.weights }));
                      return arr.length > 0 ? [{ name: `${k}-rebal`, type:'scatter', symbolSize:6, data: arr }] : [];
                    })
                  ]
                }}/>
                {backtestMetricColumns.length > 0 && (
                  <div>
                    <h4 className="text-sm font-semibold mb-2">横向指标对比</h4>
                    <HorizontalMetricComparison
                      columns={backtestMetricColumns}
                      rows={backtestMetricRows}
                      height={DEFAULT_METRIC_TABLE_HEIGHT}
                    />
                    <div className="mt-4">
                      <h5 className="text-sm font-semibold mb-2 text-gray-700">收益风险象限图</h5>
                      <PerformanceQuadrantChart
                        columns={backtestMetricColumns}
                        rows={backtestMetricRows}
                        defaultXAxis="年化波动率(%)"
                        defaultYAxis="累计收益率(%)"
                      />
                    </div>
                  </div>
                )}
              </div>
            )}

            {/*（移除：回测模块内的重复添加按钮）*/}
          </div>
        </div>
      </Section>
    </div>
  );
}
