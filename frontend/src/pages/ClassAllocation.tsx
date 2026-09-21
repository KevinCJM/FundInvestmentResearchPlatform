import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import ReactECharts from 'echarts-for-react';
import { AllocationMetricsReview } from '../components/HorizontalMetricComparison';
import { buildAnnualMetricRows } from '../utils/performance';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import {
  requestEqualWeights,
} from '../services/strategyWeights';
import { assertNativeNumericalExecution } from '../utils/fixedNjitExecution';
import {
  HistoricalRegimeBacktestSelector,
  RegimeConditioningPanel,
} from '../components/HistoricalRegimeBacktest';
import type { HistoricalRegimeBacktestReference } from '../services/portfolioRegime';
import { PortfolioRiskSection } from '../components/risk-models/PublishedRiskPanel';
import { apiErrorMessage } from '../utils/apiError'
import PitDecisionNotice from '../components/PitDecisionNotice'
import { createTaaBaseline } from '../services/tacticalAllocation'
import { FrontierGridControls, FrontierGridResults, defaultFrontierGrid, frontierGridIssue, type FrontierGridSettings } from '../components/frontier-grid/FrontierGrid'
import { allocationJourneyPath, readAllocationDraft, readAllocationJourney, updateAllocationJourney, writeAllocationDraft } from '../app/allocationJourney'

// Helper component for section titles
function Section({ title, children, plain = false }: { title: string; children: React.ReactNode; plain?: boolean }) {
  return (
    <div className={plain ? "border-t border-slate-200 py-5" : "mt-6 rounded-xl border border-slate-200 bg-white p-4 sm:p-6"}>
      {plain ? <h3 className="text-base font-semibold text-slate-800">{title}</h3> : <h2 className="text-lg font-semibold text-slate-800">{title}</h2>}
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
const allocationLabPath = (allocationName: string, universeId: string) => {
  const query = new URLSearchParams()
  if (allocationName) query.set('alloc', allocationName)
  if (universeId) query.set('universe', universeId)
  return `/pre-investment/saa/allocation-lab${query.size ? `?${query}` : ''}`
}

export default function ClassAllocation() {
  const [params] = useSearchParams();
  const journey = readAllocationJourney();
  const allocationName = params.get('alloc') ?? journey.allocationName ?? '';
  const universeId = params.get('universe') ?? journey.universeId ?? '';
  return <ClassAllocationEditor key={`${universeId}:${allocationName}`} requestedAllocation={allocationName} universeId={universeId} />;
}

type StrategyType = 'fixed' | 'risk_budget' | 'target';
type StrategyRow = { id: string; type: StrategyType; name: string; rows: { className: string; weight?: number | null; budget?: number }[]; cfg: any; rebalance?: any };
type GroupLimit = { id: string; assets: string[]; lo: number; hi: number };
type RoundConf = { id: string; samples: number; step: number; buckets: number };
type AllocationDraft = {
  startDate?: string; endDate?: string; btStart?: string; researchGoal?: string;
  strategies?: StrategyRow[]; assetNames?: string[];
  singleLimits?: Record<string, { lo: number; hi: number }>; groupLimits?: GroupLimit[];
  returnMetric?: string; riskMetric?: string; returnType?: string; riskFreePct?: number;
  annualDaysRet?: number; ewmAlpha?: number; ewmWindow?: number; annualDaysRisk?: number;
  ewmAlphaRisk?: number; ewmWindowRisk?: number; confidence?: number;
  explorationSeed?: number;
  rounds?: RoundConf[]; quantStep?: 'none' | '0.001' | '0.002' | '0.005'; useRefine?: boolean; refineCount?: number;
  historicalRegime?: HistoricalRegimeBacktestReference | null;
  frontierGrid?: FrontierGridSettings;
};

function ClassAllocationEditor({ requestedAllocation, universeId }: { requestedAllocation: string; universeId: string }) {
  const navigate = useNavigate();
  const draftScope = `saa:${universeId || 'local'}:${requestedAllocation || 'new'}`;
  const [initialDraft] = useState(() => readAllocationDraft<AllocationDraft>(draftScope) ?? {});
  const [researchGoal, setResearchGoal] = useState(initialDraft.researchGoal ?? 'manual');
  const [draftNotice, setDraftNotice] = useState(Boolean(initialDraft.strategies?.length));
  const backtestInputRef = useRef('');
  const frontierInputRef = useRef('');
  // State for UI interaction
  const [returnMetric, setReturnMetric] = useState(initialDraft.returnMetric ?? 'annual_mean');
  const [riskMetric, setRiskMetric] = useState(initialDraft.riskMetric ?? 'annual_vol');
  const [startDate, setStartDate] = useState(initialDraft.startDate ?? '2020-01-01');
  const [endDate, setEndDate] = useState(initialDraft.endDate ?? readAllocationJourney().researchDate ?? new Date().toISOString().split('T')[0]);

  // State for results
  const [frontierData, setFrontierData] = useState<any>(null);
  const [isCalculating, setIsCalculating] = useState(false);

  // State for loading data
  const [allocations, setAllocations] = useState<string[]>([]);
  const [selectedAlloc, setSelectedAlloc] = useState(requestedAllocation);
  const [configDetails, setConfigDetails] = useState<ConfigDetail[] | null>(null);
  const [assetNames, setAssetNames] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [equalWeightLoading, setEqualWeightLoading] = useState(false);

  // Form states for dynamic inputs
  const [annualDaysRet, setAnnualDaysRet] = useState(initialDraft.annualDaysRet ?? 252);
  const [ewmAlpha, setEwmAlpha] = useState(initialDraft.ewmAlpha ?? 0.94);
  const [ewmWindow, setEwmWindow] = useState(initialDraft.ewmWindow ?? 60);
  const [annualDaysRisk, setAnnualDaysRisk] = useState(initialDraft.annualDaysRisk ?? 252);
  const [ewmAlphaRisk, setEwmAlphaRisk] = useState(initialDraft.ewmAlphaRisk ?? 0.94);
  const [ewmWindowRisk, setEwmWindowRisk] = useState(initialDraft.ewmWindowRisk ?? 60);
  const [confidence, setConfidence] = useState(initialDraft.confidence ?? 95);
  const [returnType, setReturnType] = useState(initialDraft.returnType ?? 'simple');
  // 年化无风险收益率（%）
  const [riskFreePct, setRiskFreePct] = useState(initialDraft.riskFreePct ?? 1.5);

  // 约束：单一上下限 per asset
  const [singleLimits, setSingleLimits] = useState<Record<string, { lo: number; hi: number }>>(initialDraft.singleLimits ?? {});
  // 约束：组上下限列表
  const [groupLimits, setGroupLimits] = useState<GroupLimit[]>(initialDraft.groupLimits ?? []);

  // 随机探索参数：多轮
  const [rounds, setRounds] = useState<RoundConf[]>(initialDraft.rounds ?? [
    { id: 'r0', samples: 1000, step: 0.5, buckets: 10 }, // 第0轮不使用分桶参数
    { id: 'r1', samples: 2000, step: 0.4, buckets: 10 },
    { id: 'r2', samples: 3000, step: 0.3, buckets: 20 },
    { id: 'r3', samples: 4000, step: 0.2, buckets: 30 },
    { id: 'r4', samples: 5000, step: 0.15, buckets: 40 },
    { id: 'r5', samples: 5000, step: 0.1, buckets: 50 },
  ]);
  const [explorationSeed, setExplorationSeed] = useState(initialDraft.explorationSeed ?? 42);
  const [scatterView, setScatterView] = useState<'all' | 'selected'>('all');
  // 权重量化
  const [quantStep, setQuantStep] = useState<'none' | '0.001' | '0.002' | '0.005'>(initialDraft.quantStep ?? 'none');
  // 固定签名 NJIT 受约束局部精炼
  const [useRefine, setUseRefine] = useState(initialDraft.useRefine ?? false);
  const [refineCount, setRefineCount] = useState(initialDraft.refineCount ?? 20);
  // Grid density has its own draft fields: never reinterpret local-refine iterations.
  const [frontierGrid, setFrontierGrid] = useState<FrontierGridSettings>(initialDraft.frontierGrid ?? defaultFrontierGrid);
  const gridIssue = frontierGridIssue(frontierGrid, quantStep !== 'none');
  const latestFrontierKey = useRef('');
  const frontierRequestSequence = useRef(0);

  // ---- 策略制定与回测 ----
  type ScheduleEntry = { markers: { date: string; weights: number[] }[]; cacheKey?: string | null; spec?: string | null };
  const [strategies, setStrategies] = useState<StrategyRow[]>(initialDraft.strategies ?? []);
  const [btStart, setBtStart] = useState<string>(initialDraft.btStart ?? initialDraft.startDate ?? '2020-01-01');
  const [navCount, setNavCount] = useState<number>(0);
  const [btSeries, setBtSeries] = useState<any>(null);
  const [historicalRegime, setHistoricalRegime] = useState<HistoricalRegimeBacktestReference | null>(initialDraft.historicalRegime ?? null);
  const [scheduleMarkers, setScheduleMarkers] = useState<Record<string, ScheduleEntry>>({});
  const [busyStrategy, setBusyStrategy] = useState<string | null>(null);
  const studyBounds = useRef('');
  studyBounds.current = stableStringify({ selectedAlloc, endDate });
  useEffect(() => {
    setScheduleMarkers({});
    setStrategies(items => items.map(item => item.type === 'fixed' ? item : { ...item, rows: item.rows.map(row => ({ ...row, weight: null })) }));
  }, [endDate]);
  const [showAddPicker, setShowAddPicker] = useState(false);
  const [btBusy, setBtBusy] = useState(false);
  const pageRef = useRef<HTMLDivElement | null>(null);
  const backtestButtonRef = useRef<HTMLButtonElement | null>(null);
  const [overlayOffset, setOverlayOffset] = useState<number | null>(null);
  const [btYAxisRange, setBtYAxisRange] = useState<{ min: number; max: number } | null>(null);

  const [taaBusy, setTaaBusy] = useState<string | null>(null);
  const [loadedAllocation, setLoadedAllocation] = useState('');
  const enterTacticalResearch = async (strategy: StrategyRow) => {
    if (!loadedAllocation || loadedAllocation !== selectedAlloc) {
      setError('请先加载当前方案，再将所选策略锁定为 SAA。'); return;
    }
    if (strategy.rows.some(row => row.weight == null || !Number.isFinite(row.weight))) {
      setError('请先计算或填写该策略的全部大类权重。'); return;
    }
    if (!endDate) { setError('请填写方案日期后再进入 TAA。'); return; }
    setTaaBusy(strategy.id); setError('');
    try {
      const baseline = await createTaaBaseline({
        alloc_name: loadedAllocation, name: `${loadedAllocation} · ${strategy.name}`,
        as_of: endDate,
        weights: Object.fromEntries(strategy.rows.map(row => [row.className, Number(row.weight) / 100])),
        constraints: Object.fromEntries(assetNames.map(name => [name, {
          min_weight: singleLimits[name]?.lo ?? 0, max_weight: singleLimits[name]?.hi ?? 1, max_abs_tilt: .1,
        }])),
        group_limits: groupLimits,
      });
      const journey = updateAllocationJourney({ universeId: universeId || undefined, allocationName: loadedAllocation, baselineId: baseline.id });
      navigate(allocationJourneyPath('taa', journey));
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'SAA 基线保存失败。'); }
    finally { setTaaBusy(null); }
  };

  const draftInput: AllocationDraft = { startDate, endDate, btStart, researchGoal, assetNames,
    strategies: strategies.map(strategy => strategy.type === 'fixed' ? strategy : { ...strategy, rows: strategy.rows.map(row => ({ ...row, weight: null })) }),
    singleLimits, groupLimits, returnMetric, riskMetric, returnType, riskFreePct, annualDaysRet, ewmAlpha,
    ewmWindow, annualDaysRisk, ewmAlphaRisk, ewmWindowRisk, confidence, rounds, explorationSeed, quantStep, useRefine, refineCount, historicalRegime, frontierGrid };
  const draftKey = stableStringify(draftInput);
  useEffect(() => {
    if (loadedAllocation === requestedAllocation && loadedAllocation) writeAllocationDraft(draftScope, draftInput);
  }, [draftScope, draftKey, loadedAllocation, requestedAllocation]);

  const frontierInputKey = stableStringify({ selectedAlloc, startDate, endDate, singleLimits, groupLimits, returnMetric, riskMetric, returnType, riskFreePct, annualDaysRet, ewmAlpha, ewmWindow, annualDaysRisk, ewmAlphaRisk, ewmWindowRisk, confidence, rounds, explorationSeed, quantStep, useRefine, refineCount, frontierGrid });
  latestFrontierKey.current = frontierInputKey;
  const backtestInputKey = stableStringify({ selectedAlloc, btStart, endDate, strategies, historicalRegime, singleLimits, groupLimits, riskFreePct });
  useEffect(() => {
    if (frontierData && frontierInputRef.current !== frontierInputKey) { setFrontierData(null); setDraftNotice(true); }
  }, [frontierInputKey, frontierData]);
  useEffect(() => {
    if (btSeries && backtestInputRef.current !== backtestInputKey) { setBtSeries(null); setDraftNotice(true); }
  }, [backtestInputKey, btSeries]);

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
      end_date: endDate || undefined,
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
  }, [selectedAlloc, btStart, endDate, buildTargetConstraints, riskFreePct]);

  const buildComputeWeightsPayload = useCallback((strategy: StrategyRow) => {
    if (!selectedAlloc) return null;
    if (strategy.type === 'fixed') return null;
    const windowMode = strategy.cfg?.window_mode || 'all';
    const base: any = {
      alloc_name: selectedAlloc,
      end_date: endDate || undefined,
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
  }, [selectedAlloc, endDate, buildTargetConstraints, riskFreePct]);

  const fetchPointWeights = useCallback(
    async (strategy: StrategyRow) => {
      const expectedStudy = studyBounds.current;
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
        throw new Error(apiErrorMessage(data, fallback));
      }
      if (studyBounds.current !== expectedStudy) throw new Error('研究结束日或方案已变化，请重新计算权重。');
      assertNativeNumericalExecution(data?.execution, '大类权重求解');
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
        end_date: schedule.end_date ?? null,
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
      const expectedStudy = studyBounds.current;
      const payload = buildSchedulePayload(strategy);
      if (!payload) throw new Error('缺少方案配置，请先选择资产配置方案');
      const response = await fetch('/api/strategy/compute-schedule-weights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(apiErrorMessage(data, '批量调仓权重计算失败'));
      }
      if (studyBounds.current !== expectedStudy) throw new Error('研究结束日或方案已变化，请重新计算调仓计划。');
      assertNativeNumericalExecution(data?.execution, '批量调仓权重计算');
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
        annualRows: [] as any[],
      };
    }
    const columns = metrics.map((m: any) => m?.name ?? '');
    const cumulativeValues = metrics.map((metric: any) =>
      parseMetricValue(metric.cumulative_return)
    );
    const cumulativePercentValues = cumulativeValues.map(v => (Number.isFinite(v) ? v * 100 : NaN));
    const rows = [
      { label: '累计收益率(%)', values: cumulativePercentValues },
      { label: '年化收益率(%)', values: metrics.map((m: any) => parseMetricValue(m.annual_return) * 100) },
      { label: '年化波动率(%)', values: metrics.map((m: any) => parseMetricValue(m.annual_vol) * 100) },
      { label: '夏普比率', values: metrics.map((m: any) => parseMetricValue(m.sharpe)) },
      { label: '99%VaR(日)(%)', values: metrics.map((m: any) => parseMetricValue(m.var99) * 100), reverseScale: true },
      { label: '99%ES(日)(%)', values: metrics.map((m: any) => parseMetricValue(m.es99) * 100), reverseScale: true },
      { label: '最大回撤(%)', values: metrics.map((m: any) => parseMetricValue(m.max_drawdown) * 100) },
      { label: '卡玛比率', values: metrics.map((m: any) => parseMetricValue(m.calmar)) },
    ];
    const annualRows = buildAnnualMetricRows(columns, btSeries?.annual_metrics ?? {
      years: [],
      series: {},
    });
    return { columns, rows, annualRows };
  }, [btSeries?.annual_metrics, btSeries?.metrics, parseMetricValue]);

  const backtestMetricColumns = backtestMetricsSummary.columns;
  const backtestMetricRows = backtestMetricsSummary.rows;

  const loadEqualPercents = useCallback(async (names: string[]): Promise<number[]> => {
    if (names.length === 0) throw new Error('请先加载至少一个资产大类');
    setEqualWeightLoading(true);
    try {
      const result = await requestEqualWeights(names.length);
      return result.weights;
    } finally {
      setEqualWeightLoading(false);
    }
  }, []);

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
          if (!requestedAllocation) setSelectedAlloc(data[0]);
          else if (!data.includes(requestedAllocation)) setError(`未找到大类方案“${requestedAllocation}”，请重新选择。`)
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

  // Loading is driven by the URL, so back/forward selects the exact same research draft.
  useEffect(() => {
    if (!requestedAllocation) return;
    let active = true;
    const load = async () => {
      setLoading(true); setError('');
      try {
        const res = await fetch(`/api/load-allocation?name=${encodeURIComponent(requestedAllocation)}`);
        if (!res.ok) throw new Error('加载大类方案失败，请重新选择或返回大类构建检查。');
        const data = await res.json();
        const details: ConfigDetail[] = data.flatMap((ac: any) => ac.etfs.map((etf: any) => ({
          className: ac.name, code: etf.code, name: etf.name, weight: `${Number(etf.weight).toFixed(2)}%`,
        })));
        if (!active) return;
        const names = Array.from(new Set(details.map(d => d.className)));
        setConfigDetails(details); setAssetNames(names); setLoadedAllocation(requestedAllocation);
        if (stableStringify(names) !== stableStringify(initialDraft.assetNames)) {
          setSingleLimits(Object.fromEntries(names.map(name => [name, { lo: 0, hi: 1 }])));
          setGroupLimits([]); setStrategies([]);
          if (initialDraft.strategies?.length) setError('大类组成已变化，已保留研究日期；请按当前大类重新设置权重与约束。');
        }
        updateAllocationJourney({ allocationName: requestedAllocation, universeId: universeId || undefined });
        const range = await fetch(`/api/strategy/default-start?alloc_name=${encodeURIComponent(requestedAllocation)}`);
        const available = await range.json();
        if (active && range.ok) {
          if (typeof available.count === 'number') setNavCount(available.count);
          if (!initialDraft.btStart && available.default_start) setBtStart(available.default_start > startDate ? available.default_start : startDate);
        }
      } catch (caught) {
        if (active) setError(caught instanceof Error ? caught.message : '加载方案失败');
      } finally { if (active) setLoading(false); }
    };
    void load();
    return () => { active = false; };
  }, [requestedAllocation, universeId]);

  const handleSelectAndLoad = () => {
    if (!selectedAlloc) { setError('请选择一个已保存的大类方案。'); return; }
    updateAllocationJourney({ allocationName: selectedAlloc, universeId: universeId || undefined });
    navigate(allocationLabPath(selectedAlloc, universeId));
  };

  const addFixedStrategy = async () => {
    if (!assetNames.length) { setError('请先加载大类方案。'); return; }
    setError('');
    try {
      const weights = await loadEqualPercents(assetNames);
      setStrategies(current => [...current, { id: `s${Date.now()}`, name: uniqueStrategyName('固定比例策略', current), type: 'fixed',
        rows: assetNames.map((name, index) => ({ className: name, weight: weights[index] })), cfg: { mode: 'custom' } }]);
    } catch (caught) { setError(caught instanceof Error ? caught.message : '无法生成初始权重。'); }
  };

  const adoptCandidate = (label: string, point: any) => {
    const names: string[] = frontierData?.asset_names ?? [];
    if (!Array.isArray(point?.weights) || point.weights.length !== names.length || !point.weights.every(Number.isFinite)) {
      setError('这个候选没有完整的大类权重，请重新计算。'); return;
    }
    setStrategies(current => [...current, { id: `s${Date.now()}`, name: uniqueStrategyName(`${label}配置`, current), type: 'fixed',
      rows: names.map((className, index) => ({ className, weight: Number((point.weights[index] * 100).toFixed(8)) })),
      cfg: { mode: 'custom', selection_reason: `采用${label}候选；样本 ${startDate} 至 ${endDate}` } }]);
    setResearchGoal('manual');
  };

  const returnLabel = ({ annual: '年化收益率', annual_mean: '年化平均收益率', cumulative: '累计收益率', mean: '日均收益率', ewm: '加权日收益率' } as Record<string, string>)[returnMetric] ?? '收益率';
  const riskLabel = ({ vol: '日波动率', annual_vol: '年化波动率', ewm_vol: '加权波动率', var: 'VaR', es: '预期尾部损失', max_drawdown: '最大回撤', downside_vol: '下行波动率' } as Record<string, string>)[riskMetric] ?? '风险';
  const candidates = frontierData ? [
    { label: '较低风险', key: 'min_variance' }, { label: '较高收益风险比', key: 'max_sharpe' }, { label: '较高收益', key: 'max_return' },
  ].filter(item => frontierData[item.key]) : [];

  const onCalculate = async () => {
    if (gridIssue) { setError(gridIssue); return; }
    const expectedInput = latestFrontierKey.current;
    const requestSequence = ++frontierRequestSequence.current;
    if (!startDate || !endDate || startDate > endDate) { setError('请选择完整研究区间，开始日不能晚于结束日。'); return; }
    if (!selectedAlloc || loadedAllocation !== selectedAlloc) {
      setError('请先加载当前大类方案，再计算候选。');
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
        seed: explorationSeed,
        rounds: rounds.map(r => ({ samples: r.samples, step: r.step, buckets: r.buckets }))
      },
      quantization: { step: quantStep === 'none' ? 'none' : Number(quantStep) },
      refine: { enabled: useRefine, method: 'bounded_pairwise_pattern_search_njit', iterations: refineCount },
      frontier_grid: frontierGrid.enabled ? {
        point_count: frontierGrid.point_count,
        max_iterations: frontierGrid.max_iterations,
        weight_domain: 'continuous',
        accept_continuous_weights: frontierGrid.accept_continuous_weights,
      } : undefined,
    };

    try {
      setIsCalculating(true);
      setError('');
      setFrontierData(null);
      const res = await fetch('/api/efficient-frontier', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (requestSequence !== frontierRequestSequence.current || expectedInput !== latestFrontierKey.current) return;
      if (!res.ok) throw new Error(apiErrorMessage(data, '计算失败'));
      assertNativeNumericalExecution(data?.execution, '大类配置有效前沿');
      frontierInputRef.current = frontierInputKey;
      setFrontierData(data);
    } catch (e: any) {
      if (requestSequence === frontierRequestSequence.current && expectedInput === latestFrontierKey.current)
        setError(e.message || '计算失败，请检查研究日期与数据质量。');
    } finally {
      if (requestSequence === frontierRequestSequence.current) setIsCalculating(false);
    }
  };

  return (
    <div ref={pageRef} className="mx-auto min-w-0 max-w-6xl p-3 sm:p-6 relative">
      {(loading || isCalculating || btBusy) && (
        <div className={`absolute inset-0 z-50 flex justify-center bg-black/40 rounded-xl ${overlayOffset !== null ? 'items-start' : 'items-center'}`}>
          <div
            className="rounded-xl bg-white px-6 py-4 shadow text-sm"
            style={overlayOffset !== null ? { marginTop: overlayOffset } : undefined}
          >
            {btBusy ? '计算中...' : (isCalculating ? '正在计算，请稍候...' : '正在加载...')}
          </div>
        </div>
      )}

      
      <h1 className="text-2xl font-semibold">大类资产配置</h1>
      <p className="text-sm text-slate-600 mt-1">先确定长期资金比例，再检验历史表现，最后进入 TAA 研究短期调整。</p>
      <p className="mt-2 text-sm text-slate-600">{loadedAllocation ? `当前大类：${loadedAllocation} · ${assetNames.length} 类` : '选择已保存的大类方案开始。'} <Link className="ml-2 underline" to={allocationJourneyPath('classes')}>返回大类构建</Link></p>
      {draftNotice && <p role="status" className="mt-3 rounded-lg bg-amber-50 p-3 text-sm text-amber-900">已保留研究输入。历史结果需在当前数据口径下重新计算。</p>}
      {error && <p role="alert" className="sticky top-2 z-40 mt-3 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800">{error}</p>}

      <section aria-labelledby="frontier-workspace-title" className="mt-6 min-w-0 rounded-xl border border-slate-200 bg-white p-4 sm:p-6 [&_button]:min-h-10 [&_input:not([type=checkbox])]:min-h-10 [&_select]:min-h-10">
      <h2 id="frontier-workspace-title" className="text-lg font-semibold text-slate-800">可配置空间与有效前沿</h2>
      <p className="my-3 text-sm leading-6 text-slate-600">在同一处配置指标、单项与联合约束、多轮随机游走、权重精度和前沿目标，再查看并采用结果。</p>
      <Section plain title="选择大类构建方案">
        {loading && <p>正在加载方案列表...</p>}
        {!loading && (
          <div className="flex flex-wrap items-center gap-3">
            <select 
              aria-label="大类构建方案"
              value={selectedAlloc}
              onChange={event => { updateAllocationJourney({ allocationName: event.target.value, universeId: universeId || undefined }); navigate(allocationLabPath(event.target.value, universeId)); }}
              className="min-w-0 max-w-full flex-grow rounded-lg border-slate-300 shadow-sm focus:border-accent-500 focus:ring-accent-500">
              {allocations.map(name => <option key={name} value={name}>{name}</option>)}
            </select>
            <button 
              onClick={handleSelectAndLoad}
              disabled={Boolean(loadedAllocation) && loadedAllocation === selectedAlloc}
              className="rounded-lg bg-accent-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-accent-700">
              {loadedAllocation === selectedAlloc && loadedAllocation ? '已加载' : '选择该方案'}
            </button>
          </div>
        )}
        {configDetails && (
          <details className="mt-3 rounded-lg border p-3">
            <summary className="cursor-pointer text-sm">查看类内产品及权重（每类合计 100%）</summary>
            <div className="mt-2 max-h-80 overflow-auto">
              <table className="min-w-full divide-y divide-slate-200">
              <thead className="bg-slate-50">
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium tracking-wide r text-slate-600">大类名称</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium tracking-wide r text-slate-600">产品代码</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium tracking-wide r text-slate-600">产品名称</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium tracking-wide r text-slate-600">类内权重</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 bg-white">
                {configDetails.map((item, index) => (
                  <tr key={index}>
                    <td className="whitespace-nowrap px-6 py-4 text-sm text-slate-900">{item.className}</td>
                    <td className="whitespace-nowrap px-6 py-4 text-sm text-slate-600 font-mono">{item.code}</td>
                    <td className="whitespace-nowrap px-6 py-4 text-sm text-slate-900">{item.name}</td>
                    <td className="whitespace-nowrap px-6 py-4 text-sm text-slate-600">{item.weight}</td>
                  </tr>
                ))}
              </tbody>
              </table>
            </div>
          </details>
        )}
      </Section>

      <Section plain title="长期配置目标与研究区间">
        <div className="grid gap-3 sm:grid-cols-3">
          <label className="text-sm">我想先做什么<select aria-label="长期配置目标" value={researchGoal} onChange={event => setResearchGoal(event.target.value)} className="mt-1 w-full rounded-lg border p-2"><option value="manual">填写长期权重</option><option value="compare">比较不同收益与风险的候选</option></select></label>
          <label className="text-sm">研究区间开始<input aria-label="研究区间开始" type="date" value={startDate} max={endDate} onChange={event => { setStartDate(event.target.value); setBtStart(event.target.value); }} className="mt-1 w-full rounded-lg border p-2" /></label>
          <label className="text-sm">方案日期 / 构建区间结束<input aria-label="研究区间结束" type="date" value={endDate} min={startDate} onChange={event => setEndDate(event.target.value)} className="mt-1 w-full rounded-lg border p-2" /></label>
        </div>
        <p className="mt-2 text-xs text-slate-600">候选、目标网格和策略回测共享上述结束日，实际样本受数据覆盖限制；不是正式 PIT 认证。</p>

      </Section>
      <Section plain title="大类资金边界（占整个组合的 %）">
        {/* 权重约束设置 */}
        <div className="mt-6 grid grid-cols-1 gap-6 md:grid-cols-2">
          <div className="min-w-0 space-y-3">
            <h3 className="font-medium text-slate-700">单个大类资金范围</h3>
            {assetNames.length === 0 ? (
              <p className="text-sm text-slate-600 mt-2">请先选择并加载方案</p>
            ) : (
              <div className="mt-3 space-y-2">
                <div className="grid grid-cols-3 gap-2 text-xs text-slate-600"><span>大类</span><span>最低 %</span><span>最高 %</span></div>
                {assetNames.map(name => (
                  <div key={name} className="grid grid-cols-3 items-center gap-2">
                    <div className="text-sm text-slate-700">{name}</div>
                    <input aria-label={`${name} 最低权重 (%)`} type="number" min={0} max={100} step={1}
                      value={Number(((singleLimits[name]?.lo ?? 0) * 100).toFixed(4))}
                      onChange={e => setSingleLimits(prev => ({ ...prev, [name]: { ...(prev[name]||{lo:0,hi:1}), lo: Number(e.target.value) / 100 } }))}
                      className="min-w-0 w-full rounded-lg border-slate-300 px-2 py-1 text-sm" placeholder="最低 %" />
                    <input aria-label={`${name} 最高权重 (%)`} type="number" min={0} max={100} step={1}
                      value={Number(((singleLimits[name]?.hi ?? 1) * 100).toFixed(4))}
                      onChange={e => setSingleLimits(prev => ({ ...prev, [name]: { ...(prev[name]||{lo:0,hi:1}), hi: Number(e.target.value) / 100 } }))}
                      className="min-w-0 w-full rounded-lg border-slate-300 px-2 py-1 text-sm" placeholder="最高 %" />
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="min-w-0 space-y-3">
            <h3 className="font-medium text-slate-700">多个大类合计范围</h3>
            <p className="mt-1 text-xs text-slate-600">例如：权益与商品合计不超过 60%。不需要时留空。</p>
            <div className="mt-2 space-y-3">
              {groupLimits.map((g, idx) => (
                <div key={g.id} className="border-b border-slate-200 py-3">
                  <div className="flex flex-wrap gap-2">
                    {assetNames.map(n => (
                      <label key={n} className="flex items-center gap-1 text-xs">
                        <input type="checkbox" checked={g.assets.includes(n)} onChange={e => {
                          setGroupLimits(prev => prev.map(x => x.id===g.id ? { ...x, assets: e.target.checked ? [...x.assets, n] : x.assets.filter(a => a!==n) } : x))
                        }} />{n}
                      </label>
                    ))}
                  </div>
                  <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-slate-600"><span>最低 %</span><span>最高 %</span><span /></div>
                  <div className="mt-1 grid grid-cols-3 gap-2">
                    <input aria-label={`联合约束 ${idx + 1} 最低权重 (%)`} type="number" min={0} max={100} step={1} value={Number((g.lo * 100).toFixed(4))} onChange={e => setGroupLimits(prev => prev.map(x => x.id===g.id ? { ...x, lo: Number(e.target.value) / 100 } : x))} className="min-w-0 w-full rounded-lg border-slate-300 px-2 py-1 text-sm" placeholder="最低 %" />
                    <input aria-label={`联合约束 ${idx + 1} 最高权重 (%)`} type="number" min={0} max={100} step={1} value={Number((g.hi * 100).toFixed(4))} onChange={e => setGroupLimits(prev => prev.map(x => x.id===g.id ? { ...x, hi: Number(e.target.value) / 100 } : x))} className="min-w-0 w-full rounded-lg border-slate-300 px-2 py-1 text-sm" placeholder="最高 %" />
                    <button onClick={() => setGroupLimits(prev => prev.filter(x => x.id !== g.id))} className="rounded-lg bg-red-50 text-red-700 text-xs px-2">删除</button>
                  </div>
                </div>
              ))}
              <button onClick={() => setGroupLimits(prev => [...prev, { id: `g${Date.now()}`, assets: [], lo: 0, hi: 1 }])} className="rounded-lg bg-slate-100 px-3 py-1 text-xs">+ 添加联合约束</button>
            </div>
          </div>
        </div>

      </Section>
      <Section plain title="收益、风险与随机探索参数">
        <div className="grid grid-cols-1 gap-x-8 gap-y-6 md:grid-cols-2">
          {/* 收益指标 */}
          <div className="space-y-3">
            <h3 className="font-medium text-slate-700">收益指标</h3>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <label className="block text-sm font-medium text-slate-600">收益指标</label>
                <select aria-label="收益指标" onChange={e => setReturnMetric(e.target.value)} value={returnMetric} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm">
                  <option value="annual">年化收益率</option>
                  <option value="annual_mean">年化收益率均值</option>
                  <option value="cumulative">累计收益率</option>
                  <option value="mean">收益率均值</option>
                  <option value="ewm">指数加权收益率</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-600">收益类型</label>
                <select aria-label="收益类型" value={returnType} onChange={e => setReturnType(e.target.value)} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm">
                  <option value="simple">普通收益率</option>
                  <option value="log">对数收益率</option>
                </select>
              </div>
            </div>
            {(returnMetric === 'annual' || returnMetric === 'annual_mean') && (
              <div>
                <label className="block text-sm font-medium text-slate-600">年化天数</label>
                <input aria-label="收益年化天数" type="number" value={annualDaysRet} onChange={e => setAnnualDaysRet(Number(e.target.value))} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm" />
              </div>
            )}
            {returnMetric === 'ewm' && (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <label className="block text-sm font-medium text-slate-600">衰减因子 λ</label>
                  <input aria-label="收益衰减因子" type="number" step="0.01" value={ewmAlpha} onChange={e => setEwmAlpha(Number(e.target.value))} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-600">窗口长度</label>
                  <input aria-label="收益窗口长度" type="number" value={ewmWindow} onChange={e => setEwmWindow(Number(e.target.value))} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm" />
                </div>
              </div>
            )}
          </div>

          {/* 风险指标 */}
          <div className="space-y-3">
            <h3 className="font-medium text-slate-700">风险指标</h3>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                    <label className="block text-sm font-medium text-slate-600">风险指标</label>
                    <select aria-label="风险指标" onChange={e => setRiskMetric(e.target.value)} value={riskMetric} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm">
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
                        <label className="block text-sm font-medium text-slate-600">置信度 %</label>
                        <input aria-label="置信度 (%)" type="number" value={confidence} onChange={e => setConfidence(Number(e.target.value))} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm" />
                    </div>
                )}
            </div>
            {(riskMetric === 'annual_vol') && (
                <div>
                    <label className="block text-sm font-medium text-slate-600">年化天数</label>
                    <input aria-label="风险年化天数" type="number" value={annualDaysRisk} onChange={e => setAnnualDaysRisk(Number(e.target.value))} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm" />
                </div>
            )}
            {(riskMetric === 'ewm_vol') && (
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <div>
                        <label className="block text-sm font-medium text-slate-600">衰减因子 λ</label>
                        <input aria-label="风险衰减因子" type="number" step="0.01" value={ewmAlphaRisk} onChange={e => setEwmAlphaRisk(Number(e.target.value))} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm" />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-slate-600">窗口长度</label>
                    <input aria-label="风险窗口长度" type="number" value={ewmWindowRisk} onChange={e => setEwmWindowRisk(Number(e.target.value))} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm" />
                    </div>
                </div>
            )}
          </div>

          {/* 夏普比率参数 */}
          <div className="space-y-3 md:col-span-2">
            <h3 className="font-medium text-slate-700">夏普比率参数</h3>
            <div>
              <label className="block text-sm font-medium text-slate-600">年化无风险收益率(%)</label>
              <input
                aria-label="无风险收益率 (%)"
                type="number"
                step="0.1"
                value={riskFreePct}
                onChange={e => setRiskFreePct(Number(e.target.value))}
                className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm"
              />
              <p className="mt-1 text-xs text-slate-600">用于计算最大夏普率，默认 1.5%</p>
            </div>
            <p className="text-xs text-slate-600">
              计算逻辑：夏普比率 = (年化收益率均值 - 年化无风险利率) / 年化标准差。
            </p>
          </div>
        </div>

        {/* 随机探索设置 */}
        <div className="mt-6 border-t border-slate-200 pt-4">
          <h3 className="font-medium text-slate-700">多轮随机游走</h3>
          <p className="text-xs text-slate-600 mt-1">首轮随机采样；后续各轮从上轮保留点扰动，投影到约束空间，再按收益分桶选取最低风险点作为下一轮种子。步长越小越靠近种子，不保证每一点都改善。样本点数为尝试预算，实际可行数量见结果。</p>
          <label className="mt-3 block max-w-xs text-sm text-slate-700">随机种子
            <input aria-label="随机种子" type="number" min={0} max={4294967295} step={1} value={explorationSeed} onChange={event => setExplorationSeed(Number(event.target.value))} className="mt-1 w-full rounded-lg border-slate-300 p-2" />
            <span className="mt-1 block text-xs text-slate-600">相同参数和种子可复现；修改种子生成另一组随机探索。</span>
          </label>
          <div className="mt-2 space-y-2">
            {rounds.map((r, idx) => (
              <div key={r.id} className="grid grid-cols-2 items-end gap-3 border-b border-slate-100 py-2 sm:grid-cols-12">
                <div className="col-span-2 text-sm text-slate-600">第{idx}轮</div>
                <label className="min-w-0 text-xs text-slate-600 sm:col-span-3">
                  <span className="whitespace-nowrap">样本点</span>
                  <input aria-label={`第${idx}轮样本点`} type="number" min={1} value={r.samples} onChange={e => setRounds(prev => prev.map(x => x.id===r.id ? { ...x, samples: Number(e.target.value) } : x))} className="mt-1 min-w-0 w-full rounded-lg border-slate-300 px-2 py-1 text-sm" />
                </label>
                <label className="min-w-0 text-xs text-slate-600 sm:col-span-3">
                  <span className="whitespace-nowrap">步长</span>
                  <input aria-label={`第${idx}轮步长`} disabled={idx === 0} title={idx === 0 ? '首轮直接随机采样，不使用扰动步长' : undefined} type="number" step={0.01} min={0} max={1} value={r.step} onChange={e => setRounds(prev => prev.map(x => x.id===r.id ? { ...x, step: Number(e.target.value) } : x))} className="mt-1 min-w-0 w-full rounded-lg border-slate-300 px-2 py-1 text-sm" />
                </label>
                {idx > 0 && (
                  <label className="min-w-0 text-xs text-slate-600 sm:col-span-3">
                    <span className="whitespace-nowrap">分桶</span>
                    <input aria-label={`第${idx}轮分桶`} type="number" min={1} value={(r as any).buckets ?? 50} onChange={e => setRounds(prev => prev.map(x => x.id===r.id ? { ...x, buckets: Number(e.target.value) } : x))} className="mt-1 min-w-0 w-full rounded-lg border-slate-300 px-2 py-1 text-sm" />
                  </label>
                )}
                {idx > 0 && (
                  <button onClick={() => setRounds(prev => prev.filter(x => x.id !== r.id))} aria-label={`删除第${idx}轮`} className="rounded-lg bg-rose-50 text-rose-800 text-xs px-2 sm:col-span-1">删</button>
                )}
              </div>
            ))}
            <button onClick={() => setRounds(prev => [...prev, { id: `r${Date.now()}`, samples: 200, step: 0.5, buckets: 50 }])} className="rounded-lg bg-slate-100 px-3 py-1 text-xs">+ 增加一轮</button>
          </div>
          <div className="mt-3 grid gap-4 sm:grid-cols-2">
            <div>
              <label className="block text-sm font-medium text-slate-600">权重量化</label>
              <select aria-label="权重量化" value={quantStep} onChange={e => setQuantStep(e.target.value as any)} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm">
                <option value="none">不量化</option>
                <option value="0.001">0.1%</option>
                <option value="0.002">0.2%</option>
                <option value="0.005">0.5%</option>
              </select>
            </div>
            <div className="flex flex-wrap items-end gap-2">
              <label className="flex items-center gap-2 text-sm text-slate-700">
                <input type="checkbox" checked={useRefine} onChange={e => setUseRefine(e.target.checked)} /> 使用受约束局部精炼
              </label>
              {useRefine && (
                <input aria-label="局部精炼最大迭代次数" type="number" min={1} max={200} value={refineCount} onChange={e => setRefineCount(Number(e.target.value))} className="w-28 rounded-lg border-slate-300 px-2 py-1 text-sm" placeholder="迭代次数" />
              )}
            </div>
          </div>
        </div>
        <p className="mt-3 text-xs leading-5 text-slate-600">局部精炼改善三个代表候选；下方 20／200 个收益目标用于加密整条前沿。选定精度时，散点与最终可采用权重均须同时满足精度、单项及联合约束。</p>
        <div className="mt-4 border-t border-slate-200 pt-4"><FrontierGridControls value={frontierGrid} quantized={quantStep !== 'none'} busy={isCalculating} onChange={setFrontierGrid} /></div>
        <div className="mt-4 flex flex-wrap gap-3">
          <button disabled={!loadedAllocation || loadedAllocation !== selectedAlloc || equalWeightLoading || isCalculating || (researchGoal !== 'manual' && Boolean(gridIssue))} onClick={researchGoal === 'manual' ? addFixedStrategy : onCalculate} className="rounded-lg bg-accent-600 px-4 py-2 text-sm text-white disabled:opacity-50">{researchGoal === 'manual' ? '填写一组长期权重' : '计算并比较候选'}</button>
          <button disabled={!loadedAllocation || loadedAllocation !== selectedAlloc || isCalculating || Boolean(gridIssue)} onClick={onCalculate} className="rounded-lg border px-4 py-2 text-sm disabled:opacity-50">生成可配置空间与有效前沿</button>
        </div>
        <p className="mt-3 text-xs text-slate-600">采样尝试预算 {rounds.reduce((sum, round) => sum + round.samples, 0).toLocaleString()} 个 · 权重精度 {quantStep === 'none' ? '连续' : `${Number(quantStep) * 100}%`}{frontierGrid.enabled ? ` · 前沿目标 ${frontierGrid.point_count} 个` : ''}</p>
        {isCalculating && <p role="status" className="mt-2 text-sm text-slate-600">正在生成随机探索候选与前沿，请稍候…</p>}
      </Section>

      {frontierData && (
        <Section plain title="比较候选，采用长期配置">
          <p className="mb-3 text-sm text-slate-600">以下是同一区间与约束下的历史候选。采用后可改权重并回测，不代表未来最优。</p>
          {/* 这张前沿图挑出来的就是权重本身，它按哪天、哪个产品域算的必须跟着它走。 */}
          <PitDecisionNotice lineage={frontierData.pit} />
          {frontierData.research_interval && <p className="mb-3 text-xs leading-5 text-slate-600">实际样本：{frontierData.research_interval.actual_start} 至 {frontierData.research_interval.actual_end} · 净值 {frontierData.research_interval.nav_observations} 期 / 收益 {frontierData.research_interval.return_observations} 期 · 采样候选 {frontierData.sampled_candidates ?? frontierData.accepted_candidates ?? frontierData.scatter?.length ?? 0} 个{Number(frontierData.refined_candidates ?? 0) > 0 ? ` + 精炼新增 ${frontierData.refined_candidates} 个` : ''}{Number(frontierData.grid_candidates ?? 0) > 0 ? ` + 网格优化新增 ${frontierData.grid_candidates} 个` : ''} · 当前候选合计 {frontierData.accepted_candidates ?? frontierData.scatter?.length ?? 0} 个 · 有效前沿 {frontierData.frontier_candidates ?? frontierData.frontier?.length ?? 0} 个。</p>}
          {frontierData.refinement?.requested && <p role="status" className="mb-3 rounded-lg bg-slate-50 p-3 text-xs leading-5 text-slate-600">局部精炼仅处理最大夏普、最小风险、最大收益 3 个代表候选，并以原始可行候选为不可退化基准；严格改善的结果会加入候选集合并重新构造前沿。{frontierData.refinement.items?.map((item: any) => `${item.candidate} ${item.status} / ${item.iterations} 次${item.applied ? ' / 已采用' : ' / 未替换原候选'}`).join('；')}。不声明整条前沿的连续求解或全局最优。</p>}
          <div className="overflow-x-auto"><table className="min-w-[640px] w-full text-sm" aria-label="长期配置候选"><thead className="bg-slate-50"><tr><th scope="col" className="p-2 text-left">候选</th>{(frontierData.asset_names ?? []).map((name: string) => <th scope="col" key={name} className="p-2">{name}</th>)}<th scope="col" className="p-2">{returnLabel}</th><th scope="col" className="p-2">{riskLabel}</th><th scope="col" className="p-2">操作</th></tr></thead><tbody>{candidates.map(({ label, key }) => {
            const point = frontierData[key]; const value = point.value ?? point;
            return <tr key={key} className="border-t"><td className="p-2">{label}</td>{(point.weights ?? []).map((weight: number, index: number) => <td key={index} className="p-2 text-center">{(weight * 100).toFixed(2)}%</td>)}<td className="p-2 text-center">{Number.isFinite(value[1]) ? `${(value[1] * 100).toFixed(2)}%` : '—'}</td><td className="p-2 text-center">{Number.isFinite(value[0]) ? `${(value[0] * 100).toFixed(2)}%` : '—'}</td><td className="p-2"><button onClick={() => adoptCandidate(label, point)} className="whitespace-nowrap rounded-lg border border-emerald-700 px-3 py-1 text-emerald-800">采用{label}</button></td></tr>;
          })}</tbody></table></div>
          {frontierData.exploration && <div className="mt-4 space-y-3">
            <label className="block max-w-sm text-sm">散点显示
              <select aria-label="散点显示" value={scatterView} onChange={event => setScatterView(event.target.value as 'all' | 'selected')} className="mt-1 w-full rounded-lg border-slate-300 p-2">
                <option value="all">全部可行候选（含精炼）</option><option value="selected">首轮与各轮分桶保留点</option>
              </select>
            </label>
            <div className="overflow-x-auto"><table aria-label="随机探索轮次统计" className="w-full min-w-[480px] text-sm">
              <thead><tr>{['轮次', '尝试预算', '可行点', '保留点', '失败点'].map(label => <th scope="col" key={label} className="p-2 text-left">{label}</th>)}</tr></thead>
              <tbody>{frontierData.exploration.rounds.map((round: any, index: number) => <tr key={round.round} className="border-t border-slate-200"><th scope="row" className="p-2 text-left">第{index}轮</th><td className="p-2">{round.requested}</td><td className="p-2">{round.accepted}</td><td className="p-2">{round.selected}</td><td className="p-2">{round.rejected}{round.search_budget_failures > 0 ? `（搜索预算耗尽 ${round.search_budget_failures}）` : ''}{round.accepted === 0 ? '；沿用上一有效轮种子' : ''}</td></tr>)}</tbody>
            </table></div>
          </div>}
          {frontierData.frontier_grid && <FrontierGridResults result={frontierData.frontier_grid} assetNames={frontierData.asset_names} riskLabel={riskLabel} returnLabel={returnLabel} onAdopt={point => adoptCandidate(`前沿目标 ${point.target_index + 1}`, point)} />}
          <details open className="mt-4"><summary className="cursor-pointer text-sm">有效前沿图（默认展开，可收起）</summary><ReactECharts
            style={{ height: 500 }}
            notMerge
            option={{
              animation: false,
              title: {
                text: '可配置空间与有效前沿',
                textStyle: { fontSize: 16 },
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
                  const rr = (Number.isFinite(risk) ? `${(Number(risk) * 100).toFixed(2)}%` : '—') + ` (${riskLabel})`;
                  const re = (Number.isFinite(ret) ? `${(Number(ret) * 100).toFixed(2)}%` : '—') + ` (${returnLabel})`;
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
                { type: 'slider', xAxisIndex: 0, filterMode: 'none', bottom: 12, height: 18 },
                { type: 'slider', yAxisIndex: 0, filterMode: 'none', right: 0, width: 12 },
              ],
              legend: {
                type: 'scroll',
                top: 36,
                right: 0,
                left: 'center',
                data: ['其他组合', frontierData.weight_domain === 'discrete' && frontierData.frontier_grid ? '连续理论前沿' : '有效前沿', ...(frontierData.frontier_grid ? [frontierData.weight_domain === 'discrete' ? '指定精度前沿' : '候选非支配点'] : []), '最大夏普率', '最小风险', '最大收益']
              },
              grid: { top: 80, bottom: 80, left: 12, right: 30, containLabel: true },
              xAxis: { type: 'value', name: riskLabel, nameLocation: 'middle', nameGap: 28, scale: true, splitNumber: 3, axisLabel: { hideOverlap: true, fontSize: 12, formatter: (value: number) => `${(value * 100).toFixed(1)}%` } },
              yAxis: { type: 'value', name: returnLabel, nameLocation: 'middle', nameGap: 42, scale: true, axisLabel: { hideOverlap: true, fontSize: 12, formatter: (value: number) => `${(value * 100).toFixed(1)}%` } },
              series: [
                ...(frontierData.scatter ? [{
                  name: '其他组合',
                  type: 'scatter',
                  symbolSize: 3,
                  data: scatterView === 'selected' && frontierData.exploration ? frontierData.exploration.selected_indices.map((index: number) => frontierData.scatter[index]) : frontierData.scatter,
                  itemStyle: { color: 'rgba(128, 128, 128, 0.35)' }
                }] : []),
                ...(frontierData.frontier ? [{
                  name: frontierData.weight_domain === 'discrete' && frontierData.frontier_grid ? '连续理论前沿' : '有效前沿',
                  type: frontierData.frontier_grid ? 'line' : 'scatter',
                  connectNulls: false,
                  smooth: false,
                  symbolSize: 6,
                  data: frontierData.frontier_grid ? frontierData.frontier_grid.curve.map((point: any) => point ?? { value: [null, null] }) : frontierData.frontier,
                  itemStyle: { color: '#2563eb' } // Tailwind indigo-600
                }] : []),
                ...(frontierData.frontier_grid ? [{
                  name: frontierData.weight_domain === 'discrete' ? '指定精度前沿' : '候选非支配点', type: 'scatter', symbolSize: 3, data: frontierData.frontier,
                }] : []),
                ...(frontierData.max_sharpe ? [{
                  name: '最大夏普率',
                  type: 'scatter',
                  symbolSize: 10,
                  data: [frontierData.max_sharpe]
                }] : []),
                ...(frontierData.min_variance ? [{
                  name: '最小风险',
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
          /></details>
        </Section>
      )}

      </section>

      {/* 大类资产策略制定与回测 */}
      <Section title="长期权重与历史验证">
        <p className="mb-4 rounded-lg bg-emerald-50 p-3 text-sm text-emerald-900">进入 TAA 将继承本策略权重、类内产品映射及单项/联合边界。TAA 会按自己的研究日期、调仓与费用重新计算 SAA 对照；本页历史业绩不会直接带入。</p>
        {!strategies.length && <p className="mb-3 text-sm text-slate-600">可以填写一组长期权重，也可以先采用上方候选。</p>}
        <div className="space-y-4">
          {/* 顶部不再显示“添加策略”按钮，统一放在策略列表与回测之间 */}

          {strategies.map((s, idx) => (
            <div key={s.id} className="rounded-lg border p-4">
              <div className="flex flex-wrap items-center gap-3">
                <input value={s.name} onChange={e => setStrategies(prev => prev.map(x => x.id===s.id? { ...x, name: e.target.value } : x))} className="rounded-lg border-slate-300 px-2 py-1 text-sm" />
                <span className="rounded-lg bg-slate-100 px-2 py-1 text-xs text-slate-700">
                  {s.type === 'fixed' ? '固定比例' : s.type === 'risk_budget' ? '风险预算' : '指定目标'}
                </span>
                <button type="button" disabled={taaBusy !== null || loadedAllocation !== selectedAlloc}
                  onClick={() => enterTacticalResearch(s)}
                  className="rounded-lg border border-emerald-700 px-3 py-2 text-sm font-medium text-emerald-800 disabled:opacity-50">
                  {taaBusy === s.id ? '正在锁定 SAA…' : '以此为 SAA，研究战术偏离'}
                </button>
                <button onClick={() => setStrategies(prev => prev.filter(x => x.id !== s.id))} className="ml-auto rounded-lg bg-red-50 px-2 py-1 text-xs text-red-700">删除</button>
              </div>
              {/* 再平衡设置（通用） */}
              <div className="mt-3 rounded-lg border p-3 text-sm">
                <label className="flex items-center gap-2"><input type="checkbox" checked={!!s.rebalance?.enabled} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, rebalance: { ...(x.rebalance||{}), enabled: e.target.checked } } : x))}/> 是否启用再平衡</label>
                {s.rebalance?.enabled && (
                  <div className="mt-2 grid grid-cols-12 items-center gap-2">
                    <div className="col-span-3">
                      <label className="block text-xs text-slate-600">再平衡方式</label>
                      <select value={s.rebalance?.mode||'monthly'} onChange={e=> setStrategies(prev=> prev.map(x=> {
                        if (x.id!==s.id) return x as any;
                        const mode = e.target.value;
                        const rb = { ...(x.rebalance||{}), mode } as any;
                        if (mode === 'fixed' && !rb.fixedInterval) {
                          rb.fixedInterval = Math.max(1, navCount||1);
                        }
                        return { ...x, rebalance: rb } as any;
                      }))} className="mt-1 w-full rounded-lg border-slate-300">
                        <option value="weekly">每周</option>
                        <option value="monthly">每月</option>
                        <option value="yearly">每年</option>
                        <option value="fixed">固定区间</option>
                      </select>
                    </div>
                    {s.rebalance?.mode !== 'fixed' ? (
                      <>
                    <div className="col-span-2">
                      <label className="block text-xs text-slate-600">第N</label>
                      <input type="number" min={1} max={ s.rebalance?.mode==='weekly'?5: s.rebalance?.mode==='monthly'?30:360 } value={s.rebalance?.N ?? 1} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, rebalance: { ...(x.rebalance||{}), which:'nth', N: Number(e.target.value) } } : x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                    </div>
                        <div className="col-span-2">
                        <label className="block text-xs text-slate-600">个</label>
                        <div className="mt-1 text-sm text-slate-600">&nbsp;</div>
                        </div>
                        <div className="col-span-2">
                          <label className="block text-xs text-slate-600">单位</label>
                          <select value={s.rebalance?.unit||'trading'} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, rebalance: { ...(x.rebalance||{}), unit: e.target.value } } : x))} className="mt-1 w-full rounded-lg border-slate-300">
                            <option value="trading">交易日</option>
                            <option value="natural">自然日</option>
                          </select>
                        </div>
                      </>
                    ) : (
                      <div className="col-span-3">
                        <label className="block text-xs text-slate-600">固定区间(天)</label>
                        <input type="number" min={1} value={s.rebalance?.fixedInterval ?? 20} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, rebalance: { ...(x.rebalance||{}), fixedInterval: Number(e.target.value) } } : x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
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
                    <label className="flex items-center gap-2"><input type="radio" checked={(s.cfg?.mode||'equal')==='equal'} disabled={equalWeightLoading} onChange={async () => {
                      setError('');
                      try {
                        const eqArr = await loadEqualPercents(assetNames);
                        setStrategies(prev => prev.map(x=>{
                          if (x.id!==s.id) return x;
                          return { ...x, cfg:{...x.cfg, mode:'equal'}, rows: assetNames.map((n,i)=> ({ className:n, weight:eqArr[i] })) } as StrategyRow;
                        }));
                      } catch (reason) {
                        setError(reason instanceof Error ? reason.message : '等权计算失败');
                      }
                    }}/> 等权重</label>
                    <label className="flex items-center gap-2"><input type="radio" checked={(s.cfg?.mode||'equal')==='custom'} onChange={() => setStrategies(prev => prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, mode:'custom'}, rows: x.rows }:x))}/> 自定义权重</label>
                  </div>
                  <p className="text-xs text-slate-600">{(s.cfg?.mode || 'equal') === 'equal' ? '各大类平均分配资金。' : '直接填写占整个组合的比例，合计应为 100%。'}</p>
                  {s.cfg?.selection_reason && <label className="block text-sm">选择理由<input aria-label={`${s.name} 选择理由`} className="mt-1 w-full rounded-lg border p-2" value={s.cfg.selection_reason} onChange={event => setStrategies(current => current.map(item => item.id === s.id ? { ...item, cfg: { ...item.cfg, selection_reason: event.target.value } } : item))} /></label>}
                  <div className="min-w-0 overflow-x-auto rounded-lg border">
                    <table className="min-w-full">
                      <thead className="bg-slate-50 text-xs text-slate-600"><tr><th scope="col" className="px-3 py-2 text-left">大类名称</th><th scope="col" className="px-3 py-2 text-left">资金权重(%)</th></tr></thead>
                      <tbody className="text-sm">
                        {s.rows.map((r,i)=> (
                          <tr key={i} className="border-t">
                            <td className="px-3 py-2">{r.className}</td>
                            <td className="px-3 py-2"><input aria-label={`${s.name} ${r.className} 权重 (%)`} type="number" min="0" max="100" value={r.weight ?? ''} onChange={e=>{
                              const v = e.target.value === '' ? undefined : Number(e.target.value);
                              setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, rows:x.rows.map((rr,j)=> j===i?{...rr, weight:v}:rr)}:x))
                            }} className={`w-28 rounded-lg border px-2 py-1 ${ (s.cfg?.mode||'equal')==='equal' ? 'bg-slate-50 text-slate-600 border-slate-200' : 'border-slate-300' }`} disabled={(s.cfg?.mode||'equal')==='equal'}/></td>
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
                  <div className="min-w-0 overflow-x-auto rounded-lg border">
                    <table className="min-w-full">
                      <thead className="bg-slate-50 text-xs text-slate-600"><tr><th scope="col" className="px-3 py-2 text-left">大类名称</th><th scope="col" className="px-3 py-2 text-left">风险预算(%)</th><th scope="col" className="px-3 py-2 text-left">资金权重(%)</th></tr></thead>
                      <tbody className="text-sm">
                        {s.rows.map((r,i)=> (
                          <tr key={i} className="border-t">
                            <td className="px-3 py-2">{r.className}</td>
                            <td className="px-3 py-2"><input type="number" value={r.budget ?? 100} onChange={e=>{
                              const v = Number(e.target.value);
                              setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, rows:x.rows.map((rr,j)=>j===i?{...rr, budget:v}:rr)}:x))
                            }} className="w-28 rounded-lg border-slate-300 px-2 py-1"/></td>
                            <td className="px-3 py-2">{r.weight==null? '-' : (r.weight?.toFixed(2))}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <label className="block text-xs text-slate-600">风险指标</label>
                      <select value={s.cfg?.risk_metric||'vol'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_metric:e.target.value}}:x))} className="mt-1 w-full rounded-lg border-slate-300">
                        <option value="vol">波动率</option>
                        <option value="var">VaR</option>
                        <option value="es">ES</option>
                        <option value="downside_vol">下行波动率</option>
                        <option value="max_drawdown">最大回撤</option>
                      </select>
                    </div>
                    {['var','es'].includes(s.cfg?.risk_metric) && (
                      <><div>
                        <label className="block text-xs text-slate-600">置信度(%)</label>
                        <input type="number" value={s.cfg?.confidence ?? 95} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, confidence:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                      </div>
                      <div>
                        <label className="block text-xs text-slate-600">天数</label>
                        <input type="number" value={s.cfg?.days ?? 252} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, days:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                      </div></>
                    )}
                  </div>
                  {/* 模型计算区间 */}
                  <div className="rounded-lg border p-3 text-sm">
                    <div className="grid grid-cols-3 gap-3 items-end">
                      <div>
                        <label className="block text-xs text-slate-600">窗口模式</label>
                        <select value={s.cfg?.window_mode || 'all'} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, cfg: { ...(x.cfg||{}), window_mode: e.target.value } } : x))} className="mt-1 w-full rounded-lg border-slate-300">
                          <option value="all">所有数据</option>
                          <option value="rollingN">最近N条</option>
                        </select>
                      </div>
                      { (s.cfg?.window_mode==='rollingN') && (
                        <div>
                          <label className="block text-xs text-slate-600">N（交易日）</label>
                          <input type="number" min={2} value={s.cfg?.data_len ?? 60} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, cfg: { ...(x.cfg||{}), data_len: Number(e.target.value) } } : x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                        </div>
                      )}
                    </div>
                    <p className="mt-2 text-xs text-slate-600">
                      模式说明：
                      <span className="ml-1 font-medium">所有数据</span> 使用回测开始至当期的全部样本；
                      <span className="ml-1 font-medium">最近N条</span> 使用当期之前最近 N 条的滚动窗口（推荐）。
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    <button
                      disabled={busyStrategy===s.id}
                      className="rounded-lg bg-slate-100 px-3 py-1 text-sm disabled:opacity-50"
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
                          setError(e?.message || '权重计算失败');
                        }finally{
                          setBusyStrategy(null);
                          setBtBusy(false);
                        }
                      }}
                    >反推资金权重</button>
                  </div>
                  {busyStrategy===s.id && <div className="text-xs text-slate-600">计算中，请稍候…</div>}

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
                        <div className="rounded-lg border overflow-x-auto">
                          <table className="min-w-full whitespace-nowrap text-sm">
                            <thead className="bg-slate-50">
                              <tr>
                                <th scope="col" className="px-3 py-2 text-left text-xs text-slate-600">大类名称</th>
                                {dates.map((d) => (
                                  <th scope="col" key={d} className="px-3 py-2 text-left text-xs text-slate-600">{d}</th>
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
                  <div className="grid grid-cols-2 gap-4 text-sm rounded-lg border p-3">
                    <div>
                      <label className="block text-xs text-slate-600">目标类型</label>
                      <select value={s.cfg?.target||'min_risk'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, target:e.target.value}}:x))} className="mt-1 w-full rounded-lg border-slate-300">
                        <option value="min_risk">最小风险</option>
                        <option value="max_return">最大收益</option>
                        <option value="max_sharpe">最大化收益风险性价比</option>
                        <option value="max_sharpe_traditional">最大化夏普比率</option>
                        <option value="risk_min_given_return">指定收益下最小风险</option>
                        <option value="return_max_given_risk">指定风险下最大收益</option>
                      </select>
                      
                      {/* Explanations for each target type */}
                      {(s.cfg?.target === 'min_risk' || !s.cfg?.target) && (
                        <p className="mt-2 text-xs text-slate-600 bg-slate-50 p-2 rounded-lg">
                          目标：在满足所有约束条件下，寻找使组合风险（由指定的<strong>风险指标</strong>衡量）最小化的权重。
                        </p>
                      )}
                      {s.cfg?.target === 'max_return' && (
                        <p className="mt-2 text-xs text-slate-600 bg-slate-50 p-2 rounded-lg">
                          目标：在满足所有约束条件下，寻找使组合收益（由指定的<strong>收益指标</strong>衡量）最大化的权重。
                        </p>
                      )}
                      {s.cfg?.target === 'max_sharpe' && (
                        <p className="mt-2 text-xs text-slate-600 bg-slate-50 p-2 rounded-lg">
                          目标：寻找使 <strong>(指定收益指标) / (指定风险指标)</strong> 比值最大化的权重。这是一个广义的收益风险性价比优化。
                        </p>
                      )}
                      {s.cfg?.target === 'max_sharpe_traditional' && (
                        <p className="mt-2 text-xs text-slate-600 bg-slate-50 p-2 rounded-lg">
                          目标：寻找使传统夏普比率 <code>(年化收益 - 无风险利率) / 年化波动率</code> 最大化的权重。
                        </p>
                      )}
                      {s.cfg?.target === 'risk_min_given_return' && (
                        <p className="mt-2 text-xs text-slate-600 bg-slate-50 p-2 rounded-lg">
                          目标：在组合收益等于<strong>目标收益值</strong>的前提下，寻找使组合风险最小化的权重。
                        </p>
                      )}
                      {s.cfg?.target === 'return_max_given_risk' && (
                        <p className="mt-2 text-xs text-slate-600 bg-slate-50 p-2 rounded-lg">
                          目标：在组合风险不高于<strong>目标风险值</strong>的前提下，寻找使组合收益最大化的权重。
                        </p>
                      )}
                    </div>
                    <div>
                      <label className="block text-xs text-slate-600">收益率类型</label>
                      <select value={s.cfg?.return_type||'simple'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, return_type:e.target.value}}:x))} className="mt-1 w-full rounded-lg border-slate-300">
                        <option value="simple">普通收益率</option>
                        <option value="log">对数收益率</option>
                      </select>
                    </div>
                  </div>

                  {/* 根据目标类型显示不同UI */}
                  {s.cfg?.target === 'max_sharpe_traditional' ? (
                    <div className="space-y-3 rounded-lg border p-3 text-sm">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-xs text-slate-600">收益指标 (固定)</label>
                          <input type="text" value="年化收益率均值" disabled className="mt-1 w-full rounded-lg border-slate-200 bg-slate-100 px-2 py-1"/>
                        </div>
                        <div>
                          <label className="block text-xs text-slate-600">风险指标 (固定)</label>
                          <input type="text" value="年化波动率" disabled className="mt-1 w-full rounded-lg border-slate-200 bg-slate-100 px-2 py-1"/>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-xs text-slate-600">年化天数</label>
                          <input type="number" value={s.cfg?.days ?? 252} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, days:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                        </div>
                        <div>
                          <label className="block text-xs text-slate-600">年化无风险利率(%)</label>
                          <input type="number" step="0.1" value={s.cfg?.risk_free_rate_pct ?? 1.5} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_free_rate_pct:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <>
                      {/* 收益指标配置 */}
                      <div className="rounded-lg border p-3 text-sm space-y-3">
                        <div>
                          <label className="block text-xs text-slate-600">收益指标</label>
                          <select value={s.cfg?.return_metric||'cumulative'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, return_metric:e.target.value}}:x))} className="mt-1 w-full rounded-lg border-slate-300">
                            <option value="annual">年化收益率</option>
                            <option value="annual_mean">年化收益率均值</option>
                            <option value="cumulative">累计收益率</option>
                            <option value="mean">收益率均值</option>
                            <option value="ewm">指数加权收益率</option>
                          </select>
                        </div>
                        {(s.cfg?.return_metric==='annual' || s.cfg?.return_metric==='annual_mean') && (
                          <div>
                            <label className="block text-xs text-slate-600">年化天数</label>
                            <input type="number" value={s.cfg?.days ?? 252} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, days:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                          </div>
                        )}
                        {s.cfg?.return_metric==='ewm' && (
                          <div className="grid grid-cols-2 gap-3">
                            <div>
                              <label className="block text-xs text-slate-600">衰减因子 λ</label>
                              <input type="number" step={0.01} value={s.cfg?.ret_alpha ?? 0.94} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, ret_alpha:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                            </div>
                            <div>
                              <label className="block text-xs text-slate-600">窗口长度</label>
                              <input type="number" value={s.cfg?.ret_window ?? 60} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, ret_window:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                            </div>
                          </div>
                        )}
                      </div>

                      {/* 风险指标配置 */}
                      <div className="rounded-lg border p-3 text-sm space-y-3">
                        <div>
                          <label className="block text-xs text-slate-600">风险指标</label>
                          <select value={s.cfg?.risk_metric||'vol'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_metric:e.target.value}}:x))} className="mt-1 w-full rounded-lg border-slate-300">
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
                            <label className="block text-xs text-slate-600">年化天数</label>
                            <input type="number" value={s.cfg?.risk_days ?? 252} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_days:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                          </div>
                        )}
                        {s.cfg?.risk_metric==='ewm_vol' && (
                          <div className="grid grid-cols-2 gap-3">
                            <div>
                              <label className="block text-xs text-slate-600">衰减因子 λ</label>
                              <input type="number" step={0.01} value={s.cfg?.risk_alpha ?? 0.94} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_alpha:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                            </div>
                            <div>
                              <label className="block text-xs text-slate-600">窗口长度</label>
                              <input type="number" value={s.cfg?.risk_window ?? 60} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_window:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                            </div>
                          </div>
                        )}
                        {(s.cfg?.risk_metric==='var' || s.cfg?.risk_metric==='es') && (
                          <div>
                            <label className="block text-xs text-slate-600">置信度%</label>
                            <input type="number" value={s.cfg?.risk_confidence ?? 95} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_confidence:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                          </div>
                        )}
                      </div>
                    </>
                  )}
                  {(s.cfg?.target==='risk_min_given_return') && (
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <label className="block text-xs text-slate-600">目标收益率 (%)</label>
                        <input type="number" value={s.cfg?.target_return == null ? '' : Number((s.cfg.target_return * 100).toFixed(4))} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, target_return: Number(e.target.value) / 100}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                      </div>
                    </div>
                  )}
                  {(s.cfg?.target==='return_max_given_risk') && (
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <label className="block text-xs text-slate-600">目标风险 (%)</label>
                        <input type="number" value={s.cfg?.target_risk == null ? '' : Number((s.cfg.target_risk * 100).toFixed(4))} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, target_risk: Number(e.target.value) / 100}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
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
                      <div className="rounded-lg border overflow-x-auto">
                        <table className="min-w-full whitespace-nowrap text-sm">
                          <thead className="bg-slate-50">
                            <tr>
                              <th scope="col" className="px-3 py-2 text-left text-xs text-slate-600">大类名称</th>
                              {dates.map((d) => (
                                <th scope="col" key={d} className="px-3 py-2 text-left text-xs text-slate-600">{d}</th>
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
                    <div className="min-w-0 overflow-x-auto rounded-lg border">
                      <table className="min-w-full">
                        <thead className="bg-slate-50 text-xs text-slate-600"><tr><th scope="col" className="px-3 py-2 text-left">大类名称</th><th scope="col" className="px-3 py-2 text-left">资金权重(%)</th></tr></thead>
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
                  <div className="rounded-lg border p-3 text-sm">
                    <div className="grid grid-cols-3 gap-3 items-end">
                      <div>
                        <label className="block text-xs text-slate-600">窗口模式</label>
                        <select value={s.cfg?.window_mode || 'rollingN'} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, cfg: { ...(x.cfg||{}), window_mode: e.target.value } } : x))} className="mt-1 w-full rounded-lg border-slate-300">
                          <option value="all">所有数据</option>
                          <option value="rollingN">最近N条</option>
                        </select>
                      </div>
                      { (s.cfg?.window_mode==='rollingN') && (
                        <div>
                          <label className="block text-xs text-slate-600">N（交易日）</label>
                          <input type="number" min={2} value={s.cfg?.data_len ?? 60} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, cfg: { ...(x.cfg||{}), data_len: Number(e.target.value) } } : x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                        </div>
                      )}
                    </div>
                    <p className="mt-2 text-xs text-slate-600">
                      模式说明：
                      <span className="ml-1 font-medium">所有数据</span> 使用回测开始至当期的全部样本；
                      <span className="ml-1 font-medium">最近N条</span> 使用当期之前最近 N 条的滚动窗口（推荐）。
                    </p>
                  </div>

                  <div>
                    <button
                      disabled={busyStrategy===s.id}
                      className="rounded-lg bg-slate-100 px-3 py-1 text-sm disabled:opacity-50"
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
                          setError(e?.message || '权重计算失败');
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
                  if (assetNames.length === 0) { setError('请先加载方案'); return; }
                  setShowAddPicker(true);
                }}
                className="rounded-lg bg-accent-600 text-white px-3 py-2 text-sm">
                + 添加新的组合策略
              </button>
            </div>
          )}
              {showAddPicker && (
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <span className="text-sm text-slate-700">选择策略类型：</span>
                  {(['fixed','risk_budget','target'] as StrategyType[]).map(t => (
                    <button key={t} disabled={equalWeightLoading} className="rounded-lg bg-slate-100 px-3 py-1 text-sm disabled:cursor-wait disabled:opacity-60" onClick={async () => {
                      const id = `s${Date.now()}`;
                      if (t === 'fixed') {
                        setError('');
                        try {
                          const eqArr = await loadEqualPercents(assetNames);
                          setStrategies(prev => {
                          const rows = assetNames.map((n,i)=> ({ className:n, weight: eqArr[i], budget: 100 }));
                          const name = uniqueStrategyName('固定比例策略', prev);
                          return [...prev, { id, type: t, name, rows, cfg: { mode: 'equal' } }];
                          });
                        } catch (reason) {
                          setError(reason instanceof Error ? reason.message : '等权计算失败');
                          return;
                        }
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
                  <button className="ml-2 rounded-lg bg-white border px-2 py-1 text-xs" onClick={()=> setShowAddPicker(false)}>取消</button>
                </div>
              )}

          {/* 策略回测 */}
          <div className="rounded-lg border p-4">
            <h3 className="font-medium text-slate-700">策略回测</h3>
            <div className="mt-3">
              <HistoricalRegimeBacktestSelector
                value={historicalRegime}
                disabled={btBusy}
                onChange={(value) => {
                  setHistoricalRegime(value);
                  setBtSeries(null);
                }}
              />
            </div>
            <p className="mt-2 text-xs text-slate-600">回测、权重求解与上方候选构建共享结束日；实际区间以有效数据覆盖为准。调仓方式见各策略，当前回测为未扣交易费用的历史表现。</p>
            <div className="mt-2 flex flex-wrap items-center gap-3">
              <label htmlFor="saa-backtest-start" className="text-sm text-slate-600">回测开始日期</label>
              <input id="saa-backtest-start" type="date" value={btStart} onChange={e=> setBtStart(e.target.value)} className="rounded-lg border-slate-300 px-2 py-1"/>
              <label htmlFor="saa-backtest-end" className="text-sm text-slate-600">回测结束日期</label>
              <input id="saa-backtest-end" type="date" value={endDate} onChange={e => setEndDate(e.target.value)} className="rounded-lg border-slate-300 px-2 py-1" />
              <button
                ref={backtestButtonRef}
                disabled={btBusy || !strategies.length || loadedAllocation !== selectedAlloc}
                className="rounded-lg bg-accent-600 px-3 py-2 text-sm text-white disabled:opacity-50"
                onClick={async ()=>{
                try{
                  if(!selectedAlloc || loadedAllocation !== selectedAlloc){ setError('请先加载当前大类方案。'); return; }
                  setBtBusy(true);
                  const expectedStudy = studyBounds.current;
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
                    end_date: endDate || undefined,
                    historical_regime: historicalRegime ?? undefined,
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
                  if (studyBounds.current !== expectedStudy) throw new Error('研究结束日或方案已变化，请重新运行回测。');
                  if(!res.ok) throw new Error(apiErrorMessage(dat, '回测失败'));
                  assertNativeNumericalExecution(dat?.execution, '大类配置策略回测');
                  if (dat?.regime_conditioning) {
                    assertNativeNumericalExecution(dat.regime_conditioning.execution, '组合历史情景条件统计');
                  }
                  backtestInputRef.current = stableStringify({ selectedAlloc, btStart, endDate, strategies: prepared, historicalRegime, singleLimits, groupLimits, riskFreePct });
                  setBtSeries(dat);
                }catch(e:any){ setError(e?.message||'回测失败'); }
                finally { setBtBusy(false); }
              }}>开始策略回测</button>
            </div>
            {btSeries && (
              <div className="mt-4 space-y-6">
                <PitDecisionNotice lineage={btSeries.pit} />
                {btSeries.research_interval && <p className="text-xs text-slate-600">实际回测区间：{btSeries.research_interval.actual_start ?? '—'} 至 {btSeries.research_interval.actual_end ?? '—'}。</p>}
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
                  <AllocationMetricsReview
                    columns={backtestMetricColumns}
                    rows={backtestMetricRows}
                    annualRows={backtestMetricsSummary.annualRows}
                  />
                )}
                <RegimeConditioningPanel result={btSeries.regime_conditioning} />
              </div>
            )}

            {/*（移除：回测模块内的重复添加按钮）*/}
          </div>
        </div>
      </Section>
      <details className="mt-5 rounded-xl border bg-white p-4"><summary className="cursor-pointer text-sm">已保存组合的风险压测</summary><PortfolioRiskSection context="选择由本配置生成的已保存产品组合，测试实际持仓；压测结果不会更改这里的大类权重。" /></details>
    </div>
  );
}
