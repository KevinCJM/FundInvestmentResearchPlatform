import { systemText, useI18n, i18n } from '../i18n/runtime'
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
import ResumeResearch from '../components/ResumeResearch'
import { createTaaBaseline } from '../services/tacticalAllocation'
import { FrontierGridControls, FrontierGridResults, defaultFrontierGrid, frontierGridIssue, type FrontierGridSettings } from '../components/frontier-grid/FrontierGrid'
import { allocationJourneyPath, readAllocationDraft, readAllocationJourney, updateAllocationJourney, writeAllocationDraft } from '../app/allocationJourney'

// Helper component for section titles
function Section({ title, children, plain = false }: { title: string; children: React.ReactNode; plain?: boolean }) {
  useI18n()
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
  useI18n()
  const [params] = useSearchParams();
  const journey = readAllocationJourney();
  // 地址栏是唯一事实来源：没带身份就当新一轮研究，续接由用户点续接条显式发起。
  const allocationName = params.get('alloc') ?? '';
  const universeId = params.get('universe') ?? '';
  return <>
    <ResumeResearch to={allocationLabPath(journey.allocationName ?? '', journey.universeId ?? '')} />
    <ClassAllocationEditor key={`${universeId}:${allocationName}`} requestedAllocation={allocationName} universeId={universeId} />
  </>;
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
  useI18n()
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
      setError(systemText('preInvestment.classAllocation.loadTheCurrentPlanBeforeLockingThe')); return;
    }
    if (strategy.rows.some(row => row.weight == null || !Number.isFinite(row.weight))) {
      setError(systemText('preInvestment.classAllocation.calculateOrEnterAllAssetClassWeights')); return;
    }
    if (!endDate) { setError(systemText('preInvestment.classAllocation.enterThePolicyDateBeforeContinuingTo')); return; }
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
    } catch (caught) { setError(caught instanceof Error ? caught.message : systemText('preInvestment.classAllocation.unableToSaveTheSaaBaseline')); }
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
      if (!payload) throw new Error(systemText('preInvestment.classAllocation.missingPlanConfigurationSelectAnAllocationPlan'));
      const response = await fetch('/api/strategy/compute-weights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        const fallback = strategy.type === 'risk_budget' ? systemText('preInvestment.classAllocation.riskBudgetWeightCalculationFailed') : systemText('preInvestment.classAllocation.targetWeightCalculationFailed');
        throw new Error(apiErrorMessage(data, fallback));
      }
      if (studyBounds.current !== expectedStudy) throw new Error(systemText('preInvestment.classAllocation.theResearchEndDateOrPlanChanged'));
      assertNativeNumericalExecution(data?.execution, systemText('preInvestment.classAllocation.assetClassWeightSolution'));
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
      if (!payload) throw new Error(systemText('preInvestment.classAllocation.missingPlanConfigurationSelectAnAllocationPlan'));
      const response = await fetch('/api/strategy/compute-schedule-weights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(apiErrorMessage(data, systemText('preInvestment.classAllocation.batchRebalancingWeightCalculationFailed')));
      }
      if (studyBounds.current !== expectedStudy) throw new Error(systemText('preInvestment.classAllocation.theResearchEndDateOrPlanChanged2'));
      assertNativeNumericalExecution(data?.execution, systemText('preInvestment.classAllocation.batchRebalancingWeightCalculation'));
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
      { label: systemText('preInvestment.classAllocation.cumulativeReturn'), values: cumulativePercentValues },
      { label: systemText('preInvestment.classAllocation.annualReturn'), values: metrics.map((m: any) => parseMetricValue(m.annual_return) * 100) },
      { label: systemText('preInvestment.classAllocation.annualVolatility'), values: metrics.map((m: any) => parseMetricValue(m.annual_vol) * 100) },
      { label: systemText('preInvestment.classAllocation.sharpeRatio'), values: metrics.map((m: any) => parseMetricValue(m.sharpe)) },
      { label: systemText('preInvestment.classAllocation.99VarDaily'), values: metrics.map((m: any) => parseMetricValue(m.var99) * 100), reverseScale: true },
      { label: systemText('preInvestment.classAllocation.99EsDaily'), values: metrics.map((m: any) => parseMetricValue(m.es99) * 100), reverseScale: true },
      { label: systemText('preInvestment.classAllocation.maximumDrawdown'), values: metrics.map((m: any) => parseMetricValue(m.max_drawdown) * 100) },
      { label: systemText('preInvestment.classAllocation.calmarRatio'), values: metrics.map((m: any) => parseMetricValue(m.calmar)) },
    ];
    const annualRows = buildAnnualMetricRows(columns, btSeries?.annual_metrics ?? {
      years: [],
      series: {},
    });
    return { columns, rows, annualRows };
  }, [btSeries?.annual_metrics, btSeries?.metrics, parseMetricValue, i18n.language]);

  const backtestMetricColumns = backtestMetricsSummary.columns;
  const backtestMetricRows = backtestMetricsSummary.rows;

  const loadEqualPercents = useCallback(async (names: string[]): Promise<number[]> => {
    if (names.length === 0) throw new Error(systemText('preInvestment.classAllocation.loadAtLeastOneAssetClassFirst'));
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
        if (!res.ok) throw new Error(systemText('preInvestment.classAllocation.unableToRetrieveThePlanList'));
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) {
          setAllocations(data);
          if (!requestedAllocation) setSelectedAlloc(data[0]);
          else if (!data.includes(requestedAllocation)) setError(systemText('preInvestment.classAllocation.assetClassPlanNotFoundSelectAnother', { p0: requestedAllocation }))
        } else {
          setError(systemText('preInvestment.classAllocation.noSavedAssetClassSchemesFoundSave'));
        }
      } catch (e: any) {
        setError(e.message || systemText('preInvestment.classAllocation.unableToLoadThePlanList'));
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
        if (!res.ok) throw new Error(systemText('preInvestment.classAllocation.unableToLoadTheAssetClassPlan'));
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
          if (initialDraft.strategies?.length) setError(systemText('preInvestment.classAllocation.assetCompositionChangedResearchDatesAreRetained'));
        }
        updateAllocationJourney({ allocationName: requestedAllocation, universeId: universeId || undefined });
        const range = await fetch(`/api/strategy/default-start?alloc_name=${encodeURIComponent(requestedAllocation)}`);
        const available = await range.json();
        if (active && range.ok) {
          if (typeof available.count === 'number') setNavCount(available.count);
          if (!initialDraft.btStart && available.default_start) setBtStart(available.default_start > startDate ? available.default_start : startDate);
        }
      } catch (caught) {
        if (active) setError(caught instanceof Error ? caught.message : systemText('preInvestment.classAllocation.unableToLoadThePlan'));
      } finally { if (active) setLoading(false); }
    };
    void load();
    return () => { active = false; };
  }, [requestedAllocation, universeId]);

  const handleSelectAndLoad = () => {
    if (!selectedAlloc) { setError(systemText('preInvestment.classAllocation.selectASavedAssetClassPlan')); return; }
    updateAllocationJourney({ allocationName: selectedAlloc, universeId: universeId || undefined });
    navigate(allocationLabPath(selectedAlloc, universeId));
  };

  const addFixedStrategy = async () => {
    if (!assetNames.length) { setError(systemText('preInvestment.classAllocation.loadAnAssetClassPlanFirst')); return; }
    setError('');
    try {
      const weights = await loadEqualPercents(assetNames);
      setStrategies(current => [...current, { id: `s${Date.now()}`, name: uniqueStrategyName(systemText('preInvestment.classAllocation.fixedWeightStrategy'), current), type: 'fixed',
        rows: assetNames.map((name, index) => ({ className: name, weight: weights[index] })), cfg: { mode: 'custom' } }]);
    } catch (caught) { setError(caught instanceof Error ? caught.message : systemText('preInvestment.classAllocation.unableToGenerateInitialWeights')); }
  };

  const adoptCandidate = (label: string, point: any) => {
    const names: string[] = frontierData?.asset_names ?? [];
    if (!Array.isArray(point?.weights) || point.weights.length !== names.length || !point.weights.every(Number.isFinite)) {
      setError(systemText('preInvestment.classAllocation.thisCandidateLacksCompleteAssetClassWeights')); return;
    }
    setStrategies(current => [...current, { id: `s${Date.now()}`, name: uniqueStrategyName(systemText('preInvestment.classAllocation.allocation', { p0: label }), current), type: 'fixed',
      rows: names.map((className, index) => ({ className, weight: Number((point.weights[index] * 100).toFixed(8)) })),
      cfg: { mode: 'custom', selection_reason: systemText('preInvestment.classAllocation.adoptCandidateSampleTo', { p0: label, p1: startDate, p2: endDate }) } }]);
    setResearchGoal('manual');
  };

  const returnLabel = ({ annual: systemText('preInvestment.classAllocation.annualizedReturn'), annual_mean: systemText('preInvestment.classAllocation.annualizedMeanReturn'), cumulative: systemText('preInvestment.classAllocation.cumulativeReturn2'), mean: systemText('preInvestment.classAllocation.meanDailyReturn'), ewm: systemText('preInvestment.classAllocation.weightedDailyReturn') } as Record<string, string>)[returnMetric] ?? systemText('preInvestment.classAllocation.return');
  const riskLabel = ({ vol: systemText('preInvestment.classAllocation.dailyVolatility'), annual_vol: systemText('preInvestment.classAllocation.annualVolatility2'), ewm_vol: systemText('preInvestment.classAllocation.weightedVolatility'), var: 'VaR', es: systemText('preInvestment.classAllocation.expectedShortfall'), max_drawdown: systemText('preInvestment.classAllocation.maximumDrawdown2'), downside_vol: systemText('preInvestment.classAllocation.downsideVolatility') } as Record<string, string>)[riskMetric] ?? systemText('preInvestment.classAllocation.risk');
  const candidates = frontierData ? [
    { label: systemText('preInvestment.classAllocation.lowerRisk'), key: 'min_variance' }, { label: systemText('preInvestment.classAllocation.higherReturnToRiskRatio'), key: 'max_sharpe' }, { label: systemText('preInvestment.classAllocation.higherReturn'), key: 'max_return' },
  ].filter(item => frontierData[item.key]) : [];

  const onCalculate = async () => {
    if (gridIssue) { setError(gridIssue); return; }
    const expectedInput = latestFrontierKey.current;
    const requestSequence = ++frontierRequestSequence.current;
    if (!startDate || !endDate || startDate > endDate) { setError(systemText('preInvestment.classAllocation.selectTheFullResearchIntervalStartDate')); return; }
    if (!selectedAlloc || loadedAllocation !== selectedAlloc) {
      setError(systemText('preInvestment.classAllocation.loadTheCurrentAssetClassPlanBefore'));
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
      if (!res.ok) throw new Error(apiErrorMessage(data, systemText('preInvestment.classAllocation.calculationFailed')));
      assertNativeNumericalExecution(data?.execution, systemText('preInvestment.classAllocation.assetAllocationEfficientFrontier'));
      frontierInputRef.current = frontierInputKey;
      setFrontierData(data);
    } catch (e: any) {
      if (requestSequence === frontierRequestSequence.current && expectedInput === latestFrontierKey.current)
        setError(e.message || systemText('preInvestment.classAllocation.calculationFailedCheckResearchDatesAndData'));
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
            {btBusy ? systemText('preInvestment.classAllocation.calculating') : (isCalculating ? systemText('preInvestment.classAllocation.calculatingPleaseWait') : systemText('preInvestment.classAllocation.loading'))}
          </div>
        </div>
      )}

      
      <h1 className="text-2xl font-semibold">{systemText('preInvestment.classAllocation.assetAllocation')}</h1>
      <p className="text-sm text-slate-600 mt-1">{systemText('preInvestment.classAllocation.setLongTermCapitalWeightsTestHistorical')}</p>
      <p className="mt-2 text-sm text-slate-600">{loadedAllocation ? systemText('preInvestment.classAllocation.currentClassesClasses', { p0: loadedAllocation, p1: assetNames.length }) : systemText('preInvestment.classAllocation.selectASavedAssetClassPlanTo')} <Link className="ml-2 underline" to={allocationJourneyPath('classes')}>{systemText('preInvestment.classAllocation.returnToAssetConstruction')}</Link></p>
      {draftNotice && <p role="status" className="mt-3 rounded-lg bg-amber-50 p-3 text-sm text-amber-900">{systemText('preInvestment.classAllocation.researchInputsRetainedRecalculateHistoricalResultsUnder')}</p>}
      {error && <p role="alert" className="sticky top-2 z-40 mt-3 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800">{error}</p>}

      <section aria-labelledby="frontier-workspace-title" className="mt-6 min-w-0 rounded-xl border border-slate-200 bg-white p-4 sm:p-6 [&_button]:min-h-10 [&_input:not([type=checkbox])]:min-h-10 [&_select]:min-h-10">
      <h2 id="frontier-workspace-title" className="text-lg font-semibold text-slate-800">{systemText('preInvestment.classAllocation.allocationSpaceAndEfficientFrontier')}</h2>
      <p className="my-3 text-sm leading-6 text-slate-600">{systemText('preInvestment.classAllocation.configureMetricsIndividualAndJointConstraintsMulti')}</p>
      <Section plain title={systemText('preInvestment.classAllocation.selectAnAssetClassScheme')}>
        {loading && <p>{systemText('preInvestment.classAllocation.loadingPlanList')}</p>}
        {!loading && (
          <div className="flex flex-wrap items-center gap-3">
            <select 
              aria-label={systemText('preInvestment.classAllocation.assetClassScheme')}
              value={selectedAlloc}
              onChange={event => { updateAllocationJourney({ allocationName: event.target.value, universeId: universeId || undefined }); navigate(allocationLabPath(event.target.value, universeId)); }}
              className="min-w-0 max-w-full flex-grow rounded-lg border-slate-300 shadow-sm focus:border-accent-500 focus:ring-accent-500">
              {allocations.map(name => <option key={name} value={name}>{name}</option>)}
            </select>
            <button 
              onClick={handleSelectAndLoad}
              disabled={Boolean(loadedAllocation) && loadedAllocation === selectedAlloc}
              className="rounded-lg bg-accent-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-accent-700">
              {loadedAllocation === selectedAlloc && loadedAllocation ? systemText('preInvestment.classAllocation.loaded') : systemText('preInvestment.classAllocation.selectThisScheme')}
            </button>
          </div>
        )}
        {configDetails && (
          <details className="mt-3 rounded-lg border p-3">
            <summary className="cursor-pointer text-sm">{systemText('preInvestment.classAllocation.viewProductsAndWeightsWithinEachClass')}</summary>
            <div className="mt-2 max-h-80 overflow-auto">
              <table className="min-w-full divide-y divide-slate-200">
              <thead className="bg-slate-50">
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium tracking-wide text-slate-600">{systemText('preInvestment.classAllocation.assetName')}</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium tracking-wide text-slate-600">{systemText('preInvestment.classAllocation.productCode')}</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium tracking-wide text-slate-600">{systemText('preInvestment.classAllocation.productName')}</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium tracking-wide text-slate-600">{systemText('preInvestment.classAllocation.withinClassWeights')}</th>
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

      <Section plain title={systemText('preInvestment.classAllocation.longTermAllocationObjectiveAndResearchInterval')}>
        <div className="grid gap-3 sm:grid-cols-3">
          <label className="text-sm">{systemText('preInvestment.classAllocation.whatWouldYouLikeToDoFirst')}<select aria-label={systemText('preInvestment.classAllocation.longTermAllocationObjective')} value={researchGoal} onChange={event => setResearchGoal(event.target.value)} className="mt-1 w-full rounded-lg border p-2"><option value="manual">{systemText('preInvestment.classAllocation.enterLongTermWeights')}</option><option value="compare">{systemText('preInvestment.classAllocation.compareCandidatesWithDifferentReturnsAndRisks')}</option></select></label>
          <label className="text-sm">{systemText('preInvestment.classAllocation.researchIntervalStart')}<input aria-label={systemText('preInvestment.classAllocation.researchIntervalStart')} type="date" value={startDate} max={endDate} onChange={event => { setStartDate(event.target.value); setBtStart(event.target.value); }} className="mt-1 w-full rounded-lg border p-2" /></label>
          <label className="text-sm">{systemText('preInvestment.classAllocation.policyDateConstructionIntervalEnd')}<input aria-label={systemText('preInvestment.classAllocation.researchIntervalEnd')} type="date" value={endDate} min={startDate} onChange={event => setEndDate(event.target.value)} className="mt-1 w-full rounded-lg border p-2" /></label>
        </div>
        <p className="mt-2 text-xs text-slate-600">{systemText('preInvestment.classAllocation.candidatesTargetGridsAndBacktestsShareThis')}</p>

      </Section>
      <Section plain title={systemText('preInvestment.classAllocation.assetClassFundingBoundsOfTheEntire')}>
        {/* 权重约束设置 */}
        <div className="mt-6 grid grid-cols-1 gap-6 md:grid-cols-2">
          <div className="min-w-0 space-y-3">
            <h3 className="font-medium text-slate-700">{systemText('preInvestment.classAllocation.individualAssetClassFundingRange')}</h3>
            {assetNames.length === 0 ? (
              <p className="text-sm text-slate-600 mt-2">{systemText('preInvestment.classAllocation.selectAndLoadASchemeFirst')}</p>
            ) : (
              <div className="mt-3 space-y-2">
                <div className="grid grid-cols-3 gap-2 text-xs text-slate-600"><span>{systemText('preInvestment.classAllocation.assetClass')}</span><span>{systemText('preInvestment.classAllocation.minimum')}</span><span>{systemText('preInvestment.classAllocation.maximum')}</span></div>
                {assetNames.map(name => (
                  <div key={name} className="grid grid-cols-3 items-center gap-2">
                    <div className="text-sm text-slate-700">{name}</div>
                    <input aria-label={systemText('preInvestment.classAllocation.minimumWeight', { p0: name })} type="number" min={0} max={100} step={1}
                      value={Number(((singleLimits[name]?.lo ?? 0) * 100).toFixed(4))}
                      onChange={e => setSingleLimits(prev => ({ ...prev, [name]: { ...(prev[name]||{lo:0,hi:1}), lo: Number(e.target.value) / 100 } }))}
                      className="min-w-0 w-full rounded-lg border-slate-300 px-2 py-1 text-sm" placeholder={systemText('preInvestment.classAllocation.minimum')} />
                    <input aria-label={systemText('preInvestment.classAllocation.maximumWeight', { p0: name })} type="number" min={0} max={100} step={1}
                      value={Number(((singleLimits[name]?.hi ?? 1) * 100).toFixed(4))}
                      onChange={e => setSingleLimits(prev => ({ ...prev, [name]: { ...(prev[name]||{lo:0,hi:1}), hi: Number(e.target.value) / 100 } }))}
                      className="min-w-0 w-full rounded-lg border-slate-300 px-2 py-1 text-sm" placeholder={systemText('preInvestment.classAllocation.maximum')} />
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="min-w-0 space-y-3">
            <h3 className="font-medium text-slate-700">{systemText('preInvestment.classAllocation.combinedRangeForMultipleClasses')}</h3>
            <p className="mt-1 text-xs text-slate-600">{systemText('preInvestment.classAllocation.forExampleEquitiesPlusCommoditiesMustNot')}</p>
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
                  <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-slate-600"><span>{systemText('preInvestment.classAllocation.minimum')}</span><span>{systemText('preInvestment.classAllocation.maximum')}</span><span /></div>
                  <div className="mt-1 grid grid-cols-3 gap-2">
                    <input aria-label={systemText('preInvestment.classAllocation.jointConstraintMinimumWeight', { p0: idx + 1 })} type="number" min={0} max={100} step={1} value={Number((g.lo * 100).toFixed(4))} onChange={e => setGroupLimits(prev => prev.map(x => x.id===g.id ? { ...x, lo: Number(e.target.value) / 100 } : x))} className="min-w-0 w-full rounded-lg border-slate-300 px-2 py-1 text-sm" placeholder={systemText('preInvestment.classAllocation.minimum')} />
                    <input aria-label={systemText('preInvestment.classAllocation.jointConstraintMaximumWeight', { p0: idx + 1 })} type="number" min={0} max={100} step={1} value={Number((g.hi * 100).toFixed(4))} onChange={e => setGroupLimits(prev => prev.map(x => x.id===g.id ? { ...x, hi: Number(e.target.value) / 100 } : x))} className="min-w-0 w-full rounded-lg border-slate-300 px-2 py-1 text-sm" placeholder={systemText('preInvestment.classAllocation.maximum')} />
                    <button onClick={() => setGroupLimits(prev => prev.filter(x => x.id !== g.id))} className="rounded-lg bg-red-50 text-red-700 text-xs px-2">{systemText('preInvestment.classAllocation.delete')}</button>
                  </div>
                </div>
              ))}
              <button onClick={() => setGroupLimits(prev => [...prev, { id: `g${Date.now()}`, assets: [], lo: 0, hi: 1 }])} className="rounded-lg bg-slate-100 px-3 py-1 text-xs">{systemText('preInvestment.classAllocation.addJointConstraint')}</button>
            </div>
          </div>
        </div>

      </Section>
      <Section plain title={systemText('preInvestment.classAllocation.returnRiskAndRandomExplorationParameters')}>
        <div className="grid grid-cols-1 gap-x-8 gap-y-6 md:grid-cols-2">
          {/* 收益指标 */}
          <div className="space-y-3">
            <h3 className="font-medium text-slate-700">{systemText('preInvestment.classAllocation.returnMetric')}</h3>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <label className="block text-sm font-medium text-slate-600">{systemText('preInvestment.classAllocation.returnMetric')}</label>
                <select aria-label={systemText('preInvestment.classAllocation.returnMetric')} onChange={e => setReturnMetric(e.target.value)} value={returnMetric} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm">
                  <option value="annual">{systemText('preInvestment.classAllocation.annualizedReturn')}</option>
                  <option value="annual_mean">{systemText('preInvestment.classAllocation.meanAnnualizedReturn')}</option>
                  <option value="cumulative">{systemText('preInvestment.classAllocation.cumulativeReturn2')}</option>
                  <option value="mean">{systemText('preInvestment.classAllocation.meanReturn')}</option>
                  <option value="ewm">{systemText('preInvestment.classAllocation.exponentiallyWeightedReturn')}</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-600">{systemText('preInvestment.classAllocation.returnType')}</label>
                <select aria-label={systemText('preInvestment.classAllocation.returnType')} value={returnType} onChange={e => setReturnType(e.target.value)} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm">
                  <option value="simple">{systemText('preInvestment.classAllocation.simpleReturns')}</option>
                  <option value="log">{systemText('preInvestment.classAllocation.logReturns')}</option>
                </select>
              </div>
            </div>
            {(returnMetric === 'annual' || returnMetric === 'annual_mean') && (
              <div>
                <label className="block text-sm font-medium text-slate-600">{systemText('preInvestment.classAllocation.annualizationDays')}</label>
                <input aria-label={systemText('preInvestment.classAllocation.returnAnnualizationDays')} type="number" value={annualDaysRet} onChange={e => setAnnualDaysRet(Number(e.target.value))} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm" />
              </div>
            )}
            {returnMetric === 'ewm' && (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <label className="block text-sm font-medium text-slate-600">{systemText('preInvestment.classAllocation.decayFactor')}</label>
                  <input aria-label={systemText('preInvestment.classAllocation.returnDecayFactor')} type="number" step="0.01" value={ewmAlpha} onChange={e => setEwmAlpha(Number(e.target.value))} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-600">{systemText('preInvestment.classAllocation.windowLength')}</label>
                  <input aria-label={systemText('preInvestment.classAllocation.returnWindowLength')} type="number" value={ewmWindow} onChange={e => setEwmWindow(Number(e.target.value))} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm" />
                </div>
              </div>
            )}
          </div>

          {/* 风险指标 */}
          <div className="space-y-3">
            <h3 className="font-medium text-slate-700">{systemText('preInvestment.classAllocation.riskMetric')}</h3>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                    <label className="block text-sm font-medium text-slate-600">{systemText('preInvestment.classAllocation.riskMetric')}</label>
                    <select aria-label={systemText('preInvestment.classAllocation.riskMetric')} onChange={e => setRiskMetric(e.target.value)} value={riskMetric} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm">
                        <option value="vol">{systemText('preInvestment.classAllocation.volatility')}</option>
                        <option value="annual_vol">{systemText('preInvestment.classAllocation.annualVolatility2')}</option>
                        <option value="ewm_vol">{systemText('preInvestment.classAllocation.exponentiallyWeightedVolatility')}</option>
                        <option value="var">VaR</option>
                        <option value="es">ES</option>
                        <option value="max_drawdown">{systemText('preInvestment.classAllocation.maximumDrawdown2')}</option>
                        <option value="downside_vol">{systemText('preInvestment.classAllocation.downsideVolatility')}</option>
                    </select>
                </div>
                {(riskMetric === 'var' || riskMetric === 'es') && (
                    <div>
                        <label className="block text-sm font-medium text-slate-600">{systemText('preInvestment.classAllocation.confidence')}</label>
                        <input aria-label={systemText('preInvestment.classAllocation.confidence2')} type="number" value={confidence} onChange={e => setConfidence(Number(e.target.value))} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm" />
                    </div>
                )}
            </div>
            {(riskMetric === 'annual_vol') && (
                <div>
                    <label className="block text-sm font-medium text-slate-600">{systemText('preInvestment.classAllocation.annualizationDays')}</label>
                    <input aria-label={systemText('preInvestment.classAllocation.riskAnnualizationDays')} type="number" value={annualDaysRisk} onChange={e => setAnnualDaysRisk(Number(e.target.value))} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm" />
                </div>
            )}
            {(riskMetric === 'ewm_vol') && (
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <div>
                        <label className="block text-sm font-medium text-slate-600">{systemText('preInvestment.classAllocation.decayFactor')}</label>
                        <input aria-label={systemText('preInvestment.classAllocation.riskDecayFactor')} type="number" step="0.01" value={ewmAlphaRisk} onChange={e => setEwmAlphaRisk(Number(e.target.value))} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm" />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-slate-600">{systemText('preInvestment.classAllocation.windowLength')}</label>
                    <input aria-label={systemText('preInvestment.classAllocation.riskWindowLength')} type="number" value={ewmWindowRisk} onChange={e => setEwmWindowRisk(Number(e.target.value))} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm" />
                    </div>
                </div>
            )}
          </div>

          {/* 夏普比率参数 */}
          <div className="space-y-3 md:col-span-2">
            <h3 className="font-medium text-slate-700">{systemText('preInvestment.classAllocation.sharpeRatioParameters')}</h3>
            <div>
              <label className="block text-sm font-medium text-slate-600">{systemText('preInvestment.classAllocation.annualRiskFreeReturn')}</label>
              <input
                aria-label={systemText('preInvestment.classAllocation.riskFreeReturn')}
                type="number"
                step="0.1"
                value={riskFreePct}
                onChange={e => setRiskFreePct(Number(e.target.value))}
                className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm"
              />
              <p className="mt-1 text-xs text-slate-600">{systemText('preInvestment.classAllocation.usedForMaximumSharpeCalculationDefaultsTo')}</p>
            </div>
            <p className="text-xs text-slate-600">
              {systemText('preInvestment.classAllocation.sharpeRatioMeanAnnualizedReturnAnnualRisk')}</p>
          </div>
        </div>

        {/* 随机探索设置 */}
        <div className="mt-6 border-t border-slate-200 pt-4">
          <h3 className="font-medium text-slate-700">{systemText('preInvestment.classAllocation.multiRoundRandomWalk')}</h3>
          <p className="text-xs text-slate-600 mt-1">{systemText('preInvestment.classAllocation.theFirstRoundSamplesRandomlyLaterRounds')}</p>
          <label className="mt-3 block max-w-xs text-sm text-slate-700">{systemText('preInvestment.classAllocation.randomSeed')}<input aria-label={systemText('preInvestment.classAllocation.randomSeed')} type="number" min={0} max={4294967295} step={1} value={explorationSeed} onChange={event => setExplorationSeed(Number(event.target.value))} className="mt-1 w-full rounded-lg border-slate-300 p-2" />
            <span className="mt-1 block text-xs text-slate-600">{systemText('preInvestment.classAllocation.identicalParametersAndSeedsReproduceResultsChange')}</span>
          </label>
          <div className="mt-2 space-y-2">
            {rounds.map((r, idx) => (
              <div key={r.id} className="grid grid-cols-2 items-end gap-3 border-b border-slate-100 py-2 sm:grid-cols-12">
                <div className="col-span-2 text-sm text-slate-600">{systemText('preInvestment.classAllocation.month')}{idx}{systemText('preInvestment.classAllocation.round')}</div>
                <label className="min-w-0 text-xs text-slate-600 sm:col-span-3">
                  <span className="whitespace-nowrap">{systemText('preInvestment.classAllocation.samplePoints')}</span>
                  <input aria-label={systemText('preInvestment.classAllocation.roundSamplePoints', { p0: idx })} type="number" min={1} value={r.samples} onChange={e => setRounds(prev => prev.map(x => x.id===r.id ? { ...x, samples: Number(e.target.value) } : x))} className="mt-1 min-w-0 w-full rounded-lg border-slate-300 px-2 py-1 text-sm" />
                </label>
                <label className="min-w-0 text-xs text-slate-600 sm:col-span-3">
                  <span className="whitespace-nowrap">{systemText('preInvestment.classAllocation.step')}</span>
                  <input aria-label={systemText('preInvestment.classAllocation.roundStepSize', { p0: idx })} disabled={idx === 0} title={idx === 0 ? systemText('preInvestment.classAllocation.theFirstRoundSamplesDirectlyWithoutPerturbation') : undefined} type="number" step={0.01} min={0} max={1} value={r.step} onChange={e => setRounds(prev => prev.map(x => x.id===r.id ? { ...x, step: Number(e.target.value) } : x))} className="mt-1 min-w-0 w-full rounded-lg border-slate-300 px-2 py-1 text-sm" />
                </label>
                {idx > 0 && (
                  <label className="min-w-0 text-xs text-slate-600 sm:col-span-3">
                    <span className="whitespace-nowrap">{systemText('preInvestment.classAllocation.buckets')}</span>
                    <input aria-label={systemText('preInvestment.classAllocation.roundBuckets', { p0: idx })} type="number" min={1} value={(r as any).buckets ?? 50} onChange={e => setRounds(prev => prev.map(x => x.id===r.id ? { ...x, buckets: Number(e.target.value) } : x))} className="mt-1 min-w-0 w-full rounded-lg border-slate-300 px-2 py-1 text-sm" />
                  </label>
                )}
                {idx > 0 && (
                  <button onClick={() => setRounds(prev => prev.filter(x => x.id !== r.id))} aria-label={systemText('preInvestment.classAllocation.deleteRound', { p0: idx })} className="rounded-lg bg-rose-50 text-rose-800 text-xs px-2 sm:col-span-1">{systemText('preInvestment.classAllocation.delete2')}</button>
                )}
              </div>
            ))}
            <button onClick={() => setRounds(prev => [...prev, { id: `r${Date.now()}`, samples: 200, step: 0.5, buckets: 50 }])} className="rounded-lg bg-slate-100 px-3 py-1 text-xs">{systemText('preInvestment.classAllocation.addRound')}</button>
          </div>
          <div className="mt-3 grid gap-4 sm:grid-cols-2">
            <div>
              <label className="block text-sm font-medium text-slate-600">{systemText('preInvestment.classAllocation.weightQuantization')}</label>
              <select aria-label={systemText('preInvestment.classAllocation.weightQuantization')} value={quantStep} onChange={e => setQuantStep(e.target.value as any)} className="mt-1 block w-full rounded-lg border-slate-300 shadow-sm">
                <option value="none">{systemText('preInvestment.classAllocation.noQuantization')}</option>
                <option value="0.001">0.1%</option>
                <option value="0.002">0.2%</option>
                <option value="0.005">0.5%</option>
              </select>
            </div>
            <div className="flex flex-wrap items-end gap-2">
              <label className="flex items-center gap-2 text-sm text-slate-700">
                <input type="checkbox" checked={useRefine} onChange={e => setUseRefine(e.target.checked)} /> {" " + systemText('preInvestment.classAllocation.useConstrainedLocalRefinement')}</label>
              {useRefine && (
                <input aria-label={systemText('preInvestment.classAllocation.maximumLocalRefinementIterations')} type="number" min={1} max={200} value={refineCount} onChange={e => setRefineCount(Number(e.target.value))} className="w-28 rounded-lg border-slate-300 px-2 py-1 text-sm" placeholder={systemText('preInvestment.classAllocation.iterations')} />
              )}
            </div>
          </div>
        </div>
        <p className="mt-3 text-xs leading-5 text-slate-600">{systemText('preInvestment.classAllocation.localRefinementImprovesThreeRepresentativeCandidatesThe')}</p>
        <div className="mt-4 border-t border-slate-200 pt-4"><FrontierGridControls value={frontierGrid} quantized={quantStep !== 'none'} busy={isCalculating} onChange={setFrontierGrid} /></div>
        <div className="mt-4 flex flex-wrap gap-3">
          <button disabled={!loadedAllocation || loadedAllocation !== selectedAlloc || equalWeightLoading || isCalculating || (researchGoal !== 'manual' && Boolean(gridIssue))} onClick={researchGoal === 'manual' ? addFixedStrategy : onCalculate} className="rounded-lg bg-accent-600 px-4 py-2 text-sm text-white disabled:opacity-50">{researchGoal === 'manual' ? systemText('preInvestment.classAllocation.enterLongTermWeights2') : systemText('preInvestment.classAllocation.calculateAndCompareCandidates')}</button>
          <button disabled={!loadedAllocation || loadedAllocation !== selectedAlloc || isCalculating || Boolean(gridIssue)} onClick={onCalculate} className="rounded-lg border px-4 py-2 text-sm disabled:opacity-50">{systemText('preInvestment.classAllocation.generateAllocationSpaceAndEfficientFrontier')}</button>
        </div>
        <p className="mt-3 text-xs text-slate-600">{systemText('preInvestment.classAllocation.samplingAttemptBudget') + " "}{rounds.reduce((sum, round) => sum + round.samples, 0).toLocaleString()} {" " + systemText('preInvestment.classAllocation.pointsWeightPrecision') + " "}{quantStep === 'none' ? systemText('preInvestment.classAllocation.continuous') : `${Number(quantStep) * 100}%`}{frontierGrid.enabled ? systemText('preInvestment.classAllocation.frontierTargets', { p0: frontierGrid.point_count }) : ''}</p>
        {isCalculating && <p role="status" className="mt-2 text-sm text-slate-600">{systemText('preInvestment.classAllocation.generatingRandomExplorationCandidatesAndFrontierPlease')}</p>}
      </Section>

      {frontierData && (
        <Section plain title={systemText('preInvestment.classAllocation.compareCandidatesAndAdoptALongTerm')}>
          <p className="mb-3 text-sm text-slate-600">{systemText('preInvestment.classAllocation.historicalCandidatesUseTheSameIntervalAnd')}</p>
          {/* 这张前沿图挑出来的就是权重本身，它按哪天、哪个产品域算的必须跟着它走。 */}
          <PitDecisionNotice lineage={frontierData.pit} />
          {frontierData.research_interval && <p className="mb-3 text-xs leading-5 text-slate-600">{systemText('preInvestment.classAllocation.actualSample')}{frontierData.research_interval.actual_start} {" " + systemText('preInvestment.classAllocation.to') + " "}{frontierData.research_interval.actual_end} {" " + systemText('preInvestment.classAllocation.nav') + " "}{frontierData.research_interval.nav_observations} {" " + systemText('preInvestment.classAllocation.periodsReturns') + " "}{frontierData.research_interval.return_observations} {" " + systemText('preInvestment.classAllocation.periodsSampledCandidates') + " "}{frontierData.sampled_candidates ?? frontierData.accepted_candidates ?? frontierData.scatter?.length ?? 0} {" " + systemText('preInvestment.classAllocation.items')}{Number(frontierData.refined_candidates ?? 0) > 0 ? systemText('preInvestment.classAllocation.addedByRefinement', { p0: frontierData.refined_candidates }) : ''}{Number(frontierData.grid_candidates ?? 0) > 0 ? systemText('preInvestment.classAllocation.addedByGridOptimization', { p0: frontierData.grid_candidates }) : ''} {" " + systemText('preInvestment.classAllocation.totalCurrentCandidates') + " "}{frontierData.accepted_candidates ?? frontierData.scatter?.length ?? 0} {" " + systemText('preInvestment.classAllocation.pointsEfficientFrontier') + " "}{frontierData.frontier_candidates ?? frontierData.frontier?.length ?? 0} {" " + systemText('preInvestment.classAllocation.points')}</p>}
          {frontierData.refinement?.requested && <p role="status" className="mb-3 rounded-lg bg-slate-50 p-3 text-xs leading-5 text-slate-600">{systemText('preInvestment.classAllocation.localRefinementCoversOnlyTheMaximumSharpe')}{frontierData.refinement.items?.map((item: any) => systemText('preInvestment.classAllocation.iterations2', { p0: item.candidate, p1: item.status, p2: item.iterations, p3: item.applied ? " " + systemText('preInvestment.classAllocation.adopted') : " " + systemText('preInvestment.classAllocation.originalCandidateRetained') })).join('；')}{systemText('preInvestment.classAllocation.noContinuousSolutionOfTheEntireFrontier')}</p>}
          <div className="overflow-x-auto"><table className="min-w-[640px] w-full text-sm" aria-label={systemText('preInvestment.classAllocation.longTermAllocationCandidates')}><thead className="bg-slate-50"><tr><th scope="col" className="p-2 text-left">{systemText('preInvestment.classAllocation.candidate')}</th>{(frontierData.asset_names ?? []).map((name: string) => <th scope="col" key={name} className="p-2">{name}</th>)}<th scope="col" className="p-2">{returnLabel}</th><th scope="col" className="p-2">{riskLabel}</th><th scope="col" className="p-2">{systemText('preInvestment.classAllocation.actions')}</th></tr></thead><tbody>{candidates.map(({ label, key }) => {
            const point = frontierData[key]; const value = point.value ?? point;
            return <tr key={key} className="border-t"><td className="p-2">{label}</td>{(point.weights ?? []).map((weight: number, index: number) => <td key={index} className="p-2 text-center">{(weight * 100).toFixed(2)}%</td>)}<td className="p-2 text-center">{Number.isFinite(value[1]) ? `${(value[1] * 100).toFixed(2)}%` : '—'}</td><td className="p-2 text-center">{Number.isFinite(value[0]) ? `${(value[0] * 100).toFixed(2)}%` : '—'}</td><td className="p-2"><button onClick={() => adoptCandidate(label, point)} className="whitespace-nowrap rounded-lg border border-emerald-700 px-3 py-1 text-emerald-800">{systemText('preInvestment.classAllocation.adopt')}{label}</button></td></tr>;
          })}</tbody></table></div>
          {frontierData.exploration && <div className="mt-4 space-y-3">
            <label className="block max-w-sm text-sm">{systemText('preInvestment.classAllocation.scatterDisplay')}<select aria-label={systemText('preInvestment.classAllocation.scatterDisplay')} value={scatterView} onChange={event => setScatterView(event.target.value as 'all' | 'selected')} className="mt-1 w-full rounded-lg border-slate-300 p-2">
                <option value="all">{systemText('preInvestment.classAllocation.allFeasibleCandidatesIncludingRefinement')}</option><option value="selected">{systemText('preInvestment.classAllocation.initialAndPerRoundBucketRetainedPoints')}</option>
              </select>
            </label>
            <div className="overflow-x-auto"><table aria-label={systemText('preInvestment.classAllocation.randomExplorationRoundStatistics')} className="w-full min-w-[480px] text-sm">
              <thead><tr>{[systemText('preInvestment.classAllocation.round2'), systemText('preInvestment.classAllocation.attemptBudget'), systemText('preInvestment.classAllocation.feasiblePoints'), systemText('preInvestment.classAllocation.retainedPoints'), systemText('preInvestment.classAllocation.failedPoints')].map(label => <th scope="col" key={label} className="p-2 text-left">{label}</th>)}</tr></thead>
              <tbody>{frontierData.exploration.rounds.map((round: any, index: number) => <tr key={round.round} className="border-t border-slate-200"><th scope="row" className="p-2 text-left">{systemText('preInvestment.classAllocation.month')}{index}{systemText('preInvestment.classAllocation.round')}</th><td className="p-2">{round.requested}</td><td className="p-2">{round.accepted}</td><td className="p-2">{round.selected}</td><td className="p-2">{round.rejected}{round.search_budget_failures > 0 ? systemText('preInvestment.classAllocation.searchBudgetExhausted', { p0: round.search_budget_failures }) : ''}{round.accepted === 0 ? systemText('preInvestment.classAllocation.reusedSeedsFromThePreviousValidRound') : ''}</td></tr>)}</tbody>
            </table></div>
          </div>}
          {frontierData.frontier_grid && <FrontierGridResults result={frontierData.frontier_grid} assetNames={frontierData.asset_names} riskLabel={riskLabel} returnLabel={returnLabel} onAdopt={point => adoptCandidate(systemText('preInvestment.classAllocation.frontierTarget', { p0: point.target_index + 1 }), point)} />}
          <details open className="mt-4"><summary className="cursor-pointer text-sm">{systemText('preInvestment.classAllocation.efficientFrontierChartExpandedByDefaultCollapsible')}</summary><ReactECharts
            style={{ height: 500 }}
            notMerge
            option={{
              animation: false,
              title: {
                text: systemText('preInvestment.classAllocation.allocationSpaceAndEfficientFrontier'),
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
                data: [systemText('preInvestment.classAllocation.otherPortfolios'), frontierData.weight_domain === 'discrete' && frontierData.frontier_grid ? systemText('preInvestment.classAllocation.continuousTheoreticalFrontier') : systemText('preInvestment.classAllocation.efficientFrontier'), ...(frontierData.frontier_grid ? [frontierData.weight_domain === 'discrete' ? systemText('preInvestment.classAllocation.frontierAtSelectedPrecision') : systemText('preInvestment.classAllocation.nondominatedCandidates')] : []), systemText('preInvestment.classAllocation.maximumSharpeRatio'), systemText('preInvestment.classAllocation.minimumRisk'), systemText('preInvestment.classAllocation.maximumReturn')]
              },
              grid: { top: 80, bottom: 80, left: 12, right: 30, containLabel: true },
              xAxis: { type: 'value', name: riskLabel, nameLocation: 'middle', nameGap: 28, scale: true, splitNumber: 3, axisLabel: { hideOverlap: true, fontSize: 12, formatter: (value: number) => `${(value * 100).toFixed(1)}%` } },
              yAxis: { type: 'value', name: returnLabel, nameLocation: 'middle', nameGap: 42, scale: true, axisLabel: { hideOverlap: true, fontSize: 12, formatter: (value: number) => `${(value * 100).toFixed(1)}%` } },
              series: [
                ...(frontierData.scatter ? [{
                  name: systemText('preInvestment.classAllocation.otherPortfolios'),
                  type: 'scatter',
                  symbolSize: 3,
                  data: scatterView === 'selected' && frontierData.exploration ? frontierData.exploration.selected_indices.map((index: number) => frontierData.scatter[index]) : frontierData.scatter,
                  itemStyle: { color: 'rgba(128, 128, 128, 0.35)' }
                }] : []),
                ...(frontierData.frontier ? [{
                  name: frontierData.weight_domain === 'discrete' && frontierData.frontier_grid ? systemText('preInvestment.classAllocation.continuousTheoreticalFrontier') : systemText('preInvestment.classAllocation.efficientFrontier'),
                  type: frontierData.frontier_grid ? 'line' : 'scatter',
                  connectNulls: false,
                  smooth: false,
                  symbolSize: 6,
                  data: frontierData.frontier_grid ? frontierData.frontier_grid.curve.map((point: any) => point ?? { value: [null, null] }) : frontierData.frontier,
                  itemStyle: { color: '#2563eb' } // Tailwind indigo-600
                }] : []),
                ...(frontierData.frontier_grid ? [{
                  name: frontierData.weight_domain === 'discrete' ? systemText('preInvestment.classAllocation.frontierAtSelectedPrecision') : systemText('preInvestment.classAllocation.nondominatedCandidates'), type: 'scatter', symbolSize: 3, data: frontierData.frontier,
                }] : []),
                ...(frontierData.max_sharpe ? [{
                  name: systemText('preInvestment.classAllocation.maximumSharpeRatio'),
                  type: 'scatter',
                  symbolSize: 10,
                  data: [frontierData.max_sharpe]
                }] : []),
                ...(frontierData.min_variance ? [{
                  name: systemText('preInvestment.classAllocation.minimumRisk'),
                  type: 'scatter',
                  symbolSize: 10,
                  data: [frontierData.min_variance]
                }] : []),
                ...(frontierData.max_return ? [{
                  name: systemText('preInvestment.classAllocation.maximumReturn'),
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
      <Section title={systemText('preInvestment.classAllocation.longTermWeightsAndHistoricalValidation')}>
        <p className="mb-4 rounded-lg bg-emerald-50 p-3 text-sm text-emerald-900">{systemText('preInvestment.classAllocation.taaInheritsTheseWeightsWithinClassProduct')}</p>
        {!strategies.length && <p className="mb-3 text-sm text-slate-600">{systemText('preInvestment.classAllocation.enterLongTermWeightsOrAdoptA')}</p>}
        <div className="space-y-4">
          {/* 顶部不再显示“添加策略”按钮，统一放在策略列表与回测之间 */}

          {strategies.map((s, idx) => (
            <div key={s.id} className="rounded-lg border p-4">
              <div className="flex flex-wrap items-center gap-3">
                <input value={s.name} onChange={e => setStrategies(prev => prev.map(x => x.id===s.id? { ...x, name: e.target.value } : x))} className="rounded-lg border-slate-300 px-2 py-1 text-sm" />
                <span className="rounded-lg bg-slate-100 px-2 py-1 text-xs text-slate-700">
                  {s.type === 'fixed' ? systemText('preInvestment.classAllocation.fixedWeights') : s.type === 'risk_budget' ? systemText('preInvestment.classAllocation.riskBudget') : systemText('preInvestment.classAllocation.specifiedTarget')}
                </span>
                <button type="button" disabled={taaBusy !== null || loadedAllocation !== selectedAlloc}
                  onClick={() => enterTacticalResearch(s)}
                  className="rounded-lg border border-emerald-700 px-3 py-2 text-sm font-medium text-emerald-800 disabled:opacity-50">
                  {taaBusy === s.id ? systemText('preInvestment.classAllocation.lockingSaa') : systemText('preInvestment.classAllocation.useAsSaaAndResearchTacticalDeviations')}
                </button>
                <button onClick={() => setStrategies(prev => prev.filter(x => x.id !== s.id))} className="ml-auto rounded-lg bg-red-50 px-2 py-1 text-xs text-red-700">{systemText('preInvestment.classAllocation.delete')}</button>
              </div>
              {/* 再平衡设置（通用） */}
              <div className="mt-3 rounded-lg border p-3 text-sm">
                <label className="flex items-center gap-2"><input type="checkbox" checked={!!s.rebalance?.enabled} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, rebalance: { ...(x.rebalance||{}), enabled: e.target.checked } } : x))}/> {" " + systemText('preInvestment.classAllocation.enableRebalancing')}</label>
                {s.rebalance?.enabled && (
                  <div className="mt-2 grid grid-cols-12 items-center gap-2">
                    <div className="col-span-3">
                      <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.rebalancingMethod')}</label>
                      <select value={s.rebalance?.mode||'monthly'} onChange={e=> setStrategies(prev=> prev.map(x=> {
                        if (x.id!==s.id) return x as any;
                        const mode = e.target.value;
                        const rb = { ...(x.rebalance||{}), mode } as any;
                        if (mode === 'fixed' && !rb.fixedInterval) {
                          rb.fixedInterval = Math.max(1, navCount||1);
                        }
                        return { ...x, rebalance: rb } as any;
                      }))} className="mt-1 w-full rounded-lg border-slate-300">
                        <option value="weekly">{systemText('preInvestment.classAllocation.weekly')}</option>
                        <option value="monthly">{systemText('preInvestment.classAllocation.monthly')}</option>
                        <option value="yearly">{systemText('preInvestment.classAllocation.annually')}</option>
                        <option value="fixed">{systemText('preInvestment.classAllocation.fixedInterval')}</option>
                      </select>
                    </div>
                    {s.rebalance?.mode !== 'fixed' ? (
                      <>
                    <div className="col-span-2">
                      <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.nth')}</label>
                      <input type="number" min={1} max={ s.rebalance?.mode==='weekly'?5: s.rebalance?.mode==='monthly'?30:360 } value={s.rebalance?.N ?? 1} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, rebalance: { ...(x.rebalance||{}), which:'nth', N: Number(e.target.value) } } : x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                    </div>
                        <div className="col-span-2">
                        <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.items')}</label>
                        <div className="mt-1 text-sm text-slate-600">&nbsp;</div>
                        </div>
                        <div className="col-span-2">
                          <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.unit')}</label>
                          <select value={s.rebalance?.unit||'trading'} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, rebalance: { ...(x.rebalance||{}), unit: e.target.value } } : x))} className="mt-1 w-full rounded-lg border-slate-300">
                            <option value="trading">{systemText('preInvestment.classAllocation.tradingDay')}</option>
                            <option value="natural">{systemText('preInvestment.classAllocation.calendarDay')}</option>
                          </select>
                        </div>
                      </>
                    ) : (
                      <div className="col-span-3">
                        <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.fixedIntervalDays')}</label>
                        <input type="number" min={1} value={s.rebalance?.fixedInterval ?? 20} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, rebalance: { ...(x.rebalance||{}), fixedInterval: Number(e.target.value) } } : x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                      </div>
                    )}
                    {s.type !== 'fixed' && (
                      <div className="col-span-12">
                        <label className="flex items-center gap-2"><input type="checkbox" checked={!!s.rebalance?.recalc} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, rebalance: { ...(x.rebalance||{}), recalc: e.target.checked } } : x))}/> {" " + systemText('preInvestment.classAllocation.recalculateTheModelAtRebalancing')}</label>
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
                        setError(reason instanceof Error ? reason.message : systemText('preInvestment.classAllocation.equalWeightCalculationFailed'));
                      }
                    }}/> {" " + systemText('preInvestment.classAllocation.equalWeights')}</label>
                    <label className="flex items-center gap-2"><input type="radio" checked={(s.cfg?.mode||'equal')==='custom'} onChange={() => setStrategies(prev => prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, mode:'custom'}, rows: x.rows }:x))}/> {" " + systemText('preInvestment.classAllocation.customWeights')}</label>
                  </div>
                  <p className="text-xs text-slate-600">{(s.cfg?.mode || 'equal') === 'equal' ? systemText('preInvestment.classAllocation.allocateCapitalEquallyAcrossAssetClasses') : systemText('preInvestment.classAllocation.enterWeightsAsPercentagesOfTheEntire')}</p>
                  {s.cfg?.selection_reason && <label className="block text-sm">{systemText('preInvestment.classAllocation.selectionRationale')}<input aria-label={systemText('preInvestment.classAllocation.selectionRationale2', { p0: s.name })} className="mt-1 w-full rounded-lg border p-2" value={s.cfg.selection_reason} onChange={event => setStrategies(current => current.map(item => item.id === s.id ? { ...item, cfg: { ...item.cfg, selection_reason: event.target.value } } : item))} /></label>}
                  <div className="min-w-0 overflow-x-auto rounded-lg border">
                    <table className="min-w-full">
                      <thead className="bg-slate-50 text-xs text-slate-600"><tr><th scope="col" className="px-3 py-2 text-left">{systemText('preInvestment.classAllocation.assetName')}</th><th scope="col" className="px-3 py-2 text-left">{systemText('preInvestment.classAllocation.fundingWeight')}</th></tr></thead>
                      <tbody className="text-sm">
                        {s.rows.map((r,i)=> (
                          <tr key={i} className="border-t">
                            <td className="px-3 py-2">{r.className}</td>
                            <td className="px-3 py-2"><input aria-label={systemText('preInvestment.classAllocation.weight', { p0: s.name, p1: r.className })} type="number" min="0" max="100" value={r.weight ?? ''} onChange={e=>{
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
                      <thead className="bg-slate-50 text-xs text-slate-600"><tr><th scope="col" className="px-3 py-2 text-left">{systemText('preInvestment.classAllocation.assetName')}</th><th scope="col" className="px-3 py-2 text-left">{systemText('preInvestment.classAllocation.riskBudget2')}</th><th scope="col" className="px-3 py-2 text-left">{systemText('preInvestment.classAllocation.fundingWeight')}</th></tr></thead>
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
                      <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.riskMetric')}</label>
                      <select value={s.cfg?.risk_metric||'vol'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_metric:e.target.value}}:x))} className="mt-1 w-full rounded-lg border-slate-300">
                        <option value="vol">{systemText('preInvestment.classAllocation.volatility')}</option>
                        <option value="var">VaR</option>
                        <option value="es">ES</option>
                        <option value="downside_vol">{systemText('preInvestment.classAllocation.downsideVolatility')}</option>
                        <option value="max_drawdown">{systemText('preInvestment.classAllocation.maximumDrawdown2')}</option>
                      </select>
                    </div>
                    {['var','es'].includes(s.cfg?.risk_metric) && (
                      <><div>
                        <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.confidence3')}</label>
                        <input type="number" value={s.cfg?.confidence ?? 95} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, confidence:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                      </div>
                      <div>
                        <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.days')}</label>
                        <input type="number" value={s.cfg?.days ?? 252} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, days:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                      </div></>
                    )}
                  </div>
                  {/* 模型计算区间 */}
                  <div className="rounded-lg border p-3 text-sm">
                    <div className="grid grid-cols-3 gap-3 items-end">
                      <div>
                        <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.windowMode')}</label>
                        <select value={s.cfg?.window_mode || 'all'} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, cfg: { ...(x.cfg||{}), window_mode: e.target.value } } : x))} className="mt-1 w-full rounded-lg border-slate-300">
                          <option value="all">{systemText('preInvestment.classAllocation.allData')}</option>
                          <option value="rollingN">{systemText('preInvestment.classAllocation.mostRecentNObservations')}</option>
                        </select>
                      </div>
                      { (s.cfg?.window_mode==='rollingN') && (
                        <div>
                          <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.nTradingDays')}</label>
                          <input type="number" min={2} value={s.cfg?.data_len ?? 60} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, cfg: { ...(x.cfg||{}), data_len: Number(e.target.value) } } : x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                        </div>
                      )}
                    </div>
                    <p className="mt-2 text-xs text-slate-600">
                      {systemText('preInvestment.classAllocation.modeGuidance')}<span className="ml-1 font-medium">{systemText('preInvestment.classAllocation.allData')}</span> {" " + systemText('preInvestment.classAllocation.usesAllSamplesFromBacktestStartTo')}<span className="ml-1 font-medium">{systemText('preInvestment.classAllocation.mostRecentNObservations')}</span> {" " + systemText('preInvestment.classAllocation.usesTheNObservationsPrecedingTheCurrent')}</p>
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
                          setError(e?.message || systemText('preInvestment.classAllocation.weightCalculationFailed'));
                        }finally{
                          setBusyStrategy(null);
                          setBtBusy(false);
                        }
                      }}
                    >{systemText('preInvestment.classAllocation.inferFundingWeights')}</button>
                  </div>
                  {busyStrategy===s.id && <div className="text-xs text-slate-600">{systemText('preInvestment.classAllocation.calculatingPleaseWait2')}</div>}

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
                                <th scope="col" className="px-3 py-2 text-left text-xs text-slate-600">{systemText('preInvestment.classAllocation.assetName')}</th>
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
                      <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.objectiveType')}</label>
                      <select value={s.cfg?.target||'min_risk'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, target:e.target.value}}:x))} className="mt-1 w-full rounded-lg border-slate-300">
                        <option value="min_risk">{systemText('preInvestment.classAllocation.minimumRisk')}</option>
                        <option value="max_return">{systemText('preInvestment.classAllocation.maximumReturn')}</option>
                        <option value="max_sharpe">{systemText('preInvestment.classAllocation.maximizeReturnToRiskRatio')}</option>
                        <option value="max_sharpe_traditional">{systemText('preInvestment.classAllocation.maximizeSharpeRatio')}</option>
                        <option value="risk_min_given_return">{systemText('preInvestment.classAllocation.minimizeRiskAtASpecifiedReturn')}</option>
                        <option value="return_max_given_risk">{systemText('preInvestment.classAllocation.maximizeReturnUnderARiskLimit')}</option>
                      </select>
                      
                      {/* Explanations for each target type */}
                      {(s.cfg?.target === 'min_risk' || !s.cfg?.target) && (
                        <p className="mt-2 text-xs text-slate-600 bg-slate-50 p-2 rounded-lg">
                          {systemText('preInvestment.classAllocation.objectiveSubjectToAllConstraintsFindWeights')}<strong>{systemText('preInvestment.classAllocation.riskMetric')}</strong>{systemText('preInvestment.classAllocation.text')}</p>
                      )}
                      {s.cfg?.target === 'max_return' && (
                        <p className="mt-2 text-xs text-slate-600 bg-slate-50 p-2 rounded-lg">
                          {systemText('preInvestment.classAllocation.objectiveSubjectToAllConstraintsFindWeights2')}<strong>{systemText('preInvestment.classAllocation.returnMetric')}</strong>{systemText('preInvestment.classAllocation.text2')}</p>
                      )}
                      {s.cfg?.target === 'max_sharpe' && (
                        <p className="mt-2 text-xs text-slate-600 bg-slate-50 p-2 rounded-lg">
                          {systemText('preInvestment.classAllocation.objectiveFindWeightsMaximizing') + " "}<strong>{systemText('preInvestment.classAllocation.selectedReturnMetricSelectedRiskMetric')}</strong> {" " + systemText('preInvestment.classAllocation.thisIsAGeneralizedReturnToRisk')}</p>
                      )}
                      {s.cfg?.target === 'max_sharpe_traditional' && (
                        <p className="mt-2 text-xs text-slate-600 bg-slate-50 p-2 rounded-lg">
                          {systemText('preInvestment.classAllocation.objectiveMaximizeTheConventionalSharpeRatio') + " "}<code>{systemText('preInvestment.classAllocation.annualReturnRiskFreeRateAnnualVolatility')}</code> {" " + systemText('preInvestment.classAllocation.text3')}</p>
                      )}
                      {s.cfg?.target === 'risk_min_given_return' && (
                        <p className="mt-2 text-xs text-slate-600 bg-slate-50 p-2 rounded-lg">
                          {systemText('preInvestment.classAllocation.objectiveMinimizePortfolioRiskWithReturnEqual')}<strong>{systemText('preInvestment.classAllocation.targetReturn')}</strong>{systemText('preInvestment.classAllocation.text4')}</p>
                      )}
                      {s.cfg?.target === 'return_max_given_risk' && (
                        <p className="mt-2 text-xs text-slate-600 bg-slate-50 p-2 rounded-lg">
                          {systemText('preInvestment.classAllocation.objectiveMaximizePortfolioReturnWithRiskNo')}<strong>{systemText('preInvestment.classAllocation.targetRisk')}</strong>{systemText('preInvestment.classAllocation.text5')}</p>
                      )}
                    </div>
                    <div>
                      <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.returnType2')}</label>
                      <select value={s.cfg?.return_type||'simple'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, return_type:e.target.value}}:x))} className="mt-1 w-full rounded-lg border-slate-300">
                        <option value="simple">{systemText('preInvestment.classAllocation.simpleReturns')}</option>
                        <option value="log">{systemText('preInvestment.classAllocation.logReturns')}</option>
                      </select>
                    </div>
                  </div>

                  {/* 根据目标类型显示不同UI */}
                  {s.cfg?.target === 'max_sharpe_traditional' ? (
                    <div className="space-y-3 rounded-lg border p-3 text-sm">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.returnMetricFixed')}</label>
                          <input type="text" value={systemText('preInvestment.classAllocation.meanAnnualizedReturn')} disabled className="mt-1 w-full rounded-lg border-slate-200 bg-slate-100 px-2 py-1"/>
                        </div>
                        <div>
                          <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.riskMetricFixed')}</label>
                          <input type="text" value={systemText('preInvestment.classAllocation.annualVolatility2')} disabled className="mt-1 w-full rounded-lg border-slate-200 bg-slate-100 px-2 py-1"/>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.annualizationDays')}</label>
                          <input type="number" value={s.cfg?.days ?? 252} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, days:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                        </div>
                        <div>
                          <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.annualRiskFreeRate')}</label>
                          <input type="number" step="0.1" value={s.cfg?.risk_free_rate_pct ?? 1.5} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_free_rate_pct:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <>
                      {/* 收益指标配置 */}
                      <div className="rounded-lg border p-3 text-sm space-y-3">
                        <div>
                          <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.returnMetric')}</label>
                          <select value={s.cfg?.return_metric||'cumulative'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, return_metric:e.target.value}}:x))} className="mt-1 w-full rounded-lg border-slate-300">
                            <option value="annual">{systemText('preInvestment.classAllocation.annualizedReturn')}</option>
                            <option value="annual_mean">{systemText('preInvestment.classAllocation.meanAnnualizedReturn')}</option>
                            <option value="cumulative">{systemText('preInvestment.classAllocation.cumulativeReturn2')}</option>
                            <option value="mean">{systemText('preInvestment.classAllocation.meanReturn')}</option>
                            <option value="ewm">{systemText('preInvestment.classAllocation.exponentiallyWeightedReturn')}</option>
                          </select>
                        </div>
                        {(s.cfg?.return_metric==='annual' || s.cfg?.return_metric==='annual_mean') && (
                          <div>
                            <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.annualizationDays')}</label>
                            <input type="number" value={s.cfg?.days ?? 252} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, days:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                          </div>
                        )}
                        {s.cfg?.return_metric==='ewm' && (
                          <div className="grid grid-cols-2 gap-3">
                            <div>
                              <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.decayFactor')}</label>
                              <input type="number" step={0.01} value={s.cfg?.ret_alpha ?? 0.94} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, ret_alpha:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                            </div>
                            <div>
                              <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.windowLength')}</label>
                              <input type="number" value={s.cfg?.ret_window ?? 60} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, ret_window:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                            </div>
                          </div>
                        )}
                      </div>

                      {/* 风险指标配置 */}
                      <div className="rounded-lg border p-3 text-sm space-y-3">
                        <div>
                          <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.riskMetric')}</label>
                          <select value={s.cfg?.risk_metric||'vol'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_metric:e.target.value}}:x))} className="mt-1 w-full rounded-lg border-slate-300">
                            <option value="vol">{systemText('preInvestment.classAllocation.volatility')}</option>
                            <option value="annual_vol">{systemText('preInvestment.classAllocation.annualVolatility2')}</option>
                            <option value="ewm_vol">{systemText('preInvestment.classAllocation.exponentiallyWeightedVolatility')}</option>
                            <option value="var">VaR</option>
                            <option value="es">ES</option>
                            <option value="max_drawdown">{systemText('preInvestment.classAllocation.maximumDrawdown2')}</option>
                            <option value="downside_vol">{systemText('preInvestment.classAllocation.downsideVolatility')}</option>
                          </select>
                        </div>
                        {s.cfg?.risk_metric==='annual_vol' && (
                          <div>
                            <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.annualizationDays')}</label>
                            <input type="number" value={s.cfg?.risk_days ?? 252} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_days:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                          </div>
                        )}
                        {s.cfg?.risk_metric==='ewm_vol' && (
                          <div className="grid grid-cols-2 gap-3">
                            <div>
                              <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.decayFactor')}</label>
                              <input type="number" step={0.01} value={s.cfg?.risk_alpha ?? 0.94} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_alpha:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                            </div>
                            <div>
                              <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.windowLength')}</label>
                              <input type="number" value={s.cfg?.risk_window ?? 60} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_window:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                            </div>
                          </div>
                        )}
                        {(s.cfg?.risk_metric==='var' || s.cfg?.risk_metric==='es') && (
                          <div>
                            <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.confidence4')}</label>
                            <input type="number" value={s.cfg?.risk_confidence ?? 95} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_confidence:Number(e.target.value)}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                          </div>
                        )}
                      </div>
                    </>
                  )}
                  {(s.cfg?.target==='risk_min_given_return') && (
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.targetReturn2')}</label>
                        <input type="number" value={s.cfg?.target_return == null ? '' : Number((s.cfg.target_return * 100).toFixed(4))} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, target_return: Number(e.target.value) / 100}}:x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                      </div>
                    </div>
                  )}
                  {(s.cfg?.target==='return_max_given_risk') && (
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.targetRisk2')}</label>
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
                              <th scope="col" className="px-3 py-2 text-left text-xs text-slate-600">{systemText('preInvestment.classAllocation.assetName')}</th>
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
                        <thead className="bg-slate-50 text-xs text-slate-600"><tr><th scope="col" className="px-3 py-2 text-left">{systemText('preInvestment.classAllocation.assetName')}</th><th scope="col" className="px-3 py-2 text-left">{systemText('preInvestment.classAllocation.fundingWeight')}</th></tr></thead>
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
                        <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.windowMode')}</label>
                        <select value={s.cfg?.window_mode || 'rollingN'} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, cfg: { ...(x.cfg||{}), window_mode: e.target.value } } : x))} className="mt-1 w-full rounded-lg border-slate-300">
                          <option value="all">{systemText('preInvestment.classAllocation.allData')}</option>
                          <option value="rollingN">{systemText('preInvestment.classAllocation.mostRecentNObservations')}</option>
                        </select>
                      </div>
                      { (s.cfg?.window_mode==='rollingN') && (
                        <div>
                          <label className="block text-xs text-slate-600">{systemText('preInvestment.classAllocation.nTradingDays')}</label>
                          <input type="number" min={2} value={s.cfg?.data_len ?? 60} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, cfg: { ...(x.cfg||{}), data_len: Number(e.target.value) } } : x))} className="mt-1 w-full rounded-lg border-slate-300 px-2 py-1"/>
                        </div>
                      )}
                    </div>
                    <p className="mt-2 text-xs text-slate-600">
                      {systemText('preInvestment.classAllocation.modeGuidance')}<span className="ml-1 font-medium">{systemText('preInvestment.classAllocation.allData')}</span> {" " + systemText('preInvestment.classAllocation.usesAllSamplesFromBacktestStartTo')}<span className="ml-1 font-medium">{systemText('preInvestment.classAllocation.mostRecentNObservations')}</span> {" " + systemText('preInvestment.classAllocation.usesTheNObservationsPrecedingTheCurrent')}</p>
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
                          setError(e?.message || systemText('preInvestment.classAllocation.weightCalculationFailed'));
                        }finally{
                          setBusyStrategy(null);
                          setBtBusy(false);
                        }
                      }}
                    >{systemText('preInvestment.classAllocation.inferFundingWeights')}</button>
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
                  if (assetNames.length === 0) { setError(systemText('preInvestment.classAllocation.loadASchemeFirst')); return; }
                  setShowAddPicker(true);
                }}
                className="rounded-lg bg-accent-600 text-white px-3 py-2 text-sm">
                {systemText('preInvestment.classAllocation.addPortfolioStrategy')}</button>
            </div>
          )}
              {showAddPicker && (
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <span className="text-sm text-slate-700">{systemText('preInvestment.classAllocation.selectStrategyType')}</span>
                  {(['fixed','risk_budget','target'] as StrategyType[]).map(t => (
                    <button key={t} disabled={equalWeightLoading} className="rounded-lg bg-slate-100 px-3 py-1 text-sm disabled:cursor-wait disabled:opacity-60" onClick={async () => {
                      const id = `s${Date.now()}`;
                      if (t === 'fixed') {
                        setError('');
                        try {
                          const eqArr = await loadEqualPercents(assetNames);
                          setStrategies(prev => {
                          const rows = assetNames.map((n,i)=> ({ className:n, weight: eqArr[i], budget: 100 }));
                          const name = uniqueStrategyName(systemText('preInvestment.classAllocation.fixedWeightStrategy'), prev);
                          return [...prev, { id, type: t, name, rows, cfg: { mode: 'equal' } }];
                          });
                        } catch (reason) {
                          setError(reason instanceof Error ? reason.message : systemText('preInvestment.classAllocation.equalWeightCalculationFailed'));
                          return;
                        }
                      } else if (t==='risk_budget') {
                        setStrategies(prev => {
                          const rows = assetNames.map(n=> ({ className:n, budget:100, weight: null }));
                          const name = uniqueStrategyName(systemText('preInvestment.classAllocation.riskBudgetStrategy'), prev);
                          return [...prev, { id, type: t, name, rows, cfg: { risk_metric:'vol', window_mode:'rollingN', data_len:60 } }];
                        });
                      } else {
                        setStrategies(prev => {
                          const rows = assetNames.map(n=> ({ className:n, weight: null }));
                          const name = uniqueStrategyName(systemText('preInvestment.classAllocation.targetStrategy'), prev);
                          return [...prev, { id, type: t, name, rows, cfg: { target:'min_risk', return_metric:'cumulative', window_mode:'all', data_len:60 } }];
                        });
                      }
                      setShowAddPicker(false);
                    }}>{t==='fixed'?systemText('preInvestment.classAllocation.fixedWeights'): t==='risk_budget'?systemText('preInvestment.classAllocation.riskBudget'):systemText('preInvestment.classAllocation.specifiedTarget')}</button>
                  ))}
                  <button className="ml-2 rounded-lg bg-white border px-2 py-1 text-xs" onClick={()=> setShowAddPicker(false)}>{systemText('preInvestment.classAllocation.cancel')}</button>
                </div>
              )}

          {/* 策略回测 */}
          <div className="rounded-lg border p-4">
            <h3 className="font-medium text-slate-700">{systemText('preInvestment.classAllocation.strategyBacktest')}</h3>
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
            <p className="mt-2 text-xs text-slate-600">{systemText('preInvestment.classAllocation.backtestsWeightSolvingAndCandidateConstructionShare')}</p>
            <div className="mt-2 flex flex-wrap items-center gap-3">
              <label htmlFor="saa-backtest-start" className="text-sm text-slate-600">{systemText('preInvestment.classAllocation.backtestStartDate')}</label>
              <input id="saa-backtest-start" type="date" value={btStart} onChange={e=> setBtStart(e.target.value)} className="rounded-lg border-slate-300 px-2 py-1"/>
              <label htmlFor="saa-backtest-end" className="text-sm text-slate-600">{systemText('preInvestment.classAllocation.backtestEndDate')}</label>
              <input id="saa-backtest-end" type="date" value={endDate} onChange={e => setEndDate(e.target.value)} className="rounded-lg border-slate-300 px-2 py-1" />
              <button
                ref={backtestButtonRef}
                disabled={btBusy || !strategies.length || loadedAllocation !== selectedAlloc}
                className="rounded-lg bg-accent-600 px-3 py-2 text-sm text-white disabled:opacity-50"
                onClick={async ()=>{
                try{
                  if(!selectedAlloc || loadedAllocation !== selectedAlloc){ setError(systemText('preInvestment.classAllocation.loadTheCurrentAssetClassPlanFirst')); return; }
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
                  if (studyBounds.current !== expectedStudy) throw new Error(systemText('preInvestment.classAllocation.theResearchEndDateOrPlanChanged3'));
                  if(!res.ok) throw new Error(apiErrorMessage(dat, systemText('preInvestment.classAllocation.backtestFailed')));
                  assertNativeNumericalExecution(dat?.execution, systemText('preInvestment.classAllocation.assetAllocationStrategyBacktest'));
                  if (dat?.regime_conditioning) {
                    assertNativeNumericalExecution(dat.regime_conditioning.execution, systemText('preInvestment.classAllocation.historicalRegimeConditionalPortfolioStatistics'));
                  }
                  backtestInputRef.current = stableStringify({ selectedAlloc, btStart, endDate, strategies: prepared, historicalRegime, singleLimits, groupLimits, riskFreePct });
                  setBtSeries(dat);
                }catch(e:any){ setError(e?.message||systemText('preInvestment.classAllocation.backtestFailed')); }
                finally { setBtBusy(false); }
              }}>{systemText('preInvestment.classAllocation.runStrategyBacktest')}</button>
            </div>
            {btSeries && (
              <div className="mt-4 space-y-6">
                <PitDecisionNotice lineage={btSeries.pit} />
                {btSeries.research_interval && <p className="text-xs text-slate-600">{systemText('preInvestment.classAllocation.actualBacktestInterval')}{btSeries.research_interval.actual_start ?? '—'} {" " + systemText('preInvestment.classAllocation.to') + " "}{btSeries.research_interval.actual_end ?? '—'}。</p>}
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
                    name:systemText('preInvestment.classAllocation.portfolioNav'),
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
      <details className="mt-5 rounded-xl border bg-white p-4"><summary className="cursor-pointer text-sm">{systemText('preInvestment.classAllocation.riskStressTestForSavedPortfolios')}</summary><PortfolioRiskSection context={systemText('preInvestment.classAllocation.selectASavedProductPortfolioGeneratedBy')} /></details>
    </div>
  );
}
