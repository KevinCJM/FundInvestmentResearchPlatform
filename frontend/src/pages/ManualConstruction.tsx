import { systemText, useI18n, i18n } from '../i18n/runtime'
import React, { useMemo, useState, useEffect } from 'react'
import ReactECharts from 'echarts-for-react'
import {
  AllocationMetricsReview,
  type HorizontalMetricRow,
} from '../components/HorizontalMetricComparison'
import { buildAnnualMetricRows, type AnnualMetricsResult } from '../utils/performance'
import { Link, useLocation, useNavigate, useSearchParams } from 'react-router-dom'
import { buildReturnNavigationState, type ReturnNavigationState } from '../utils/returnNavigation'
import { evaluateNumericControls, type NumericControlResult } from '../services/businessNumeric'
import { requestEqualWeights as requestEqualWeightsResult } from '../services/strategyWeights'
import {
  getInvestableUniverse,
  investableUniverseEligibleCount,
  searchInvestableUniverseProducts,
  type InvestableUniverseSnapshot,
} from '../services/productPools'
import {
  assertNativeNumericalExecution,
  assertNativeNumericalExecutionLanes,
  type NativeNumericalExecutionAudit,
} from '../utils/fixedNjitExecution'
import { apiErrorMessage } from '../utils/apiError'
import { Button, DataTable } from '../components/ui'
import PitProvenance from '../components/PitProvenance'
import type { PitRunLineage } from '../services/pit'
import { useResearchDay } from '../app/ResearchContext'
import { allocationJourneyPath, readAllocationJourney, updateAllocationJourney, useAllocationDraft } from '../app/allocationJourney'

type WeightMode = 'custom' | 'equal' | 'risk'
type RiskMetric = 'vol' | 'var' | 'es'

interface ETFItem {
  code: string
  name: string
  weight?: number
  riskContribution?: number
  solved?: boolean
  management?: string
  found_date?: string
  instrument_type?: 'etf' | 'fund'
  evaluation_plan_names?: string[]
  pool_names?: string[]
  max_weight?: number | null
}

interface AssetClass {
  id: string
  name: string
  mode: WeightMode
  etfs: ETFItem[]
  riskMetric?: RiskMetric
  maxLeverage?: number
}

function uid() {
  return Math.random().toString(36).slice(2, 10)
}

function clamp(n: number, a: number, b: number) {
  return Math.max(a, Math.min(b, n))
}

async function requestEqualWeights(assetCount: number, signal?: AbortSignal): Promise<number[]> {
  return (await requestEqualWeightsResult(assetCount, signal)).weights
}

function productKind(item: ETFItem): 'etf' | 'fund' {
  if (item.instrument_type === 'fund' || item.code.toUpperCase().endsWith('.OF')) return 'fund'
  return 'etf'
}

export default function AssetClassConstructionPage() {
  useI18n()
  const navigate = useNavigate()
  const location = useLocation()
  const [searchParams] = useSearchParams()
  // 地址栏没带可投资域就是未选择，不回落到上次研究。
  const universeId = searchParams.get('universe') ?? ''
  const platformAsOf = useResearchDay()
  const [draft, setDraft] = useAllocationDraft(`classes:${universeId || 'new'}`, () => ({
    classes: [
      { id: uid(), name: systemText('preInvestment.manualConstruction.equities'), mode: 'custom', etfs: [], riskMetric: 'vol', maxLeverage: 0 },
      { id: uid(), name: systemText('preInvestment.manualConstruction.fixedIncome'), mode: 'equal', etfs: [], riskMetric: 'vol', maxLeverage: 0 },
    ] as AssetClass[], startDate: '2020-01-01', allocName: '', savedName: '', rollWindow: 60, rollTargetClass: '',
  }))
  const classes = draft.classes
  const duplicateProducts = useMemo(() => {
    const owners = new Map<string, string>()
    const conflicts: string[] = []
    for (const assetClass of classes) for (const product of assetClass.etfs) {
      const key = `${productKind(product)}:${product.code}`
      const owner = owners.get(key)
      if (owner) conflicts.push(systemText('preInvestment.manualConstruction.appearsInBothAndRetainOneAssignment', { p0: product.name, p1: owner, p2: assetClass.name }))
      else owners.set(key, assetClass.name)
    }
    return conflicts
  }, [classes, i18n.language])
  const setClasses: React.Dispatch<React.SetStateAction<AssetClass[]>> = action => setDraft(current => ({ ...current, savedName: '', classes: typeof action === 'function' ? action(current.classes) : action }))
  const [actionMessage, setActionMessage] = useState('')
  const [actionError, setActionError] = useState('')
  const [selectedProducts, setSelectedProducts] = useState<ETFItem[]>([])
  const productReturnState = useMemo(
    () => buildReturnNavigationState(location, systemText('preInvestment.manualConstruction.returnToManualAssetConstruction')),
    [location.hash, location.pathname, location.search, i18n.language],
  )
  const [equalWeightCache, setEqualWeightCache] = useState<Record<number, number[]>>({})
  const [classControls, setClassControls] = useState<Record<string, NumericControlResult>>({})
  const [classControlError, setClassControlError] = useState('')

  const [loading, setLoading] = useState(false)
  const [universe, setUniverse] = useState<InvestableUniverseSnapshot | null>(null)
  const [universeLoading, setUniverseLoading] = useState(false)
  const [universeError, setUniverseError] = useState('')
  const [searchOpen, setSearchOpen] = useState<{ open: boolean; classId?: string }>({ open: false })
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<ETFItem[]>([])
  const [sortBy, setSortBy] = useState<'name' | 'code'>('name')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc')
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [total, setTotal] = useState(0)
  const [productSearchBusy, setProductSearchBusy] = useState(false)
  const [fitLoading, setFitLoading] = useState(false)
  const startDate = draft.startDate
  const setStartDate = (value: string) => setDraft(current => ({ ...current, startDate: value }))
  const [fitResult, setFitResult] = useState<null | { dates: string[]; navs: Record<string, number[]>; corr: Array<Array<number | null>>; corr_labels: string[]; metrics: { name: string; cumulative_return?: number | null; annual_return?: number | null; annual_vol?: number | null; sharpe?: number | null; var99?: number | null; es99?: number | null; max_drawdown?: number | null; calmar?: number | null }[]; consistency: { name: string; mean_corr?: number; pca_evr1?: number; max_te?: number }[]; annual_metrics: AnnualMetricsResult; execution: NativeNumericalExecutionAudit; pit?: PitRunLineage }>(null)
  const [rollLoading, setRollLoading] = useState(false)
  const [rollResult, setRollResult] = useState<null | { dates: string[]; series: Record<string, Array<number | null>>; metrics: { name: string; overall:number | null; mean:number | null; median:number | null; std:number | null; skew:number | null; kurtosis:number | null }[]; execution: NativeNumericalExecutionAudit }>(null)
  const rollWindow = draft.rollWindow
  const setRollWindow = (value: number) => setDraft(current => ({ ...current, rollWindow: value }))
  const classOptions = useMemo(()=> classes.map(c=> c.name), [classes])
  const rollTargetClass = draft.rollTargetClass
  const setRollTargetClass = (value: string) => setDraft(current => ({ ...current, rollTargetClass: value }))

  useEffect(() => { setFitResult(null); setRollResult(null); setActionError('') }, [classes, startDate, universeId, platformAsOf])

  useEffect(() => {
    if (!universeId) {
      setUniverse(null)
      setUniverseError('')
      return
    }
    let active = true
    setUniverseLoading(true)
    setUniverse(null)
    setUniverseError('')
    getInvestableUniverse(universeId)
      .then((snapshot) => { if (active) { setUniverse(snapshot); updateAllocationJourney({ name: snapshot.name, researchDate: snapshot.research_date, universeId: snapshot.id, poolVersionIds: snapshot.version_ids }) } })
      .catch((caught) => {
        if (active) {
          setUniverse(null)
          setUniverseError(caught instanceof Error ? caught.message : systemText('preInvestment.manualConstruction.unableToLoadTheInvestableUniverseSnapshot'))
        }
      })
      .finally(() => { if (active) setUniverseLoading(false) })
    return () => { active = false }
  }, [universeId])

  useEffect(() => {
    const controller = new AbortController()
    const counts = Array.from(new Set(
      classes
        .filter((item) => item.mode === 'equal' && item.etfs.length > 0)
        .map((item) => item.etfs.length),
    ))
    Promise.all(counts.filter((count) => !equalWeightCache[count]).map(async (count) => {
      const weights = await requestEqualWeights(count, controller.signal)
      return [count, weights] as const
    }))
      .then((entries) => {
        if (!entries.length) return
        setEqualWeightCache((current) => ({
          ...current,
          ...Object.fromEntries(entries),
        }))
      })
      .catch((reason) => {
        if ((reason as DOMException)?.name !== 'AbortError') {
          console.error(systemText('preInvestment.manualConstruction.unableToLoadNjitEqualWeights'), reason)
        }
      })
    return () => controller.abort()
  }, [classes, equalWeightCache])

  useEffect(() => {
    const controller = new AbortController()
    const groups = classes.flatMap((assetClass) => [
      {
        key: `${assetClass.id}:weight`,
        values: assetClass.mode === 'equal'
          ? (equalWeightCache[assetClass.etfs.length] ?? [])
          : assetClass.etfs.map((item) => item.weight ?? 0),
        target: 100,
        tolerance: 0.000001,
      },
      {
        key: `${assetClass.id}:risk`,
        values: assetClass.etfs.map((item) => item.riskContribution ?? 0),
        target: 100,
        tolerance: 0.000001,
      },
    ])
    setClassControls({})
    setClassControlError('')
    if (groups.length === 0) return () => controller.abort()
    evaluateNumericControls(groups, controller.signal)
      .then((response) => setClassControls(Object.fromEntries(response.items.map((item) => [item.key, item]))))
      .catch((reason) => {
        if ((reason as DOMException)?.name !== 'AbortError') setClassControlError(systemText('preInvestment.manualConstruction.njitWeightValidationIsTemporarilyUnavailable'))
      })
    return () => controller.abort()
  }, [classes, equalWeightCache])

  const metricsSummary = useMemo(() => {
    if (!fitResult?.metrics || !Array.isArray(fitResult.metrics)) {
      return { columns: [] as string[], rows: [] as HorizontalMetricRow[], annualRows: [] as HorizontalMetricRow[] }
    }
    const columns = fitResult.metrics.map(m => m.name)
    const cumulativeValues = fitResult.metrics.map(metric => Number(metric.cumulative_return ?? NaN))
    const cumulativePercentValues = cumulativeValues.map(v => Number.isFinite(v) ? v * 100 : NaN)
    const rows = [
      { label: systemText('preInvestment.manualConstruction.cumulativeReturn'), values: cumulativePercentValues },
      { label: systemText('preInvestment.manualConstruction.annualReturn'), values: fitResult.metrics.map(m => Number((m.annual_return ?? NaN) * 100)) },
      { label: systemText('preInvestment.manualConstruction.annualVolatility'), values: fitResult.metrics.map(m => Number((m.annual_vol ?? NaN) * 100)) },
      { label: systemText('preInvestment.manualConstruction.sharpeRatio'), values: fitResult.metrics.map(m => Number(m.sharpe ?? NaN)) },
      { label: systemText('preInvestment.manualConstruction.99VarDaily'), values: fitResult.metrics.map(m => Number((m.var99 ?? NaN) * 100)) },
      { label: systemText('preInvestment.manualConstruction.99EsDaily'), values: fitResult.metrics.map(m => Number((m.es99 ?? NaN) * 100)) },
      { label: systemText('preInvestment.manualConstruction.maximumDrawdown'), values: fitResult.metrics.map(m => Number((m.max_drawdown ?? NaN) * 100)) },
      { label: systemText('preInvestment.manualConstruction.calmarRatio'), values: fitResult.metrics.map(m => Number(m.calmar ?? NaN)) },
    ]
    const annualRows = buildAnnualMetricRows(columns, fitResult.annual_metrics)
    return { columns, rows, annualRows }
  }, [fitResult, i18n.language])

  const metricColumns = metricsSummary.columns
  const metricRows = metricsSummary.rows

  // --- Save/Load States ---
  const [saveModal, setSaveModal] = useState(false)
  const [loadModal, setLoadModal] = useState(false)
  const allocName = draft.allocName
  const setAllocName = (value: string) => setDraft(current => ({ ...current, allocName: value }))
  const [allocList, setAllocList] = useState<string[]>([])
  const [allocSearch, setAllocSearch] = useState('')

  function updateClass(id: string, updater: (c: AssetClass) => AssetClass) {
    setClasses((prev) => prev.map((c) => (c.id === id ? updater(c) : c)))
  }

  function setCustomWeight(classId: string, idx: number, val: number) {
    updateClass(classId, (c) => {
      const etfs = c.etfs.map((e, i) => (i === idx ? { ...e, weight: clamp(val, 0, 100) } : e))
      return { ...c, etfs }
    })
  }

  function setRiskContribution(classId: string, idx: number, val: number) {
    updateClass(classId, (c) => {
      const etfs = c.etfs.map((e, i) => (i === idx ? { ...e, riskContribution: clamp(val, 0, 100) } : e))
      return { ...c, etfs }
    })
  }

  function setMaxLeverage(classId: string, val: number) {
    updateClass(classId, (c) => ({ ...c, maxLeverage: clamp(val, 0, 100) }))
  }

  function addProductsToClass(classId: string, products: ETFItem[]) {
    const occupied = products.find(product => classes.some(assetClass => assetClass.id !== classId && assetClass.etfs.some(item => productKind(item) === productKind(product) && item.code === product.code)))
    if (occupied) { setActionError(systemText('preInvestment.manualConstruction.alreadyBelongsToAnotherAssetClassRemove', { p0: occupied.name })); return }
    updateClass(classId, assetClass => {
      const added = products.filter(product => !assetClass.etfs.some(item => productKind(item) === productKind(product) && item.code === product.code))
      const single = assetClass.etfs.length === 0 && added.length === 1
      return { ...assetClass, etfs: [...assetClass.etfs, ...added.map(product => ({ ...product, weight: assetClass.mode === 'custom' ? (single ? 100 : 0) : undefined, riskContribution: assetClass.mode === 'risk' ? (single ? 100 : 0) : undefined, solved: false }))] }
    })
    setSelectedProducts([]); setSearchOpen({ open: false }); setSearchQuery('')
    setActionMessage(products.length > 1 ? systemText('preInvestment.manualConstruction.addedProductsChooseEqualWeightsOrEnter', { p0: products.length }) : systemText('preInvestment.manualConstruction.productAddedASingleProductClassHas'))
  }

  function removeETF(classId: string, idx: number) {
    updateClass(classId, (c) => ({ ...c, etfs: c.etfs.filter((_, i) => i !== idx) }))
  }

  function addAssetClass() {
    setClasses((prev) => [...prev, { id: uid(), name: systemText('preInvestment.manualConstruction.newAssetClass'), mode: 'custom', etfs: [], riskMetric: 'vol', maxLeverage: 0 }])
  }

  function deleteAssetClass(id: string) {
    setClasses((prev) => prev.filter((c) => c.id !== id))
  }

  async function onSolveRiskWeights(classId: string) {
    const ac = classes.find((c) => c.id === classId)
    if (!ac) return
    if (ac.mode !== 'risk') {
      alert(systemText('preInvestment.manualConstruction.switchToRiskParityFirst'))
      return
    }
    const riskControl = classControls[`${ac.id}:risk`]
    if (!riskControl) {
      alert(classControlError || systemText('preInvestment.manualConstruction.theNjitKernelIsValidatingRiskContributions'))
      return
    }
    if (!riskControl.within_tolerance) {
      alert(systemText('preInvestment.manualConstruction.riskContributionsMustTotal100AdjustThem'))
      return
    }
    try {
      setLoading(true)
      const payload = {
        assetClassId: ac.id,
        riskMetric: ac.riskMetric ?? 'vol',
        maxLeverage: ac.maxLeverage ?? 0,
        etfs: ac.etfs.map((e) => ({ code: e.code, name: e.name, riskContribution: e.riskContribution ?? 0 })),
      }
      const resp = await fetch('/api/risk-parity/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!resp.ok) throw new Error(systemText('preInvestment.manualConstruction.backendReturnedErrorStatus', { p0: resp.status }))
      const data: { weights: number[]; execution: NativeNumericalExecutionAudit } = await resp.json()
      assertNativeNumericalExecution(data.execution, systemText('preInvestment.manualConstruction.riskParityWeightSolution'))
      if (!Array.isArray(data.weights) || data.weights.length !== ac.etfs.length || data.weights.some((weight) => typeof weight !== 'number' || !Number.isFinite(weight) || weight < 0)) {
        throw new Error(systemText('preInvestment.manualConstruction.returnedWeightCountOrValuesViolateThe'))
      }
      updateClass(classId, (c) => ({
        ...c,
        etfs: c.etfs.map((e, i) => ({ ...e, weight: data.weights[i], solved: true })),
      }))
      // 成功后直接回显（不弹窗）
    } catch (err: any) {
      console.error(err)
      alert(systemText('preInvestment.manualConstruction.calculationFailed') + err.message + "\n" + systemText('preInvestment.manualConstruction.confirmThatThePythonBackendIsRunning'))
    } finally {
      setLoading(false)
    }
  }

  async function onFit() {
    setActionError('')
    if (duplicateProducts.length) { setActionError(duplicateProducts[0]); return }
    if (platformAsOf && startDate > platformAsOf) { setActionError(systemText('preInvestment.manualConstruction.startDateIsLaterThanPlatformCutoff', { p0: startDate, p1: platformAsOf })); return }
    // 校验参数
    if (!startDate) {
      alert(systemText('preInvestment.manualConstruction.selectAStartDate'))
      return
    }
    try {
      setFitLoading(true)
      setFitResult(null)
      const classWeights = await Promise.all(classes.map(async (ac) => ({
        assetClass: ac,
        weights: ac.mode === 'equal'
          ? (equalWeightCache[ac.etfs.length] ?? await requestEqualWeights(ac.etfs.length))
          : ac.etfs.map((e) => Number(e.weight || 0)),
      })))
      const validation = await evaluateNumericControls(classWeights.map(({ assetClass, weights }) => ({
        key: assetClass.id,
        values: weights,
        target: 100,
        tolerance: 0.0001,
      })))
      const validationById = Object.fromEntries(validation.items.map((item) => [item.key, item]))
      const payloadClasses = classWeights.map(({ assetClass: ac, weights }) => {
        const control = validationById[ac.id]
        if (ac.mode === 'custom' && !control?.within_tolerance) {
          throw new Error(systemText('preInvestment.manualConstruction.fundingWeightsForAssetClassMustTotal', { p0: ac.name }))
        }
        if (ac.mode === 'risk' && !control?.positive) {
          throw new Error(systemText('preInvestment.manualConstruction.completeInferFundingWeightsForAssetClass', { p0: ac.name }))
        }
        return {
          id: ac.id,
          name: ac.name,
          etfs: ac.etfs.map((e, i) => ({ code: e.code, name: e.name, weight: weights[i] || 0 })),
        }
      })
      const resp = await fetch('/api/fit-classes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ startDate, classes: payloadClasses, universe_snapshot_id: universe?.id ?? null }),
      })
      // 严格 PIT 模式下产品域前视会被后端判掉，理由只在 detail 里；
      // 原来的 `后端错误 400` 把它吃掉了。
      if (!resp.ok) throw new Error(apiErrorMessage(await resp.json().catch(() => null), systemText('preInvestment.manualConstruction.backendError', { p0: resp.status })))
      const data = await resp.json() as NonNullable<typeof fitResult>
      assertNativeNumericalExecutionLanes(data.execution, systemText('preInvestment.manualConstruction.assetClassFitting'))
      setFitResult(data)
    } catch (e: any) {
      setActionError(systemText('preInvestment.manualConstruction.fittingFailed') + (e?.message || e))
    } finally {
      setFitLoading(false)
    }
  }

  async function onRoll() {
    if (!rollTargetClass) {
      alert(systemText('preInvestment.manualConstruction.selectTheAssetClassToStudy'))
      return
    }
    try {
      setRollLoading(true)
      setRollResult(null)
      // 准备大类及资金权重
      const payloadClasses = await Promise.all(classes.map(async (ac) => {
        const weights = ac.mode === 'equal'
          ? (equalWeightCache[ac.etfs.length] ?? await requestEqualWeights(ac.etfs.length))
          : ac.etfs.map((e)=> Number(e.weight||0))
        return {
          id: ac.id,
          name: ac.name,
          etfs: ac.etfs.map((e,i)=> ({ code: e.code, name: e.name, weight: weights[i]||0 }))
        }
      }))
      const resp = await fetch('/api/rolling-corr-classes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ startDate, window: rollWindow, targetClassName: rollTargetClass, classes: payloadClasses })
      })
      if (!resp.ok) throw new Error(systemText('preInvestment.manualConstruction.backendError', { p0: resp.status }))
      const data = await resp.json() as NonNullable<typeof rollResult>
      assertNativeNumericalExecution(data.execution, systemText('preInvestment.manualConstruction.rollingAssetClassCorrelations'))
      setRollResult(data)
    } catch (e:any) {
      alert(systemText('preInvestment.manualConstruction.rollingCorrelationCalculationFailed') + (e?.message||e))
    } finally {
      setRollLoading(false)
    }
  }

  const enterPortfolioResearch = () => {
    if (!universe) {
      navigate('/pre-investment/product-pool')
      return
    }
    const constituents = classes.flatMap((assetClass) => assetClass.etfs.map((item) => ({
      kind: item.instrument_type ?? 'etf',
      product_id: item.code,
      code: item.code,
      name: item.name,
      weight: 0,
      risk_budget: 0,
      asset_class_id: assetClass.id,
      asset_class_name: assetClass.name,
    })))
    sessionStorage.setItem('portfolioResearchImport', JSON.stringify({
      name: systemText('preInvestment.manualConstruction.researchPortfolioFromManualAssetClasses'),
      method: 'equal_weight',
      universe_snapshot_id: universe.id,
      constituents,
    }))
    navigate(`/pre-investment/product-allocation-timing/construction?universe=${encodeURIComponent(universe.id)}`)
  }

  // Automatic construction is an explicit handoff, scoped to the same universe.
  useEffect(() => {
    const imported = sessionStorage.getItem('autoClassificationImport')
    if (!imported) return
    try {
      const parsed = JSON.parse(imported)
      if (parsed.universe_snapshot_id === universeId && Array.isArray(parsed.classes) && parsed.classes.length) {
        setClasses(parsed.classes)
        sessionStorage.removeItem('autoClassificationImport')
      }
    } catch { sessionStorage.removeItem('autoClassificationImport') }
  }, [universeId])

  useEffect(() => {
    const controller = new AbortController()
    if (!universeId || !universe) {
      setSearchResults([])
      setTotal(0)
      return () => controller.abort()
    }
    setProductSearchBusy(true)
    // 与风险等级配置中心的选择器一致，防抖 180ms 再发请求，不随打字逐键调用接口。
    const timer = window.setTimeout(() => {
      searchInvestableUniverseProducts(universeId, {
        query: searchQuery,
        eligibleOnly: true,
        page,
        pageSize,
        signal: controller.signal,
      })
        .then((response) => {
          const mapped = response.items.map((item) => {
            const sources = Array.isArray(item.evaluation_sources)
              ? item.evaluation_sources
              : []
            return {
              code: item.product_id,
              name: item.name || item.code || item.product_id,
              instrument_type: item.kind,
              evaluation_plan_names: [...new Set(sources.map((source) => source.evaluation_plan_name).filter(Boolean))],
              pool_names: [...new Set(sources.map((source) => source.pool_name).filter(Boolean))],
              max_weight: item.max_weight,
            }
          })
          mapped.sort((left, right) => {
            const leftValue = sortBy === 'code' ? left.code : left.name
            const rightValue = sortBy === 'code' ? right.code : right.name
            return leftValue.localeCompare(rightValue) * (sortDir === 'asc' ? 1 : -1)
          })
          setSearchResults(mapped)
          setTotal(response.total)
          setProductSearchBusy(false)
        })
        .catch((caught) => {
          if ((caught as DOMException)?.name !== 'AbortError') {
            setSearchResults([])
            setTotal(0)
            setUniverseError(caught instanceof Error ? caught.message : systemText('preInvestment.manualConstruction.unableToLoadInvestableUniverseProducts'))
            setProductSearchBusy(false)
          }
        })
    }, 180)
    return () => { window.clearTimeout(timer); controller.abort() }
  }, [page, pageSize, searchQuery, sortBy, sortDir, universe, universeId])

  const busy = loading || fitLoading || rollLoading || universeLoading

  // --- Save/Load Handlers ---
  async function handleSave(continueToSaa = false) {
    if (duplicateProducts.length) { setActionError(duplicateProducts[0]); setSaveModal(false); return }
    const name = allocName.trim()
    if (!name) {
      alert(systemText('preInvestment.manualConstruction.configurationNameCannotBeBlank'))
      return
    }
    if (allocList.includes(name)) {
      setActionError(systemText('preInvestment.manualConstruction.configurationAlreadyExistsSaveUnderANew', { p0: name }))
      return
    }
    try {
      setLoading(true)
      const payloadClasses = await Promise.all(classes.map(async (ac) => {
        const weights = ac.mode === 'equal'
          ? (equalWeightCache[ac.etfs.length] ?? await requestEqualWeights(ac.etfs.length))
          : ac.etfs.map((e) => Number(e.weight || 0))
        return {
          id: ac.id,
          name: ac.name,
          etfs: ac.etfs.map((e, i) => ({ code: e.code, name: e.name, weight: weights[i] || 0 })),
        }
      }))
      const resp = await fetch('/api/save-allocation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ asset_alloc_name: name, classes: payloadClasses, universe_snapshot_id: universe?.id ?? null }),
      })
      const data = await resp.json()
      if (!resp.ok) throw new Error(apiErrorMessage(data, systemText('preInvestment.manualConstruction.error', { p0: resp.status })))
      setActionError('')
      setActionMessage(systemText('preInvestment.manualConstruction.assetConfigurationSavedContinueToLongTerm', { p0: name }))
      setDraft(current => ({ ...current, savedName: name, allocName: name }))
      updateAllocationJourney({ allocationName: name, universeId: universe?.id, baselineId: undefined, taaRunId: undefined })
      setSaveModal(false)
      setAllocList(p => Array.from(new Set([...p, name])).sort())
      if (continueToSaa) navigate(allocationJourneyPath('saa', { ...readAllocationJourney(), allocationName: name, universeId }))
    } catch (e: any) {
      setActionError(systemText('preInvestment.manualConstruction.saveFailed') + (e.message || e))
      setSaveModal(false)
    } finally {
      setLoading(false)
    }
  }

  async function handleLoad(name: string) {
    if (!name) return
    try {
      setLoading(true)
      const resp = await fetch(`/api/load-allocation?name=${encodeURIComponent(name)}`)
      const data = await resp.json()
      if (!resp.ok) throw new Error(apiErrorMessage(data, systemText('preInvestment.manualConstruction.error', { p0: resp.status })))
      if (!Array.isArray(data)) throw new Error(systemText('preInvestment.manualConstruction.savedAssetConfigurationHasAnInvalidFormat'))
      setClasses(data.map(assetClass => ({ ...assetClass, mode: assetClass.mode || 'custom' })))
      setAllocName(name)
      setActionMessage(systemText('preInvestment.manualConstruction.importedCheckThatProductsBelongToThe', { p0: name }))
      setLoadModal(false)
      setAllocSearch('')
    } catch (e: any) {
      alert(systemText('preInvestment.manualConstruction.loadingFailed') + (e.message || e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    const fetchAllocations = async () => {
      try {
        const resp = await fetch('/api/list-allocations')
        if (resp.ok) {
          const data = await resp.json()
          if (Array.isArray(data)) setAllocList(data.sort())
        }
      } catch (e) {
        console.error("Failed to fetch allocation list", e)
      }
    }
    fetchAllocations()
  }, [])

  // 内嵌批量选择器：托盘随搜索词、排序与翻页保留，只在确认时一次性写回所属大类。
  const productPicker = searchOpen.open && universe && (
    <section className="mt-2 space-y-3 rounded-lg border border-slate-200 bg-slate-50/60 p-3" aria-label={systemText('preInvestment.manualConstruction.addProductsFromTheInvestableUniverse')}>
      <div className="flex items-center justify-between gap-2">
        <h4 className="text-sm font-semibold text-slate-800">{systemText('preInvestment.manualConstruction.addProductsFromTheInvestableUniverse')}</h4>
        <button type="button" className="text-slate-600" aria-label={systemText('preInvestment.manualConstruction.cancelAddingProducts')} onClick={() => setSearchOpen({ open: false })}>✕</button>
      </div>
      <input
        autoFocus
        value={searchQuery}
        onChange={(e) => { setSearchQuery(e.target.value); setPage(1) }}
        placeholder={systemText('preInvestment.manualConstruction.searchByCodeNameOrEvaluationPlan')}
        className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-accent-500"
      />
      <DataTable<ETFItem>
        caption={systemText('preInvestment.manualConstruction.investableUniverseProductResults')}
        rows={searchResults}
        rowKey={(item) => `${productKind(item)}:${item.code}`}
        minWidth="480px"
        maxHeight="24rem"
        loading={productSearchBusy ? systemText('preInvestment.manualConstruction.searching') : undefined}
        empty={systemText('preInvestment.manualConstruction.noMatches')}
        columns={[
          { header: systemText('preInvestment.manualConstruction.products'), cell: (etf) => <span className="block min-w-0"><span className="block break-words font-medium text-slate-900">{etf.name}</span><span className="block break-all text-xs text-slate-600">{etf.code} · {etf.instrument_type === 'etf' ? 'ETF' : systemText('preInvestment.manualConstruction.mutualFund')}</span></span> },
          { header: systemText('preInvestment.manualConstruction.evaluationPlanProductPool'), cell: (etf) => <span className="block break-words text-xs text-slate-600"><span className="block truncate">{systemText('preInvestment.manualConstruction.evaluationPlan')}{etf.evaluation_plan_names?.join('、') || '—'}</span><span className="block truncate">{systemText('preInvestment.manualConstruction.productPool')}{etf.pool_names?.join('、') || '—'}</span></span> },
          { header: systemText('preInvestment.manualConstruction.select'), cell: (etf) => {
            const owner = classes.find((assetClass) => assetClass.etfs.some((item) => productKind(item) === productKind(etf) && item.code === etf.code))
            const selected = selectedProducts.some((item) => item.code === etf.code && productKind(item) === productKind(etf))
            return <label className="flex min-h-10 items-center gap-2"><input type="checkbox" aria-label={`${etf.name} ${etf.code}`} disabled={Boolean(owner)} checked={Boolean(owner) || selected} onChange={() => setSelectedProducts((current) => current.some((item) => item.code === etf.code && productKind(item) === productKind(etf)) ? current.filter((item) => item.code !== etf.code || productKind(item) !== productKind(etf)) : [...current, etf])} /><span className="text-xs text-slate-600">{owner ? systemText('preInvestment.manualConstruction.alreadyAssignedTo', { p0: owner.name }) : selected ? systemText('preInvestment.manualConstruction.selected') : systemText('preInvestment.manualConstruction.select')}</span></label>
          } },
        ]}
      />
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600">
        <div className="flex flex-wrap items-center gap-2">
          <label>{systemText('preInvestment.manualConstruction.sort')}</label>
          <select className="rounded-lg border border-slate-300 px-2 py-1" value={sortBy} onChange={(e) => { setSortBy(e.target.value as any); setPage(1) }}><option value="name">{systemText('preInvestment.manualConstruction.name')}</option><option value="code">{systemText('preInvestment.manualConstruction.code')}</option></select>
          <select className="rounded-lg border border-slate-300 px-2 py-1" value={sortDir} onChange={(e) => { setSortDir(e.target.value as any); setPage(1) }}><option value="asc">{systemText('preInvestment.manualConstruction.ascending')}</option><option value="desc">{systemText('preInvestment.manualConstruction.descending')}</option></select>
          <label>{systemText('preInvestment.manualConstruction.perPage')}</label>
          <select className="rounded-lg border border-slate-300 px-2 py-1" value={pageSize} onChange={(e) => { setPageSize(Number(e.target.value)); setPage(1) }}>{[5, 10, 20, 50].map((n) => <option key={n} value={n}>{n}</option>)}</select>
        </div>
        <div className="flex items-center gap-2">
          <span>{systemText('preInvestment.manualConstruction.total') + " "}{total} {" " + systemText('preInvestment.manualConstruction.items')}</span>
          <button type="button" className="rounded-lg border border-slate-300 px-2 py-1 disabled:opacity-40" disabled={page <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))}>{systemText('preInvestment.manualConstruction.previous')}</button>
          <span>{systemText('preInvestment.manualConstruction.month') + " "}{page} {" " + systemText('preInvestment.manualConstruction.pageS')}</span>
          <button type="button" className="rounded-lg border border-slate-300 px-2 py-1 disabled:opacity-40" disabled={page * pageSize >= total} onClick={() => setPage((p) => p + 1)}>{systemText('preInvestment.manualConstruction.next')}</button>
        </div>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-200 pt-3">
        <span className="text-xs text-slate-600">{systemText('preInvestment.manualConstruction.selected') + " "}{selectedProducts.length} {" " + systemText('preInvestment.manualConstruction.productsProductsAlreadyAssignedCannotBeAdded')}</span>
        <div className="flex flex-wrap gap-2">
          <Button tone="primary" disabled={!selectedProducts.length} onClick={() => searchOpen.classId && addProductsToClass(searchOpen.classId, selectedProducts)}>{systemText('preInvestment.manualConstruction.addSelectedProducts')}{selectedProducts.length}）</Button>
          <Button onClick={() => setSearchOpen({ open: false })}>{systemText('preInvestment.manualConstruction.cancel')}</Button>
        </div>
      </div>
    </section>
  )

  return (
    <div className="mx-auto min-w-0 max-w-5xl p-2 sm:p-4 relative">
      {busy && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="rounded-xl bg-white px-6 py-4 shadow text-sm">{systemText('preInvestment.manualConstruction.calculatingPleaseWait')}</div>
        </div>
      )}
      <h1 className="text-2xl font-semibold">{systemText('preInvestment.manualConstruction.assetClassConstruction')}</h1>
      <p className="text-sm text-slate-600 mt-1">{systemText('preInvestment.manualConstruction.assignProductsToAssetClassesThenSet')}</p>
      <p className="mt-2 text-xs text-slate-600">{systemText('preInvestment.manualConstruction.inputsAreSavedAutomaticallyInThisBrowser')}</p>
      {(actionMessage || actionError) && <div role={actionError ? 'alert' : 'status'} className={`mt-3 rounded-lg border p-3 text-sm ${actionError ? 'border-rose-200 bg-rose-50 text-rose-800' : 'border-emerald-200 bg-emerald-50 text-emerald-900'}`}>{actionError || actionMessage}</div>}
      {duplicateProducts.length > 0 && <div role="alert" className="mt-3 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">{duplicateProducts.map(message => <p key={message}>{message}</p>)}</div>}
      {draft.savedName && <Link to={allocationJourneyPath('saa', { ...readAllocationJourney(), allocationName: draft.savedName, universeId })} className="mt-3 inline-block rounded-lg bg-emerald-800 px-4 py-2 text-sm font-semibold text-white">{systemText('preInvestment.manualConstruction.continueToSaa')}{draft.savedName} →</Link>}
      {universe ? (
        <div className="mt-4 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900">
          {systemText('preInvestment.manualConstruction.lockedInvestableUniverse')}<b>{universe.name}</b> · {investableUniverseEligibleCount(universe)} {" " + systemText('preInvestment.manualConstruction.availableProductsResearchDate') + " "}{universe.research_date}
        </div>
      ) : (
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          <span>{universeError || systemText('preInvestment.manualConstruction.noProductPoolVersionIsLockedProducts')}</span>
          <Link to="/pre-investment/product-pool" className="rounded-lg bg-amber-800 px-3 py-2 font-medium text-white">{systemText('preInvestment.manualConstruction.selectProductPoolVersions')}</Link>
        </div>
      )}

      <div className="mt-5 rounded-xl border border-slate-200 bg-white p-4">
        <div className="flex justify-between items-center">
          <SectionTitle title={systemText('preInvestment.manualConstruction.buildAssetClasses')} />
          <button 
            className="rounded-lg bg-slate-100 px-3 py-1 text-xs text-slate-700 hover:bg-slate-200"
            onClick={() => setLoadModal(true)} >
              {systemText('preInvestment.manualConstruction.importAssetConfiguration')}</button>
        </div>
        <div className="space-y-6">
          {classes.map((ac) => (
            <AssetClassCard
              key={ac.id}
              ac={ac}
              equalWeights={equalWeightCache[ac.etfs.length] ?? []}
              weightControl={classControls[`${ac.id}:weight`]}
              riskControl={classControls[`${ac.id}:risk`]}
              controlError={classControlError}
              on重命名={(name) => updateClass(ac.id, (c) => ({ ...c, name }))}
              onModeChange={(mode) => updateClass(ac.id, (c) => ({ ...c, mode, etfs: mode === 'custom' && c.mode === 'equal' ? c.etfs.map((item, index) => ({ ...item, weight: equalWeightCache[c.etfs.length]?.[index] ?? item.weight })) : c.etfs }))}
              onRiskMetricChange={(metric) => updateClass(ac.id, (c) => ({ ...c, riskMetric: metric }))}
              on删除={() => deleteAssetClass(ac.id)}
              onAddETF={() => { setSelectedProducts([]); setPage(1); return universe ? setSearchOpen({ open: true, classId: ac.id }) : navigate('/pre-investment/product-pool') }}
              picker={searchOpen.classId === ac.id ? productPicker : undefined}
              onRemoveETF={(idx) => removeETF(ac.id, idx)}
              onSetCustomWeight={(idx, v) => setCustomWeight(ac.id, idx, v)}
              onSetRiskContribution={(idx, v) => setRiskContribution(ac.id, idx, v)}
              onSetMaxLeverage={(v) => setMaxLeverage(ac.id, v)}
              onSolve={() => onSolveRiskWeights(ac.id)}
              onCompare={() => {
                const params = new URLSearchParams()
                params.set('ids', ac.etfs.map((item) => item.code).join(','))
                params.set('kinds', ac.etfs.map(productKind).join(','))
                navigate(`/product-research/compare?${params.toString()}`, { state: productReturnState })
              }}
              returnState={productReturnState}
              loading={loading}
            />
          ))}

          <button className="w-full rounded-xl border border-dashed border-slate-300 py-3 text-sm bg-accent-50/50 hover:bg-accent-50" onClick={addAssetClass}>
            {systemText('preInvestment.manualConstruction.addAssetClass')}</button>
        </div>
      </div>

      <div className="sticky bottom-0 z-20 mt-6 flex flex-wrap justify-center gap-3 rounded-xl border border-slate-200 bg-white/95 p-3 text-center shadow-sm">
          <button type="button" onClick={onFit} disabled={busy} className="rounded-lg border border-accent-600 bg-white px-4 py-2 text-sm font-semibold text-accent-700">{systemText('preInvestment.manualConstruction.checkAssetClassData')}</button>
          <button 
            className="rounded-lg bg-accent-600 px-6 py-2 text-sm font-semibold text-white shadow-sm hover:bg-accent-700"
            onClick={() => setSaveModal(true)} >
              {systemText('preInvestment.manualConstruction.saveCurrentAssetConfiguration')}</button>
          <details className="text-sm text-slate-600"><summary className="cursor-pointer px-3 py-2">{systemText('preInvestment.manualConstruction.otherResearchUses')}</summary><button
            className="rounded-lg border border-emerald-700 bg-white px-6 py-2 text-sm font-semibold text-emerald-800 hover:bg-accent-50 disabled:cursor-not-allowed disabled:opacity-50"
            onClick={enterPortfolioResearch}
            disabled={!universe || !classes.some((assetClass) => assetClass.etfs.length > 0)}
          >
            {systemText('preInvestment.manualConstruction.saveAsResearchPortfolioOpenPortfolioMetrics')}</button></details>
      </div>

      {/* Save Modal */}
      {saveModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-md rounded-xl bg-white p-5 shadow-xl">
            <h3 className="text-lg font-semibold">{systemText('preInvestment.manualConstruction.saveAssetConfiguration')}</h3>
            <input
              autoFocus
              value={allocName}
              onChange={(e) => setAllocName(e.target.value)}
              placeholder={systemText('preInvestment.manualConstruction.enterAConfigurationName')}
              className="mt-3 w-full rounded-lg border border-slate-300 px-3 py-2 outline-none focus:ring-2 focus:ring-accent-500"
            />
            <div className="mt-4 flex justify-end gap-3">
              <button className="rounded-lg bg-slate-100 px-4 py-2 text-sm hover:bg-slate-200" onClick={() => setSaveModal(false)}>{systemText('preInvestment.manualConstruction.cancel')}</button>
              <button className="rounded-lg bg-accent-600 px-4 py-2 text-sm text-white hover:bg-accent-700" onClick={() => void handleSave()}>{systemText('preInvestment.manualConstruction.save')}</button>
              <button className="rounded-lg bg-emerald-800 px-4 py-2 text-sm text-white" onClick={() => void handleSave(true)}>{systemText('preInvestment.manualConstruction.saveAndContinueToSaa')}</button>
            </div>
          </div>
        </div>
      )}

      {/* Load Modal */}
      {loadModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-xl rounded-xl bg-white p-5 shadow-xl">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">{systemText('preInvestment.manualConstruction.importAssetConfiguration')}</h3>
              <button className="text-slate-600" onClick={() => setLoadModal(false)}>✕</button>
            </div>
            <input
              autoFocus
              value={allocSearch}
              onChange={(e) => setAllocSearch(e.target.value)}
              placeholder={systemText('preInvestment.manualConstruction.searchByName')}
              className="mt-3 w-full rounded-lg border border-slate-300 px-3 py-2 outline-none focus:ring-2 focus:ring-accent-500"
            />
            <div className="mt-3 max-h-80 overflow-auto rounded-lg border border-slate-100">
              {allocList.filter(name => name.toLowerCase().includes(allocSearch.toLowerCase())).map(name => (
                <button
                  key={name}
                  onClick={() => handleLoad(name)}
                  className="flex w-full items-center justify-between border-b px-4 py-2 text-left hover:bg-slate-50"
                >
                  <span className="text-sm text-slate-800">{name}</span>
                  <span className="text-xs text-slate-600">{systemText('preInvestment.manualConstruction.import')}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* 拟合区域 */}
      <div className="mt-6 rounded-xl border border-slate-200 bg-white p-4">
        <SectionTitle title={systemText('preInvestment.manualConstruction.assetClassReturnFitting')} />
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 text-sm">
            <label htmlFor="class-start-date" className="text-slate-700">{systemText('preInvestment.manualConstruction.selectStartDate')}</label>
            <input id="class-start-date" type="date" max={platformAsOf || undefined} className="border rounded-lg px-2 py-1" value={startDate} onChange={(e)=>setStartDate(e.target.value)} />
          </div>
          <button className="rounded-lg bg-accent-600 px-3 py-1 text-xs text-white hover:bg-accent-700" onClick={onFit} disabled={busy}>
            {systemText('preInvestment.manualConstruction.fitAssetClassReturns')}</button>
        </div>
        {fitResult && (
          <div className="mt-4 space-y-6">
            {/* 这些数字是按哪一天、哪个产品域算出来的——自动构建大类一直印着，手动这边没有。 */}
            <PitProvenance lineage={fitResult.pit} />
            <div>
              {(() => {
                const keys = Object.keys(fitResult.navs)
                return (
                  <ReactECharts style={{ height: 360 }} option={{
                    title: { text: systemText('preInvestment.manualConstruction.syntheticNavStartsAt1'), left: 0, top: 0, textStyle: { fontSize: 13, fontWeight: 600 } },
                    tooltip: { trigger: 'axis', valueFormatter: (v:any)=> Number(v).toFixed(2) },
                    legend: { top: 0, right: 0 },
                    grid: { left: 56, right: 16, top: 36, bottom: 86 },
                    xAxis: { type: 'category', data: fitResult.dates, axisLabel: { showMaxLabel: true, hideOverlap: true, margin: 12 } },
                    // 自动范围 + padding，通过函数形式按数据动态设置范围
                    yAxis: {
                      type: 'value',
                      scale: true,
                      min: (v:any) => (Number.isFinite(v.min) && Number.isFinite(v.max)) ? v.min - (v.max - v.min) * 0.05 : 'dataMin',
                      max: (v:any) => (Number.isFinite(v.min) && Number.isFinite(v.max)) ? v.max + (v.max - v.min) * 0.05 : 'dataMax',
                      axisLabel: { formatter: (val:any)=> Number(val).toFixed(2) },
                    },
                    dataZoom: [
                      { type: 'inside' },
                      { type: 'slider', bottom: 36, height: 18 },
                    ],
                    series: keys.map((k)=>({
                      name: k,
                      type: 'line',
                      smooth: false, // 不使用平滑曲线
                      symbol: 'none',
                      lineStyle: { width: 2 },
                      data: fitResult.navs[k]
                    }))
                  }} />
                )
              })()}
            </div>
            <div>
              <h3 className="text-sm font-semibold mb-2">{systemText('preInvestment.manualConstruction.correlationMatrix')}</h3>
              <ReactECharts style={{ height: 320 }} option={(function(){
                const labels = fitResult.corr_labels
                const data: any[] = []
                for(let i=0;i<labels.length;i++){
                  for(let j=0;j<labels.length;j++){
                    data.push([i, j, fitResult.corr[i][j]])
                  }
                }
                return {
                  tooltip: { position: 'top', formatter: (p:any)=> `${labels[p.data[1]]} vs ${labels[p.data[0]]}: ${p.data[2] == null ? '—' : Number(p.data[2]).toFixed(2)}` },
                  grid: { left: 80, right: 16, top: 16, bottom: 40 },
                  xAxis: { type: 'category', data: labels, axisLabel: { rotate: 30 } },
                  yAxis: { type: 'category', data: labels },
                  // 配色：白色 → 天蓝色，隐藏色度图例
                  visualMap: {
                    min: -1, max: 1, show: false,
                    inRange: { color: ['#ffffff','#e0f2fe','#bae6fd','#7dd3fc','#38bdf8','#0ea5e9'] }
                  },
                  series: [{
                    type: 'heatmap',
                    data,
                    label: { show: true, formatter: (p:any)=> p.data[2] == null ? '—' : Number(p.data[2]).toFixed(2), color: '#111827' },
                    emphasis: { itemStyle: { shadowBlur: 5, shadowColor: 'rgba(0,0,0,0.3)' } }
                  }]
                }
              })()} />
            </div>
            <AllocationMetricsReview
              columns={metricColumns}
              rows={metricRows}
              annualRows={metricsSummary.annualRows}
              detailedContent={Array.isArray(fitResult.consistency) && fitResult.consistency.length > 0 ? (
                <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
                  <h4 className="mb-3 text-sm font-semibold">{systemText('preInvestment.manualConstruction.withinClassConsistency')}</h4>
                  <div className="overflow-auto">
                    <table className="text-xs border" style={{ width: '100%', tableLayout: 'fixed' }}>
                      <thead>
                        <tr>
                          <th scope="col" className="border px-2 py-2">{systemText('preInvestment.manualConstruction.assetClass')}</th>
                          <th scope="col" className="border px-2 py-2">{systemText('preInvestment.manualConstruction.meanCorrelation')}</th>
                          <th scope="col" className="border px-2 py-2">{systemText('preInvestment.manualConstruction.principalComponentExplainedVariance')}</th>
                          <th scope="col" className="border px-2 py-2">{systemText('preInvestment.manualConstruction.maximumTrackingError')}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {fitResult.consistency.map((r)=> {
                          const meanCorr = r.mean_corr as number
                          const pcaEvr1 = r.pca_evr1 as number
                          const maxTe = r.max_te as number
                          const meanCorrStyle = Number.isFinite(meanCorr) && meanCorr < 0.6 ? { backgroundColor: '#fee2e2' } : {}
                          const pcaEvr1Style = Number.isFinite(pcaEvr1) && pcaEvr1 < 0.8 ? { backgroundColor: '#fee2e2' } : {}
                          const maxTeStyle = Number.isFinite(maxTe) && maxTe * 100 > 5 ? { backgroundColor: '#fee2e2' } : {}

                          return (
                            <tr key={r.name}>
                              <td className="border px-2 py-2">{r.name}</td>
                              <td className="border px-2 py-2 text-right" style={meanCorrStyle}>{Number.isFinite(meanCorr) ? meanCorr.toFixed(3) : '-'}</td>
                              <td className="border px-2 py-2 text-right" style={pcaEvr1Style}>{Number.isFinite(pcaEvr1) ? (pcaEvr1 * 100).toFixed(2) : '-'}</td>
                              <td className="border px-2 py-2 text-right" style={maxTeStyle}>{Number.isFinite(maxTe) ? (maxTe * 100).toFixed(2) : '-'}</td>
                            </tr>
                          )
                        })}</tbody>
                    </table>
                  </div>
                </div>
              ) : undefined}
            />
          </div>
        )}
      </div>

      {/* 滚动相关性研究 */}
      <details className="mt-6 rounded-xl border border-slate-200 bg-white p-4">
        <summary className="mb-3 cursor-pointer text-lg font-semibold">{systemText('preInvestment.manualConstruction.rollingCorrelationResearch')}</summary>
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 text-sm">
            <label className="text-slate-700">{systemText('preInvestment.manualConstruction.rollingWindowDays')}</label>
            <input type="number" min={5} step={1} className="border rounded-lg px-2 py-1 w-24 text-right" value={rollWindow} onChange={(e)=> setRollWindow(Number(e.target.value)||60)} />
          </div>
          <div className="flex items-center gap-2 text-sm">
            <label className="text-slate-700">{systemText('preInvestment.manualConstruction.researchTarget')}</label>
            <select className="border rounded-lg px-2 py-1" value={rollTargetClass} onChange={(e)=> setRollTargetClass(e.target.value)}>
              <option value="">{systemText('preInvestment.manualConstruction.selectAnAssetClass')}</option>
              {classOptions.map(n=> <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
          <button className="rounded-lg bg-accent-600 px-3 py-1 text-xs text-white hover:bg-accent-700" onClick={onRoll} disabled={busy}>
            {systemText('preInvestment.manualConstruction.calculateRollingCorrelations')}</button>
        </div>
        {rollResult && (
          <div className="mt-4 space-y-4">
            <ReactECharts style={{ height: 320 }} option={{
              tooltip: { trigger: 'axis', valueFormatter: (v:any)=> Number(v).toFixed(2) },
              legend: { top: 0 },
              grid: { left: 48, right: 16, top: 16, bottom: 40 },
              xAxis: { type: 'category', data: rollResult.dates },
              yAxis: { type: 'value', min: -1, max: 1 },
              dataZoom: [{ type: 'inside' }, { type: 'slider' }],
              series: Object.keys(rollResult.series).map(k=> ({ name:k, type:'line', symbol:'none', data: rollResult.series[k] }))
            }} />
            <div className="overflow-auto">
              <table className="text-xs border" style={{ width: '100%', tableLayout: 'fixed' }}>
                <thead>
                  <tr>
                    <th scope="col" className="border px-2 py-2">{systemText('preInvestment.manualConstruction.assetClass')}</th>
                    <th scope="col" className="border px-2 py-2">{systemText('preInvestment.manualConstruction.overallCorrelation')}</th>
                    <th scope="col" className="border px-2 py-2">{systemText('preInvestment.manualConstruction.mean')}</th>
                    <th scope="col" className="border px-2 py-2">{systemText('preInvestment.manualConstruction.median')}</th>
                    <th scope="col" className="border px-2 py-2">{systemText('preInvestment.manualConstruction.standardDeviation')}</th>
                    <th scope="col" className="border px-2 py-2">{systemText('preInvestment.manualConstruction.skewness')}</th>
                    <th scope="col" className="border px-2 py-2">{systemText('preInvestment.manualConstruction.kurtosis')}</th>
                  </tr>
                </thead>
                <tbody>
                  {rollResult.metrics.map(m=> (
                    <tr key={m.name}>
                      <td className="border px-2 py-2">{m.name}</td>
                      <td className="border px-2 py-2 text-right">{m.overall !== null && Number.isFinite(m.overall) ? m.overall.toFixed(2) : '-'}</td>
                      <td className="border px-2 py-2 text-right">{m.mean !== null && Number.isFinite(m.mean) ? m.mean.toFixed(2) : '-'}</td>
                      <td className="border px-2 py-2 text-right">{m.median !== null && Number.isFinite(m.median) ? m.median.toFixed(2) : '-'}</td>
                      <td className="border px-2 py-2 text-right">{m.std !== null && Number.isFinite(m.std) ? m.std.toFixed(2) : '-'}</td>
                      <td className="border px-2 py-2 text-right">{m.skew !== null && Number.isFinite(m.skew) ? m.skew.toFixed(2) : '-'}</td>
                      <td className="border px-2 py-2 text-right">{m.kurtosis !== null && Number.isFinite(m.kurtosis) ? m.kurtosis.toFixed(2) : '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </details>
    </div>
  )
}

function SectionTitle({ title }: { title: string }) {
  useI18n()
  return (
    <div className="mb-3 flex items-center gap-2">
      <span className="text-accent-600">◆</span>
      <h2 className="text-lg font-semibold">{title}</h2>
    </div>
  )
}

function ModePill({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  useI18n()
  return (
    <button
      onClick={onClick}
      className={`rounded-lg px-2 py-1 text-xs ${active ? 'bg-accent-600 text-white' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'}`}
      style={{ padding: '0.25rem 0.5rem' }}
    >
      {label}
    </button>
  )
}

function AssetClassCard({
  ac,
  equalWeights,
  weightControl,
  riskControl,
  controlError,
  on重命名,
  onModeChange,
  onRiskMetricChange,
  on删除,
  onAddETF,
  picker,
  onRemoveETF,
  onSetCustomWeight,
  onSetRiskContribution,
  onSetMaxLeverage,
  onSolve,
  onCompare,
  returnState,
  loading,
}: {
  ac: AssetClass
  equalWeights: number[]
  weightControl?: NumericControlResult
  riskControl?: NumericControlResult
  controlError: string
  on重命名: (name: string) => void
  onModeChange: (mode: WeightMode) => void
  onRiskMetricChange: (metric: RiskMetric) => void
  on删除: () => void
  onAddETF: () => void
  /** 打开时渲染在「+ 添加新的产品」下方的内嵌选择器；只有当前大类有值，其余大类为 undefined。 */
  picker?: React.ReactNode
  onRemoveETF: (idx: number) => void
  onSetCustomWeight: (idx: number, val: number) => void
  onSetRiskContribution: (idx: number, val: number) => void
  onSetMaxLeverage: (val: number) => void
  onSolve: () => void
  onCompare: () => void
  returnState: ReturnNavigationState
  loading: boolean
}) {
  useI18n()
  const [editing, setEditing] = useState(false)
  const [tempName, setTempName] = useState(ac.name)
  useEffect(() => setTempName(ac.name), [ac.name])

  const isRisk = ac.mode === 'risk'
  const showSolved = isRisk && ac.etfs.some((e) => e.solved)

  return (
    <div className="rounded-xl border border-slate-200">
      <div className="flex flex-col gap-3 border-b bg-slate-50/80 px-3 py-2 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          {editing ? (
            <input
              value={tempName}
              onChange={(e) => setTempName(e.target.value)}
              onBlur={() => {
                on重命名(tempName.trim() || ac.name)
                setEditing(false)
              }}
              className="rounded-lg border border-slate-300 px-2 py-1 text-sm"
            />
          ) : (
            <div className="text-sm font-medium">{ac.name}</div>
          )}
          <button className="text-xs text-accent-600 underline" onClick={() => setEditing((v) => !v)} title={systemText('preInvestment.manualConstruction.rename')}>
            {editing ? systemText('preInvestment.manualConstruction.save') : systemText('preInvestment.manualConstruction.rename')}
          </button>

          <div className="flex flex-wrap items-center gap-2 lg:ml-3">
            <ModePill label={systemText('preInvestment.manualConstruction.customWeights')} active={ac.mode === 'custom'} onClick={() => onModeChange('custom')} />
            <ModePill label={systemText('preInvestment.manualConstruction.equalWeights')} active={ac.mode === 'equal'} onClick={() => onModeChange('equal')} />
            <ModePill label={systemText('preInvestment.manualConstruction.riskParity')} active={ac.mode === 'risk'} onClick={() => onModeChange('risk')} />

            {isRisk && (
              <>
                <select
                  className="ml-2 rounded-lg border border-slate-300 px-2 py-1 text-xs"
                  value={ac.riskMetric || 'vol'}
                  onChange={(e) => onRiskMetricChange(e.target.value as RiskMetric)}
                >
                  <option value="vol">{systemText('preInvestment.manualConstruction.volatility')}</option>
                  <option value="var">VaR</option>
                  <option value="es">ES</option>
                </select>

                <div className="ml-2 flex items-center gap-2 text-xs">
                  <span className="text-slate-600">{systemText('preInvestment.manualConstruction.maximumLeverage')}</span>
                  <input
                    type="number"
                    min={0}
                    step={0.01}
                    value={ac.maxLeverage ?? 0}
                    onChange={(e) => onSetMaxLeverage(Number(e.target.value))}
                    className="w-20 rounded-lg border border-slate-300 px-2 py-1 text-right"
                    title={systemText('preInvestment.manualConstruction.maximumPortfolioLeverageAllowed0MeansNo')}
                  />
                </div>

                <button
                  className="ml-2 rounded-lg bg-accent-600 px-3 py-1 text-xs text-white hover:bg-accent-700 disabled:opacity-60"
                  onClick={onSolve}
                  disabled={loading || riskControl?.within_tolerance !== true}
                  title={riskControl?.within_tolerance !== true ? systemText('preInvestment.manualConstruction.njitMustConfirmRiskContributionsTotal100') : ''}
                >
                  {loading ? systemText('preInvestment.manualConstruction.calculating') : systemText('preInvestment.manualConstruction.inferFundingWeights')}
                </button>
              </>
            )}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2 self-end lg:self-auto">
          <button
            type="button"
            onClick={onCompare}
            disabled={ac.etfs.length < 2}
            title={ac.etfs.length < 2 ? systemText('preInvestment.manualConstruction.addAtLeastTwoProductsToCompare') : systemText('preInvestment.manualConstruction.compareProductsIn', { p0: ac.name, p1: ac.etfs.length })}
            className="rounded-lg border border-accent-200 bg-white px-3 py-1 text-xs font-medium text-accent-700 hover:bg-accent-50 focus:outline-none focus:ring-2 focus:ring-accent-500 focus:ring-offset-1 disabled:cursor-not-allowed disabled:border-slate-200 disabled:text-slate-600"
          >
            {systemText('preInvestment.manualConstruction.compareProducts')}{ac.etfs.length >= 2 ? `（${ac.etfs.length}）` : ''}
          </button>
          <button className="rounded-lg border border-red-200 px-3 py-1 text-xs text-red-700 hover:bg-red-50" onClick={on删除}>
            {systemText('preInvestment.manualConstruction.deleteThisAssetClass')}</button>
        </div>
      </div>

      <div className="grid grid-cols-12 items-center gap-2 px-3 py-2 text-xs text-slate-600">
        <div className="min-w-0 col-span-7">{systemText('preInvestment.manualConstruction.productEtfMutualFund')}</div>
        <div className="min-w-0 col-span-3 text-right">{isRisk ? (showSolved ? systemText('preInvestment.manualConstruction.riskContributionFundingWeight') : systemText('preInvestment.manualConstruction.riskContribution')) : systemText('preInvestment.manualConstruction.withinClassWeight')}</div>
        <div className="col-span-2 text-right">{systemText('preInvestment.manualConstruction.actions')}</div>
      </div>

      <div className="divide-y">
        {ac.etfs.map((e, idx) => (
          <div key={idx} className="grid grid-cols-12 items-center gap-2 px-3 py-2">
            <div className="min-w-0 col-span-7">
              <div className="flex items-center gap-2">
                <span className="w-5 text-xs text-slate-600">{idx + 1}.</span>
                <div className="min-w-0 flex-1 truncate text-sm">
                  <div className="flex flex-col items-start gap-1 sm:flex-row sm:items-center sm:gap-2">
                    <span className="font-mono">{e.code}</span>
                    <Link
                      to={`/product-research/products/${encodeURIComponent(e.code)}?kind=${productKind(e)}`}
                      state={returnState}
                      className="truncate font-medium text-accent-700 hover:text-accent-600 hover:underline focus:outline-none focus:ring-2 focus:ring-accent-500 focus:ring-offset-2"
                      title={systemText('preInvestment.manualConstruction.viewProductResearchFor', { p0: e.name })}
                    >
                      {e.name}
                    </Link>
                  </div>
                  <p className="mt-1 truncate text-xs text-slate-600">{e.evaluation_plan_names?.join('、') || systemText('preInvestment.manualConstruction.fromTheLockedProductScope')}</p>
                </div>
              </div>
            </div>

            <div className="min-w-0 col-span-3 text-right">
              {ac.mode === 'custom' ? (
                <div className="inline-flex items-center gap-2">
                  <input
                    type="number"
                    min={0}
                    max={100}
                    step={0.01}
                    aria-label={systemText('preInvestment.manualConstruction.withinClassWeight2', { p0: ac.name, p1: e.name })}
                    value={e.weight ?? 0}
                    onChange={(ev) => onSetCustomWeight(idx, Number(ev.target.value))}
                    className="w-16 sm:w-24 rounded-lg border border-slate-300 px-2 py-1 text-right text-sm"
                  />
                  <span className="text-sm text-slate-600">%</span>
                </div>
              ) : ac.mode === 'equal' ? (
                <div className="pr-2 text-sm text-slate-700">{equalWeights[idx] == null ? systemText('preInvestment.manualConstruction.calculating2') : `${equalWeights[idx].toFixed(2)}%`}</div>
              ) : (
                <div className="inline-flex items-center gap-3 justify-end">
                  <input
                    type="number"
                    min={0}
                    max={100}
                    step={0.01}
                    value={e.riskContribution ?? 0}
                    onChange={(ev) => onSetRiskContribution(idx, Number(ev.target.value))}
                    className="w-16 sm:w-24 rounded-lg border border-slate-300 px-2 py-1 text-right text-sm"
                  />
                  <span className="text-sm text-slate-600">%</span>
                  {showSolved && <span className="text-xs text-slate-600">{systemText('preInvestment.manualConstruction.fundingWeight') + " "}{(e.weight ?? 0).toFixed(2)}%</span>}
                </div>
              )}
            </div>

            <div className="col-span-2 text-right">
              <button onClick={() => onRemoveETF(idx)} className="rounded-lg border border-slate-300 px-3 py-1 text-xs hover:bg-slate-50">
                {systemText('preInvestment.manualConstruction.delete')}</button>
            </div>
          </div>
        ))}

        <div className="px-3 py-2">
          <button onClick={onAddETF} className="w-full rounded-lg border border-dashed border-slate-300 py-2 text-sm hover:bg-slate-50">
            {systemText('preInvestment.manualConstruction.addProduct')}</button>
          {picker}
        </div>
      </div>

      <div className="border-t px-3 py-2 text-xs">
        {ac.mode === 'custom' ? (
          <>
            <span
              className={
                'rounded-lg px-2 py-0.5 ' + (weightControl?.within_tolerance ? 'bg-green-50 text-green-700' : 'bg-yellow-50 text-yellow-700')
              }
            >
              {systemText('preInvestment.manualConstruction.totalWithinClassWeight') + " "}{weightControl ? `${weightControl.total.toFixed(2)}%` : controlError || systemText('preInvestment.manualConstruction.njitValidationInProgress')}
            </span>
            {weightControl && !weightControl.within_tolerance && <span className="ml-2 text-yellow-700">{systemText('preInvestment.manualConstruction.mustEqual100')}</span>}
          </>
        ) : ac.mode === 'equal' ? (
          <span className={`rounded-lg px-2 py-0.5 ${weightControl?.within_tolerance ? 'bg-green-50 text-green-700' : 'bg-yellow-50 text-yellow-700'}`}>{systemText('preInvestment.manualConstruction.totalWithinClassWeight') + " "}{weightControl ? `${weightControl.total.toFixed(2)}%` : controlError || systemText('preInvestment.manualConstruction.njitValidationInProgress')}</span>
        ) : (
          <>
            <span
              className={
                'rounded-lg px-2 py-0.5 ' + (riskControl?.within_tolerance ? 'bg-green-50 text-green-700' : 'bg-yellow-50 text-yellow-700')
              }
            >
              {systemText('preInvestment.manualConstruction.totalRiskContribution') + " "}{riskControl ? `${riskControl.total.toFixed(2)}%` : controlError || systemText('preInvestment.manualConstruction.njitValidationInProgress')}
            </span>
            {riskControl && !riskControl.within_tolerance && <span className="ml-2 text-yellow-700">{systemText('preInvestment.manualConstruction.mustEqual100')}</span>}
            {showSolved && (
              <span className="ml-3 rounded-lg bg-accent-50 px-2 py-0.5 text-accent-700">{systemText('preInvestment.manualConstruction.totalWithinClassFundingWeight') + " "}{weightControl ? `${weightControl.total.toFixed(2)}%` : controlError || systemText('preInvestment.manualConstruction.njitValidationInProgress')}</span>
            )}
          </>
        )}
      </div>
    </div>
  )
}
