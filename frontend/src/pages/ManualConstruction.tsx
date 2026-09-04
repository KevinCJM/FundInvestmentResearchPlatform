import React, { useMemo, useState, useEffect } from 'react'
import ReactECharts from 'echarts-for-react'
import HorizontalMetricComparison, {
  DEFAULT_METRIC_TABLE_HEIGHT,
  PerformanceQuadrantChart,
} from '../components/HorizontalMetricComparison'
import { buildAnnualMetricRows, type AnnualMetricsResult } from '../utils/performance'
import { Link, useLocation, useNavigate, useSearchParams } from 'react-router-dom'
import { buildReturnNavigationState, type ReturnNavigationState } from '../utils/returnNavigation'
import { evaluateNumericControls, type NumericControlResult } from '../services/businessNumeric'
import { requestEqualWeights as requestEqualWeightsResult } from '../services/strategyWeights'
import {
  getInvestableUniverse,
  searchInvestableUniverseProducts,
  type InvestableUniverseSnapshot,
} from '../services/productPools'
import {
  assertFixedNjitExecution,
  type FixedNjitExecutionAudit,
} from '../utils/fixedNjitExecution'

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
  const navigate = useNavigate()
  const location = useLocation()
  const [searchParams] = useSearchParams()
  const universeId = searchParams.get('universe') ?? ''
  const productReturnState = useMemo(
    () => buildReturnNavigationState(location, '返回手动构建大类'),
    [location.hash, location.pathname, location.search],
  )
  const [classes, setClasses] = useState<AssetClass[]>([])
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
  const [fitLoading, setFitLoading] = useState(false)
  const [startDate, setStartDate] = useState<string>('2020-01-01')
  const [fitResult, setFitResult] = useState<null | { dates: string[]; navs: Record<string, number[]>; corr: Array<Array<number | null>>; corr_labels: string[]; metrics: { name: string; cumulative_return?: number | null; annual_return?: number | null; annual_vol?: number | null; sharpe?: number | null; var99?: number | null; es99?: number | null; max_drawdown?: number | null; calmar?: number | null }[]; consistency: { name: string; mean_corr?: number; pca_evr1?: number; max_te?: number }[]; annual_metrics: AnnualMetricsResult; execution: FixedNjitExecutionAudit }>(null)
  const [rollLoading, setRollLoading] = useState(false)
  const [rollResult, setRollResult] = useState<null | { dates: string[]; series: Record<string, Array<number | null>>; metrics: { name: string; overall:number | null; mean:number | null; median:number | null; std:number | null; skew:number | null; kurtosis:number | null }[]; execution: FixedNjitExecutionAudit }>(null)
  const [rollWindow, setRollWindow] = useState<number>(60)
  const classOptions = useMemo(()=> classes.map(c=> c.name), [classes])
  const [rollTargetClass, setRollTargetClass] = useState<string>('')

  useEffect(() => {
    if (!universeId) {
      setUniverse(null)
      setUniverseError('')
      return
    }
    let active = true
    setUniverseLoading(true)
    setUniverseError('')
    getInvestableUniverse(universeId)
      .then((snapshot) => { if (active) setUniverse(snapshot) })
      .catch((caught) => {
        if (active) {
          setUniverse(null)
          setUniverseError(caught instanceof Error ? caught.message : '无法加载可投资域快照。')
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
          console.error('等权 NJIT 结果加载失败', reason)
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
        if ((reason as DOMException)?.name !== 'AbortError') setClassControlError('NJIT 权重校验暂不可用')
      })
    return () => controller.abort()
  }, [classes, equalWeightCache])

  const metricsSummary = useMemo(() => {
    if (!fitResult?.metrics || !Array.isArray(fitResult.metrics)) {
      return { columns: [] as string[], rows: [] as any[] }
    }
    const columns = fitResult.metrics.map(m => m.name)
    const cumulativeValues = fitResult.metrics.map(metric => Number(metric.cumulative_return ?? NaN))
    const cumulativePercentValues = cumulativeValues.map(v => Number.isFinite(v) ? v * 100 : NaN)
    const rows = [
      { label: '累计收益率', values: cumulativeValues },
      { label: '累计收益率(%)', values: cumulativePercentValues },
      { label: '年化收益率(%)', values: fitResult.metrics.map(m => Number((m.annual_return ?? NaN) * 100)) },
      { label: '年化波动率(%)', values: fitResult.metrics.map(m => Number((m.annual_vol ?? NaN) * 100)) },
      { label: '夏普比率', values: fitResult.metrics.map(m => Number(m.sharpe ?? NaN)) },
      { label: '99%VaR(日)(%)', values: fitResult.metrics.map(m => Number((m.var99 ?? NaN) * 100)) },
      { label: '99%ES(日)(%)', values: fitResult.metrics.map(m => Number((m.es99 ?? NaN) * 100)) },
      { label: '最大回撤(%)', values: fitResult.metrics.map(m => Number((m.max_drawdown ?? NaN) * 100)) },
      { label: '卡玛比率', values: fitResult.metrics.map(m => Number(m.calmar ?? NaN)) },
    ]
    const annualRows = buildAnnualMetricRows(columns, fitResult.annual_metrics)
    const mergedRows = annualRows.length > 0 ? [...rows, ...annualRows] : rows
    return { columns, rows: mergedRows }
  }, [fitResult])

  const metricColumns = metricsSummary.columns
  const metricRows = metricsSummary.rows

  // --- Save/Load States ---
  const [saveModal, setSaveModal] = useState(false)
  const [loadModal, setLoadModal] = useState(false)
  const [allocName, setAllocName] = useState('')
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

  function addETFToClass(classId: string, etf: { code: string; name: string }) {
    updateClass(classId, (c) => {
      const exists = c.etfs.some((x) => x.code === etf.code && x.name === etf.name)
      const etfs = exists
        ? c.etfs
        : [
            ...c.etfs,
            {
              ...etf,
              weight: c.mode === 'custom' ? 0 : undefined,
              riskContribution: c.mode === 'risk' ? 0 : undefined,
              solved: false,
            },
          ]
      return { ...c, etfs }
    })
  }

  function removeETF(classId: string, idx: number) {
    updateClass(classId, (c) => ({ ...c, etfs: c.etfs.filter((_, i) => i !== idx) }))
  }

  function addAssetClass() {
    setClasses((prev) => [...prev, { id: uid(), name: '新大类', mode: 'custom', etfs: [], riskMetric: 'vol', maxLeverage: 0 }])
  }

  function deleteAssetClass(id: string) {
    setClasses((prev) => prev.filter((c) => c.id !== id))
  }

  async function onSolveRiskWeights(classId: string) {
    const ac = classes.find((c) => c.id === classId)
    if (!ac) return
    if (ac.mode !== 'risk') {
      alert('请先切换到“风险平价”模式')
      return
    }
    const riskControl = classControls[`${ac.id}:risk`]
    if (!riskControl) {
      alert(classControlError || '风险贡献正在由 NJIT 内核校验，请稍候')
      return
    }
    if (!riskControl.within_tolerance) {
      alert('风险贡献合计需等于 100%，请调整后再计算')
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
      if (!resp.ok) throw new Error(`后端返回错误状态 ${resp.status}`)
      const data: { weights: number[]; execution: FixedNjitExecutionAudit } = await resp.json()
      assertFixedNjitExecution(data.execution, '风险平价权重求解')
      if (!Array.isArray(data.weights) || data.weights.length !== ac.etfs.length || data.weights.some((weight) => typeof weight !== 'number' || !Number.isFinite(weight) || weight < 0)) {
        throw new Error('后端权重数量或数值不符合契约')
      }
      updateClass(classId, (c) => ({
        ...c,
        etfs: c.etfs.map((e, i) => ({ ...e, weight: data.weights[i], solved: true })),
      }))
      // 成功后直接回显（不弹窗）
    } catch (err: any) {
      console.error(err)
      alert('计算失败：' + err.message + '\n请确认已启动 Python 后端 (POST /api/risk-parity/solve)。')
    } finally {
      setLoading(false)
    }
  }

  async function onFit() {
    // 校验参数
    if (!startDate) {
      alert('请选择开始日期')
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
          throw new Error(`大类【${ac.name}】资金权重合计应为 100%`)
        }
        if (ac.mode === 'risk' && !control?.positive) {
          throw new Error(`大类【${ac.name}】请先完成“反推资金权重”计算`)
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
        body: JSON.stringify({ startDate, classes: payloadClasses }),
      })
      if (!resp.ok) throw new Error(`后端错误 ${resp.status}`)
      const data = await resp.json() as NonNullable<typeof fitResult>
      assertFixedNjitExecution(data.execution, '资产大类拟合')
      setFitResult(data)
    } catch (e: any) {
      alert('拟合失败：' + (e?.message || e))
    } finally {
      setFitLoading(false)
    }
  }

  async function onRoll() {
    if (!rollTargetClass) {
      alert('请选择研究对象大类')
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
      if (!resp.ok) throw new Error(`后端错误 ${resp.status}`)
      const data = await resp.json() as NonNullable<typeof rollResult>
      assertFixedNjitExecution(data.execution, '大类滚动相关性')
      setRollResult(data)
    } catch (e:any) {
      alert('滚动相关性计算失败：' + (e?.message||e))
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
      name: '来自手动大类的研究组合',
      method: 'equal_weight',
      universe_snapshot_id: universe.id,
      constituents,
    }))
    navigate(`/pre-investment/product-allocation-timing/construction?universe=${encodeURIComponent(universe.id)}`)
  }

  // 大类名称由研究员维护，具体产品只能从已锁定可投资域中加入。
  useEffect(() => {
    if (classes.length > 0) return
    // Handoff from 自动构建大类: adopt the generated draft instead of the empty default.
    const imported = sessionStorage.getItem('autoClassificationImport')
    if (imported) {
      sessionStorage.removeItem('autoClassificationImport')
      try {
        const parsed = JSON.parse(imported)
        if (Array.isArray(parsed?.classes) && parsed.classes.length > 0) {
          setClasses(parsed.classes)
          return
        }
      } catch {
        // Fall through to the empty default pool.
      }
    }
    setClasses([
      { id: uid(), name: '权益类', mode: 'custom', etfs: [], riskMetric: 'vol', maxLeverage: 0 },
      { id: uid(), name: '固收类', mode: 'equal', etfs: [], riskMetric: 'vol', maxLeverage: 0 },
    ])
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    if (!universeId || !universe) {
      setSearchResults([])
      setTotal(0)
      return () => controller.abort()
    }
    searchInvestableUniverseProducts(universeId, {
      query: searchQuery,
      eligibleOnly: true,
      page,
      pageSize,
      signal: controller.signal,
    })
      .then((response) => {
        const mapped = response.items.map((item) => ({
          code: item.product_id,
          name: item.name,
          instrument_type: item.kind,
          evaluation_plan_names: [...new Set(item.evaluation_sources.map((source) => source.evaluation_plan_name))],
          pool_names: [...new Set(item.evaluation_sources.map((source) => source.pool_name))],
          max_weight: item.max_weight,
        }))
        mapped.sort((left, right) => {
          const leftValue = sortBy === 'code' ? left.code : left.name
          const rightValue = sortBy === 'code' ? right.code : right.name
          return leftValue.localeCompare(rightValue) * (sortDir === 'asc' ? 1 : -1)
        })
        setSearchResults(mapped)
        setTotal(response.total)
      })
      .catch((caught) => {
        if ((caught as DOMException)?.name !== 'AbortError') {
          setSearchResults([])
          setTotal(0)
          setUniverseError(caught instanceof Error ? caught.message : '无法读取可投资域产品。')
        }
      })
    return () => controller.abort()
  }, [page, pageSize, searchQuery, sortBy, sortDir, universe, universeId])

  const busy = loading || fitLoading || rollLoading || universeLoading

  // --- Save/Load Handlers ---
  async function handleSave() {
    const name = allocName.trim()
    if (!name) {
      alert('配置名称不能为空')
      return
    }
    if (allocList.includes(name)) {
      if (!confirm(`配置名称 “${name}” 已存在，要覆盖吗？（此操作不可逆）`)) {
        return
      }
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
        body: JSON.stringify({ asset_alloc_name: name, classes: payloadClasses }),
      })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.detail || `错误 ${resp.status}`)
      alert(`配置 “${name}” 保存成功！`)
      setSaveModal(false)
      setAllocName('')
      setAllocList(p => Array.from(new Set([...p, name])).sort())
    } catch (e: any) {
      alert('保存失败：' + (e.message || e))
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
      if (!resp.ok) throw new Error(data.detail || `错误 ${resp.status}`)
      setClasses(data)
      setLoadModal(false)
      setAllocSearch('')
    } catch (e: any) {
      alert('加载失败：' + (e.message || e))
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


  return (
    <div className="mx-auto max-w-5xl p-6 relative">
      {busy && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="rounded-xl bg-white px-6 py-4 shadow text-sm">正在计算，请稍候...</div>
        </div>
      )}
      <h1 className="text-2xl font-semibold">资产大类构建模块</h1>
      <p className="text-sm text-gray-500 mt-1">基于已锁定产品池快照配置大类、类内代理产品与拟合权重。</p>
      {universe ? (
        <div className="mt-4 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900">
          已锁定可投资域：<b>{universe.name}</b> · {universe.summary.eligible_count} 只可用产品 · 研究日期 {universe.research_date}
        </div>
      ) : (
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          <span>{universeError || '尚未锁定产品池版本，不能从全市场直接加入产品。'}</span>
          <Link to="/pre-investment/product-pool" className="rounded bg-amber-800 px-3 py-2 font-medium text-white">选择产品池版本</Link>
        </div>
      )}

      <div className="mt-5 rounded-2xl border border-gray-200 bg-white p-4">
        <div className="flex justify-between items-center">
          <SectionTitle title="构建资产大类" />
          <button 
            className="rounded-md bg-gray-100 px-3 py-1 text-xs text-gray-700 hover:bg-gray-200"
            onClick={() => setLoadModal(true)} >
              导入大类配置
          </button>
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
              onModeChange={(mode) => updateClass(ac.id, (c) => ({ ...c, mode }))}
              onRiskMetricChange={(metric) => updateClass(ac.id, (c) => ({ ...c, riskMetric: metric }))}
              on删除={() => deleteAssetClass(ac.id)}
              onAddETF={() => universe ? setSearchOpen({ open: true, classId: ac.id }) : navigate('/pre-investment/product-pool')}
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

          <button className="w-full rounded-xl border border-dashed border-gray-300 py-3 text-sm bg-blue-50/50 hover:bg-blue-50" onClick={addAssetClass}>
            + 添加新大类
          </button>
        </div>
      </div>

      <div className="mt-6 flex flex-wrap justify-center gap-3 text-center">
          <button 
            className="rounded-lg bg-blue-600 px-6 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-700"
            onClick={() => setSaveModal(true)} >
              保存当前大类配置
          </button>
          <button
            className="rounded-lg border border-emerald-700 bg-white px-6 py-2 text-sm font-semibold text-emerald-800 hover:bg-emerald-50 disabled:cursor-not-allowed disabled:opacity-50"
            onClick={enterPortfolioResearch}
            disabled={!universe || !classes.some((assetClass) => assetClass.etfs.length > 0)}
          >
            保存为研究组合 / 进入组合指标
          </button>
      </div>

      {/* Save Modal */}
      {saveModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-md rounded-2xl bg-white p-5 shadow-xl">
            <h3 className="text-lg font-semibold">保存大类配置</h3>
            <input
              autoFocus
              value={allocName}
              onChange={(e) => setAllocName(e.target.value)}
              placeholder="请输入配置名称..."
              className="mt-3 w-full rounded-lg border border-gray-300 px-3 py-2 outline-none focus:ring-2 focus:ring-blue-500"
            />
            <div className="mt-4 flex justify-end gap-3">
              <button className="rounded-lg bg-gray-100 px-4 py-2 text-sm hover:bg-gray-200" onClick={() => setSaveModal(false)}>取消</button>
              <button className="rounded-lg bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700" onClick={handleSave}>保存</button>
            </div>
          </div>
        </div>
      )}

      {/* Load Modal */}
      {loadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-xl rounded-2xl bg-white p-5 shadow-xl">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">导入大类配置</h3>
              <button className="text-gray-500" onClick={() => setLoadModal(false)}>✕</button>
            </div>
            <input
              autoFocus
              value={allocSearch}
              onChange={(e) => setAllocSearch(e.target.value)}
              placeholder="按名称搜索..."
              className="mt-3 w-full rounded-lg border border-gray-300 px-3 py-2 outline-none focus:ring-2 focus:ring-blue-500"
            />
            <div className="mt-3 max-h-80 overflow-auto rounded-lg border border-gray-100">
              {allocList.filter(name => name.toLowerCase().includes(allocSearch.toLowerCase())).map(name => (
                <button
                  key={name}
                  onClick={() => handleLoad(name)}
                  className="flex w-full items-center justify-between border-b px-4 py-2 text-left hover:bg-gray-50"
                >
                  <span className="text-sm text-gray-800">{name}</span>
                  <span className="text-xs text-gray-400">导入</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {searchOpen.open && universe && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-xl rounded-2xl bg-white p-5 shadow-xl">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">从可投资域添加产品</h3>
              <button className="text-gray-500" onClick={() => setSearchOpen({ open: false })}>✕</button>
            </div>
            <input
              autoFocus
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="按代码、名称或评价方案搜索..."
              className="mt-3 w-full rounded-lg border border-gray-300 px-3 py-2 outline-none focus:ring-2 focus:ring-blue-500"
            />
            <div className="mt-3 max-h-80 overflow-auto rounded-lg border border-gray-100">
              {searchResults.length === 0 && <div className="p-4 text-sm text-gray-500">无匹配结果</div>}
              {searchResults.map((etf) => (
                <button
                  key={etf.code + etf.name}
                  onClick={() => {
                    if (searchOpen.classId) addETFToClass(searchOpen.classId, etf)
                    setSearchQuery('')
                    setSearchOpen({ open: false })
                  }}
                  className="flex w-full items-center justify-between border-b px-4 py-2 text-left hover:bg-gray-50"
                >
                  <div className="flex-1 flex items-center gap-2">
                    <span className="font-mono text-sm">{etf.code}</span>
                    <span className="truncate px-2 text-sm text-gray-700">{etf.name}</span>
                    {etf.instrument_type && (
                      <span className="rounded bg-blue-50 px-2 py-0.5 text-[10px] font-semibold text-blue-600">
                        {etf.instrument_type === 'etf' ? 'ETF' : '公募基金'}
                      </span>
                    )}
                  </div>
                  <div className="hidden md:block max-w-56 text-right text-xs text-gray-500 mr-3">
                    <div className="truncate">评价方案：{etf.evaluation_plan_names?.join('、') || '—'}</div>
                    <div className="truncate">产品池：{etf.pool_names?.join('、') || '—'}</div>
                  </div>
                  <span className="text-xs text-gray-400">添加</span>
                </button>
              ))}
            </div>
            {/* 分页与排序控制 */}
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <label className="text-xs text-gray-600">排序</label>
                <select className="border rounded px-2 py-1 text-xs" value={sortBy} onChange={(e) => { setSortBy(e.target.value as any); setPage(1) }}>
                  <option value="name">名称</option>
                  <option value="code">代码</option>
                </select>
                <select className="border rounded px-2 py-1 text-xs" value={sortDir} onChange={(e) => { setSortDir(e.target.value as any); setPage(1) }}>
                  <option value="asc">升序</option>
                  <option value="desc">降序</option>
                </select>
              </div>
              <div className="flex items-center gap-2">
                <label className="text-xs text-gray-600">每页</label>
                <select className="border rounded px-2 py-1 text-xs" value={pageSize} onChange={(e) => { setPageSize(Number(e.target.value)); setPage(1) }}>
                  {[5,10,20,50].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
                <span className="text-xs text-gray-600">共 {total} 条</span>
                <button className="border rounded px-2 py-1 text-xs" disabled={page<=1} onClick={() => setPage(p=>Math.max(1,p-1))}>上一页</button>
                <span className="text-xs">第 {page} 页</span>
                <button className="border rounded px-2 py-1 text-xs" disabled={page*pageSize>=total} onClick={() => setPage(p=>p+1)}>下一页</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 拟合区域 */}
      <div className="mt-6 rounded-2xl border border-gray-200 bg-white p-4">
        <SectionTitle title="大类收益率拟合" />
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 text-sm">
            <label className="text-gray-700">选择开始日期</label>
            <input type="date" className="border rounded px-2 py-1" value={startDate} onChange={(e)=>setStartDate(e.target.value)} />
          </div>
          <button className="rounded-md bg-blue-600 px-3 py-1 text-xs text-white hover:bg-blue-700" onClick={onFit} disabled={busy}>
            拟合大类收益率
          </button>
        </div>
        {fitResult && (
          <div className="mt-4 space-y-6">
            <div>
              {(() => {
                const keys = Object.keys(fitResult.navs)
                return (
                  <ReactECharts style={{ height: 360 }} option={{
                    title: { text: '虚拟净值走势（起始=1）', left: 0, top: 0, textStyle: { fontSize: 13, fontWeight: 600 } },
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
              <h3 className="text-sm font-semibold mb-2">相关系数矩阵</h3>
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
            <div>
              <h3 className="text-sm font-semibold mb-2">横向指标对比</h3>
              <HorizontalMetricComparison
                columns={metricColumns}
                rows={metricRows}
                height={DEFAULT_METRIC_TABLE_HEIGHT}
              />
              <div className="mt-4">
                <h4 className="text-sm font-semibold mb-2 text-gray-700">收益风险象限图</h4>
                <PerformanceQuadrantChart
                  columns={metricColumns}
                  rows={metricRows}
                  defaultXAxis="年化波动率(%)"
                  defaultYAxis="累计收益率(%)"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 同类资产一致性 */}
      {fitResult && Array.isArray(fitResult.consistency) && fitResult.consistency.length > 0 && (
        <div className="mt-4 rounded-2xl border border-gray-200 bg-white p-4">
          <SectionTitle title="同类资产一致性" />
          <div className="overflow-auto">
            <table className="text-xs border" style={{ width: '100%', tableLayout: 'fixed' }}>
              <thead>
                <tr>
                  <th className="border px-2 py-2">大类</th>
                  <th className="border px-2 py-2">相关性均值</th>
                  <th className="border px-2 py-2">主成分解释度(%)</th>
                  <th className="border px-2 py-2">最大跟踪误差(%)</th>
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
      )}

      {/* 滚动相关性研究 */}
      <div className="mt-6 rounded-2xl border border-gray-200 bg-white p-4">
        <SectionTitle title="滚动相关性研究" />
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 text-sm">
            <label className="text-gray-700">滚动天数</label>
            <input type="number" min={5} step={1} className="border rounded px-2 py-1 w-24 text-right" value={rollWindow} onChange={(e)=> setRollWindow(Number(e.target.value)||60)} />
          </div>
          <div className="flex items-center gap-2 text-sm">
            <label className="text-gray-700">研究对象</label>
            <select className="border rounded px-2 py-1" value={rollTargetClass} onChange={(e)=> setRollTargetClass(e.target.value)}>
              <option value="">请选择大类</option>
              {classOptions.map(n=> <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
          <button className="rounded-md bg-blue-600 px-3 py-1 text-xs text-white hover:bg-blue-700" onClick={onRoll} disabled={busy}>
            计算滚动相关性
          </button>
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
                    <th className="border px-2 py-2">大类</th>
                    <th className="border px-2 py-2">整体相关系数</th>
                    <th className="border px-2 py-2">均值</th>
                    <th className="border px-2 py-2">中位数</th>
                    <th className="border px-2 py-2">标准差</th>
                    <th className="border px-2 py-2">偏度</th>
                    <th className="border px-2 py-2">峰度</th>
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
      </div>
    </div>
  )
}

function SectionTitle({ title }: { title: string }) {
  return (
    <div className="mb-3 flex items-center gap-2">
      <span className="text-blue-600">◆</span>
      <h2 className="text-lg font-semibold">{title}</h2>
    </div>
  )
}

function ModePill({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`rounded-md px-2 py-1 text-xs ${active ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
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
  onRemoveETF: (idx: number) => void
  onSetCustomWeight: (idx: number, val: number) => void
  onSetRiskContribution: (idx: number, val: number) => void
  onSetMaxLeverage: (val: number) => void
  onSolve: () => void
  onCompare: () => void
  returnState: ReturnNavigationState
  loading: boolean
}) {
  const [editing, setEditing] = useState(false)
  const [tempName, setTempName] = useState(ac.name)
  useEffect(() => setTempName(ac.name), [ac.name])

  const isRisk = ac.mode === 'risk'
  const showSolved = isRisk && ac.etfs.some((e) => e.solved)

  return (
    <div className="rounded-xl border border-gray-200">
      <div className="flex flex-col gap-3 border-b bg-gray-50/80 px-3 py-2 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          {editing ? (
            <input
              value={tempName}
              onChange={(e) => setTempName(e.target.value)}
              onBlur={() => {
                on重命名(tempName.trim() || ac.name)
                setEditing(false)
              }}
              className="rounded-md border border-gray-300 px-2 py-1 text-sm"
            />
          ) : (
            <div className="text-sm font-medium">{ac.name}</div>
          )}
          <button className="text-xs text-blue-600 underline" onClick={() => setEditing((v) => !v)} title="重命名">
            {editing ? '保存' : '重命名'}
          </button>

          <div className="flex flex-wrap items-center gap-2 lg:ml-3">
            <ModePill label="自定义权重" active={ac.mode === 'custom'} onClick={() => onModeChange('custom')} />
            <ModePill label="等权重" active={ac.mode === 'equal'} onClick={() => onModeChange('equal')} />
            <ModePill label="风险平价" active={ac.mode === 'risk'} onClick={() => onModeChange('risk')} />

            {isRisk && (
              <>
                <select
                  className="ml-2 rounded-md border border-gray-300 px-2 py-1 text-xs"
                  value={ac.riskMetric || 'vol'}
                  onChange={(e) => onRiskMetricChange(e.target.value as RiskMetric)}
                >
                  <option value="vol">波动率</option>
                  <option value="var">VaR</option>
                  <option value="es">ES</option>
                </select>

                <div className="ml-2 flex items-center gap-2 text-xs">
                  <span className="text-gray-600">最大杠杆</span>
                  <input
                    type="number"
                    min={0}
                    step={0.01}
                    value={ac.maxLeverage ?? 0}
                    onChange={(e) => onSetMaxLeverage(Number(e.target.value))}
                    className="w-20 rounded-md border border-gray-300 px-2 py-1 text-right"
                    title="允许的组合最大杠杆率，例如 0 表示不允许杠杆；2 表示最多 2x"
                  />
                </div>

                <button
                  className="ml-2 rounded-md bg-blue-600 px-3 py-1 text-xs text-white hover:bg-blue-700 disabled:opacity-60"
                  onClick={onSolve}
                  disabled={loading || riskControl?.within_tolerance !== true}
                  title={riskControl?.within_tolerance !== true ? '风险贡献合计需经 NJIT 校验等于 100% 才能计算' : ''}
                >
                  {loading ? '计算中...' : '反推资金权重'}
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
            title={ac.etfs.length < 2 ? '至少添加两个产品后才能进行对比' : `对比“${ac.name}”下的 ${ac.etfs.length} 个产品`}
            className="rounded-md border border-blue-200 bg-white px-3 py-1 text-xs font-medium text-blue-700 hover:bg-blue-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 disabled:cursor-not-allowed disabled:border-gray-200 disabled:text-gray-400"
          >
            产品对比{ac.etfs.length >= 2 ? `（${ac.etfs.length}）` : ''}
          </button>
          <button className="rounded-md border border-red-200 px-3 py-1 text-xs text-red-600 hover:bg-red-50" onClick={on删除}>
            删除这个大类
          </button>
        </div>
      </div>

      <div className="grid grid-cols-12 items-center gap-2 px-3 py-2 text-xs text-gray-500">
        <div className="col-span-7">产品（ETF / 公募基金）</div>
        <div className="col-span-3 text-right">{isRisk ? (showSolved ? '风险贡献（%） / 资金权重（%）' : '风险贡献（%）') : '权重（%）'}</div>
        <div className="col-span-2 text-right">操作</div>
      </div>

      <div className="divide-y">
        {ac.etfs.map((e, idx) => (
          <div key={idx} className="grid grid-cols-12 items-center gap-2 px-3 py-2">
            <div className="col-span-7">
              <div className="flex items-center gap-2">
                <span className="w-5 text-xs text-gray-500">{idx + 1}.</span>
                <div className="min-w-0 flex-1 truncate text-sm">
                  <div className="flex items-center gap-2">
                    <span className="font-mono">{e.code}</span>
                    <Link
                      to={`/product-research/products/${encodeURIComponent(e.code)}?kind=${productKind(e)}`}
                      state={returnState}
                      className="truncate font-medium text-blue-700 hover:text-blue-600 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                      title={`查看${e.name}的产品研究`}
                    >
                      {e.name}
                    </Link>
                  </div>
                  <div className="text-xs text-gray-500 mt-0.5">基金公司：{e.management || '—'} ｜ 成立日期：{e.found_date || '—'}</div>
                </div>
              </div>
            </div>

            <div className="col-span-3 text-right">
              {ac.mode === 'custom' ? (
                <div className="inline-flex items-center gap-2">
                  <input
                    type="number"
                    min={0}
                    max={100}
                    step={0.01}
                    value={e.weight ?? 0}
                    onChange={(ev) => onSetCustomWeight(idx, Number(ev.target.value))}
                    className="w-24 rounded-md border border-gray-300 px-2 py-1 text-right text-sm"
                  />
                  <span className="text-sm text-gray-500">%</span>
                </div>
              ) : ac.mode === 'equal' ? (
                <div className="pr-2 text-sm text-gray-700">{equalWeights[idx] == null ? '计算中' : `${equalWeights[idx].toFixed(2)}%`}</div>
              ) : (
                <div className="inline-flex items-center gap-3 justify-end">
                  <input
                    type="number"
                    min={0}
                    max={100}
                    step={0.01}
                    value={e.riskContribution ?? 0}
                    onChange={(ev) => onSetRiskContribution(idx, Number(ev.target.value))}
                    className="w-24 rounded-md border border-gray-300 px-2 py-1 text-right text-sm"
                  />
                  <span className="text-sm text-gray-500">%</span>
                  {showSolved && <span className="text-xs text-gray-500">/ 资金权重 {(e.weight ?? 0).toFixed(2)}%</span>}
                </div>
              )}
            </div>

            <div className="col-span-2 text-right">
              <button onClick={() => onRemoveETF(idx)} className="rounded-md border border-gray-300 px-3 py-1 text-xs hover:bg-gray-50">
                删除
              </button>
            </div>
          </div>
        ))}

        <div className="px-3 py-2">
          <button onClick={onAddETF} className="w-full rounded-lg border border-dashed border-gray-300 py-2 text-sm hover:bg-gray-50">
            + 添加新的产品
          </button>
        </div>
      </div>

      <div className="border-t px-3 py-2 text-xs">
        {ac.mode === 'custom' ? (
          <>
            <span
              className={
                'rounded-md px-2 py-0.5 ' + (weightControl?.within_tolerance ? 'bg-green-50 text-green-700' : 'bg-yellow-50 text-yellow-700')
              }
            >
              权重合计： {weightControl ? `${weightControl.total.toFixed(2)}%` : controlError || 'NJIT 校验中…'}
            </span>
            {weightControl && !weightControl.within_tolerance && <span className="ml-2 text-yellow-700">（需等于 100%）</span>}
          </>
        ) : ac.mode === 'equal' ? (
          <span className={`rounded-md px-2 py-0.5 ${weightControl?.within_tolerance ? 'bg-green-50 text-green-700' : 'bg-yellow-50 text-yellow-700'}`}>权重合计： {weightControl ? `${weightControl.total.toFixed(2)}%` : controlError || 'NJIT 校验中…'}</span>
        ) : (
          <>
            <span
              className={
                'rounded-md px-2 py-0.5 ' + (riskControl?.within_tolerance ? 'bg-green-50 text-green-700' : 'bg-yellow-50 text-yellow-700')
              }
            >
              风险贡献合计： {riskControl ? `${riskControl.total.toFixed(2)}%` : controlError || 'NJIT 校验中…'}
            </span>
            {riskControl && !riskControl.within_tolerance && <span className="ml-2 text-yellow-700">（需等于 100%）</span>}
            {showSolved && (
              <span className="ml-3 rounded-md bg-blue-50 px-2 py-0.5 text-blue-700">资金权重合计： {weightControl ? `${weightControl.total.toFixed(2)}%` : controlError || 'NJIT 校验中…'}</span>
            )}
          </>
        )}
      </div>
    </div>
  )
}
