import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import ClassFitPanel, { ClassConsistencyTable, type ClassFitResult } from '../components/ClassFitPanel'
import { assertFixedNjitExecution, assertFixedNjitExecutionLanes, type FixedNjitExecutionAudit } from '../utils/fixedNjitExecution'
import {
  getInvestableUniverse,
  investableUniverseEligibleCount,
  searchInvestableUniverseProducts,
  type InvestableUniverseSnapshot,
} from '../services/productPools'

interface PoolItem {
  code: string
  name: string
  instrument_type: 'etf' | 'fund'
  evaluation_plan_names?: string[]
  pool_names?: string[]
  max_weight?: number | null
}

interface AutoClassMember {
  code: string
  name: string
  weight: number
  instrument_type: string
  fund_type: string
  invest_type: string
  management: string
  contract_label: string
  // Absent on drafts produced before the contract taxonomy shipped.
  taxonomy?: { asset_class: string; category: string; detail: string; path: string; matched: string }
  affinity: number | null
  is_medoid: boolean
  max_weight: number | null
  capped: boolean
}

interface AutoClassGroup {
  id: string
  name: string
  size: number
  medoid: string
  silhouette: number | null
  mean_corr: number | null
  max_class_weight: number | null
  etfs: AutoClassMember[]
}

interface AutoClassResult {
  algorithm: string
  features: string
  taxonomy_level: string
  block_by: string
  k: number
  weight_mode: string
  observations: number
  start_date: string
  end_date: string
  classes: AutoClassGroup[]
  unassigned: { code: string; name: string; reason: string; detail: string }[]
  skipped: { code: string; name: string; reason: string; detail: string }[]
  warnings: string[]
  diagnostics: {
    silhouette: number | null
    cross_class_corr: Array<Array<number | null>>
    cross_class_labels: string[]
    significant_eigenvalues: number
    k_suggestions: { k: number; silhouette: number | null }[]
    blocks: { block: string; size: number; k: number; silhouette: number | null }[]
    contract_deviations: { code: string; name: string; assigned_class: string; contract_label: string }[]
    winsorized: { code: string; name: string; clipped: number; max_raw_return: number | null }[]
  }
  execution: FixedNjitExecutionAudit
  universe_snapshot: {
    id: string
    name: string
    research_date: string
    content_hash: string
    version_refs: unknown[]
    immutable: true
  }
}

interface MetaOption { id: string; label: string }
interface AutoClassMeta {
  algorithms: MetaOption[]
  features: MetaOption[]
  linkages: MetaOption[]
  weight_modes: MetaOption[]
  taxonomy_levels: MetaOption[]
  block_modes: MetaOption[]
  limits: { min_observations: number; max_auto_k: number; min_products: number }
}

const ALGORITHM_HINTS: Record<string, string> = {
  rule: '按基金合同类型、投资类型、业绩基准与跟踪指数做确定性映射，直接输出所选层级的合同分类。大类个数由标签自然决定，是其它算法的对照基线。',
  hierarchical: '在收益相关性距离上做凝聚层次聚类。默认选项：无需初值、结果确定，且“为什么这两个产品在一起”可以回溯到合并树。',
  kmedoids: '以真实产品作为类中心，每个大类天然得到一只代表产品，适合直接拿来做大类代理。',
  kmeans: '在标准化特征空间上做质心聚类。速度快，但类中心是虚拟点，对特征量纲更敏感。',
}
const FEATURE_HINTS: Record<string, string> = {
  correlation: '用日收益相关性距离 √(0.5(1-ρ))。大类资产的本质是同涨同跌，这是最贴合的度量。',
  metrics: '用收益、波动、回撤、夏普、卡玛、折溢价与成交额画像做欧氏距离。',
  pca: '对相关矩阵做特征分解，用前几个主成分载荷聚类，识别共同的系统性驱动。',
  blend: '风险收益画像与主成分载荷拼接后聚类。',
}

const BLOCK_HINTS: Record<string, string> = {
  none: '只按净值行为聚类。相关性窗口可能把黄金 ETF 和权益 ETF 放进同一类。',
  asset_class: '权益/固收/商品/货币/海外/混合 之间不可混合，统计聚类只在同一资产大类内部细分。做 SAA 时的推荐口径。',
  category: '在资产大类之下再锁定二级类型（宽基规模/风格因子/行业/主题、利率债/信用债/可转债/同业存单…），聚类只区分同一类型内的差异。',
  detail: '锁定到三级明细（大小盘、红利/低波/价值、八大行业、六大主题、国债/城投债…）。层级越细，块越多、每块可拆的类越少。',
}

const emptyMeta: AutoClassMeta = {
  algorithms: [],
  features: [],
  linkages: [],
  weight_modes: [],
  taxonomy_levels: [],
  block_modes: [],
  limits: { min_observations: 60, max_auto_k: 8, min_products: 2 },
}

function SectionTitle({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="mb-3">
      <div className="flex items-center gap-2">
        <span className="text-violet-600">◆</span>
        <h2 className="text-lg font-semibold">{title}</h2>
      </div>
      {hint && <p className="mt-1 pl-6 text-xs text-gray-500">{hint}</p>}
    </div>
  )
}

function formatNumber(value: number | null | undefined, digits = 3) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—'
}

export default function AutoAssetClassification() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const universeId = searchParams.get('universe') ?? ''
  const [universe, setUniverse] = useState<InvestableUniverseSnapshot | null>(null)
  const [universeLoading, setUniverseLoading] = useState(false)
  const [universeError, setUniverseError] = useState('')
  const [productsError, setProductsError] = useState('')
  const [meta, setMeta] = useState<AutoClassMeta>(emptyMeta)
  const [pool, setPool] = useState<PoolItem[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<PoolItem[]>([])
  const [searchTotal, setSearchTotal] = useState(0)

  const [algorithm, setAlgorithm] = useState('hierarchical')
  const [features, setFeatures] = useState('correlation')
  const [linkage, setLinkage] = useState('average')
  const [autoK, setAutoK] = useState(false)
  const [k, setK] = useState(5)
  const [sizeMin, setSizeMin] = useState(2)
  const [sizeMax, setSizeMax] = useState(4)
  const [unassignedPolicy, setUnassignedPolicy] = useState<'park' | 'force'>('park')
  const [weightMode, setWeightMode] = useState('inv_vol')
  const [taxonomyLevel, setTaxonomyLevel] = useState('asset_class')
  const [blockBy, setBlockBy] = useState('none')
  const [startDate, setStartDate] = useState('2020-01-01')

  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<AutoClassResult | null>(null)
  const [fitResult, setFitResult] = useState<ClassFitResult | null>(null)
  const [fitError, setFitError] = useState('')
  const [saveName, setSaveName] = useState('')
  const [saveMessage, setSaveMessage] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    fetch('/api/asset-classes/auto/meta', { signal: controller.signal })
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error(String(response.status)))))
      // Merge instead of replace: a meta payload from an older backend that has
      // no taxonomy options must not leave the option lists undefined.
      .then((payload: Partial<AutoClassMeta>) => setMeta({ ...emptyMeta, ...payload }))
      .catch(() => undefined)
    return () => controller.abort()
  }, [])

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
        if (!active) return
        setUniverse(null)
        setUniverseError(caught instanceof Error ? caught.message : '无法加载可投资域快照。')
      })
      .finally(() => { if (active) setUniverseLoading(false) })
    return () => { active = false }
  }, [universeId])

  useEffect(() => {
    const controller = new AbortController()
    if (!universeId || !universe) {
      setSearchResults([])
      setSearchTotal(0)
      return () => controller.abort()
    }
    setProductsError('')
    searchInvestableUniverseProducts(universeId, {
      query: searchQuery,
      eligibleOnly: true,
      page: 1,
      pageSize: 100,
      signal: controller.signal,
    })
      .then((response) => {
        setSearchResults(response.items.map((item) => ({
          code: item.product_id,
          name: item.name,
          instrument_type: item.kind,
          evaluation_plan_names: [...new Set(item.evaluation_sources.map((source) => source.evaluation_plan_name))],
          pool_names: [...new Set(item.evaluation_sources.map((source) => source.pool_name))],
          max_weight: item.max_weight,
        })))
        setSearchTotal(response.total)
      })
      .catch((caught) => {
        if ((caught as DOMException)?.name !== 'AbortError') {
          setSearchResults([])
          setSearchTotal(0)
          // `universeError` only renders while the universe is missing, so a
          // failed product fetch used to leave an empty picker and no reason.
          setProductsError(caught instanceof Error ? caught.message : '无法读取可投资域产品。')
        }
      })
    return () => controller.abort()
  }, [searchQuery, universe, universeId])

  const poolCodes = useMemo(() => new Set(pool.map((item) => item.code)), [pool])

  // /api/fit-classes deliberately runs on the raw series, so any class holding a
  // product with an anomalous NAV print shows performance that cannot be trusted.
  const affectedClasses = useMemo(() => {
    if (!result) return []
    const flagged = new Map(result.diagnostics.winsorized.map((item) => [item.code, item]))
    if (flagged.size === 0) return []
    return result.classes
      .map((group) => ({
        className: group.name,
        products: group.etfs.map((member) => flagged.get(member.code)).filter(Boolean) as AutoClassResult['diagnostics']['winsorized'],
      }))
      .filter((item) => item.products.length > 0)
  }, [result])

  function addAllSearchResults() {
    setPool((current) => {
      const known = new Set(current.map((entry) => entry.code))
      return [...current, ...searchResults.filter((item) => !known.has(item.code))]
    })
  }

  function toggleProduct(item: PoolItem, checked: boolean) {
    setPool((current) => checked
      ? (current.some((entry) => entry.code === item.code) ? current : [...current, item])
      : current.filter((entry) => entry.code !== item.code))
  }

  const allVisibleSelected = searchResults.length > 0 && searchResults.every((item) => poolCodes.has(item.code))

  function toggleAllVisible(checked: boolean) {
    if (checked) {
      addAllSearchResults()
      return
    }
    const visible = new Set(searchResults.map((item) => item.code))
    setPool((current) => current.filter((entry) => !visible.has(entry.code)))
  }

  async function runFit(classes: AutoClassGroup[]) {
    setFitError('')
    setFitResult(null)
    const payload = {
      startDate,
      classes: classes.map((group) => ({
        id: group.id,
        name: group.name,
        etfs: group.etfs.map((member) => ({ code: member.code, name: member.name, weight: member.weight })),
      })),
    }
    try {
      const response = await fetch('/api/fit-classes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!response.ok) throw new Error(`后端错误 ${response.status}`)
      const data = await response.json() as ClassFitResult
      assertFixedNjitExecutionLanes(data.execution, '自动大类净值拟合')
      setFitResult(data)
    } catch (reason: any) {
      setFitError(`大类净值与指标计算失败：${reason?.message || reason}`)
    }
  }

  async function onRun() {
    setError('')
    setSaveMessage('')
    if (!universe) {
      setError('请先选择并锁定产品池版本。')
      return
    }
    if (pool.length < meta.limits.min_products) {
      setError(`请至少选择 ${meta.limits.min_products} 个产品`)
      return
    }
    if (sizeMax < sizeMin) {
      setError('每类最多产品数不能小于最少产品数')
      return
    }
    setRunning(true)
    setResult(null)
    setFitResult(null)
    try {
      const response = await fetch('/api/asset-classes/auto/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          universe_snapshot_id: universe.id,
          products: pool.map((item) => ({ code: item.code, name: item.name, kind: item.instrument_type })),
          startDate,
          algorithm,
          features,
          linkage,
          k: algorithm === 'rule' || autoK ? null : k,
          sizeMin,
          sizeMax,
          unassignedPolicy,
          weightMode,
          taxonomyLevel,
          blockBy,
        }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data?.detail || `后端错误 ${response.status}`)
      assertFixedNjitExecution(data.execution, '自动构建大类')
      setResult(data as AutoClassResult)
      await runFit((data as AutoClassResult).classes)
    } catch (reason: any) {
      setError(reason?.message || String(reason))
    } finally {
      setRunning(false)
    }
  }

  async function onSave() {
    if (!result) return
    const name = saveName.trim()
    if (!name) {
      setSaveMessage('配置名称不能为空')
      return
    }
    try {
      const response = await fetch('/api/save-allocation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          asset_alloc_name: name,
          universe_snapshot_id: universe?.id ?? null,
          classes: result.classes.map((group) => ({
            id: group.id,
            name: group.name,
            etfs: group.etfs.map((member) => ({ code: member.code, name: member.name, weight: member.weight })),
          })),
        }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data?.detail || `错误 ${response.status}`)
      setSaveMessage(`配置「${name}」已保存，可在大类资产配置与回测中引用`)
    } catch (reason: any) {
      setSaveMessage(`保存失败：${reason?.message || reason}`)
    }
  }

  function openInManualWorkspace() {
    if (!result) return
    sessionStorage.setItem('autoClassificationImport', JSON.stringify({
      universe_snapshot_id: universe?.id ?? null,
      classes: result.classes.map((group) => ({
        id: group.id,
        name: group.name,
        mode: 'custom',
        riskMetric: 'vol',
        maxLeverage: 0,
        etfs: group.etfs.map((member) => ({
          code: member.code,
          name: member.name,
          weight: member.weight,
          instrument_type: member.instrument_type === 'fund' ? 'fund' : 'etf',
        })),
      })),
    }))
    navigate(universe ? `/pre-investment/saa/asset-classes?universe=${encodeURIComponent(universe.id)}` : '/pre-investment/product-pool')
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      <header className="mb-6">
        <p className="text-xs font-semibold uppercase tracking-wide text-violet-700">Pre-investment · SAA</p>
        <h1 className="mt-1 text-2xl font-bold text-gray-900">自动构建大类</h1>
        <p className="mt-2 max-w-4xl text-sm text-gray-600">
          在已锁定可投资域内，按合同标签、收益相关性和风险特征自动划分资产大类；结果可保存或进入手动构建继续调整。
        </p>
      </header>

      {universe ? (
        <div className="mb-4 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900">
          已锁定可投资域：<b>{universe.name}</b> · {investableUniverseEligibleCount(universe)} 只可用产品 · 研究日期 {universe.research_date}
        </div>
      ) : (
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          <span>{universeLoading ? '正在读取可投资域…' : universeError || '尚未锁定产品池版本，不能运行自动分类。'}</span>
          <Link to="/pre-investment/product-pool" className="rounded bg-amber-800 px-3 py-2 font-medium text-white">选择产品池版本</Link>
        </div>
      )}

      {/* 1. 产品池 */}
      <section className="rounded-2xl border border-gray-200 bg-white p-4">
        <SectionTitle title="① 从可投资域选择产品" hint="只展示所选产品池版本中当前可用于新增配置的产品。" />
        <div className="grid gap-4 lg:grid-cols-2">
          <div>
            <div className="flex items-center gap-2">
              <input
                className="w-full rounded border px-2 py-1 text-sm"
                placeholder="按代码、名称或评价方案搜索"
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                aria-label="可投资域产品搜索"
              />
            </div>
            <div className="mt-2 flex items-center justify-between gap-2">
              <label className="flex items-center gap-2 text-xs font-medium text-gray-700">
                <input
                  type="checkbox"
                  className="h-3.5 w-3.5"
                  disabled={searchResults.length === 0}
                  checked={allVisibleSelected}
                  onChange={(event) => toggleAllVisible(event.target.checked)}
                  aria-label="全选当前结果"
                />
                全选当前结果
              </label>
              <span className="text-xs text-gray-500">匹配 {searchTotal} 个产品，显示前 {searchResults.length} 个</span>
            </div>
            {productsError && (
              <p className="mt-2 rounded border border-red-300 bg-red-50 px-2 py-1 text-xs text-red-700">
                读取可投资域产品失败：{productsError}
              </p>
            )}
            <ul className="mt-2 max-h-64 divide-y overflow-auto rounded border">
              {searchResults.map((item) => (
                <li key={item.code}>
                  <label className="flex cursor-pointer items-center gap-2 px-2 py-1 text-xs hover:bg-violet-50">
                    <input
                      type="checkbox"
                      className="h-3.5 w-3.5 shrink-0"
                      checked={poolCodes.has(item.code)}
                      onChange={(event) => toggleProduct(item, event.target.checked)}
                      aria-label={`选择 ${item.name}`}
                    />
                    <span className="min-w-0 flex-1 truncate">
                      <span className="font-mono text-gray-500">{item.code}</span> {item.name}
                      {typeof item.max_weight === 'number' && (
                        <span className="ml-1 rounded bg-amber-100 px-1 py-0.5 text-[10px] font-semibold text-amber-800">
                          限额 {(item.max_weight * 100).toFixed(1)}%
                        </span>
                      )}
                      {item.pool_names && item.pool_names.length > 0 && (
                        <span className="ml-1 text-[10px] text-gray-400">{item.pool_names.join('/')}</span>
                      )}
                    </span>
                  </label>
                </li>
              ))}
              {searchResults.length === 0 && !productsError && (
                <li className="px-2 py-3 text-xs text-gray-400">没有匹配的产品</li>
              )}
            </ul>
            <p className="mt-3 rounded bg-slate-50 px-3 py-2 text-xs text-slate-600">不允许粘贴任意代码，避免绕过产品池准入与使用限制。</p>
          </div>
          <div>
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold">已选产品（{pool.length}）</h3>
              <button className="rounded border px-2 py-0.5 text-xs hover:bg-gray-50" onClick={() => setPool([])}>清空</button>
            </div>
            <ul className="mt-2 max-h-80 divide-y overflow-auto rounded border">
              {pool.map((item) => (
                <li key={item.code} className="flex items-center justify-between gap-2 px-2 py-1 text-xs">
                  <span className="min-w-0 truncate">
                    <span className="font-mono text-gray-500">{item.code}</span> {item.name || '（按代码解析）'}
                    {typeof item.max_weight === 'number' && (
                      <span className="ml-1 rounded bg-amber-100 px-1 py-0.5 text-[10px] font-semibold text-amber-800">
                        限额 {(item.max_weight * 100).toFixed(1)}%
                      </span>
                    )}
                    {item.pool_names && item.pool_names.length > 0 && (
                      <span className="ml-1 text-[10px] text-gray-400">{item.pool_names.join('/')}</span>
                    )}
                  </span>
                  <button className="rounded border px-2 py-0.5 text-xs" onClick={() => setPool((current) => current.filter((entry) => entry.code !== item.code))}>移除</button>
                </li>
              ))}
              {pool.length === 0 && <li className="px-2 py-3 text-xs text-gray-400">尚未选择产品</li>}
            </ul>
          </div>
        </div>
      </section>

      {/* 2. 参数 */}
      <section className="mt-4 rounded-2xl border border-gray-200 bg-white p-4">
        <SectionTitle title="② 分类参数" hint="算法只负责给出产品与大类的亲和度；“分几类、每类几个”由统一的容量约束选择器处理。" />
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <label className="text-sm">
            <span className="mb-1 block font-medium text-gray-700">算法</span>
            <select className="w-full rounded border px-2 py-1 text-sm" value={algorithm} onChange={(event) => setAlgorithm(event.target.value)}>
              {meta.algorithms.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
            </select>
          </label>
          <label className="text-sm">
            <span className="mb-1 block font-medium text-gray-700">特征集</span>
            <select className="w-full rounded border px-2 py-1 text-sm" value={features} onChange={(event) => setFeatures(event.target.value)}>
              {meta.features.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
            </select>
          </label>
          <label className="text-sm">
            <span className="mb-1 block font-medium text-gray-700">连接方式（层次聚类）</span>
            <select className="w-full rounded border px-2 py-1 text-sm" value={linkage} disabled={algorithm !== 'hierarchical'} onChange={(event) => setLinkage(event.target.value)}>
              {meta.linkages.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
            </select>
          </label>
          <label className="text-sm">
            <span className="mb-1 block font-medium text-gray-700">合同分类层级</span>
            <select className="w-full rounded border px-2 py-1 text-sm" value={taxonomyLevel} onChange={(event) => setTaxonomyLevel(event.target.value)}>
              {meta.taxonomy_levels.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
            </select>
          </label>
          <label className="text-sm">
            <span className="mb-1 block font-medium text-gray-700">按合同分层（硬约束）</span>
            <select className="w-full rounded border px-2 py-1 text-sm" value={blockBy} disabled={algorithm === 'rule'} onChange={(event) => setBlockBy(event.target.value)}>
              {meta.block_modes.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
            </select>
          </label>
          <label className="text-sm">
            <span className="mb-1 block font-medium text-gray-700">类内权重</span>
            <select className="w-full rounded border px-2 py-1 text-sm" value={weightMode} onChange={(event) => setWeightMode(event.target.value)}>
              {meta.weight_modes.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
            </select>
          </label>
          <div className="text-sm">
            <span className="mb-1 block font-medium text-gray-700">大类个数 K</span>
            <div className="flex items-center gap-2">
              <input
                type="number"
                min={2}
                className="w-24 rounded border px-2 py-1 text-sm disabled:bg-gray-100"
                value={k}
                disabled={autoK || algorithm === 'rule'}
                onChange={(event) => setK(Math.max(2, Number(event.target.value) || 2))}
                aria-label="大类个数"
              />
              <label className="flex items-center gap-1 text-xs text-gray-600">
                <input
                  type="checkbox"
                  checked={autoK}
                  disabled={algorithm === 'rule'}
                  onChange={(event) => setAutoK(event.target.checked)}
                  aria-label="自动建议大类个数"
                />
                自动建议
              </label>
            </div>
            {algorithm === 'rule' && <p className="mt-1 text-xs text-amber-700">规则映射的大类个数由合同标签自然决定</p>}
          </div>
          <label className="text-sm">
            <span className="mb-1 block font-medium text-gray-700">每类最少产品数</span>
            <input type="number" min={1} className="w-full rounded border px-2 py-1 text-sm" value={sizeMin} onChange={(event) => setSizeMin(Math.max(1, Number(event.target.value) || 1))} aria-label="每类最少产品数" />
          </label>
          <label className="text-sm">
            <span className="mb-1 block font-medium text-gray-700">每类最多产品数</span>
            <input type="number" min={1} className="w-full rounded border px-2 py-1 text-sm" value={sizeMax} onChange={(event) => setSizeMax(Math.max(1, Number(event.target.value) || 1))} aria-label="每类最多产品数" />
          </label>
          <label className="text-sm">
            <span className="mb-1 block font-medium text-gray-700">样本开始日期</span>
            <input type="date" className="w-full rounded border px-2 py-1 text-sm" value={startDate} onChange={(event) => setStartDate(event.target.value)} aria-label="样本开始日期" />
          </label>
          <label className="text-sm md:col-span-2">
            <span className="mb-1 block font-medium text-gray-700">装不下的产品</span>
            <select className="w-full rounded border px-2 py-1 text-sm" value={unassignedPolicy} onChange={(event) => setUnassignedPolicy(event.target.value as 'park' | 'force')}>
              <option value="park">进入待观察池（推荐）</option>
              <option value="force">强制归入最相近大类</option>
            </select>
          </label>
        </div>
        <div className="mt-3 rounded-lg bg-violet-50 p-3 text-xs text-violet-900">
          <p><strong>{meta.algorithms.find((item) => item.id === algorithm)?.label ?? algorithm}</strong>：{ALGORITHM_HINTS[algorithm]}</p>
          <p className="mt-1"><strong>{meta.features.find((item) => item.id === features)?.label ?? features}</strong>：{FEATURE_HINTS[features]}</p>
          <p className="mt-1"><strong>分层</strong>：{BLOCK_HINTS[blockBy]}</p>
        </div>
        <div className="mt-3 flex items-center gap-3">
          <button
            className="rounded-md bg-violet-700 px-4 py-1.5 text-sm font-medium text-white hover:bg-violet-600 disabled:opacity-50"
            onClick={onRun}
            disabled={running || !universe}
          >
            {running ? '计算中…' : '运行自动分类'}
          </button>
          {error && <span className="text-sm text-red-600">{error}</span>}
        </div>
      </section>

      {result && (
        <>
          {/* 3. 诊断 */}
          <section className="mt-4 rounded-2xl border border-gray-200 bg-white p-4">
            <SectionTitle title="③ 分类诊断" hint="轮廓系数衡量类内紧密与类间分离，越接近 1 越好；负值说明该产品更像别的大类。" />
            <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
              {[
                { label: '大类个数', value: String(result.k) },
                { label: '整体轮廓系数', value: formatNumber(result.diagnostics.silhouette) },
                { label: '共同样本交易日', value: String(result.observations) },
                { label: '显著特征值数', value: String(result.diagnostics.significant_eigenvalues) },
                { label: '样本区间', value: `${result.start_date} ~ ${result.end_date}` },
              ].map((card) => (
                <div key={card.label} className="rounded-lg border bg-gray-50 p-3">
                  <p className="text-xs text-gray-500">{card.label}</p>
                  <p className="mt-1 text-sm font-semibold text-gray-900">{card.value}</p>
                </div>
              ))}
            </div>
            {result.diagnostics.k_suggestions.length > 0 && (
              <div className="mt-3">
                <h4 className="text-xs font-semibold text-gray-700">K 建议（按轮廓系数）</h4>
                <div className="mt-1 flex flex-wrap gap-2">
                  {result.diagnostics.k_suggestions.map((item) => (
                    <span key={item.k} className={`rounded border px-2 py-0.5 text-xs ${item.k === result.k ? 'border-violet-500 bg-violet-50 font-semibold text-violet-800' : 'text-gray-600'}`}>
                      K={item.k}：{formatNumber(item.silhouette)}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {result.warnings.length > 0 && (
              <ul className="mt-3 space-y-1 rounded-lg bg-amber-50 p-3 text-xs text-amber-900">
                {result.warnings.map((warning) => <li key={warning}>· {warning}</li>)}
              </ul>
            )}
          </section>

          {/* 4. 大类结果 */}
          <section className="mt-4 rounded-2xl border border-gray-200 bg-white p-4">
            <SectionTitle title="④ 大类映射草案" hint="★ 为该大类的代表产品（medoid）；权重按所选类内权重方式归一到 100%。" />
            <div className="grid gap-3 lg:grid-cols-2">
              {result.classes.map((group) => (
                <div key={group.id} className="rounded-xl border border-violet-200 bg-violet-50/40 p-3">
                  <div className="flex items-baseline justify-between">
                    <h3 className="text-sm font-semibold text-violet-900">{group.name}</h3>
                    <span className="text-xs text-gray-600">
                      {group.size} 个产品 · 类内相关 {formatNumber(group.mean_corr, 2)} · 轮廓 {formatNumber(group.silhouette, 2)}
                    </span>
                  </div>
                  <table className="mt-2 w-full text-xs">
                    <thead>
                      <tr className="text-left text-gray-500">
                        <th className="py-1">产品</th>
                        <th className="py-1">合同分类</th>
                        <th className="py-1 text-right">池限额</th>
                        <th className="py-1 text-right">权重(%)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {group.etfs.map((member) => (
                        <tr key={member.code} className="border-t border-violet-100">
                          <td className="py-1">
                            {member.is_medoid && <span className="mr-1 text-amber-500" title="代表产品">★</span>}
                            <span className="font-mono text-gray-500">{member.code}</span> {member.name}
                          </td>
                          <td className="py-1 text-gray-600" title={member.taxonomy?.path ?? member.contract_label}>{member.contract_label}</td>
                          <td className="py-1 text-right text-gray-500">
                            {member.max_weight === null ? '—' : `${(member.max_weight * 100).toFixed(1)}%`}
                          </td>
                          <td className={`py-1 text-right font-medium ${member.capped ? 'text-amber-700' : ''}`}>
                            {member.weight.toFixed(2)}
                            {member.capped && <span className="ml-1 text-[10px]" title="已被产品池限额压低">▼</span>}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {group.max_class_weight !== null && group.max_class_weight < 100 && (
                    <p className="mt-2 rounded bg-amber-100 px-2 py-1 text-[11px] text-amber-900">
                      成员限额合计 {group.max_class_weight.toFixed(1)}%：该大类的 SAA 权重不得超过此值，否则会突破产品池限额。
                    </p>
                  )}
                </div>
              ))}
            </div>

            {result.diagnostics.blocks.length > 0 && (
              <div className="mt-4 rounded-lg border border-violet-200 bg-violet-50 p-3">
                <h4 className="text-xs font-semibold text-violet-900">
                  合同分层（{meta.block_modes.find((item) => item.id === result.block_by)?.label ?? result.block_by}）
                </h4>
                <p className="mt-1 text-xs text-violet-800">
                  每个合同块内部单独聚类，块之间不会合并，也不会互相拉产品。
                </p>
                <table className="mt-2 w-full text-xs">
                  <thead>
                    <tr className="text-left text-violet-700">
                      <th className="py-1">合同块</th>
                      <th className="py-1 text-right">产品数</th>
                      <th className="py-1 text-right">块内大类数</th>
                      <th className="py-1 text-right">块内轮廓系数</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.diagnostics.blocks.map((item) => (
                      <tr key={item.block} className="border-t border-violet-200">
                        <td className="py-1">{item.block}</td>
                        <td className="py-1 text-right">{item.size}</td>
                        <td className="py-1 text-right">{item.k}</td>
                        <td className="py-1 text-right">{item.silhouette === null ? '—' : item.silhouette.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <div className="mt-4 grid gap-4 lg:grid-cols-2">
              <div>
                <h4 className="text-xs font-semibold text-gray-700">行为分类与合同分类的偏离（{result.diagnostics.contract_deviations.length}）</h4>
                <p className="mt-1 text-xs text-gray-500">这些产品按净值行为进入的大类与合同标签不一致，是研究线索而不是错误。</p>
                <ul className="mt-2 max-h-48 divide-y overflow-auto rounded border text-xs">
                  {result.diagnostics.contract_deviations.map((item) => (
                    <li key={item.code} className="px-2 py-1">
                      <span className="font-mono text-gray-500">{item.code}</span> {item.name}
                      <span className="ml-1 text-gray-600">：合同「{item.contract_label}」→ 归入「{item.assigned_class}」</span>
                    </li>
                  ))}
                  {result.diagnostics.contract_deviations.length === 0 && <li className="px-2 py-2 text-gray-400">没有偏离项</li>}
                </ul>
              </div>
              <div>
                <h4 className="text-xs font-semibold text-gray-700">待观察与排除（{result.unassigned.length + result.skipped.length}）</h4>
                <p className="mt-1 text-xs text-gray-500">超出每类上限、相似度不足，或样本期内缺少可用净值的产品。</p>
                <ul className="mt-2 max-h-48 divide-y overflow-auto rounded border text-xs">
                  {[...result.unassigned, ...result.skipped].map((item) => (
                    <li key={`${item.reason}-${item.code}`} className="px-2 py-1">
                      <span className="font-mono text-gray-500">{item.code}</span> {item.name}
                      <span className="ml-1 text-gray-500">（{item.detail}）</span>
                    </li>
                  ))}
                  {result.unassigned.length + result.skipped.length === 0 && <li className="px-2 py-2 text-gray-400">全部产品已归类</li>}
                </ul>
              </div>
            </div>

            {result.diagnostics.winsorized.length > 0 && (
              <div className="mt-4 rounded-lg bg-rose-50 p-3 text-xs text-rose-900">
                <h4 className="font-semibold">复权净值疑似异常（{result.diagnostics.winsorized.length}）</h4>
                <p className="mt-1">下列产品存在超出稳健区间的单日跳变，已在聚类特征中做稳健处理；展示的净值与绩效指标仍使用原始序列。</p>
                <ul className="mt-1">
                  {result.diagnostics.winsorized.map((item) => (
                    <li key={item.code}>
                      · <span className="font-mono">{item.code}</span> {item.name}：{item.clipped} 个观测，最大单日 {item.max_raw_return === null ? '—' : `${(item.max_raw_return * 100).toFixed(1)}%`}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="mt-4 flex flex-wrap items-center gap-3 border-t pt-3">
              <input
                className="rounded border px-2 py-1 text-sm"
                placeholder="保存为大类配置名称"
                value={saveName}
                onChange={(event) => setSaveName(event.target.value)}
                aria-label="大类配置名称"
              />
              <button className="rounded-md bg-violet-700 px-3 py-1 text-sm text-white hover:bg-violet-600" onClick={onSave}>保存为大类配置</button>
              <button className="rounded-md border px-3 py-1 text-sm hover:bg-gray-50" onClick={openInManualWorkspace}>在手动构建大类中打开</button>
              {saveMessage && <span className="text-sm text-gray-700">{saveMessage}</span>}
            </div>
          </section>

          {/* 5. 净值与指标 */}
          <section className="mt-4 rounded-2xl border border-gray-200 bg-white p-4">
            <SectionTitle title="⑤ 大类净值与指标" hint="与手动构建大类使用同一条拟合链路：类内权重合成大类虚拟净值，再计算收益、风险与相关性。" />
            {fitError && <p className="text-sm text-red-600">{fitError}</p>}
            {!fitError && !fitResult && <p className="text-sm text-gray-500">正在计算大类净值…</p>}
            {affectedClasses.length > 0 && (
              <div className="mb-4 rounded-lg border border-rose-300 bg-rose-50 p-3 text-xs text-rose-900">
                <p className="font-semibold">以下大类的净值与指标不可直接采信</p>
                <p className="mt-1">
                  聚类特征已对异常净值做稳健处理，但本节的净值、收益、波动与回撤走的是<b>原始复权净值序列</b>，
                  仍包含下列产品的异常跳变。请先核实数据再解读这些大类的表现。
                </p>
                <ul className="mt-1">
                  {affectedClasses.map((item) => (
                    <li key={item.className}>
                      · <b>{item.className}</b> ← {item.products.map((product) => `${product.name}（${product.code}，单日 ${product.max_raw_return === null ? '—' : `${(product.max_raw_return * 100).toFixed(1)}%`}）`).join('、')}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {fitResult && <ClassFitPanel result={fitResult} />}
          </section>

          {fitResult && Array.isArray(fitResult.consistency) && fitResult.consistency.length > 0 && (
            <section className="mt-4 rounded-2xl border border-gray-200 bg-white p-4">
              <SectionTitle title="⑥ 同类资产一致性" hint="相关性均值 < 0.6、主成分解释度 < 80% 或跟踪误差 > 5% 时标红，说明该大类内部并不同质。" />
              <ClassConsistencyTable rows={fitResult.consistency} />
            </section>
          )}
        </>
      )}
    </div>
  )
}
