import { systemText, useI18n } from '../i18n/runtime'
import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import ClassFitPanel, { ClassConsistencyTable, type ClassFitResult } from '../components/ClassFitPanel'
import { assertNativeNumericalExecution, assertNativeNumericalExecutionLanes, type NativeNumericalExecutionAudit } from '../utils/fixedNjitExecution'
import {
  getInvestableUniverse,
  investableUniverseEligibleCount,
  searchInvestableUniverseProducts,
  type InvestableUniverseSnapshot,
} from '../services/productPools'
import { apiErrorMessage } from '../utils/apiError'
import PitProvenance from '../components/PitProvenance'

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
  pit?: import('../services/pit').PitRunLineage
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
  execution: NativeNumericalExecutionAudit
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




// One place decides what each algorithm ignores; the control, its label, its
// reason line and the hint panel all read from here so they cannot disagree.
function inactiveReason(parameter: 'linkage' | 'blockBy' | 'k', algorithm: string): string {
  if (parameter === 'linkage' && algorithm !== 'hierarchical') {
    return systemText('preInvestment.autoAssetClassification.linkageIsUsedOnlyByHierarchicalClustering')
  }
  if (parameter === 'blockBy' && algorithm === 'rule') {
    return systemText('preInvestment.autoAssetClassification.ruleMappingAlreadyUsesContractCategoriesAdditional')
  }
  if (parameter === 'k' && algorithm === 'rule') {
    return systemText('preInvestment.autoAssetClassification.contractLabelsDetermineTheNumberOfRule')
  }
  return ''
}

const SELECT_CLASS =
  'w-full rounded-lg border px-2 py-1 text-sm disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-100 disabled:text-slate-600'

function labelClass(inactive: boolean): string {
  return `mb-1 block font-medium ${inactive ? 'text-slate-600' : 'text-slate-700'}`
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
  useI18n()
  return (
    <div className="mb-3">
      <div className="flex items-center gap-2">
        <span className="text-accent-600">◆</span>
        <h2 className="text-lg font-semibold">{title}</h2>
      </div>
      {hint && <p className="mt-1 pl-6 text-xs text-slate-600">{hint}</p>}
    </div>
  )
}

function formatNumber(value: number | null | undefined, digits = 3) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—'
}

export default function AutoAssetClassification() {
  const ALGORITHM_HINTS: Record<string, string> = {
  rule: systemText('preInvestment.autoAssetClassification.deterministicMappingFromFundContractTypeInvestment'),
  hierarchical: systemText('preInvestment.autoAssetClassification.agglomerativeHierarchicalClusteringOnReturnCorrelationDistances'),
  kmedoids: systemText('preInvestment.autoAssetClassification.usesActualProductsAsClassCentersNaturally'),
  kmeans: systemText('preInvestment.autoAssetClassification.centroidClusteringInStandardizedFeatureSpaceFast'),
  spectral: systemText('preInvestment.autoAssetClassification.transformsCorrelationDistancesIntoAGraphAnd'),
  gmm: systemText('preInvestment.autoAssetClassification.gaussianMixtureWithEmGivesEachClass'),
}
  const FEATURE_HINTS: Record<string, string> = {
  correlation: systemText('preInvestment.autoAssetClassification.usesDailyReturnCorrelationDistance05'),
  denoised: systemText('preInvestment.autoAssetClassification.usesCorrelationDistanceAfterRandomMatrixDenoising'),
  metrics: systemText('preInvestment.autoAssetClassification.euclideanDistanceOnProfilesOfReturnVolatility'),
  pca: systemText('preInvestment.autoAssetClassification.decomposesTheCorrelationMatrixAndClustersLeading'),
  blend: systemText('preInvestment.autoAssetClassification.clustersCombinedRiskReturnProfilesAndPrincipal'),
}
  const BLOCK_HINTS: Record<string, string> = {
  none: systemText('preInvestment.autoAssetClassification.clustersNavBehaviorOnlyACorrelationWindow'),
  asset_class: systemText('preInvestment.autoAssetClassification.keepsEquitiesFixedIncomeCommoditiesMoneyMarkets'),
  category: systemText('preInvestment.autoAssetClassification.alsoFixesSecondLevelTypesWithinAsset'),
  detail: systemText('preInvestment.autoAssetClassification.fixesThirdLevelDetailsSizeDividendLow'),
}
  useI18n()
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

  // Empty string means the parameter is live; a message means it is greyed out
  // and says why.
  const linkageInactive = inactiveReason('linkage', algorithm)
  const blockInactive = inactiveReason('blockBy', algorithm)
  const kInactive = inactiveReason('k', algorithm)

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
        setUniverseError(caught instanceof Error ? caught.message : systemText('preInvestment.autoAssetClassification.unableToLoadTheInvestableUniverseSnapshot'))
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
          setProductsError(caught instanceof Error ? caught.message : systemText('preInvestment.autoAssetClassification.unableToLoadInvestableUniverseProducts'))
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
      if (!response.ok) throw new Error(systemText('preInvestment.autoAssetClassification.backendError', { p0: response.status }))
      const data = await response.json() as ClassFitResult
      assertNativeNumericalExecutionLanes(data.execution, systemText('preInvestment.autoAssetClassification.automaticAssetClassNavFitting'))
      setFitResult(data)
    } catch (reason: any) {
      setFitError(systemText('preInvestment.autoAssetClassification.assetClassNavAndMetricCalculationFailed', { p0: reason?.message || reason }))
    }
  }

  async function onRun() {
    setError('')
    setSaveMessage('')
    if (!universe) {
      setError(systemText('preInvestment.autoAssetClassification.selectAndLockAProductPoolVersion'))
      return
    }
    if (pool.length < meta.limits.min_products) {
      setError(systemText('preInvestment.autoAssetClassification.selectAtLeastProducts', { p0: meta.limits.min_products }))
      return
    }
    if (sizeMax < sizeMin) {
      setError(systemText('preInvestment.autoAssetClassification.maximumProductsPerClassMustNotBe'))
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
      if (!response.ok) throw new Error(apiErrorMessage(data, systemText('preInvestment.autoAssetClassification.backendError', { p0: response.status })))
      assertNativeNumericalExecution(data.execution, systemText('preInvestment.autoAssetClassification.automaticAssetClassification'))
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
      setSaveMessage(systemText('preInvestment.autoAssetClassification.configurationNameCannotBeBlank'))
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
      if (!response.ok) throw new Error(apiErrorMessage(data, systemText('preInvestment.autoAssetClassification.error', { p0: response.status })))
      setSaveMessage(systemText('preInvestment.autoAssetClassification.configurationSavedAndAvailableInAssetAllocation', { p0: name }))
    } catch (reason: any) {
      setSaveMessage(systemText('preInvestment.autoAssetClassification.saveFailed', { p0: reason?.message || reason }))
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
        <p className="text-xs font-semibold uppercase tracking-wide text-accent-700">Pre-investment · SAA</p>
        <h1 className="mt-1 text-2xl font-bold text-slate-900">{systemText('preInvestment.autoAssetClassification.automaticAssetClassification')}</h1>
        <p className="mt-2 max-w-4xl text-sm text-slate-600">
          {systemText('preInvestment.autoAssetClassification.automaticallyClassifyTheLockedInvestableUniverseUsing')}</p>
      </header>

      {universe ? (
        <div className="mb-4 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900">
          {systemText('preInvestment.autoAssetClassification.lockedInvestableUniverse')}<b>{universe.name}</b> · {investableUniverseEligibleCount(universe)} {" " + systemText('preInvestment.autoAssetClassification.availableProductsResearchDate') + " "}{universe.research_date}
        </div>
      ) : (
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          <span>{universeLoading ? systemText('preInvestment.autoAssetClassification.loadingInvestableUniverse') : universeError || systemText('preInvestment.autoAssetClassification.lockAProductPoolVersionBeforeRunning')}</span>
          <Link to="/pre-investment/product-pool" className="rounded-lg bg-amber-800 px-3 py-2 font-medium text-white">{systemText('preInvestment.autoAssetClassification.selectProductPoolVersions')}</Link>
        </div>
      )}

      {/* 1. 产品池 */}
      <section className="rounded-xl border border-slate-200 bg-white p-4">
        <SectionTitle title={systemText('preInvestment.autoAssetClassification.1SelectProductsFromTheInvestableUniverse')} hint={systemText('preInvestment.autoAssetClassification.onlyProductsCurrentlyEligibleForNewConfigurations')} />
        <div className="grid gap-4 lg:grid-cols-2">
          <div>
            <div className="flex items-center gap-2">
              <input
                className="w-full rounded-lg border px-2 py-1 text-sm"
                placeholder={systemText('preInvestment.autoAssetClassification.searchByCodeNameOrEvaluationPlan')}
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                aria-label={systemText('preInvestment.autoAssetClassification.investableUniverseProductSearch')}
              />
            </div>
            <div className="mt-2 flex items-center justify-between gap-2">
              <label className="flex items-center gap-2 text-xs font-medium text-slate-700">
                <input
                  type="checkbox"
                  className="h-3.5 w-3.5"
                  disabled={searchResults.length === 0}
                  checked={allVisibleSelected}
                  onChange={(event) => toggleAllVisible(event.target.checked)}
                  aria-label={systemText('preInvestment.autoAssetClassification.selectAllCurrentResults')}
                />
                {systemText('preInvestment.autoAssetClassification.selectAllCurrentResults')}</label>
              <span className="text-xs text-slate-600">{systemText('preInvestment.autoAssetClassification.matched') + " "}{searchTotal} {" " + systemText('preInvestment.autoAssetClassification.productsShowingTheFirst') + " "}{searchResults.length} {" " + systemText('preInvestment.autoAssetClassification.items')}</span>
            </div>
            {productsError && (
              <p className="mt-2 rounded-lg border border-red-300 bg-red-50 px-2 py-1 text-xs text-red-700">
                {systemText('preInvestment.autoAssetClassification.unableToLoadInvestableUniverseProducts2')}{productsError}
              </p>
            )}
            <ul className="mt-2 max-h-64 divide-y overflow-auto rounded-lg border">
              {searchResults.map((item) => (
                <li key={item.code}>
                  <label className="flex cursor-pointer items-center gap-2 px-2 py-1 text-xs hover:bg-accent-50">
                    <input
                      type="checkbox"
                      className="h-3.5 w-3.5 shrink-0"
                      checked={poolCodes.has(item.code)}
                      onChange={(event) => toggleProduct(item, event.target.checked)}
                      aria-label={systemText('preInvestment.autoAssetClassification.select', { p0: item.name })}
                    />
                    <span className="min-w-0 flex-1 truncate">
                      <span className="font-mono text-slate-600">{item.code}</span> {item.name}
                      {typeof item.max_weight === 'number' && (
                        <span className="ml-1 rounded-lg bg-amber-100 px-1 py-0.5 text-xs font-semibold text-amber-800">
                          {systemText('preInvestment.autoAssetClassification.limit') + " "}{(item.max_weight * 100).toFixed(1)}%
                        </span>
                      )}
                      {item.pool_names && item.pool_names.length > 0 && (
                        <span className="ml-1 text-xs text-slate-600">{item.pool_names.join('/')}</span>
                      )}
                    </span>
                  </label>
                </li>
              ))}
              {searchResults.length === 0 && !productsError && (
                <li className="px-2 py-3 text-xs text-slate-600">{systemText('preInvestment.autoAssetClassification.noMatchingProducts')}</li>
              )}
            </ul>
            <p className="mt-3 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600">{systemText('preInvestment.autoAssetClassification.arbitraryCodesCannotBePastedPreventingBypass')}</p>
          </div>
          <div>
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold">{systemText('preInvestment.autoAssetClassification.selectedProducts')}{pool.length}）</h3>
              <button className="rounded-lg border px-2 py-0.5 text-xs hover:bg-slate-50" onClick={() => setPool([])}>{systemText('preInvestment.autoAssetClassification.clear')}</button>
            </div>
            <ul className="mt-2 max-h-80 divide-y overflow-auto rounded-lg border">
              {pool.map((item) => (
                <li key={item.code} className="flex items-center justify-between gap-2 px-2 py-1 text-xs">
                  <span className="min-w-0 truncate">
                    <span className="font-mono text-slate-600">{item.code}</span> {item.name || systemText('preInvestment.autoAssetClassification.resolvedByCode')}
                    {typeof item.max_weight === 'number' && (
                      <span className="ml-1 rounded-lg bg-amber-100 px-1 py-0.5 text-xs font-semibold text-amber-800">
                        {systemText('preInvestment.autoAssetClassification.limit') + " "}{(item.max_weight * 100).toFixed(1)}%
                      </span>
                    )}
                    {item.pool_names && item.pool_names.length > 0 && (
                      <span className="ml-1 text-xs text-slate-600">{item.pool_names.join('/')}</span>
                    )}
                  </span>
                  <button className="rounded-lg border px-2 py-0.5 text-xs" onClick={() => setPool((current) => current.filter((entry) => entry.code !== item.code))}>{systemText('preInvestment.autoAssetClassification.remove')}</button>
                </li>
              ))}
              {pool.length === 0 && <li className="px-2 py-3 text-xs text-slate-600">{systemText('preInvestment.autoAssetClassification.noProductsSelected')}</li>}
            </ul>
          </div>
        </div>
      </section>

      {/* 2. 参数 */}
      <section className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
        <SectionTitle title={systemText('preInvestment.autoAssetClassification.2ClassificationParameters')} hint={systemText('preInvestment.autoAssetClassification.algorithmsEstimateProductToClassAffinityA')} />
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <label className="text-sm">
            <span className="mb-1 block font-medium text-slate-700">{systemText('preInvestment.autoAssetClassification.algorithm')}</span>
            <select className={SELECT_CLASS} value={algorithm} onChange={(event) => setAlgorithm(event.target.value)}>
              {meta.algorithms.map((option) => <option key={option.id} value={option.id}>{systemText(`preInvestment.autoOptions.algorithms.${option.id}`, {}, option.label)}</option>)}
            </select>
          </label>
          <label className="text-sm">
            <span className="mb-1 block font-medium text-slate-700">{systemText('preInvestment.autoAssetClassification.featureSet')}</span>
            <select className={SELECT_CLASS} value={features} onChange={(event) => setFeatures(event.target.value)}>
              {meta.features.map((option) => <option key={option.id} value={option.id}>{systemText(`preInvestment.autoOptions.features.${option.id}`, {}, option.label)}</option>)}
            </select>
          </label>
          <div>
            <label className="text-sm">
              <span className={labelClass(Boolean(linkageInactive))}>{systemText('preInvestment.autoAssetClassification.linkageHierarchicalClustering')}</span>
              <select className={SELECT_CLASS} value={linkage} disabled={Boolean(linkageInactive)} onChange={(event) => setLinkage(event.target.value)}>
                {meta.linkages.map((option) => <option key={option.id} value={option.id}>{systemText(`preInvestment.autoOptions.linkages.${option.id}`, {}, option.label)}</option>)}
              </select>
            </label>
            {linkageInactive && <p className="mt-1 text-xs text-slate-600">{linkageInactive}</p>}
          </div>
          <label className="text-sm">
            <span className="mb-1 block font-medium text-slate-700">{systemText('preInvestment.autoAssetClassification.contractClassificationLevel')}</span>
            <select className={SELECT_CLASS} value={taxonomyLevel} onChange={(event) => setTaxonomyLevel(event.target.value)}>
              {meta.taxonomy_levels.map((option) => <option key={option.id} value={option.id}>{systemText(`preInvestment.autoOptions.taxonomy_levels.${option.id}`, {}, option.label)}</option>)}
            </select>
          </label>
          <div>
            <label className="text-sm">
              <span className={labelClass(Boolean(blockInactive))}>{systemText('preInvestment.autoAssetClassification.contractStratificationHardConstraint')}</span>
              <select className={SELECT_CLASS} value={blockBy} disabled={Boolean(blockInactive)} onChange={(event) => setBlockBy(event.target.value)}>
                {meta.block_modes.map((option) => <option key={option.id} value={option.id}>{systemText(`preInvestment.autoOptions.block_modes.${option.id}`, {}, option.label)}</option>)}
              </select>
            </label>
            {blockInactive && <p className="mt-1 text-xs text-slate-600">{blockInactive}</p>}
          </div>
          <label className="text-sm">
            <span className="mb-1 block font-medium text-slate-700">{systemText('preInvestment.autoAssetClassification.withinClassWeights')}</span>
            <select className={SELECT_CLASS} value={weightMode} onChange={(event) => setWeightMode(event.target.value)}>
              {meta.weight_modes.map((option) => <option key={option.id} value={option.id}>{systemText(`preInvestment.autoOptions.weight_modes.${option.id}`, {}, option.label)}</option>)}
            </select>
          </label>
          <div className="text-sm">
            <span className={labelClass(Boolean(kInactive))}>{systemText('preInvestment.autoAssetClassification.numberOfClassesK')}</span>
            <div className="flex items-center gap-2">
              <input
                type="number"
                min={2}
                className="w-24 rounded-lg border px-2 py-1 text-sm disabled:bg-slate-100"
                value={k}
                disabled={autoK || Boolean(kInactive)}
                onChange={(event) => setK(Math.max(2, Number(event.target.value) || 2))}
                aria-label={systemText('preInvestment.autoAssetClassification.numberOfAssetClasses')}
              />
              <label className={`flex items-center gap-1 text-xs ${kInactive ? 'text-slate-600' : 'text-slate-600'}`}>
                <input
                  type="checkbox"
                  checked={autoK}
                  disabled={Boolean(kInactive)}
                  onChange={(event) => setAutoK(event.target.checked)}
                  aria-label={systemText('preInvestment.autoAssetClassification.suggestNumberOfAssetClasses')}
                />
                {systemText('preInvestment.autoAssetClassification.suggestAutomatically')}</label>
            </div>
            {kInactive && <p className="mt-1 text-xs text-slate-600">{kInactive}</p>}
          </div>
          <label className="text-sm">
            <span className="mb-1 block font-medium text-slate-700">{systemText('preInvestment.autoAssetClassification.minimumProductsPerClass')}</span>
            <input type="number" min={1} className={SELECT_CLASS} value={sizeMin} onChange={(event) => setSizeMin(Math.max(1, Number(event.target.value) || 1))} aria-label={systemText('preInvestment.autoAssetClassification.minimumProductsPerClass')} />
          </label>
          <label className="text-sm">
            <span className="mb-1 block font-medium text-slate-700">{systemText('preInvestment.autoAssetClassification.maximumProductsPerClass')}</span>
            <input type="number" min={1} className={SELECT_CLASS} value={sizeMax} onChange={(event) => setSizeMax(Math.max(1, Number(event.target.value) || 1))} aria-label={systemText('preInvestment.autoAssetClassification.maximumProductsPerClass')} />
          </label>
          <label className="text-sm">
            <span className="mb-1 block font-medium text-slate-700">{systemText('preInvestment.autoAssetClassification.sampleStartDate')}</span>
            <input type="date" className={SELECT_CLASS} value={startDate} onChange={(event) => setStartDate(event.target.value)} aria-label={systemText('preInvestment.autoAssetClassification.sampleStartDate')} />
          </label>
          <label className="text-sm md:col-span-2">
            <span className="mb-1 block font-medium text-slate-700">{systemText('preInvestment.autoAssetClassification.productsExceedingCapacity')}</span>
            <select className={SELECT_CLASS} value={unassignedPolicy} onChange={(event) => setUnassignedPolicy(event.target.value as 'park' | 'force')}>
              <option value="park">{systemText('preInvestment.autoAssetClassification.moveToWatchlistRecommended')}</option>
              <option value="force">{systemText('preInvestment.autoAssetClassification.assignToTheNearestClass')}</option>
            </select>
          </label>
        </div>
        <div className="mt-3 rounded-lg bg-accent-50 p-3 text-xs text-accent-900">
          <p><strong>{meta.algorithms.find((item) => item.id === algorithm)?.label ?? algorithm}</strong>：{ALGORITHM_HINTS[algorithm]}</p>
          <p className="mt-1"><strong>{meta.features.find((item) => item.id === features)?.label ?? features}</strong>：{FEATURE_HINTS[features]}</p>
          {!blockInactive && <p className="mt-1"><strong>{systemText('preInvestment.autoAssetClassification.stratification')}</strong>：{BLOCK_HINTS[blockBy]}</p>}
        </div>
        <div className="mt-3 flex items-center gap-3">
          <button
            className="rounded-lg bg-accent-700 px-4 py-1.5 text-sm font-medium text-white hover:bg-accent-600 disabled:opacity-50"
            onClick={onRun}
            disabled={running || !universe}
          >
            {running ? systemText('preInvestment.autoAssetClassification.calculating') : systemText('preInvestment.autoAssetClassification.runAutomaticClassification')}
          </button>
          {error && <span className="text-sm text-red-600">{error}</span>}
        </div>
      </section>

      {result && (
        <>
          {/* 3. 诊断 */}
          <section className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
            <SectionTitle title={systemText('preInvestment.autoAssetClassification.3ClassificationDiagnostics')} hint={systemText('preInvestment.autoAssetClassification.silhouetteScoresMeasureWithinClassCohesionAnd')} />
            <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
              {[
                { label: systemText('preInvestment.autoAssetClassification.numberOfAssetClasses'), value: String(result.k) },
                { label: systemText('preInvestment.autoAssetClassification.overallSilhouetteScore'), value: formatNumber(result.diagnostics.silhouette) },
                { label: systemText('preInvestment.autoAssetClassification.commonSampleTradingDays'), value: String(result.observations) },
                { label: systemText('preInvestment.autoAssetClassification.significantEigenvalueCount'), value: String(result.diagnostics.significant_eigenvalues) },
                { label: systemText('preInvestment.autoAssetClassification.samplePeriod'), value: `${result.start_date} ~ ${result.end_date}` },
              ].map((card) => (
                <div key={card.label} className="rounded-lg border bg-slate-50 p-3">
                  <p className="text-xs text-slate-600">{card.label}</p>
                  <p className="mt-1 text-sm font-semibold text-slate-900">{card.value}</p>
                </div>
              ))}
            </div>
            {result.diagnostics.k_suggestions.length > 0 && (
              <div className="mt-3">
                <h4 className="text-xs font-semibold text-slate-700">{systemText('preInvestment.autoAssetClassification.suggestedKBySilhouetteScore')}</h4>
                <div className="mt-1 flex flex-wrap gap-2">
                  {result.diagnostics.k_suggestions.map((item) => (
                    <span key={item.k} className={`rounded-lg border px-2 py-0.5 text-xs ${item.k === result.k ? 'border-accent-500 bg-accent-50 font-semibold text-accent-800' : 'text-slate-600'}`}>
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
          <section className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
            <SectionTitle title={systemText('preInvestment.autoAssetClassification.4DraftAssetClassMapping')} hint={systemText('preInvestment.autoAssetClassification.marksTheRepresentativeProductMedoidWeightsFollow')} />
            {/* Provenance rides with the result, not with the window frame. */}
            <PitProvenance lineage={result.pit} />
            <div className="grid gap-3 lg:grid-cols-2">
              {result.classes.map((group) => (
                <div key={group.id} className="rounded-xl border border-accent-200 bg-accent-50/40 p-3">
                  <div className="flex items-baseline justify-between">
                    <h3 className="text-sm font-semibold text-accent-900">{group.name}</h3>
                    <span className="text-xs text-slate-600">
                      {group.size} {" " + systemText('preInvestment.autoAssetClassification.productsWithinClassCorrelation') + " "}{formatNumber(group.mean_corr, 2)} {" " + systemText('preInvestment.autoAssetClassification.silhouette') + " "}{formatNumber(group.silhouette, 2)}
                    </span>
                  </div>
                  <table className="mt-2 w-full text-xs">
                    <thead>
                      <tr className="text-left text-slate-600">
                        <th scope="col" className="py-1">{systemText('preInvestment.autoAssetClassification.products')}</th>
                        <th scope="col" className="py-1">{systemText('preInvestment.autoAssetClassification.contractClassification')}</th>
                        <th scope="col" className="py-1 text-right">{systemText('preInvestment.autoAssetClassification.poolLimit')}</th>
                        <th scope="col" className="py-1 text-right">{systemText('preInvestment.autoAssetClassification.weight')}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {group.etfs.map((member) => (
                        <tr key={member.code} className="border-t border-accent-100">
                          <td className="py-1">
                            {member.is_medoid && <span className="mr-1 text-amber-500" title={systemText('preInvestment.autoAssetClassification.representativeProduct')}>★</span>}
                            <span className="font-mono text-slate-600">{member.code}</span> {member.name}
                          </td>
                          <td className="py-1 text-slate-600" title={member.taxonomy?.path ?? member.contract_label}>{member.contract_label}</td>
                          <td className="py-1 text-right text-slate-600">
                            {member.max_weight === null ? '—' : `${(member.max_weight * 100).toFixed(1)}%`}
                          </td>
                          <td className={`py-1 text-right font-medium ${member.capped ? 'text-amber-700' : ''}`}>
                            {member.weight.toFixed(2)}
                            {member.capped && <span className="ml-1 text-xs" title={systemText('preInvestment.autoAssetClassification.reducedByProductPoolLimits')}>▼</span>}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {group.max_class_weight !== null && group.max_class_weight < 100 && (
                    <p className="mt-2 rounded-lg bg-amber-100 px-2 py-1 text-xs text-amber-900">
                      {systemText('preInvestment.autoAssetClassification.combinedMemberLimits') + " "}{group.max_class_weight.toFixed(1)}{systemText('preInvestment.autoAssetClassification.thisClassSSaaWeightMustNot')}</p>
                  )}
                </div>
              ))}
            </div>

            {result.diagnostics.blocks.length > 0 && (
              <div className="mt-4 rounded-lg border border-accent-200 bg-accent-50 p-3">
                <h4 className="text-xs font-semibold text-accent-900">
                  {systemText('preInvestment.autoAssetClassification.contractStrata')}{meta.block_modes.find((item) => item.id === result.block_by)?.label ?? result.block_by}）
                </h4>
                <p className="mt-1 text-xs text-accent-800">
                  {systemText('preInvestment.autoAssetClassification.eachContractBlockIsClusteredSeparatelyBlocks')}</p>
                <table className="mt-2 w-full text-xs">
                  <thead>
                    <tr className="text-left text-accent-700">
                      <th scope="col" className="py-1">{systemText('preInvestment.autoAssetClassification.contractBlock')}</th>
                      <th scope="col" className="py-1 text-right">{systemText('preInvestment.autoAssetClassification.productCount')}</th>
                      <th scope="col" className="py-1 text-right">{systemText('preInvestment.autoAssetClassification.classesWithinBlock')}</th>
                      <th scope="col" className="py-1 text-right">{systemText('preInvestment.autoAssetClassification.withinBlockSilhouette')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.diagnostics.blocks.map((item) => (
                      <tr key={item.block} className="border-t border-accent-200">
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
                <h4 className="text-xs font-semibold text-slate-700">{systemText('preInvestment.autoAssetClassification.behavioralVersusContractClassificationDifferences')}{result.diagnostics.contract_deviations.length}）</h4>
                <p className="mt-1 text-xs text-slate-600">{systemText('preInvestment.autoAssetClassification.theseProductsNavBasedClassesDifferFrom')}</p>
                <ul className="mt-2 max-h-48 divide-y overflow-auto rounded-lg border text-xs">
                  {result.diagnostics.contract_deviations.map((item) => (
                    <li key={item.code} className="px-2 py-1">
                      <span className="font-mono text-slate-600">{item.code}</span> {item.name}
                      <span className="ml-1 text-slate-600">{systemText('preInvestment.autoAssetClassification.contract')}{item.contract_label}{systemText('preInvestment.autoAssetClassification.assignedTo')}{item.assigned_class}」</span>
                    </li>
                  ))}
                  {result.diagnostics.contract_deviations.length === 0 && <li className="px-2 py-2 text-slate-600">{systemText('preInvestment.autoAssetClassification.noDifferences')}</li>}
                </ul>
              </div>
              <div>
                <h4 className="text-xs font-semibold text-slate-700">{systemText('preInvestment.autoAssetClassification.watchlistAndExclusions')}{result.unassigned.length + result.skipped.length}）</h4>
                <p className="mt-1 text-xs text-slate-600">{systemText('preInvestment.autoAssetClassification.productsExceedingClassCapacityWithInsufficientSimilarity')}</p>
                <ul className="mt-2 max-h-48 divide-y overflow-auto rounded-lg border text-xs">
                  {[...result.unassigned, ...result.skipped].map((item) => (
                    <li key={`${item.reason}-${item.code}`} className="px-2 py-1">
                      <span className="font-mono text-slate-600">{item.code}</span> {item.name}
                      <span className="ml-1 text-slate-600">（{item.detail}）</span>
                    </li>
                  ))}
                  {result.unassigned.length + result.skipped.length === 0 && <li className="px-2 py-2 text-slate-600">{systemText('preInvestment.autoAssetClassification.allProductsClassified')}</li>}
                </ul>
              </div>
            </div>

            {result.diagnostics.winsorized.length > 0 && (
              <div className="mt-4 rounded-lg bg-rose-50 p-3 text-xs text-rose-900">
                <h4 className="font-semibold">{systemText('preInvestment.autoAssetClassification.suspectedAdjustedNavAnomalies')}{result.diagnostics.winsorized.length}）</h4>
                <p className="mt-1">{systemText('preInvestment.autoAssetClassification.theseProductsHaveDailyJumpsOutsideRobust')}</p>
                <ul className="mt-1">
                  {result.diagnostics.winsorized.map((item) => (
                    <li key={item.code}>
                      · <span className="font-mono">{item.code}</span> {item.name}：{item.clipped} {" " + systemText('preInvestment.autoAssetClassification.observationsLargestOneDayChange') + " "}{item.max_raw_return === null ? '—' : `${(item.max_raw_return * 100).toFixed(1)}%`}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="mt-4 flex flex-wrap items-center gap-3 border-t pt-3">
              <input
                className="rounded-lg border px-2 py-1 text-sm"
                placeholder={systemText('preInvestment.autoAssetClassification.saveAsAssetConfigurationName')}
                value={saveName}
                onChange={(event) => setSaveName(event.target.value)}
                aria-label={systemText('preInvestment.autoAssetClassification.assetConfigurationName')}
              />
              <button className="rounded-lg bg-accent-700 px-3 py-1 text-sm text-white hover:bg-accent-600" onClick={onSave}>{systemText('preInvestment.autoAssetClassification.saveAssetConfiguration')}</button>
              <button className="rounded-lg border px-3 py-1 text-sm hover:bg-slate-50" onClick={openInManualWorkspace}>{systemText('preInvestment.autoAssetClassification.openInManualAssetConstruction')}</button>
              {saveMessage && <span className="text-sm text-slate-700">{saveMessage}</span>}
            </div>
          </section>

          {/* 5. 净值与指标 */}
          <section className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
            <SectionTitle title={systemText('preInvestment.autoAssetClassification.5AssetClassNavAndMetrics')} hint={systemText('preInvestment.autoAssetClassification.usesTheSameFittingPipelineAsManual')} />
            {fitError && <p className="text-sm text-red-600">{fitError}</p>}
            {!fitError && !fitResult && <p className="text-sm text-slate-600">{systemText('preInvestment.autoAssetClassification.calculatingAssetClassNav')}</p>}
            {affectedClasses.length > 0 && (
              <div className="mb-4 rounded-lg border border-rose-300 bg-rose-50 p-3 text-xs text-rose-900">
                <p className="font-semibold">{systemText('preInvestment.autoAssetClassification.reviewDataBeforeRelyingOnTheseClasses')}</p>
                <p className="mt-1">
                  {systemText('preInvestment.autoAssetClassification.clusteringFeaturesUseRobustTreatmentForAnomalous')}<b>{systemText('preInvestment.autoAssetClassification.originalAdjustedNavSeries')}</b>{systemText('preInvestment.autoAssetClassification.whichStillContainTheFollowingProductJumps')}</p>
                <ul className="mt-1">
                  {affectedClasses.map((item) => (
                    <li key={item.className}>
                      · <b>{item.className}</b> ← {item.products.map((product) => systemText('preInvestment.autoAssetClassification.oneDayChange', { p0: product.name, p1: product.code, p2: product.max_raw_return === null ? '—' : `${(product.max_raw_return * 100).toFixed(1)}%` })).join('、')}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {fitResult && <ClassFitPanel result={fitResult} />}
          </section>

          {fitResult && Array.isArray(fitResult.consistency) && fitResult.consistency.length > 0 && (
            <section className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
              <SectionTitle title={systemText('preInvestment.autoAssetClassification.6WithinClassConsistency')} hint={systemText('preInvestment.autoAssetClassification.redHighlightsIndicateMeanCorrelation06')} />
              <ClassConsistencyTable rows={fitResult.consistency} />
            </section>
          )}
        </>
      )}
    </div>
  )
}
