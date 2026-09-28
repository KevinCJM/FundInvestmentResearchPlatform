import { useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useResearchDay } from '../app/ResearchContext'
import { readAllocationJourney, updateAllocationJourney, useAllocationJourney } from '../app/allocationJourney'
import {
  listInvestableUniverseSnapshots,
  retireInvestableUniverseSnapshot,
  type InvestableUniverseSummary,
} from '../services/productPools'
import { getMandate, getStrategicCatalog, mandateCashFloor, type MandateVersion, type StrategicCatalog } from '../services/strategicAllocation'
import { cashFloorIssue, retireUniverse } from '../services/strategicScope'
import { actionClass, Badge, Button, DataTable, ErrorPanel, SectionHeader } from '../components/ui'
import { Field, inputClass, sectionClass } from '../components/risk-models/ResearchUI'
import { useI18n, systemText } from '../i18n/runtime'
import { WarningMark } from '../components/WarningMark'
import { UpstreamLink, UsabilityNote, VersionTag, upstreamOf } from '../components/versioning'
import type { Versioned } from '../services/versioning'

function messageOf(reason: unknown, fallback: string) {
  return reason instanceof Error && reason.message ? reason.message : fallback
}

/** 研究日与页面 PIT 日期不一致的提醒。 */
function ResearchDateWarning({ date, platformAsOf }: { date: string; platformAsOf: string }) {
  useI18n()
  return <WarningMark label={systemText('preInvestment.productPoolSelection.theResearchDateDiffersFromTheCurrent')}>
    {systemText('preInvestment.productPoolSelection.researchDate') + " "}{date} {" " + systemText('preInvestment.productPoolSelection.andTheCurrentPagePitDate') + " "}{platformAsOf} {" " + systemText('preInvestment.productPoolSelection.differYouCanContinueResearch')}
  </WarningMark>
}

interface SavedScopeRow {
  id: string
  name: string
  researchDate: string
  mandateId?: string
  mandateHash?: string
  kind: string
  count: number
  countLabel: string
  href: string
  copyHref: string
  editHref: string
  selected: boolean
  /** 战略范围的现金下限偏离所绑定投资目标时的提醒。 */
  cashIssue?: string
  /** 后端派生的版本、上游目标与可用性。 */
  lineage: Versioned
}

export default function ProductPoolSelection() {
  const { s } = useI18n()
  const [params] = useSearchParams()
  const [journey] = useAllocationJourney()
  const [mandates, setMandates] = useState<MandateVersion[]>([])
  const [catalogLoading, setCatalogLoading] = useState(true)
  const [catalogError, setCatalogError] = useState('')
  const [catalog, setCatalog] = useState<StrategicCatalog | null>(null)
  const [savedUniverses, setSavedUniverses] = useState<InvestableUniverseSummary[]>([])
  const [savedLoading, setSavedLoading] = useState(true)
  const [savedError, setSavedError] = useState('')
  const [savedReload, setSavedReload] = useState(0)
  const [catalogReload, setCatalogReload] = useState(0)
  const [scopeQuery, setScopeQuery] = useState('')
  const [deletingId, setDeletingId] = useState('')
  const [deleteBusy, setDeleteBusy] = useState('')
  const [deleteError, setDeleteError] = useState('')
  const deleteToken = useRef(0)
  const strategic = params.get('scope') === 'strategic'
  // 地址栏没带目标就是未选择：侧栏裸路径进来不替用户认领上次的目标。
  const mandateId = params.get('mandate') ?? ''
  const platformAsOf = useResearchDay()
  const [historicalMandates, setHistoricalMandates] = useState<Record<string, MandateVersion | null>>({})

  useEffect(() => {
    const controller = new AbortController()
    setCatalogLoading(true); setCatalogError('')
    getStrategicCatalog(controller.signal)
      .then(value => {
        if (controller.signal.aborted) return
        setMandates(value.mandates); setCatalog(value)
      })
      .catch(reason => { if (!controller.signal.aborted) setCatalogError(messageOf(reason, systemText('preInvestment.productPoolSelection.unableToLoadTheInvestmentObjective'))) })
      .finally(() => { if (!controller.signal.aborted) setCatalogLoading(false) })
    return () => controller.abort()
  }, [catalogReload])

  // 产品范围库读取不可变摘要；战略范围库来自同一份战略目录，避免两套真相。
  useEffect(() => {
    let active = true
    setSavedLoading(true); setSavedError('')
    listInvestableUniverseSnapshots()
      .then(value => { if (active) setSavedUniverses(value.items) })
      .catch(reason => { if (active) setSavedError(messageOf(reason, systemText('preInvestment.productPoolSelection.unableToLoadTheSavedResearchScope'))) })
      .finally(() => { if (active) setSavedLoading(false) })
    return () => { active = false }
  }, [savedReload])

  const modeHref = (nextStrategic: boolean) => {
    const next = new URLSearchParams(params)
    // 路径切换清理另一条路径的恢复/复制/新建参数，避免带着旧身份进入。
    next.delete('copy'); next.delete('new')
    if (nextStrategic) { next.set('scope', 'strategic'); next.delete('version') }
    else { next.delete('scope'); next.delete('strategic_universe'); next.delete('mapping') }
    if (mandateId) next.set('mandate', mandateId)
    const query = next.toString()
    return `/pre-investment/product-pool${query ? `?${query}` : ''}`
  }

  // 「新建」跳独立页面：new 必须每次渲染都重算，两次点击不能撞上同一个草稿 key。
  const newScopeQuery = new URLSearchParams()
  if (mandateId) newScopeQuery.set('mandate', mandateId)
  if (strategic) newScopeQuery.set('scope', 'strategic')
  newScopeQuery.set('new', String(Date.now()))
  const newScopeHref = `/pre-investment/product-pool/new?${newScopeQuery.toString()}`
  const clearScopeIdentity = () => updateAllocationJourney({ universeId: undefined, strategicUniverseId: undefined, implementationMappingId: undefined })

  const scopeHref = (query: Record<string, string>, boundMandate?: string) => {
    const next = new URLSearchParams(query)
    if (boundMandate) next.set('mandate', boundMandate)
    return `/pre-investment/product-pool/new?${next.toString()}`
  }
  const removeScope = async (row: SavedScopeRow) => {
    if (deleteBusy) return
    const token = ++deleteToken.current
    setDeleteBusy(row.id); setDeleteError('')
    try {
      if (strategic) await retireUniverse(row.id)
      else await retireInvestableUniverseSnapshot(row.id)
      if (token !== deleteToken.current) return
      setDeletingId('')
      // 用完成时的当前旅程判断：等待期间切换了选择就不清掉新选择。
      const current = readAllocationJourney()
      if (strategic && current.strategicUniverseId === row.id) {
        updateAllocationJourney({ strategicUniverseId: undefined, implementationMappingId: undefined })
      }
      if (!strategic && current.universeId === row.id) {
        updateAllocationJourney({ universeId: undefined })
      }
      if (strategic) setCatalogReload(n => n + 1)
      else setSavedReload(n => n + 1)
    } catch (reason) {
      if (token === deleteToken.current) setDeleteError(messageOf(reason, systemText('preInvestment.productPoolSelection.unableToRemoveThisItemFromThe')))
    } finally {
      if (token === deleteToken.current) setDeleteBusy('')
    }
  }
  // 路径切换或离开页面时作废在途删除，避免迟到结果清掉新选择。
  useEffect(() => {
    setDeletingId(''); setDeleteBusy(''); setDeleteError('')
    return () => { ++deleteToken.current }
  }, [strategic])
  const normalizedQuery = scopeQuery.trim().toLowerCase()
  const strategicScopes = catalog?.strategic_universes ?? []
  const scopesLoading = strategic ? catalogLoading : savedLoading
  const scopesError = strategic ? catalogError : savedError
  const scopeRows: SavedScopeRow[] = strategic
    ? strategicScopes
      .filter(item => !normalizedQuery || item.name.toLowerCase().includes(normalizedQuery))
      .map(item => ({
        id: item.id,
        name: item.name,
        researchDate: item.definition.as_of,
        mandateId: item.mandate_id, mandateHash: item.mandate_hash,
        kind: systemText('preInvestment.productPoolSelection.strategicScope'),
        count: item.definition.assets.length,
        countLabel: systemText('preInvestment.productPoolSelection.strategicAssets'),
        href: scopeHref({ scope: 'strategic', strategic_universe: item.id }, item.mandate_id),
        copyHref: scopeHref({ scope: 'strategic', strategic_universe: item.id, copy: '1' }, item.mandate_id),
        editHref: scopeHref({ scope: 'strategic', strategic_universe: item.id, edit: '1' }, item.mandate_id),
        selected: journey.strategicUniverseId === item.id,
        lineage: item,
        cashIssue: cashFloorIssue(item.definition.assets, mandateCashFloor(mandates.find(mandate => mandate.id === item.mandate_id))),
      }))
    : savedUniverses
      .filter(item => !normalizedQuery || item.name.toLowerCase().includes(normalizedQuery))
      .map(item => ({
        id: item.id,
        name: item.name,
        researchDate: item.research_date,
        mandateId: item.mandate_id, mandateHash: item.mandate_hash,
        kind: systemText('preInvestment.productPoolSelection.productPoolScope'),
        // product_count 是冻结总数；只有确实存过 eligible_count 时才敢写“可投资”。
        count: item.summary?.eligible_count ?? item.product_count ?? 0,
        countLabel: item.summary?.eligible_count != null ? systemText('preInvestment.productPoolSelection.investableProducts') : systemText('preInvestment.productPoolSelection.products'),
        href: scopeHref({ universe: item.id }, item.mandate_id),
        copyHref: scopeHref({ universe: item.id, copy: item.id }, item.mandate_id),
        editHref: scopeHref({ universe: item.id, edit: item.id }, item.mandate_id),
        selected: journey.universeId === item.id,
        lineage: item,
      }))

  // 活动目录不含已停用目标；按保存时的精确 ID 补读，不能用当前目标或名称猜测。
  useEffect(() => {
    if (catalogLoading || catalogError) return
    const controller = new AbortController()
    const ids = [...new Set([...strategicScopes, ...savedUniverses].map(item => item.mandate_id)
      .filter((id): id is string => Boolean(id) && !mandates.some(mandate => mandate.id === id)))]
    setHistoricalMandates({})
    void Promise.all(ids.map(async id => {
      try {
        const value = await getMandate(id, controller.signal)
        return [id, value.id === id ? value : null] as const
      } catch { return [id, null] as const }
    })).then(entries => { if (!controller.signal.aborted) setHistoricalMandates(Object.fromEntries(entries)) })
    return () => controller.abort()
  }, [catalog, savedUniverses, catalogLoading, catalogError])

  const mandateCell = (row: SavedScopeRow) => {
    if (!row.mandateId) return s('researchScope.mandateUnbound')
    if (catalogError) return s('researchScope.mandateUnavailable')
    const mandate = mandates.find(item => item.id === row.mandateId) ?? historicalMandates[row.mandateId]
    if (mandate === null || (mandate && row.mandateHash && mandate.content_hash !== row.mandateHash)) return s('researchScope.mandateUnavailable')
    if (!mandate) return s('researchScope.mandateLoading')
    const ref = upstreamOf(row.lineage, 'mandate')
    if (ref?.id === row.mandateId) return <><UpstreamLink item={ref} /><UsabilityNote usable={row.lineage.usable} /></>
    return <Link to={`/pre-investment/objectives/new?view=${encodeURIComponent(row.mandateId)}`}
      className="inline-flex min-h-10 items-center font-medium text-accent-800 underline focus-visible:ring-2 focus-visible:ring-accent-500">{mandate.name}</Link>
  }

  return <div className="space-y-5">
    <section aria-labelledby="research-start-heading" className={`${sectionClass} shadow-sm`}>
      <h1 id="research-start-heading" className="text-lg font-semibold text-slate-900">{systemText('preInvestment.productPoolSelection.chooseResearchPathAndScope')}</h1>
      <p className="mt-2 text-sm leading-6 text-slate-600">{s('researchScopeIntro.description')}</p>
      <p className="mt-1 text-sm leading-6 text-slate-600">{s('researchScopeIntro.workflow')}</p>
      <nav className="mt-3 grid gap-1 rounded-lg border border-slate-300 bg-slate-50 p-1 sm:inline-grid sm:grid-cols-2" aria-label={systemText('preInvestment.productPoolSelection.researchPath')}>
        <Link aria-label={systemText('preInvestment.productPoolSelection.existingProductsChooseProductPools')} aria-current={!strategic ? 'page' : undefined} to={modeHref(false)} onClick={() => updateAllocationJourney({ strategicUniverseId: undefined, implementationMappingId: undefined })} className={`flex min-h-10 items-center rounded-lg px-3 text-sm font-semibold sm:justify-center ${!strategic ? 'bg-accent-600 text-white shadow-sm' : 'text-slate-700 hover:bg-white'}`}>{systemText('preInvestment.productPoolSelection.productFirstChooseProductPools')}</Link>
        <Link aria-label={systemText('preInvestment.productPoolSelection.startWithStrategyIndependentAssetScope')} aria-current={strategic ? 'page' : undefined} to={modeHref(true)} className={`flex min-h-10 items-center rounded-lg px-3 text-sm font-semibold sm:justify-center ${strategic ? 'bg-accent-600 text-white shadow-sm' : 'text-slate-700 hover:bg-white'}`}>{systemText('preInvestment.productPoolSelection.strategyFirstDefineStrategicAssets')}</Link>
      </nav>
      {/* 只说明当前选中的路径。两条并排时各占半栏被迫折行，读者还要先分辨哪条与自己有关。 */}
      <p aria-label={systemText('preInvestment.productPoolSelection.researchPathGuidance')} className="mt-3 text-sm leading-6 text-slate-600">
        {strategic
          ? <><strong className="font-semibold text-slate-800">Strategy first：</strong>{systemText('preInvestment.productPoolSelection.useThisWhenDefiningTheAllocationFramework')}</>
          : <><strong className="font-semibold text-slate-800">Product first：</strong>{systemText('preInvestment.productPoolSelection.useThisWhenYouAlreadyHaveA')}</>}
      </p>
    </section>
    <section className={`${sectionClass} shadow-sm`} aria-label={systemText('preInvestment.productPoolSelection.savedResearchScopes')}>
      <SectionHeader title={systemText('preInvestment.productPoolSelection.savedResearchScopes')} description={systemText('preInvestment.productPoolSelection.resumeRestoresTheSavedVersionEditingSaves')} />
      <div className="mt-3 flex flex-wrap items-end gap-3">
        <Field label={systemText('preInvestment.productPoolSelection.searchScopeNames')}><input value={scopeQuery} onChange={event => setScopeQuery(event.target.value)} className={inputClass} /></Field>
        <Link to={newScopeHref} onClick={clearScopeIdentity} className={actionClass('primary')}>{systemText('preInvestment.productPoolSelection.newResearchScope')}</Link>
      </div>
      {((!strategic && catalogError) || Object.values(historicalMandates).some(value => value === null)) && <p role="alert" className="mt-2 text-sm text-rose-800">{s('researchScope.mandateLoadFailed')} <button type="button" className="min-h-10 font-semibold underline" onClick={() => setCatalogReload(n => n + 1)}>{s('preInvestment.scopeMandateSummary.retryLoadingObjective')}</button></p>}
      {deleteError && <p role="alert" className="mt-2 text-sm text-rose-800">{deleteError}</p>}
      {/* 范围库没读出来时表里一行都没有，空态文案会把读取失败说成「还没有范围」；整块换成公共错误态。 */}
      {scopesError ? <ErrorPanel className="mt-3" message={scopesError} action={<Button onClick={() => strategic ? setCatalogReload(n => n + 1) : setSavedReload(n => n + 1)}>{systemText('preInvestment.productPoolSelection.retryLoadingScopes')}</Button>} />
      : <div className="mt-3"><DataTable<SavedScopeRow>
        caption={systemText('preInvestment.productPoolSelection.savedResearchScopes')}
        rows={scopeRows}
        rowKey={row => row.id}
        minWidth="720px"
        loading={scopesLoading ? systemText('preInvestment.productPoolSelection.loadingSavedResearchScopes') : undefined}
        empty={normalizedQuery ? systemText('preInvestment.productPoolSelection.noMatchingResearchScopesClearTheSearch') : systemText('preInvestment.productPoolSelection.noSavedYetSelectNewResearchScope', { p0: strategic ? systemText('preInvestment.productPoolSelection.strategicScope') : systemText('preInvestment.productPoolSelection.productScope') })}
        columns={[
          { header: systemText('preInvestment.productPoolSelection.name'), cell: row => <span className="flex flex-wrap items-center gap-2"><Link to={row.href} aria-current={row.selected ? 'true' : undefined} className="inline-flex min-h-10 items-center font-medium text-accent-800 underline">{row.name}</Link><VersionTag version={row.lineage.version} />{row.selected && <Badge tone="neutral">{systemText('preInvestment.productPoolSelection.current')}</Badge>}{row.cashIssue && <WarningMark label={systemText('preInvestment.scopeCashFloor.label')}>{row.cashIssue}</WarningMark>}</span> },
          { header: systemText('preInvestment.productPoolSelection.investmentObjectivesAndConstraints'), cell: row => <span className="block min-w-32 break-words text-slate-700">{mandateCell(row)}</span> },
          { header: systemText('preInvestment.productPoolSelection.researchDate'), nowrap: true, cell: row => <span className="inline-flex items-center gap-1.5">
            {row.researchDate}
            {typeof platformAsOf === 'string' && row.researchDate !== platformAsOf && <ResearchDateWarning date={row.researchDate} platformAsOf={platformAsOf} />}
          </span> },
          { header: systemText('preInvestment.productPoolSelection.type'), nowrap: true, cell: row => row.kind },
          { header: systemText('preInvestment.productPoolSelection.size'), nowrap: true, cell: row => systemText('preInvestment.productPoolSelection.text', { p0: row.count, p1: row.countLabel }) },
          { header: systemText('preInvestment.productPoolSelection.actions'), nowrap: true, cell: row => deletingId === row.id
            ? <span className="flex flex-wrap items-center gap-3"><span className="text-xs text-slate-700">{systemText('preInvestment.productPoolSelection.remove')}{row.name}{systemText('preInvestment.productPoolSelection.fromTheListHistoricalResearchReferencingIt')}</span><Button disabled={deleteBusy === row.id} onClick={() => void removeScope(row)}>{deleteBusy === row.id ? systemText('preInvestment.productPoolSelection.removing') : systemText('preInvestment.productPoolSelection.confirmRemoval')}</Button><Button disabled={deleteBusy === row.id} onClick={() => setDeletingId('')}>{systemText('preInvestment.productPoolSelection.cancel')}</Button></span>
            : <span className="flex flex-wrap gap-3"><Link to={row.href} className="inline-flex min-h-10 items-center text-accent-800 underline">{systemText('preInvestment.productPoolSelection.continueResearch')}</Link><Link to={row.editHref} className="inline-flex min-h-10 items-center text-accent-800 underline">{systemText('preInvestment.productPoolSelection.edit')}</Link><Link to={row.copyHref} className="inline-flex min-h-10 items-center text-slate-600 underline">{systemText('preInvestment.productPoolSelection.copyAsNew')}</Link><button type="button" onClick={() => { setDeleteError(''); setDeletingId(row.id) }} className="inline-flex min-h-10 items-center text-rose-800 underline">{systemText('preInvestment.productPoolSelection.delete')}</button></span> },
        ]}
      /></div>}
    </section>
  </div>
}
