import { systemText, useI18n } from '../../i18n/runtime'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { readAllocationJourney, updateAllocationJourney, useAllocationDraft } from '../../app/allocationJourney'
import { useResearchDay } from '../../app/ResearchContext'
import { getStrategicCatalog, type StrategicCatalog } from '../../services/strategicAllocation'
import { cashFloorIssue, confirmUniverse, getStrategicUniverse, previewUniverse, proxySummaryText, researchProxyIssue, weightLimitsIssue, withCashFloor, type StrategicAsset, type UniverseDefinition, type UniverseVersion } from '../../services/strategicScope'
import { actionClass, Button, DataTable, SectionHeader } from '../ui'
import { Feedback, Field, inputClass, sectionClass, today } from '../risk-models/ResearchUI'
import UniverseFields from './UniverseFields'
import type { CategoryReuseItem } from '../risk-scales/CategoryEditor'
import ImplementationMappingEditor from './ImplementationMappingEditor'
import { useScopeOperation } from './useScopeOperation'
import ScopeFeasibility from './ScopeFeasibility'
import { WarningMark } from '../WarningMark'

interface StrategicScopeWorkspaceProps {
  mandateField?: ReactNode
  onCatalogLoaded?: (catalog: StrategicCatalog) => void
  onCatalogError?: (message: string) => void
  /** 由第 02 步传入：非空时不能确认保存或继续，直到选定已发布投资目标与约束。 */
  mandateBlockedReason?: string
  /** 第 02 步「新建研究范围」的显式命令标识；变化时清空编辑草稿。 */
  freshKey?: string
  /** 编辑保存：当前正在替代的已保存范围 ID。 */
  editSource?: string
  /** 同名预检用的活动范围名称（后端仍为权威校验）。 */
  existingScopeNames?: string[]
  /** 所选投资目标生效的现金下限：用于补齐现金大类及偏离提醒。 */
  cashFloor?: number
}

export default function StrategicScopeWorkspace({ mandateField, onCatalogLoaded, onCatalogError, mandateBlockedReason, freshKey = '', editSource = '', existingScopeNames = [], cashFloor = 0 }: StrategicScopeWorkspaceProps = {}) {
  useI18n()
  const [params, setParams] = useSearchParams()
  const clock = useResearchDay()
  const previousClock = useRef(clock)
  const copyRequested = params.get('copy') === '1'
  const editRequested = Boolean(editSource)
  const freshRequested = Boolean(freshKey)
  const requestedScopeId = params.get('strategic_universe') ?? ''
  const domainId = params.get('universe') ?? ''
  // 新建使用独立草稿身份：不恢复旧旅程，也不破坏上一份工作草稿。
  const [draft, setDraft] = useAllocationDraft<UniverseDefinition>(freshRequested ? `strategic-universe:editor:new:${freshKey}` : 'strategic-universe:editor', () => ({ name: systemText('preInvestment.strategicScopeWorkspace.independentStrategicScope', { p0: clock ?? today() }), as_of: clock ?? today(), currency: 'CNY', source: '', assets: [] }))
  const [catalog, setCatalog] = useState<StrategicCatalog | null>(null)
  const [catalogError, setCatalogError] = useState('')
  const [reload, setReload] = useState(0)
  const [loading, setLoading] = useState(true)
  const [saved, setSaved] = useState<UniverseVersion | null>(null)
  const [mappingId, setMappingId] = useState(params.get('mapping') ?? '')
  // 编辑/复制要等原版本读回后再补现金大类，否则补在旧草稿上会被读回结果覆盖。
  const [loadedView, setLoadedView] = useState('')
  const seededCash = useRef('')
  const pitLocked = typeof clock === 'string'
  const cutoff = clock ?? today()
  const operation = useScopeOperation()
  // Immutable history reads do not depend on the current research clock. Keep
  // them separate so initial PIT settings cannot cancel version restoration.
  const historyRead = useScopeOperation()
  useEffect(() => {
    const controller = new AbortController(); setLoading(true); setCatalogError('')
    getStrategicCatalog(controller.signal).then(value => { if (!controller.signal.aborted) { setCatalog(value); onCatalogLoaded?.(value) } })
      .catch(error => { if (!controller.signal.aborted) { const message = error instanceof Error ? error.message : systemText('preInvestment.strategicScopeWorkspace.unableToLoadTheCatalog'); setCatalogError(message); onCatalogError?.(message) } })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [reload, onCatalogLoaded, onCatalogError])
  function read(id: string, mapping = '') {
    operation.invalidate(); setMappingId(mapping)
    void historyRead.run(signal => getStrategicUniverse(id, signal), result => {
      if (result.id !== id) throw new Error(systemText('preInvestment.strategicScopeWorkspace.theLoadedStrategicScopeIdentityDoesNot'))
      if (editRequested) {
        // 编辑：载入原字段到可编辑草稿，确认时以新版本替代当前版本。
        setDraft({ ...result.definition })
        setSaved(null); setLoadedView(viewKey)
        return
      }
      if (copyRequested) {
        // 复制是显式命令：只产生可编辑草稿，不改写已保存版本或旅程引用。
        setDraft({ ...result.definition, name: systemText('preInvestment.strategicScopeWorkspace.copy', { p0: result.name }) })
        setSaved(null); setLoadedView(viewKey)
        return
      }
      setSaved(result); setDraft(result.definition)
      updateAllocationJourney({ strategicUniverseId: result.id, implementationMappingId: mapping || undefined })
    })
  }
  // 已保存、复制、新建是不同视图身份：切换时作废在途读写并按身份重读，
  // 旧的只读摘要或迟到保存不能覆盖当前身份。
  const viewKey = `${requestedScopeId}|${editRequested ? 'edit' : copyRequested ? 'copy' : 'view'}|${freshKey}`
  useEffect(() => {
    operation.invalidate(); historyRead.invalidate()
    setSaved(null)
    if (freshRequested) { setMappingId(''); return }
    if (requestedScopeId) read(requestedScopeId, params.get('mapping') ?? '')
  }, [viewKey])
  useEffect(() => { if (previousClock.current === clock) return; previousClock.current = clock; operation.invalidate() }, [clock])
  // PIT 开启时研究日由平台口径决定，不靠用户记得改：口径变了、或草稿是旧口径存下来的，都跟过来。
  // 只读的历史版本有自己的研究日，不对齐。
  useEffect(() => {
    if (!pitLocked || saved || draft.as_of === clock) return
    operation.invalidate()
    setDraft(current => ({ ...current, as_of: clock }))
  }, [clock, pitLocked, saved, draft.as_of])
  // 目标设了现金下限：每个编辑身份补一次现金大类及下限；之后用户调低或删除都只提醒、不再改回。
  const mandateParam = params.get('mandate') ?? ''
  useEffect(() => {
    const key = `${viewKey}|${mandateParam}|${cashFloor}`
    const ready = freshRequested || !requestedScopeId || loadedView === viewKey
    if (saved || !ready || !(cashFloor > 0) || seededCash.current === key) return
    seededCash.current = key
    setDraft(current => withCashFloor(current, cashFloor))
  }, [viewKey, mandateParam, cashFloor, saved, loadedView])
  const change = (value: UniverseDefinition) => { historyRead.invalidate(); operation.invalidate(); setSaved(null); setDraft(value) }
  // 常用大类来自已保存的战略范围；同名保留最近一次出现的代理配置。
  const reuseCategories = useMemo<CategoryReuseItem[]>(() => {
    const byName = new Map<string, CategoryReuseItem>()
    for (const universe of catalog?.strategic_universes ?? []) for (const asset of universe.definition.assets) {
      const proxy = asset.research_proxy, name = asset.name.trim()
      if (!name || !proxy || (proxy.asset_type === 'market' && !proxy.components.length)) continue
      byName.set(name, { name, summary: proxySummaryText(proxy), proxy, sourceLabels: proxy.source_labels })
    }
    return [...byName.values()]
  }, [catalog])
  const duplicateName = Boolean(draft.name.trim()) && existingScopeNames.some(existing => existing.trim().toLowerCase() === draft.name.trim().toLowerCase())
  const nameError = duplicateName ? systemText('preInvestment.strategicScopeWorkspace.aResearchScopeWithThisNameAlready') : operation.error.includes(systemText('preInvestment.strategicScopeWorkspace.nameAlreadyExists')) ? systemText('preInvestment.strategicScopeWorkspace.aResearchScopeWithThisNameAlready') : ''
  const issue = clock === undefined ? systemText('preInvestment.strategicScopeWorkspace.theKnowledgeCutoffIsUnconfirmedSavingIs') : !draft.name.trim() ? systemText('preInvestment.strategicScopeWorkspace.enterAStrategicScopeName') : nameError ? nameError : !draft.as_of || draft.as_of > (clock ?? today()) ? systemText('preInvestment.strategicScopeWorkspace.theStrategicResearchDateMustNotExceed') : !draft.assets.length ? systemText('preInvestment.strategicScopeWorkspace.addAtLeastOneStrategicAsset') : new Set(draft.assets.map(a => a.id)).size !== draft.assets.length ? systemText('preInvestment.strategicScopeWorkspace.duplicateAssetRecordsRemoveTheDuplicatesAnd') : draft.assets.some(a => !/^[a-z][a-z0-9_-]{0,79}$/.test(a.id) || !a.name.trim() || a.currency !== draft.currency) ? systemText('preInvestment.strategicScopeWorkspace.nameEveryAssetClassAndUseA') : researchProxyIssue(draft) || weightLimitsIssue(draft)
  const mandateId = mandateParam
  useEffect(() => { operation.invalidate() }, [mandateId, mandateBlockedReason])
  const saveBlockedReason = historyRead.busy ? systemText('preInvestment.strategicScopeWorkspace.loadingTheSavedScopePleaseWait') : historyRead.error || issue || mandateBlockedReason || ''
  const cashIssue = cashFloorIssue((saved?.definition ?? draft).assets, cashFloor)
  const saveMessage = operation.busy ? systemText('preInvestment.strategicScopeWorkspace.checkingAndSavingPleaseWait') : operation.error ? systemText('preInvestment.strategicScopeWorkspace.saveFailed', { p0: operation.error }) : saveBlockedReason
  function save() {
    if (saveBlockedReason || operation.busy) return
    void operation.run(async signal => {
      const checked = await previewUniverse(draft, signal)
      // 编辑、切换目标或离开页面后，迟到的校验结果不能继续写入。
      if (signal.aborted) throw new DOMException(systemText('preInvestment.strategicScopeWorkspace.saveCancelled'), 'AbortError')
      return confirmUniverse(draft, checked.preview_hash, signal, editRequested ? editSource : undefined, mandateId || undefined)
    }, result => {
      setSaved(result)
      updateAllocationJourney({ strategicUniverseId: result.id, mandateId: mandateId || readAllocationJourney().mandateId })
      setReload(n => n + 1)
      // 保存成功即写规范 URL：新 ID 成为唯一身份，清掉来源参数与过期映射。
      setParams(current => {
        current.set('scope', 'strategic')
        current.set('strategic_universe', result.id)
        current.delete('edit'); current.delete('copy'); current.delete('new'); current.delete('mapping')
        return current
      }, { replace: true })
    })
  }
  return <section className={`${sectionClass} space-y-5 shadow-sm`} aria-label={systemText('preInvestment.strategicScopeWorkspace.independentStrategicScope2')}>
    <SectionHeader title={systemText('preInvestment.strategicScopeWorkspace.configureAssetClassesAndResearchProxies')} description={systemText('preInvestment.strategicScopeWorkspace.defineAssetClassesThenSelectRepresentativeIndices')} />
    {mandateField}
    <Feedback error={historyRead.error || catalogError} />
    {historyRead.busy && <p role="status" className="text-sm text-slate-600">{systemText('preInvestment.strategicScopeWorkspace.loadingSavedScope')}</p>}
    {saved && <div className="space-y-3">
      <p className="text-sm text-slate-700">{systemText('preInvestment.strategicScopeWorkspace.readOnlyStrategicScope')}{saved.name} · {saved.definition.as_of}</p>
      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <div className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-600">{systemText('preInvestment.strategicScopeWorkspace.strategicBaseCurrency')}</dt><dd className="mt-1 text-sm font-semibold">{saved.definition.currency}</dd></div>
        <div className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-600">{systemText('preInvestment.strategicScopeWorkspace.strategicAssets')}</dt><dd className="mt-1 text-sm font-semibold">{saved.definition.assets.length} {" " + systemText('preInvestment.strategicScopeWorkspace.items')}</dd></div>
        <div className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-600">{systemText('preInvestment.strategicScopeWorkspace.researchDate')}</dt><dd className="mt-1 text-sm font-semibold">{saved.definition.as_of}</dd></div>
      </dl>
      <DataTable<StrategicAsset>
        caption={systemText('preInvestment.strategicScopeWorkspace.readOnlyStrategicAssetSummary')}
        rows={saved.definition.assets}
        rowKey={asset => asset.id}
        minWidth="560px"
        empty={systemText('preInvestment.strategicScopeWorkspace.thisScopeContainsNoStrategicAssets')}
        columns={[
          { header: systemText('preInvestment.strategicScopeWorkspace.asset'), cell: asset => <span className="block break-words font-medium text-slate-900">{asset.name}</span> },
          { header: systemText('preInvestment.strategicScopeWorkspace.researchProxyCashReturn'), cell: asset => asset.research_proxy?.asset_type === 'cash' ? systemText('preInvestment.strategicScopeWorkspace.expectedAnnualCashReturn', { p0: Number((asset.research_proxy.cash_return ?? 0) * 100).toFixed(2) }) : asset.research_proxy?.components.length ? asset.research_proxy.components.map(component => <span key={component.series_id} className="block break-words">{asset.research_proxy?.source_labels[component.series_id] || component.series_id.split(':').pop()} · {Number(component.weight * 100).toFixed(2)}%</span>) : systemText('preInvestment.strategicScopeWorkspace.notConfigured') },
          { header: systemText('preInvestment.strategicScopeWorkspace.assetType'), nowrap: true, cell: asset => asset.research_proxy ? asset.research_proxy.asset_type === 'cash' ? systemText('preInvestment.strategicScopeWorkspace.cash') : systemText('preInvestment.strategicScopeWorkspace.nonCash') : systemText('preInvestment.strategicScopeWorkspace.notSet') },
          { header: systemText('preInvestment.universeFields.weightLimits'), nowrap: true, cell: asset => {
            const limits = asset.weight_limits
            const issue = asset.role === 'liquidity' && asset.liquidity === 'liquid' ? cashFloorIssue([asset], cashFloor) : ''
            return <span className="inline-flex items-center gap-1.5 tabular-nums">
              {limits ? `${(limits.min_weight * 100).toFixed(2)}% – ${(limits.max_weight * 100).toFixed(2)}%` : systemText('preInvestment.strategicScopeWorkspace.followsMandate')}
              {issue && <WarningMark label={systemText('preInvestment.scopeCashFloor.label')}>{issue}</WarningMark>}
            </span>
          } },
        ]}
      />
    </div>}
    {!saved && <fieldset disabled={historyRead.busy} className="min-w-0"><UniverseFields value={draft} onChange={change} cutoff={cutoff} pitLocked={pitLocked} nameError={nameError} reuseCategories={reuseCategories} cashFloor={cashFloor} /></fieldset>}
    <ScopeFeasibility clock={clock}
      input={mandateId && draft.assets.length ? { mandate_id: mandateId, as_of: draft.as_of, strategic_definition: saved?.definition ?? draft } : null}
      blockedReason={historyRead.busy ? systemText('preInvestment.strategicScopeWorkspace.loadingResearchScopePleaseWait') : historyRead.error || mandateBlockedReason || (!saved && issue) || (draft.as_of > cutoff ? systemText('preInvestment.strategicScopeWorkspace.setTheResearchDateOnOrBefore') : '')} />
    {!saved && <>
      <div className="sticky bottom-0 -mx-4 space-y-3 border-t border-slate-200 bg-white px-4 py-3 sm:-mx-5 sm:px-5" aria-label={systemText('preInvestment.strategicScopeWorkspace.saveStrategicScope')}>
        {cashIssue && <p role="status" className="text-sm text-amber-800">{cashIssue}</p>}
        {saveMessage && <p id="scope-save-reason" role={operation.error ? 'alert' : 'status'} className={operation.error ? 'rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900' : 'text-sm text-amber-800'}>{saveMessage}</p>}
        <div className="flex flex-wrap gap-3">
          <Button tone="primary" disabled={Boolean(saveBlockedReason) || operation.busy} aria-describedby={saveMessage ? 'scope-save-reason' : undefined} onClick={save}>
            {operation.busy ? systemText('preInvestment.strategicScopeWorkspace.saving') : editRequested ? systemText('preInvestment.strategicScopeWorkspace.saveChanges') : systemText('preInvestment.strategicScopeWorkspace.saveStrategicScope')}
          </Button>
          {editRequested && <Button disabled={operation.busy || historyRead.busy} onClick={() => { operation.invalidate(); setParams(current => { current.delete('edit'); current.delete('copy'); return current }, { replace: true }) }}>{systemText('preInvestment.strategicScopeWorkspace.cancelEditing')}</Button>}
        </div>
      </div>
    </>}
    {(saved || catalogError) && <details className="space-y-3 border-t border-slate-200 pt-4"><summary className="min-h-10 cursor-pointer text-sm font-medium">{systemText('preInvestment.strategicScopeWorkspace.loadHistoricalMappingsAndRetry')}</summary>
      {saved && <Field label={systemText('preInvestment.strategicScopeWorkspace.savedImplementationMappings')} optional><select className={inputClass} value={mappingId} onChange={e => { setMappingId(e.target.value); if (!e.target.value) updateAllocationJourney({ implementationMappingId: undefined }) }}><option value="">{systemText('preInvestment.strategicScopeWorkspace.createANewMappingOrLeaveUnmatched')}</option>{catalog?.implementation_maps?.filter(m => m.definition.strategic_universe_id === saved.id).map(m => <option key={m.id} value={m.id}>{m.name} · {m.implementation_status === 'complete' ? systemText('preInvestment.strategicScopeWorkspace.fullCoverage') : systemText('preInvestment.strategicScopeWorkspace.gapsRemain')}</option>)}</select></Field>}
      <Button disabled={loading} onClick={() => setReload(n => n + 1)}>{systemText('preInvestment.strategicScopeWorkspace.retryLoadingScopeCatalog')}</Button>
    </details>}
    {saved && catalog && <details key={mappingId || 'new'} open={Boolean(mappingId)} className="border-t border-slate-200 pt-4"><summary className="min-h-10 cursor-pointer text-sm font-medium text-slate-700">{systemText('preInvestment.strategicScopeWorkspace.matchActualProxyProductsCanBeCompleted')}</summary><div className="mt-3"><ImplementationMappingEditor heading={false} key={`${saved.id}:${mappingId}`} universe={saved} catalog={catalog} domainId={domainId} requestedMapping={mappingId} onSaved={mapping => { setMappingId(mapping.id); setReload(n => n + 1); setParams(current => { current.set('scope', 'strategic'); current.set('strategic_universe', saved.id); current.set('mapping', mapping.id); return current }, { replace: true }) }} /></div></details>}
    {saved && <div role="group" aria-label={systemText('preInvestment.strategicScopeWorkspace.strategicScopeActions')} className="flex flex-col items-start gap-3 border-t border-slate-200 pt-4 sm:flex-row sm:flex-wrap sm:items-center">
      {!mandateBlockedReason && <Link className={actionClass('primary', 'w-full sm:w-28')} title={systemText('preInvestment.strategicScopeWorkspace.researchLongTermReturnsAndRiskLtcma')} to={`/pre-investment/ltcma/new?${new URLSearchParams({ strategic_universe: saved.id, ...(params.get('mandate') ? { mandate: params.get('mandate')! } : {}) })}`}>{systemText('preInvestment.strategicScopeWorkspace.next')}</Link>}
      <Link onClick={() => { operation.invalidate(); historyRead.invalidate(); setSaved(null); setMappingId('') }} to={(() => { const next = new URLSearchParams(params); next.set('scope', 'strategic'); next.set('strategic_universe', saved.id); next.set('edit', '1'); next.delete('copy'); return `/pre-investment/product-pool/new?${next.toString()}` })()} className={actionClass('secondary', 'w-full sm:w-28')}>{systemText('preInvestment.strategicScopeWorkspace.backToEditing')}</Link>
    </div>}
  </section>
}
