import { systemText, useI18n } from '../i18n/runtime'
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import StrategicScopeWorkspace from '../components/strategic-scope/StrategicScopeWorkspace'
import ScopeMandateSummary from '../components/strategic-scope/ScopeMandateSummary'
import { useResearchDay } from '../app/ResearchContext'
import { updateAllocationJourney, useAllocationDraft, writeAllocationDraft, useAllocationJourney } from '../app/allocationJourney'
import {
  createInvestableUniverseSnapshot,
  bindInvestableUniverseMandate,
  getInvestableUniverse,
  listInvestableUniverseSnapshots,
  listProductPoolVersions,
  poolVersionDataAsOf,
  type InvestableUniverseSnapshot,
  type InvestableUniverseSummary,
  type ProductPoolVersion,
} from '../services/productPools'
import { replayPoolVersion, type PoolReplayResult } from '../services/pit'
import { getStrategicCatalog, getMandate, mandateCashFloor, type MandateVersion, type StrategicCatalog } from '../services/strategicAllocation'
import { bindUniverseMandate, getStrategicUniverse } from '../services/strategicScope'
import { DataTable, SectionHeader, Button, ErrorPanel, LoadingPanel } from '../components/ui'
import { Field, inputClass, sectionClass } from '../components/risk-models/ResearchUI'
import ScopeFeasibility from '../components/strategic-scope/ScopeFeasibility'
import { upstreamOf } from '../components/versioning'

const today = () => new Date().toISOString().slice(0, 10)

function messageOf(reason: unknown, fallback: string) {
  return reason instanceof Error && reason.message ? reason.message : fallback
}

const backLinkClass = 'inline-flex min-h-10 items-center text-sm font-semibold text-accent-800 underline'

export default function ProductPoolWorkspacePage() {
  const { s } = useI18n()
  const [params, setParams] = useSearchParams()
  const researchDay = useResearchDay()
  const strategic = params.get('scope') === 'strategic'
  const requestedMandate = params.get('mandate') ?? ''
  const requestedNew = params.get('new') ?? ''
  const requestedEdit = params.get('edit') ?? ''

  const [mandates, setMandates] = useState<MandateVersion[]>([])
  const [catalog, setCatalog] = useState<StrategicCatalog | null>(null)
  const [catalogLoading, setCatalogLoading] = useState(true)
  const [catalogError, setCatalogError] = useState('')
  const [catalogReload, setCatalogReload] = useState(0)
  const [savedUniverses, setSavedUniverses] = useState<InvestableUniverseSummary[]>([])
  const [savedReload, setSavedReload] = useState(0)
  const sourceId = requestedNew ? '' : strategic ? params.get('strategic_universe') ?? '' : requestedEdit || params.get('copy') || params.get('universe') || ''
  const copying = Boolean(params.get('copy'))
  const scopeKey = `${strategic ? 'strategic' : 'product'}:${sourceId}:${copying ? 'copy' : 'view'}`
  const [scopeContext, setScopeContext] = useState<{ key: string; mandateId: string; error: string; mandate?: MandateVersion } | null>(null)
  const context = scopeContext?.key === scopeKey ? scopeContext : null
  const restoringScope = Boolean(sourceId && !context)
  const mandateId = sourceId ? context?.mandateId ?? '' : requestedMandate

  // 独立页面自己拉一份目录：不再和列表页共享 state，两处各自新鲜读取。
  useEffect(() => {
    const controller = new AbortController()
    setCatalogLoading(true); setCatalogError('')
    getStrategicCatalog(controller.signal)
      .then(value => { if (!controller.signal.aborted) { setMandates(value.mandates); setCatalog(value) } })
      .catch(reason => { if (!controller.signal.aborted) setCatalogError(messageOf(reason, systemText('preInvestment.productPoolWorkspacePage.unableToLoadTheInvestmentObjective'))) })
      .finally(() => { if (!controller.signal.aborted) setCatalogLoading(false) })
    return () => controller.abort()
  }, [catalogReload])

  useEffect(() => {
    let active = true
    listInvestableUniverseSnapshots()
      .then(value => { if (active) setSavedUniverses(value.items) })
      .catch(() => { /* 同名预检失败时退化为不拦截，后端仍是权威校验。 */ })
    return () => { active = false }
  }, [savedReload])

  // 已保存范围的关联为准；URL 只负责显式新建，不能把历史范围换成别的目标。
  // 旧版本只允许从用户明确指定目标的链接补记关联，书签不能证明历史归属。
  useEffect(() => {
    if (!sourceId || catalogLoading || catalogError) return
    const controller = new AbortController()
    setScopeContext(null)
    const load = strategic ? getStrategicUniverse(sourceId, controller.signal) : getInvestableUniverse(sourceId)
    void load.then(async record => {
      if (controller.signal.aborted) return
      let bound = record.mandate_id || ''
      const legacyTarget = requestedMandate
      if (!bound && legacyTarget && mandates.some(item => item.id === legacyTarget) && !copying) {
        const linked = strategic
          ? await bindUniverseMandate(sourceId, legacyTarget, controller.signal)
          : await bindInvestableUniverseMandate(sourceId, legacyTarget, controller.signal)
        if (linked.mandate_id !== legacyTarget) throw new Error(systemText('preInvestment.productPoolWorkspacePage.unableToRestoreResearchScopeLinksPlease'))
        bound = linked.mandate_id
      }
      if (controller.signal.aborted) return
      const resolved = copying ? requestedMandate || bound : bound
      const mandate = mandates.find(item => item.id === resolved)
        ?? (resolved && bound === resolved ? await getMandate(resolved, controller.signal) : undefined)
      if (controller.signal.aborted) return
      if (mandate && (mandate.id !== resolved || (bound === resolved && record.mandate_hash && record.mandate_hash !== mandate.content_hash))) {
        throw new Error(systemText('preInvestment.productPoolWorkspacePage.theObjectiveVersionLinkedToThisScope'))
      }
      setScopeContext({ key: scopeKey, mandateId: resolved, mandate, error: '' })
      setParams(current => {
        const next = new URLSearchParams(current)
        if (resolved) next.set('mandate', resolved)
        else next.delete('mandate')
        return next
      }, { replace: true })
      if (resolved && !copying) updateAllocationJourney({
        mandateId: resolved,
        ...(strategic ? { strategicUniverseId: sourceId } : { universeId: sourceId }),
        name: record.name,
      })
    }).catch(reason => {
      if (!controller.signal.aborted) setScopeContext({ key: scopeKey, mandateId: '', error: messageOf(reason, systemText('preInvestment.productPoolWorkspacePage.unableToLoadResearchScopeLinksPlease')) })
    })
    return () => controller.abort()
  }, [scopeKey, catalogLoading, catalogError, catalogReload])

  // 战略范围工作区自己会读目录并在保存后重新拉取；接回来复用同一份 state，避免同名预检读到旧列表。
  const receiveCatalog = useCallback((value: StrategicCatalog) => {
    setCatalog(value); setMandates(value.mandates); setCatalogError(''); setCatalogLoading(false)
  }, [])
  const receiveCatalogError = useCallback((message: string) => {
    setCatalogError(message); setCatalogLoading(false)
  }, [])

  const selectedMandate = mandates.find(item => item.id === mandateId) ?? context?.mandate ?? null
  const mandateBlockedReason = context?.error || (catalogLoading || restoringScope
    ? systemText('preInvestment.productPoolWorkspacePage.loadingInvestmentObjectivesAndConstraintsPleaseWait')
    : catalogError
      ? systemText('preInvestment.productPoolWorkspacePage.unableToLoadInvestmentObjectivesAndConstraints')
      : selectedMandate && !mandates.some(item => item.id === mandateId)
        ? systemText('preInvestment.productPoolWorkspacePage.thisObjectiveIsRetiredOrSupersededHistorical')
      : mandates.length === 0
        ? systemText('preInvestment.productPoolWorkspacePage.noPublishedInvestmentObjectivesAndConstraintsPublish')
        : !mandateId
          ? systemText('preInvestment.productPoolWorkspacePage.investmentObjectivesAndConstraintsAreRequiredSelect')
          : !selectedMandate
            ? systemText('preInvestment.productPoolWorkspacePage.thePreviouslySelectedObjectiveIsMissingOr')
            : '')

  const strategicScopes = catalog?.strategic_universes ?? []
  // 前端先做同名预检（后端仍是权威）；编辑排除自身。
  const strategicEditId = strategic && requestedEdit ? params.get('strategic_universe') ?? '' : ''
  const editSourceId = strategic ? strategicEditId : requestedEdit
  const existingScopeNames = [
    ...strategicScopes.filter(item => item.id !== editSourceId).map(item => item.name),
    ...savedUniverses.filter(item => item.id !== editSourceId).map(item => item.name),
  ]

  // 返回列表时带回模式与投资目标，不让用户回去后重新选一遍。
  const backQuery = new URLSearchParams()
  if (mandateId) backQuery.set('mandate', mandateId)
  if (strategic) backQuery.set('scope', 'strategic')
  const backHref = `/pre-investment/product-pool${backQuery.size ? `?${backQuery}` : ''}`

  const fixedMandate = Boolean(sourceId && !copying && context?.mandateId)
  // 修改已保存范围时，若绑定目标已有新版本，可升级为同一系列的当前版本（后端校验同系列）。
  const sourceRow = requestedEdit && !copying ? (strategic ? strategicScopes : savedUniverses).find(item => item.id === sourceId) : undefined
  const boundRef = upstreamOf(sourceRow, 'mandate')
  const upgradeTo = boundRef?.status === 'superseded' && boundRef.id === context?.mandateId
    ? mandates.find(item => item.id === boundRef.latest_id) : undefined
  const upgradeMandate = () => {
    if (!context || !upgradeTo) return
    setScopeContext({ ...context, mandateId: upgradeTo.id, mandate: upgradeTo })
    updateAllocationJourney({ mandateId: upgradeTo.id })
    setParams(current => { const next = new URLSearchParams(current); next.set('mandate', upgradeTo.id); return next }, { replace: true })
  }
  const upgraded = Boolean(boundRef && context?.mandateId && boundRef.id !== context.mandateId)
  const chooseMandate = (id: string) => {
    if (fixedMandate) return
    if (copying && context) setScopeContext({ ...context, mandateId: id, mandate: mandates.find(item => item.id === id) })
    updateAllocationJourney({ mandateId: id || undefined })
    setParams(current => {
      const next = new URLSearchParams(current)
      if (id) next.set('mandate', id)
      else next.delete('mandate')
      return next
    }, { replace: true })
    // 未绑定的旧范围仍通过现有明确关联接口补记，不改写冻结内容。
    if (sourceId && !copying) setCatalogReload(value => value + 1)
  }
  const mandateField = <div className="space-y-2">
    <div className="max-w-xl"><Field label={s('preInvestment.productPoolSelection.investmentObjectivesAndConstraints')} required>
      <select required aria-label={s('preInvestment.productPoolSelection.investmentObjectivesAndConstraints')} aria-describedby="scope-mandate-help"
        value={selectedMandate ? mandateId : ''} disabled={catalogLoading || restoringScope || Boolean(catalogError || context?.error) || fixedMandate}
        onChange={event => chooseMandate(event.target.value)} className={inputClass}>
        <option value="">{s('preInvestment.productPoolSelection.selectPublishedInvestmentObjectivesAndConstraints')}</option>
        {selectedMandate && !mandates.some(item => item.id === selectedMandate.id) && <option value={selectedMandate.id}>{selectedMandate.name}</option>}
        {mandates.map(mandate => <option key={mandate.id} value={mandate.id}>{mandate.name} · {mandate.definition.currency} · {mandate.definition.horizon_years} {s('preInvestment.productPoolSelection.years')}</option>)}
      </select>
    </Field></div>
    <p id="scope-mandate-help" className="text-xs leading-5 text-slate-600">{s(upgraded ? 'versioning.scopeMandateUpgraded' : fixedMandate ? 'researchScope.mandateFixed' : 'researchScope.mandateHelp')}</p>
    {upgradeTo && boundRef && <div className="flex flex-wrap items-center gap-2 text-sm text-amber-800"><span>{s('versioning.scopeMandateHasNewer', { number: boundRef.number ?? '', latest: boundRef.latest_number ?? '' })}</span><Button onClick={upgradeMandate}>{s('versioning.scopeMandateUpgrade', { latest: boundRef.latest_number ?? '' })}</Button></div>}
    {mandateBlockedReason && <p role={catalogError || context?.error ? 'alert' : 'status'} className="text-sm text-amber-800">{mandateBlockedReason}</p>}
    {(catalogError || context?.error) && <Button onClick={() => setCatalogReload(value => value + 1)}>{s('preInvestment.scopeMandateSummary.retryLoadingObjective')}</Button>}
    {!catalogLoading && !catalogError && !restoringScope && mandates.length === 0 && <Link className={backLinkClass} to="/pre-investment/objectives">{s('preInvestment.productPoolSelection.createAnInvestmentObjectiveFirst')}</Link>}
    {selectedMandate && researchDay === undefined && <p role="status" className="text-xs text-slate-600">{s('preInvestment.productPoolSelection.thePagePitContextIsNotYet')}</p>}
    {selectedMandate && researchDay === null && <p role="status" className="text-xs text-slate-600">{s('preInvestment.productPoolSelection.pitIsOffOnThisPageSo')} {selectedMandate.definition.as_of} {s('preInvestment.productPoolSelection.isRetained')}</p>}
    {selectedMandate && typeof researchDay === 'string' && selectedMandate.definition.as_of !== researchDay && <p role="status" className="text-sm text-amber-800">{s('preInvestment.productPoolSelection.theSelectedObjectiveSResearchDate')} {selectedMandate.definition.as_of} {s('preInvestment.productPoolSelection.andTheCurrentPagePitDate')} {researchDay} {s('preInvestment.productPoolSelection.differYouCanContinueResearch')}</p>}
  </div>

  return <div className="space-y-5">
    <Link className={backLinkClass} to={backHref}>{systemText('preInvestment.productPoolWorkspacePage.backToResearchScopes')}</Link>
    {restoringScope && (catalogError
      ? <ErrorPanel message={catalogError} action={<Button onClick={() => setCatalogReload(value => value + 1)}>{s('preInvestment.scopeMandateSummary.retryLoadingObjective')}</Button>} />
      : <LoadingPanel text={s('preInvestment.productPoolWorkspacePage.loadingInvestmentObjectivesAndConstraintsPleaseWait')} />)}
    {selectedMandate && !catalogLoading && !catalogError && <ScopeMandateSummary mandate={selectedMandate} loading={false}
      blockedReason={mandateBlockedReason} backHref={backHref} researchDay={researchDay}
      onRetry={catalogError || context?.error ? () => setCatalogReload(value => value + 1) : undefined} />}
    {!restoringScope && (strategic
      ? <StrategicScopeWorkspace mandateField={mandateField} cashFloor={mandateCashFloor(selectedMandate)} freshKey={requestedNew} editSource={strategicEditId} existingScopeNames={existingScopeNames} mandateBlockedReason={mandateBlockedReason} onCatalogLoaded={receiveCatalog} onCatalogError={receiveCatalogError} />
      : <ProductPoolWorkspace mandateField={mandateField} freshKey={requestedNew} editSource={requestedEdit} existingScopeNames={existingScopeNames} mandateBlockedReason={mandateBlockedReason} onScopeSaved={() => setSavedReload(n => n + 1)} />)}
  </div>
}

function ProductPoolWorkspace({ mandateField, mandateBlockedReason, freshKey, onScopeSaved, editSource, existingScopeNames }: { mandateField: ReactNode; mandateBlockedReason: string; freshKey: string; onScopeSaved: () => void; editSource: string; existingScopeNames: string[] }) {
  useI18n()
  const generation = useRef(0)
  useEffect(() => () => { ++generation.current }, [])
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const [journey] = useAllocationJourney()
  const platformAsOf = useResearchDay()
  const requestedVersion = params.get('version') || ''
  const requestedUniverse = params.get('universe') || ''
  const copySource = params.get('copy') || ''
  const requestedMandate = params.get('mandate') || ''
  useEffect(() => {
    ++generation.current
    setCreating(false)
    if (requestedMandate) updateAllocationJourney({ mandateId: requestedMandate })
  }, [requestedMandate])
  // 只读来源、复制草稿与全新草稿各用独立 key：复制不会污染冻结范围，新建不会回落到旧旅程。
  const draftScope = editSource
    ? `edit:${editSource}`
    : requestedUniverse
      ? (copySource ? `copy:${copySource}` : `universe:${requestedUniverse}`)
    : requestedVersion
      ? `version:${requestedVersion}`
      : freshKey
        ? `new:${freshKey}`
        : 'resume'
  const [draft, setDraft] = useAllocationDraft(`pool:${draftScope}`, { researchDate: '', name: systemText('preInvestment.productPoolWorkspacePage.preInvestmentResearchUniverse'), selectedIds: [] as string[], snapshotId: '', selectionEdited: false, excludedKeys: [] as string[] })
  // 切换只读/复制/新建身份时作废在途创建，并清掉上一个身份的锁定结果与前向操作。
  useEffect(() => {
    ++generation.current
    createdRef.current = ''
    setCreating(false); setJustCreated(false); setSnapshot(null); setServerNameError('')
  }, [draftScope])
  const researchDate = draft.researchDate || platformAsOf || journey.researchDate || today()
  const name = draft.name
  const selectedIds = draft.selectedIds
  const [serverNameError, setServerNameError] = useState('')
  const duplicateName = Boolean(name.trim()) && existingScopeNames.some(existing => existing.trim().toLowerCase() === name.trim().toLowerCase())
  const nameError = duplicateName ? systemText('preInvestment.productPoolWorkspacePage.aResearchScopeWithThisNameAlready') : serverNameError
  const [showHistory, setShowHistory] = useState(false)
  const [versionQuery, setVersionQuery] = useState('')
  const setName = (value: string) => { ++generation.current; setCreating(false); setServerNameError(''); setDraft(current => ({ ...current, name: value })) }
  const setResearchDate = (value: string) => { ++generation.current; setCreating(false); setDraft(current => ({ ...current, researchDate: value, snapshotId: '', selectedIds: [], selectionEdited: true })); setSnapshot(null) }
  const [versions, setVersions] = useState<ProductPoolVersion[]>([])
  const [allVersions, setAllVersions] = useState<ProductPoolVersion[]>([])
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState('')
  const [snapshot, setSnapshot] = useState<InvestableUniverseSnapshot | null>(null)
  const [justCreated, setJustCreated] = useState(false)
  const [replays, setReplays] = useState<Record<string, PoolReplayResult | string>>({})

  // A version whose data cut is later than the research day was screened with
  // information that day did not have. Reproducible is not the same as causal,
  // and this is the one place the difference is still fixable.
  const replay = async (versionId: string) => {
    setReplays((current) => ({ ...current, [versionId]: systemText('preInvestment.productPoolWorkspacePage.replaying') }))
    try {
      const result = await replayPoolVersion(versionId, researchDate)
      setReplays((current) => ({ ...current, [versionId]: result }))
    } catch (reason) {
      setReplays((current) => ({ ...current, [versionId]: messageOf(reason, systemText('preInvestment.productPoolWorkspacePage.replayFailed')) }))
    }
  }

  useEffect(() => {
    let active = true
    setLoading(true); setError(''); setReplays({})
    // 生效版本用于选择；全部发布版本用于区分“尚未发布”与“该日无生效版本”，
    // 并保证已保存范围引用的旧版本不会被当前筛选静默丢弃。
    Promise.all([
      listProductPoolVersions({ activeOn: researchDate }),
      listProductPoolVersions({}),
    ])
      .then(([activeResponse, allResponse]) => {
        if (!active) return
        setVersions(activeResponse.items)
        setAllVersions(allResponse.items)
        const known = new Set([...activeResponse.items, ...allResponse.items].map(item => item.id))
        setDraft(current => ({ ...current, selectedIds: current.selectedIds.length ? current.selectedIds.filter(id => known.has(id)) : requestedVersion && known.has(requestedVersion) ? [requestedVersion] : [] }))
      })
      .catch((reason) => { if (active) setError(messageOf(reason, systemText('preInvestment.productPoolWorkspacePage.unableToLoadEffectiveProductPoolVersions'))) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [researchDate, requestedVersion])

  // 显式新建不恢复旧旅程；复制与编辑只读取来源身份，草稿各自独立。
  const restoreId = draft.snapshotId || (!draft.selectionEdited
    ? (freshKey ? '' : editSource || copySource || requestedUniverse)
    : '')
  const copiedRef = useRef('')
  const createdRef = useRef('')
  const restoreToken = useRef(0)
  useEffect(() => {
    const token = ++restoreToken.current
    // 刚创建的快照就是当前状态；只有非空且相同的已创建 ID 才跳过恢复，
    // 空 restoreId 仍要清掉旧快照，避免残留已锁定区与前向操作。
    if (createdRef.current && restoreId === createdRef.current) return
    setSnapshot(null)
    if (!restoreId) return
    let active = true
    getInvestableUniverse(restoreId).then(result => {
      if (!active || token !== restoreToken.current) return
      if (editSource && editSource === restoreId) {
        if (copiedRef.current === editSource) return
        copiedRef.current = editSource
        setJustCreated(false)
        // 编辑：载入原字段到可编辑草稿，保存时以新版本替代当前版本。
        setDraft(current => ({ ...current, snapshotId: '', selectionEdited: true, name: result.name, researchDate: result.research_date, selectedIds: result.version_ids ?? [], excludedKeys: result.excluded_product_keys ?? [] }))
        return
      }
      if (copySource && copySource === restoreId) {
        if (copiedRef.current === copySource) return
        copiedRef.current = copySource
        setJustCreated(false)
        // 复制是显式命令：产生可编辑草稿，绝不改写已保存快照身份。
        setDraft(current => ({ ...current, snapshotId: '', selectionEdited: true, name: systemText('preInvestment.productPoolWorkspacePage.copy', { p0: result.name }), researchDate: result.research_date, selectedIds: result.version_ids ?? [], excludedKeys: result.excluded_product_keys ?? [] }))
        return
      }
      setSnapshot(result)
      setJustCreated(false)
      setDraft(current => ({ ...current, snapshotId: result.id, name: current.snapshotId === result.id ? current.name : result.name, researchDate: result.research_date, selectedIds: result.version_ids ?? current.selectedIds, selectionEdited: false }))
      updateAllocationJourney({ universeId: result.id, name: result.name, researchDate: result.research_date, poolVersionIds: result.version_ids })
    }).catch(() => { if (active && token === restoreToken.current) setError(systemText('preInvestment.productPoolWorkspacePage.thePreviouslyLockedProductScopeCouldNot')) })
    return () => { active = false }
  }, [restoreId, copySource, draftScope])

  const toggle = (versionId: string) => {
    ++generation.current; setCreating(false)
    const target = versions.find(item => item.id === versionId)
    setDraft(current => ({ ...current, snapshotId: '', selectionEdited: true, selectedIds: current.selectedIds.includes(versionId)
      ? current.selectedIds.filter(id => id !== versionId)
      : [...current.selectedIds.filter(id => versions.find(item => item.id === id)?.pool_id !== target?.pool_id), versionId] }))
    setSnapshot(null)
  }
  const versionMatches = (version: ProductPoolVersion) => !versionQuery.trim() || `${version.pool_name} V${version.version}`.toLowerCase().includes(versionQuery.trim().toLowerCase())
  const visibleVersions = (showHistory ? versions : versions.filter(version => selectedIds.includes(version.id) || !versions.some(other => other.pool_id === version.pool_id && other.version > version.version)))
    .filter(version => versionMatches(version) || selectedIds.includes(version.id))

  const createSnapshot = async () => {
    if (mandateBlockedReason) {
      setError(mandateBlockedReason)
      return
    }
    if (nameError) {
      setError(nameError)
      return
    }
    if (!name.trim() || selectedIds.length === 0) {
      setError(systemText('preInvestment.productPoolWorkspacePage.enterANameAndSelectAtLeast'))
      return
    }
    const token = ++generation.current
    setCreating(true); setError('')
    try {
      const result = await createInvestableUniverseSnapshot({
        name: name.trim(),
        mandate_id: requestedMandate || undefined,
        research_date: researchDate,
        version_ids: selectedIds,
        ...(draft.excludedKeys?.length ? { excluded_product_keys: draft.excludedKeys } : {}),
        ...(editSource ? { replaces_snapshot_id: editSource } : {}),
      })
      if (token !== generation.current) return
      setSnapshot(result)
      setJustCreated(true)
      createdRef.current = result.id
      const savedDraft = { ...draft, snapshotId: result.id, selectionEdited: false, researchDate: result.research_date, selectedIds: result.version_ids ?? selectedIds }
      // 每次保存都落到规范身份：持久化只读草稿并把 URL 指向新 ID，清掉来源参数。
      writeAllocationDraft(`pool:universe:${result.id}`, savedDraft)
      setParams(current => {
        const next = new URLSearchParams(current)
        next.set('universe', result.id)
        next.delete('copy'); next.delete('edit'); next.delete('new'); next.delete('version')
        return next
      }, { replace: true })
      updateAllocationJourney({ name: result.name, researchDate: result.research_date, universeId: result.id, poolVersionIds: result.version_ids })
      onScopeSaved()
    } catch (reason) {
      if (token !== generation.current) return
      const message = messageOf(reason, systemText('preInvestment.productPoolWorkspacePage.unableToCreateTheInvestableUniverseSnapshot'))
      if (message.includes(systemText('preInvestment.productPoolWorkspacePage.nameAlreadyExists'))) setServerNameError(message)
      else setError(message)
    } finally {
      if (token === generation.current) setCreating(false)
    }
  }

  // 选中的冻结范围默认只读：先复制新建，再进入编辑，避免在已保存身份上就地改动。
  const readOnly = Boolean(snapshot && !draft.selectionEdited && !copySource && !freshKey && !justCreated)
  const copyHref = (() => {
    const next = new URLSearchParams(params)
    next.set('copy', snapshot?.id ?? '')
    return `/pre-investment/product-pool/new?${next.toString()}`
  })()
  const editHref = (() => {
    const next = new URLSearchParams(params)
    next.set('universe', snapshot?.id ?? '')
    next.set('edit', snapshot?.id ?? '')
    next.delete('copy')
    return `/pre-investment/product-pool/new?${next.toString()}`
  })()
  const cancelEdit = () => {
    if (!editSource) return
    writeAllocationDraft(`pool:edit:${editSource}`, { researchDate: '', name: systemText('preInvestment.productPoolWorkspacePage.preInvestmentResearchUniverse'), selectedIds: [], snapshotId: '', selectionEdited: false, excludedKeys: [] })
    const next = new URLSearchParams(params)
    next.delete('edit'); next.delete('copy')
    setParams(next, { replace: true })
  }
  const forwardButtons = snapshot && <div className="flex flex-wrap gap-2"><Button tone="primary" disabled={Boolean(mandateBlockedReason)} title={mandateBlockedReason || undefined} onClick={() => navigate(`/pre-investment/saa/auto-classification?universe=${encodeURIComponent(snapshot.id)}`)}>{systemText('preInvestment.productPoolWorkspacePage.continueToAutomaticAssetClassification')}</Button><Button disabled={Boolean(mandateBlockedReason)} title={mandateBlockedReason || undefined} onClick={() => navigate(`/pre-investment/saa/asset-classes?universe=${encodeURIComponent(snapshot.id)}`)}>{systemText('preInvestment.productPoolWorkspacePage.continueToManualAssetConstruction')}</Button></div>
  const snapshotFacts = snapshot && <dl className="grid grid-cols-2 gap-3 md:grid-cols-4"><div className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-600">{systemText('preInvestment.productPoolWorkspacePage.productPoolVersion')}</dt><dd className="mt-1 text-lg font-semibold">{snapshot.version_ids?.length ?? 0}</dd></div><div className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-600">{systemText('preInvestment.productPoolWorkspacePage.evaluationPlanGroup')}</dt><dd className="mt-1 text-lg font-semibold">{snapshot.groups?.length ?? 0}</dd></div><div className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-600">{systemText('preInvestment.productPoolWorkspacePage.investableProducts')}</dt><dd className="mt-1 text-lg font-semibold">{snapshot.product_count}</dd></div><div className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-600">{systemText('preInvestment.productPoolWorkspacePage.researchDate')}</dt><dd className="mt-1 text-sm font-semibold">{snapshot.research_date}</dd></div></dl>
  const snapshotGroups = snapshot && <div className="grid gap-4 xl:grid-cols-2">{(snapshot.groups ?? []).map((group) => <div key={`${group.evaluation_plan_id}:${group.evaluation_plan_revision}`} className="rounded-xl border border-slate-200 p-4"><h3 className="font-semibold text-slate-900">{group.evaluation_plan_name}</h3><p className="mt-1 text-xs text-slate-600">{systemText('preInvestment.productPoolWorkspacePage.evaluationPlanV')}{group.evaluation_plan_revision} · {group.product_count} {" " + systemText('preInvestment.productPoolWorkspacePage.products')}</p><ul className="mt-3 max-h-56 divide-y divide-slate-100 overflow-auto">{group.products.map((product) => <li key={product.key} className="flex items-center justify-between gap-3 py-2 text-sm"><span className="min-w-0 break-words"><b>{product.name || product.code}</b><span className="ml-2 font-mono text-xs text-slate-600">{product.code}</span></span><span className="text-xs text-slate-600">{product.usage_status === 'limited' ? systemText('preInvestment.productPoolWorkspacePage.limit', { p0: product.max_weight == null ? '—' : `${(product.max_weight * 100).toFixed(1)}%` }) : systemText('preInvestment.productPoolWorkspacePage.valid')}</span></li>)}</ul></div>)}</div>
  const compositionTable = <div className="mt-3"><DataTable<ProductPoolVersion>
    caption={systemText('preInvestment.productPoolWorkspacePage.scopeCompositionReadOnlyVersion')}
    rows={visibleVersions}
    rowKey={version => version.id}
    minWidth="620px"
    loading={loading ? systemText('preInvestment.productPoolWorkspacePage.loadingVersionComposition') : undefined}
    empty={systemText('preInvestment.productPoolWorkspacePage.noVersionsToDisplay')}
    columns={[
      { header: systemText('preInvestment.productPoolWorkspacePage.select'), cell: version => <input type="checkbox" disabled aria-label={`${version.pool_name} · V${version.version}`} checked={selectedIds.includes(version.id)} readOnly /> },
      { header: systemText('preInvestment.productPoolWorkspacePage.productPoolVersion'), cell: version => `${version.pool_name} · V${version.version}` },
      { header: systemText('preInvestment.productPoolWorkspacePage.effectivePeriod'), nowrap: true, cell: version => `${version.effective_from} ～ ${version.effective_to || systemText('preInvestment.productPoolWorkspacePage.openEnded')}` },
      { header: systemText('preInvestment.productPoolWorkspacePage.evaluationData'), nowrap: true, cell: version => systemText('preInvestment.productPoolWorkspacePage.evaluationDataThrough', { p0: poolVersionDataAsOf(version) ?? systemText('preInvestment.productPoolWorkspacePage.notRecorded') }) },
    ]}
  /></div>

  return <div className="space-y-5" aria-busy={loading || creating}>
    {readOnly && snapshot && <section className={`${sectionClass} space-y-5 shadow-sm`}>
      <SectionHeader title={systemText('preInvestment.productPoolWorkspacePage.savedProductScope')} description={systemText('preInvestment.productPoolWorkspacePage.thisSavedScopeIsReadOnlyEditing')} />
      {mandateField}
      {error && <div role="alert" className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">{error}</div>}
      <div className="flex flex-wrap items-start justify-between gap-3"><h2 className="text-lg font-semibold text-slate-900">{snapshot.name}</h2><div className="flex flex-wrap gap-2">{forwardButtons}<Link to={editHref} className="inline-flex min-h-10 items-center rounded-lg border border-slate-300 bg-white px-4 text-sm font-medium text-slate-700">{systemText('preInvestment.productPoolWorkspacePage.editThisScope')}</Link><Link to={copyHref} className="inline-flex min-h-10 items-center rounded-lg border border-slate-300 bg-white px-4 text-sm font-medium text-slate-700">{systemText('preInvestment.productPoolWorkspacePage.copyThisScopeAsNew')}</Link></div></div>
      {snapshotFacts}
      <details className="rounded-xl border border-slate-200 px-3 py-2"><summary className="min-h-10 cursor-pointer text-sm font-medium text-slate-700">{systemText('preInvestment.productPoolWorkspacePage.scopeCompositionReadOnly')}</summary>{compositionTable}</details>
      {snapshotGroups}
    </section>}
    {!readOnly && <>
    <section className={`${sectionClass} space-y-5 shadow-sm`}>
      <SectionHeader title={systemText('preInvestment.productPoolWorkspacePage.selectProductPoolVersions')} description={systemText('preInvestment.productPoolWorkspacePage.selectProductScopesAndNameThisStudy')} />
      {mandateField}
      {error && <div role="alert" className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">{error}</div>}
      <div className="grid gap-4 md:grid-cols-[180px_auto] md:items-end">
        <Field label={systemText('preInvestment.productPoolWorkspacePage.researchDate')} required><input required type="date" value={researchDate} onChange={(event) => setResearchDate(event.target.value)} className={inputClass} /></Field>
        <div className="flex flex-wrap gap-2"><Button tone="primary" disabled={creating || selectedIds.length === 0 || Boolean(mandateBlockedReason) || Boolean(nameError)} title={mandateBlockedReason || nameError || undefined} onClick={() => void createSnapshot()}>{editSource ? systemText('preInvestment.productPoolWorkspacePage.saveChanges') : systemText('preInvestment.productPoolWorkspacePage.createLockedSnapshot')}</Button>{editSource && <Button disabled={creating} onClick={cancelEdit}>{systemText('preInvestment.productPoolWorkspacePage.cancelEditing')}</Button>}</div>
      </div>
      <details open={Boolean(editSource) || Boolean(nameError)} className="rounded-xl border border-slate-200 px-3 py-2">
        <summary className="min-h-10 cursor-pointer text-sm font-medium text-slate-700">{systemText('preInvestment.productPoolWorkspacePage.researchNameGeneratedByDefaultEditable')}</summary>
        <div className="mt-3 max-w-md"><Field label={systemText('preInvestment.productPoolWorkspacePage.researchName')} required><input required value={name} aria-invalid={nameError ? true : undefined} aria-describedby={nameError ? 'scope-name-error' : undefined} onChange={(event) => setName(event.target.value)} className={inputClass} />{nameError && <span id="scope-name-error" role="alert" className="mt-1 block text-xs text-rose-800">{nameError}</span>}</Field></div>
      </details>
      <p className="mt-3 text-xs text-slate-600">{platformAsOf === undefined ? systemText('preInvestment.productPoolWorkspacePage.thePlatformPitContextIsUnconfirmedThis') : platformAsOf ? systemText('preInvestment.productPoolWorkspacePage.platformKnowledgeCutoffThisDateFiltersEffective', { p0: platformAsOf }) : systemText('preInvestment.productPoolWorkspacePage.thePlatformUsesAllOnDiskData')}</p>
      {platformAsOf && researchDate > platformAsOf && <p role="alert" className="mt-2 text-sm text-amber-800">{systemText('preInvestment.productPoolWorkspacePage.theProductScopeResearchDateIsLater')}</p>}
    </section>

    <section className={`${sectionClass} shadow-sm`}>
      <div className="flex flex-wrap items-end justify-between gap-3"><div><h2 className="text-lg font-semibold text-slate-900">{systemText('preInvestment.productPoolWorkspacePage.effectiveProductPoolVersions')}<span aria-hidden="true" className="ml-0.5 text-rose-700">*</span></h2><p className="mt-1 text-sm text-slate-600">{systemText('preInvestment.productPoolWorkspacePage.selectAtLeastOneVersionSelected') + " "}{selectedIds.length} {" " + systemText('preInvestment.productPoolWorkspacePage.items')}</p></div><span className="text-xs text-slate-600">{systemText('preInvestment.productPoolWorkspacePage.researchDate2')}{researchDate}</span></div>
      <div className="mt-3 flex flex-wrap items-end gap-3">
        <Field label={systemText('preInvestment.productPoolWorkspacePage.searchProductPoolVersions')}><input value={versionQuery} onChange={event => setVersionQuery(event.target.value)} className={inputClass} /></Field>
        <label className="flex min-h-10 items-center gap-2 text-sm text-slate-600"><input aria-label={systemText('preInvestment.productPoolWorkspacePage.showHistoricalPublishedVersions')} type="checkbox" checked={showHistory} onChange={event => setShowHistory(event.target.checked)} /><span data-optional-label={systemText('common.optional')} className="after:ml-1 after:text-xs after:text-slate-500 after:content-[attr(data-optional-label)]">{systemText('preInvestment.productPoolWorkspacePage.showHistoricalPublishedVersions')}</span></label>
      </div>
      <div className="mt-4"><DataTable<ProductPoolVersion>
        caption={systemText('preInvestment.productPoolWorkspacePage.effectiveProductPoolVersions')}
        rows={visibleVersions}
        rowKey={version => version.id}
        minWidth="760px"
        loading={loading ? systemText('preInvestment.productPoolWorkspacePage.loadingEffectiveProductPoolVersions') : undefined}
        empty={versions.length === 0 && allVersions.length > 0 ? systemText('preInvestment.productPoolWorkspacePage.noVersionsWereEffectiveOnTheResearch') : systemText('preInvestment.productPoolWorkspacePage.noMatchingEffectiveProductPoolVersions')}
        columns={[
          { header: systemText('preInvestment.productPoolWorkspacePage.select'), cell: version => <input type="checkbox" aria-label={`${version.pool_name} · V${version.version}`} checked={selectedIds.includes(version.id)} onChange={() => toggle(version.id)} className="h-4 w-4" /> },
          { header: systemText('preInvestment.productPoolWorkspacePage.productPoolVersion'), cell: version => <span className="block min-w-0"><b>{version.pool_name} · V{version.version}</b><span className="block text-xs text-slate-600">{version.effective_from} ～ {version.effective_to || systemText('preInvestment.productPoolWorkspacePage.openEnded')} · {version.evaluation_plans.length} {" " + systemText('preInvestment.productPoolWorkspacePage.evaluationPlans')}</span></span> },
          { header: systemText('preInvestment.productPoolWorkspacePage.evaluationData'), cell: version => {
            const dataAsOf = poolVersionDataAsOf(version)
            const lookahead = Boolean(dataAsOf && dataAsOf > researchDate)
            const replayed = replays[version.id]
            return <span className="block text-xs leading-5">
              <span className={`rounded-lg border px-1.5 py-0.5 font-semibold ${lookahead ? 'border-rose-200 bg-rose-50 text-rose-800' : 'border-slate-200 bg-slate-50 text-slate-600'}`}>{systemText('preInvestment.productPoolWorkspacePage.evaluationDataThrough2') + " "}{dataAsOf ?? systemText('preInvestment.productPoolWorkspacePage.notRecorded')}</span>
              {lookahead && <span className="ml-2 text-rose-700">{systemText('preInvestment.productPoolWorkspacePage.isLaterThanTheResearchDate') + " "}{researchDate}{systemText('preInvestment.productPoolWorkspacePage.theListContainsFutureInformation')}</span>}
              <button type="button" onClick={() => void replay(version.id)} className="ml-2 underline hover:no-underline">{systemText('preInvestment.productPoolWorkspacePage.replayAsOfResearchDate')}</button>
              {typeof replayed === 'string' && <span className="mt-1 block text-slate-600">{replayed}</span>}
              {replayed && typeof replayed !== 'string' && <span className="mt-1 block text-slate-700">{systemText('preInvestment.productPoolWorkspacePage.replayThrough') + " "}{replayed.as_of}{systemText('preInvestment.productPoolWorkspacePage.retained') + " "}{replayed.summary.kept} {" " + systemText('preInvestment.productPoolWorkspacePage.added') + " "}{replayed.summary.added} {" " + systemText('preInvestment.productPoolWorkspacePage.removed') + " "}{replayed.summary.removed}{replayed.summary.manual_only > 0 && systemText('preInvestment.productPoolWorkspacePage.manuallyAdmittedCannotReplay', { p0: replayed.summary.manual_only })}</span>}
            </span>
          } },
          { header: systemText('preInvestment.productPoolWorkspacePage.investableProducts'), numeric: true, cell: version => version.investable_count },
        ]}
      /></div>
      {!loading && versions.length === 0 && (allVersions.length === 0
        ? <p className="mt-4 rounded-lg bg-slate-50 p-6 text-center text-sm text-slate-600">{systemText('preInvestment.productPoolWorkspacePage.noPublishedProductPoolVersionsYet')}<Link to="/product-research/pools" className="ml-2 text-accent-800 underline">{systemText('preInvestment.productPoolWorkspacePage.createOrPublishAProductPool')}</Link></p>
        : <div className="mt-4 space-y-3">
          <p className="rounded-lg bg-amber-50 p-4 text-sm text-amber-900">{systemText('preInvestment.productPoolWorkspacePage.researchDate3') + " "}{researchDate} {" " + systemText('preInvestment.productPoolWorkspacePage.noEffectiveProductPoolVersionsPublishedVersions')}</p>
          <DataTable<ProductPoolVersion>
            caption={systemText('preInvestment.productPoolWorkspacePage.publishedProductPoolVersionsNotEffectiveOn')}
            rows={allVersions}
            rowKey={version => version.id}
            minWidth="620px"
            empty={systemText('preInvestment.productPoolWorkspacePage.noPublishedVersions')}
            columns={[
              { header: systemText('preInvestment.productPoolWorkspacePage.productPoolVersion'), cell: version => `${version.pool_name} · V${version.version}` },
              { header: systemText('preInvestment.productPoolWorkspacePage.effectivePeriod'), nowrap: true, cell: version => `${version.effective_from} ～ ${version.effective_to || systemText('preInvestment.productPoolWorkspacePage.openEnded')}` },
              { header: systemText('preInvestment.productPoolWorkspacePage.evaluationData'), nowrap: true, cell: version => systemText('preInvestment.productPoolWorkspacePage.evaluationDataThrough', { p0: poolVersionDataAsOf(version) ?? systemText('preInvestment.productPoolWorkspacePage.notRecorded') }) },
            ]}
          />
        </div>)}
    </section>

    {snapshot && <section className={`${sectionClass} space-y-5 shadow-sm`}>
      <div className="flex flex-wrap items-start justify-between gap-3"><div><p className="text-sm font-medium text-emerald-800">{systemText('preInvestment.productPoolWorkspacePage.investableUniverseSnapshotLocked')}</p><h2 className="mt-1 text-lg font-semibold text-slate-900">{snapshot.name}</h2></div>{forwardButtons}</div>
      {snapshotFacts}
      {snapshotGroups}
    </section>}
    </>}
    <div className={sectionClass}><ScopeFeasibility clock={platformAsOf}
      input={requestedMandate && (snapshot || selectedIds.length) ? {
        mandate_id: requestedMandate, as_of: snapshot?.research_date ?? researchDate,
        ...(snapshot ? { universe_snapshot_id: snapshot.id } : { product_version_ids: selectedIds, product_excluded_keys: draft.excludedKeys ?? [] }),
      } : null}
      blockedReason={mandateBlockedReason || (loading || (restoreId && !snapshot) ? systemText('preInvestment.productPoolWorkspacePage.loadingResearchScopePleaseWait') : platformAsOf && researchDate > platformAsOf ? systemText('preInvestment.productPoolWorkspacePage.setTheResearchDateOnOrBefore') : '')} /></div>
  </div>
}
